"""llama.cpp OpenAI-compatible inference backend (managed-process aware)."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import urlparse

from neuroswarm_arm.runtime.dipa.interfaces.backend import InferenceBackend
from neuroswarm_arm.runtime.dipa.interfaces.types import (
    BackendCapabilities,
    DecodeRequest,
    DeviceClass,
    GenerateRequest,
    GenerateResult,
    HealthState,
    HealthStatus,
    PrefillRequest,
    PrefillResult,
    TokenChunk,
)

from ...execution.execution_context import ExecutionContext
from ...control.telemetry_exporter import TelemetryExporter
from neuroswarm_arm.runtime.slot_registry import SlotRegistry
from neuroswarm_arm.runtime.slot_router import SlotRouter
from neuroswarm_arm.runtime.radix_slot_router import RadixSlotRouter
from neuroswarm_arm.runtime.okf_slot_affinity import OkfSlotAffinity, block_hashes_from_baggage

from neuroswarm_arm.runtime.dipa.backends.llama_cpp.kleidiai_verifier import (
    KleidiaiVerifier,
    probe_cpu_features,
)
from .process_supervisor import ProcessSupervisor
from .slot_client import SlotClient
from .kv_bridge import MAKStoLlamaKVBridge


def _slot_kv_reuse_enabled() -> bool:
    return os.getenv("NSA_LLAMA_SLOT_KV_REUSE", "1").strip() not in {
        "0",
        "false",
        "False",
        "no",
        "NO",
    }


def _resolve_slot_dir() -> Path:
    if LlamaCppBackend.slot_dir is not None:
        return LlamaCppBackend.slot_dir
    return Path(os.getenv("NSA_LLAMA_SLOT_DIR", "/tmp/neuroswarm-slots"))


class SpecDecodeMetrics:
    """Local ASR counters/gauges for token-level speculative decoding (no prometheus_client)."""

    _EWMA_ALPHA = 0.2

    def __init__(self) -> None:
        self._local: dict[str, float] = {
            "asr_draft_tokens_total": 0.0,
            "asr_accepted_tokens_total": 0.0,
            "asr_verify_calls_total": 0.0,
            "asr_tok_per_s": 0.0,
        }

    def reset(self) -> None:
        for key in self._local:
            self._local[key] = 0.0

    def snapshot(self) -> dict[str, float]:
        return dict(self._local)

    def get(self, name: str) -> float:
        return float(self._local.get(name, 0.0))

    def inc(self, name: str, value: float = 1.0) -> None:
        self._local[name] = self._local.get(name, 0.0) + value

    def set(self, name: str, value: float) -> None:
        self._local[name] = value

    def observe_tok_per_s(self, tokens: float, elapsed_s: float) -> None:
        if elapsed_s <= 0 or tokens <= 0:
            return
        sample = tokens / elapsed_s
        prev = self._local.get("asr_tok_per_s", 0.0)
        if prev <= 0:
            self._local["asr_tok_per_s"] = sample
        else:
            a = self._EWMA_ALPHA
            self._local["asr_tok_per_s"] = a * sample + (1.0 - a) * prev


ASR_METRICS = SpecDecodeMetrics()


@dataclass(slots=True)
class LlamaHttpClient:
    """HTTP client for OpenAI-compatible llama.cpp servers."""

    base_url: str
    timeout_s: float = 120.0
    health_timeout_s: float = 5.0

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        return self._post("/v1/chat/completions", payload)

    def generate_with_logits(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_logprobs: int = 5,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        if extra:
            payload.update(extra)
        return self._post("/v1/chat/completions", payload)

    def generate_with_logits_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_logprobs: int = 5,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Same payload as generate_with_logits but stream: True → SSE lines."""
        merged: dict[str, Any] = {
            **(extra or {}),
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
        }
        return self.chat_stream_raw(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra=merged,
        )

    def chat_stream_raw(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        extra: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if extra:
            payload.update(extra)
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line.decode("utf-8", errors="ignore")
        except error.HTTPError as exc:
            raise RuntimeError(f"llama server HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"llama server unavailable: {exc.reason}") from exc

    def is_ready(self) -> bool:
        for path in ("/health", "/v1/models"):
            try:
                self._get(path)
                return True
            except Exception:
                continue
        return False

    def wait_ready(self, timeout_s: float = 300.0, interval_s: float = 2.0) -> bool:
        """Poll /health until llama-server reports ready (model loaded, mmap done).
        
        Args:
            timeout_s: Maximum seconds to wait for readiness
            interval_s: Polling interval between checks
            
        Returns:
            True if server becomes ready within timeout, False otherwise
        """
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.is_ready():
                return True
            time.sleep(interval_s)
        return False

    def tokenize(self, content: str) -> list[int] | None:
        try:
            data = self._post("/tokenize", {"content": content})
            tokens = data.get("tokens")
            if isinstance(tokens, list):
                return [int(t) for t in tokens]
        except Exception:
            return None
        return None

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.base_url.rstrip("/") + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"llama server HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"llama server unavailable: {exc.reason}") from exc

    def _get(self, path: str) -> dict[str, Any]:
        req = request.Request(self.base_url.rstrip("/") + path, method="GET")
        try:
            with request.urlopen(req, timeout=self.health_timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {"status": "ok"}
        except error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError("llama server endpoint not found") from exc
            raise RuntimeError(f"llama server HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"llama server unavailable: {exc.reason}") from exc


class LlamaCppBackend(InferenceBackend):
    """Token-level speculative decoding via llama.cpp --model-draft on
    tier-spec; draft tokens are produced by the draft model and verified by
    the target using top-τ acceptance (G14) and n-gram fallback (G13)."""

    slot_dir: Path | None = None

    def __init__(
        self,
        name: str = "llama_cpp",
        base_url: str = "http://127.0.0.1:8080",
        tier: int = 0,
        *,
        kleidiai: bool | None = None,
        continuous_batching: bool = True,
        prefix_caching: bool = True,
        speculation: bool = False,
        supervisor: ProcessSupervisor | None = None,
        managed_command: list[str] | None = None,
        draft_base_url: str | None = None,
        draft_command: list[str] | None = None,
        numa_bind: list[str] | None = None,
        telemetry: TelemetryExporter | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.tier = tier
        self.draft_base_url = (
            draft_base_url or os.getenv("NSA_TIER_SPEC_URL", "")
        ).strip()
        self._spec_url = self.draft_base_url
        self._draft_client = (
            LlamaHttpClient(base_url=self.draft_base_url)
            if self.draft_base_url
            else None
        )
        env_k = os.getenv("NSA_DIPA_KLEIDIAI", "").strip() in {"1", "true", "TRUE", "yes"}
        self._kleidiai = env_k if kleidiai is None else kleidiai
        self.capabilities = BackendCapabilities(
            streaming=True,
            batching=True,
            continuous_batching=continuous_batching,
            prefill_decode_split=False,  # honest: OpenAI chat path is fused
            prefix_caching=prefix_caching,
            tokenize=True,
            speculation=bool(speculation or self._draft_client is not None),
            self_speculation=bool(speculation or self._draft_client is not None),
            kleidiai=self._kleidiai,
            device_classes=(DeviceClass.CPU,),
        )
        self._client = LlamaHttpClient(base_url=base_url)
        slot_path = self.slot_dir or _resolve_slot_dir()
        self._slots = SlotClient(base_url, slot_dir=slot_path)
        total_slots = int(os.getenv("NSA_LLAMA_SLOTS", "4"))
        registry = SlotRegistry(total_slots)
        if os.getenv("NSA_RADIX_ENABLED", "1").strip() not in {"0", "false", "False"}:
            self._okf_affinity = OkfSlotAffinity()
            self._slot_router: SlotRouter = RadixSlotRouter(
                registry=registry,
                okf_affinity=self._okf_affinity,
            )
        else:
            self._okf_affinity = None
            self._slot_router = SlotRouter(registry=registry)
        self._telemetry = telemetry
        self._supervisor = supervisor
        self._managed_command = managed_command
        self._draft_command = list(draft_command) if draft_command else None
        self._numa_bind = numa_bind
        self._verifier = KleidiaiVerifier(
            require=os.getenv("NSA_REQUIRE_KLEIDIAI", "0").strip()
            in {"1", "true", "TRUE", "yes"}
        )
        self._kleidiai_active: bool | None = None

        # Initialize KV bridge for zero-copy SHM transfer
        self._kv_bridge: MAKStoLlamaKVBridge | None = None
        if os.getenv("NSA_KV_SHM_BRIDGE", "0").strip() in {"1", "true", "TRUE", "yes"}:
            # MAKS manager would be injected; for now create bridge with None manager
            # The MAKS manager should be set via set_maks_manager() before use
            self._kv_bridge = MAKStoLlamaKVBridge(maks_manager=None, slot_client=self._slots)

    def set_maks_manager(self, maks_manager: Any) -> None:
        """Set the MAKS manager for the KV bridge."""
        if self._kv_bridge is not None:
            self._kv_bridge._maks_manager = maks_manager

    def configure_draft(
        self,
        *,
        draft_base_url: str,
        draft_command: list[str] | None = None,
        speculation: bool = True,
    ) -> None:
        """Attach a draft llama-server endpoint for speculative decoding."""
        self.draft_base_url = draft_base_url.strip()
        self._spec_url = self.draft_base_url
        self._draft_client = (
            LlamaHttpClient(base_url=self.draft_base_url)
            if self.draft_base_url
            else None
        )
        self._draft_command = list(draft_command) if draft_command else None
        active = bool(speculation and self._draft_client is not None)
        self.capabilities.speculation = active
        self.capabilities.self_speculation = active

    def record_spec_verify(self, draft: Any, *, accepted: bool) -> None:
        """Record ASR metrics for a draft verification outcome (no-op if spec URL unset)."""
        if not self._spec_url:
            return
        words: list[str] = []
        tokens = getattr(draft, "tokens", None)
        if tokens:
            words = [str(getattr(t, "text", "") or "") for t in tokens]
        else:
            text = str(getattr(draft, "text", "") or "")
            words = text.split() if text.strip() else []
        n = len(words)
        ASR_METRICS.inc("asr_draft_tokens_total", float(n))
        ASR_METRICS.inc("asr_verify_calls_total", 1.0)
        if accepted:
            ASR_METRICS.inc("asr_accepted_tokens_total", float(n))

    def _probe_kleidiai_runtime(self) -> bool:
        """Check llama-server for KleidiAI kernel evidence (not env assumption)."""
        verifier = KleidiaiVerifier(require=False)
        for path in ("/props", "/health"):
            try:
                data = self._client._get(path)
                text = json.dumps(data) if isinstance(data, (dict, list)) else str(data)
                verifier.feed_many(text)
            except Exception:
                continue
        result = verifier.result()
        active = bool(result.kernel_ok)
        self._kleidiai_active = active
        self.capabilities.kleidiai = active or self._kleidiai
        return active

    @property
    def kleidiai_active(self) -> bool:
        if self._kleidiai_active is None:
            self._probe_kleidiai_runtime()
        return bool(self._kleidiai_active)

    def start(self) -> None:
        if self._supervisor is not None and self._managed_command:
            self._supervisor.start(
                self.name,
                self._managed_command,
                base_url=self.base_url,
                numa_bind=self._numa_bind,
            )
            ok = self._supervisor.wait_kleidiai(self.name, timeout_s=180.0)
            self.capabilities.kleidiai = bool(ok or self._kleidiai)
            draft_command = self._draft_command or _derive_draft_command(
                self._managed_command,
                self.draft_base_url,
            )
            if draft_command and self.draft_base_url:
                self._supervisor.start_draft(
                    self.name,
                    draft_command,
                    base_url=self.draft_base_url,
                )
        self._probe_kleidiai_runtime()

    def stop(self) -> None:
        if self._supervisor is not None:
            self._supervisor.stop(self.name)

    def warmup(self, model: str | None = None) -> None:
        self._client.is_ready()

    def wait_ready(self, timeout_s: float = 300.0) -> bool:
        """Wait for llama-server to be fully ready (model loaded, mmap complete).
        
        Delegates to the HTTP client's wait_ready method.
        
        Args:
            timeout_s: Maximum seconds to wait for readiness
            
        Returns:
            True if server becomes ready within timeout, False otherwise
        """
        return self._client.wait_ready(timeout_s=timeout_s)

    def tokenize(self, text: str) -> list[int]:
        tokens = self._client.tokenize(text)
        if tokens is not None:
            return tokens
        return list(range(max(1, len(text.split()))))

    async def health(self) -> HealthStatus:
        t0 = time.perf_counter()
        ready = await asyncio.to_thread(self._client.is_ready)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        details: dict[str, Any] = {
            "base_url": self.base_url,
            "kleidiai": self.capabilities.kleidiai,
            "continuous_batching": self.capabilities.continuous_batching,
            "prefix_caching": self.capabilities.prefix_caching,
            "prefill_decode_split": self.capabilities.prefill_decode_split,
            "draft_base_url": self.draft_base_url,
        }
        if self._supervisor is not None:
            details["supervisor"] = self._supervisor.snapshot().get(self.name, {})
        if ready:
            self._probe_kleidiai_runtime()
            details["kleidiai_active"] = self.kleidiai_active
            details["cpu_features"] = asdict(probe_cpu_features())
            try:
                details["slot_busy_ratio"] = await asyncio.to_thread(
                    self._slots.busy_ratio
                )
            except Exception:
                pass
            return HealthStatus(
                state=HealthState.HEALTHY,
                latency_ms=latency_ms,
                message="llama.cpp ready",
                details=details,
            )
        return HealthStatus(
            state=HealthState.UNHEALTHY,
            latency_ms=latency_ms,
            message="llama.cpp unavailable",
            details=details,
        )

    async def draft_health(self) -> HealthStatus:
        t0 = time.perf_counter()
        details = {"base_url": self.draft_base_url}
        if self._draft_client is None:
            return HealthStatus(
                state=HealthState.UNKNOWN,
                latency_ms=0.0,
                message="draft backend not configured",
                details=details,
            )
        ready = await asyncio.to_thread(self._draft_client.is_ready)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if ready:
            return HealthStatus(
                state=HealthState.HEALTHY,
                latency_ms=latency_ms,
                message="draft llama.cpp ready",
                details=details,
            )
        return HealthStatus(
            state=HealthState.UNHEALTHY,
            latency_ms=latency_ms,
            message="draft llama.cpp unavailable",
            details=details,
        )

    async def prefill(self, req: PrefillRequest, ctx: ExecutionContext) -> PrefillResult:
        # Honest fused path: no separate server prefill API on chat endpoint.
        prompt_tokens = _approx_tokens_from_messages(req.messages)
        tokens = await asyncio.to_thread(
            self.tokenize, " ".join(m.get("content", "") for m in req.messages)
        )
        return PrefillResult(
            prefix_tokens=len(tokens) if tokens else prompt_tokens,
            kv_handle=req.kv_handle,
            latency_ms=0.0,
            backend=self.name,
        )

    async def decode(
        self, req: DecodeRequest, ctx: ExecutionContext
    ) -> AsyncIterator[TokenChunk]:
        index = 0
        extra, _slot_meta = _llama_chat_extra(
            session_id=req.session_id,
            messages=req.messages,
            slot_router=self._slot_router,
            tokenize_fn=self.tokenize,
            okf_block_hashes=_okf_hashes_from_request(req),
            cache_prompt_tokens=list(getattr(req, "cache_prompt_tokens", None) or []),
            response_format=_json_response_format(req, tier=self.tier),
        )
        sync_q: queue.Queue[Any] = queue.Queue()

        def _producer() -> None:
            try:
                finished = False
                for line in self._client.chat_stream_raw(
                    req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    extra=extra,
                ):
                    for text, channel in _parse_sse_line(line):
                        if text is None:
                            sync_q.put((None, "answer", True))
                            return
                        if text:
                            sync_q.put((text, channel, False))
                if not finished:
                    sync_q.put((None, "answer", True))
            except Exception as exc:
                sync_q.put(exc)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, sync_q.get)
            if isinstance(item, Exception):
                raise item
            text, channel, finished = item
            if finished:
                yield TokenChunk(text="", index=index, finished=True, channel=channel)
                return
            yield TokenChunk(
                text=str(text),
                index=index,
                finished=False,
                channel=channel,
            )
            index += 1

    async def generate(
        self, req: GenerateRequest, ctx: ExecutionContext
    ) -> GenerateResult:
        slot_reused = False
        extra, slot_meta = _llama_chat_extra(
            session_id=req.session_id,
            messages=req.messages,
            slot_router=self._slot_router,
            tokenize_fn=self.tokenize,
            okf_block_hashes=_okf_hashes_from_request(req),
            cache_prompt_tokens=list(getattr(req, "cache_prompt_tokens", None) or []),
            response_format=_json_response_format(req, tier=self.tier),
        )
        extra, slot_meta = _llama_chat_extra(
            session_id=req.session_id,
            messages=req.messages,
            slot_router=self._slot_router,
            tokenize_fn=self.tokenize,
            okf_block_hashes=_okf_hashes_from_request(req),
            cache_prompt_tokens=list(getattr(req, "cache_prompt_tokens", None) or []),
            response_format=_json_response_format(req, tier=self.tier),
        )  # FIX: prevent NameError
        t0 = time.perf_counter()
        # Prepare telemetry attributes for this generation request
        span_attrs = {
            "backend": self.name,
            "tier": self.tier,
            "slot.reused": slot_reused,
            "gen_ai.arm.kleidiai_active": self.kleidiai_active,
        }

        if isinstance(slot_id, int):
            span_attrs["slot.id"] = slot_id
        tel = self._telemetry
        with tel.span("chat", **span_attrs) if tel else _null_span():
            raw = await asyncio.to_thread(
                self._client.chat,
                req.messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stream=False,
                extra=extra,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        ttft_seconds = latency_ms / 1000.0
        text = _extract_chat_content(raw)
        prompt_tokens = _usage_or_approx(raw, "prompt_tokens", req.messages)
        completion_tokens = _usage_or_approx_text(raw, "completion_tokens", text)
        cached_tokens = _cached_prompt_tokens(raw)
        response_slot = raw.get("id_slot") if isinstance(raw, dict) else None
        if isinstance(response_slot, int):
            slot_id = response_slot
        metrics = {
            "cached_prompt_tokens": float(cached_tokens),
            "slot_reused": 1.0 if slot_reused else 0.0,
            "ttft_seconds": ttft_seconds,
        }
        if isinstance(raw, dict) and isinstance(raw.get("timings"), dict):
            t = raw["timings"]
            for src, dst in (
                ("prompt_ms", "llama_prompt_ms"),
                ("predicted_ms", "llama_predicted_ms"),
                ("prompt_n", "llama_prompt_n"),
                ("predicted_n", "llama_predicted_n"),
                ("prompt_per_second", "llama_prompt_per_second"),
                ("predicted_per_second", "llama_predicted_per_second"),
                ("prompt_per_token_ms", "llama_prompt_per_token_ms"),
                ("predicted_per_token_ms", "llama_predicted_per_token_ms"),
            ):
                if t.get(src) is None:
                    continue
                try:
                    metrics[dst] = float(t[src])
                except (TypeError, ValueError):
                    continue
        if isinstance(slot_id, int):
            metrics["slot_id"] = float(slot_id)
            metrics["id_slot"] = float(slot_id)
        # Use SHM bridge for zero-copy KV transfer when enabled
        if (
            self._kv_bridge is not None
            and req.kv_handle
            and isinstance(slot_id, int)
        ):
            try:
                shm_name = await self._kv_bridge.share_session_kv(
                    req.session_id, slot_id, self
                )
                # Store shm_name in kv_handle for future use
                req.kv_handle = shm_name
                metrics["slot_kv_saved"] = 1.0
                metrics["slot_kv_shm"] = 1.0
            except Exception:
                # Soft-fail: fall back to file-based or skip
                metrics["slot_kv_saved"] = 0.0
                metrics["slot_kv_shm"] = 0.0
        elif (
            _slot_kv_reuse_enabled()
            and req.kv_handle
            and isinstance(slot_id, int)
        ):
            slot_file = self._slots.resolve_filename(req.kv_handle)
            try:
                await asyncio.to_thread(self._slots.kv_export, slot_id, slot_file)
                metrics["slot_kv_saved"] = 1.0
            except Exception:
                # Soft-fail: next turn prefills; do not fail the completion.
                metrics["slot_kv_saved"] = 0.0
        if isinstance(self._slot_router, RadixSlotRouter):
            prompt_text = " ".join(str(m.get("content", "")) for m in req.messages)
            token_ids = await asyncio.to_thread(self.tokenize, prompt_text)
            self._slot_router.record_after_inference(
                token_ids,
                slot_id,
                okf_block_hashes=_okf_hashes_from_request(req),
            )
            snap = self._slot_router.metrics.snapshot()
            metrics["radix_prefix_hit_total"] = float(snap["radix_prefix_hit_total"])
            if slot_meta and slot_meta.get("radix_match_len"):
                metrics["radix_match_len"] = float(slot_meta["radix_match_len"])
        if tel:
            tel.event(
                "neuroswarm.slot.bind",
                session_id=req.session_id,
                slot_id=slot_id,
                tier=self.tier,
                slot_reused=slot_reused,
            )
            if cached_tokens:
                tel.event(
                    "gen_ai.cache_read",
                    session_id=req.session_id,
                    cached_tokens=cached_tokens,
                    gen_ai_usage_cache_read_input_tokens=cached_tokens,
                )
        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            backend=self.name,
            quant=req.quant,
            tier_used=self.tier,
            raw=raw if isinstance(raw, dict) else {},
            metrics=metrics,
        )

    async def generate_with_logits(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_logprobs: int = 5,
        session_id: str = "",
        quant: str = "",
        kv_handle: str | None = None,
        id_slot: int | None = None,
        ctx: ExecutionContext | None = None,
    ) -> GenerateResult:
        """Target forward with OpenAI logprobs / top_logprobs."""
        t0 = time.perf_counter()
        n_probs = int(os.getenv("NSA_LLAMA_N_PROBS", "0") or "0")
        if n_probs <= 0:
            os.environ["NSA_LLAMA_N_PROBS"] = str(int(top_logprobs))
        extra, slot_meta = _llama_chat_extra(
            session_id=session_id,
            messages=messages,
            slot_router=self._slot_router,
            tokenize_fn=self.tokenize,
            okf_block_hashes=None,
            cache_prompt_tokens=[],
            response_format=None,
            request_logprobs=True,
        )
        if isinstance(id_slot, int):
            extra["id_slot"] = id_slot
        slot_id = slot_meta.get("slot_id")
        slot_reused = bool(slot_meta.get("slot_reused"))
        span_attrs = {
            "session_id": session_id,
            "backend": self.name,
            "tier": self.tier,
            "slot.reused": slot_reused,
            "gen_ai.arm.kleidiai_active": self.kleidiai_active,
        }
        if isinstance(slot_id, int):
            span_attrs["slot.id"] = slot_id
        tel = self._telemetry
        with tel.span("chat_logits", **span_attrs) if tel else _null_span():
            raw = await asyncio.to_thread(
                self._client.generate_with_logits,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_logprobs=top_logprobs,
                extra=extra,
            )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = _extract_chat_content(raw)
        prompt_tokens = _usage_or_approx(raw, "prompt_tokens", messages)
        completion_tokens = _usage_or_approx_text(raw, "completion_tokens", text)
        cached_tokens = _cached_prompt_tokens(raw)
        response_slot = raw.get("id_slot") if isinstance(raw, dict) else None
        if isinstance(response_slot, int):
            slot_id = response_slot
        metrics: dict[str, float] = {
            "cached_prompt_tokens": float(cached_tokens),
            "slot_reused": 1.0 if slot_reused else 0.0,
            "ttft_seconds": latency_ms / 1000.0,
            "logits_available": 1.0,
        }
        if isinstance(slot_id, int):
            metrics["slot_id"] = float(slot_id)
            metrics["id_slot"] = float(slot_id)
        if isinstance(self._slot_router, RadixSlotRouter):
            prompt_text = " ".join(str(m.get("content", "")) for m in messages)
            token_ids = await asyncio.to_thread(self.tokenize, prompt_text)
            self._slot_router.record_after_inference(
                token_ids,
                slot_id,
                okf_block_hashes=None,
            )
        return GenerateResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            ttft_ms=latency_ms,
            backend=self.name,
            quant=quant,
            tier_used=self.tier,
            raw=raw if isinstance(raw, dict) else {},
            metrics=metrics,
        )

    async def generate_with_logits_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_logprobs: int = 5,
        session_id: str = "",
        quant: str = "",
        kv_handle: str | None = None,
        id_slot: int | None = None,
        ctx: ExecutionContext | None = None,
        draft: Any = None,
        tau_floor: float = 0.0,
    ) -> AsyncIterator[TokenChunk]:
        """Stream target logprobs; yield accepted draft tokens as steps arrive."""
        from neuroswarm_arm.runtime.armcascade.interfaces.types import (
            LogitsBundle,
            Proposal,
        )
        from neuroswarm_arm.runtime.armcascade.verification.logits_verifier import (
            _parse_step,
            accept_one_draft_position,
        )

        n_probs = int(os.getenv("NSA_LLAMA_N_PROBS", "0") or "0")
        if n_probs <= 0:
            os.environ["NSA_LLAMA_N_PROBS"] = str(int(top_logprobs))
        extra, slot_meta = _llama_chat_extra(
            session_id=session_id,
            messages=messages,
            slot_router=self._slot_router,
            tokenize_fn=self.tokenize,
            okf_block_hashes=None,
            cache_prompt_tokens=[],
            response_format=None,
            request_logprobs=True,
        )
        if isinstance(id_slot, int):
            extra["id_slot"] = id_slot
        elif isinstance(slot_meta.get("slot_id"), int):
            extra.setdefault("id_slot", slot_meta["slot_id"])

        prop = draft if isinstance(draft, Proposal) else None
        if prop is None and draft is not None:
            text = str(getattr(draft, "text", "") or "")
            prop = Proposal.from_text(text, strategy="draft_model")
        if prop is None:
            prop = Proposal.from_text("", strategy="draft_model")

        greedy = float(temperature) == 0.0
        sync_q: queue.Queue[Any] = queue.Queue()
        spec_enabled = bool(self._spec_url)
        stream_t0 = time.perf_counter()

        def _producer() -> None:
            try:
                words = (
                    [t.text for t in prop.tokens]
                    if prop.tokens
                    else (prop.text.split() if prop.text.strip() else [])
                )
                bundle = LogitsBundle(
                    draft_tokens=list(words),
                    draft_token_ids=[
                        t.token_id if t.token_id is not None else 0
                        for t in prop.tokens
                    ]
                    if prop.tokens
                    else [0 for _ in words],
                    draft_logprobs=[float(t.logprob or 0.0) for t in prop.tokens]
                    if prop.tokens
                    else [0.0 for _ in words],
                    draft_ranks=[int(t.rank) for t in prop.tokens]
                    if prop.tokens
                    else [0 for _ in words],
                    top_n=int(top_logprobs),
                )
                position = 0
                index = 0
                finished = False
                seen_steps = 0
                accepted_count = 0

                def _draft_token_at(draft_position: int) -> Any | None:
                    if draft_position < 0:
                        return None
                    if draft_position < len(prop.tokens):
                        return prop.tokens[draft_position]
                    if draft_position < len(words):
                        return Proposal.from_text(
                            words[draft_position],
                            strategy="draft_model",
                        )
                    return None

                def _record_accept_position(pos: Any, draft_position: int) -> None:
                    if not spec_enabled or pos.waiting:
                        return
                    draft_token = _draft_token_at(draft_position)
                    if draft_token is None:
                        return
                    self.record_spec_verify(
                        draft_token,
                        accepted=pos.accepted_token is not None,
                    )

                for line in self._client.generate_with_logits_stream(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_logprobs=top_logprobs,
                    extra=extra,
                ):
                    if finished:
                        break
                    parsed = _parse_sse_logprobs_payload(line)
                    if parsed is None:
                        continue
                    if parsed is False:
                        break
                    # Cumulative content grows; incremental is usually len==1.
                    if len(parsed) > seen_steps:
                        new_entries = parsed[seen_steps:]
                        seen_steps = len(parsed)
                    else:
                        new_entries = parsed
                        seen_steps += len(parsed)
                    for entry in new_entries:
                        step = _parse_step(entry)
                        if step is None:
                            continue
                        bundle.steps.append(step)
                        while True:
                            pos = accept_one_draft_position(
                                bundle,
                                prop,
                                position,
                                greedy=greedy,
                                tau_floor=tau_floor,
                            )
                            if pos.waiting:
                                break
                            _record_accept_position(pos, position)
                            if pos.accepted_token is not None:
                                accepted_count += 1
                                sync_q.put(
                                    TokenChunk(
                                        text=pos.accepted_token,
                                        index=index,
                                        finished=False,
                                        metrics={
                                            "accepted_prefix_len": float(position + 1),
                                            "logits_available": 1.0,
                                        },
                                    )
                                )
                                index += 1
                                position += 1
                                if pos.top_tau_used and pos.residual_or_bonus:
                                    sync_q.put(
                                        TokenChunk(
                                            text=pos.residual_or_bonus,
                                            index=index,
                                            finished=False,
                                            metrics={
                                                "accepted_prefix_len": float(position),
                                                "bonus": 1.0,
                                                "top_tau_used": 1.0,
                                            },
                                        )
                                    )
                                    index += 1
                                    finished = True
                                    break
                                if pos.is_final:
                                    if position == len(bundle.draft_tokens):
                                        bonus = accept_one_draft_position(
                                            bundle,
                                            prop,
                                            position,
                                            greedy=greedy,
                                            tau_floor=tau_floor,
                                        )
                                        _record_accept_position(bonus, position)
                                        if (
                                            not bonus.waiting
                                            and bonus.residual_or_bonus
                                        ):
                                            sync_q.put(
                                                TokenChunk(
                                                    text=bonus.residual_or_bonus,
                                                    index=index,
                                                    finished=False,
                                                    metrics={
                                                        "accepted_prefix_len": float(
                                                            position
                                                        ),
                                                        "bonus": 1.0,
                                                    },
                                                )
                                            )
                                            index += 1
                                            finished = True
                                    break
                                continue
                            if pos.residual_or_bonus:
                                sync_q.put(
                                    TokenChunk(
                                        text=pos.residual_or_bonus,
                                        index=index,
                                        finished=False,
                                        metrics={
                                            "accepted_prefix_len": float(position),
                                            "rejected": 1.0,
                                        },
                                    )
                                )
                                index += 1
                            finished = True
                            break
                        if finished:
                            break
                # Flush pending bonus if all draft accepted and bonus step arrived late.
                if (
                    not finished
                    and position == len(bundle.draft_tokens)
                    and len(bundle.steps) > position
                ):
                    bonus = accept_one_draft_position(
                        bundle,
                        prop,
                        position,
                        greedy=greedy,
                        tau_floor=tau_floor,
                    )
                    _record_accept_position(bonus, position)
                    if not bonus.waiting and bonus.residual_or_bonus:
                        sync_q.put(
                            TokenChunk(
                                text=bonus.residual_or_bonus,
                                index=index,
                                finished=False,
                                metrics={
                                    "accepted_prefix_len": float(position),
                                    "bonus": 1.0,
                                },
                            )
                        )
                        index += 1
                if spec_enabled and accepted_count > 0:
                    ASR_METRICS.observe_tok_per_s(
                        float(accepted_count),
                        time.perf_counter() - stream_t0,
                    )
                sync_q.put(
                    TokenChunk(
                        text="",
                        index=index,
                        finished=True,
                        metrics={"accepted_prefix_len": float(position)},
                    )
                )
            except Exception as exc:
                sync_q.put(exc)

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, sync_q.get)
            if isinstance(item, Exception):
                raise item
            assert isinstance(item, TokenChunk)
            yield item
            if item.finished:
                return

    async def cancel(self, session_id: str) -> None:
        # llama-server cancel is slot-specific; best-effort no-op when unmanaged.
        return None


def _derive_draft_command(
    managed_command: list[str] | None,
    draft_base_url: str,
) -> list[str] | None:
    if not managed_command or not draft_base_url:
        return None
    cmd = [str(part) for part in managed_command]
    draft_path = (
        os.getenv("NSA_DRAFT_MODEL_PATH", "").strip()
        or _arg_after(cmd, "--model-draft", "-md", "--draft-model")
    )
    if not draft_path:
        return None
    port = os.getenv("NSA_DRAFT_PORT", "").strip() or _port_from_url(draft_base_url)
    ctx_size = (
        os.getenv("NSA_DRAFT_CTX_SIZE", "").strip()
        or _arg_after(cmd, "-c", "--ctx-size", "--ctx_size")
        or "2048"
    )
    n_threads = (
        os.getenv("NSA_DRAFT_N_THREADS", "").strip()
        or _arg_after(cmd, "-t", "--threads")
        or "4"
    )
    return [
        _llama_server_executable(cmd),
        "-m",
        draft_path,
        "--port",
        port,
        "-c",
        ctx_size,
        "-t",
        n_threads,
        "--host",
        "127.0.0.1",
    ]


def _arg_after(command: list[str], *names: str) -> str:
    for index, part in enumerate(command[:-1]):
        if part in names:
            return command[index + 1]
    return ""


def _port_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.port is not None:
        return str(parsed.port)
    return "8081"


def _llama_server_executable(command: list[str]) -> str:
    for part in command:
        if "llama-server" in Path(part).name:
            return part
    return "llama-server"


def _okf_hashes_from_request(req: GenerateRequest | DecodeRequest) -> list[str] | None:
    baggage = getattr(req, "baggage", None) or {}
    if not isinstance(baggage, dict):
        return None
    hashes = block_hashes_from_baggage(baggage)
    return hashes or None


def _json_response_format(
    req: GenerateRequest | DecodeRequest,
    *,
    tier: int = 0,
) -> dict[str, Any] | None:
    if tier < 3:
        return None
    if os.getenv("NSA_TIER3_JSON_TOOLS", "1").strip() in {"0", "false", "False"}:
        return None
    baggage = getattr(req, "baggage", None) or {}
    if not isinstance(baggage, dict):
        return None
    fmt = baggage.get("response_format")
    if isinstance(fmt, dict) and fmt:
        return fmt
    if baggage.get("tool_call") or baggage.get("json_tools"):
        return {"type": "json_object"}
    return None


def _llama_logprobs_payload(*, request_logprobs: bool = False) -> dict[str, Any]:
    """Attach llama-server logprob fields when env or caller requests speculation."""
    out: dict[str, Any] = {}
    try:
        n_probs = int(os.getenv("NSA_LLAMA_N_PROBS", "0") or "0")
    except ValueError:
        n_probs = 0
    if n_probs <= 0 and request_logprobs:
        n_probs = int(os.getenv("NSA_LLAMA_N_PROBS_DEFAULT", "5") or "5")
    if n_probs > 0:
        out["n_probs"] = n_probs
        out["logprobs"] = True
        top = int(os.getenv("NSA_LLAMA_TOP_LOGPROBS", str(min(n_probs, 5))) or min(n_probs, 5))
        if top > 0:
            out["top_logprobs"] = top
    return out


def _llama_chat_extra(
    *,
    session_id: str = "",
    messages: list[dict[str, str]] | None = None,
    slot_router: SlotRouter | None = None,
    tokenize_fn: Any | None = None,
    okf_block_hashes: list[str] | None = None,
    cache_prompt_tokens: list[int] | None = None,
    response_format: dict[str, Any] | None = None,
    request_logprobs: bool = False,
    explicit_id_slot: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build llama-server slot reuse payload (id_slot + cache_prompt)."""
    prompt = " ".join(str(m.get("content", "")) for m in (messages or []))
    token_ids: list[int] | None = list(cache_prompt_tokens) if cache_prompt_tokens else None
    if tokenize_fn is not None and prompt and not token_ids:
        token_ids = tokenize_fn(prompt)
    if slot_router is not None:
        if isinstance(slot_router, RadixSlotRouter):
            extra, meta = slot_router.prepare_payload(
                session_id,
                prompt,
                {},
                token_ids=token_ids,
                okf_block_hashes=okf_block_hashes,
            )
            if cache_prompt_tokens:
                extra["cache_prompt_tokens"] = list(cache_prompt_tokens)
            if response_format:
                extra["response_format"] = response_format
            extra.update(_llama_logprobs_payload(request_logprobs=request_logprobs))
            if explicit_id_slot is not None:
                extra["id_slot"] = int(explicit_id_slot)
            return extra, meta
        extra, meta = slot_router.prepare_payload(session_id, prompt, {})
        if cache_prompt_tokens:
            extra["cache_prompt_tokens"] = list(cache_prompt_tokens)
        if response_format:
            extra["response_format"] = response_format
        extra.update(_llama_logprobs_payload(request_logprobs=request_logprobs))
        if explicit_id_slot is not None:
            extra["id_slot"] = int(explicit_id_slot)
        return extra, meta
    extra: dict[str, Any] = {"cache_prompt": True}
    if cache_prompt_tokens:
        extra["cache_prompt_tokens"] = list(cache_prompt_tokens)
    if response_format:
        extra["response_format"] = response_format
    extra.update(_llama_logprobs_payload(request_logprobs=request_logprobs))
    if explicit_id_slot is not None:
        extra["id_slot"] = int(explicit_id_slot)
    return extra, {"slot_reused": False, "slot_id": explicit_id_slot}


def _cached_prompt_tokens(payload: dict[str, Any]) -> int:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return 0
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if isinstance(cached, int):
            return cached
    return 0


@contextmanager
def _null_span() -> Iterator[None]:
    yield None


def _parse_sse_line(line: str) -> list[tuple[str | None, str]]:
    """Return ``(text, channel)`` pieces; ``(None, _)`` means stream done."""
    line = line.strip()
    if not line or line.startswith(":"):
        return []
    if not line.startswith("data:"):
        return []
    data = line[5:].strip()
    if data == "[DONE]":
        return [(None, "answer")]
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return []
    choices = payload.get("choices") or []
    if not choices:
        return []
    delta = choices[0].get("delta") or {}
    # Reasoning models (R1 / Qwen3-thinking) may stream only reasoning_content
    # until the final answer lands in content.
    content = delta.get("content")
    reasoning = delta.get("reasoning_content")
    if reasoning is None:
        reasoning = delta.get("reasoning")
    if content is not None and str(content) != "":
        return [(str(content), "answer")]
    if reasoning is not None and str(reasoning) != "":
        return [(str(reasoning), "thinking")]
    return []


def _parse_sse_logprobs_payload(line: str) -> list[dict[str, Any]] | None | bool:
    """Parse SSE line for logprobs.content entries.

    Returns:
        list of logprob step dicts to append
        None — ignore line
        False — stream done ([DONE])
    """
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return False
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    c0 = choices[0]
    logprobs = c0.get("logprobs")
    content: list[Any] = []
    if isinstance(logprobs, dict):
        raw_content = logprobs.get("content") or []
        if isinstance(raw_content, list):
            content = list(raw_content)
    if not content:
        # Some servers put incremental logprobs on delta.
        delta = c0.get("delta") or {}
        if isinstance(delta, dict):
            dlp = delta.get("logprobs")
            if isinstance(dlp, dict):
                raw_content = dlp.get("content") or []
                if isinstance(raw_content, list):
                    content = list(raw_content)
    out: list[dict[str, Any]] = []
    for entry in content:
        if isinstance(entry, dict):
            out.append(entry)
        elif isinstance(entry, str) and entry.strip():
            out.append({"token": entry, "logprob": 0.0, "top_logprobs": []})
    return out or None


def _extract_chat_content(payload: dict[str, Any]) -> str:
    """Prefer assistant content; fall back to reasoning_content when empty.

    Some llama-server builds / reasoning GGUFs put all tokens in
    ``message.reasoning_content`` and leave ``message.content`` empty
    (especially when max_tokens is consumed by the think phase).
    """
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    text = str(content) if content is not None else ""
    if text.strip():
        return text
    for key in ("reasoning_content", "reasoning"):
        alt = message.get(key)
        if alt is not None and str(alt).strip():
            return str(alt)
    return text


def _approx_word_tokens(text: str) -> int:
    return max(1, len(text.split())) if text.strip() else 0


def _approx_tokens_from_messages(messages: list[dict[str, str]]) -> int:
    parts = [str(m.get("content", "")) for m in messages]
    return _approx_word_tokens(" ".join(parts))


def _usage_or_approx(
    payload: dict[str, Any], key: str, messages: list[dict[str, str]]
) -> int:
    usage = payload.get("usage") or {}
    value = usage.get(key)
    if isinstance(value, int):
        return value
    return _approx_tokens_from_messages(messages)


def _usage_or_approx_text(payload: dict[str, Any], key: str, text: str) -> int:
    usage = payload.get("usage") or {}
    value = usage.get(key)
    if isinstance(value, int):
        return value
    return _approx_word_tokens(text)
