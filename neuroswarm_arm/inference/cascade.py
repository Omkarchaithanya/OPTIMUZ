"""Cascade facade — delegates to DIPA CascadeEngine / DIPARuntime.

Prefer ``neuroswarm_arm.runtime.dipa.build_dipa`` for new code. This module
keeps the historical ``CascadeRouter.handle`` API used by gateway / tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..aqr import pick_quant
from ..governor import ReasoningGovernor
from ..schemas import ChatRequest, ChatResponse
from .llama_client import LlamaClient

if TYPE_CHECKING:
    from ..runtime.dipa import DIPARuntime
    from ..runtime.kv.manager.runtime import KVRuntimeManager


@dataclass
class CascadeRouter:
    """Compat wrapper: prefers injected DIPARuntime; else legacy tier clients."""

    tier1: LlamaClient | None = None
    tier2: LlamaClient | None = None
    tier3: LlamaClient | None = None
    governor: ReasoningGovernor = field(default_factory=ReasoningGovernor)
    confidence_threshold: float = 0.85
    kv_runtime: KVRuntimeManager | None = None
    dipa: DIPARuntime | None = None

    def handle(self, req: ChatRequest, tool_names: list[str] | None = None, **kwargs: Any) -> ChatResponse:
        names = list(tool_names or [])
        if False and self.dipa is not None:
            return self._handle_via_dipa(req, names, **kwargs)
        return self._handle_legacy(req, names, **kwargs)

    def _handle_via_dipa(
        self, req: ChatRequest, tool_names: list[str], **kwargs: Any
    ) -> ChatResponse:
        from ..schemas import PlanState

        prompt_text = req.messages[-1].content if req.messages else ""
        kv_fields = self._kv_plan_fields(req)
        confidence = float(kwargs.get("tool_confidence") or (0.9 if tool_names else 0.4))
        router_result = kwargs.get("router_result")
        plan = PlanState(
            tool_confidence_top1=confidence,
            slo_remaining_ms=4000.0,
            self_consistency_score=min(1.0, len(prompt_text.split()) / 256.0),
            session_id=req.session_id or "",
            **kv_fields,
        )
        cap = self.governor.cap(plan, router_result=router_result)
        system = self.governor.prompt(plan)
        # Mutate a copy so DIPA sees governor system prompt + token cap.
        payload = req.model_copy(
            update={
                "max_tokens": min(req.max_tokens, cap),
            }
        )
        assert self.dipa is not None
        # Attach system prompt via normalize path on ChatRequest-like object
        # by temporarily stuffing into a thin adapter.
        adapter = _GovernorRequest(payload, system, cap, tool_names)
        return self.dipa.handle(
            adapter,
            tool_names,
            tool_schemas=kwargs.get("tool_schemas"),
            tool_confidence=confidence,
            tool_prompt_block=kwargs.get("tool_prompt_block"),
        )

    def _handle_legacy(self, req: ChatRequest, tool_names: list[str], **kwargs: Any) -> ChatResponse:
        # Preserved path for unit tests that construct CascadeRouter with clients only.
        from ..metrics import metrics
        from ..schemas import ChatChoice, ChatUsage, Message, PlanState
        from neuroswarm_arm.runtime.armcascade.classifier.hardness import HardnessTierMapper
        from neuroswarm_arm.runtime.dipa.interfaces.types import InferenceRequest
        import time

        if self.tier1 is None or self.tier2 is None or self.tier3 is None:
            raise RuntimeError("CascadeRouter requires dipa or tier LlamaClients")

        prompt_text = req.messages[-1].content if req.messages else ""
        kv_fields = self._kv_plan_fields(req)
        router_result = kwargs.get("router_result")
        plan = PlanState(
            tool_confidence_top1=0.9 if tool_names else 0.4,
            slo_remaining_ms=4000.0,
            self_consistency_score=min(1.0, len(prompt_text.split()) / 256.0),
            session_id=req.session_id or "",
            **kv_fields,
        )
        cap = self.governor.cap(plan, router_result=router_result)
        quant = pick_quant(req.agent_role, prompt_text)
        messages = [m.model_dump() for m in req.messages]
        messages = [{"role": "system", "content": self.governor.prompt(plan)}] + messages

        infer_req = InferenceRequest(
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            agent_role=req.agent_role,
            session_id=req.session_id or "",
            tool_names=list(tool_names),
        )
        hardness = HardnessTierMapper().classify(infer_req)
        start_tier = hardness.start_tier

        start = time.monotonic()
        tier_used = start_tier
        content = ""
        conf = 0.0
        tier_clients = {1: self.tier1, 2: self.tier2, 3: self.tier3}
        thresholds = {1: self.confidence_threshold, 2: 0.5, 3: 0.0}

        tier_messages = messages
        if req.tools:
            tier_messages = self._inject_xlam_tool_prompt(messages, req.tools)

        for tier_id in range(start_tier, 4):
            client = tier_clients[tier_id]
            kwargs: dict[str, Any] = {}
            if tier_id == 3:
                kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            max_tok = min(req.max_tokens, cap) if tier_id == start_tier else req.max_tokens
            payload = client.chat(
                tier_messages, max_tokens=max_tok, temperature=req.temperature, **kwargs
            )
            content = self._extract_text(payload)
            tool_calls = self._extract_tool_calls(payload, content)
            conf = self._confidence(content)
            tier_used = tier_id
            threshold = thresholds.get(tier_id, 0.0)
            if conf >= threshold or tier_id >= 3:
                break

        elapsed_ms = (time.monotonic() - start) * 1000.0
        prompt_tokens = self._approx_tokens(prompt_text)
        completion_tokens = self._approx_tokens(content)
        metrics.inc("neuroswarm_requests_total")
        metrics.set("neuroswarm_last_request_latency_ms", elapsed_ms)
        metrics.set("neuroswarm_last_tier_used", float(tier_used))
        metrics.inc(f"neuroswarm_cascade_tier_{tier_used}_total")
        metrics.set("neuroswarm_last_thinking_token_cap", float(cap))
        metrics.set("neuroswarm_last_tool_schema_count", float(len(tool_names)))

        return ChatResponse(
            model=req.model,
            tier_used=tier_used,
            content=content,
            choices=[ChatChoice(message=Message(role="assistant", content=content, tool_calls=tool_calls or None))],
            usage=ChatUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            tool_schemas_used=tool_names,
            thinking_token_cap=cap,
            tool_calls=tool_calls,
            metrics={
                "latency_ms": elapsed_ms,
                "tool_schema_count": float(len(tool_names)),
                "quant_policy": quant,
                "tier_used": float(tier_used),
                "prompt_tokens": float(prompt_tokens),
                "completion_tokens": float(completion_tokens),
                "kv_pressure": float(plan.kv_pressure),
                "kv_hit_rate": float(plan.kv_hit_rate),
                "kv_storage_tier": float(plan.kv_storage_tier),
            },
        )

    def _kv_plan_fields(self, req: ChatRequest) -> dict:
        if self.kv_runtime is None:
            return {
                "kv_pressure": min(1.0, max(0.0, req.max_tokens / 16384.0)),
                "kv_hit_rate": 0.0,
                "kv_storage_tier": 1,
                "kv_migration_latency_ms": 0.0,
                "memory_pressure": min(1.0, max(0.0, req.max_tokens / 16384.0)),
            }
        snap = self.kv_runtime.pressure_snapshot()
        # Plane-2 facade may return MAKS dict or KV PressureSnapshot object.
        if isinstance(snap, dict):
            pressure = float(snap.get("pressure", 0.0) or 0.0)
            hit_rate = float(snap.get("hit_rate", 0.0) or 0.0)
            dominant = snap.get("dominant_tier", 1)
            migration = float(snap.get("migration_latency_ms", 0.0) or 0.0)
        else:
            pressure = float(getattr(snap, "pressure", 0.0) or 0.0)
            hit_rate = float(getattr(snap, "hit_rate", 0.0) or 0.0)
            dominant = getattr(snap, "dominant_tier", 1)
            migration = float(getattr(snap, "migration_latency_ms", 0.0) or 0.0)
        return {
            "kv_pressure": pressure,
            "kv_hit_rate": hit_rate,
            "kv_storage_tier": int(dominant),
            "kv_migration_latency_ms": migration,
            "memory_pressure": pressure,
        }

    def _confidence(self, text: str) -> float:
        if not text.strip():
            return 0.0
        score = 0.5
        score += min(0.4, len(text) / 8000.0)
        if "I don't know" in text or "cannot" in text.lower():
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _inject_xlam_tool_prompt(self, messages: list[dict], tools: list[dict]) -> list[dict]:
        """Build Salesforce xLAM native tool-call prompt.

        xLAM-fc-r models are NOT compatible with llama.cpp's native
        tools/tool_choice grammar enforcement (confirmed: sending a
        `tools` field triggers a fatal "peg-native format" parse error).
        Instead, tool definitions must be embedded as instruction text in
        this exact bracketed-block format, and the model replies with a
        raw JSON array in plain content — no native tool_calls field.
        """
        import json

        xlam_tools = []
        for tool in tools:
            if isinstance(tool, dict) and "function" in tool:
                fn = tool["function"]
                xlam_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": {
                        k: v for k, v in fn.get("parameters", {}).get("properties", {}).items()
                    },
                })
            else:
                xlam_tools.append(tool)

        task_instruction = (
            "You are an expert in composing functions. You are given a question "
            "and a set of possible functions. Based on the question, you will "
            "need to make one or more function/tool calls to achieve the "
            "purpose. If none of the functions can be used, point it out."
        )
        format_instruction = (
            "The output MUST strictly be a JSON array, and NO other text MUST "
            "be included.\nExample: "
            '[{"name": "func_name", "arguments": {"arg1": "val1"}}]\n'
            "If no function call is needed, output an empty array: []"
        )
        query = messages[-1]["content"] if messages else ""

        prompt = (
            f"[BEGIN OF TASK INSTRUCTION]\n{task_instruction}\n[END OF TASK INSTRUCTION]\n\n"
            f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
            f"[BEGIN OF FORMAT INSTRUCTION]\n{format_instruction}\n[END OF FORMAT INSTRUCTION]\n\n"
            f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n"
        )

        new_messages = list(messages[:-1])
        new_messages.append({"role": "user", "content": prompt})
        return new_messages

    def _extract_text(self, payload: dict) -> str:
        try:
            msg = payload["choices"][0]["message"]
            content = str(msg.get("content") or "").strip()
            if not content:
                content = str(msg.get("reasoning_content") or "").strip()
            think_end = "" + "/think>"
            content = re.sub(rf"(?s)^[\s\S]*?(?:{re.escape(think_end)})\s*", "", content).strip()
            return content
        except Exception:
            return str(payload)

    def _extract_tool_calls(self, payload: dict, content: str = "") -> list[dict]:
        try:
            native = list(payload["choices"][0]["message"].get("tool_calls") or [])
            if native:
                return native
        except Exception:
            pass
        stripped = (content or "").strip()
        if stripped.startswith("["):
            try:
                import json
                arr = json.loads(stripped)
                if isinstance(arr, list):
                    return [
                        {
                            "type": "function",
                            "function": {
                                "name": call.get("name", ""),
                                "arguments": call.get("arguments", {}),
                            },
                        }
                        for call in arr
                        if isinstance(call, dict)
                    ]
            except Exception:
                pass
        return []

    def _approx_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        return max(1, len(text.split()))


@dataclass
class _GovernorRequest:
    """Thin adapter so DIPA RequestRouter sees system_prompt + thinking cap."""

    req: ChatRequest
    system_prompt: str
    thinking_token_cap: int
    tool_names: list[str]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.req, name)
