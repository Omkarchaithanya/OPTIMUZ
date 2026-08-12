from __future__ import annotations

import asyncio
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar
from uuid import uuid4

from .runtime.haoe.integration.chat import build_chat_handlers, correlation_from_request
from .runtime.router.models import RouteContext
from .schemas import ChatRequest, ChatResponse
from .tools.registry import ToolRegistry
from .tools.semantic_mcp_router import SemanticMCPRouter

if TYPE_CHECKING:
    from .inference.cascade import CascadeRouter
    from .runtime.dipa import DIPARuntime
from .runtime.dipa.speculative.engine import SpeculativeEngine
from .runtime.dipa.speculative.predictor import ToolCallPredictor
from .runtime.haoe import HAOERuntime
from .runtime.kv.manager.runtime import KVRuntimeManager
from .runtime.router import SemanticToolRouter
from .runtime.router.speculative_tool_executor import SpeculativeToolExecutor

_T = TypeVar("_T")


def _tool_spec_enabled() -> bool:
    raw = os.getenv("NSA_TOOL_SPEC_ENABLED", "")
    return raw in {"1", "true", "True", "yes", "YES"}


def _run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run async coroutine from sync gateway (safe with/without running loop)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: list[_T] = []
    error: list[BaseException] = []

    def _target() -> None:
        try:
            result.append(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result[0]


@dataclass
class AgentGateway:
    registry: ToolRegistry
    semantic_router: SemanticMCPRouter
    cascade: CascadeRouter | None = None
    dipa: DIPARuntime | None = None
    kv_runtime: KVRuntimeManager | None = None
    haoe: HAOERuntime | None = None
    tool_router: SemanticToolRouter | None = None
    # Optional Ye/Nichols Speculator (Tier-1 draft). Inject later — Prompt B3 wires call sites.
    tool_predictor: ToolCallPredictor | None = None
    # Optional SpeculativeEngine (arxiv 2512.15834). None = byte-identical legacy path.
    speculative_engine: SpeculativeEngine | None = None
    # B3: Speculative tool executor — predict + pre-warm tool calls
    spec_executor: SpeculativeToolExecutor | None = None
    armora_policy: Any | None = None
    budget_service: Any | None = None
    rcis: Any | None = None
    rpf: Any | None = None
    rof: Any | None = None
    aqr_quant: str = ""
    inference_tier_hint: int = 1
    okf_runtime: Any | None = None
    memory: Any | None = None
    acr: Any | None = None

    def _seed_envelope(
        self, req: ChatRequest
    ) -> tuple[str, float, dict[str, float], list[str]]:
        """Create/freeze BudgetEnvelope; return id, remaining_usd, remaining_map, policies."""
        policies: list[str] = []
        budget = 1.0
        remaining_map: dict[str, float] = {}
        envelope_id = ""

        if self.armora_policy is not None:
            try:
                budget = float(getattr(self.armora_policy, "max_cost_usd", budget) or budget)
                policies = list(getattr(self.armora_policy, "policies", []) or [])
            except Exception:
                pass

        svc = self.budget_service
        if svc is None and self.armora_policy is not None:
            svc = getattr(self.armora_policy, "service", None)

        if svc is not None:
            try:
                if self.rof is not None:
                    from neuroswarm_arm.armora.telemetry.schemas import SpanNames

                    with self.rof.span(SpanNames.ADMISSION):
                        with self.rof.span(SpanNames.POLICY):
                            with self.rof.span(SpanNames.BUDGET):
                                env, state, decision = svc.create_and_freeze_sync(
                                    request_id=getattr(req, "request_id", None)
                                    or f"chat-{uuid4().hex[:12]}",
                                    tenant_id=getattr(req, "tenant_id", "") or "default",
                                    agent_role=req.agent_role or "chat",
                                    agent_id=req.agent_id or "default",
                                    workflow="chat",
                                )
                else:
                    env, state, decision = svc.create_and_freeze_sync(
                        request_id=getattr(req, "request_id", None)
                        or f"chat-{uuid4().hex[:12]}",
                        tenant_id=getattr(req, "tenant_id", "") or "default",
                        agent_role=req.agent_role or "chat",
                        agent_id=req.agent_id or "default",
                        workflow="chat",
                    )
                envelope_id = str(env.envelope_id)
                remaining_map = state.remaining_map()
                budget = float(remaining_map.get("cost_usd", budget))
                if hasattr(self.armora_policy, "bind_request_envelope"):
                    self.armora_policy.bind_request_envelope(envelope_id)
                if self.rof is not None:
                    from neuroswarm_arm.armora.telemetry.instrumentation import bind_envelope

                    bind_envelope(self.rof, envelope_id, agent_id=req.agent_id or "default")
                if not decision.accepted:
                    policies = list(policies) + ["budget_rejected"]
            except Exception:
                envelope_id = ""

        return envelope_id, budget, remaining_map, policies

    def _route_context(self, req: ChatRequest, query: str) -> RouteContext:
        envelope_id, budget, remaining_map, policies = self._seed_envelope(req)
        pressure = 0.0
        if self.kv_runtime is not None:
            try:
                snap = self.kv_runtime.pressure_snapshot()
                pressure = float(
                    getattr(snap, "pressure", 0.0)
                    if not isinstance(snap, dict)
                    else snap.get("pressure", 0.0)
                )
            except Exception:
                pressure = 0.0
        latency = float(remaining_map.get("latency_ms", 4000.0)) if remaining_map else 4000.0
        return RouteContext(
            agent_id=req.agent_id or "default",
            agent_role=req.agent_role or "tool_call",
            conversation_excerpt=query,
            memory_pressure=pressure,
            quantization=self.aqr_quant,
            inference_tier=self.inference_tier_hint,
            budget_remaining_usd=budget,
            budget_envelope_id=envelope_id,
            budget_remaining=remaining_map,
            security_policies=policies,
            latency_slo_ms=latency,
        )

    def _router_has_tool_schemas(self, req: ChatRequest) -> bool:
        """True when semantic router injects ≥1 tool schema for this request."""
        query = req.messages[-1].content if req.messages else ""
        try:
            if hasattr(self.semantic_router, "route_result"):
                ctx = self._route_context(req, query)
                routed = self.semantic_router.route_result(query, context=ctx)
                schemas = list(getattr(routed, "schemas", None) or [])
                if schemas:
                    return True
                names = list(getattr(routed, "tool_names", None) or [])
                return bool(names)
            selected = self.semantic_router.route(query)
            return bool(selected)
        except Exception:
            return False

    def handle_chat(self, req: ChatRequest) -> ChatResponse:
        """Execute chat as a HAOE task graph (never a single unstructured coroutine)."""
        # Speculative tool path — only when engine wired + flag + router schemas.
        # Default (engine=None) is byte-identical to legacy HAOE/cascade path.
        if (
            self.speculative_engine is not None
            and _tool_spec_enabled()
            and self._router_has_tool_schemas(req)
        ):
            return self._attach_runtime_cost_report(
                req, _run_coro_sync(self.speculative_engine.generate(req))
            )

        profile_session_id = ""
        if self.rpf is not None and getattr(getattr(self.rpf, "config", None), "enabled", False):
            try:
                ctx = self.rpf.open_session(
                    request_id=str(
                        getattr(req, "request_id", None)
                        or getattr(req, "session_id", None)
                        or ""
                    ),
                    workflow_id="chat",
                    agent_id=req.agent_id or "default",
                    tenant_id=str(getattr(req, "tenant_id", "") or ""),
                )
                profile_session_id = str(getattr(ctx, "session_id", "") or "")
                try:
                    self.rpf.sample(profile_session_id)
                except Exception:
                    pass
            except Exception:
                profile_session_id = ""
        try:
            if self.rof is not None:
                with self.rof.start_request(
                    request_id=str(
                        getattr(req, "request_id", None)
                        or getattr(req, "session_id", "")
                        or ""
                    ),
                    agent_id=req.agent_id or "default",
                    workflow_id="chat",
                ):
                    return self._handle_chat_body(req)
            return self._handle_chat_body(req)
        finally:
            if profile_session_id and self.rpf is not None:
                try:
                    profile = self.rpf.finalize_sync(profile_session_id)
                    del profile
                except Exception:
                    pass

    async def handle_chat_async(self, req: ChatRequest) -> ChatResponse:
        """Async chat path with speculative tool pre-warming."""
        # Run speculative tool prediction/execution before the main chat
        if self.spec_executor is not None and req.session_id:
            try:
                query = req.messages[-1].content if req.messages else ""
                ctx = self._route_context(req, query) if req.messages else None
                specs = await self.spec_executor.speculate(
                    query, context=ctx.__dict__ if ctx else None, session_id=req.session_id
                )
                # Stash on request for downstream DIPA inference to inject
                # We can't easily modify pydantic model, so we'll inject via metrics
                # after the main chat completes
                req = req.model_copy(update={"_spec_results": specs})
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("speculative_tool_executor_failed: %s", exc)

        # Delegate to sync handler (which runs in thread pool from main.py)
        return await asyncio.to_thread(self.handle_chat, req)

    def _fast_path_eligible(self, routed: Any) -> bool:
        """HAOE bypass when high-confidence and no ACR memory-plane work."""
        if self.acr is not None:
            return False
        return bool(getattr(routed, "high_confidence", False))

    def _handle_chat_fastpath(self, req: ChatRequest, routed: Any) -> ChatResponse:
        """Pre-routed high-conf chat → cascade/DIPA directly (skip HAOE DAG)."""
        import anyio

        from neuroswarm_arm.runtime.router.orchestration import build_routed_inference_hints

        tool_names = list(getattr(routed, "tool_names", None) or [])
        tool_schemas = list(getattr(routed, "schemas", None) or [])
        tool_confidence = float(getattr(routed, "confidence_top1", 0.0) or 0.0)
        tool_high_confidence = bool(getattr(routed, "high_confidence", False))
        tool_prompt_block = ""
        if hasattr(self.semantic_router, "prompt_block"):
            try:
                tool_prompt_block = self.semantic_router.prompt_block(routed) or ""
            except Exception:
                tool_prompt_block = ""
        cost_meta: dict[str, Any] = {}
        try:
            query = req.messages[-1].content if req.messages else ""
            hints = build_routed_inference_hints(
                query,
                routed,
                prompt_block=tool_prompt_block,
                schemas=tool_schemas,
            )
            tool_names = list(hints.tool_names) or tool_names
            tool_schemas = list(hints.tool_schemas) or tool_schemas
            tool_prompt_block = hints.tool_prompt_block or tool_prompt_block
            tool_confidence = float(hints.tool_confidence)
            tool_high_confidence = bool(hints.high_confidence)
            cost_meta = hints.cost_decision.as_dict() if hints.cost_decision else {}
        except Exception:
            cost_meta = {}
        high_conf_budget = int(
            getattr(
                getattr(self.semantic_router, "config", None),
                "high_conf_thinking_budget",
                256,
            )
            or 256
        )
        session_id = req.session_id or f"chat-{uuid4().hex[:16]}"
        if self.kv_runtime is not None:
            self.kv_runtime.create_session(session_id, agent_id=req.agent_id)
            prompt = req.messages[-1].content if req.messages else ""
            payload = prompt.encode("utf-8")[:4096]

            async def _persist() -> None:
                assert self.kv_runtime is not None
                await self.kv_runtime.allocate(
                    session_id,
                    payload,
                    agent_id=req.agent_id,
                )

            anyio.run(_persist)

        engine = self.dipa or self.cascade
        if engine is None:
            raise RuntimeError("AgentGateway requires dipa or cascade")
        handle_kwargs: dict[str, Any] = {
            "tool_schemas": tool_schemas or None,
            "tool_confidence": tool_confidence,
            "tool_prompt_block": tool_prompt_block or None,
        }
        if tool_high_confidence:
            handle_kwargs["tool_high_confidence"] = True
            handle_kwargs["high_conf_thinking_budget"] = high_conf_budget
        try:
            response = engine.handle(
                req.model_copy(update={"session_id": session_id}),
                tool_names,
                **handle_kwargs,
            )
        except TypeError:
            handle_kwargs.pop("tool_high_confidence", None)
            handle_kwargs.pop("high_conf_thinking_budget", None)
            response = engine.handle(
                req.model_copy(update={"session_id": session_id}),
                tool_names,
                **handle_kwargs,
            )
        if self.kv_runtime is not None:

            async def _ckpt() -> None:
                assert self.kv_runtime is not None
                await self.kv_runtime.checkpoint(session_id)

            anyio.run(_ckpt)
        metrics = dict(getattr(response, "metrics", None) or {})
        metrics["haoe_bypassed"] = 1
        metrics["haoe_fast_path"] = 1
        if cost_meta:
            metrics["cost_router_tier"] = cost_meta.get("tier")
            metrics["cost_router_reason"] = cost_meta.get("reason")
        response = response.model_copy(update={"metrics": metrics})
        return self._attach_runtime_cost_report(req, response)

    def _handle_chat_body(self, req: ChatRequest) -> ChatResponse:
        if self.haoe is None:
            return self._handle_chat_inline(req)

        # Pre-route for HAOE fast-path decision (empty tools / no ACR).
        query = req.messages[-1].content if req.messages else ""
        ctx = self._route_context(req, query)
        routed = None
        if hasattr(self.semantic_router, "route_result"):
            try:
                routed = self.semantic_router.route_result(query, context=ctx)
            except Exception:
                routed = None
        if routed is not None and self._fast_path_eligible(routed):
            return self._handle_chat_fastpath(req, routed)

        numa_hint = 0
        if self.kv_runtime is not None:
            numa_hint = self.kv_runtime.block_manager.numa_policy.preferred

        inference = self.dipa or self.cascade
        handlers = build_chat_handlers(
            semantic_router=self.semantic_router,
            cascade=self.cascade,
            inference=inference,
            kv_runtime=self.kv_runtime,
            request=req,
            route_context_factory=self._route_context,
            okf_runtime=self.okf_runtime,
            memory=self.memory,
            acr=self.acr,
        )
        # Reuse pre-route inside handlers if we already routed (avoid double embed).
        if routed is not None and hasattr(self.semantic_router, "route_result"):
            # Stash on request extensions for chat handlers — handlers re-route today;
            # acceptable: fast path is the win; tool path still uses full DAG.
            pass
        ids = correlation_from_request(req)
        if self.rof is not None:
            from neuroswarm_arm.armora.telemetry.instrumentation import bridge_haoe_ids
            from neuroswarm_arm.armora.telemetry.schemas import SpanNames

            bridge_haoe_ids(self.rof, ids)
            with self.rof.span(SpanNames.HAOE_WORKFLOW):
                result = self.haoe.submit_workflow(
                    "chat",
                    handlers,
                    ids=ids,
                    context={"agent_role": getattr(req, "agent_role", "")},
                    numa_node=numa_hint,
                )
        else:
            result = self.haoe.submit_workflow(
                "chat",
                handlers,
                ids=ids,
                context={"agent_role": getattr(req, "agent_role", "")},
                numa_node=numa_hint,
            )
        response = result.output
        if response is None:
            raise RuntimeError("HAOE chat workflow produced no response")
        return self._attach_runtime_cost_report(req, response)

    def _attach_runtime_cost_report(
        self, req: ChatRequest, response: ChatResponse
    ) -> ChatResponse:
        """Emit ExecutionResult + RuntimeCostReport dual output via RCIS."""
        if self.rcis is None or not getattr(self.rcis, "config", None) or not self.rcis.config.enabled:
            return response
        try:
            from neuroswarm_arm.armora.cost.schemas import RequestContext

            usage = getattr(response, "usage", None)
            metrics = dict(getattr(response, "metrics", None) or {})
            tier = int(getattr(response, "tier_used", self.inference_tier_hint) or 1)
            request_id = str(getattr(req, "session_id", None) or response.id)
            execution_id = f"exec-{uuid4().hex[:16]}"
            context = RequestContext(
                request_id=request_id,
                execution_id=execution_id,
                workflow_id="chat",
                user_id=str(getattr(req, "tenant_id", "") or ""),
                agent_id=req.agent_id or "default",
                planner_id="haoe.chat",
                model=str(getattr(response, "model", req.model) or req.model),
                model_tier=f"tier{tier}",
                backend=str(metrics.get("backend", "llama.cpp") or "llama.cpp"),
                quantization=self.aqr_quant or str(metrics.get("quantization", "") or ""),
                prompt_token_estimate=int(getattr(usage, "prompt_tokens", 0) or 0),
                planner_decision_trace={
                    "tier_used": tier,
                    "tool_schemas_used": ([t.get("function", {}).get("name", t.get("name", "unknown")) for t in (request.tools or [])] or list(getattr(response, "tool_schemas_used", []) or []),),  # FIX: prefer request.tools
                },
                extensions={"source": "AgentGateway"},
            )

            def _finalize_cost() -> ChatResponse:
                prediction = self.rcis.predict_sync(context)
                self.rcis.open_session(context, prediction=prediction)
                observed = self.rcis.from_execution_accounting(
                    {
                        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                        "reasoning_tokens": int(metrics.get("reasoning_tokens", 0) or 0),
                        "cpu_seconds": float(metrics.get("cpu_seconds", 0) or 0),
                        "wall_clock_ms": float(metrics.get("latency_ms", 0) or 0),
                        "execution_time_ms": float(metrics.get("latency_ms", 0) or 0),
                        "peak_memory_bytes": float(metrics.get("peak_memory_bytes", 0) or 0),
                        "average_memory_bytes": float(metrics.get("average_memory_bytes", 0) or 0),
                        "kv_cache_bytes": float(metrics.get("kv_bytes", 0) or 0),
                        "tool_schemas_used": ([t.get("function", {}).get("name", t.get("name", "unknown")) for t in (request.tools or [])] or float(len(getattr(response, "tool_schemas_used", []) or [])),),  # FIX: prefer request.tools
                        "retries": float(metrics.get("retries", 0) or 0),
                        "estimated_energy_joules": float(metrics.get("energy_joules", 0) or 0),
                    },
                    context=context,
                    prediction=prediction,
                    extras={
                        "accepted_speculative_tokens": int(
                            metrics.get("accepted_speculative_tokens", 0) or 0
                        ),
                        "rejected_speculative_tokens": int(
                            metrics.get("rejected_speculative_tokens", 0) or 0
                        ),
                        "kv_cache_hits": int(metrics.get("kv_hits", 0) or 0),
                        "kv_cache_misses": int(metrics.get("kv_misses", 0) or 0),
                        "quality_score": float(metrics.get("quality_score", 1.0) or 1.0),
                        "success": True,
                        "wall_time_ms": float(metrics.get("latency_ms", 0) or 0),
                    },
                )
                report = self.rcis.finalize_sync(
                    context=context,
                    observed=observed,
                    predicted=prediction,
                    execution_id=execution_id,
                )
                if self.rpf is not None and getattr(getattr(self.rpf, "config", None), "enabled", False):
                    try:
                        sid = getattr(self.rpf.signal_bus, "_session_hint", "") or ""
                        if sid:
                            self.rpf.record_phase(
                                sid,
                                execution_ms=float(metrics.get("latency_ms", 0) or 0),
                                backend_ms=float(metrics.get("latency_ms", 0) or 0),
                                backend=str(metrics.get("backend", "llama.cpp") or "llama.cpp"),
                                model=str(getattr(response, "model", req.model) or req.model),
                                model_tier=f"tier{tier}",
                                quantization=self.aqr_quant or "",
                                kv_memory_bytes=float(metrics.get("kv_bytes", 0) or 0),
                                accepted_speculative_tokens=int(
                                    metrics.get("accepted_speculative_tokens", 0) or 0
                                ),
                                rejected_speculative_tokens=int(
                                    metrics.get("rejected_speculative_tokens", 0) or 0
                                ),
                            )
                    except Exception:
                        pass
                if self.rof is not None:
                    from neuroswarm_arm.armora.telemetry.schemas import EventType

                    self.rof.emit_builtin(
                        EventType.COST_REPORT_GENERATED,
                        payload={"execution_id": execution_id},
                    )
                return response.model_copy(
                    update={"runtime_cost_report": report.model_dump(mode="json")}
                )

            if self.rof is not None:
                from neuroswarm_arm.armora.telemetry.schemas import SpanNames

                with self.rof.span(SpanNames.COST):
                    return _finalize_cost()
            return _finalize_cost()
        except Exception:
            return response

    def _handle_chat_inline(self, req: ChatRequest) -> ChatResponse:
        """Fallback when HAOE is not wired (tests / degraded mode)."""
        import anyio

        query = req.messages[-1].content if req.messages else ""
        ctx = self._route_context(req, query)
        if hasattr(self.semantic_router, "route_result"):
            routed = self.semantic_router.route_result(query, context=ctx)
            tool_names = list(routed.tool_names)
            tool_schemas = list(routed.schemas)
            tool_confidence = float(routed.confidence_top1)
            tool_high_confidence = bool(getattr(routed, "high_confidence", False))
            tool_prompt_block = self.semantic_router.prompt_block(routed)
            high_conf_budget = int(
                getattr(getattr(self.semantic_router, "config", None), "high_conf_thinking_budget", 256)
                or 256
            )
        else:
            selected_tools = self.semantic_router.route(query)
            tool_names = [t.name for t in selected_tools]
            tool_schemas = []
            tool_confidence = 0.0
            tool_high_confidence = False
            tool_prompt_block = ""
            high_conf_budget = 256
        session_id = req.session_id or f"chat-{uuid4().hex[:16]}"
        if self.kv_runtime is not None:
            self.kv_runtime.create_session(session_id, agent_id=req.agent_id)
            prompt = req.messages[-1].content if req.messages else ""
            payload = prompt.encode("utf-8")[:4096]

            async def _persist() -> None:
                assert self.kv_runtime is not None
                await self.kv_runtime.allocate(
                    session_id,
                    payload,
                    agent_id=req.agent_id,
                )

            anyio.run(_persist)

        engine = self.dipa or self.cascade
        if engine is None:
            raise RuntimeError("AgentGateway requires dipa or cascade")
        handle_kwargs = {
            "tool_schemas": tool_schemas or None,
            "tool_confidence": tool_confidence,
            "tool_prompt_block": tool_prompt_block or None,
        }
        if tool_high_confidence:
            handle_kwargs["tool_high_confidence"] = True
            handle_kwargs["high_conf_thinking_budget"] = high_conf_budget
        try:
            response = engine.handle(
                req.model_copy(update={"session_id": session_id}),
                tool_names,
                **handle_kwargs,
            )
        except TypeError:
            handle_kwargs.pop("tool_high_confidence", None)
            handle_kwargs.pop("high_conf_thinking_budget", None)
            response = engine.handle(
                req.model_copy(update={"session_id": session_id}),
                tool_names,
                **handle_kwargs,
            )
        if self.kv_runtime is not None:

            async def _ckpt() -> None:
                assert self.kv_runtime is not None
                await self.kv_runtime.checkpoint(session_id)

            anyio.run(_ckpt)
        return self._attach_runtime_cost_report(req, response)