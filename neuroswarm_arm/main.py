from __future__ import annotations

import os
import platform
from pathlib import Path
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .aqr import pick_quant
from .armora import ArmoraBudgetPolicy, build_armora, build_budget_service, build_rcis, build_rof, build_rpf
from .armora.profiling.arop_provider import ProfilingObservationProvider
from .armora.telemetry.bridges import (
    BudgetTelemetrySource,
    CallablePrometheusSource,
    MetricsStoreSource,
    RCISTelemetrySource,
    ROFCostTelemetryBridge,
    RPFTelemetrySource,
)
from .armora.telemetry.bridges.arop_provider import ROFObservationProvider
from .armora.telemetry.bridges.performix import PerformixMetricSource
from .armora.telemetry.middleware import ROFMiddleware
from .config import get_config
from .evolution import build_arop, load_arop_config
from .evolution.api import create_arop_router
from .evolution.performix_client import PerformixClient
from .gateway import AgentGateway
from .governor import ReasoningGovernor
from .inference.cascade import CascadeRouter
from .inference.llama_client import LlamaClient
from .metrics import metrics
from .metrics.bridges import PlaneMetricBridge, RMFObservationProvider
from .metrics.lifecycle import build_rmf
from .metrics.middleware import install_rmf_middleware
from .memory.mem0_client import build_memory
from .runtime.dipa import build_dipa
from .runtime.haoe import build_haoe
from .runtime.kv.api import create_kv_router
from .runtime.kv.factory import build_kv_runtime
from .runtime.kv.utils.config import KVRuntimeConfig
from .runtime.maks import build_maks, load_maks_config
from .runtime.dipa.cache.maks_connector import MAKSConnector
from .runtime.okf import build_okf
from .runtime.okf.api import create_okf_router
from .runtime.okf.factory import OKFConfig
from .runtime.acr import build_acr
from .runtime.rtg import build_rtg
from .runtime.rtg.hooks import DIPAReasoningHook
from .runtime.router import build_router, create_tool_router, load_router_config
from .runtime.router.speculative_tool_executor import SpeculativeToolExecutor
from .runtime.router.tool_call_predictor import ToolCallPredictor
from .runtime.router.tool_output_cache import ToolOutputCache
from .schemas import ChatRequest
from .tools.registry import ToolRegistry
from .tools.semantic_mcp_router import SemanticMCPRouter


cfg = get_config()
router_cfg = load_router_config()
router_cfg.top_k = cfg.router_top_k
router_cfg.tool_metadata_root = cfg.tool_metadata_root
router_cfg.okf_root = cfg.okf_root
router_cfg.mem_store = cfg.mem_store

# Runtime Metrics Framework — owns metrics; Prometheus is exporter only
rmf = build_rmf()
metrics.bind(rmf)

tool_router = build_router(router_cfg, metrics_bridge=metrics, start_sync=True)
registry = ToolRegistry()
registry.bind(tool_router.registry)
semantic_router = SemanticMCPRouter(
    registry=registry,
    top_k=cfg.router_top_k,
)
semantic_router.bind(tool_router)

# Speculative tool execution layer (B3) — additive, disabled by default via env
_tool_cache = ToolOutputCache()
_tool_predictor = ToolCallPredictor(
    registry=tool_router.registry,
    embedder=tool_router.embedder,
    tier1_url=cfg.tier1_url,
)
_spec_executor = SpeculativeToolExecutor(
    cache=_tool_cache,
    predictor=_tool_predictor,
)

kv_cfg = KVRuntimeConfig(
    root=cfg.kv_store,
    block_size_tokens=cfg.kv_block_size,
    pressure_threshold=cfg.kv_pressure_threshold,
    ram_budget_bytes=cfg.kv_ram_budget,
    compression=cfg.kv_compression,
    redis_url=cfg.kv_redis_url,
    sharing_backend=cfg.kv_sharing_backend,
    enable_background_migration=cfg.kv_bg_migration,
)
kv_runtime = build_kv_runtime(kv_cfg, metrics_bridge=metrics)

maks_cfg = load_maks_config(cfg.kv_store)
maks_runtime = build_maks(
    maks_cfg,
    kv_runtime=kv_runtime,
    metrics_bridge=metrics,
    enable_scheduler=False,
)
maks_connector = MAKSConnector(sharing=maks_runtime)

# ARMORA Runtime Observability Framework — single telemetry control plane
rof = build_rof()
rof.register_metric_source(MetricsStoreSource(metrics))
rof.register_metric_source(PerformixMetricSource("work/haoe/performix_snapshot.json"))

# ARMORA Budget Envelope service — single per-process ledger; shared with MAKS policy
budget_service = build_budget_service(okf_root=cfg.okf_root)
rof.register_metric_source(BudgetTelemetrySource(budget_service))
# ARMORA Runtime Cost Intelligence System — learning signal (not admit gate)
rcis = build_rcis()
# Wire RCIS cost reports into ROF counters (nexus_*_tokens) so Grafana token panels move.
rcis.telemetry = ROFCostTelemetryBridge(rcis.telemetry, rof=rof)
rof.register_metric_source(RCISTelemetrySource(rcis))
# ARMORA Runtime Profiling Framework — observation plane (not admit / not cost)
rpf = build_rpf()
rof.register_metric_source(RPFTelemetrySource(rpf))
_armora_policy = getattr(maks_runtime, "armora", None)
if isinstance(_armora_policy, ArmoraBudgetPolicy):
    _armora_policy.service = budget_service
elif _armora_policy is None:
    _armora_policy = ArmoraBudgetPolicy(service=budget_service)
    maks_runtime.armora = _armora_policy

memory = build_memory(cfg.mem_store)
if isinstance(_armora_policy, ArmoraBudgetPolicy):
    _armora_policy.memory = getattr(memory, "neuro", memory)

rtg = build_rtg(
    metrics_bridge=metrics,
    kv_pressure=maks_runtime,
    semantic_router=semantic_router,
    performix_path="work/haoe/performix_snapshot.json",
)
rtg_hook = DIPAReasoningHook(rtg, memory=getattr(memory, "neuro", memory))
governor = ReasoningGovernor(rtg=rtg)

dipa = build_dipa(
    metrics_bridge=metrics,
    tier_urls={
        "tier1": cfg.tier1_url,
        "tier2": cfg.tier2_url,
        "tier3": cfg.tier3_url,
    },
    topology_cores=None,
    maks=maks_connector,
    reasoning_hook=rtg_hook,
    memory=getattr(memory, "neuro", memory),
    tool_router=tool_router,
)
armora = build_armora(
    dipa.engine,
    budget=_armora_policy if isinstance(_armora_policy, ArmoraBudgetPolicy) else None,
)
# Bind DIPA afford gate to shared budget service
try:
    from neuroswarm_arm.armora.budget.dipa_gate import BudgetAffordGate

    _gate = BudgetAffordGate(budget_service)
    planner = getattr(getattr(dipa, "decision_engine", None), "planner", None) or getattr(
        dipa, "planner", None
    )
    if planner is not None and hasattr(planner, "bind_afford_gate"):
        planner.bind_afford_gate(_gate)
except Exception:
    pass
# Inject RCIS planner feedback port (repositories only — no planner mutation)
try:
    _de = getattr(dipa, "decision_engine", None)
    if _de is not None:
        _de.cost_feedback = rcis.feedback
        _de.profiler_feedback = rpf.feedback
except Exception:
    pass
# Compat cascade facade delegates to DIPA.
cascade = CascadeRouter(
    tier1=LlamaClient(cfg.tier1_url),
    tier2=LlamaClient(cfg.tier2_url),
    tier3=LlamaClient(cfg.tier3_url),
    governor=governor,
    confidence_threshold=cfg.cascade_confidence_threshold,
    kv_runtime=kv_runtime,
    dipa=dipa,
)
_affinity_cores = kv_runtime.block_manager.numa_policy.affinity_cores()
haoe = build_haoe(
    metrics_bridge=metrics,
    kv_pressure=maks_runtime.pressure_snapshot,
    fast_cores=_affinity_cores[:4] or list(range(4)),
    slow_cores=_affinity_cores[4:] or list(range(4, 8)),
    topology_cores=_affinity_cores or list(range(8)),
)
# HAOE locality → MAKS allocator hints
from neuroswarm_arm.runtime.maks.models import LocalityHint

maks_runtime.scheduler.set_locality(
    "default",
    LocalityHint(
        numa_node=0,
        affinity_cores=list(_affinity_cores or []),
        agent_id="default",
        priority=0,
    ),
)

okf_runtime = build_okf(
    OKFConfig(
        source_root=cfg.okf_root,
        artifact_root=cfg.okf_artifacts,
        token_budget=cfg.okf_token_budget,
        enabled=cfg.okf_enabled,
    ),
    metrics_bridge=metrics,
)

# Adaptive Context Runtime (ArmCascade Layer 4 Context OS) — wraps Mem0 + OKF
acr_runtime = build_acr(
    work_dir=Path("work/acr"),
    memory=getattr(memory, "neuro", memory),
    okf=okf_runtime if cfg.okf_enabled else None,
    metrics_bridge=metrics,
)
# Late-bind ACR into AWPP predictor (DIPA builds before ACR)
try:
    _awpp_conn = getattr(dipa, "awpp", None)
    if _awpp_conn is not None and hasattr(_awpp_conn, "bind_runtime"):
        _awpp_conn.bind_runtime(acr=acr_runtime)
except Exception:
    pass

def _tool_spec_enabled() -> bool:
    raw = os.getenv("NSA_TOOL_SPEC_ENABLED", "")
    return raw in {"1", "true", "True", "yes", "YES"}


speculative_engine = None
_mcp_manager = None
if _tool_spec_enabled():
    import asyncio as _asyncio

    from neuroswarm_arm.runtime.dipa.backends.llama_cpp.backend import LlamaHttpClient
    from neuroswarm_arm.runtime.dipa.speculative.engine import SpeculativeEngine
    from neuroswarm_arm.runtime.dipa.speculative.executor import SpeculativeExecutor
    from neuroswarm_arm.runtime.dipa.speculative.predictor import ToolCallPredictor
    from neuroswarm_arm.runtime.dipa.speculative.tool_cache import ToolOutputCache
    from neuroswarm_arm.runtime.router.mcp_executor import call_tool, get_mcp_manager

    class _CascadeGenerateAdapter:
        """Wrap sync CascadeRouter.handle as async SpeculativeEngine cascade."""

        def __init__(self, cascade_router: Any, router: Any) -> None:
            self._cascade = cascade_router
            self._router = router

        async def generate(self, request: ChatRequest):
            def _run():
                tool_names: list[str] = []
                kwargs: dict[str, Any] = {}
                try:
                    query = request.messages[-1].content if request.messages else ""
                    if hasattr(self._router, "route_result"):
                        routed = self._router.route_result(query)
                        tool_names = list(getattr(routed, "tool_names", None) or [])
                        schemas = list(getattr(routed, "schemas", None) or [])
                        if schemas:
                            kwargs["tool_schemas"] = schemas
                        conf = float(getattr(routed, "confidence_top1", 0.0) or 0.0)
                        kwargs["tool_confidence"] = conf
                except Exception:
                    pass
                return self._cascade.handle(request, tool_names or None, **kwargs)

            return await _asyncio.to_thread(_run)

    class _MCPExecuteAdapter:
        """Public MCPManager execute surface for SpeculativeExecutor."""

        def __init__(self, manager: Any, tool_registry: Any = None) -> None:
            self._manager = manager
            self._registry = tool_registry

        def _resolve_tool_id(self, tool_name: str) -> str:
            """Map draft short names (echo) / display names → registry ids (echo.echo)."""
            name = (tool_name or "").strip()
            if not name:
                return name
            if self._manager.is_executable(name):
                return name
            lower = name.lower()
            compact = "".join(ch for ch in lower if ch.isalnum())
            # Prefer exact registry id / leaf match among reconciled tools.
            executable = list(getattr(self._manager, "executable_tools", None) or [])
            for tid in executable:
                if tid.lower() == lower:
                    return tid
                leaf = tid.split(".", 1)[-1].lower()
                if leaf == lower or leaf == compact:
                    return tid
            # Fall back to registry catalog (name / id / aliases).
            reg = self._registry
            tools: list[Any] = []
            if reg is not None:
                raw_tools = getattr(reg, "tools", None)
                if isinstance(raw_tools, dict):
                    tools = list(raw_tools.values())
                elif hasattr(reg, "all"):
                    try:
                        tools = list(reg.all() or [])
                    except Exception:
                        tools = []
                elif isinstance(raw_tools, list):
                    tools = list(raw_tools)
            for tool in tools:
                tid = str(getattr(tool, "id", "") or "")
                tname = str(getattr(tool, "name", "") or "")
                tcompact = "".join(ch for ch in tname.lower() if ch.isalnum())
                if tid.lower() == lower or tname.lower() == lower or tcompact == compact:
                    return tid or name
                aliases = list(getattr(tool, "aliases", None) or [])
                if any(str(a).lower() == lower for a in aliases):
                    return tid or name
                if tid and tid.split(".", 1)[-1].lower() in {lower, compact}:
                    return tid
            return name

        async def execute(self, tool_name: str, args: dict, **kwargs: Any) -> Any:
            del kwargs
            return await call_tool(self._resolve_tool_id(tool_name), args, pool=self._manager)

    _mcp_manager = get_mcp_manager()
    _spec_cache = ToolOutputCache()
    _draft = LlamaHttpClient(base_url=cfg.tier1_url)
    _predictor = ToolCallPredictor(
        draft_client=_draft,
        registry=registry,
        semantic_router=semantic_router,
    )
    _inflight = _asyncio.Semaphore(int(os.getenv("NSA_TOOL_SPEC_INFLIGHT", "4") or 4))
    _executor = SpeculativeExecutor(
        mcp_manager=_MCPExecuteAdapter(_mcp_manager, registry),
        cache=_spec_cache,
        inflight_sem=_inflight,
    )
    speculative_engine = SpeculativeEngine(
        predictor=_predictor,
        executor=_executor,
        cascade=_CascadeGenerateAdapter(cascade, semantic_router),
        cache=_spec_cache,
        metrics=metrics,
    )

gateway = AgentGateway(
    registry=registry,
    semantic_router=semantic_router,
    cascade=cascade,
    dipa=dipa,
    kv_runtime=kv_runtime,
    haoe=haoe,
    tool_router=tool_router,
    armora_policy=_armora_policy,
    budget_service=budget_service,
    rcis=rcis,
    rpf=rpf,
    rof=rof,
    aqr_quant=pick_quant("tool_call"),
    inference_tier_hint=1,
    okf_runtime=okf_runtime,
    memory=memory,
    acr=acr_runtime,
    speculative_engine=speculative_engine,
    spec_executor=_spec_executor,
)
performix = PerformixClient()

arop_cfg = load_arop_config(work_dir=Path("work/arop"), okf_root=cfg.okf_root)
_aqr = None
try:
    _reg = getattr(dipa, "registry", None)
    if _reg is not None and hasattr(_reg, "get"):
        _aqr = _reg.get("aqr")
except Exception:
    _aqr = None
if _aqr is None:
    _qr = getattr(dipa, "quant_router", None)
    if _qr is not None:
        _aqr = getattr(_qr, "connector", None)
_aqr = _aqr or getattr(dipa, "aqr", None) or getattr(dipa, "quant_connector", None)
arop = build_arop(
    arop_cfg,
    memory=memory,
    metrics_bridge=metrics,
    # Live ASCR sink is cascade_engine (not dipa.ascr / dipa.cascade — those are missing).
    ascr=getattr(dipa, "cascade_engine", None),
    rtg=rtg,
    router=tool_router,
    haoe=haoe,
    maks=maks_runtime,
    aqr=_aqr,
    rcis=rcis,
)
# Per-request thresholds resolve from AROP PolicyRegistry (rule/bandit policies).
# Not online PPO — ADR 0005.
_cascade = getattr(dipa, "cascade_engine", None)
if _cascade is not None and hasattr(_cascade, "set_threshold_agent"):
    try:
        _cascade.set_threshold_agent(arop.policy_agent)
    except Exception:
        pass
elif _cascade is not None and hasattr(getattr(_cascade, "thresholds", None), "agent"):
    try:
        _cascade.thresholds.agent = arop.policy_agent
    except Exception:
        pass
try:
    arop.aggregator.add(ROFObservationProvider(rof))
except Exception:
    pass
try:
    arop.aggregator.add(ProfilingObservationProvider(rpf))
except Exception:
    pass
try:
    arop.aggregator.add(RMFObservationProvider(rmf))
except Exception:
    pass
# Late-bind KV / ACR / HAOE metric sources into ROF scrape registry
rof.register_metric_source(CallablePrometheusSource("kv", kv_runtime.telemetry))
try:
    rof.register_metric_source(
        CallablePrometheusSource("acr", acr_runtime, method="prometheus_text")
    )
except Exception:
    pass

# RMF plane bridges — normalize subsystem metrics into the registry
_rmf_bridges = PlaneMetricBridge(rmf)
_rmf_bridges.wire_budget(budget_service)
_rmf_bridges.wire_rcis(rcis)
_rmf_bridges.wire_haoe(haoe)
_rmf_bridges.wire_dipa(dipa)
_rmf_bridges.wire_maks(maks_runtime)
_rmf_bridges.wire_kv(kv_runtime)
try:
    _rmf_bridges.wire_acr(acr_runtime)
except Exception:
    pass
try:
    _awpp = getattr(dipa, "awpp", None) or getattr(dipa, "warm", None)
    if _awpp is not None:
        _rmf_bridges.wire_awpp(_awpp)
except Exception:
    pass
try:
    _rmf_bridges.wire_arop(arop)
except Exception:
    pass
# ROF meter series only (not full ROF scrape — avoids RMF↔MetricsStore recursion)
rmf.register_source(rof.meter.export_prometheus)

app = FastAPI(title="NeuroSwarm-Arm", version="0.1.0")
app.add_middleware(ROFMiddleware, rof=rof)
install_rmf_middleware(app, rmf=rmf)
app.include_router(create_kv_router(kv_runtime))
from .runtime.maks.api import create_maks_router

app.include_router(create_maks_router(maks_runtime))
app.include_router(create_tool_router(tool_router))
app.include_router(create_arop_router(arop))
if cfg.okf_enabled:
    try:
        app.include_router(create_okf_router(okf_runtime.runtime))
    except Exception:
        pass

# Durable long-horizon workflows (Meta Orchestrator + checkpoint + experience)
from .runtime.swarm.api import (
    WorkflowService,
    create_experience_router,
    create_workflow_router,
)

workflow_service = WorkflowService(Path("work/swarm"))
app.include_router(create_workflow_router(workflow_service))
app.include_router(create_experience_router(workflow_service))


@app.on_event("startup")
async def _startup_tool_cache() -> None:
    """Shared Speculative Tool Cache (Nichols et al. §3) for Layer 2."""
    from neuroswarm_arm.runtime.dipa.speculative.tool_cache import ToolOutputCache

    # Reuse engine cache when speculative path is live; else fresh singleton.
    if speculative_engine is not None:
        app.state.tool_cache = speculative_engine._cache  # noqa: SLF001
        app.state.speculative_engine = speculative_engine
        app.state.mcp_manager = _mcp_manager
    else:
        app.state.tool_cache = ToolOutputCache()
        app.state.speculative_engine = None
        app.state.mcp_manager = None


@app.on_event("startup")
async def _startup_mcp_reconcile() -> None:
    """When NSA_MCP_EXECUTE=1, discover tools/list and mark reconciled tools executable."""
    try:
        from neuroswarm_arm.runtime.router.mcp_executor import mcp_execute_enabled

        if not mcp_execute_enabled():
            return
        result = await tool_router.reconcile_mcp_execute()
        import logging

        logging.getLogger(__name__).info("mcp_reconcile_startup %s", result)
    except Exception as exc:  # noqa: BLE001
        import logging

        logging.getLogger(__name__).warning("mcp_reconcile_startup failed: %s", exc)


_arop_loop_task = None


@app.on_event("startup")
async def _startup_arop_loop() -> None:
    """Periodic RuntimeOptimizer.run_once when NSA_AROP_LOOP=1 (auto_promote stays off)."""
    global _arop_loop_task
    if not getattr(arop_cfg, "loop_enabled", False):
        return
    import asyncio
    import logging

    log = logging.getLogger(__name__)
    interval = max(1, int(getattr(arop_cfg, "interval_seconds", 3600) or 3600))

    async def _loop() -> None:
        while True:
            try:
                await asyncio.to_thread(arop.optimizer.run_once)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("arop_loop run_once failed: %s", exc)
            await asyncio.sleep(interval)

    _arop_loop_task = asyncio.create_task(_loop(), name="arop_loop")
    log.info("arop_loop started interval_seconds=%s", interval)


@app.on_event("shutdown")
def _shutdown_runtime() -> None:
    global _arop_loop_task
    if _arop_loop_task is not None:
        try:
            _arop_loop_task.cancel()
        except Exception:
            pass
        _arop_loop_task = None
    if hasattr(app.state, "tool_cache"):
        app.state.tool_cache = None
    try:
        rmf.shutdown()
    except Exception:
        pass
    try:
        rof.shutdown()
    except Exception:
        pass
    try:
        rpf.shutdown()
    except Exception:
        pass
    tool_router.shutdown()
    haoe.shutdown()
    try:
        armora.shutdown()
    except Exception:
        dipa.shutdown()
    maks_runtime.stop()
    kv_runtime.shutdown()


@app.get("/health")
def health() -> dict[str, object]:
    mem_status: dict[str, object] = {"status": "unknown"}
    try:
        neuro = getattr(memory, "neuro", memory)
        if hasattr(neuro, "health"):
            hs = neuro.health()
            mem_status = {
                "healthy": bool(getattr(hs, "healthy", False)),
                "provider": getattr(hs, "provider", ""),
                "details": getattr(hs, "details", {}),
            }
    except Exception as exc:  # noqa: BLE001
        mem_status = {"healthy": False, "error": str(exc)}
    numa_payload: dict[str, object] = {}
    try:
        from neuroswarm_arm.runtime.haoe.topology.numa_status import collect_numa_status

        numa_payload = collect_numa_status().to_dict()
    except Exception as exc:  # noqa: BLE001
        numa_payload = {"error": str(exc)}
    awpp_payload: dict[str, object] = {}
    try:
        _conn = getattr(dipa, "awpp", None)
        if _conn is not None and hasattr(_conn, "status"):
            awpp_payload = dict(_conn.status())
        else:
            awpp_payload = {"status": "unavailable"}
    except Exception as exc:  # noqa: BLE001
        awpp_payload = {"error": str(exc)}
    return {"status": "ok", "memory": mem_status, "numa": numa_payload, "awpp": awpp_payload}


@app.get("/ready")
def ready() -> dict[str, object]:
    """Always HTTP 200 — use body status ready|degraded so bootstrap curl -fsS never 500s."""
    try:
        models = {
            "tier1": {"path": cfg.model_tier1, "exists": Path(cfg.model_tier1).exists()},
            "tier2": {"path": cfg.model_tier2, "exists": Path(cfg.model_tier2).exists()},
            "tier3": {"path": cfg.model_tier3, "exists": Path(cfg.model_tier3).exists()},
        }
        try:
            health_payload = dipa.health()
            backends = health_payload.get("backends", health_payload) if isinstance(health_payload, dict) else {}
        except Exception:
            backends = {}
        llama_ready = {
            name: str(info.get("state", "unknown")) == "healthy"
            for name, info in backends.items()
            if isinstance(info, dict) and name.startswith("tier")
        }
        for tier in ("tier1", "tier2", "tier3"):
            llama_ready.setdefault(tier, False)
        try:
            tools_indexed = len(registry.as_list())
        except Exception:
            tools_indexed = 0
        reasons: list[str] = []
        for tier, model in models.items():
            if not model["exists"]:
                reasons.append(f"{tier} model missing: {model['path']}")
        for tier, is_ready in llama_ready.items():
            if not is_ready:
                reasons.append(f"{tier} llama server unavailable")
        if tools_indexed == 0:
            reasons.append(f"no tools indexed from {cfg.tool_metadata_root}")
        try:
            from neuroswarm_arm.runtime.router.mcp_executor import (
                get_mcp_manager,
                mcp_execute_enabled,
            )

            if mcp_execute_enabled():
                mgr = get_mcp_manager()
                if int(getattr(mgr, "executable_count", None) or len(mgr.executable_tools)) == 0:
                    reasons.append(
                        "MCP execute on but tools/list reconcile incomplete (executable_count=0)"
                    )
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"mcp_manager: {exc}")

        def _safe(label: str, fn):
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"{label}: {exc}")
                return {"error": str(exc)}

        mem_details: dict[str, object] = {}
        try:
            neuro = getattr(memory, "neuro", memory)
            if hasattr(neuro, "health"):
                hs = neuro.health()
                details = getattr(hs, "details", {}) or {}
                mem_details = dict(details) if isinstance(details, dict) else {"raw": details}
                mem_details.setdefault("provider", getattr(hs, "provider", "unknown"))
                mem_details.setdefault("healthy", getattr(hs, "healthy", True))
            # Router history ranker honesty
            hist = getattr(tool_router, "history", None)
            if hist is not None and hasattr(hist, "status"):
                hr = hist.status()
                mem_details["history_ranker"] = hr
                if hr.get("history_ranker_degraded"):
                    reasons.append("history_ranker using json_emergency / degraded memory")
        except Exception as exc:  # noqa: BLE001
            mem_details = {"error": str(exc), "emergency_active": True}

        numa_info: dict[str, object] = {}
        try:
            from neuroswarm_arm.runtime.haoe.topology.numa_status import collect_numa_status

            numa_info = collect_numa_status().to_dict()
        except Exception as exc:  # noqa: BLE001
            numa_info = {"error": str(exc)}
        system = {
            "arch": platform.machine(),
            "cpu_features": _cpu_features(),
            "expected_docker_network": bool(
                os.getenv("NSA_TIER1_URL") or os.getenv("NSA_TIER2_URL") or os.getenv("NSA_TIER3_URL")
            ),
            "numa": numa_info,
        }
        dipa_status = _safe("dipa", dipa.status)
        router_health = _safe("router", tool_router.health)
        kv_status = _safe("kv", kv_runtime.status)
        haoe_status = _safe("haoe", haoe.status)
        rtg_status = _safe("rtg", rtg.status)
        okf_status = _safe("okf", okf_runtime.status) if cfg.okf_enabled else {"enabled": False}
        acr_status = _safe("acr", acr_runtime.health)
        arop_status = _safe("arop", arop.health)
        return {
            "status": "ready" if not reasons else "degraded",
            "system": system,
            "models": models,
            "llama": llama_ready,
            "dipa": dipa_status,
            "tools": {
                "indexed_count": tools_indexed,
                "metadata_root": str(cfg.tool_metadata_root),
                "router": router_health,
            },
            "kv": kv_status,
            "haoe": haoe_status,
            "rtg": rtg_status,
            "okf": okf_status,
            "acr": acr_status,
            "arop": arop_status,
            "memory": mem_details,
            "reasons": reasons,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "degraded",
            "reasons": [f"ready_handler: {exc}"],
            "error": str(exc),
        }


def _cpu_features() -> list[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return []
    text = cpuinfo.read_text(encoding="utf-8", errors="ignore")
    wanted = ("asimd", "sve", "sve2", "i8mm", "dotprod", "bf16", "sme", "sme2")
    return sorted({flag for flag in wanted if re.search(rf"\b{re.escape(flag)}\b", text)})


@app.get("/build-info")
def build_info() -> dict[str, object]:
    """KleidiAI / GGML / Axion feature honesty for operators and demos."""
    features = _cpu_features()
    has_sve2 = "sve2" in features
    has_i8mm = "i8mm" in features
    has_sme = "sme" in features or "sme2" in features
    kleidi_env = {
        "GGML_KLEIDIAI_SME": os.getenv("GGML_KLEIDIAI_SME"),
        "NSA_REQUIRE_KLEIDIAI": os.getenv("NSA_REQUIRE_KLEIDIAI"),
        "NSA_ROUTER_EMBEDDING_BACKEND": os.getenv("NSA_ROUTER_EMBEDDING_BACKEND", "fastembed"),
        "NSA_AROP_CANARY_PCT": os.getenv("NSA_AROP_CANARY_PCT", "5"),
        "NSA_RTG_PPO": os.getenv("NSA_RTG_PPO", "0"),
    }
    return {
        "arch": platform.machine(),
        "cpu_features": features,
        "axion_profile": {
            "expected": "Neoverse-V2",
            "acceleration": "SVE2+I8MM",
            "sme2": False,
            "note": "GCP Axion C4A is Neoverse V2: SVE2+I8MM present; SME2 not available.",
        },
        "detected": {
            "sve2": has_sve2,
            "i8mm": has_i8mm,
            "sme2": has_sme,
            "dotprod": "dotprod" in features,
            "bf16": "bf16" in features,
        },
        "kleidiai": {
            "sme_auto": kleidi_env["GGML_KLEIDIAI_SME"] in (None, "", "auto"),
            "sme_forced_off": kleidi_env["GGML_KLEIDIAI_SME"] in {"0", "false", "off"},
            "require": kleidi_env["NSA_REQUIRE_KLEIDIAI"] in {"1", "true", "yes"},
        },
        "rtg": {
            "default_policy": "bandit",
            "ppo_enabled": kleidi_env["NSA_RTG_PPO"] in {"1", "true", "yes"},
            "note": "PPO is optional offline scaffold; bandit/heuristics are the default live path.",
        },
        "env": kleidi_env,
    }


@app.get("/v1/models")
def models() -> dict[str, object]:
    return {
        "object": "list",
        "data": [
            {"id": "cascade", "object": "model", "owned_by": "neuroswarm"},
            {"id": "tier1", "object": "model", "owned_by": "neuroswarm"},
            {"id": "tier2", "object": "model", "owned_by": "neuroswarm"},
            {"id": "tier3", "object": "model", "owned_by": "neuroswarm"},
        ],
    }


@app.get("/metrics")
def export_metrics(request: Request) -> Response:
    # RMF owns scrape; Prometheus is exporter only.
    if not rmf.check_auth(request.headers.get("Authorization")):
        raise HTTPException(status_code=401, detail="unauthorized")
    accept = request.headers.get("Accept", "")
    # When we merge memory/arop extras, classic Prometheus text avoids OpenMetrics
    # `# EOF` mid-stream (Prometheus: "unexpected data after # EOF").
    want_om = "openmetrics" in accept
    body, ctype = rmf.export("openmetrics" if want_om else "prometheus")
    mem = ""
    try:
        neuro = getattr(memory, "neuro", None)
        if neuro is not None and hasattr(neuro, "runtime"):
            mem = neuro.runtime.metrics.prometheus_text()
    except Exception:
        mem = ""
    arop_txt = ""
    try:
        from neuroswarm_arm.evolution.observation.otel_provider import PrometheusObservationProvider

        prom = next(
            (p for p in arop.aggregator.providers if isinstance(p, PrometheusObservationProvider)),
            None,
        )
        if prom is not None:
            snap = arop.aggregator.snapshot()
            arop_txt = prom.prometheus_text(dict(snap.aggregate))
    except Exception:
        arop_txt = ""
    # Strip every OpenMetrics EOF marker before merge — RMF/extras may already emit one.
    def _strip_eof(text: str) -> str:
        return "\n".join(ln for ln in text.splitlines() if ln.strip() != "# EOF").rstrip()

    core = _strip_eof(body)
    extras = _strip_eof(f"{mem}{arop_txt}")
    if extras:
        payload = f"{core}\n{extras}\n"
    else:
        payload = f"{core}\n"
    bench_path = Path(
        os.getenv(
            "NSA_BENCHMARK_METRICS_PATH",
            "work/benchmarks/arop_closed_loop/benchmark.prom",
        )
    )
    try:
        if bench_path.is_file():
            bench_txt = _strip_eof(
                bench_path.read_text(encoding="utf-8", errors="replace")
            )
            if bench_txt:
                payload = f"{payload.rstrip()}\n{bench_txt}\n"
    except Exception:
        pass
    if want_om:
        payload = f"{payload.rstrip()}\n# EOF\n"
        ctype = "application/openmetrics-text; version=1.0.0; charset=utf-8"
    else:
        # Classic Prometheus text: never emit # EOF (scrape Accept may still prefer OM).
        ctype = "text/plain; version=0.0.4; charset=utf-8"
    return Response(content=payload, media_type=ctype)


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if req.stream:
        return await _chat_stream(req)
    # handle_chat is sync and may call _run_coro_sync (thread.join). Never run it
    # on the event-loop thread — that deadlocks /health and every other request.
    import asyncio as _aio
    import os

    # Branch: async path with speculative tool pre-warming when enabled
    if os.getenv("NSA_SPEC_TOOL_ENABLED", "1") == "1" and not req.stream:
        response = await gateway.handle_chat_async(req)
    else:
        response = await _aio.to_thread(gateway.handle_chat, req)

    # Add speculative tools metrics to response
    if hasattr(req, "_spec_results") and req._spec_results:
        specs = req._spec_results
        metrics_dict = dict(getattr(response, "metrics", None) or {})
        cache_hits = sum(1 for s in specs if s.is_cache_hit)
        metrics_dict["speculative_tools"] = {
            "drafts": len(specs),
            "cache_hits": cache_hits,
            "confirmed": 0,  # incremented by post-call hook
            "prediction_latency_ms": specs[0].prediction_latency_ms if specs else 0.0,
        }
        response = response.model_copy(update={"metrics": metrics_dict})

    return response.model_dump()


async def _chat_stream(req: ChatRequest) -> StreamingResponse:
    """OpenAI SSE stream via ASCREngine.run_stream (spec accept as logits arrive)."""
    import json
    from time import time
    from uuid import uuid4

    from neuroswarm_arm.runtime.dipa.execution.execution_context import ExecutionContext
    from neuroswarm_arm.runtime.dipa.interfaces.types import InferenceRequest

    session_id = req.session_id or f"chat-{uuid4().hex[:16]}"
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    infer_req = InferenceRequest(
        messages=messages,
        model=req.model,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        session_id=session_id,
        agent_role=req.agent_role,
        agent_id=req.agent_id,
        stream=True,
    )
    plan = dipa.decision_engine.decide(infer_req)
    plan.stream = True
    ctx = ExecutionContext(request=infer_req, ids=infer_req.ids)
    ctx.plan = plan
    ctx.quant = plan.quant

    engine = getattr(dipa, "cascade_engine", None)
    if engine is None or not hasattr(engine, "run_stream"):
        raise HTTPException(status_code=501, detail="streaming cascade unavailable")

    chunk_id = f"chatcmpl-{uuid4().hex[:24]}"
    created = int(time())
    model_name = req.model or "cascade"

    async def event_gen():
        accepted_prefix = 0
        mode = "speculative"
        first_sent = False
        async for token in engine.run_stream(infer_req, plan, ctx):
            if token.metrics and "accepted_prefix_len" in token.metrics:
                accepted_prefix = int(token.metrics["accepted_prefix_len"])
            mode = str(ctx.baggage.get("ascr_mode") or mode)
            if token.finished and not token.text:
                # Final metadata comment for clients / evidence.
                yield (
                    f": X-ASCR-Accepted-Prefix: {accepted_prefix}\n"
                    f": X-ASCR-Mode: {mode}\n\n"
                )
                yield "data: [DONE]\n\n"
                return
            if not token.text:
                continue
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token.text}
                        if first_sent
                        else {"role": "assistant", "content": token.text},
                        "finish_reason": None,
                    }
                ],
                "ascr": {
                    "accepted_prefix_len": accepted_prefix,
                    "mode": mode,
                },
            }
            first_sent = True
            yield f"data: {json.dumps(payload)}\n\n"
        mode = str(ctx.baggage.get("ascr_mode") or mode)
        accepted_prefix = int(ctx.baggage.get("ascr_accepted_prefix") or accepted_prefix)
        yield (
            f": X-ASCR-Accepted-Prefix: {accepted_prefix}\n"
            f": X-ASCR-Mode: {mode}\n\n"
        )
        yield "data: [DONE]\n\n"

    # Mode known after first draft decision — use plan speculation strategy hint.
    initial_mode = "speculative"
    meta = dict(plan.metadata or {})
    spec = meta.get("speculation") or {}
    if isinstance(spec, dict) and spec.get("strategy"):
        initial_mode = str(spec["strategy"])

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-ASCR-Mode": initial_mode,
            "X-ASCR-Accepted-Prefix": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/v1/kv-cache/status")
def kv_cache_status(tier: int | None = None) -> dict:
    """Live per-tier llama-server KV slot occupancy (transformer cache, not MAKS)."""
    from neuroswarm_arm.runtime.dipa.backends.llama_cpp.kv_cache_status import (
        fetch_all_tier_kv_cache_status,
        fetch_tier_kv_cache_status,
    )

    if tier is not None:
        statuses = [fetch_tier_kv_cache_status(int(tier))]
    else:
        statuses = fetch_all_tier_kv_cache_status()
    return {"tiers": [s.to_dict() for s in statuses]}


class CacheInvalidateBody(BaseModel):
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    all: bool = False


@app.get("/v1/tools/cache")
async def tools_cache_status() -> dict[str, Any]:
    """Debug: speculative tool-cache size / hit rate / top keys."""
    cache = getattr(app.state, "tool_cache", None)
    if cache is None:
        return {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "top_keys": []}
    snap = cache.snapshot() if hasattr(cache, "snapshot") else dict(cache.metrics())
    snap.setdefault("top_keys", [])
    return snap


@app.post("/v1/tools/cache/invalidate")
async def tools_cache_invalidate(body: CacheInvalidateBody) -> dict[str, int]:
    """Invalidate one tool+args key or clear the entire speculative cache."""
    cache = getattr(app.state, "tool_cache", None)
    if cache is None:
        return {"invalidated": 0}
    if body.all:
        n = await cache.invalidate(all_entries=True)
    else:
        n = await cache.invalidate(body.tool, body.args, all_entries=False)
    return {"invalidated": int(n)}


@app.get("/v1/tools/spec_debug")
def tools_spec_debug(reset: int = 0) -> dict[str, Any]:
    """Demo ring-buffer of last N speculative events (cap 200)."""
    eng = getattr(app.state, "speculative_engine", None) or speculative_engine
    if eng is None or not hasattr(eng, "debug_snapshot"):
        return {"events": [], "count": 0}
    events = eng.debug_snapshot(reset=bool(reset))
    return {"events": events, "count": len(events)}


@app.get("/v1/speculative-tools/stats")
def speculative_tools_stats() -> dict[str, Any]:
    """Stats for the speculative tool executor (B3)."""
    if gateway.spec_executor is None:
        return {"enabled": False}
    return {"enabled": True, **gateway.spec_executor.stats()}


@app.get("/v1/cost/economics")
def cost_economics(limit: int = 200) -> dict:
    return rcis.unit_economics(limit=limit).model_dump()


@app.get("/v1/cost/comparisons")
def cost_comparisons(limit: int = 200) -> dict:
    return dict(rcis.comparison_bundle(limit=limit))


@app.get("/v1/cost/feedback/backends")
def cost_feedback_backends() -> dict:
    from neuroswarm_arm.armora.cost import WorkloadKey

    return rcis.feedback.lowest_cost_backend_sync(WorkloadKey()).model_dump()


@app.post("/bench/run")
def bench_run(payload: dict) -> dict:
    recipe = payload.get("recipe", "system-characterization")
    result = performix.run_recipe(recipe, output=cfg.benchmarks_dir / f"{recipe}.json")
    return result


def serve() -> None:
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    serve()
