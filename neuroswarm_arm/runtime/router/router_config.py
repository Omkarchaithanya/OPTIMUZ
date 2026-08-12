"""Router configuration via NSA_ROUTER_* environment variables."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _b(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "NO"}


@dataclass(slots=True)
class HybridWeights:
    semantic: float = 0.40
    keyword: float = 0.12
    param: float = 0.10
    capability: float = 0.08
    workflow: float = 0.08
    agent_role: float = 0.06
    popularity: float = 0.04
    history_success: float = 0.06
    latency_cost: float = 0.06

    def as_dict(self) -> dict[str, float]:
        return {
            "semantic": self.semantic,
            "keyword": self.keyword,
            "param": self.param,
            "capability": self.capability,
            "workflow": self.workflow,
            "agent_role": self.agent_role,
            "popularity": self.popularity,
            "history_success": self.history_success,
            "latency_cost": self.latency_cost,
        }


@dataclass(slots=True)
class RerankWeights:
    semantic: float = 0.22
    param_compat: float = 0.08
    agent_type: float = 0.06
    workflow_stage: float = 0.06
    conversation: float = 0.08
    history_success: float = 0.08
    latency: float = 0.05
    failure_rate: float = 0.05
    dependencies: float = 0.04
    permissions: float = 0.04
    output_format: float = 0.04
    okf_relevance: float = 0.06
    cost: float = 0.05
    budget: float = 0.04
    confidence: float = 0.05


@dataclass(slots=True)
class RouterConfig:
    top_k: int = 3
    candidate_multiplier: int = 5
    threshold: float = 0.42
    high_conf_gate: float = 0.70
    high_conf_thinking_budget: int = 256
    encoder_name: str = "nomic-embed-text-v1.5"
    embedding_backend: str = "fastembed"
    fastembed_cache_dir: str | None = None
    fallback_dims: int = 64
    matryoshka_dim: int = 256
    ann_backend: str = "turbovec"
    # Use TurboVec IdMapIndex only at/above this tool count; below → exact float32.
    # Default 0: activate TurboVec whenever the wheel imports (exact only as real fallback).
    turbovec_min_tools: int = 0
    metric: str = "cosine"
    cache_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/1"
    index_path: Path = Path("work/router/index")
    snapshot_dir: Path = Path("work/router/snapshots")
    cache_dir: Path = Path("work/router/cache")
    tool_metadata_root: Path = Path("templates/mcp-servers")
    okf_root: Path = Path("okf")
    mem_store: Path = Path("work/memory")
    use_onnx: bool = False
    use_int8: bool = False
    onnx_path: str | None = None
    tokenizer_path: str | None = None
    allow_hash: bool = False
    embed_batch_size: int = 32
    embed_workers: int = 2
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 10_000
    turbovec_bit_width: int = 4
    enable_hot_reload: bool = True
    hot_reload_interval_s: float = 5.0
    affinity_cores: list[int] = field(default_factory=list)
    hybrid: HybridWeights = field(default_factory=HybridWeights)
    rerank: RerankWeights = field(default_factory=RerankWeights)
    otel_enabled: bool = False
    otel_endpoint: str = ""

    @classmethod
    def from_env(cls, *, root: Path | None = None) -> RouterConfig:
        base = root or Path(".")
        hybrid = HybridWeights(
            semantic=_f("NSA_ROUTER_W_SEMANTIC", 0.40),
            keyword=_f("NSA_ROUTER_W_KEYWORD", 0.12),
            param=_f("NSA_ROUTER_W_PARAM", 0.10),
            capability=_f("NSA_ROUTER_W_CAPABILITY", 0.08),
            workflow=_f("NSA_ROUTER_W_WORKFLOW", 0.08),
            agent_role=_f("NSA_ROUTER_W_AGENT", 0.06),
            popularity=_f("NSA_ROUTER_W_POPULARITY", 0.04),
            history_success=_f("NSA_ROUTER_W_HISTORY", 0.06),
            latency_cost=_f("NSA_ROUTER_W_LATENCY_COST", 0.06),
        )
        cores_raw = os.getenv("NSA_ROUTER_AFFINITY_CORES", "")
        cores = [int(x) for x in cores_raw.split(",") if x.strip().isdigit()]
        # Dual naming: THRESHOLD and RERANK_TRIGGER both control expand gate.
        threshold = _f("NSA_ROUTER_THRESHOLD", 0.42)
        if os.getenv("NSA_ROUTER_RERANK_TRIGGER") is not None:
            threshold = _f("NSA_ROUTER_RERANK_TRIGGER", threshold)
        return cls(
            top_k=_i("NSA_ROUTER_TOP_K", 3),
            candidate_multiplier=_i("NSA_ROUTER_CANDIDATE_MULT", 5),
            threshold=threshold,
            high_conf_gate=_f("NSA_ROUTER_HIGH_CONF_GATE", 0.70),
            high_conf_thinking_budget=_i("NSA_ROUTER_HIGH_CONF_THINKING_BUDGET", 256),
            encoder_name=os.getenv("NSA_ROUTER_ENCODER", "nomic-embed-text-v1.5"),
            embedding_backend=os.getenv("NSA_ROUTER_EMBEDDING_BACKEND", "fastembed").lower(),
            fastembed_cache_dir=os.getenv("NSA_ROUTER_FASTEMBED_CACHE")
            or os.getenv("FASTEMBED_CACHE_PATH"),
            fallback_dims=_i("NSA_ROUTER_FALLBACK_DIMS", 64),
            matryoshka_dim=_i("NSA_ROUTER_MATRYOSHKA_DIM", 256),
            ann_backend=os.getenv("NSA_ROUTER_ANN_BACKEND", "turbovec").lower(),
            turbovec_min_tools=_i("NSA_ROUTER_TURBOVEC_MIN_TOOLS", 0),
            metric=os.getenv("NSA_ROUTER_METRIC", "cosine").lower(),
            cache_backend=os.getenv("NSA_ROUTER_CACHE", "memory").lower(),
            redis_url=os.getenv("NSA_ROUTER_REDIS_URL", "redis://localhost:6379/1"),
            index_path=Path(os.getenv("NSA_ROUTER_INDEX_PATH", str(base / "work/router/index"))),
            snapshot_dir=Path(os.getenv("NSA_ROUTER_SNAPSHOT_DIR", str(base / "work/router/snapshots"))),
            cache_dir=Path(os.getenv("NSA_ROUTER_CACHE_DIR", str(base / "work/router/cache"))),
            tool_metadata_root=Path(
                os.getenv("NSA_TOOL_METADATA_ROOT", str(base / "templates/mcp-servers"))
            ),
            okf_root=Path(os.getenv("NSA_OKF_ROOT", str(base / "okf"))),
            mem_store=Path(os.getenv("NSA_MEM_STORE", str(base / "work/memory"))),
            use_onnx=_b("NSA_ROUTER_ONNX", False),
            use_int8=_b("NSA_ROUTER_INT8", False),
            onnx_path=os.getenv("NSA_ROUTER_ONNX_PATH"),
            tokenizer_path=os.getenv("NSA_ROUTER_TOKENIZER_PATH"),
            allow_hash=_b("NSA_ROUTER_ALLOW_HASH", False),
            embed_batch_size=_i("NSA_ROUTER_EMBED_BATCH", 32),
            embed_workers=_i("NSA_ROUTER_EMBED_WORKERS", 2),
            cache_ttl_s=_f("NSA_ROUTER_CACHE_TTL", 3600.0),
            cache_max_entries=_i("NSA_ROUTER_CACHE_MAX", 10_000),
            turbovec_bit_width=_i("NSA_ROUTER_TURBOVEC_BITS", 4),
            enable_hot_reload=_b("NSA_ROUTER_HOT_RELOAD", True),
            hot_reload_interval_s=_f("NSA_ROUTER_HOT_RELOAD_INTERVAL", 5.0),
            affinity_cores=cores,
            hybrid=hybrid,
            otel_enabled=_b("NSA_ROUTER_OTEL", False),
            otel_endpoint=os.getenv("NSA_ROUTER_OTEL_ENDPOINT", ""),
        )

    def ensure_dirs(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mem_store.mkdir(parents=True, exist_ok=True)


def load_router_config(root: Path | None = None) -> RouterConfig:
    cfg = RouterConfig.from_env(root=root)
    cfg.ensure_dirs()
    return cfg
