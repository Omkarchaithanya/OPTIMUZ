"""DI factory for Semantic MCP Tool Router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backends import build_vector_index
from .embedding_cache import EmbeddingCache
from .embedding_service import EmbeddingService
from .history_ranker import HistoryRanker
from .hybrid_search import HybridSearch
from .incremental_index import IncrementalIndexer
from .index_snapshot import IndexSnapshotManager
from .models import EmbeddingSpec, MetricKind
from .registry import ToolRegistry
from .registry_loader import RegistryLoader
from .reranker import Reranker
from .router_config import RouterConfig, load_router_config
from .router_events import RouterEventBus
from .router_metrics import RouterMetrics
from .telemetry import RouterTelemetry
from .tool_registry_sync import ToolRegistrySync
from .tool_router import SemanticToolRouter


def build_router(
    cfg: RouterConfig | None = None,
    *,
    root: Path | None = None,
    metrics_bridge: Any | None = None,
    memory: Any | None = None,
    start_sync: bool = True,
) -> SemanticToolRouter:
    config = cfg or load_router_config(root)
    config.ensure_dirs()
    events = RouterEventBus()
    metrics = RouterMetrics(bridge=metrics_bridge)
    telemetry = RouterTelemetry(enabled=config.otel_enabled, endpoint=config.otel_endpoint)

    cache = EmbeddingCache(
        backend=config.cache_backend,
        redis_url=config.redis_url,
        disk_dir=config.cache_dir,
        max_entries=config.cache_max_entries,
        ttl_s=config.cache_ttl_s,
        metrics=metrics,
    )
    embedder = EmbeddingService(
        EmbeddingSpec(
            model_name=config.encoder_name,
            dims=config.matryoshka_dim,
            normalize=True,
            backend=config.embedding_backend,
            use_onnx=config.use_onnx,
            use_int8=config.use_int8,
            onnx_path=config.onnx_path,
            tokenizer_path=config.tokenizer_path,
            fastembed_cache_dir=config.fastembed_cache_dir,
            matryoshka_dim=config.matryoshka_dim,
        ),
        cache=cache,
        metrics=metrics,
        workers=config.embed_workers,
        fallback_dims=config.fallback_dims,
        allow_hash=config.allow_hash,
    )
    metric = MetricKind(config.metric) if config.metric in {m.value for m in MetricKind} else MetricKind.COSINE
    index = build_vector_index(
        config.ann_backend,
        embedder.dims,
        metric=metric,
        bit_width=config.turbovec_bit_width,
        events=events,
        turbovec_min_tools=config.turbovec_min_tools,
    )
    if int(embedder.dims) != int(getattr(index, "dims", embedder.dims)):
        raise ValueError(
            f"embedder.dims ({embedder.dims}) != index.dims ({getattr(index, 'dims', None)}); "
            "refusing to wire mismatched ANN index"
        )
    registry = ToolRegistry(events=events)
    loader = RegistryLoader()
    loaded: list = []
    for path in [config.tool_metadata_root, config.okf_root]:
        if path.exists():
            loaded.extend(loader.load_path(path))

    # Optional: index only tools whose MCP server answers tools/list.
    from .live_mcp_index import filter_tools_by_live_mcp, live_index_enabled

    if live_index_enabled() and loaded:
        kept = filter_tools_by_live_mcp(loaded)
        registry.bulk_register(kept)
        metrics.set("router_live_index", 1.0)
        metrics.set("router_live_index_kept", float(len(kept)))
        metrics.set("router_live_index_skipped", float(max(0, len(loaded) - len(kept))))
    else:
        registry.bulk_register(loaded)
        metrics.set("router_live_index", 0.0)

    indexer = IncrementalIndexer(index, embedder, events=events)
    indexer.rebuild(registry.as_list())
    metrics.set("router_tools_registered", float(registry.size()))
    metrics.set("router_index_size", float(index.size()))

    history = HistoryRanker(memory=memory, root=config.mem_store)
    hybrid = HybridSearch(config.hybrid)
    reranker = Reranker(config.rerank)
    snapshots = IndexSnapshotManager(registry, index, snapshot_dir=config.snapshot_dir, events=events)
    sync = ToolRegistrySync(
        registry,
        indexer,
        roots=[config.tool_metadata_root, config.okf_root],
        events=events,
        interval_s=config.hot_reload_interval_s,
        enabled=config.enable_hot_reload,
    )
    router = SemanticToolRouter(
        config=config,
        registry=registry,
        embedder=embedder,
        index=index,
        metrics=metrics,
        events=events,
        history=history,
        hybrid=hybrid,
        reranker=reranker,
        indexer=indexer,
        snapshots=snapshots,
        sync=sync,
        telemetry=telemetry,
    )
    if start_sync and config.enable_hot_reload:
        sync.start()
    return router
