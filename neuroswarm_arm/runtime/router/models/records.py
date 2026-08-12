"""Domain records for the Semantic MCP Tool Router."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MetricKind(str, Enum):
    COSINE = "cosine"
    IP = "ip"
    L2 = "l2"


@dataclass(slots=True)
class EmbeddingSpec:
    model_name: str = "nomic-embed-text-v1.5"
    dims: int = 384
    normalize: bool = True
    matryoshka_dim: int = 768
    matryoshka_dim: int = 768
    # fastembed | sentence-transformers | onnx | hash | auto
    backend: str = "fastembed"
    use_onnx: bool = False
    use_int8: bool = False
    onnx_path: str | None = None
    tokenizer_path: str | None = None
    fastembed_cache_dir: str | None = None


@dataclass(slots=True)
class ToolRecord:
    id: str
    name: str
    description: str = ""
    namespace: str = "default"
    version: str = "1.0.0"
    category: str = "general"
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    rate_limits: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    p50_latency_ms: float = 50.0
    tags: list[str] = field(default_factory=list)
    params: dict[str, str] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    example_prompts: list[str] = field(default_factory=list)
    example_args: dict[str, Any] = field(default_factory=dict)
    endpoint: str | None = None
    auth: str | None = None
    okf_path: str | None = None
    checksum: str = ""
    popularity: float = 0.0
    success_rate: float = 1.0
    failure_rate: float = 0.0
    recent_usage: float = 0.0
    reliability: float = 1.0
    agent_roles: list[str] = field(default_factory=list)
    workflow_stages: list[str] = field(default_factory=list)
    # True only after live MCP tools/list reconciliation (not YAML-only).
    executable: bool = False
    destructive_hint: bool = False
    readonly_hint: bool = False

    def index_text(self) -> str:
        parts = [
            self.name,
            self.description,
            " ".join(self.params.keys()),
            " ".join(self.capabilities),
            " ".join(self.tags),
            " ".join(self.example_prompts),
            self.category,
            self.namespace,
        ]
        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass(slots=True)
class RouteContext:
    agent_id: str = "default"
    agent_role: str = "tool_call"
    workflow_stage: str = "execute"
    conversation_excerpt: str = ""
    previous_tools: list[str] = field(default_factory=list)
    task_type: str = "general"
    budget_remaining_usd: float = 1.0
    budget_envelope_id: str = ""
    budget_remaining: dict[str, float] = field(default_factory=dict)
    latency_slo_ms: float = 4000.0
    security_policies: list[str] = field(default_factory=list)
    memory_pressure: float = 0.0
    quantization: str = ""
    inference_tier: int = 1
    mem0_hits: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    required_permissions: list[str] = field(default_factory=list)
    expected_output_format: str = ""
    tool_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScoredTool:
    tool: ToolRecord
    score: float
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    confidence: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    schema: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.tool.id

    @property
    def name(self) -> str:
        return self.tool.name


@dataclass(slots=True)
class RoutingResult:
    tools: list[ScoredTool] = field(default_factory=list)
    top_k: int = 3
    confidence_top1: float = 0.0
    high_confidence: bool = False
    prompt_tokens_before: int = 0
    prompt_tokens_after: int = 0
    latency_breakdown_ms: dict[str, float] = field(default_factory=dict)
    features_debug: dict[str, Any] = field(default_factory=dict)
    query: str = ""
    candidate_count: int = 0

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.tools]

    @property
    def tool_ids(self) -> list[str]:
        return [t.id for t in self.tools]

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self.tools if t.schema]

    def token_reduction_ratio(self) -> float:
        if self.prompt_tokens_before <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.prompt_tokens_after / self.prompt_tokens_before))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "id": t.id,
                    "name": t.name,
                    "score": t.score,
                    "confidence": t.confidence,
                    "semantic_score": t.semantic_score,
                    "hybrid_score": t.hybrid_score,
                    "rerank_score": t.rerank_score,
                    "features": t.features,
                    "schema": t.schema,
                }
                for t in self.tools
            ],
            "top_k": self.top_k,
            "confidence_top1": self.confidence_top1,
            "high_confidence": self.high_confidence,
            "prompt_tokens_before": self.prompt_tokens_before,
            "prompt_tokens_after": self.prompt_tokens_after,
            "token_reduction_ratio": self.token_reduction_ratio(),
            "latency_breakdown_ms": self.latency_breakdown_ms,
            "features_debug": self.features_debug,
            "query": self.query,
            "candidate_count": self.candidate_count,
            "tool_names": self.tool_names,
            "tool_ids": self.tool_ids,
        }
