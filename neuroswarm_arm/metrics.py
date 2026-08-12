from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class MetricsStore:
    lock: Lock = field(default_factory=Lock)
    counters: dict[str, float] = field(default_factory=dict)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self.lock:
            self.counters[name] = self.counters.get(name, 0.0) + value

    def set(self, name: str, value: float) -> None:
        with self.lock:
            self.counters[name] = value

    def export_prometheus(self) -> str:
        with self.lock:
            lines = []
            # Metric metadata for embedding metrics
            meta = {
                "embedding_backend_active": ("gauge", "Active embedding backend (1=active)"),
                "embedding_matryoshka_dim": ("gauge", "Current Matryoshka truncation dimension"),
                "embedding_model_dims": ("gauge", "Native model embedding dimensions"),
                "embedding_backend_st": ("gauge", "SentenceTransformers backend active"),
                "embedding_backend_fastembed": ("gauge", "FastEmbed backend active"),
                "embedding_backend_hash": ("gauge", "Hash fallback backend active"),
                "embedding_last_dim": ("gauge", "Last encoded embedding dimension"),
                "embedding_last_backend": ("gauge", "Last backend used (1=ST, 2=FastEmbed, 0=Hash)"),
            }
            for key, value in sorted(self.counters.items()):
                mtype, mhelp = meta.get(key, ("gauge", f"Auto-registered gauge {key}"))
                lines.append(f"# HELP {key} {mhelp}")
                lines.append(f"# TYPE {key} {mtype}")
                lines.append(f"{key} {value}")
            return "\n".join(lines) + "\n"


metrics = MetricsStore()

