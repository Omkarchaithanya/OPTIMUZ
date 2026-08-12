"""Embedding service: FastEmbed BGE-small, Sentence-Transformers, ONNX, hash (opt-in)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from typing import Any, Iterator

import numpy as np

from .embedding_cache import EmbeddingCache
from .models import EmbeddingSpec
from .router_exceptions import EmbeddingError
from .router_metrics import RouterMetrics
from .similarity import l2_normalize, validate_embedding
def _truncate_matryoshka(embedding: np.ndarray, target_dim: int) -> np.ndarray:
    """Nomic v1.5 Matryoshka truncation: layer-norm -> slice -> renormalize."""
    vec = embedding.astype(np.float32)
    mean = np.mean(vec)
    var = np.var(vec)
    vec = (vec - mean) / np.sqrt(var + 1e-5)
    vec = vec[:target_dim]
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec



KNOWN_MODELS = {
    "nomic-embed-text-v1.5": 768,
    "bge-small-en-v1.5": 384,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "all-MiniLM-L6-v2": 384,
    "intfloat/e5-small-v2": 384,
    "e5-small-v2": 384,
}

# BGE retrieval query instruction (FlagEmbedding / BAAI card). Documents stay unprefixed.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_HASH_REMEDIATION = (
    "No real embedding backend available (FastEmbed / Sentence-Transformers / ONNX). "
    "Install fastembed (preferred on ARM gateway) or sentence-transformers, "
    "or set NSA_ROUTER_ONNX=1 with NSA_ROUTER_ONNX_PATH + tokenizer. "
    "For local tests only, set NSA_ROUTER_ALLOW_HASH=1."
)


def _hash_embed(text: str, dims: int) -> np.ndarray:
    vec = np.zeros(dims, dtype=np.float32)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    for i, ch in enumerate(text.lower()):
        vec[i % dims] += ((ord(ch) + digest[i % len(digest)]) % 31) / 31.0
    norm = float(np.linalg.norm(vec)) or 1.0
    return vec / norm


def _allow_hash() -> bool:
    raw = os.getenv("NSA_ROUTER_ALLOW_HASH", "")
    return raw in {"1", "true", "True", "yes", "YES"}


def _env_backend() -> str:
    return (os.getenv("NSA_ROUTER_EMBEDDING_BACKEND") or "").strip().lower()


def bge_query_prefix_enabled() -> bool:
    """Kill-switch: NSA_ROUTER_BGE_QUERY_PREFIX=0 disables instruction prefix."""
    raw = os.getenv("NSA_ROUTER_BGE_QUERY_PREFIX", "1")
    return raw not in {"0", "false", "False", "no", "NO"}


def apply_bge_query_prefix(text: str, *, model_name: str) -> str:
    """Prefix query text for BGE-small retrieval; no-op for other models / when disabled."""
    if not text or not bge_query_prefix_enabled():
        return text
    name = (model_name or "").lower()
    if "bge-small" not in name and not name.endswith("bge-small-en-v1.5"):
        return text
    if text.startswith(BGE_QUERY_PREFIX):
        return text
    return BGE_QUERY_PREFIX + text


class _FastEmbedAdapter:
    """Adapt fastembed.TextEmbedding to a SentenceTransformer-like encode API."""

    def __init__(self, model: Any, dims: int = 384) -> None:
        self._model = model
        self._dims = int(dims)

    def encode(self, text_or_list: Any, normalize_embeddings: bool = False) -> np.ndarray:
        if isinstance(text_or_list, str):
            vecs = list(self._model.embed([text_or_list]))
            arr = np.asarray(vecs[0], dtype=np.float32).reshape(-1)
        else:
            texts = [str(t) for t in text_or_list]
            vecs = list(self._model.embed(texts))
            arr = np.asarray(vecs, dtype=np.float32)
        if normalize_embeddings:
            if arr.ndim == 1:
                n = float(np.linalg.norm(arr)) or 1.0
                arr = arr / n
            else:
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                arr = arr / norms
        return arr

    def get_sentence_embedding_dimension(self) -> int:
        return self._dims


class EmbeddingService:
    def __init__(
        self,
        spec: EmbeddingSpec | None = None,
        *,
        cache: EmbeddingCache | None = None,
        metrics: RouterMetrics | None = None,
        workers: int = 2,
        fallback_dims: int = 64,
        allow_hash: bool | None = None,
    ) -> None:
        self.spec = spec or EmbeddingSpec()
        self.cache = cache
        self.metrics = metrics
        self.fallback_dims = fallback_dims
        self._allow_hash = _allow_hash() if allow_hash is None else bool(allow_hash)
        self._model: Any = None
        self._onnx: Any = None
        self._tokenizer: Any = None
        self._dims: int | None = KNOWN_MODELS.get(self.spec.model_name)
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="emb")
        self._backend = "hash"
        self._ensure_backend()

    @property
    def dims(self) -> int:
        target = getattr(self.spec, "matryoshka_dim", None)
        if target:
            return int(target)
        return int(self._dims or self.spec.dims or self.fallback_dims)

    @property
    def backend_name(self) -> str:
        return self._backend

    def _requested_backend(self) -> str:
        env = _env_backend()
        if env:
            return env
        if self.spec.use_onnx:
            return "onnx"
        return (self.spec.backend or "fastembed").strip().lower() or "fastembed"

    def _ensure_backend(self) -> None:
        requested = self._requested_backend()

        if requested == "hash":
            self._backend = "hash"
            self._dims = self._dims or self.fallback_dims
            if not self._allow_hash:
                raise EmbeddingError(_HASH_REMEDIATION)
            return

        if requested in {"onnx", "onnx-int8"} or self.spec.use_onnx:
            if self._try_onnx():
                return
            raise EmbeddingError(
                "ONNX embedding backend requested but session could not be loaded. "
                "Set NSA_ROUTER_ONNX_PATH to a valid model and provide a tokenizer "
                f"(NSA_ROUTER_TOKENIZER_PATH or encoder '{self.spec.model_name}')."
            )

        if requested in {"fastembed", "auto", "default", ""}:
            if self._try_fastembed():
                return
            if requested == "fastembed":
                # Fall through to ST then fail/hash — still prefer fail-loud for prod.
                pass

        if requested in {"sentence-transformers", "st", "auto", "default", "fastembed", ""}:
            if self._try_sentence_transformers():
                return

        if requested in {"sentence-transformers", "st"} and not self._allow_hash:
            raise EmbeddingError(
                "NSA_ROUTER_EMBEDDING_BACKEND=sentence-transformers but "
                "sentence-transformers failed to load. " + _HASH_REMEDIATION
            )

        if requested == "fastembed" and not self._allow_hash:
            raise EmbeddingError(
                "NSA_ROUTER_EMBEDDING_BACKEND=fastembed but fastembed failed to load. "
                "Install fastembed+onnxruntime into the gateway image. " + _HASH_REMEDIATION
            )

        self._backend = "hash"
        self._dims = self._dims or self.fallback_dims
        if not self._allow_hash:
            raise EmbeddingError(_HASH_REMEDIATION)

    def _try_fastembed(self) -> bool:
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception:
            return False
        try:
            cache_dir = (
                self.spec.fastembed_cache_dir
                or os.getenv("NSA_ROUTER_FASTEMBED_CACHE")
                or os.getenv("FASTEMBED_CACHE_PATH")
            )
            kwargs: dict[str, Any] = {"model_name": self.spec.model_name}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            raw = TextEmbedding(**kwargs)
            dims = int(KNOWN_MODELS.get(self.spec.model_name, self.spec.dims or 384))
            self._model = _FastEmbedAdapter(raw, dims=dims)
            self._dims = dims
            self._backend = "fastembed"
            return True
        except Exception:
            self._model = None
            return False

    def _try_sentence_transformers(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            print(f"[ST DEBUG] import failed: {exc}")
            return False
        try:
            target_dim = getattr(self.spec, "matryoshka_dim", None) or self.dims or 768
            model_map = {
                "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
                "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
            }
            model_name = model_map.get(self.spec.model_name, self.spec.model_name)
            print(f"[ST DEBUG] Loading: {model_name} truncate_dim={target_dim}")
            self._model = SentenceTransformer(
                model_name, device="cpu", trust_remote_code=True, truncate_dim=target_dim
            )
            self._dims = int(self._model.get_sentence_embedding_dimension())
            self._backend = "sentence-transformers"
            print(f"[ST DEBUG] Success! dims={self._dims}")
            return True
        except Exception as exc:
            print(f"[ST DEBUG] Load failed: {exc}")
            import traceback
            traceback.print_exc()
            self._model = None
            return False

    def _load_tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:
            raise EmbeddingError(
                "ONNX embedding requires transformers.AutoTokenizer. "
                "Install transformers or unset NSA_ROUTER_ONNX."
            ) from exc
        tok_id = (
            self.spec.tokenizer_path
            or os.getenv("NSA_ROUTER_TOKENIZER_PATH")
            or self.spec.model_name
        )
        try:
            return AutoTokenizer.from_pretrained(tok_id)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load ONNX tokenizer from '{tok_id}'. "
                "Set NSA_ROUTER_TOKENIZER_PATH to a local HF tokenizer directory "
                "or HF model id. Hash-encode under the ONNX label is not allowed."
            ) from exc

    def _try_onnx(self) -> bool:
        path = self.spec.onnx_path or os.getenv("NSA_ROUTER_ONNX_PATH")
        if not path:
            return False
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise EmbeddingError(
                "NSA_ROUTER_ONNX=1 requires onnxruntime. Install onnxruntime."
            ) from exc
        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            providers = ["CPUExecutionProvider"]
            self._onnx = ort.InferenceSession(path, sess_options=sess_options, providers=providers)
            out = self._onnx.get_outputs()[0]
            shape = out.shape
            self._dims = int(shape[-1]) if shape and isinstance(shape[-1], int) else self.spec.dims
            self._tokenizer = self._load_tokenizer()
            self._backend = "onnx-int8" if self.spec.use_int8 else "onnx"
            return True
        except EmbeddingError:
            self._onnx = None
            self._tokenizer = None
            raise
        except Exception as exc:
            self._onnx = None
            self._tokenizer = None
            raise EmbeddingError(f"Failed to load ONNX embedding model at '{path}': {exc}") from exc

    def encode(self, text: str, *, normalize: bool | None = None) -> np.ndarray:
        timer = self.metrics.timer() if self.metrics else None
        model_name = self.spec.model_name
        if self.cache is not None:
            cached = self.cache.get(model_name, text)
            if cached is not None:
                if self.metrics and timer:
                    self.metrics.set("router_embedding_latency_ms", timer.ms())
                return validate_embedding(cached, self.dims)
        vec = self._encode_uncached(text)
        do_norm = self.spec.normalize if normalize is None else normalize
        if do_norm:
            vec = l2_normalize(vec)
        vec = validate_embedding(vec, self.dims)
        if self.cache is not None:
            self.cache.set(model_name, text, vec)
        if self.metrics and timer:
            self.metrics.set("router_embedding_latency_ms", timer.ms())
        return vec

    def encode_query(self, text: str, *, normalize: bool | None = None) -> np.ndarray:
        """Encode a retrieval query (BGE instruction prefix when applicable).

        Tool/document embeddings must keep using ``encode`` (unprefixed).
        Hash backend skips the prefix so deterministic toy embeddings stay aligned.
        """
        if self._backend == "hash":
            return self.encode(text, normalize=normalize)
        prefixed = apply_bge_query_prefix(text, model_name=self.spec.model_name)
        return self.encode(prefixed, normalize=normalize)

    def _encode_uncached(self, text: str) -> np.ndarray:
        if self._onnx is not None:
            emb = self._encode_onnx(text)
            target_dim = getattr(self.spec, "matryoshka_dim", None)
            if target_dim and target_dim < emb.shape[-1]:
                emb = _truncate_matryoshka(emb, target_dim)
            return emb
        if self._model is not None:
            arr = self._model.encode(text, normalize_embeddings=False)
            emb = np.asarray(arr, dtype=np.float32).reshape(-1)
            target_dim = getattr(self.spec, "matryoshka_dim", None)
            if target_dim and target_dim < emb.shape[-1]:
                emb = _truncate_matryoshka(emb, target_dim)
            return emb
        if not self._allow_hash:
            raise EmbeddingError(_HASH_REMEDIATION)
        return _hash_embed(text, self.dims)

    def _encode_onnx(self, text: str) -> np.ndarray:
        if self._tokenizer is None:
            raise EmbeddingError(
                "ONNX backend has no tokenizer; refusing hash fallback under ONNX label."
            )
        try:
            inputs = self._tokenizer(text, return_tensors="np", padding=True, truncation=True)
            feeds = {k: v for k, v in inputs.items()}
            outs = self._onnx.run(None, feeds)
            return np.asarray(outs[0], dtype=np.float32).reshape(-1)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"onnx encode failed: {exc}") from exc

    def encode_batch(self, texts: list[str], *, normalize: bool | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        if self._model is not None and self.cache is None:
            arr = self._model.encode(texts, normalize_embeddings=bool(normalize or self.spec.normalize))
            return np.asarray(arr, dtype=np.float32)
        rows = [self.encode(t, normalize=normalize) for t in texts]
        return np.stack(rows).astype(np.float32)

    async def encode_async(self, text: str, *, normalize: bool | None = None) -> np.ndarray:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self.encode(text, normalize=normalize))

    def encode_stream(self, texts: Iterator[str], *, normalize: bool | None = None) -> Iterator[np.ndarray]:
        for text in texts:
            yield self.encode(text, normalize=normalize)

    def persist_stats(self) -> dict[str, Any]:
        return {
            "backend": self._backend,
            "model": self.spec.model_name,
            "dims": self.dims,
            "onnx": self.spec.use_onnx,
            "int8": self.spec.use_int8,
            "cache": self.cache.stats() if self.cache else None,
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
