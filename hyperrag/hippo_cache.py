"""High-performance RAG cache with semantic similarity search."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


@dataclass
class CacheEntry:
    query_embedding: np.ndarray
    retrieved_docs: list[Any]
    relevance_scores: list[float]
    citation_metadata: dict[str, Any]
    timestamp: datetime
    access_count: int = 0


class HippoCache:
    """Thread-safe LRU cache with cosine-similarity search."""

    def __init__(
        self,
        max_size: int = 10_000,
        ttl_hours: int = 24,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self.similarity_threshold = similarity_threshold

        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._embeddings: list[np.ndarray] = []
        self._keys: list[str] = []
        self._matrix: np.ndarray | None = None

        # Metrics
        self._hits = 0
        self._misses = 0
        self._latency_total_ms = 0.0

    # ------------------------------------------------------------------
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _update_matrix(self) -> None:
        if self._embeddings:
            self._matrix = np.vstack(self._embeddings)
        else:
            self._matrix = None

    def _is_expired(self, entry: CacheEntry) -> bool:
        return datetime.utcnow() - entry.timestamp > self.ttl

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.max_size:
            old_key, _ = self._cache.popitem(last=False)
            if old_key in self._keys:
                idx = self._keys.index(old_key)
                self._keys.pop(idx)
                self._embeddings.pop(idx)
                self._update_matrix()

    # ------------------------------------------------------------------
    def get(self, query_embedding: np.ndarray) -> CacheEntry | None:
        start = datetime.utcnow()
        with self._lock:
            if self._matrix is None:
                self._misses += 1
                return None
            query = self._normalize(query_embedding)
            sims = self._matrix @ query
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            if score < self.similarity_threshold:
                self._misses += 1
                return None
            key = self._keys[idx]
            entry = self._cache.get(key)
            if entry is None or self._is_expired(entry):
                self._cache.pop(key, None)
                self._hits += 0
                self._misses += 1
                return None
            entry.access_count += 1
            self._cache.move_to_end(key)
            self._hits += 1
        self._latency_total_ms += (datetime.utcnow() - start).total_seconds() * 1000
        return entry

    def set(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            normalized = self._normalize(entry.query_embedding)
            if key in self._cache:
                # update existing embedding
                idx = self._keys.index(key)
                self._embeddings[idx] = normalized
            else:
                self._keys.append(key)
                self._embeddings.append(normalized)
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._update_matrix()
            self._evict_if_needed()

    async def warm_cache(self, items: Iterable[tuple[str, CacheEntry]]) -> None:
        for key, entry in items:
            self.set(key, entry)
            await asyncio.sleep(0)

    def get_or_retrieve(
        self,
        key: str,
        query_embedding: np.ndarray,
        retrieval_fn: Callable[[], tuple[list[Any], list[float], dict[str, Any]]],
    ) -> CacheEntry:
        start = datetime.utcnow()
        entry = self.get(query_embedding)
        if entry is not None:
            return entry
        docs, scores, meta = retrieval_fn()
        entry = CacheEntry(
            query_embedding=query_embedding,
            retrieved_docs=docs,
            relevance_scores=scores,
            citation_metadata=meta,
            timestamp=datetime.utcnow(),
            access_count=1,
        )
        self.set(key, entry)
        self._latency_total_ms += (datetime.utcnow() - start).total_seconds() * 1000
        return entry

    def metrics(self) -> dict[str, float]:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        avg_latency = self._latency_total_ms / total if total else 0.0
        return {
            "hit_rate": hit_rate,
            "avg_latency_ms": avg_latency,
            "size": len(self._cache),
        }
