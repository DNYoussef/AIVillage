"""Very small vector retriever."""

from __future__ import annotations

from typing import Any

import numpy as np


class VectorRetriever:
    """Stores vectors in memory and performs cosine similarity search."""

    def __init__(self, **_kwargs: Any) -> None:
        """Initialise empty in-memory store."""
        self._store: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
        self._next_id = 0

    def add_vector(self, vector: np.ndarray, metadata: dict[str, Any]) -> int:
        vid = self._next_id
        self._store[vid] = (vector, metadata)
        self._next_id += 1
        return vid

    def retrieve(
        self, query: np.ndarray, top_k: int = 1
    ) -> list[tuple[float, dict[str, Any]]]:
        if not self._store:
            return []
        query_norm = query / np.linalg.norm(query)
        scored: list[tuple[float, dict[str, Any]]] = []
        for vec, meta in self._store.values():
            vec_norm = vec / np.linalg.norm(vec)
            score = float(np.dot(query_norm, vec_norm))
            scored.append((score, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
