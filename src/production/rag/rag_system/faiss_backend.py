"""FAISS-based adapter used by VectorStore as a local fallback."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import faiss
import numpy as np

from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore as PickledStore

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.json")
DEFAULT_DIM = 768


class FaissAdapter:
    """Thin wrapper around a FAISS index with optional disk loading."""

    def __init__(self, path: str | None = DEFAULT_STORE_PATH, dimension: int = DEFAULT_DIM) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata: list[dict[str, Any]] = []
        if path and os.path.exists(path):
            store = PickledStore.load(path, UnifiedConfig())
            vecs = [d["embedding"] for d in store.documents]
            self.index.add(np.array(vecs, dtype="float32"))
            self.metadata = [{k: v for k, v in doc.items() if k != "embedding"} for doc in store.documents]

    # ----------------------------- iteration -----------------------------
    def iter_embeddings(self, batch_size: int = 100) -> Iterable[tuple[list[Any], list[Any], list[dict[str, Any]]]]:
        for i in range(0, len(self.metadata), batch_size):
            batch_meta = self.metadata[i : i + batch_size]
            ids = [d["id"] for d in batch_meta]
            vecs = [self.index.reconstruct(j) for j in range(i, i + len(batch_meta))]
            payload = [{k: v for k, v in d.items() if k != "id"} for d in batch_meta]
            yield ids, vecs, payload

    # ----------------------------- add/search ----------------------------
    def add(self, ids: list[str], embeddings: np.ndarray, payload: list[dict[str, Any]]) -> None:
        vecs = np.asarray(embeddings, dtype="float32")
        self.index.add(vecs)
        for pid, meta in zip(ids, payload, strict=False):
            m = dict(meta)
            m["id"] = pid
            self.metadata.append(m)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        q = np.asarray([query_vec], dtype="float32")
        D, indices = self.index.search(q, k)
        results: list[dict[str, Any]] = []
        for dist, idx in zip(D[0], indices[0], strict=False):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({"id": meta["id"], "score": float(-dist), "meta": meta})
        return results
