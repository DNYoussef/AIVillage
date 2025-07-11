"""Helper utilities for working with ``VectorStore`` files."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Tuple

from rag_system.retrieval.vector_store import VectorStore
from rag_system.core.config import UnifiedConfig


DEFAULT_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store.pkl")


class FaissAdapter:
    """Thin wrapper exposing batches of embeddings from a pickled store."""

    def __init__(self, path: str = DEFAULT_STORE_PATH) -> None:
        self.path = path
        self.store = VectorStore.load(path, UnifiedConfig())

    def iter_embeddings(
        self, batch_size: int = 100
    ) -> Iterable[Tuple[List[Any], List[Any], List[Dict[str, Any]]]]:
        """Yield ``(ids, vectors, payload)`` tuples from the store."""

        docs = list(self.store.documents)
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            ids = [d["id"] for d in batch]
            vectors = [d["embedding"] for d in batch]
            payload = [
                {k: v for k, v in d.items() if k not in {"embedding"}}
                for d in batch
            ]
            yield ids, vectors, payload

