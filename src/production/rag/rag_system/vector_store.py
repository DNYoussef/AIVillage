import logging
import os
import time
from typing import Any

import numpy as np
from qdrant_client import QdrantClient

from .faiss_backend import FaissAdapter

logger = logging.getLogger(__name__)

USE_QDRANT = os.getenv("RAG_USE_QDRANT", "0") == "1"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ai_village_vectors")


class VectorStore:
    """Unified interface backed by Qdrant or FAISS with automatic fallback."""

    def __init__(self) -> None:
        self.faiss = FaissAdapter()
        self.backend = self.faiss
        if USE_QDRANT:
            try:
                self.qdrant = QdrantClient(url=QDRANT_URL, timeout=5)
                self.qdrant.get_collections()
                self.backend = self.qdrant
                logger.info("VectorStore: using Qdrant backend")
            except Exception as e:  # pragma: no cover - network side effects
                logger.warning("Qdrant unavailable (%s) – falling back to FAISS", e)

    def add(
        self, ids: list[str], embeddings: np.ndarray, payload: list[dict[str, Any]]
    ) -> None:
        if self.backend is self.faiss:
            self.faiss.add(ids, embeddings, payload)
        else:
            try:
                self.backend.upload_collection(
                    collection_name=COLLECTION_NAME,
                    ids=ids,
                    vectors=embeddings,
                    payload=payload,
                    parallel=1,
                )
                self.faiss.add(ids, embeddings, payload)
            except Exception as e:  # pragma: no cover - network side effects
                logger.exception("Qdrant add failed – reverting to FAISS: %s", e)
                self.backend = self.faiss
                self.faiss.add(ids, embeddings, payload)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        started = time.time()
        try:
            if self.backend is self.faiss:
                return self.faiss.search(query_vec, k)
            res = self.backend.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vec.tolist(),
                limit=k,
                with_payload=True,
            )
            return [{"id": p.id, "score": p.score, "meta": p.payload} for p in res]
        except Exception as e:  # pragma: no cover - network side effects
            logger.exception("Qdrant search failed – falling back (%s)", e)
            self.backend = self.faiss
            return self.faiss.search(query_vec, k)
        finally:
            dur = (time.time() - started) * 1000
            logger.debug("vector search %s ms", round(dur, 2))
