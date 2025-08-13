import base64
from datetime import datetime
import json
import os
from typing import Any
from urllib.parse import urlparse
import uuid

import faiss
import numpy as np

USE_QDRANT = os.getenv("RAG_USE_QDRANT") == "1"
if USE_QDRANT:
    try:
        from qdrant_client import QdrantClient
    except Exception:  # pragma: no cover - optional
        QdrantClient = None

from ..core.config import UnifiedConfig
from ..core.structures import RetrievalResult


def _get_qdrant_url() -> str:
    """Return validated Qdrant URL based on environment."""
    # Use HTTPS by default in production, HTTP only in development
    default_url = (
        "https://qdrant.aivillage.internal:6333"
        if os.getenv("AIVILLAGE_ENV") == "production"
        else "https://localhost:6333"
    )
    url = os.getenv("QDRANT_URL", default_url)

    if os.getenv("AIVILLAGE_ENV") == "production" and urlparse(url).scheme == "http":
        msg = "QDRANT_URL must use https:// in production environment"
        raise ValueError(msg)
    return url


DEFAULT_DIMENSION = 768


class VectorStore:
    def __init__(
        self,
        config: UnifiedConfig | None = None,
        dimension: int = DEFAULT_DIMENSION,
        embedding_model: Any | None = None,
    ) -> None:
        """Create a VectorStore.

        The previous version of :class:`VectorStore` required ``config`` and
        ``dimension`` arguments.  Many parts of the codebase still instantiate
        this class without any parameters which resulted in ``TypeError`` being
        raised at runtime.  To maintain backwards compatibility we allow both
        arguments to be optional and provide sensible defaults.
        """
        self.config = config or UnifiedConfig()
        self.embedding_model = embedding_model
        self.dimension = (
            getattr(embedding_model, "hidden_size", dimension)
            if embedding_model is not None
            else dimension
        )
        self.documents: list[dict[str, Any]] = []
        if USE_QDRANT and QdrantClient is not None:
            self.qdrant = QdrantClient(url=_get_qdrant_url())
            self.collection = "documents"
            try:  # pragma: no cover - network side effects
                self.qdrant.get_collection(self.collection)
            except Exception:
                self.qdrant.recreate_collection(
                    self.collection, vector_size=self.dimension, distance="Cosine"
                )
            self.index = None
        else:
            self.index = faiss.IndexFlatL2(self.dimension)

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        vectors = [doc["embedding"] for doc in documents]
        if USE_QDRANT and QdrantClient is not None:
            payload = [
                {
                    "id": doc["id"],
                    "vector": vec,
                    "payload": {"content": doc["content"]},
                }
                for doc, vec in zip(documents, vectors, strict=False)
            ]
            try:  # pragma: no cover - network side effects
                self.qdrant.upsert(collection_name=self.collection, points=payload)
            except Exception as e:
                logger.warning(f"Failed to upsert to Qdrant: {e}. Falling back to local index.")
        else:
            self.index.add(np.array(vectors).astype("float32"))
        self.documents.extend(documents)

    async def add_texts(self, texts: list[str]) -> None:
        """Convenience helper used by learning layers to store raw text."""
        docs = []
        for text in texts:
            if self.embedding_model is not None:
                _, emb = self.embedding_model.encode(text)
                try:
                    vec = emb.mean(dim=0).detach().cpu().numpy().astype("float32")
                except Exception:
                    vec = np.asarray(emb, dtype="float32")
            else:
                vec = np.random.rand(self.dimension).astype("float32")

            docs.append(
                {
                    "id": str(uuid.uuid4()),
                    "content": text,
                    "embedding": vec,
                    "timestamp": datetime.now(),
                }
            )
        self.add_documents(docs)

    def update_document(self, doc_id: str, new_doc: dict[str, Any]) -> None:
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                np.array([doc["embedding"]]).astype("float32")
                new_vector = np.array([new_doc["embedding"]]).astype("float32")
                self.index.remove_ids(np.array([i]))
                self.index.add(new_vector)
                self.documents[i] = new_doc
                break

    def delete_document(self, doc_id: str) -> None:
        for i, doc in enumerate(self.documents):
            if doc["id"] == doc_id:
                self.index.remove_ids(np.array([i]))
                del self.documents[i]
                break

    async def retrieve(
        self,
        query_vector: list[float],
        k: int,
        timestamp: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        if USE_QDRANT and QdrantClient is not None:
            try:  # pragma: no cover - network side effects
                resp = self.qdrant.search(
                    collection_name=self.collection, query_vector=query_vector, limit=k
                )
                entries = [(p.payload.get("content", ""), p.score, p.id) for p in resp]
            except Exception:
                entries = []
            distances = [1.0 - s for _, s, _ in entries]
            indices = list(range(len(entries)))
            for content, _s, pid in entries:
                self.documents.append(
                    {
                        "id": pid,
                        "content": content,
                        "embedding": query_vector,
                        "timestamp": datetime.now(),
                    }
                )
        else:
            query_vector_np = np.array([query_vector]).astype("float32")
            distances, indices = self.index.search(query_vector_np, k)

        results = []
        for i, idx in enumerate(
            indices[0] if isinstance(indices, np.ndarray) else indices
        ):
            doc = self.documents[idx]
            if (timestamp is None or doc["timestamp"] <= timestamp) and (
                metadata_filter is None
                or all(doc.get(key) == value for key, value in metadata_filter.items())
            ):
                result = RetrievalResult(
                    id=doc["id"],
                    content=doc["content"],
                    score=1
                    / (1 + distances[0][i]),  # Convert distance to similarity score
                    timestamp=doc["timestamp"],
                )
                results.append(result)

        return results

    def get_document_by_id(self, doc_id: str) -> dict[str, Any] | None:
        for doc in self.documents:
            if doc["id"] == doc_id:
                return doc
        return None

    def get_size(self) -> int:
        return len(self.documents)

    async def get_count(self) -> int:
        """Return the number of stored vector documents."""
        return len(self.documents)

    def save(self, file_path: str) -> None:
        index_bytes = (
            faiss.serialize_index(self.index) if self.index is not None else b""
        )
        data = {
            "index": base64.b64encode(index_bytes).decode("utf-8"),
            "documents": self.documents,
            "dimension": self.dimension,
        }
        with open(file_path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, file_path: str, config: UnifiedConfig) -> "VectorStore":
        with open(file_path) as f:
            data = json.load(f)

        vector_store = cls(config, data["dimension"])
        index_data = data.get("index", "")
        if index_data:
            idx_bytes = base64.b64decode(index_data)
            vector_store.index = faiss.deserialize_index(idx_bytes)
        else:
            vector_store.index = faiss.IndexFlatL2(vector_store.dimension)
        vector_store.documents = data.get("documents", [])

        return vector_store
