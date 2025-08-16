from __future__ import annotations

from typing import Any

import numpy as np

try:
    from .faiss_backend import FaissAdapter
except Exception:  # pragma: no cover - faiss may be missing
    FaissAdapter = None  # type: ignore

from .core.interface import (
    EmbeddingModel,
    KnowledgeConstructor,
    ReasoningEngine,
    Retriever,
)
from .utils.embedding import BERTEmbeddingModel


class SimpleEmbeddingModel(EmbeddingModel):
    """Minimal embedding model using BERT with fallback."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self._model = BERTEmbeddingModel(model_name)
        self.dimension = getattr(self._model, "hidden_size", 768)

    async def get_embedding(self, text: str) -> list[float]:
        _tokens, emb = self._model.encode(text)
        # Mean pool to a single vector
        return emb.mean(dim=0).detach().cpu().tolist()


class SimpleRetriever(Retriever):
    """In-memory retriever with optional FAISS acceleration."""

    def __init__(self, embedding_model: EmbeddingModel | None = None) -> None:
        self.embedding_model = embedding_model or SimpleEmbeddingModel()
        self._faiss = (
            FaissAdapter(dimension=self.embedding_model.dimension) if FaissAdapter else None
        )
        self._store: list[tuple[np.ndarray, dict[str, Any]]] = []
        self._next_id = 0

    async def add_documents(self, texts: list[str]) -> None:
        embeddings = [
            await self.embedding_model.get_embedding(t) for t in texts
        ]
        ids = [str(self._next_id + i) for i in range(len(texts))]
        payload = [{"content": t} for t in texts]
        if self._faiss is not None:
            self._faiss.add(ids, np.array(embeddings, dtype="float32"), payload)
        else:
            for emb, meta in zip(embeddings, payload, strict=False):
                self._store.append((np.array(emb, dtype="float32"), meta))
        self._next_id += len(texts)

    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        query_vec = np.array(await self.embedding_model.get_embedding(query), dtype="float32")
        if self._faiss is not None:
            results = self._faiss.search(query_vec, k)
            return [
                {"id": r["id"], "score": r["score"], "content": r["meta"].get("content", "")}
                for r in results
            ]
        if not self._store:
            return []
        qn = query_vec / np.linalg.norm(query_vec)
        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, (vec, meta) in enumerate(self._store):
            vn = vec / np.linalg.norm(vec)
            score = float(np.dot(qn, vn))
            scored.append((score, {"id": str(idx), "content": meta.get("content", "")}))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": s[1]["id"], "score": s[0], "content": s[1]["content"]}
            for s in scored[:k]
        ]


class SimpleKnowledgeConstructor(KnowledgeConstructor):
    """Create a trivial knowledge bundle from retrieved documents."""

    async def construct(
        self, query: str, retrieved_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        summary = " ".join(doc.get("content", "") for doc in retrieved_docs)
        return {"query": query, "documents": retrieved_docs, "summary": summary}


class SimpleReasoningEngine(ReasoningEngine):
    """Return a basic answer based on constructed knowledge."""

    async def reason(self, query: str, constructed_knowledge: dict[str, Any]) -> str:
        summary = constructed_knowledge.get("summary", "")
        if not summary:
            return f"No information found for: {query}"
        return f"Based on the documents, {summary}"
