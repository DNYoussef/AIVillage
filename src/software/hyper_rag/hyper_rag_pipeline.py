"""Minimal HyperRAG pipeline with deterministic offline embeddings.

This implementation provides a lightweight retrieval system used by the
HyperRAG MCP server. It avoids any network calls and uses a deterministic
SimHash based embedder. Retrieval uses a hybrid score combining cosine
similarity, token overlap and an optional belief prior stored in-memory.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from .simhash_embedder import SimHashEmbedder


class RAGType(Enum):
    """Supported retrieval modes."""

    HYBRID = "hybrid"


@dataclass
class ContextTag:
    book_summary: str
    chapter_summary: str
    tag_id: str
    probability_weight: float = 1.0


@dataclass
class KnowledgeItem:
    content: str
    item_id: str
    context_tags: List[ContextTag]
    embedding: np.ndarray
    belief_probability: float = 0.0


@dataclass
class RetrievalResult:
    items: List[KnowledgeItem]
    retrieval_method: RAGType
    confidence_score: float
    bayesian_scores: Dict[str, float]
    semantic_coherence: float
    context_relevance: float
    total_items_considered: int
    metrics: Dict[str, float] = field(default_factory=dict)


class HyperRAGPipeline:
    """Simple local-only RAG pipeline."""

    def __init__(self) -> None:
        self.embedder = SimHashEmbedder()
        self.knowledge_items: Dict[str, KnowledgeItem] = {}
        # Belief priors are stored separately so they can be updated
        self.belief_priors: Dict[str, float] = {}

    async def ingest_knowledge(
        self,
        content: str,
        book_summary: str = "general",
        chapter_summary: str = "general",
        source_confidence: float = 0.0,
    ) -> str:
        """Store a knowledge item and return its id."""

        item_id = hashlib.md5(content.encode()).hexdigest()
        tag = ContextTag(book_summary, chapter_summary, item_id[:8], source_confidence)
        embedding = self.embedder.embed(content)
        item = KnowledgeItem(content, item_id, [tag], embedding, source_confidence)
        self.knowledge_items[item_id] = item
        if source_confidence:
            self.belief_priors[item_id] = source_confidence
        return item_id

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if t]

    async def search(self, query: str, max_results: int = 5) -> RetrievalResult:
        """Return top matching items using hybrid scoring."""

        start = time.perf_counter()
        query_vec = self.embedder.embed(query)
        query_tokens = self._tokenize(query)

        scored: List[tuple[float, KnowledgeItem]] = []
        for item in self.knowledge_items.values():
            cos = float(np.dot(query_vec, item.embedding))
            overlap = len(set(query_tokens) & set(self._tokenize(item.content)))
            overlap_bonus = overlap / max(len(set(query_tokens)), 1)
            prior = self.belief_priors.get(item.item_id, 0.0)
            score = cos + overlap_bonus + prior
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [itm for _, itm in scored[:max_results]]
        metrics = {
            "retrieval_ms": (time.perf_counter() - start) * 1000,
            "n_candidates": len(scored),
            "hybrid_score_top1": scored[0][0] if scored else 0.0,
        }
        bayes = {i.item_id: self.belief_priors.get(i.item_id, 0.0) for i in top_items}
        return RetrievalResult(
            items=top_items,
            retrieval_method=RAGType.HYBRID,
            confidence_score=metrics["hybrid_score_top1"],
            bayesian_scores=bayes,
            semantic_coherence=0.0,
            context_relevance=0.0,
            total_items_considered=len(scored),
            metrics=metrics,
        )
