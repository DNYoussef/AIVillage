"""Minimal RAG pipeline for import validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """Simple text document."""

    id: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalResult:
    """Result returned by :class:`EnhancedRAGPipeline.retrieve`."""

    id: int
    text: str
    score: float


class EnhancedRAGPipeline:
    """Tiny RAG pipeline that stores documents and retrieves none."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.documents: list[Document] = []

    def add_document(self, doc: Document) -> None:
        self.documents.append(doc)

    def retrieve(self, query: str, top_k: int = 1) -> list[RetrievalResult]:
        return []


class RAGPipeline(EnhancedRAGPipeline):
    """Backward compatible alias for :class:`EnhancedRAGPipeline`."""


__all__ = [
    "Document",
    "EnhancedRAGPipeline",
    "RAGPipeline",
    "RetrievalResult",
]
