"""Component factory registering minimal RAG implementations."""

from __future__ import annotations

from typing import Any

from .core.interface import (
    EmbeddingModel,
    KnowledgeConstructor,
    ReasoningEngine,
    Retriever,
)
from .simple_components import (
    SimpleEmbeddingModel,
    SimpleKnowledgeConstructor,
    SimpleReasoningEngine,
    SimpleRetriever,
)


class ComponentFactory:
    """Factory for creating RAG system components with defaults."""

    _retrievers: dict[str, type[Retriever]] = {"default": SimpleRetriever}
    _constructors: dict[str, type[KnowledgeConstructor]] = {
        "default": SimpleKnowledgeConstructor
    }
    _reasoners: dict[str, type[ReasoningEngine]] = {
        "default": SimpleReasoningEngine
    }
    _embeddings: dict[str, type[EmbeddingModel]] = {
        "default": SimpleEmbeddingModel
    }

    @classmethod
    def create_retriever(cls, name: str = "default", **kwargs: Any) -> Retriever:
        return cls._retrievers[name](**kwargs)

    @classmethod
    def create_knowledge_constructor(
        cls, name: str = "default", **kwargs: Any
    ) -> KnowledgeConstructor:
        return cls._constructors[name](**kwargs)

    @classmethod
    def create_reasoning_engine(
        cls, name: str = "default", **kwargs: Any
    ) -> ReasoningEngine:
        return cls._reasoners[name](**kwargs)

    @classmethod
    def create_embedding_model(
        cls, name: str = "default", **kwargs: Any
    ) -> EmbeddingModel:
        return cls._embeddings[name](**kwargs)
