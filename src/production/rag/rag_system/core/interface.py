# rag_system/core/interfaces.py

from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        raise RuntimeError("Retriever.retrieve must be implemented by subclasses")


class KnowledgeConstructor(ABC):
    @abstractmethod
    async def construct(
        self, query: str, retrieved_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        raise RuntimeError("KnowledgeConstructor.construct must be implemented by subclasses")


class ReasoningEngine(ABC):
    @abstractmethod
    async def reason(self, query: str, constructed_knowledge: dict[str, Any]) -> str:
        raise RuntimeError("ReasoningEngine.reason must be implemented by subclasses")


class EmbeddingModel(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        raise RuntimeError("EmbeddingModel.get_embedding must be implemented by subclasses")
