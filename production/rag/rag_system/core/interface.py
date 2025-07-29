# rag_system/core/interfaces.py

from abc import ABC, abstractmethod
from typing import Any


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, k: int) -> list[dict[str, Any]]:
        pass

class KnowledgeConstructor(ABC):
    @abstractmethod
    async def construct(self, query: str, retrieved_docs: list[dict[str, Any]]) -> dict[str, Any]:
        pass

class ReasoningEngine(ABC):
    @abstractmethod
    async def reason(self, query: str, constructed_knowledge: dict[str, Any]) -> str:
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass
