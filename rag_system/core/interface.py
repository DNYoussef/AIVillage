# rag_system/core/interfaces.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        pass

class KnowledgeConstructor(ABC):
    @abstractmethod
    async def construct(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

class ReasoningEngine(ABC):
    @abstractmethod
    async def reason(self, query: str, constructed_knowledge: Dict[str, Any]) -> str:
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        pass
