# rag_system/core/agent_interface.py

from abc import ABC, abstractmethod

class AgentInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response based on the given prompt."""
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        """Get the embedding for the given text."""
        pass

    @abstractmethod
    async def rerank(self, query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        """Rerank the given results based on the query."""
        pass
