from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

class AgentInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def rerank(self, query: str, results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def introspect(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def communicate(self, message: str, recipient: 'AgentInterface') -> str:
        pass

    @abstractmethod
    async def activate_latent_space(self, query: str) -> Tuple[str, str]:
        pass
