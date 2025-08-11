from abc import ABC, abstractmethod
from typing import Any


class AgentInterface(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        pass

    @abstractmethod
    async def rerank(self, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def introspect(self) -> dict[str, Any]:
        pass

    @abstractmethod
    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        pass

    @abstractmethod
    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        pass
