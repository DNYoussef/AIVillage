"""Base component for RAG system modules."""

from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseComponent(ABC):
    """Abstract base class for RAG system components."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        pass

    @abstractmethod
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        pass
