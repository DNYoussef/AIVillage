from abc import ABC, abstractmethod
from typing import Any


class BaseComponent(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return the result."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Perform any necessary cleanup operations."""

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Return the current status of the component."""

    @abstractmethod
    async def update_config(self, config: dict[str, Any]) -> None:
        """Update the component's configuration."""
