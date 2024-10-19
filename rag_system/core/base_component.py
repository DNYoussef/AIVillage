from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseComponent(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input data and return the result."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Perform any necessary cleanup operations."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Return the current status of the component."""
        pass

    @abstractmethod
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update the component's configuration."""
        pass
