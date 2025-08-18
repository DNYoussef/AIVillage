"""
Base Agent Interface

Provides common functionality for all meta-agents in the AIVillage architecture.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all AIVillage meta-agents."""

    def __init__(self, agent_id: str):
        """Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
        """
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{agent_id}")
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the agent.

        Returns:
            bool: True if initialization successful
        """
        try:
            await self._initialize_agent()
            self._initialized = True
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            return False

    @abstractmethod
    async def _initialize_agent(self) -> None:
        """Subclass-specific initialization logic."""
        pass

    async def shutdown(self) -> bool:
        """Shutdown the agent gracefully."""
        try:
            await self._shutdown_agent()
            self._initialized = False
            self.logger.info(f"Agent {self.agent_id} shutdown successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to shutdown agent {self.agent_id}: {e}")
            return False

    async def _shutdown_agent(self) -> None:
        """Subclass-specific shutdown logic."""
        pass

    @abstractmethod
    async def process_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Process an incoming message.

        Args:
            message: The message to process

        Returns:
            Optional response message
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """Get agent status information."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "initialized": self._initialized,
            "status": "active" if self._initialized else "inactive",
        }

    async def think_encrypted(self, query: str) -> str:
        """Generate encrypted thought bubbles (placeholder implementation).

        Args:
            query: The query to think about

        Returns:
            Encrypted thought result
        """
        # Simple placeholder - in real implementation this would use proper encryption
        import base64

        thought = f"Thinking about: {query} (Agent: {self.agent_id})"
        encoded = base64.b64encode(thought.encode()).decode()
        return f"ENCRYPTED_THOUGHT:{encoded}"


class AgentInterface(BaseAgent):
    """Alternative name for BaseAgent to match existing code."""

    pass
