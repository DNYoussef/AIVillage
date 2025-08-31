"""
Service interfaces for SageAgent components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime


class ICognitiveService(Protocol):
    """Interface for cognitive processing services."""

    async def process(self, data: Any) -> Any:
        """Process cognitive data."""
        ...

    async def evolve(self) -> None:
        """Evolve cognitive capabilities."""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Get current cognitive state."""
        ...


class IProcessingService(Protocol):
    """Interface for processing chain services."""

    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a query through the service chain."""
        ...

    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute a task."""
        ...

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the processing service."""
        ...


class IKnowledgeService(Protocol):
    """Interface for knowledge management services."""

    async def store_knowledge(self, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store knowledge and return identifier."""
        ...

    async def retrieve_knowledge(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge."""
        ...

    async def update_knowledge(self, knowledge_id: str, data: Any) -> bool:
        """Update existing knowledge."""
        ...


class ILearningService(Protocol):
    """Interface for learning and adaptation services."""

    async def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Learn from user/system feedback."""
        ...

    async def adapt_behavior(self, performance_metrics: Dict[str, Any]) -> None:
        """Adapt behavior based on performance."""
        ...

    def get_learning_state(self) -> Dict[str, Any]:
        """Get current learning state."""
        ...


class IErrorHandlingService(Protocol):
    """Interface for error handling services."""

    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Handle errors with context-aware recovery."""
        ...

    def configure_recovery_strategies(self, strategies: Dict[str, Any]) -> None:
        """Configure error recovery strategies."""
        ...


class ICollaborationService(Protocol):
    """Interface for agent collaboration services."""

    async def collaborate_on_task(self, task: Dict[str, Any], agents: List[str]) -> Dict[str, Any]:
        """Collaborate with other agents on a task."""
        ...

    async def share_knowledge(self, knowledge: Any, target_agents: List[str]) -> bool:
        """Share knowledge with other agents."""
        ...

    def get_collaboration_state(self) -> Dict[str, Any]:
        """Get current collaboration state."""
        ...


class IResearchService(Protocol):
    """Interface for research capability services."""

    async def conduct_research(self, topic: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conduct research on a topic."""
        ...

    async def web_scrape(self, url: str) -> Dict[str, Any]:
        """Scrape web content."""
        ...

    async def web_search(self, query: str) -> Dict[str, Any]:
        """Perform web search."""
        ...


# Abstract base classes for concrete implementations

class AbstractServiceBase(ABC):
    """Base class for all services with common functionality."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        self._last_used = datetime.now()

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service gracefully."""
        pass

    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized

    def update_last_used(self) -> None:
        """Update last used timestamp."""
        self._last_used = datetime.now()

    def get_last_used(self) -> datetime:
        """Get last used timestamp."""
        return self._last_used