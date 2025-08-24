"""
Composable Agent Services

Extracted common agent functionality into focused, reusable services.
Follows single responsibility principle and dependency injection.
"""

from abc import ABC, abstractmethod
import hashlib
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class AgentInterface(Protocol):
    """Protocol for agent interface - weak connascence."""

    async def generate(self, prompt: str) -> str:
        """Generate response to prompt."""
        ...


class EmbeddingService:
    """
    Service for generating embeddings from text.

    Focused on text embedding concerns with pluggable implementations.
    """

    def __init__(self, embedding_strategy: "EmbeddingStrategy | None" = None):
        self._strategy = embedding_strategy or HashEmbeddingStrategy()

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for given text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the embedding
        """
        return await self._strategy.generate_embedding(text)

    def set_strategy(self, strategy: "EmbeddingStrategy") -> None:
        """Set a new embedding strategy."""
        self._strategy = strategy


class EmbeddingStrategy(ABC):
    """Abstract strategy for text embedding generation."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class HashEmbeddingStrategy(EmbeddingStrategy):
    """Simple hash-based embedding strategy for testing/fallback."""

    EMBEDDING_DIM = 384

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate a simple hash-based embedding for text."""
        hash_value = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(), 16)  # nosec B324
        return [(hash_value % 1000) / 1000.0] * self.EMBEDDING_DIM


class CommunicationService:
    """
    Service for inter-agent communication.

    Handles message formatting and agent-to-agent interactions.
    """

    def __init__(self, message_formatter: "MessageFormatter | None" = None):
        self._formatter = message_formatter or DefaultMessageFormatter()

    async def send_message(
        self, sender_agent: AgentInterface, recipient_agent: AgentInterface, message: str, sender_type: str = "Agent"
    ) -> str:
        """
        Send message from one agent to another.

        Args:
            sender_agent: The agent sending the message
            recipient_agent: The agent receiving the message
            message: Message content to send
            sender_type: Type/role of the sending agent

        Returns:
            Response from the recipient agent
        """
        formatted_message = self._formatter.format_message(sender_type, message)
        response = await recipient_agent.generate(formatted_message)
        return self._formatter.format_response(response)

    def set_formatter(self, formatter: "MessageFormatter") -> None:
        """Set a new message formatter."""
        self._formatter = formatter


class MessageFormatter(ABC):
    """Abstract formatter for agent messages."""

    @abstractmethod
    def format_message(self, sender_type: str, message: str) -> str:
        """Format outgoing message."""
        ...

    @abstractmethod
    def format_response(self, response: str) -> str:
        """Format incoming response."""
        ...


class DefaultMessageFormatter(MessageFormatter):
    """Default message formatter with simple formatting."""

    def format_message(self, sender_type: str, message: str) -> str:
        """Format message with sender type prefix."""
        return f"{sender_type} says: {message}"

    def format_response(self, response: str) -> str:
        """Format response with received prefix."""
        return f"Received response: {response}"


class IntrospectionService:
    """
    Service for agent self-analysis and status reporting.

    Provides standardized introspection capabilities.
    """

    def __init__(self):
        self._status_providers: list[StatusProvider] = []

    def add_status_provider(self, provider: "StatusProvider") -> None:
        """Add a status information provider."""
        self._status_providers.append(provider)

    async def get_agent_status(self, agent_context: dict[str, Any]) -> dict[str, Any]:
        """
        Get comprehensive agent status information.

        Args:
            agent_context: Basic agent information

        Returns:
            Dictionary with complete status information
        """
        status = dict(agent_context)  # Start with basic info

        # Collect status from all providers
        for provider in self._status_providers:
            try:
                provider_status = await provider.get_status()
                status.update(provider_status)
            except Exception as e:
                logger.warning(f"Status provider {provider.__class__.__name__} failed: {e}")
                status[f"{provider.__class__.__name__}_error"] = str(e)

        return status


class StatusProvider(ABC):
    """Abstract provider for status information."""

    @abstractmethod
    async def get_status(self) -> dict[str, Any]:
        """Get status information."""
        ...


class BasicStatusProvider(StatusProvider):
    """Provides basic runtime status information."""

    async def get_status(self) -> dict[str, Any]:
        """Get basic status information."""
        import time

        import psutil

        return {
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.Process().cpu_percent(),
            "uptime_seconds": time.time() - psutil.Process().create_time(),
        }


class LatentSpaceService:
    """
    Service for latent space operations and representation.

    Handles latent space navigation and representation generation.
    """

    def __init__(self, space_analyzer: "LatentSpaceAnalyzer | None" = None):
        self._analyzer = space_analyzer or DefaultLatentSpaceAnalyzer()

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """
        Activate latent space representation for query.

        Args:
            query: Input query to analyze

        Returns:
            Tuple of (space_type, representation)
        """
        return await self._analyzer.analyze_query(query)

    def set_analyzer(self, analyzer: "LatentSpaceAnalyzer") -> None:
        """Set a new latent space analyzer."""
        self._analyzer = analyzer


class LatentSpaceAnalyzer(ABC):
    """Abstract analyzer for latent space operations."""

    @abstractmethod
    async def analyze_query(self, query: str) -> tuple[str, str]:
        """Analyze query and return space type and representation."""
        ...


class DefaultLatentSpaceAnalyzer(LatentSpaceAnalyzer):
    """Default latent space analyzer with basic functionality."""

    MAX_QUERY_LENGTH = 50

    async def analyze_query(self, query: str) -> tuple[str, str]:
        """Return generic latent space representation."""
        truncated_query = query[: self.MAX_QUERY_LENGTH]
        return "general", f"LATENT[general:{truncated_query}]"


class AgentCapabilityRegistry:
    """
    Registry for agent capabilities and features.

    Manages capability discovery and validation.
    """

    def __init__(self):
        self._capabilities: dict[str, CapabilityDefinition] = {}

    def register_capability(self, name: str, description: str, validator: "CapabilityValidator | None" = None) -> None:
        """Register a new capability."""
        self._capabilities[name] = CapabilityDefinition(
            name=name, description=description, validator=validator or DefaultCapabilityValidator()
        )

    def get_capabilities(self) -> list[str]:
        """Get list of registered capability names."""
        return list(self._capabilities.keys())

    async def validate_capability(self, name: str, context: dict[str, Any]) -> bool:
        """Validate if an agent has a specific capability."""
        if name not in self._capabilities:
            return False

        capability = self._capabilities[name]
        return await capability.validator.validate(context)


class CapabilityDefinition:
    """Definition of an agent capability."""

    def __init__(self, name: str, description: str, validator: "CapabilityValidator"):
        self.name = name
        self.description = description
        self.validator = validator


class CapabilityValidator(ABC):
    """Abstract validator for agent capabilities."""

    @abstractmethod
    async def validate(self, context: dict[str, Any]) -> bool:
        """Validate capability in given context."""
        ...


class DefaultCapabilityValidator(CapabilityValidator):
    """Default capability validator that always returns True."""

    async def validate(self, context: dict[str, Any]) -> bool:
        """Default validation always passes."""
        return True
