"""Refactored shared base class for specialized agents.

This module provides a composition-based agent base class that uses
specialized services instead of monolithic functionality. Domain specific
agents should subclass BaseAgent and extend its behavior only where necessary.

The refactored design follows connascence principles and reduces coupling
through dependency injection and single responsibility services.
"""

from __future__ import annotations

import logging
from typing import Any

from .agent_interface import AgentInterface
from .agent_services import (
    AgentCapabilityRegistry,
    BasicStatusProvider,
    CommunicationService,
    EmbeddingService,
    IntrospectionService,
    LatentSpaceService,
)

logger = logging.getLogger(__name__)


class BaseAgent(AgentInterface):
    """
    Refactored minimal agent base class with composable services.

    Uses composition and dependency injection instead of inheritance
    to provide agent functionality through specialized services.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list[str] | None = None,
        embedding_service: EmbeddingService | None = None,
        communication_service: CommunicationService | None = None,
        introspection_service: IntrospectionService | None = None,
        latent_space_service: LatentSpaceService | None = None,
        capability_registry: AgentCapabilityRegistry | None = None,
    ) -> None:
        """
        Initialize agent with core properties and injected services.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type/role of the agent
            capabilities: List of capability names
            embedding_service: Service for text embeddings
            communication_service: Service for inter-agent communication
            introspection_service: Service for self-analysis
            latent_space_service: Service for latent space operations
            capability_registry: Registry for capability management
        """
        # Core agent properties
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.initialized = False

        # Initialize services with dependency injection
        self._embedding_service = embedding_service or EmbeddingService()
        self._communication_service = communication_service or CommunicationService()
        self._introspection_service = introspection_service or IntrospectionService()
        self._latent_space_service = latent_space_service or LatentSpaceService()
        self._capability_registry = capability_registry or AgentCapabilityRegistry()

        # Setup introspection with basic status provider
        self._introspection_service.add_status_provider(BasicStatusProvider())

        # Register provided capabilities
        self._register_initial_capabilities()

    async def get_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for text using the embedding service.

        Args:
            text: Text to generate embedding for

        Returns:
            List of float values representing the embedding
        """
        return await self._embedding_service.get_embedding(text)

    async def communicate(self, message: str, recipient: AgentInterface) -> str:
        """
        Communicate with another agent using the communication service.

        Args:
            message: Message to send
            recipient: Agent to send message to

        Returns:
            Response from the recipient agent
        """
        return await self._communication_service.send_message(
            sender_agent=self, recipient_agent=recipient, message=message, sender_type=self.agent_type
        )

    async def introspect(self) -> dict[str, Any]:
        """
        Get comprehensive agent status using the introspection service.

        Returns:
            Dictionary with agent status and runtime information
        """
        base_context = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "initialized": self.initialized,
        }

        return await self._introspection_service.get_agent_status(base_context)

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """
        Activate latent space representation using the latent space service.

        Args:
            query: Query to process in latent space

        Returns:
            Tuple of (space_type, representation)
        """
        return await self._latent_space_service.activate_latent_space(query)

    async def has_capability(self, capability_name: str) -> bool:
        """
        Check if agent has a specific capability.

        Args:
            capability_name: Name of capability to check

        Returns:
            True if agent has the capability
        """
        if capability_name not in self.capabilities:
            return False

        context = await self.introspect()
        return await self._capability_registry.validate_capability(capability_name, context)

    async def list_capabilities(self) -> list[str]:
        """
        Get list of available capabilities.

        Returns:
            List of capability names
        """
        return self._capability_registry.get_capabilities()

    def add_capability(self, capability_name: str, description: str = "") -> None:
        """
        Add a new capability to this agent.

        Args:
            capability_name: Name of the capability
            description: Description of the capability
        """
        if capability_name not in self.capabilities:
            self.capabilities.append(capability_name)

        self._capability_registry.register_capability(
            name=capability_name, description=description or f"Capability: {capability_name}"
        )

    # Service accessor methods for customization

    def get_embedding_service(self) -> EmbeddingService:
        """Get the embedding service for customization."""
        return self._embedding_service

    def get_communication_service(self) -> CommunicationService:
        """Get the communication service for customization."""
        return self._communication_service

    def get_introspection_service(self) -> IntrospectionService:
        """Get the introspection service for customization."""
        return self._introspection_service

    def get_latent_space_service(self) -> LatentSpaceService:
        """Get the latent space service for customization."""
        return self._latent_space_service

    def get_capability_registry(self) -> AgentCapabilityRegistry:
        """Get the capability registry for customization."""
        return self._capability_registry

    # Private helper methods

    def _register_initial_capabilities(self) -> None:
        """Register the initial capabilities provided by this agent."""
        for capability in self.capabilities:
            self._capability_registry.register_capability(
                name=capability, description=f"Initial capability: {capability}"
            )

    # Abstract method that subclasses should implement
    async def generate(self, prompt: str) -> str:
        """
        Generate response to prompt.

        This method should be implemented by concrete agent subclasses
        to provide their specific generation logic.

        Args:
            prompt: Input prompt to process

        Returns:
            Generated response
        """
        # Default implementation for base agent
        return f"BaseAgent({self.agent_type}) received: {prompt}"


# Backward compatibility support
class LegacyBaseAgent(BaseAgent):
    """
    Legacy compatibility wrapper for existing code.

    Provides the original method signatures while using the new
    service-based implementation underneath.
    """

    def __init__(self, agent_id: str, agent_type: str, capabilities: list[str]) -> None:
        """Legacy constructor signature."""
        super().__init__(agent_id, agent_type, capabilities)


# Export both classes for different use cases
__all__ = ["BaseAgent", "LegacyBaseAgent"]
