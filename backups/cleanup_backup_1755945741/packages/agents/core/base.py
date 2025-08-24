#!/usr/bin/env python3
"""
Base Agent Implementation - Core Agent Framework

This module provides the foundational BaseAgent class that all AI agents inherit from.
Restored during reorganization validation to fix missing packages.agents.core.base module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import time
from typing import Any
import uuid

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response object returned by agent message processing."""

    content: str
    status: str = "success"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    conversation_id: str | None = None


@dataclass
class AgentMessage:
    """Message object for agent communication."""

    content: str
    sender: str
    recipient: str
    message_type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class BaseAgent(ABC):
    """
    Base class for all AI agents in the AIVillage system.

    Provides core functionality for:
    - Message processing and response generation
    - Agent identification and metadata management
    - Basic logging and error handling
    - Communication protocols
    """

    def __init__(
        self,
        agent_id: str,
        name: str | None = None,
        description: str | None = None,
        capabilities: list[str] | None = None,
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent
            name: Human-readable name for the agent
            description: Description of agent capabilities
            capabilities: List of capabilities this agent supports
        """
        self.agent_id = agent_id
        self.name = name or f"Agent-{agent_id}"
        self.description = description or "Base AI Agent"
        self.capabilities = capabilities or ["message_processing", "basic_reasoning"]

        # Core agent state
        self.is_active = True
        self.created_at = time.time()
        self.message_count = 0
        self.last_activity = time.time()

        # Configuration
        self.config = {"max_message_length": 10000, "response_timeout": 30.0, "logging_enabled": True}

        # Initialize logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        self.logger.info(f"Initialized agent {self.name} ({self.agent_id})")

    def process_message(self, message: str | AgentMessage) -> AgentResponse:
        """
        Process an incoming message and generate a response.

        Args:
            message: Message content (string) or AgentMessage object

        Returns:
            AgentResponse object with response content and metadata
        """
        try:
            # Update activity tracking
            self.last_activity = time.time()
            self.message_count += 1

            # Convert string to AgentMessage if needed
            if isinstance(message, str):
                agent_message = AgentMessage(content=message, sender="user", recipient=self.agent_id)
            else:
                agent_message = message

            # Validate message
            if not agent_message.content.strip():
                return AgentResponse(content="Error: Empty message received", status="error", agent_id=self.agent_id)

            if len(agent_message.content) > self.config["max_message_length"]:
                return AgentResponse(
                    content=f"Error: Message too long (max {self.config['max_message_length']} chars)",
                    status="error",
                    agent_id=self.agent_id,
                )

            # Log message processing
            if self.config["logging_enabled"]:
                self.logger.info(f"Processing message from {agent_message.sender}: {agent_message.content[:100]}...")

            # Process the message using the abstract method
            response_content = self._process_message_impl(agent_message)

            # Create response object
            response = AgentResponse(
                content=response_content,
                status="success",
                agent_id=self.agent_id,
                metadata={
                    "message_id": agent_message.message_id,
                    "processing_time": time.time() - self.last_activity,
                    "capabilities_used": self.capabilities,
                },
            )

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return AgentResponse(
                content=f"Error processing message: {str(e)}",
                status="error",
                agent_id=self.agent_id,
                metadata={"error_type": type(e).__name__},
            )

    @abstractmethod
    def _process_message_impl(self, message: AgentMessage) -> str:
        """
        Abstract method for implementing message processing logic.

        Subclasses must implement this method to provide specific agent behavior.

        Args:
            message: AgentMessage object to process

        Returns:
            String response content
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """
        Get current agent status and metadata.

        Returns:
            Dictionary with agent status information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "message_count": self.message_count,
            "uptime": time.time() - self.created_at,
        }

    def get_capabilities(self) -> list[str]:
        """Get list of agent capabilities."""
        return self.capabilities.copy()

    def add_capability(self, capability: str) -> bool:
        """
        Add a new capability to the agent.

        Args:
            capability: Capability name to add

        Returns:
            True if capability was added, False if already exists
        """
        if capability not in self.capabilities:
            self.capabilities.append(capability)
            self.logger.info(f"Added capability: {capability}")
            return True
        return False

    def remove_capability(self, capability: str) -> bool:
        """
        Remove a capability from the agent.

        Args:
            capability: Capability name to remove

        Returns:
            True if capability was removed, False if not found
        """
        if capability in self.capabilities:
            self.capabilities.remove(capability)
            self.logger.info(f"Removed capability: {capability}")
            return True
        return False

    def shutdown(self):
        """Gracefully shutdown the agent."""
        self.is_active = False
        self.logger.info(f"Agent {self.name} ({self.agent_id}) shutting down")

    def __str__(self) -> str:
        return f"BaseAgent(id={self.agent_id}, name={self.name}, active={self.is_active})"

    def __repr__(self) -> str:
        return (
            f"BaseAgent(agent_id='{self.agent_id}', name='{self.name}', "
            f"capabilities={self.capabilities}, active={self.is_active})"
        )


class SimpleAgent(BaseAgent):
    """
    Simple implementation of BaseAgent for testing and basic use cases.

    Provides basic echo functionality with simple processing patterns.
    """

    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id, **kwargs)
        self.add_capability("echo_processing")
        self.add_capability("simple_responses")

    def _process_message_impl(self, message: AgentMessage) -> str:
        """
        Simple message processing implementation.

        Provides basic responses and echo functionality.
        """
        content = message.content.lower().strip()

        # Basic response patterns
        if content in ["hello", "hi", "hey"]:
            return f"Hello! I'm {self.name}, ready to help."

        elif content in ["status", "health", "ping"]:
            status = self.get_status()
            return f"Agent {self.name} is active. Processed {status['message_count']} messages."

        elif content.startswith("echo "):
            echo_content = message.content[5:]
            return f"Echo: {echo_content}"

        elif content in ["capabilities", "help"]:
            return f"I can: {', '.join(self.get_capabilities())}"

        elif content in ["bye", "goodbye", "exit"]:
            return f"Goodbye! Agent {self.name} signing off."

        else:
            # Default response with message reflection
            return f"I received your message: '{message.content}'. I'm a simple agent and can respond to basic commands like 'hello', 'status', 'echo <text>', and 'capabilities'."


# Export the main classes
__all__ = ["BaseAgent", "SimpleAgent", "AgentResponse", "AgentMessage"]
