"""Communication protocol classes for AIVillage.

This module provides standardized communication classes for inter-agent messaging
and system-wide communication protocols.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class AgentMessageType(Enum):
    """Types of messages that can be sent between agents."""

    TASK = "TASK"
    RESPONSE = "RESPONSE"
    QUERY = "QUERY"
    NOTIFICATION = "NOTIFICATION"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"


class Priority(Enum):
    """Priority levels for message processing."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """A standardized message for inter-agent communication.

    This class provides a consistent format for all messages sent between
    agents in the AIVillage system.
    """

    type: AgentMessageType
    sender: str
    receiver: str
    content: Any
    id: str = field(default_factory=lambda: str(uuid4()))
    parent_id: str | None = None
    priority: Priority = Priority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "parent_id": self.parent_id,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMessage":
        """Create message from dictionary format."""
        return cls(
            type=AgentMessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            id=data["id"],
            parent_id=data.get("parent_id"),
            priority=Priority(data["priority"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


class AgentCommunicationProtocol:
    """Standard communication protocol for agent messaging.

    This class provides a unified interface for agents to communicate with
    each other using standardized message formats and protocols.
    """

    def __init__(self):
        self.subscribers: dict[str, Any] = {}
        self.message_history: list[AgentMessage] = []
        self.logger = None  # Will be set by get_component_logger

    def subscribe(self, agent_name: str, handler: Any):
        """Subscribe an agent to receive messages.

        Args:
            agent_name: Name of the subscribing agent
            handler: Async function to handle incoming messages
        """
        self.subscribers[agent_name] = handler

    def unsubscribe(self, agent_name: str):
        """Unsubscribe an agent from receiving messages.

        Args:
            agent_name: Name of the agent to unsubscribe
        """
        if agent_name in self.subscribers:
            del self.subscribers[agent_name]

    async def send_message(self, message: AgentMessage):
        """Send a message to a specific agent.

        Args:
            message: The message to send
        """
        self.message_history.append(message)

        if message.receiver in self.subscribers:
            handler = self.subscribers[message.receiver]
            await handler(message)

    async def broadcast(self, message: AgentMessage, exclude: list[str] = None):
        """Broadcast a message to all subscribed agents.

        Args:
            message: The message to broadcast
            exclude: List of agent names to exclude from broadcast
        """
        exclude = exclude or []
        self.message_history.append(message)

        for agent_name, handler in self.subscribers.items():
            if agent_name not in exclude:
                broadcast_message = AgentMessage(
                    type=message.type,
                    sender=message.sender,
                    receiver=agent_name,
                    content=message.content,
                    parent_id=message.parent_id,
                    priority=message.priority,
                    metadata=message.metadata,
                )
                await handler(broadcast_message)

    async def query(self, sender: str, receiver: str, content: Any) -> Any:
        """Send a query and wait for response.

        Args:
            sender: Name of the sending agent
            receiver: Name of the receiving agent
            content: Query content

        Returns:
            Response from the receiver
        """
        query_message = AgentMessage(
            type=AgentMessageType.QUERY,
            sender=sender,
            receiver=receiver,
            content=content,
            priority=Priority.HIGH,
        )

        await self.send_message(query_message)

        # In a real implementation, this would wait for response
        # For now, return a placeholder response
        return {"status": "query_sent", "query_id": query_message.id}

    def get_message_history(
        self, agent_name: str = None, limit: int = 100
    ) -> list[AgentMessage]:
        """Get message history for an agent or all messages.

        Args:
            agent_name: Optional agent name to filter messages
            limit: Maximum number of messages to return

        Returns:
            List of messages
        """
        if agent_name:
            messages = [
                msg
                for msg in self.message_history
                if msg.sender == agent_name or msg.receiver == agent_name
            ]
        else:
            messages = self.message_history

        return messages[-limit:] if limit else messages

    def clear_history(self, agent_name: str = None):
        """Clear message history.

        Args:
            agent_name: Optional agent name to clear only their messages
        """
        if agent_name:
            self.message_history = [
                msg
                for msg in self.message_history
                if msg.sender != agent_name and msg.receiver != agent_name
            ]
        else:
            self.message_history.clear()
