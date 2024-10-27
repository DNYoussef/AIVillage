from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class MessageType(Enum):
    """Types of messages that can be exchanged."""
    TASK = "task"
    RESPONSE = "response"
    QUERY = "query"
    NOTIFICATION = "notification"

class Priority(Enum):
    """Message priority levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2

@dataclass(frozen=True)
class Message:
    """
    Immutable message class for agent communication.
    
    Attributes:
        type: The type of message (task, response, query, notification)
        sender: ID of the sending agent
        receiver: ID of the receiving agent
        content: The message payload
        id: Unique message identifier
        parent_id: ID of the parent message if this is a response
        timestamp: When the message was created
        priority: Message priority level
    """
    type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary format."""
        return cls(
            id=data.get("id"),
            type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=Priority(data["priority"])
        )
