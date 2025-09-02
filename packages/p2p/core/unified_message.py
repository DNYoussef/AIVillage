"""
Unified Message Implementation for P2P Communication
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class MessageType(Enum):
    DATA = "data"
    CONTROL = "control"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class UnifiedMessage:
    """Unified message structure for P2P communication"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: str
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    routing_hints: List[str] = field(default_factory=list)
    encryption_level: str = "standard"
    ttl: int = 3600  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format"""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "routing_hints": self.routing_hints,
            "encryption_level": self.encryption_level,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedMessage':
        """Create message from dictionary"""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            recipient_id=data["recipient_id"],
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            timestamp=data["timestamp"],
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            metadata=data.get("metadata", {}),
            routing_hints=data.get("routing_hints", []),
            encryption_level=data.get("encryption_level", "standard"),
            ttl=data.get("ttl", 3600)
        )