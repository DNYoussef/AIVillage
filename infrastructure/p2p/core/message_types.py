"""Core message types for P2P communication."""

from dataclasses import dataclass
from enum import Enum
import time
from typing import Any, Dict, Optional
import uuid


class MessageType(Enum):
    """Standard message types for P2P communication."""

    HEARTBEAT = "heartbeat"
    DATA = "data"
    CONTROL = "control"
    DISCOVERY = "discovery"
    ACK = "acknowledgment"


class MessagePriority(Enum):
    """Message priority levels."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class MessageMetadata:
    """Message metadata for routing and processing."""

    sender_id: str
    receiver_id: str
    timestamp: float
    message_id: str
    ttl: int = 10
    hop_count: int = 0
    signature: Optional[str] = None

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if self.timestamp <= 0:
            self.timestamp = time.time()


@dataclass
class Message:
    """Basic P2P message structure."""

    message_type: MessageType
    sender_id: str
    receiver_id: str
    payload: Dict[str, Any]
    timestamp: float = None
    message_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
        }


@dataclass
class UnifiedMessage:
    """Unified message format for all P2P communications."""

    metadata: MessageMetadata
    content: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    encrypted: bool = False

    def __post_init__(self):
        if not self.metadata.message_id:
            self.metadata.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert unified message to dictionary."""
        return {
            "metadata": {
                "sender_id": self.metadata.sender_id,
                "receiver_id": self.metadata.receiver_id,
                "timestamp": self.metadata.timestamp,
                "message_id": self.metadata.message_id,
                "ttl": self.metadata.ttl,
                "hop_count": self.metadata.hop_count,
                "signature": self.metadata.signature,
            },
            "content": self.content,
            "priority": self.priority.value,
            "encrypted": self.encrypted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create UnifiedMessage from dictionary."""
        metadata_data = data.get("metadata", {})
        metadata = MessageMetadata(
            sender_id=metadata_data.get("sender_id", ""),
            receiver_id=metadata_data.get("receiver_id", ""),
            timestamp=metadata_data.get("timestamp", time.time()),
            message_id=metadata_data.get("message_id", str(uuid.uuid4())),
            ttl=metadata_data.get("ttl", 10),
            hop_count=metadata_data.get("hop_count", 0),
            signature=metadata_data.get("signature"),
        )

        return cls(
            metadata=metadata,
            content=data.get("content", {}),
            priority=MessagePriority(data.get("priority", MessagePriority.NORMAL.value)),
            encrypted=data.get("encrypted", False),
        )
