from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, Optional
import time
import uuid

class MessageType(Enum):
    TASK = auto()
    QUERY = auto()
    RESPONSE = auto()
    UPDATE = auto()
    COMMAND = auto()
    BULK_UPDATE = auto()
    PROJECT_UPDATE = auto()
    SYSTEM_STATUS_UPDATE = auto()
    CONFIG_UPDATE = auto()

class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass(frozen=True)
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    priority: Priority = Priority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updated_content(self, new_content: Dict[str, Any]) -> 'Message':
        return Message(
            id=self.id,
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content={**self.content, **new_content},
            priority=self.priority,
            timestamp=self.timestamp,
            parent_id=self.parent_id,
            metadata=self.metadata
        )

    def with_updated_priority(self, new_priority: Priority) -> 'Message':
        return Message(
            id=self.id,
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content=self.content,
            priority=new_priority,
            timestamp=self.timestamp,
            parent_id=self.parent_id,
            metadata=self.metadata
        )