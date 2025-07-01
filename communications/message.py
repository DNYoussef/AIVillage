from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
import uuid


class MessageType(Enum):
    TASK = "task"
    RESPONSE = "response"
    QUERY = "query"
    NOTIFICATION = "notification"
    # Additional types used by collaboration features
    COLLABORATION_REQUEST = "collaboration_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    TASK_RESULT = "task_result"
    JOINT_REASONING_RESULT = "joint_reasoning_result"
    UPDATE = "update"
    COMMAND = "command"
    BULK_UPDATE = "bulk_update"
    PROJECT_UPDATE = "project_update"
    SYSTEM_STATUS_UPDATE = "system_status_update"
    CONFIG_UPDATE = "config_update"
    TOOL_CALL = "tool_call"
    EVIDENCE = "evidence"


class Priority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True)
class Message:
    type: MessageType
    sender: str
    receiver: str
    content: Dict[str, Any]
    metadata: Dict[str, Any] | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: Priority = Priority.MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            id=data.get("id"),
            type=MessageType(data["type"]),
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            parent_id=data.get("parent_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=Priority(data["priority"]),
            metadata=data.get("metadata"),
        )

    def with_updated_content(self, new_content: Dict[str, Any]) -> "Message":
        """Return a new Message with updated content."""
        return Message(
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content=new_content,
            id=self.id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            priority=self.priority,
            metadata=self.metadata,
        )

    def with_updated_priority(self, new_priority: Priority) -> "Message":
        """Return a new Message with a different priority."""
        return Message(
            type=self.type,
            sender=self.sender,
            receiver=self.receiver,
            content=self.content,
            id=self.id,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            priority=new_priority,
            metadata=self.metadata,
        )
