"""
Session Domain Entity

Represents user sessions and interactions with the system,
tracking context, state, and conversation history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .agent_entity import AgentId
from .user_entity import UserId


class SessionType(Enum):
    """Types of sessions"""

    CHAT = "chat"
    TASK_EXECUTION = "task_execution"
    COLLABORATION = "collaboration"
    TRAINING = "training"
    SYSTEM = "system"


class SessionStatus(Enum):
    """Session status"""

    ACTIVE = "active"
    IDLE = "idle"
    TERMINATED = "terminated"
    EXPIRED = "expired"


@dataclass
class SessionId:
    """Session identifier value object"""

    value: str

    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("SessionId must be a non-empty string")

    @classmethod
    def generate(cls) -> SessionId:
        """Generate new unique session ID"""
        return cls(str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class SessionMessage:
    """Individual message within a session"""

    timestamp: datetime
    sender_id: str  # Can be user or agent ID
    sender_type: str  # "user" or "agent"
    content: str
    message_type: str = "text"  # text, command, response, error
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """
    Core Session domain entity

    Represents an interaction session between users and agents,
    tracking conversation history, context, and state.
    """

    # Identity
    id: SessionId
    session_type: SessionType

    # Participants
    user_id: UserId
    agent_ids: list[AgentId] = field(default_factory=list)

    # Session state
    status: SessionStatus = SessionStatus.ACTIVE
    context: dict[str, Any] = field(default_factory=dict)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None

    # Conversation history
    messages: list[SessionMessage] = field(default_factory=list)

    # Session metrics
    total_messages: int = 0
    total_user_messages: int = 0
    total_agent_messages: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate session invariants"""
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError("Expiration time must be after creation time")

    def add_agent(self, agent_id: AgentId) -> None:
        """Add agent to session"""
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
            self._update_last_activity()

    def remove_agent(self, agent_id: AgentId) -> None:
        """Remove agent from session"""
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
            self._update_last_activity()

    def add_message(
        self,
        sender_id: str,
        sender_type: str,
        content: str,
        message_type: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add message to session"""
        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot add message to {self.status.value} session")

        message = SessionMessage(
            timestamp=datetime.now(),
            sender_id=sender_id,
            sender_type=sender_type,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )

        self.messages.append(message)
        self.total_messages += 1

        if sender_type == "user":
            self.total_user_messages += 1
        elif sender_type == "agent":
            self.total_agent_messages += 1

        self._update_last_activity()

    def update_context(self, key: str, value: Any) -> None:
        """Update session context"""
        self.context[key] = value
        self._update_last_activity()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get value from session context"""
        return self.context.get(key, default)

    def set_idle(self) -> None:
        """Mark session as idle"""
        if self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.IDLE

    def reactivate(self) -> None:
        """Reactivate idle session"""
        if self.status == SessionStatus.IDLE:
            self.status = SessionStatus.ACTIVE
            self._update_last_activity()

    def terminate(self) -> None:
        """Terminate session"""
        self.status = SessionStatus.TERMINATED
        self._update_last_activity()

    def is_expired(self) -> bool:
        """Check if session is expired"""
        if not self.expires_at:
            return False

        return datetime.now() > self.expires_at

    def expire(self) -> None:
        """Mark session as expired"""
        self.status = SessionStatus.EXPIRED

    def get_recent_messages(self, count: int = 10) -> list[SessionMessage]:
        """Get recent messages from session"""
        return self.messages[-count:] if self.messages else []

    def get_messages_by_sender(self, sender_id: str) -> list[SessionMessage]:
        """Get all messages from specific sender"""
        return [msg for msg in self.messages if msg.sender_id == sender_id]

    def get_session_duration_minutes(self) -> float:
        """Get session duration in minutes"""
        end_time = self.last_activity
        duration = end_time - self.created_at
        return duration.total_seconds() / 60.0

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get summary of conversation"""
        return {
            "total_messages": self.total_messages,
            "user_messages": self.total_user_messages,
            "agent_messages": self.total_agent_messages,
            "duration_minutes": self.get_session_duration_minutes(),
            "participants": {"user": str(self.user_id), "agents": [str(aid) for aid in self.agent_ids]},
            "status": self.status.value,
            "session_type": self.session_type.value,
        }

    def _update_last_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary representation"""
        return {
            "id": str(self.id),
            "session_type": self.session_type.value,
            "user_id": str(self.user_id),
            "agent_ids": [str(aid) for aid in self.agent_ids],
            "status": self.status.value,
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "total_messages": self.total_messages,
            "total_user_messages": self.total_user_messages,
            "total_agent_messages": self.total_agent_messages,
            "metadata": self.metadata,
            "messages": [
                {
                    "timestamp": msg.timestamp.isoformat(),
                    "sender_id": msg.sender_id,
                    "sender_type": msg.sender_type,
                    "content": msg.content,
                    "message_type": msg.message_type,
                    "metadata": msg.metadata,
                }
                for msg in self.messages
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Create session from dictionary representation"""
        session = cls(
            id=SessionId(data["id"]),
            session_type=SessionType(data["session_type"]),
            user_id=UserId(data["user_id"]),
            agent_ids=[AgentId(aid) for aid in data.get("agent_ids", [])],
            status=SessionStatus(data["status"]),
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            total_messages=data.get("total_messages", 0),
            total_user_messages=data.get("total_user_messages", 0),
            total_agent_messages=data.get("total_agent_messages", 0),
            metadata=data.get("metadata", {}),
        )

        # Reconstruct messages
        for msg_data in data.get("messages", []):
            session.messages.append(
                SessionMessage(
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    sender_id=msg_data["sender_id"],
                    sender_type=msg_data["sender_type"],
                    content=msg_data["content"],
                    message_type=msg_data.get("message_type", "text"),
                    metadata=msg_data.get("metadata", {}),
                )
            )

        return session
