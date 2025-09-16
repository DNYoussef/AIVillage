"""
Unified Message Types - Consolidation of all communication patterns.

This module defines the core message structures that replace and unify:
1. Message (P2P communications)  
2. AgentMessage (core agents)
3. Chat messages (edge communications)
4. Twin communication messages  
5. Infrastructure bridge messages
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4


class MessageType(Enum):
    """
    Unified message types covering all communication patterns.
    
    Consolidates message types from all 5 communication systems.
    """
    
    # Agent communication types (from AgentMessage)
    TASK = "TASK"
    RESPONSE = "RESPONSE" 
    QUERY = "QUERY"
    NOTIFICATION = "NOTIFICATION"
    ERROR = "ERROR"
    SYSTEM = "SYSTEM"
    
    # P2P communication types
    PEER_DISCOVERY = "PEER_DISCOVERY"
    PEER_ANNOUNCEMENT = "PEER_ANNOUNCEMENT"
    HEARTBEAT = "HEARTBEAT"
    SYNC = "SYNC"
    
    # Edge communication types
    CHAT = "CHAT"
    CHAT_RESPONSE = "CHAT_RESPONSE"
    STATUS_REQUEST = "STATUS_REQUEST"
    STATUS_RESPONSE = "STATUS_RESPONSE"
    
    # Infrastructure types
    SERVICE_DISCOVERY = "SERVICE_DISCOVERY"
    HEALTH_CHECK = "HEALTH_CHECK"
    RESOURCE_UPDATE = "RESOURCE_UPDATE"
    COORDINATION = "COORDINATION"
    
    # Request-response patterns
    REQUEST = "REQUEST"
    REPLY = "REPLY"
    ACK = "ACK"
    NACK = "NACK"
    
    # Broadcast types
    BROADCAST = "BROADCAST"
    MULTICAST = "MULTICAST"


class Priority(Enum):
    """
    Message priority levels for queue management.
    
    Higher values = higher priority in processing.
    """
    
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    URGENT = 4
    CRITICAL = 5
    EMERGENCY = 10  # For system emergencies


@dataclass
class MessageMetadata:
    """
    Comprehensive metadata for message tracking and routing.
    
    Consolidates metadata patterns from all communication systems.
    """
    
    # Core tracking
    message_id: str = field(default_factory=lambda: str(uuid4()))
    correlation_id: Optional[str] = None
    parent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Timing information  
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    # Routing and delivery
    routing_key: Optional[str] = None
    reply_to: Optional[str] = None
    expects_response: bool = False
    is_response: bool = False
    
    # Retry and reliability
    attempt_count: int = 0
    max_attempts: int = 3
    
    # Transport specific
    transport_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Custom attributes
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    
    def create_response_metadata(self) -> "MessageMetadata":
        """Create metadata for a response message."""
        return MessageMetadata(
            correlation_id=self.correlation_id or self.message_id,
            parent_id=self.message_id,
            conversation_id=self.conversation_id,
            reply_to=None,  # Clear reply_to in response
            expects_response=False,
            is_response=True,
        )
    
    def increment_attempt(self) -> None:
        """Increment the attempt counter."""
        self.attempt_count += 1
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.attempt_count < self.max_attempts
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def add_transport_hint(self, key: str, value: Any) -> None:
        """Add a transport-specific hint."""
        self.transport_hints[key] = value
    
    def add_custom_attribute(self, key: str, value: Any) -> None:
        """Add a custom attribute."""
        self.custom_attributes[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "parent_id": self.parent_id, 
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "routing_key": self.routing_key,
            "reply_to": self.reply_to,
            "expects_response": self.expects_response,
            "is_response": self.is_response,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "transport_hints": self.transport_hints,
            "custom_attributes": self.custom_attributes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageMetadata":
        """Create metadata from dictionary format."""
        metadata = cls()
        metadata.message_id = data.get("message_id", str(uuid4()))
        metadata.correlation_id = data.get("correlation_id")
        metadata.parent_id = data.get("parent_id")
        metadata.conversation_id = data.get("conversation_id")
        
        if data.get("created_at"):
            metadata.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            metadata.expires_at = datetime.fromisoformat(data["expires_at"])
            
        metadata.routing_key = data.get("routing_key")
        metadata.reply_to = data.get("reply_to")
        metadata.expects_response = data.get("expects_response", False)
        metadata.is_response = data.get("is_response", False)
        metadata.attempt_count = data.get("attempt_count", 0)
        metadata.max_attempts = data.get("max_attempts", 3)
        metadata.transport_hints = data.get("transport_hints", {})
        metadata.custom_attributes = data.get("custom_attributes", {})
        
        return metadata


@dataclass 
class Message:
    """
    Unified message structure consolidating all communication patterns.
    
    Replaces and unifies:
    1. Message (P2P communications)
    2. AgentMessage (core agents) 
    3. Chat requests/responses (edge)
    4. Twin communication messages
    5. Infrastructure messages
    """
    
    # Core message identification
    type: MessageType
    sender_id: str
    recipient_id: str
    
    # Message content
    payload: Any
    
    # Message metadata and routing
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    priority: Priority = Priority.NORMAL
    
    # Timestamps
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for serialization."""
        return {
            "type": self.type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "payload": self.payload,
            "priority": self.priority.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary format."""
        return cls(
            type=MessageType(data["type"]),
            sender_id=data["sender_id"], 
            recipient_id=data["recipient_id"],
            payload=data["payload"],
            priority=Priority(data.get("priority", Priority.NORMAL.value)),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=MessageMetadata.from_dict(data.get("metadata", {})),
        )
    
    def create_response(
        self, 
        response_payload: Any, 
        response_type: Optional[MessageType] = None
    ) -> "Message":
        """Create a response message to this message."""
        if response_type is None:
            response_type = MessageType.RESPONSE
            
        return Message(
            type=response_type,
            sender_id=self.recipient_id,  # Swap sender/recipient
            recipient_id=self.sender_id,
            payload=response_payload,
            priority=self.priority,  # Inherit priority
            metadata=self.metadata.create_response_metadata(),
        )
    
    def create_error_response(self, error_message: str, error_code: Optional[str] = None) -> "Message":
        """Create an error response to this message."""
        error_payload = {
            "error": error_message,
            "error_code": error_code,
            "original_message_id": self.metadata.message_id,
        }
        
        return self.create_response(error_payload, MessageType.ERROR)
    
    def is_request(self) -> bool:
        """Check if this message expects a response."""
        return self.metadata.expects_response
    
    def is_response(self) -> bool:
        """Check if this message is a response to another message."""
        return self.metadata.is_response
    
    def is_expired(self) -> bool:
        """Check if this message has expired.""" 
        return self.metadata.is_expired()
    
    def can_retry(self) -> bool:
        """Check if this message can be retried."""
        return self.metadata.can_retry()
    
    def increment_attempt(self) -> None:
        """Increment the retry attempt counter."""
        self.metadata.increment_attempt()
    
    def __str__(self) -> str:
        """String representation of message."""
        return (
            f"Message(type={self.type.value}, "
            f"sender={self.sender_id}, "
            f"recipient={self.recipient_id}, "
            f"id={self.metadata.message_id[:8]}...)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of message."""
        return (
            f"Message("
            f"type={self.type.value}, "
            f"sender_id='{self.sender_id}', "
            f"recipient_id='{self.recipient_id}', "
            f"priority={self.priority.value}, "
            f"metadata={self.metadata.message_id}"
            f")"
        )


# Utility functions for backward compatibility

def create_agent_message(
    message_type: str,
    sender: str, 
    receiver: str,
    content: Any,
    **kwargs
) -> Message:
    """
    Create a Message using the old AgentMessage pattern.
    
    Provides backward compatibility for existing agent code.
    """
    try:
        msg_type = MessageType(message_type.upper())
    except ValueError:
        msg_type = MessageType.NOTIFICATION
    
    metadata = MessageMetadata()
    if "parent_id" in kwargs:
        metadata.parent_id = kwargs["parent_id"]
    if "conversation_id" in kwargs:
        metadata.conversation_id = kwargs["conversation_id"]
    
    priority = Priority.NORMAL
    if "priority" in kwargs:
        try:
            priority = Priority(kwargs["priority"])
        except ValueError:
            priority = Priority.NORMAL
    
    return Message(
        type=msg_type,
        sender_id=sender,
        recipient_id=receiver, 
        payload=content,
        priority=priority,
        metadata=metadata,
    )


def create_p2p_message(
    sender_id: str,
    recipient_id: str, 
    message_type: str,
    payload: Any,
    **kwargs
) -> Message:
    """
    Create a Message using the old P2P message pattern.
    
    Provides backward compatibility for P2P communication code.
    """
    try:
        msg_type = MessageType(message_type.upper())
    except ValueError:
        msg_type = MessageType.NOTIFICATION
    
    metadata = MessageMetadata()
    if "correlation_id" in kwargs:
        metadata.correlation_id = kwargs["correlation_id"]
    if "expects_response" in kwargs:
        metadata.expects_response = kwargs["expects_response"]
    
    return Message(
        type=msg_type,
        sender_id=sender_id,
        recipient_id=recipient_id,
        payload=payload,
        metadata=metadata,
    )