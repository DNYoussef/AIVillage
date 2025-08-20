"""
Unified message types and formats for P2P communication.

Provides standardized message formats that work across all transport types
including BitChat, BetaNet, and direct QUIC connections. Payloads are
base64-encoded to maintain interoperability across different transports.
"""

import base64
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MessagePriority(Enum):
    """Message priority levels for transport selection and handling."""

    CRITICAL = 1  # Emergency/critical system messages
    HIGH = 2  # High priority agent coordination
    NORMAL = 3  # Standard application messages
    LOW = 4  # Background synchronization
    BULK = 5  # Large data transfers


class MessageType(Enum):
    """Core message types supported across all transports."""

    DATA = "data"  # General data payload
    AGENT_TASK = "agent_task"  # Agent coordination/task assignment
    PARAMETER_UPDATE = "param_sync"  # Model parameter synchronization
    GRADIENT_SHARE = "gradient"  # Federated learning gradients
    HEARTBEAT = "heartbeat"  # Keep-alive/presence
    DISCOVERY = "discovery"  # Peer discovery
    ROUTING = "routing"  # Routing table updates
    SYSTEM = "system"  # System control messages


@dataclass
class MessageMetadata:
    """Metadata associated with messages for routing and processing."""

    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    # Routing information
    sender_id: str = ""
    recipient_id: str = ""
    hop_count: int = 0
    max_hops: int = 7
    route_path: list[str] = field(default_factory=list)

    # Transport preferences
    transport_hint: str | None = None
    encryption_required: bool = True
    compression_enabled: bool = True

    # Quality of Service
    priority: MessagePriority = MessagePriority.NORMAL
    ttl_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

    # Mobile optimization
    battery_aware: bool = True
    data_cost_aware: bool = True
    offline_capable: bool = True

    # Custom extensions
    custom_headers: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedMessage:
    """
    Unified message format that works across all P2P transport types.

    This format provides a common interface for BitChat, BetaNet, and QUIC
    transports while preserving transport-specific optimizations.
    """

    # Core message data
    message_type: MessageType = MessageType.DATA
    payload: bytes = b""

    # Message metadata
    metadata: MessageMetadata = field(default_factory=MessageMetadata)

    # Chunking support for large messages
    chunk_index: int = 0
    total_chunks: int = 1
    max_chunk_size: int = 16384  # 16KB default

    def __post_init__(self):
        """Validate message after initialization."""
        if not self.metadata.message_id:
            self.metadata.message_id = str(uuid.uuid4())

        if not self.metadata.timestamp:
            self.metadata.timestamp = time.time()

    @property
    def is_chunked(self) -> bool:
        """Check if this message is part of a chunked transfer."""
        return self.total_chunks > 1

    @property
    def is_final_chunk(self) -> bool:
        """Check if this is the final chunk in a chunked transfer."""
        return self.chunk_index == self.total_chunks - 1

    @property
    def size_bytes(self) -> int:
        """Get message payload size in bytes."""
        return len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "message_type": self.message_type.value,
            "payload": base64.b64encode(self.payload).decode("utf-8"),  # Base64 encode for transport interoperability
            "metadata": {
                "message_id": self.metadata.message_id,
                "correlation_id": self.metadata.correlation_id,
                "timestamp": self.metadata.timestamp,
                "sender_id": self.metadata.sender_id,
                "recipient_id": self.metadata.recipient_id,
                "hop_count": self.metadata.hop_count,
                "max_hops": self.metadata.max_hops,
                "route_path": self.metadata.route_path,
                "transport_hint": self.metadata.transport_hint,
                "encryption_required": self.metadata.encryption_required,
                "compression_enabled": self.metadata.compression_enabled,
                "priority": self.metadata.priority.value,
                "ttl_seconds": self.metadata.ttl_seconds,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "battery_aware": self.metadata.battery_aware,
                "data_cost_aware": self.metadata.data_cost_aware,
                "offline_capable": self.metadata.offline_capable,
                "custom_headers": self.metadata.custom_headers,
            },
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "max_chunk_size": self.max_chunk_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedMessage":
        """Create message from dictionary."""
        metadata = MessageMetadata(
            message_id=data["metadata"]["message_id"],
            correlation_id=data["metadata"].get("correlation_id"),
            timestamp=data["metadata"]["timestamp"],
            sender_id=data["metadata"]["sender_id"],
            recipient_id=data["metadata"]["recipient_id"],
            hop_count=data["metadata"]["hop_count"],
            max_hops=data["metadata"]["max_hops"],
            route_path=data["metadata"]["route_path"],
            transport_hint=data["metadata"].get("transport_hint"),
            encryption_required=data["metadata"]["encryption_required"],
            compression_enabled=data["metadata"]["compression_enabled"],
            priority=MessagePriority(data["metadata"]["priority"]),
            ttl_seconds=data["metadata"]["ttl_seconds"],
            retry_count=data["metadata"]["retry_count"],
            max_retries=data["metadata"]["max_retries"],
            battery_aware=data["metadata"]["battery_aware"],
            data_cost_aware=data["metadata"]["data_cost_aware"],
            offline_capable=data["metadata"]["offline_capable"],
            custom_headers=data["metadata"]["custom_headers"],
        )

        return cls(
            message_type=MessageType(data["message_type"]),
            payload=base64.b64decode(data["payload"]),
            metadata=metadata,
            chunk_index=data["chunk_index"],
            total_chunks=data["total_chunks"],
            max_chunk_size=data["max_chunk_size"],
        )

    def create_chunk(self, chunk_data: bytes, chunk_index: int) -> "UnifiedMessage":
        """Create a new chunk message from this message."""
        chunk_msg = UnifiedMessage(
            message_type=self.message_type,
            payload=chunk_data,
            metadata=MessageMetadata(
                message_id=str(uuid.uuid4()),  # New ID for chunk
                correlation_id=self.metadata.message_id,  # Link to parent
                timestamp=time.time(),
                sender_id=self.metadata.sender_id,
                recipient_id=self.metadata.recipient_id,
                hop_count=self.metadata.hop_count,
                max_hops=self.metadata.max_hops,
                route_path=self.metadata.route_path.copy(),
                transport_hint=self.metadata.transport_hint,
                encryption_required=self.metadata.encryption_required,
                compression_enabled=self.metadata.compression_enabled,
                priority=self.metadata.priority,
                ttl_seconds=self.metadata.ttl_seconds,
                retry_count=0,  # Reset retry count for chunk
                max_retries=self.metadata.max_retries,
                battery_aware=self.metadata.battery_aware,
                data_cost_aware=self.metadata.data_cost_aware,
                offline_capable=self.metadata.offline_capable,
                custom_headers=self.metadata.custom_headers.copy(),
            ),
            chunk_index=chunk_index,
            total_chunks=self.total_chunks,
            max_chunk_size=self.max_chunk_size,
        )
        return chunk_msg


def create_data_message(
    recipient_id: str, payload: bytes, sender_id: str = "", priority: MessagePriority = MessagePriority.NORMAL, **kwargs
) -> UnifiedMessage:
    """Factory function to create a data message."""
    metadata = MessageMetadata(sender_id=sender_id, recipient_id=recipient_id, priority=priority, **kwargs)

    return UnifiedMessage(
        message_type=MessageType.DATA,
        payload=payload,
        metadata=metadata,
    )


def create_agent_task_message(
    recipient_id: str, task_data: bytes, sender_id: str = "", priority: MessagePriority = MessagePriority.HIGH, **kwargs
) -> UnifiedMessage:
    """Factory function to create an agent task message."""
    metadata = MessageMetadata(sender_id=sender_id, recipient_id=recipient_id, priority=priority, **kwargs)

    return UnifiedMessage(
        message_type=MessageType.AGENT_TASK,
        payload=task_data,
        metadata=metadata,
    )


def create_heartbeat_message(
    sender_id: str, recipient_id: str = "broadcast", priority: MessagePriority = MessagePriority.LOW, **kwargs
) -> UnifiedMessage:
    """Factory function to create a heartbeat message."""
    metadata = MessageMetadata(
        sender_id=sender_id,
        recipient_id=recipient_id,
        priority=priority,
        ttl_seconds=60,  # Short TTL for heartbeats
        **kwargs,
    )

    return UnifiedMessage(
        message_type=MessageType.HEARTBEAT,
        payload=b"",
        metadata=metadata,
    )
