"""
Core P2P transport interfaces and coordination.

Provides the fundamental abstractions and coordination layer for all P2P transports.
"""

from .message_types import MessageMetadata, MessagePriority, UnifiedMessage
from .transport_manager import TransportManager, TransportPriority, TransportType

__all__ = [
    "TransportManager",
    "TransportType",
    "TransportPriority",
    "UnifiedMessage",
    "MessagePriority",
    "MessageMetadata",
]
