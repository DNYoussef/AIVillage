"""
Core messaging types and implementations.

Contains the fundamental message structures and metadata that power
the unified messaging system.
"""

from .message import Message, MessageType, Priority, MessageMetadata
from .message_bus import UnifiedMessageBus

__all__ = [
    "Message",
    "MessageType", 
    "Priority",
    "MessageMetadata",
    "UnifiedMessageBus",
]