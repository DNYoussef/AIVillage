"""
AIVillage Unified P2P Communication System

Provides comprehensive peer-to-peer communication capabilities including:
- BitChat: Bluetooth Low Energy mesh networking for offline-first communication
- BetaNet: Encrypted internet backbone with HTX transport protocol
- Unified transport management with automatic failover and routing
- Mobile platform optimizations and cross-platform compatibility

This package consolidates all P2P implementations into a single, coherent system
based on the production-ready betanet bounty implementation.
"""

from .betanet.htx_transport import HtxClient, HtxServer
from .bitchat.ble_transport import BitChatTransport
from .bridges.compatibility import create_legacy_bridge
from .core.message_types import MessagePriority, UnifiedMessage
from .core.transport_manager import TransportManager, TransportPriority, TransportType

__version__ = "2.0.0"
__all__ = [
    # Core transport management
    "TransportManager",
    "TransportType",
    "TransportPriority",
    # Message types
    "UnifiedMessage",
    "MessagePriority",
    # Transport implementations
    "HtxClient",
    "HtxServer",
    "BitChatTransport",
    # Compatibility
    "create_legacy_bridge",
]
