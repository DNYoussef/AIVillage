"""P2P module for mobile integration.

This module provides the interface that mobile JNI bridge expects to import.
It exports the core LibP2P functionality needed by the Android bridge.
"""

# Export main classes that mobile bridge expects
# Export convenience functions
from .libp2p_mesh import (
    LibP2PMeshNetwork,
    MeshConfiguration,
    MeshMessage,
    MeshMessageType,
    MeshPeerCapabilities,
    MeshStatus,
    create_mesh_message,
    get_default_config,
)

__all__ = [
    "LibP2PMeshNetwork",
    "MeshConfiguration",
    "MeshMessage",
    "MeshMessageType",
    "MeshPeerCapabilities",
    "MeshStatus",
    "create_mesh_message",
    "get_default_config",
]
