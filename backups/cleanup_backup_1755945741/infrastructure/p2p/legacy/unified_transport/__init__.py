"""Unified Communications System

This package consolidates ALL overlapping communication systems:

CONSOLIDATED FROM:
- src/communications/* (WebSocket + encryption)
- src/core/p2p/betanet_*.py (7 different Betanet implementations)
- src/core/p2p/libp2p_mesh.py (Mesh networking)
- src/core/p2p/mesh_network.py (High-level mesh)
- betanet-gateway/* (Rust performance layer)
- agents/*/navigator_agent.py (Routing logic)

RESULT: One unified, non-overlapping communication architecture
"""

# Backward compatibility for existing imports
from .core.unified_transport import (
    BetanetTransportV2,
    CommunicationsProtocol,
    RoutingMode,
    TransportType,
    UnifiedCommunicationHub,
    UnifiedMessage,
    create_communication_hub,
)

__all__ = [
    "UnifiedCommunicationHub",
    "UnifiedMessage",
    "TransportType",
    "RoutingMode",
    "create_communication_hub",
    # Backward compatibility
    "CommunicationsProtocol",
    "BetanetTransportV2",
]

__version__ = "2.0.0"
__status__ = "Consolidated"
