"""DEPRECATED: P2P Node Implementation

This module has been consolidated into src/production/communications/p2p/p2p_node.py

The production implementation provides all features from this module plus:
- Better Windows compatibility
- Enhanced validation with Pydantic
- Improved error handling
- More efficient resource management

Please update your imports to use the production version:
  from src.production.communications.p2p.p2p_node import P2PNode, NodeStatus, PeerInfo

This shim will be removed in a future version.
"""

import warnings

from src.production.communications.p2p.p2p_node import (
    HandshakePayload as _HandshakePayload,
)
from src.production.communications.p2p.p2p_node import (
    HeartbeatPayload as _HeartbeatPayload,
)
from src.production.communications.p2p.p2p_node import MessageType as _MessageType
from src.production.communications.p2p.p2p_node import NodeStatus as _NodeStatus
from src.production.communications.p2p.p2p_node import P2PNode as _P2PNode
from src.production.communications.p2p.p2p_node import PeerInfo as _PeerInfo
from src.production.communications.p2p.p2p_node import (
    validate_payload as _validate_payload,
)

warnings.warn(
    "src.core.p2p.p2p_node is deprecated. "
    "Use src.production.communications.p2p.p2p_node instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Export the production implementations with deprecation warning
P2PNode = _P2PNode
NodeStatus = _NodeStatus
PeerInfo = _PeerInfo
MessageType = _MessageType
HandshakePayload = _HandshakePayload
HeartbeatPayload = _HeartbeatPayload
validate_payload = _validate_payload


# For backward compatibility with evolution-specific features
class PeerCapabilities(_PeerInfo):
    """Backward compatibility wrapper for PeerCapabilities -> PeerInfo."""

    def __init__(self, device_id: str, cpu_cores: int, ram_mb: int, **kwargs):
        warnings.warn(
            "PeerCapabilities is deprecated. Use PeerInfo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map old fields to new PeerInfo structure
        super().__init__(
            peer_id=device_id,
            address=kwargs.get("address", "localhost"),
            port=kwargs.get("port", 0),
            capabilities={
                "cpu_cores": cpu_cores,
                "ram_mb": ram_mb,
                "battery_percent": kwargs.get("battery_percent"),
                "network_type": kwargs.get("network_type", "unknown"),
                "trust_score": kwargs.get("trust_score", 0.5),
                "latency_ms": kwargs.get("latency_ms", 0.0),
                "bandwidth_kbps": kwargs.get("bandwidth_kbps"),
                "can_evolve": kwargs.get("can_evolve", True),
                "evolution_capacity": kwargs.get("evolution_capacity", 1.0),
                "available_for_evolution": kwargs.get("available_for_evolution", True),
            },
        )


# Evolution-specific helper functions for backward compatibility
def is_suitable_for_evolution(peer_info: _PeerInfo) -> bool:
    """Check if peer is suitable for evolution tasks."""
    warnings.warn(
        "is_suitable_for_evolution is deprecated. "
        "Use peer_info.capabilities.get('can_evolve', True) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return peer_info.capabilities.get("can_evolve", True)


def get_evolution_priority(peer_info: _PeerInfo) -> float:
    """Get evolution priority for peer."""
    warnings.warn(
        "get_evolution_priority is deprecated. "
        "Use peer_info.capabilities.get('evolution_capacity', 1.0) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return peer_info.capabilities.get("evolution_capacity", 1.0)
