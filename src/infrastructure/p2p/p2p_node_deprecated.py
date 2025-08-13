"""DEPRECATED: Infrastructure P2P Node Implementation

This module has been consolidated into src/production/communications/p2p/p2p_node.py

The production implementation provides all features from this module plus:
- Enhanced message validation
- Better error handling and recovery
- Improved peer discovery
- More efficient resource usage

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
    "src.infrastructure.p2p.p2p_node is deprecated. "
    "Use src.production.communications.p2p.p2p_node instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Export the production implementations
P2PNode = _P2PNode
NodeStatus = _NodeStatus
PeerInfo = _PeerInfo
MessageType = _MessageType
HandshakePayload = _HandshakePayload
HeartbeatPayload = _HeartbeatPayload
validate_payload = _validate_payload


# Backward compatibility for infrastructure-specific features
class PeerCapabilities(_PeerInfo):
    """Backward compatibility wrapper for PeerCapabilities -> PeerInfo."""

    def __init__(self, device_id: str, cpu_cores: int, ram_mb: int, **kwargs):
        warnings.warn(
            "PeerCapabilities is deprecated. Use PeerInfo instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Map infrastructure fields to production PeerInfo
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
            },
        )
