"""Legacy adapter for P2PNode within production communications."""

from AIVillage.src.core.p2p.legacy import (
    MessageType,
    NodeStatus,
    P2PMessage,
    PeerInfo,
    P2PNode,
)

__all__ = ["P2PNode", "NodeStatus", "PeerInfo", "MessageType", "P2PMessage"]
