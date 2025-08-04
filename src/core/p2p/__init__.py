"""P2P Communication Layer for Sprint 6 - Foundation for Distributed Evolution."""

from .encryption_layer import EncryptionLayer
from .message_protocol import EvolutionMessage, MessageProtocol
from .p2p_node import P2PNode, PeerCapabilities
from .peer_discovery import PeerDiscovery

__all__ = [
    "EncryptionLayer",
    "EvolutionMessage",
    "MessageProtocol",
    "P2PNode",
    "PeerCapabilities",
    "PeerDiscovery",
]
