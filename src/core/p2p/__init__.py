"""P2P Communication Layer for Sprint 6 - Foundation for Distributed Evolution."""

from .encryption_layer import EncryptionLayer
from .fallback_transports import FallbackTransportManager
from .libp2p_mesh import LibP2PMeshNetwork
from .mdns_discovery import mDNSDiscovery
from .mesh_network import MeshNetwork, MeshNetworkNode, create_mesh_network
from .message_protocol import EvolutionMessage, MessageProtocol
from .p2p_node import P2PNode, PeerCapabilities
from .peer_discovery import PeerDiscovery

__all__ = [
    "EncryptionLayer",
    "EvolutionMessage",
    "FallbackTransportManager",
    "LibP2PMeshNetwork",
    "MeshNetwork",
    "MeshNetworkNode",
    "MessageProtocol",
    "P2PNode",
    "PeerCapabilities",
    "PeerDiscovery",
    "create_mesh_network",
    "mDNSDiscovery",
]
