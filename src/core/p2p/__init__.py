"""P2P Communication Layer for Sprint 6 - Foundation for Distributed Evolution"""

from .p2p_node import P2PNode, PeerCapabilities
from .peer_discovery import PeerDiscovery
from .message_protocol import MessageProtocol, EvolutionMessage
from .encryption_layer import EncryptionLayer

__all__ = [
    'P2PNode',
    'PeerCapabilities', 
    'PeerDiscovery',
    'MessageProtocol',
    'EvolutionMessage',
    'EncryptionLayer'
]