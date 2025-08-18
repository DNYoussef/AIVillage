"""
BitChat Bluetooth Low Energy mesh networking implementation.

Provides offline-first peer-to-peer communication using Bluetooth Low Energy
with mesh networking capabilities. Optimized for mobile devices and
battery-conscious operation.

Based on existing Android/iOS implementations and core BitChat protocol.
"""

from .ble_transport import BitChatMessage, BitChatTransport
from .mesh_network import MeshNetwork, MeshNode
from .mobile_bridge import MobileBridge

__all__ = [
    "BitChatTransport",
    "BitChatMessage",
    "MeshNetwork",
    "MeshNode",
    "MobileBridge",
]
