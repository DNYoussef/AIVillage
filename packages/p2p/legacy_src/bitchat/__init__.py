"""
BitChat - Bluetooth Mesh Protocol

Handles Bluetooth-based mesh networking for:
- Offline/local communications
- Edge device discovery
- Low-power mesh routing
- Data cost-aware communications
"""

from .bitchat_transport import BitChatTransport
from .mesh_manager import BluetoothMeshManager

__all__ = ["BitChatTransport", "BluetoothMeshManager"]
