"""P2P Communication Layer for AIVillage.

This module provides peer-to-peer networking capabilities including:
- Node discovery and connection management
- Device mesh networking with multi-protocol support
- Tensor streaming for model distribution
- Fault-tolerant message delivery
"""

from .device_mesh import DeviceMesh
from .p2p_node import P2PNode, PeerCapabilities
from .tensor_streaming import TensorStreamer

__all__ = ["DeviceMesh", "P2PNode", "PeerCapabilities", "TensorStreamer"]

__version__ = "1.0.0"
