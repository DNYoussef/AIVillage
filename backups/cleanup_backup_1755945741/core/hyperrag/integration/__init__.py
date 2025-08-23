"""
HyperRAG Integration Bridges

System integration for distributed and edge computing:
- EdgeDeviceBridge: Mobile/edge optimization
- P2PNetworkBridge: Peer-to-peer knowledge sharing
- FogComputeBridge: Distributed fog computing
"""

try:
    from .edge_device_bridge import EdgeDeviceRAGBridge
except ImportError:
    EdgeDeviceRAGBridge = None

try:
    from .p2p_network_bridge import P2PNetworkRAGBridge
except ImportError:
    P2PNetworkRAGBridge = None

try:
    from .fog_compute_bridge import FogComputeBridge
except ImportError:
    FogComputeBridge = None

__all__ = ["EdgeDeviceRAGBridge", "P2PNetworkRAGBridge", "FogComputeBridge"]
