"""
Mesh Module - Intelligent Message Routing

Intelligent message routing with failover and mobile-aware protocols.
"""

# Import MeshNetwork from BitChat implementation
try:
    from ..bitchat.mesh_network import MeshNetwork
except ImportError:
    # Fallback stub for testing
    class MeshNetwork:
        def __init__(self):
            pass


__all__ = ["MeshNetwork"]
