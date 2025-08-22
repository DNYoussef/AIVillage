"""
Edge Device Integration Bridges

Provides compatibility and integration layers between edge devices and other AIVillage systems.
"""

from .legacy_compatibility import LegacyEdgeCompatibility
from .p2p_integration import EdgeP2PBridge

__all__ = [
    "EdgeP2PBridge",
    "LegacyEdgeCompatibility",
]
