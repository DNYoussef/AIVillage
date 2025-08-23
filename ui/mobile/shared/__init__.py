"""
Mobile Edge Device Infrastructure

Provides comprehensive mobile device support including:
- Battery/thermal-aware resource management
- Cross-platform mobile app frameworks (iOS/Android)
- BitChat mesh networking integration
- Mobile-optimized AI deployment
- Device profiling and optimization
"""

from .resource_manager import (
    ChunkingConfig,
    MobileDeviceProfile,
    MobileResourceManager,
    PowerMode,
    ResourceOptimization,
    ResourcePolicy,
    TransportPreference,
)

__all__ = [
    "MobileResourceManager",
    "MobileDeviceProfile",
    "ResourceOptimization",
    "ResourcePolicy",
    "ChunkingConfig",
    "PowerMode",
    "TransportPreference",
]
