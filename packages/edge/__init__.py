"""
Unified Edge Device Infrastructure for AIVillage

This package consolidates all edge device, mobile app, and fog computing
implementations into a coherent architecture supporting:

- Cross-platform mobile deployment (iOS/Android)
- Battery/thermal-aware resource management
- Distributed edge computing and fog architecture
- Device-specific AI model optimization
- Offline-first mesh networking
- Production deployment and monitoring

Architecture:
    core/           - Core edge management and orchestration
    mobile/         - Cross-platform mobile infrastructure
    deployment/     - Device deployment and lifecycle management
    fog_compute/    - Distributed fog computing architecture
    monitoring/     - Edge device monitoring and analytics
    bridges/        - Platform compatibility and migration support
"""

__version__ = "1.0.0"
__author__ = "AIVillage Team"

# Core exports for unified edge device management
from .core.device_registry import DeviceRegistry
from .core.edge_manager import EdgeManager
from .deployment.edge_deployer import EdgeDeployer
from .mobile.resource_manager import MobileResourceManager

__all__ = [
    "EdgeManager",
    "DeviceRegistry",
    "MobileResourceManager",
    "EdgeDeployer",
]
