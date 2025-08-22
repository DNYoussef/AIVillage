"""
Core Edge Device Management

Provides the foundational components for edge device orchestration.
"""

from .device_registry import DeviceRegistry
from .edge_manager import DeviceCapabilities, DeviceType, EdgeManager, EdgeState

__all__ = [
    "EdgeManager",
    "DeviceType",
    "EdgeState",
    "DeviceCapabilities",
    "DeviceRegistry",
]
