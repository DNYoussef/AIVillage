"""
Device Registry - Central device management and lifecycle tracking

Provides device registration, discovery, and lifecycle management for the edge computing system.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DeviceRegistration:
    """Device registration record"""

    device_id: str
    device_name: str
    device_type: str
    registered_at: datetime
    last_seen: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class DeviceRegistry:
    """Central device registry for edge computing system"""

    def __init__(self):
        self.devices: dict[str, DeviceRegistration] = {}
        self.stats = {
            "total_registered": 0,
            "active_devices": 0,
            "inactive_devices": 0,
        }

        logger.info("Device Registry initialized")

    def register_device(
        self, device_id: str, device_name: str, device_type: str, metadata: dict[str, Any] | None = None
    ) -> DeviceRegistration:
        """Register a new device"""

        registration = DeviceRegistration(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            registered_at=datetime.now(UTC),
            last_seen=datetime.now(UTC),
            metadata=metadata or {},
            is_active=True,
        )

        self.devices[device_id] = registration
        self.stats["total_registered"] += 1
        self.stats["active_devices"] += 1

        logger.info(f"Registered device {device_name} ({device_type})")
        return registration

    def get_device(self, device_id: str) -> DeviceRegistration | None:
        """Get device registration"""
        return self.devices.get(device_id)

    def list_devices(self, active_only: bool = True) -> list[DeviceRegistration]:
        """List registered devices"""
        if active_only:
            return [d for d in self.devices.values() if d.is_active]
        return list(self.devices.values())

    def update_last_seen(self, device_id: str) -> bool:
        """Update device last seen timestamp"""
        if device_id in self.devices:
            self.devices[device_id].last_seen = datetime.now(UTC)
            return True
        return False

    def deactivate_device(self, device_id: str) -> bool:
        """Deactivate a device"""
        if device_id in self.devices:
            self.devices[device_id].is_active = False
            self.stats["active_devices"] -= 1
            self.stats["inactive_devices"] += 1
            return True
        return False
