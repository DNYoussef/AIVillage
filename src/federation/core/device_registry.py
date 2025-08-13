"""Device Registry - Core Federation Framework

Manages device discovery, role assignment, and capability registration
for the AIVillage federated network. Supports 5 device types: Beacon,
Worker, Relay, Storage, and Edge nodes.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import time
from typing import Any
import uuid

# Cryptography imports
try:
    import nacl.encoding
    import nacl.secret
    import nacl.signing
    import nacl.utils

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("PyNaCl not available - using fallback crypto")

logger = logging.getLogger(__name__)


class DeviceRole(Enum):
    """Device roles in the federation"""

    BEACON = "beacon"  # Always-on coordinators (99.9% uptime)
    WORKER = "worker"  # Compute contributors
    RELAY = "relay"  # Network infrastructure
    STORAGE = "storage"  # Distributed persistence
    EDGE = "edge"  # Local processing nodes


class DeviceCapability(Enum):
    """Device capabilities for role assignment"""

    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    CELLULAR = "cellular"
    ETHERNET = "ethernet"
    HIGH_COMPUTE = "high_compute"  # 4+ CPU cores
    LOW_POWER = "low_power"  # Battery device
    ALWAYS_ON = "always_on"  # Mains powered
    STORAGE_LARGE = "storage_large"  # 100GB+ available
    BANDWIDTH_HIGH = "bandwidth_high"  # 100Mbps+


@dataclass
class DeviceIdentity:
    """Cryptographic identity for federation devices"""

    device_id: str
    public_key: bytes = field(default=b"")
    signing_key_public: bytes = field(default=b"")
    validator_key: bytes | None = None  # For beacon nodes
    created_at: float = field(default_factory=time.time)
    reputation_score: float = 0.5  # Start at neutral
    last_seen: float = field(default_factory=time.time)

    def __post_init__(self):
        """Generate keys if not provided"""
        if CRYPTO_AVAILABLE and not self.public_key:
            # Generate encryption keypair
            private_key = nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
            self.public_key = private_key  # Simplified for demo

            # Generate signing keypair
            signing_key = nacl.signing.SigningKey.generate()
            self.signing_key_public = bytes(signing_key.verify_key)

    def sign_message(self, message: bytes, private_signing_key: bytes) -> bytes:
        """Sign a message with the device's signing key"""
        if not CRYPTO_AVAILABLE:
            return b"mock_signature"

        signing_key = nacl.signing.SigningKey(private_signing_key)
        signed = signing_key.sign(message)
        return signed.signature

    def verify_signature(self, message: bytes, signature: bytes) -> bool:
        """Verify a message signature"""
        if not CRYPTO_AVAILABLE:
            return True  # Mock verification

        try:
            verify_key = nacl.signing.VerifyKey(self.signing_key_public)
            verify_key.verify(message, signature)
            return True
        except Exception:
            return False


@dataclass
class DeviceProfile:
    """Complete device profile for federation"""

    identity: DeviceIdentity
    role: DeviceRole
    capabilities: set[DeviceCapability] = field(default_factory=set)
    protocols: set[str] = field(default_factory=set)  # bitchat, betanet, tor, i2p
    resources: dict[str, Any] = field(default_factory=dict)
    region: str | None = None
    beacon_affinity: str | None = None  # Preferred beacon node

    # Resource metrics
    cpu_cores: int = 1
    memory_gb: float = 1.0
    storage_gb: float = 10.0
    bandwidth_mbps: float = 1.0
    battery_percent: int | None = None
    uptime_hours: float = 0.0

    # Federation metrics
    messages_relayed: int = 0
    storage_contributed_gb: float = 0.0
    compute_contributed_hours: float = 0.0
    credits_earned: int = 0

    def calculate_device_score(self) -> float:
        """Calculate overall device contribution score"""
        base_score = self.identity.reputation_score

        # Uptime bonus (up to 0.2)
        uptime_bonus = min(0.2, self.uptime_hours / (24 * 30))  # 30 days for max

        # Resource contribution bonus (up to 0.3)
        resource_score = (self.cpu_cores / 8) * 0.1 + (self.memory_gb / 16) * 0.1 + (self.bandwidth_mbps / 100) * 0.1
        resource_bonus = min(0.3, resource_score)

        return min(1.0, base_score + uptime_bonus + resource_bonus)

    def is_suitable_for_role(self, role: DeviceRole) -> bool:
        """Check if device meets requirements for specific role"""
        if role == DeviceRole.BEACON:
            return (
                DeviceCapability.ALWAYS_ON in self.capabilities
                and self.uptime_hours > 24 * 7
                and self.calculate_device_score() > 0.8  # Minimum 1 week uptime
            )

        if role == DeviceRole.WORKER:
            return DeviceCapability.HIGH_COMPUTE in self.capabilities and self.cpu_cores >= 2

        if role == DeviceRole.RELAY:
            return (
                DeviceCapability.BANDWIDTH_HIGH in self.capabilities
                and len(self.protocols) >= 2  # Support multiple protocols
            )

        if role == DeviceRole.STORAGE:
            return DeviceCapability.STORAGE_LARGE in self.capabilities and self.storage_gb >= 100

        if role == DeviceRole.EDGE:
            return True  # Any device can be an edge node

        return False


class DeviceRegistry:
    """Central registry for federation device management"""

    def __init__(self, local_device_id: str = None):
        self.local_device_id = local_device_id or f"device_{uuid.uuid4().hex[:12]}"
        self.devices: dict[str, DeviceProfile] = {}
        self.local_profile: DeviceProfile | None = None

        # Beacon tracking
        self.known_beacons: dict[str, float] = {}  # device_id -> last_heartbeat
        self.active_beacon: str | None = None

        # Regional clustering
        self.region_devices: dict[str, set[str]] = {}

        # Federation statistics
        self.total_devices = 0
        self.total_cpu_cores = 0
        self.total_storage_gb = 0.0
        self.total_bandwidth_mbps = 0.0

        logger.info(f"DeviceRegistry initialized for device: {self.local_device_id}")

    async def initialize_local_device(
        self, capabilities: set[DeviceCapability], region: str = "unknown"
    ) -> DeviceProfile:
        """Initialize and register the local device"""
        # Create device identity
        identity = DeviceIdentity(device_id=self.local_device_id)

        # Detect hardware capabilities
        detected_capabilities = await self._detect_capabilities()
        all_capabilities = capabilities | detected_capabilities

        # Detect supported protocols
        protocols = await self._detect_protocols()

        # Gather resource information
        resources = await self._gather_resources()

        # Determine optimal role
        optimal_role = self._determine_optimal_role(all_capabilities, resources)

        # Create device profile
        self.local_profile = DeviceProfile(
            identity=identity,
            role=optimal_role,
            capabilities=all_capabilities,
            protocols=protocols,
            resources=resources,
            region=region,
            cpu_cores=resources.get("cpu_cores", 1),
            memory_gb=resources.get("memory_gb", 1.0),
            storage_gb=resources.get("storage_gb", 10.0),
            bandwidth_mbps=resources.get("bandwidth_mbps", 1.0),
            battery_percent=resources.get("battery_percent"),
        )

        # Register locally
        self.devices[self.local_device_id] = self.local_profile

        logger.info(
            f"Local device initialized: role={optimal_role.value}, "
            f"capabilities={len(all_capabilities)}, protocols={len(protocols)}"
        )

        return self.local_profile

    async def register_device(self, profile: DeviceProfile) -> bool:
        """Register a remote device in the federation"""
        device_id = profile.identity.device_id

        # Verify device signature (simplified)
        if not self._verify_device_profile(profile):
            logger.warning(f"Failed to verify device profile: {device_id}")
            return False

        # Update registry
        self.devices[device_id] = profile

        # Update beacon tracking
        if profile.role == DeviceRole.BEACON:
            self.known_beacons[device_id] = time.time()

        # Update regional clustering
        if profile.region:
            if profile.region not in self.region_devices:
                self.region_devices[profile.region] = set()
            self.region_devices[profile.region].add(device_id)

        # Update federation statistics
        self._update_statistics()

        logger.info(f"Registered device: {device_id} ({profile.role.value})")
        return True

    def get_devices_by_role(self, role: DeviceRole) -> list[DeviceProfile]:
        """Get all devices with specific role"""
        return [profile for profile in self.devices.values() if profile.role == role]

    def get_devices_by_region(self, region: str) -> list[DeviceProfile]:
        """Get all devices in specific region"""
        return [profile for profile in self.devices.values() if profile.region == region]

    def get_devices_by_capability(self, capability: DeviceCapability) -> list[DeviceProfile]:
        """Get all devices with specific capability"""
        return [profile for profile in self.devices.values() if capability in profile.capabilities]

    def find_best_beacon(self, region: str = None) -> DeviceProfile | None:
        """Find the best beacon node for coordination"""
        beacons = self.get_devices_by_role(DeviceRole.BEACON)

        if region:
            beacons = [b for b in beacons if b.region == region]

        if not beacons:
            return None

        # Sort by device score and uptime
        beacons.sort(key=lambda x: (x.calculate_device_score(), x.uptime_hours), reverse=True)

        return beacons[0]

    def assign_device_to_beacon(self, device_id: str, beacon_id: str) -> bool:
        """Assign a device to a specific beacon for coordination"""
        if device_id not in self.devices or beacon_id not in self.devices:
            return False

        device = self.devices[device_id]
        beacon = self.devices[beacon_id]

        if beacon.role != DeviceRole.BEACON:
            return False

        device.beacon_affinity = beacon_id
        logger.info(f"Assigned device {device_id} to beacon {beacon_id}")
        return True

    def get_federation_status(self) -> dict[str, Any]:
        """Get comprehensive federation status"""
        role_counts = {}
        for role in DeviceRole:
            role_counts[role.value] = len(self.get_devices_by_role(role))

        return {
            "total_devices": len(self.devices),
            "role_distribution": role_counts,
            "regional_distribution": {region: len(devices) for region, devices in self.region_devices.items()},
            "active_beacons": len([b for b, t in self.known_beacons.items() if time.time() - t < 300]),  # 5 minutes
            "total_resources": {
                "cpu_cores": self.total_cpu_cores,
                "storage_gb": self.total_storage_gb,
                "bandwidth_mbps": self.total_bandwidth_mbps,
            },
            "local_device": {
                "id": self.local_device_id,
                "role": self.local_profile.role.value if self.local_profile else None,
                "score": self.local_profile.calculate_device_score() if self.local_profile else 0,
            },
        }

    async def _detect_capabilities(self) -> set[DeviceCapability]:
        """Auto-detect device capabilities"""
        capabilities = set()

        try:
            # Check for Bluetooth
            try:
                import bluetooth

                capabilities.add(DeviceCapability.BLUETOOTH)
            except ImportError:
                pass

            # Check CPU count
            cpu_count = os.cpu_count() or 1
            if cpu_count >= 4:
                capabilities.add(DeviceCapability.HIGH_COMPUTE)

            # Check power source (simplified)
            try:
                import psutil

                battery = psutil.sensors_battery()
                if battery:
                    capabilities.add(DeviceCapability.LOW_POWER)
                else:
                    capabilities.add(DeviceCapability.ALWAYS_ON)
            except Exception:
                # Assume battery device if can't detect
                capabilities.add(DeviceCapability.LOW_POWER)

            # Network capabilities (simplified detection)
            capabilities.add(DeviceCapability.WIFI)  # Most devices have WiFi

        except Exception as e:
            logger.warning(f"Error detecting capabilities: {e}")

        return capabilities

    async def _detect_protocols(self) -> set[str]:
        """Detect which protocols are available"""
        protocols = set()

        # Check BitChat (Bluetooth)
        try:
            import bluetooth

            protocols.add("bitchat")
        except ImportError:
            pass

        # Check Betanet (always available in software)
        protocols.add("betanet")

        # Check Tor
        try:
            import stem

            protocols.add("tor")
        except ImportError:
            pass

        # Check I2P (SAM API)
        try:
            import socket

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", 7656))
            if result == 0:
                protocols.add("i2p")
            sock.close()
        except Exception:
            pass

        return protocols

    async def _gather_resources(self) -> dict[str, Any]:
        """Gather device resource information"""
        resources = {}

        try:
            # CPU information
            resources["cpu_cores"] = os.cpu_count() or 1

            # Memory information
            try:
                import psutil

                memory = psutil.virtual_memory()
                resources["memory_gb"] = memory.total / (1024**3)

                # Disk space
                disk = psutil.disk_usage("/")
                resources["storage_gb"] = disk.free / (1024**3)

                # Battery
                battery = psutil.sensors_battery()
                if battery:
                    resources["battery_percent"] = int(battery.percent)

            except ImportError:
                resources["memory_gb"] = 2.0  # Default assumption
                resources["storage_gb"] = 20.0

            # Simplified bandwidth detection
            resources["bandwidth_mbps"] = 10.0  # Default assumption

        except Exception as e:
            logger.warning(f"Error gathering resources: {e}")
            # Safe defaults
            resources.update(
                {
                    "cpu_cores": 1,
                    "memory_gb": 1.0,
                    "storage_gb": 10.0,
                    "bandwidth_mbps": 1.0,
                }
            )

        return resources

    def _determine_optimal_role(self, capabilities: set[DeviceCapability], resources: dict[str, Any]) -> DeviceRole:
        """Determine optimal role for device based on capabilities"""
        # Check beacon suitability (highest priority)
        if (
            DeviceCapability.ALWAYS_ON in capabilities
            and resources.get("memory_gb", 0) >= 4
            and resources.get("bandwidth_mbps", 0) >= 10
        ):
            return DeviceRole.BEACON

        # Check storage node suitability
        if DeviceCapability.STORAGE_LARGE in capabilities and resources.get("storage_gb", 0) >= 100:
            return DeviceRole.STORAGE

        # Check worker node suitability
        if DeviceCapability.HIGH_COMPUTE in capabilities and resources.get("cpu_cores", 1) >= 2:
            return DeviceRole.WORKER

        # Check relay node suitability
        if DeviceCapability.BANDWIDTH_HIGH in capabilities and resources.get("bandwidth_mbps", 0) >= 50:
            return DeviceRole.RELAY

        # Default to edge node
        return DeviceRole.EDGE

    def _verify_device_profile(self, profile: DeviceProfile) -> bool:
        """Verify device profile authenticity (simplified)"""
        # In production, would verify cryptographic signatures
        # and check against reputation blacklists
        return True

    def _update_statistics(self):
        """Update federation-wide statistics"""
        self.total_devices = len(self.devices)
        self.total_cpu_cores = sum(d.cpu_cores for d in self.devices.values())
        self.total_storage_gb = sum(d.storage_gb for d in self.devices.values())
        self.total_bandwidth_mbps = sum(d.bandwidth_mbps for d in self.devices.values())

    async def cleanup_stale_devices(self, max_age_hours: int = 24):
        """Remove devices that haven't been seen recently"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        stale_devices = []
        for device_id, profile in self.devices.items():
            if current_time - profile.identity.last_seen > max_age_seconds:
                stale_devices.append(device_id)

        for device_id in stale_devices:
            del self.devices[device_id]
            # Clean up beacon tracking
            self.known_beacons.pop(device_id, None)
            # Clean up regional clustering
            for region_devices in self.region_devices.values():
                region_devices.discard(device_id)

        if stale_devices:
            logger.info(f"Cleaned up {len(stale_devices)} stale devices")
            self._update_statistics()
