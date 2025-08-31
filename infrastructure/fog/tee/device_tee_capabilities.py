"""
Device TEE Capabilities Detection and Management

Provides functionality to detect, report, and manage TEE capabilities
on fog computing devices including mobile devices, edge nodes, and servers.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import os
from pathlib import Path
import platform
from typing import Any

from .tee_types import TEECapability, TEEType

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of devices that can provide TEE capabilities"""

    MOBILE = "mobile"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    SERVER = "server"
    EDGE_NODE = "edge_node"
    EMBEDDED = "embedded"


class SecurityLevel(Enum):
    """Security levels for different TEE types"""

    SOFTWARE = "software"  # Software isolation only
    HARDWARE = "hardware"  # Hardware-based TEE
    CERTIFIED = "certified"  # Certified hardware TEE
    VALIDATED = "validated"  # Formally validated TEE


@dataclass
class DeviceTEEProfile:
    """Profile of TEE capabilities for a device"""

    device_id: str
    device_type: DeviceType
    platform: str  # OS platform
    architecture: str  # CPU architecture

    # TEE capabilities
    available_tee_types: list[TEECapability] = field(default_factory=list)
    primary_tee: TEECapability | None = None
    security_level: SecurityLevel = SecurityLevel.SOFTWARE

    # Hardware features
    cpu_features: list[str] = field(default_factory=list)
    secure_boot_enabled: bool = False
    tpm_available: bool = False
    hardware_rng: bool = False

    # Resource constraints
    max_tee_memory_mb: int = 0
    max_concurrent_enclaves: int = 0
    attestation_support: bool = False

    # Dynamic properties
    current_load: float = 0.0
    thermal_state: str = "normal"
    battery_level: int | None = None
    power_state: str = "plugged"

    # Metadata
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    detection_confidence: float = 1.0

    def get_best_tee_for_task(
        self, memory_mb: int, requires_attestation: bool = False, preferred_type: TEEType | None = None
    ) -> TEECapability | None:
        """Select best TEE capability for a specific task"""

        # Filter capabilities by requirements
        suitable_tees = []

        for tee_cap in self.available_tee_types:
            # Check memory requirement
            if tee_cap.max_memory_mb < memory_mb:
                continue

            # Check attestation requirement
            if requires_attestation and not tee_cap.supports_remote_attestation:
                continue

            # Check preferred type
            if preferred_type and tee_cap.tee_type != preferred_type:
                continue

            suitable_tees.append(tee_cap)

        if not suitable_tees:
            return None

        # Rank by preference: Hardware > Software, Attestation > No Attestation
        def tee_score(tee_cap: TEECapability) -> float:
            score = 0.0

            # Hardware TEE bonus
            if tee_cap.tee_type != TEEType.SOFTWARE_ISOLATION:
                score += 10.0

            # Attestation support bonus
            if tee_cap.supports_remote_attestation:
                score += 5.0

            # Memory capacity bonus
            score += min(5.0, tee_cap.max_memory_mb / 1024)  # Up to 5GB bonus

            # Enclave capacity bonus
            score += min(2.0, tee_cap.max_enclaves / 10)  # Up to 20 enclaves bonus

            return score

        # Sort by score and return best
        suitable_tees.sort(key=tee_score, reverse=True)
        return suitable_tees[0]

    def is_suitable_for_fog_computing(self) -> bool:
        """Check if device is suitable for fog computing"""

        # Minimum requirements for fog participation
        if not self.available_tee_types:
            return False

        if self.max_tee_memory_mb < 128:  # At least 128MB TEE memory
            return False

        if self.current_load > 0.9:  # Device too busy
            return False

        if self.thermal_state in ["hot", "critical"]:
            return False

        # Battery devices need sufficient charge
        if self.battery_level is not None and self.battery_level < 30:
            return False

        return True


class DeviceTEEDetector:
    """Detects TEE capabilities on different device types"""

    def __init__(self):
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        self.detection_cache: dict[str, DeviceTEEProfile] = {}

        logger.info(f"TEE detector initialized for {self.platform}/{self.architecture}")

    async def detect_device_capabilities(self, device_id: str) -> DeviceTEEProfile:
        """Detect TEE capabilities for a device"""

        # Check cache first
        if device_id in self.detection_cache:
            cached_profile = self.detection_cache[device_id]
            # Refresh if older than 1 hour
            if (datetime.now(UTC) - cached_profile.last_updated).seconds < 3600:
                return cached_profile

        logger.info(f"Detecting TEE capabilities for device {device_id}")

        # Determine device type
        device_type = self._determine_device_type()

        # Create profile
        profile = DeviceTEEProfile(
            device_id=device_id, device_type=device_type, platform=self.platform, architecture=self.architecture
        )

        # Detect CPU features
        profile.cpu_features = await self._detect_cpu_features()

        # Detect TEE capabilities
        profile.available_tee_types = await self._detect_tee_capabilities()

        # Determine primary TEE
        if profile.available_tee_types:
            profile.primary_tee = profile.available_tee_types[0]  # First is usually best

            # Set security level based on primary TEE
            if profile.primary_tee.tee_type == TEEType.SOFTWARE_ISOLATION:
                profile.security_level = SecurityLevel.SOFTWARE
            else:
                profile.security_level = SecurityLevel.HARDWARE

        # Detect hardware security features
        profile.secure_boot_enabled = await self._detect_secure_boot()
        profile.tpm_available = await self._detect_tpm()
        profile.hardware_rng = await self._detect_hardware_rng()

        # Calculate resource limits
        profile.max_tee_memory_mb = await self._calculate_max_tee_memory()
        profile.max_concurrent_enclaves = await self._calculate_max_enclaves()
        profile.attestation_support = any(tee.supports_remote_attestation for tee in profile.available_tee_types)

        # Get current system state
        profile.current_load = await self._get_system_load()
        profile.thermal_state = await self._get_thermal_state()
        profile.battery_level = await self._get_battery_level()
        profile.power_state = await self._get_power_state()

        # Cache result
        self.detection_cache[device_id] = profile

        logger.info(f"Detected {len(profile.available_tee_types)} TEE types for {device_id}")
        return profile

    def _determine_device_type(self) -> DeviceType:
        """Determine device type based on platform and hardware"""

        # Check environment variables for explicit type
        env_type = os.getenv("DEVICE_TYPE")
        if env_type:
            try:
                return DeviceType(env_type.lower())
            except ValueError as e:
                logger.warning(f"Invalid device type in environment variable '{env_type}': {e}")

        # Platform-based detection
        if self.platform in ["android", "ios"]:
            return DeviceType.MOBILE
        elif self.platform == "linux":
            # Check for common embedded/edge indicators
            if any(keyword in os.uname().nodename.lower() for keyword in ["pi", "edge", "nano", "jetson"]):
                return DeviceType.EDGE_NODE
            elif "server" in os.uname().nodename.lower():
                return DeviceType.SERVER
            else:
                return DeviceType.DESKTOP
        elif self.platform == "windows":
            # Simple heuristic based on processor info
            try:
                with open("/proc/cpuinfo") as f:
                    cpu_info = f.read().lower()
                    if "mobile" in cpu_info or "atom" in cpu_info:
                        return DeviceType.TABLET
            except Exception:
                logging.exception("Failed to read CPU info for device type detection")
            return DeviceType.DESKTOP
        elif self.platform == "darwin":
            # macOS - could be laptop or desktop
            return DeviceType.LAPTOP
        else:
            return DeviceType.DESKTOP

    async def _detect_cpu_features(self) -> list[str]:
        """Detect CPU security features"""
        features = []

        try:
            if self.platform == "linux":
                # Read CPU info
                with open("/proc/cpuinfo") as f:
                    cpu_info = f.read()

                # Check for common security features
                if "sev" in cpu_info:
                    features.append("amd_sev")
                if "tdx" in cpu_info:
                    features.append("intel_tdx")
                if "sgx" in cpu_info:
                    features.append("intel_sgx")
                if "aes" in cpu_info:
                    features.append("hardware_aes")
                if "rng" in cpu_info:
                    features.append("hardware_rng")

        except Exception as e:
            logger.debug(f"Error reading CPU features: {e}")

        return features

    async def _detect_tee_capabilities(self) -> list[TEECapability]:
        """Detect available TEE technologies"""
        capabilities = []

        # Always add software isolation as fallback
        software_cap = TEECapability(
            tee_type=TEEType.SOFTWARE_ISOLATION,
            available=True,
            version="1.0",
            max_memory_mb=2048,  # Will be adjusted based on system memory
            max_enclaves=8,
            supports_migration=True,
            supports_remote_attestation=True,
            supports_sealed_storage=False,
            secure_boot=False,
            memory_encryption=False,
            io_protection=True,
            debug_disabled=True,
        )
        capabilities.append(software_cap)

        # Try to detect hardware TEEs
        try:
            # AMD SEV-SNP detection
            if await self._check_amd_sev_snp():
                sev_cap = TEECapability(
                    tee_type=TEEType.AMD_SEV_SNP,
                    available=True,
                    version="1.0",
                    max_memory_mb=8192,
                    max_enclaves=16,
                    supports_migration=True,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=True,
                    debug_disabled=True,
                )
                capabilities.insert(0, sev_cap)  # Prefer hardware TEE

            # Intel TDX detection
            if await self._check_intel_tdx():
                tdx_cap = TEECapability(
                    tee_type=TEEType.INTEL_TDX,
                    available=True,
                    version="1.0",
                    max_memory_mb=16384,
                    max_enclaves=8,
                    supports_migration=False,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=True,
                    debug_disabled=True,
                )
                capabilities.insert(0, tdx_cap)  # TDX preferred over SEV-SNP

            # Intel SGX detection
            if await self._check_intel_sgx():
                sgx_cap = TEECapability(
                    tee_type=TEEType.INTEL_SGX,
                    available=True,
                    version="2.0",
                    max_memory_mb=256,  # SGX has limited memory
                    max_enclaves=64,
                    supports_migration=False,
                    supports_remote_attestation=True,
                    supports_sealed_storage=True,
                    secure_boot=True,
                    memory_encryption=True,
                    io_protection=False,
                    debug_disabled=True,
                )
                capabilities.append(sgx_cap)  # Lower priority due to memory limits

        except Exception as e:
            logger.debug(f"Hardware TEE detection failed: {e}")

        return capabilities

    async def _check_amd_sev_snp(self) -> bool:
        """Check for AMD SEV-SNP support"""
        try:
            # Check for SEV device
            if os.path.exists("/dev/sev"):
                # Check kernel support
                proc = await asyncio.create_subprocess_exec(
                    "dmesg", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()

                if b"SEV-SNP" in stdout:
                    return True

        except Exception:
            logging.exception("Failed to check AMD SEV-SNP support")

        return False

    async def _check_intel_tdx(self) -> bool:
        """Check for Intel TDX support"""
        try:
            # Check CPUID for TDX support
            proc = await asyncio.create_subprocess_exec(
                "cpuid", "-1", "-l", "0x21", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0 and b"TDX" in stdout:
                return True

        except Exception:
            logging.exception("Failed to check Intel TDX support")

        return False

    async def _check_intel_sgx(self) -> bool:
        """Check for Intel SGX support"""
        try:
            # Check for SGX device nodes
            sgx_devices = ["/dev/sgx_enclave", "/dev/sgx/enclave", "/dev/isgx"]
            if any(os.path.exists(dev) for dev in sgx_devices):
                return True

        except Exception:
            logging.exception("Failed to check Intel SGX support")

        return False

    async def _detect_secure_boot(self) -> bool:
        """Detect if secure boot is enabled"""
        try:
            if self.platform == "linux":
                # Check EFI secure boot status
                secure_boot_path = "/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c"
                if os.path.exists(secure_boot_path):
                    with open(secure_boot_path, "rb") as f:
                        data = f.read()
                        # Check if secure boot is enabled (last byte should be 1)
                        return len(data) > 4 and data[-1] == 1

        except Exception:
            logging.exception("Failed to detect secure boot status")

        return False

    async def _detect_tpm(self) -> bool:
        """Detect TPM availability"""
        try:
            # Check for TPM device
            tpm_devices = ["/dev/tpm0", "/dev/tpmrm0"]
            if any(os.path.exists(dev) for dev in tpm_devices):
                return True

        except Exception:
            logging.exception("Failed to detect TPM availability")

        return False

    async def _detect_hardware_rng(self) -> bool:
        """Detect hardware random number generator"""
        try:
            return os.path.exists("/dev/hwrng")
        except Exception:
            return False

    async def _calculate_max_tee_memory(self) -> int:
        """Calculate maximum TEE memory based on system memory"""
        try:
            if self.platform == "linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_kb = int(line.split()[1])
                            total_mb = total_kb // 1024

                            # TEE can use up to 25% of system memory
                            max_tee_mb = min(total_mb // 4, 8192)  # Cap at 8GB
                            return max_tee_mb

        except Exception:
            logging.exception("Failed to calculate TEE memory allocation from system memory info")

        # Default allocation
        return 1024  # 1GB default

    async def _calculate_max_enclaves(self) -> int:
        """Calculate maximum concurrent enclaves"""
        try:
            # Base on CPU cores and memory
            cpu_count = os.cpu_count() or 1
            max_tee_memory = await self._calculate_max_tee_memory()

            # Estimate enclaves based on resources
            memory_based = max_tee_memory // 256  # 256MB per enclave
            cpu_based = cpu_count * 2  # 2 enclaves per core

            return min(memory_based, cpu_based, 32)  # Cap at 32

        except Exception as e:
            logger.warning(f"Failed to estimate TEE enclave count: {e}")

        return 4  # Default

    async def _get_system_load(self) -> float:
        """Get current system load average"""
        try:
            if hasattr(os, "getloadavg"):
                return os.getloadavg()[0]  # 1-minute load average
        except Exception as e:
            logger.debug(f"Failed to get system load average: {e}")

        return 0.0

    async def _get_thermal_state(self) -> str:
        """Get thermal state of the device"""
        try:
            # Check thermal zone (Linux)
            thermal_zones = Path("/sys/class/thermal").glob("thermal_zone*")
            for zone in thermal_zones:
                temp_file = zone / "temp"
                if temp_file.exists():
                    with open(temp_file) as f:
                        temp_millic = int(f.read().strip())
                        temp_c = temp_millic / 1000

                        if temp_c > 80:
                            return "critical"
                        elif temp_c > 70:
                            return "hot"
                        elif temp_c > 60:
                            return "warm"
                        else:
                            return "normal"

        except Exception as e:
            logger.debug(f"Failed to read thermal state from sysfs: {e}")

        # Check environment variable for testing
        return os.getenv("THERMAL_STATE", "normal")

    async def _get_battery_level(self) -> int | None:
        """Get battery level percentage"""
        try:
            # Check power supply (Linux)
            power_supplies = Path("/sys/class/power_supply").glob("BAT*")
            for supply in power_supplies:
                capacity_file = supply / "capacity"
                if capacity_file.exists():
                    with open(capacity_file) as f:
                        return int(f.read().strip())

        except Exception as e:
            logger.debug(f"Failed to read battery level from sysfs: {e}")

        # Check environment variable for testing
        battery_env = os.getenv("BATTERY_LEVEL")
        if battery_env:
            try:
                return int(battery_env)
            except ValueError as e:
                logger.warning(f"Invalid battery level in environment variable '{battery_env}': {e}")

        return None  # No battery (desktop/server)

    async def _get_power_state(self) -> str:
        """Get power state (plugged, battery, charging)"""
        try:
            # Check AC adapter status
            power_supplies = Path("/sys/class/power_supply").glob("A{C,DP}*")
            for supply in power_supplies:
                online_file = supply / "online"
                if online_file.exists():
                    with open(online_file) as f:
                        if f.read().strip() == "1":
                            return "plugged"

            # Check battery status
            battery_supplies = Path("/sys/class/power_supply").glob("BAT*")
            for supply in battery_supplies:
                status_file = supply / "status"
                if status_file.exists():
                    with open(status_file) as f:
                        status = f.read().strip()
                        if status == "Charging":
                            return "charging"
                        elif status == "Discharging":
                            return "battery"

        except Exception as e:
            logger.debug(f"Failed to read power status from sysfs: {e}")

        return "plugged"  # Default assumption


class DeviceTEERegistry:
    """Registry for managing device TEE capabilities"""

    def __init__(self):
        self.devices: dict[str, DeviceTEEProfile] = {}
        self.detector = DeviceTEEDetector()

    async def register_device(self, device_id: str) -> DeviceTEEProfile:
        """Register a device and detect its TEE capabilities"""
        profile = await self.detector.detect_device_capabilities(device_id)
        self.devices[device_id] = profile

        logger.info(f"Registered device {device_id} with {len(profile.available_tee_types)} TEE types")
        return profile

    async def update_device_status(self, device_id: str) -> DeviceTEEProfile | None:
        """Update dynamic status of a registered device"""
        if device_id not in self.devices:
            logger.warning(f"Device {device_id} not registered")
            return None

        profile = self.devices[device_id]

        # Update dynamic properties
        profile.current_load = await self.detector._get_system_load()
        profile.thermal_state = await self.detector._get_thermal_state()
        profile.battery_level = await self.detector._get_battery_level()
        profile.power_state = await self.detector._get_power_state()
        profile.last_updated = datetime.now(UTC)

        return profile

    def get_device(self, device_id: str) -> DeviceTEEProfile | None:
        """Get device profile"""
        return self.devices.get(device_id)

    def list_devices_with_tee(self, tee_type: TEEType | None = None, min_memory_mb: int = 0) -> list[DeviceTEEProfile]:
        """List devices with specific TEE capabilities"""
        suitable_devices = []

        for profile in self.devices.values():
            if not profile.is_suitable_for_fog_computing():
                continue

            if tee_type:
                has_tee_type = any(cap.tee_type == tee_type for cap in profile.available_tee_types)
                if not has_tee_type:
                    continue

            if profile.max_tee_memory_mb < min_memory_mb:
                continue

            suitable_devices.append(profile)

        return suitable_devices

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        total_devices = len(self.devices)
        tee_capable = sum(1 for p in self.devices.values() if p.available_tee_types)
        hardware_tee = sum(
            1
            for p in self.devices.values()
            if any(cap.tee_type != TEEType.SOFTWARE_ISOLATION for cap in p.available_tee_types)
        )

        # TEE type breakdown
        tee_type_counts = {}
        for tee_type in TEEType:
            count = sum(
                1 for p in self.devices.values() if any(cap.tee_type == tee_type for cap in p.available_tee_types)
            )
            tee_type_counts[tee_type.value] = count

        return {
            "total_devices": total_devices,
            "tee_capable_devices": tee_capable,
            "hardware_tee_devices": hardware_tee,
            "tee_type_breakdown": tee_type_counts,
            "suitable_for_fog": len(self.list_devices_with_tee()),
        }


# Global registry instance
device_tee_registry = DeviceTEERegistry()


# Convenience functions
async def detect_local_device_tee() -> DeviceTEEProfile:
    """Detect TEE capabilities of the local device"""
    import socket

    local_device_id = f"local_{socket.gethostname()}"
    return await device_tee_registry.register_device(local_device_id)


async def get_best_tee_device(
    memory_mb: int, requires_attestation: bool = False, preferred_type: TEEType | None = None
) -> DeviceTEEProfile | None:
    """Find the best device for a TEE task"""
    suitable_devices = device_tee_registry.list_devices_with_tee(tee_type=preferred_type, min_memory_mb=memory_mb)

    if not suitable_devices:
        return None

    # Score devices based on suitability
    def device_score(profile: DeviceTEEProfile) -> float:
        score = 0.0

        # Get best TEE for this task
        best_tee = profile.get_best_tee_for_task(memory_mb, requires_attestation, preferred_type)
        if not best_tee:
            return 0.0

        # Hardware TEE bonus
        if best_tee.tee_type != TEEType.SOFTWARE_ISOLATION:
            score += 10.0

        # Power state bonus (plugged > charging > battery)
        if profile.power_state == "plugged":
            score += 5.0
        elif profile.power_state == "charging":
            score += 3.0

        # Thermal state penalty
        thermal_penalties = {"normal": 0, "warm": -1, "hot": -3, "critical": -10}
        score += thermal_penalties.get(profile.thermal_state, 0)

        # Load penalty
        score -= profile.current_load * 2

        # Memory capacity bonus
        score += min(3.0, profile.max_tee_memory_mb / 1024)

        return score

    # Sort by score and return best
    suitable_devices.sort(key=device_score, reverse=True)
    return suitable_devices[0]
