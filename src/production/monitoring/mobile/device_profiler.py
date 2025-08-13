"""Enhanced Device Profiler - Consolidated Implementation.

This is the canonical device profiler implementation for AIVillage, consolidating
features from multiple implementations:
- Comprehensive resource monitoring from core version
- Mobile platform-specific features (Android, iOS)
- Real-time profiling capabilities
- Evolution suitability scoring
- Cross-platform compatibility

Replaces:
- src/core/resources/device_profiler.py (comprehensive features merged)
- Previous mobile-specific implementation (platform features retained)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import platform
import queue
import threading
import time
from typing import Any
import uuid

# System monitoring
import psutil

# Platform-specific imports
ANDROID_AVAILABLE = False
MACOS_AVAILABLE = False

# Default fallbacks for macOS-specific modules
NSBundle = None  # type: ignore[assignment]
NSProcessInfo = None  # type: ignore[assignment]
objc = None  # type: ignore[assignment]

if platform.system() == "Android":
    try:
        from jnius import autoclass

        Build = autoclass("android.os.Build")
        BatteryManager = autoclass("android.os.BatteryManager")
        ActivityManager = autoclass("android.app.ActivityManager")
        ThermalManager = autoclass("android.os.ThermalManager")
        ANDROID_AVAILABLE = True
    except ImportError:
        logger.debug("Android platform modules not available")
elif platform.system() == "Darwin":  # iOS/macOS
    try:
        from Foundation import NSBundle, NSProcessInfo  # type: ignore
        import objc  # type: ignore

        MACOS_AVAILABLE = True
    except ImportError:
        # Fallback when Foundation isn't available (e.g., non-macOS platforms)
        MACOS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Device type classification for resource management."""

    PHONE = "phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    SERVER = "server"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"


class PowerState(Enum):
    """Device power state for evolution scheduling."""

    PLUGGED_IN = "plugged_in"
    BATTERY_HIGH = "battery_high"  # > 80%
    BATTERY_MEDIUM = "battery_medium"  # 20-80%
    BATTERY_LOW = "battery_low"  # 5-20%
    BATTERY_CRITICAL = "battery_critical"  # < 5%
    UNKNOWN = "unknown"


class ThermalState(Enum):
    """Device thermal state for performance management."""

    NORMAL = "normal"  # < 60°C
    WARM = "warm"  # 60-75°C
    HOT = "hot"  # 75-85°C
    CRITICAL = "critical"  # > 85°C
    THROTTLING = "throttling"  # CPU throttling detected
    UNKNOWN = "unknown"


@dataclass
class ResourceSnapshot:
    """Real-time resource usage snapshot."""

    timestamp: float

    # Memory (bytes)
    memory_total: int
    memory_available: int
    memory_used: int
    memory_percent: float

    # CPU
    cpu_percent: float
    cpu_cores: int
    cpu_freq_current: float | None = None
    cpu_freq_max: float | None = None
    cpu_temp: float | None = None

    # Storage (bytes)
    storage_total: int = 0
    storage_used: int = 0
    storage_free: int = 0
    storage_percent: float = 0.0

    # Power
    battery_percent: float | None = None
    power_plugged: bool | None = None
    power_state: PowerState = PowerState.UNKNOWN

    # Thermal
    thermal_state: ThermalState = ThermalState.UNKNOWN

    # Network
    network_sent: int = 0
    network_received: int = 0
    network_connections: int = 0

    # Process metrics
    process_count: int = 0

    # GPU (if available)
    gpu_memory_used: int | None = None
    gpu_memory_total: int | None = None
    gpu_utilization: float | None = None

    @property
    def memory_usage_gb(self) -> float:
        """Memory usage in GB."""
        return self.memory_used / (1024**3)

    @property
    def memory_available_gb(self) -> float:
        """Available memory in GB."""
        return self.memory_available / (1024**3)

    @property
    def storage_used_gb(self) -> float:
        """Storage used in GB."""
        return self.storage_used / (1024**3)

    @property
    def is_resource_constrained(self) -> bool:
        """Check if device is resource constrained."""
        return (
            self.memory_percent > 85
            or self.cpu_percent > 90
            or self.storage_percent > 90
            or self.power_state in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL]
            or self.thermal_state
            in [ThermalState.HOT, ThermalState.CRITICAL, ThermalState.THROTTLING]
        )

    @property
    def evolution_suitability_score(self) -> float:
        """Calculate suitability for evolution tasks (0-1, higher = better)."""
        score = 1.0

        # Memory penalty
        if self.memory_percent > 80:
            score -= (self.memory_percent - 80) / 20 * 0.3

        # CPU penalty
        if self.cpu_percent > 70:
            score -= (self.cpu_percent - 70) / 30 * 0.2

        # Power penalty
        if self.battery_percent is not None and not self.power_plugged:
            if self.battery_percent < 30:
                score -= (30 - self.battery_percent) / 30 * 0.3

        # Thermal penalty
        thermal_penalties = {
            ThermalState.WARM: 0.1,
            ThermalState.HOT: 0.3,
            ThermalState.CRITICAL: 0.7,
            ThermalState.THROTTLING: 0.5,
        }
        score -= thermal_penalties.get(self.thermal_state, 0.0)

        # Storage penalty
        if self.storage_percent > 90:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-1, higher = better)."""
        # Base score from available resources
        memory_score = max(0.0, 1.0 - self.memory_percent / 100)
        cpu_score = max(0.0, 1.0 - self.cpu_percent / 100)
        storage_score = max(0.0, 1.0 - self.storage_percent / 100)

        # Weight the scores
        base_score = memory_score * 0.4 + cpu_score * 0.3 + storage_score * 0.3

        # Apply power and thermal adjustments
        if self.power_state == PowerState.BATTERY_CRITICAL:
            base_score *= 0.3
        elif self.power_state == PowerState.BATTERY_LOW:
            base_score *= 0.6
        elif self.power_state == PowerState.BATTERY_MEDIUM:
            base_score *= 0.8

        if self.thermal_state in [ThermalState.HOT, ThermalState.CRITICAL]:
            base_score *= 0.5
        elif self.thermal_state == ThermalState.WARM:
            base_score *= 0.8

        return max(0.0, min(1.0, base_score))


@dataclass
class DeviceProfile:
    """Comprehensive device profile with real-time monitoring."""

    # Device identification
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    device_name: str = ""
    device_type: DeviceType = DeviceType.UNKNOWN
    platform: str = ""
    architecture: str = ""

    # Hardware specs
    cpu_cores: int = 1
    cpu_brand: str = ""
    memory_total_gb: float = 0.0
    storage_total_gb: float = 0.0

    # Current resource state
    current_snapshot: ResourceSnapshot | None = None

    # Historical tracking
    snapshot_history: list[ResourceSnapshot] = field(default_factory=list)
    max_history_size: int = 100

    # Capabilities
    supports_gpu: bool = False
    supports_neural_engine: bool = False
    supports_background_processing: bool = True

    # Mobile-specific
    is_mobile: bool = False
    battery_optimization_enabled: bool = False

    # Threading for real-time monitoring
    _monitoring_thread: threading.Thread | None = field(default=None, init=False)
    _monitoring_active: bool = field(default=False, init=False)
    _snapshot_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _callbacks: list[Callable[[ResourceSnapshot], None]] = field(
        default_factory=list, init=False
    )

    def __post_init__(self):
        """Initialize device profile after creation."""
        if not self.device_name:
            self.device_name = platform.node() or "Unknown Device"
        if not self.platform:
            self.platform = platform.system()
        if not self.architecture:
            self.architecture = platform.machine()

        # Detect device type and mobile status
        self._detect_device_type()
        self._initialize_capabilities()

    def _detect_device_type(self) -> None:
        """Detect device type based on platform and hardware."""
        system = platform.system().lower()

        if ANDROID_AVAILABLE:
            self.device_type = DeviceType.PHONE  # Assume phone for Android
            self.is_mobile = True
        elif system == "ios":
            self.device_type = DeviceType.PHONE
            self.is_mobile = True
        elif system == "darwin":
            # Could be iPhone, iPad, or Mac
            try:
                if MACOS_AVAILABLE and NSProcessInfo:
                    # Check if running on iOS via Catalyst or native
                    self.device_type = DeviceType.LAPTOP  # Default to laptop for macOS
                else:
                    self.device_type = DeviceType.LAPTOP
            except:
                self.device_type = DeviceType.LAPTOP
        elif system in ["windows", "linux"]:
            # Use memory as a rough guide
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            if total_memory_gb < 4:
                self.device_type = DeviceType.EMBEDDED
            elif total_memory_gb < 8:
                self.device_type = DeviceType.LAPTOP
            elif total_memory_gb < 32:
                self.device_type = DeviceType.DESKTOP
            else:
                self.device_type = DeviceType.SERVER
        else:
            self.device_type = DeviceType.UNKNOWN

    def _initialize_capabilities(self) -> None:
        """Initialize device capabilities based on platform."""
        # Get basic hardware info
        self.cpu_cores = psutil.cpu_count() or 1
        self.memory_total_gb = psutil.virtual_memory().total / (1024**3)

        # Storage info
        try:
            disk_usage = psutil.disk_usage("/")
            self.storage_total_gb = disk_usage.total / (1024**3)
        except:
            self.storage_total_gb = 0.0

        # Platform-specific capabilities
        if ANDROID_AVAILABLE:
            self.supports_background_processing = True
            self.battery_optimization_enabled = True
        elif platform.system() == "Darwin":
            # Check for Neural Engine on Apple Silicon
            if "arm" in self.architecture.lower():
                self.supports_neural_engine = True

        # GPU detection (basic)
        try:
            # This is a basic check - would need platform-specific GPU detection for accuracy
            if platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "path", "win32_VideoController", "get", "name"],
                    capture_output=True,
                    text=True,
                )
                if "nvidia" in result.stdout.lower() or "amd" in result.stdout.lower():
                    self.supports_gpu = True
        except:
            pass

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start real-time resource monitoring."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Started resource monitoring for device {self.device_id}")

    def stop_monitoring(self) -> None:
        """Stop real-time resource monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=2.0)
        logger.info(f"Stopped resource monitoring for device {self.device_id}")

    def _monitoring_loop(self, interval: float) -> None:
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                snapshot = self._capture_snapshot()
                self.update_profile(snapshot)

                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.exception(f"Error in monitoring callback: {e}")

                time.sleep(interval)
            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource snapshot."""
        current_time = time.time()

        # Memory info
        memory = psutil.virtual_memory()

        # CPU info
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()

        # Storage info
        try:
            disk = psutil.disk_usage("/")
            storage_total = disk.total
            storage_used = disk.used
            storage_free = disk.free
            storage_percent = (storage_used / storage_total) * 100
        except:
            storage_total = storage_used = storage_free = 0
            storage_percent = 0.0

        # Battery info
        battery = psutil.sensors_battery()
        battery_percent = battery.percent if battery else None
        power_plugged = battery.power_plugged if battery else None

        # Power state
        power_state = self._determine_power_state(battery_percent, power_plugged)

        # Thermal info
        thermal_state = self._determine_thermal_state()

        # Network info
        net_io = psutil.net_io_counters()
        network_sent = net_io.bytes_sent
        network_received = net_io.bytes_recv
        network_connections = len(psutil.net_connections())

        # Process count
        process_count = len(psutil.pids())

        # CPU temperature (if available)
        cpu_temp = self._get_cpu_temperature()

        return ResourceSnapshot(
            timestamp=current_time,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_used=memory.used,
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
            cpu_cores=self.cpu_cores,
            cpu_freq_current=cpu_freq.current if cpu_freq else None,
            cpu_freq_max=cpu_freq.max if cpu_freq else None,
            cpu_temp=cpu_temp,
            storage_total=storage_total,
            storage_used=storage_used,
            storage_free=storage_free,
            storage_percent=storage_percent,
            battery_percent=battery_percent,
            power_plugged=power_plugged,
            power_state=power_state,
            thermal_state=thermal_state,
            network_sent=network_sent,
            network_received=network_received,
            network_connections=network_connections,
            process_count=process_count,
        )

    def _determine_power_state(
        self, battery_percent: float | None, power_plugged: bool | None
    ) -> PowerState:
        """Determine power state from battery info."""
        if power_plugged:
            return PowerState.PLUGGED_IN
        if battery_percent is None:
            return PowerState.UNKNOWN
        if battery_percent > 80:
            return PowerState.BATTERY_HIGH
        elif battery_percent > 20:
            return PowerState.BATTERY_MEDIUM
        elif battery_percent > 5:
            return PowerState.BATTERY_LOW
        else:
            return PowerState.BATTERY_CRITICAL

    def _determine_thermal_state(self) -> ThermalState:
        """Determine thermal state from temperature sensors."""
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return ThermalState.UNKNOWN

            # Get highest temperature
            max_temp = 0.0
            for _name, entries in temps.items():
                for entry in entries:
                    if entry.current and entry.current > max_temp:
                        max_temp = entry.current

            if max_temp < 60:
                return ThermalState.NORMAL
            elif max_temp < 75:
                return ThermalState.WARM
            elif max_temp < 85:
                return ThermalState.HOT
            else:
                return ThermalState.CRITICAL

        except:
            return ThermalState.UNKNOWN

    def _get_cpu_temperature(self) -> float | None:
        """Get CPU temperature if available."""
        try:
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps:
                # Intel CPUs
                for entry in temps["coretemp"]:
                    if "Package" in entry.label or "Physical" in entry.label:
                        return entry.current
            elif "k10temp" in temps:
                # AMD CPUs
                for entry in temps["k10temp"]:
                    return entry.current
            elif "cpu_thermal" in temps:
                # ARM/Mobile CPUs
                for entry in temps["cpu_thermal"]:
                    return entry.current
        except:
            pass
        return None

    def update_profile(self, snapshot: ResourceSnapshot) -> None:
        """Update profile with new snapshot."""
        self.current_snapshot = snapshot

        # Add to history
        self.snapshot_history.append(snapshot)

        # Trim history if too long
        if len(self.snapshot_history) > self.max_history_size:
            self.snapshot_history = self.snapshot_history[-self.max_history_size :]

    def get_evolution_constraints(self) -> dict[str, Any]:
        """Get constraints for evolution tasks based on current state."""
        if not self.current_snapshot:
            return {"available": False, "reason": "No resource data"}

        snapshot = self.current_snapshot
        constraints = {
            "available": True,
            "max_memory_gb": max(
                0.5, snapshot.memory_available_gb * 0.7
            ),  # Use 70% of available
            "max_cpu_percent": max(
                10, 100 - snapshot.cpu_percent - 20
            ),  # Leave 20% headroom
            "max_duration_minutes": 60,  # Default 1 hour limit
            "power_sensitive": snapshot.power_state
            in [PowerState.BATTERY_LOW, PowerState.BATTERY_CRITICAL],
            "thermal_sensitive": snapshot.thermal_state
            in [ThermalState.HOT, ThermalState.CRITICAL],
        }

        # Adjust based on constraints
        if snapshot.is_resource_constrained:
            constraints["available"] = False
            constraints["reason"] = "Device is resource constrained"
        elif snapshot.evolution_suitability_score < 0.3:
            constraints["max_duration_minutes"] = 15
            constraints["max_memory_gb"] *= 0.5
            constraints["max_cpu_percent"] *= 0.5

        return constraints

    def add_monitoring_callback(
        self, callback: Callable[[ResourceSnapshot], None]
    ) -> None:
        """Add callback for monitoring updates."""
        self._callbacks.append(callback)

    def remove_monitoring_callback(
        self, callback: Callable[[ResourceSnapshot], None]
    ) -> None:
        """Remove monitoring callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type.value,
            "platform": self.platform,
            "architecture": self.architecture,
            "cpu_cores": self.cpu_cores,
            "cpu_brand": self.cpu_brand,
            "memory_total_gb": self.memory_total_gb,
            "storage_total_gb": self.storage_total_gb,
            "current_snapshot": self.current_snapshot.__dict__
            if self.current_snapshot
            else None,
            "supports_gpu": self.supports_gpu,
            "supports_neural_engine": self.supports_neural_engine,
            "supports_background_processing": self.supports_background_processing,
            "is_mobile": self.is_mobile,
            "battery_optimization_enabled": self.battery_optimization_enabled,
            "monitoring_active": self._monitoring_active,
        }


class DeviceProfiler:
    """Enhanced device profiler with consolidated features."""

    def __init__(self) -> None:
        self.profile = DeviceProfile()
        self._initialized = False

    def _generate_device_id(self) -> str:
        """Generate unique device ID."""
        import hashlib

        # Use platform-specific identifiers when available
        identifiers = [
            platform.node(),
            platform.system(),
            platform.machine(),
            platform.processor(),
        ]

        # Add platform-specific IDs
        if ANDROID_AVAILABLE:
            try:
                identifiers.append(Build.SERIAL)
                identifiers.append(Build.MODEL)
            except:
                pass
        elif MACOS_AVAILABLE:
            try:
                # Use system profiler for Mac hardware UUID
                import subprocess

                result = subprocess.run(
                    ["system_profiler", "SPHardwareDataType"],
                    capture_output=True,
                    text=True,
                )
                for line in result.stdout.split("\n"):
                    if "Hardware UUID" in line:
                        identifiers.append(line.split(":")[1].strip())
                        break
            except:
                pass

        # Create hash from identifiers
        combined = "".join(str(i) for i in identifiers if i)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def initialize(self) -> DeviceProfile:
        """Initialize and return device profile."""
        if not self._initialized:
            self.profile.device_id = self._generate_device_id()

            # Capture initial snapshot
            snapshot = self.profile._capture_snapshot()
            self.profile.update_profile(snapshot)

            self._initialized = True
            logger.info(
                f"Initialized device profiler for {self.profile.device_type.value} device"
            )

        return self.profile

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start real-time monitoring."""
        if not self._initialized:
            self.initialize()
        self.profile.start_monitoring(interval)

    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.profile.stop_monitoring()

    def get_current_profile(self) -> DeviceProfile:
        """Get current device profile."""
        if not self._initialized:
            self.initialize()
        return self.profile

    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture immediate resource snapshot."""
        return self.profile._capture_snapshot()


# Global profiler instance
_global_profiler: DeviceProfiler | None = None


def get_device_profiler() -> DeviceProfiler:
    """Get global device profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = DeviceProfiler()
    return _global_profiler


def get_device_profile() -> DeviceProfile:
    """Get current device profile."""
    return get_device_profiler().get_current_profile()
