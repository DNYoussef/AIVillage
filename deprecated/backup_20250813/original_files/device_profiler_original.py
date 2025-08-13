"""Real-time mobile device resource profiling for Sprint 6."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import platform
import queue
import threading
import time
from typing import Any

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
        pass
elif platform.system() == "Darwin":  # iOS/macOS
    try:
        from Foundation import NSBundle, NSProcessInfo  # type: ignore
        import objc  # type: ignore

        MACOS_AVAILABLE = True
    except ImportError:
        # Fallback when Foundation isn't available (e.g., non-macOS platforms)
        MACOS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DeviceProfile:
    """Current device resource state - Enhanced for Sprint 6."""

    timestamp: float
    cpu_percent: float
    cpu_freq_mhz: float
    cpu_temp_celsius: float | None
    cpu_cores: int
    ram_used_mb: int
    ram_available_mb: int
    ram_total_mb: int
    battery_percent: int | None
    battery_charging: bool
    battery_temp_celsius: float | None
    battery_health: str | None
    network_type: str
    network_bandwidth_mbps: float | None
    network_latency_ms: float | None
    storage_available_gb: float
    storage_total_gb: float
    gpu_available: bool
    gpu_memory_mb: int | None
    thermal_state: str
    power_mode: str
    screen_brightness: int | None
    device_type: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "cpu": {
                "percent": self.cpu_percent,
                "freq_mhz": self.cpu_freq_mhz,
                "temp_celsius": self.cpu_temp_celsius,
                "cores": self.cpu_cores,
            },
            "memory": {
                "used_mb": self.ram_used_mb,
                "available_mb": self.ram_available_mb,
                "total_mb": self.ram_total_mb,
                "usage_percent": (self.ram_used_mb / self.ram_total_mb) * 100,
            },
            "battery": {
                "percent": self.battery_percent,
                "charging": self.battery_charging,
                "temp_celsius": self.battery_temp_celsius,
                "health": self.battery_health,
            },
            "network": {
                "type": self.network_type,
                "bandwidth_mbps": self.network_bandwidth_mbps,
                "latency_ms": self.network_latency_ms,
            },
            "storage": {
                "available_gb": self.storage_available_gb,
                "total_gb": self.storage_total_gb,
                "usage_percent": ((self.storage_total_gb - self.storage_available_gb) / self.storage_total_gb) * 100,
            },
            "gpu": {"available": self.gpu_available, "memory_mb": self.gpu_memory_mb},
            "system": {
                "thermal_state": self.thermal_state,
                "power_mode": self.power_mode,
                "screen_brightness": self.screen_brightness,
                "device_type": self.device_type,
            },
        }


class DeviceType(Enum):
    """Device type classification."""

    PHONE = "phone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    EMBEDDED = "embedded"
    UNKNOWN = "unknown"


class PowerState(Enum):
    """Device power state."""

    PLUGGED_IN = "plugged_in"
    BATTERY_HIGH = "battery_high"  # > 80%
    BATTERY_MEDIUM = "battery_medium"  # 20-80%
    BATTERY_LOW = "battery_low"  # 5-20%
    BATTERY_CRITICAL = "battery_critical"  # < 5%
    UNKNOWN = "unknown"


class ThermalState(Enum):
    """Device thermal state."""

    NORMAL = "normal"  # < 60°C
    WARM = "warm"  # 60-75°C
    HOT = "hot"  # 75-85°C
    CRITICAL = "critical"  # > 85°C
    THROTTLING = "throttling"  # CPU throttling detected
    UNKNOWN = "unknown"


@dataclass
class ResourceSnapshot:
    """Snapshot of device resources at a point in time."""

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

    # Network
    network_sent: int = 0
    network_received: int = 0
    network_connections: int = 0

    # Process count
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
        )

    @property
    def performance_score(self) -> float:
        """Calculate performance score (0-1, higher is better)."""
        # Weight factors
        memory_score = max(0, (100 - self.memory_percent) / 100)
        cpu_score = max(0, (100 - self.cpu_percent) / 100)

        storage_score = 1.0
        if self.storage_percent > 0:
            storage_score = max(0, (100 - self.storage_percent) / 100)

        power_score = 1.0
        if self.battery_percent is not None:
            power_score = 1.0 if self.power_plugged else self.battery_percent / 100

        # Weighted average
        return memory_score * 0.4 + cpu_score * 0.3 + storage_score * 0.2 + power_score * 0.1


@dataclass
class DeviceProfile:
    """Complete device profile and capabilities."""

    device_id: str

    # Hardware info
    device_type: DeviceType
    os_type: str
    os_version: str
    architecture: str

    # Capabilities
    total_memory_gb: float
    cpu_cores: int
    cpu_model: str

    # Feature support
    supports_gpu: bool = False
    supports_bluetooth: bool = False
    supports_wifi: bool = False
    supports_cellular: bool = False

    # Performance characteristics
    benchmark_score: float | None = None
    typical_performance: float | None = None

    # Resource constraints
    memory_limit_mb: int | None = None
    cpu_limit_percent: float | None = None
    battery_optimization: bool = False

    # Monitoring configuration
    monitoring_interval: float = 10.0  # seconds

    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def update_profile(self, **kwargs) -> None:
        """Update profile with new information."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "os_type": self.os_type,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "total_memory_gb": self.total_memory_gb,
            "cpu_cores": self.cpu_cores,
            "cpu_model": self.cpu_model,
            "supports_gpu": self.supports_gpu,
            "supports_bluetooth": self.supports_bluetooth,
            "supports_wifi": self.supports_wifi,
            "supports_cellular": self.supports_cellular,
            "benchmark_score": self.benchmark_score,
            "typical_performance": self.typical_performance,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "battery_optimization": self.battery_optimization,
            "monitoring_interval": self.monitoring_interval,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }


class DeviceProfiler:
    """Real-time device profiler for mobile resource monitoring."""

    def __init__(
        self,
        device_id: str | None = None,
        monitoring_interval: float = 5.0,
        history_size: int = 1000,
        enable_background_monitoring: bool = True,
    ) -> None:
        self.device_id = device_id or self._generate_device_id()
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_background_monitoring = enable_background_monitoring

        # Device profile
        self.profile = self._create_device_profile()

        # Resource monitoring
        self.snapshots: list[ResourceSnapshot] = []
        self.current_snapshot: ResourceSnapshot | None = None

        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread: threading.Thread | None = None
        self.snapshot_queue = queue.Queue(maxsize=100)

        # Callbacks
        self.alert_callbacks: list[Callable[[str, ResourceSnapshot], None]] = []
        self.threshold_callbacks: dict[str, list[Callable[[ResourceSnapshot], None]]] = {}

        # Thresholds for alerts
        self.thresholds = {
            "memory_critical": 95.0,
            "memory_warning": 85.0,
            "cpu_critical": 95.0,
            "cpu_warning": 80.0,
            "battery_critical": 5.0,
            "battery_low": 15.0,
            "temperature_critical": 85.0,
            "temperature_warning": 75.0,
        }

        # Statistics
        self.stats = {
            "snapshots_taken": 0,
            "alerts_triggered": 0,
            "monitoring_uptime": 0.0,
            "last_alert_time": None,
        }

        logger.info(f"Device profiler initialized for {self.profile.device_type.value} device")

    def _generate_device_id(self) -> str:
        """Generate unique device ID."""
        import hashlib
        import uuid

        # Use MAC address and hostname for consistent ID
        try:
            mac = hex(uuid.getnode())[2:]
            hostname = platform.node()
            device_string = f"{mac}-{hostname}-{platform.system()}"
            return hashlib.md5(device_string.encode()).hexdigest()[:16]
        except:
            return str(uuid.uuid4())[:16]

    def _create_device_profile(self) -> DeviceProfile:
        """Create comprehensive device profile."""
        # Detect device type
        device_type = self._detect_device_type()

        # Get system information
        memory = psutil.virtual_memory()
        cpu_info = self._get_cpu_info()

        return DeviceProfile(
            device_id=self.device_id,
            device_type=device_type,
            os_type=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            total_memory_gb=memory.total / (1024**3),
            cpu_cores=psutil.cpu_count(logical=True),
            cpu_model=cpu_info.get("model", "Unknown"),
            supports_gpu=self._detect_gpu_support(),
            supports_bluetooth=self._detect_bluetooth_support(),
            supports_wifi=self._detect_wifi_support(),
            supports_cellular=self._detect_cellular_support(device_type),
            monitoring_interval=self.monitoring_interval,
        )

    def _detect_device_type(self) -> DeviceType:
        """Detect device type based on system characteristics."""
        system = platform.system().lower()

        # Memory-based heuristics
        memory_gb = psutil.virtual_memory().total / (1024**3)

        if system == "android":
            return DeviceType.PHONE if memory_gb < 6 else DeviceType.TABLET
        if system == "darwin":
            if platform.machine().startswith("iP"):
                return DeviceType.PHONE if "iPhone" in platform.machine() else DeviceType.TABLET
            return DeviceType.LAPTOP if memory_gb < 16 else DeviceType.DESKTOP
        if system in ["linux", "windows"]:
            if memory_gb < 4:
                return DeviceType.EMBEDDED
            if memory_gb < 8:
                return DeviceType.LAPTOP
            return DeviceType.DESKTOP
        return DeviceType.UNKNOWN

    def _get_cpu_info(self) -> dict[str, Any]:
        """Get CPU information."""
        try:
            # Try to get CPU model
            if platform.system() == "Windows":
                import subprocess

                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    if len(lines) > 1:
                        return {"model": lines[1].strip()}
            elif platform.system() in ["Linux", "Darwin"]:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            return {"model": line.split(":")[1].strip()}
        except:
            pass

        return {"model": f"{platform.processor()} ({psutil.cpu_count()} cores)"}

    def _detect_gpu_support(self) -> bool:
        """Detect if GPU is available."""
        try:
            # Try to detect NVIDIA GPU
            import subprocess

            result = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
            return result.returncode == 0
        except:
            pass

        # Check for integrated graphics or other indicators
        return False

    def _detect_bluetooth_support(self) -> bool:
        """Detect Bluetooth support."""
        # Simplified detection - would use platform-specific APIs in production
        system = platform.system().lower()
        return system in ["android", "darwin", "linux"]

    def _detect_wifi_support(self) -> bool:
        """Detect WiFi support."""
        # Most modern devices have WiFi
        return True

    def _detect_cellular_support(self, device_type: DeviceType) -> bool:
        """Detect cellular support."""
        # Mainly mobile devices
        return device_type in [DeviceType.PHONE, DeviceType.TABLET]

    def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource usage."""
        timestamp = time.time()

        # Memory information
        memory = psutil.virtual_memory()

        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = None
        cpu_temp = None

        try:
            freq = psutil.cpu_freq()
            if freq:
                cpu_freq = freq.current
        except:
            pass

        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature from any available sensor
                for sensor_name, sensor_list in temps.items():
                    if "cpu" in sensor_name.lower() or "core" in sensor_name.lower():
                        if sensor_list:
                            cpu_temp = sensor_list[0].current
                            break
        except:
            pass

        # Storage information
        storage_total = storage_used = storage_free = 0
        storage_percent = 0.0

        try:
            disk = psutil.disk_usage("/")
            storage_total = disk.total
            storage_used = disk.used
            storage_free = disk.free
            storage_percent = (disk.used / disk.total) * 100
        except:
            pass

        # Battery information
        battery_percent = None
        power_plugged = None
        power_state = PowerState.UNKNOWN

        try:
            battery = psutil.sensors_battery()
            if battery:
                battery_percent = battery.percent
                power_plugged = battery.power_plugged

                if power_plugged:
                    power_state = PowerState.PLUGGED_IN
                elif battery_percent > 80:
                    power_state = PowerState.BATTERY_HIGH
                elif battery_percent > 20:
                    power_state = PowerState.BATTERY_MEDIUM
                elif battery_percent > 5:
                    power_state = PowerState.BATTERY_LOW
                else:
                    power_state = PowerState.BATTERY_CRITICAL
        except:
            pass

        # Network information
        network_stats = psutil.net_io_counters()
        network_connections = len(psutil.net_connections())

        # Process count
        process_count = len(psutil.pids())

        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            memory_total=memory.total,
            memory_available=memory.available,
            memory_used=memory.used,
            memory_percent=memory.percent,
            cpu_percent=cpu_percent,
            cpu_cores=psutil.cpu_count(),
            cpu_freq_current=cpu_freq,
            cpu_temp=cpu_temp,
            storage_total=storage_total,
            storage_used=storage_used,
            storage_free=storage_free,
            storage_percent=storage_percent,
            battery_percent=battery_percent,
            power_plugged=power_plugged,
            power_state=power_state,
            network_sent=network_stats.bytes_sent,
            network_received=network_stats.bytes_recv,
            network_connections=network_connections,
            process_count=process_count,
        )

        # Store snapshot
        self.current_snapshot = snapshot
        self.snapshots.append(snapshot)

        # Maintain history size
        if len(self.snapshots) > self.history_size:
            self.snapshots = self.snapshots[-self.history_size :]

        # Update stats
        self.stats["snapshots_taken"] += 1

        # Check thresholds and trigger alerts
        self._check_thresholds(snapshot)

        # Queue for background processing
        try:
            self.snapshot_queue.put_nowait(snapshot)
        except queue.Full:
            pass  # Queue full, skip this snapshot

        return snapshot

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True

        if self.enable_background_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

        logger.info("Device monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)

        logger.info("Device monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        start_time = time.time()

        while self.monitoring_active:
            try:
                self.take_snapshot()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.exception(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

        self.stats["monitoring_uptime"] = time.time() - start_time

    def _check_thresholds(self, snapshot: ResourceSnapshot) -> None:
        """Check resource thresholds and trigger alerts."""
        alerts_triggered = []

        # Memory checks
        if snapshot.memory_percent > self.thresholds["memory_critical"]:
            alerts_triggered.append("memory_critical")
        elif snapshot.memory_percent > self.thresholds["memory_warning"]:
            alerts_triggered.append("memory_warning")

        # CPU checks
        if snapshot.cpu_percent > self.thresholds["cpu_critical"]:
            alerts_triggered.append("cpu_critical")
        elif snapshot.cpu_percent > self.thresholds["cpu_warning"]:
            alerts_triggered.append("cpu_warning")

        # Battery checks
        if snapshot.battery_percent is not None:
            if snapshot.battery_percent < self.thresholds["battery_critical"]:
                alerts_triggered.append("battery_critical")
            elif snapshot.battery_percent < self.thresholds["battery_low"]:
                alerts_triggered.append("battery_low")

        # Temperature checks
        if snapshot.cpu_temp is not None:
            if snapshot.cpu_temp > self.thresholds["temperature_critical"]:
                alerts_triggered.append("temperature_critical")
            elif snapshot.cpu_temp > self.thresholds["temperature_warning"]:
                alerts_triggered.append("temperature_warning")

        # Trigger alerts
        for alert_type in alerts_triggered:
            self._trigger_alert(alert_type, snapshot)

    def _trigger_alert(self, alert_type: str, snapshot: ResourceSnapshot) -> None:
        """Trigger alert for resource threshold violation."""
        self.stats["alerts_triggered"] += 1
        self.stats["last_alert_time"] = time.time()

        logger.warning(f"Resource alert: {alert_type} - {self._format_alert_message(alert_type, snapshot)}")

        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, snapshot)
            except Exception as e:
                logger.exception(f"Error in alert callback: {e}")

        # Call threshold-specific callbacks
        for callback in self.threshold_callbacks.get(alert_type, []):
            try:
                callback(snapshot)
            except Exception as e:
                logger.exception(f"Error in threshold callback: {e}")

    def _format_alert_message(self, alert_type: str, snapshot: ResourceSnapshot) -> str:
        """Format alert message."""
        if "memory" in alert_type:
            return f"Memory usage: {snapshot.memory_percent:.1f}% ({snapshot.memory_usage_gb:.1f}GB used)"
        if "cpu" in alert_type:
            return f"CPU usage: {snapshot.cpu_percent:.1f}%"
        if "battery" in alert_type:
            return f"Battery level: {snapshot.battery_percent:.1f}%"
        if "temperature" in alert_type:
            return f"CPU temperature: {snapshot.cpu_temp:.1f}°C"
        return f"Resource threshold exceeded: {alert_type}"

    def register_alert_callback(self, callback: Callable[[str, ResourceSnapshot], None]) -> None:
        """Register callback for all alerts."""
        self.alert_callbacks.append(callback)

    def register_threshold_callback(self, threshold: str, callback: Callable[[ResourceSnapshot], None]) -> None:
        """Register callback for specific threshold."""
        if threshold not in self.threshold_callbacks:
            self.threshold_callbacks[threshold] = []
        self.threshold_callbacks[threshold].append(callback)

    def set_threshold(self, threshold_name: str, value: float) -> None:
        """Set resource threshold."""
        self.thresholds[threshold_name] = value
        logger.info(f"Set threshold {threshold_name} to {value}")

    def get_current_status(self) -> dict[str, Any]:
        """Get current device status."""
        current = self.current_snapshot

        if not current:
            return {"status": "no_data"}

        return {
            "device_id": self.device_id,
            "device_type": self.profile.device_type.value,
            "timestamp": current.timestamp,
            "resource_constrained": current.is_resource_constrained,
            "performance_score": current.performance_score,
            "memory": {
                "used_gb": current.memory_usage_gb,
                "available_gb": current.memory_available_gb,
                "percent": current.memory_percent,
            },
            "cpu": {
                "percent": current.cpu_percent,
                "cores": current.cpu_cores,
                "temperature": current.cpu_temp,
            },
            "power": {
                "battery_percent": current.battery_percent,
                "plugged_in": current.power_plugged,
                "state": current.power_state.value,
            },
            "storage": {
                "used_gb": current.storage_used_gb,
                "percent": current.storage_percent,
            },
        }

    def get_historical_data(
        self,
        duration_minutes: int | None = None,
        metric: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get historical monitoring data."""
        snapshots = self.snapshots

        if duration_minutes:
            cutoff_time = time.time() - (duration_minutes * 60)
            snapshots = [s for s in snapshots if s.timestamp > cutoff_time]

        if metric:
            # Return specific metric over time
            data = []
            for snapshot in snapshots:
                value = getattr(snapshot, metric, None)
                if value is not None:
                    data.append(
                        {
                            "timestamp": snapshot.timestamp,
                            "value": value,
                        }
                    )
            return data
        # Return full snapshots
        return [
            {
                "timestamp": s.timestamp,
                "memory_percent": s.memory_percent,
                "cpu_percent": s.cpu_percent,
                "battery_percent": s.battery_percent,
                "power_state": s.power_state.value,
                "performance_score": s.performance_score,
            }
            for s in snapshots
        ]

    def get_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends."""
        if len(self.snapshots) < 10:
            return {"status": "insufficient_data"}

        recent_snapshots = self.snapshots[-100:]  # Last 100 snapshots

        # Calculate averages and trends
        memory_values = [s.memory_percent for s in recent_snapshots]
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        performance_scores = [s.performance_score for s in recent_snapshots]

        return {
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
                "trend": ("increasing" if memory_values[-5:] > memory_values[:5] else "stable"),
            },
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
                "trend": "increasing" if cpu_values[-5:] > cpu_values[:5] else "stable",
            },
            "performance": {
                "avg": sum(performance_scores) / len(performance_scores),
                "min": min(performance_scores),
                "max": max(performance_scores),
                "trend": ("improving" if performance_scores[-5:] > performance_scores[:5] else "stable"),
            },
            "analysis_period": f"{len(recent_snapshots)} snapshots",
        }

    def export_profile(self, include_history: bool = False) -> dict[str, Any]:
        """Export device profile and optionally history."""
        data = {
            "profile": self.profile.to_dict(),
            "current_status": self.get_current_status(),
            "statistics": self.stats.copy(),
            "thresholds": self.thresholds.copy(),
        }

        if include_history:
            data["history"] = self.get_historical_data()

        return data

    def get_monitoring_stats(self) -> dict[str, Any]:
        """Get monitoring statistics."""
        return {
            **self.stats,
            "monitoring_active": self.monitoring_active,
            "snapshots_in_memory": len(self.snapshots),
            "queue_size": self.snapshot_queue.qsize(),
            "alerts_configured": len(self.thresholds),
            "callbacks_registered": len(self.alert_callbacks),
        }
