"""
BetaNet Mobile Optimization - Consolidated Mobile Features

Consolidates mobile-specific BetaNet optimizations from deprecated files:
- Battery-aware transport selection and resource management
- Data budget management for cellular connections
- Thermal throttling and adaptive performance scaling
- Network type detection and optimization
- Chunk size adaptation for mobile constraints
"""

import logging
import platform
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class NetworkType(Enum):
    """Network connection types for mobile optimization"""

    WIFI = "wifi"
    CELLULAR_5G = "cellular_5g"
    CELLULAR_4G = "cellular_4g"
    CELLULAR_3G = "cellular_3g"
    CELLULAR_2G = "cellular_2g"
    ETHERNET = "ethernet"
    BLUETOOTH = "bluetooth"
    UNKNOWN = "unknown"


class BatteryState(Enum):
    """Battery power states"""

    FULL = "full"  # 95-100%
    HIGH = "high"  # 70-94%
    MEDIUM = "medium"  # 30-69%
    LOW = "low"  # 10-29%
    CRITICAL = "critical"  # 0-9%
    CHARGING = "charging"  # Any level while plugged in


class ThermalState(Enum):
    """Device thermal states"""

    COOL = "cool"  # < 35°C
    WARM = "warm"  # 35-45°C
    HOT = "hot"  # 45-60°C
    OVERHEATING = "overheating"  # > 60°C


@dataclass
class MobileDeviceProfile:
    """Mobile device performance and capability profile"""

    device_id: str
    platform: str = field(default_factory=platform.system)
    cpu_cores: int = field(default_factory=lambda: psutil.cpu_count())
    memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    battery_percent: float = 100.0
    battery_state: BatteryState = BatteryState.FULL
    thermal_state: ThermalState = ThermalState.COOL
    network_type: NetworkType = NetworkType.WIFI
    data_budget_mb: float | None = None
    data_used_mb: float = 0.0
    performance_mode: str = "balanced"  # power_saver, balanced, performance


@dataclass
class MobileTransportConfig:
    """Configuration optimized for mobile device constraints"""

    max_chunk_size: int = 1024  # Start conservative for mobile
    min_chunk_size: int = 256  # Minimum viable chunk
    compression_enabled: bool = True  # Always compress on mobile
    encryption_level: str = "standard"  # standard, high, paranoid
    battery_optimization: bool = True
    data_budget_enforcement: bool = True
    thermal_throttling: bool = True
    adaptive_retry: bool = True
    connection_pooling: bool = True  # Reuse connections
    keepalive_interval: int = 30  # Seconds between keepalives


class BatteryMonitor:
    """Battery state monitoring and optimization"""

    def __init__(self):
        self.last_check = 0
        self.check_interval = 10  # Check every 10 seconds
        self.battery_history = []
        self.charging_detected = False

    def get_battery_state(self) -> tuple[float, BatteryState]:
        """Get current battery percentage and state"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            # Use cached value
            if self.battery_history:
                return self.battery_history[-1]

        try:
            # Try to get battery info via psutil
            battery = psutil.sensors_battery()
            if battery:
                percent = battery.percent
                is_charging = battery.power_plugged

                if is_charging:
                    state = BatteryState.CHARGING
                elif percent >= 95:
                    state = BatteryState.FULL
                elif percent >= 70:
                    state = BatteryState.HIGH
                elif percent >= 30:
                    state = BatteryState.MEDIUM
                elif percent >= 10:
                    state = BatteryState.LOW
                else:
                    state = BatteryState.CRITICAL

                self.last_check = current_time
                result = (percent, state)
                self.battery_history.append(result)

                # Keep only last 10 readings
                if len(self.battery_history) > 10:
                    self.battery_history = self.battery_history[-10:]

                return result

        except Exception as e:
            logger.debug(f"Battery monitoring error: {e}")

        # Fallback - assume decent battery on desktop
        return (75.0, BatteryState.HIGH)

    def is_low_battery(self, threshold_percent: float = 15.0) -> bool:
        """Check if battery is below threshold"""
        percent, state = self.get_battery_state()
        return percent < threshold_percent and state != BatteryState.CHARGING


class ThermalMonitor:
    """Device thermal monitoring and throttling"""

    def __init__(self):
        self.last_check = 0
        self.check_interval = 15  # Check every 15 seconds
        self.temp_history = []

    def get_thermal_state(self) -> tuple[float, ThermalState]:
        """Get current CPU temperature and thermal state"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            if self.temp_history:
                return self.temp_history[-1]

        try:
            # Try to get temperature readings
            temps = psutil.sensors_temperatures()
            if temps:
                # Find CPU temperature
                cpu_temps = []
                for name, entries in temps.items():
                    if "cpu" in name.lower() or "core" in name.lower():
                        for entry in entries:
                            if entry.current:
                                cpu_temps.append(entry.current)

                if cpu_temps:
                    avg_temp = sum(cpu_temps) / len(cpu_temps)

                    if avg_temp < 35:
                        state = ThermalState.COOL
                    elif avg_temp < 45:
                        state = ThermalState.WARM
                    elif avg_temp < 60:
                        state = ThermalState.HOT
                    else:
                        state = ThermalState.OVERHEATING

                    self.last_check = current_time
                    result = (avg_temp, state)
                    self.temp_history.append(result)

                    if len(self.temp_history) > 10:
                        self.temp_history = self.temp_history[-10:]

                    return result

        except Exception as e:
            logger.debug(f"Thermal monitoring error: {e}")

        # Fallback - assume normal temperature
        return (40.0, ThermalState.WARM)

    def needs_thermal_throttling(self, threshold_temp: float = 55.0) -> bool:
        """Check if thermal throttling is needed"""
        temp, state = self.get_thermal_state()
        return temp > threshold_temp or state == ThermalState.OVERHEATING


class NetworkDetector:
    """Network type detection and optimization"""

    def __init__(self):
        self.current_type = NetworkType.UNKNOWN
        self.last_detection = 0
        self.detection_interval = 30  # Check every 30 seconds

    def detect_network_type(self) -> NetworkType:
        """Detect current network connection type"""
        current_time = time.time()
        if current_time - self.last_detection < self.detection_interval:
            return self.current_type

        try:
            # Get network interface stats
            net_stats = psutil.net_if_stats()
            psutil.net_if_addrs()

            # Look for active interfaces
            active_interfaces = []
            for interface, stats in net_stats.items():
                if stats.isup and not interface.startswith("lo"):  # Skip loopback
                    active_interfaces.append(interface)

            # Classify network type based on interface names
            for interface in active_interfaces:
                interface_lower = interface.lower()

                if any(x in interface_lower for x in ["wifi", "wlan", "wireless"]):
                    self.current_type = NetworkType.WIFI
                    break
                elif any(x in interface_lower for x in ["cellular", "4g", "5g", "lte"]):
                    self.current_type = NetworkType.CELLULAR_4G  # Default cellular
                    break
                elif any(x in interface_lower for x in ["ethernet", "eth0", "en0"]):
                    self.current_type = NetworkType.ETHERNET
                    break
                elif any(x in interface_lower for x in ["bluetooth", "bt"]):
                    self.current_type = NetworkType.BLUETOOTH
                    break

            self.last_detection = current_time

        except Exception as e:
            logger.debug(f"Network detection error: {e}")

        return self.current_type

    def is_metered_connection(self) -> bool:
        """Check if connection is metered (cellular)"""
        network_type = self.detect_network_type()
        return network_type in [
            NetworkType.CELLULAR_2G,
            NetworkType.CELLULAR_3G,
            NetworkType.CELLULAR_4G,
            NetworkType.CELLULAR_5G,
        ]


class DataBudgetManager:
    """Data usage budget management for cellular connections"""

    def __init__(self, daily_budget_mb: float | None = None):
        self.daily_budget_mb = daily_budget_mb or 100.0  # 100MB default
        self.usage_today_mb = 0.0
        self.usage_history = {}
        self.last_reset = time.time()
        self.warning_threshold = 0.8  # Warn at 80%

    def record_usage(self, bytes_used: int):
        """Record data usage"""
        mb_used = bytes_used / (1024 * 1024)
        self.usage_today_mb += mb_used

        # Reset daily counter if needed
        current_day = time.strftime("%Y-%m-%d")
        if current_day not in self.usage_history:
            self.usage_history[current_day] = 0.0

        self.usage_history[current_day] += mb_used

    def get_remaining_budget(self) -> float:
        """Get remaining data budget in MB"""
        return max(0, self.daily_budget_mb - self.usage_today_mb)

    def is_over_budget(self) -> bool:
        """Check if over daily budget"""
        return self.usage_today_mb >= self.daily_budget_mb

    def is_approaching_budget(self) -> bool:
        """Check if approaching budget threshold"""
        return self.usage_today_mb >= (self.daily_budget_mb * self.warning_threshold)

    def get_usage_stats(self) -> dict[str, Any]:
        """Get usage statistics"""
        return {
            "daily_budget_mb": self.daily_budget_mb,
            "usage_today_mb": self.usage_today_mb,
            "remaining_mb": self.get_remaining_budget(),
            "usage_percent": min(100, (self.usage_today_mb / self.daily_budget_mb) * 100),
            "is_over_budget": self.is_over_budget(),
            "is_approaching_budget": self.is_approaching_budget(),
        }


class AdaptiveChunkingManager:
    """Adaptive chunk size management based on device constraints"""

    def __init__(self, config: MobileTransportConfig):
        self.config = config
        self.current_chunk_size = config.max_chunk_size
        self.performance_history = []
        self.adaptation_factor = 0.8  # Reduce by 20% on issues

    def calculate_optimal_chunk_size(
        self,
        battery_state: BatteryState,
        thermal_state: ThermalState,
        network_type: NetworkType,
        available_memory_mb: float,
    ) -> int:
        """Calculate optimal chunk size based on current conditions"""

        base_size = self.config.max_chunk_size

        # Battery optimization
        if battery_state in [BatteryState.LOW, BatteryState.CRITICAL]:
            base_size = int(base_size * 0.5)  # Halve chunk size on low battery
        elif battery_state == BatteryState.MEDIUM:
            base_size = int(base_size * 0.7)

        # Thermal throttling
        if thermal_state == ThermalState.OVERHEATING:
            base_size = int(base_size * 0.3)  # Aggressive reduction
        elif thermal_state == ThermalState.HOT:
            base_size = int(base_size * 0.6)

        # Network type optimization
        network_factors = {
            NetworkType.ETHERNET: 1.0,
            NetworkType.WIFI: 0.9,
            NetworkType.CELLULAR_5G: 0.8,
            NetworkType.CELLULAR_4G: 0.6,
            NetworkType.CELLULAR_3G: 0.4,
            NetworkType.CELLULAR_2G: 0.2,
            NetworkType.BLUETOOTH: 0.3,
            NetworkType.UNKNOWN: 0.5,
        }
        base_size = int(base_size * network_factors.get(network_type, 0.5))

        # Memory constraints
        if available_memory_mb < 512:  # Less than 512MB available
            base_size = int(base_size * 0.5)
        elif available_memory_mb < 1024:  # Less than 1GB available
            base_size = int(base_size * 0.7)

        # Enforce bounds
        optimal_size = max(self.config.min_chunk_size, min(base_size, self.config.max_chunk_size))

        self.current_chunk_size = optimal_size
        return optimal_size

    def record_performance(self, chunk_size: int, transfer_time_ms: float, success: bool):
        """Record chunk transfer performance for adaptive optimization"""
        perf_record = {
            "chunk_size": chunk_size,
            "transfer_time_ms": transfer_time_ms,
            "success": success,
            "timestamp": time.time(),
        }

        self.performance_history.append(perf_record)

        # Keep only recent history
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

        # Adapt chunk size based on recent performance
        recent_failures = sum(1 for p in self.performance_history[-10:] if not p["success"])
        if recent_failures >= 3:  # 3 failures in last 10 transfers
            self.current_chunk_size = int(self.current_chunk_size * self.adaptation_factor)
            self.current_chunk_size = max(self.config.min_chunk_size, self.current_chunk_size)
            logger.info(f"Adapted chunk size to {self.current_chunk_size} due to failures")


class MobileBetaNetOptimizer:
    """Unified mobile optimization manager for BetaNet transport"""

    def __init__(self, device_profile: MobileDeviceProfile, transport_config: MobileTransportConfig):
        self.device_profile = device_profile
        self.transport_config = transport_config

        # Initialize monitors
        self.battery_monitor = BatteryMonitor()
        self.thermal_monitor = ThermalMonitor()
        self.network_detector = NetworkDetector()
        self.data_budget = DataBudgetManager(device_profile.data_budget_mb)
        self.chunk_manager = AdaptiveChunkingManager(transport_config)

        # Optimization state
        self.optimization_active = True
        self.last_optimization = 0
        self.optimization_interval = 5  # Optimize every 5 seconds

    async def optimize_transport_settings(self) -> dict[str, Any]:
        """Optimize transport settings based on current device state"""

        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return self._get_current_settings()

        # Get current device state
        battery_percent, battery_state = self.battery_monitor.get_battery_state()
        cpu_temp, thermal_state = self.thermal_monitor.get_thermal_state()
        network_type = self.network_detector.detect_network_type()

        # Get available memory
        memory_info = psutil.virtual_memory()
        available_memory_mb = memory_info.available / (1024 * 1024)

        # Update device profile
        self.device_profile.battery_percent = battery_percent
        self.device_profile.battery_state = battery_state
        self.device_profile.thermal_state = thermal_state
        self.device_profile.network_type = network_type

        # Calculate optimal chunk size
        optimal_chunk_size = self.chunk_manager.calculate_optimal_chunk_size(
            battery_state, thermal_state, network_type, available_memory_mb
        )

        # Determine if we should throttle operations
        should_throttle = (
            battery_state in [BatteryState.LOW, BatteryState.CRITICAL]
            or thermal_state in [ThermalState.HOT, ThermalState.OVERHEATING]
            or self.data_budget.is_over_budget()
        )

        # Determine compression level
        compression_level = "high" if network_type in [NetworkType.CELLULAR_2G, NetworkType.CELLULAR_3G] else "standard"

        settings = {
            "chunk_size": optimal_chunk_size,
            "compression_level": compression_level,
            "throttle_enabled": should_throttle,
            "max_concurrent_transfers": 1 if should_throttle else 3,
            "retry_attempts": 2 if should_throttle else 5,
            "connection_timeout": 10 if should_throttle else 30,
            "use_data_budget": network_type
            in [NetworkType.CELLULAR_2G, NetworkType.CELLULAR_3G, NetworkType.CELLULAR_4G, NetworkType.CELLULAR_5G],
            "prefer_compression": True,
            "adaptive_retry_delay": True,
            "battery_state": battery_state.value,
            "thermal_state": thermal_state.value,
            "network_type": network_type.value,
            "available_memory_mb": available_memory_mb,
        }

        self.last_optimization = current_time

        logger.debug(
            f"Mobile optimization: chunk_size={optimal_chunk_size}, "
            f"throttle={should_throttle}, network={network_type.value}"
        )

        return settings

    def _get_current_settings(self) -> dict[str, Any]:
        """Get current optimization settings without recalculation"""
        return {
            "chunk_size": self.chunk_manager.current_chunk_size,
            "battery_state": self.device_profile.battery_state.value,
            "thermal_state": self.device_profile.thermal_state.value,
            "network_type": self.device_profile.network_type.value,
        }

    async def record_transfer_stats(
        self, bytes_transferred: int, transfer_time_ms: float, success: bool, chunk_size: int
    ):
        """Record transfer statistics for optimization"""

        # Record data usage for budget tracking
        if self.network_detector.is_metered_connection():
            self.data_budget.record_usage(bytes_transferred)

        # Record performance for chunk size adaptation
        self.chunk_manager.record_performance(chunk_size, transfer_time_ms, success)

    def should_pause_transfers(self) -> bool:
        """Check if transfers should be paused due to device constraints"""
        battery_percent, battery_state = self.battery_monitor.get_battery_state()
        cpu_temp, thermal_state = self.thermal_monitor.get_thermal_state()

        # Pause on critical battery or overheating
        if battery_state == BatteryState.CRITICAL and battery_percent < 5:
            return True

        if thermal_state == ThermalState.OVERHEATING:
            return True

        # Pause if severely over data budget
        if self.data_budget.is_over_budget() and self.network_detector.is_metered_connection():
            return True

        return False

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics"""
        battery_percent, battery_state = self.battery_monitor.get_battery_state()
        cpu_temp, thermal_state = self.thermal_monitor.get_thermal_state()

        return {
            "device_profile": {
                "platform": self.device_profile.platform,
                "cpu_cores": self.device_profile.cpu_cores,
                "memory_gb": self.device_profile.memory_gb,
                "performance_mode": self.device_profile.performance_mode,
            },
            "current_state": {
                "battery_percent": battery_percent,
                "battery_state": battery_state.value,
                "cpu_temp": cpu_temp,
                "thermal_state": thermal_state.value,
                "network_type": self.network_detector.detect_network_type().value,
                "is_metered": self.network_detector.is_metered_connection(),
            },
            "optimization": {
                "current_chunk_size": self.chunk_manager.current_chunk_size,
                "should_pause": self.should_pause_transfers(),
                "optimization_active": self.optimization_active,
            },
            "data_budget": self.data_budget.get_usage_stats(),
        }


# Factory function for easy integration
def create_mobile_optimized_client(
    device_id: str = "mobile_device", data_budget_mb: float | None = None, max_chunk_size: int = 2048
) -> MobileBetaNetOptimizer:
    """
    Create mobile-optimized BetaNet client with automatic device detection

    Args:
        device_id: Unique device identifier
        data_budget_mb: Daily data budget (None for unlimited)
        max_chunk_size: Maximum chunk size in bytes

    Returns:
        Configured mobile optimizer
    """
    # Create device profile with auto-detection
    device_profile = MobileDeviceProfile(device_id=device_id, data_budget_mb=data_budget_mb)

    # Create transport config optimized for mobile
    transport_config = MobileTransportConfig(
        max_chunk_size=max_chunk_size,
        min_chunk_size=256,
        compression_enabled=True,
        battery_optimization=True,
        data_budget_enforcement=True,
        thermal_throttling=True,
    )

    optimizer = MobileBetaNetOptimizer(device_profile, transport_config)

    logger.info(
        f"Created mobile-optimized BetaNet client: device={device_id}, "
        f"budget={data_budget_mb}MB, max_chunk={max_chunk_size}"
    )

    return optimizer
