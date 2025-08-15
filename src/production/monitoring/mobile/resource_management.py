"""Battery/Thermal-Aware Resource Management for Mobile Devices

Implements intelligent resource management with:
- Battery-aware transport selection (BitChat-first under low power)
- Thermal throttling with progressive limits
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Network cost-aware routing decisions
- Real-time policy adaptation

Priority: P2 - Mobile Resource Optimization
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Handle cross-platform import - add current directory to path for safe imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from device_profiler import DeviceProfile
    from resource_allocator import ResourceAllocation, ResourceAllocator
except ImportError:
    # Fallback for development/testing - define minimal mock classes
    from dataclasses import dataclass
    from typing import Any

    @dataclass
    class DeviceProfile:
        """Mock DeviceProfile for safe importing - matches device_profiler.py signature"""

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
        battery_temp_celsius: float | None = None
        battery_health: str | None = None
        network_type: str = "wifi"
        network_bandwidth_mbps: float | None = None
        network_latency_ms: float | None = None
        storage_available_gb: float = 50.0
        storage_total_gb: float = 100.0
        gpu_available: bool = False
        gpu_memory_mb: int | None = None
        thermal_state: str = "normal"
        power_mode: str = "balanced"
        screen_brightness: int | None = None
        device_type: str = "laptop"

        def to_dict(self) -> dict[str, Any]:
            return {
                "timestamp": self.timestamp,
                "cpu_percent": self.cpu_percent,
                "cpu_temp_celsius": self.cpu_temp_celsius,
                "battery_percent": self.battery_percent,
                "battery_charging": self.battery_charging,
                "ram_used_mb": self.ram_used_mb,
                "ram_total_mb": self.ram_total_mb,
                "ram_available_mb": self.ram_available_mb,
                "network_type": self.network_type,
                "network_latency_ms": self.network_latency_ms,
                "device_type": self.device_type,
            }

    @dataclass
    class ResourceAllocation:
        """Mock ResourceAllocation for safe importing"""

        cpu_limit_percent: float = 100.0
        memory_limit_mb: int = 2048
        reason: str = "default"

        def to_dict(self) -> dict[str, Any]:
            return {
                "cpu_limit_percent": self.cpu_limit_percent,
                "memory_limit_mb": self.memory_limit_mb,
                "reason": self.reason,
            }

    class ResourceAllocator:
        """Mock ResourceAllocator for safe importing"""

        def __init__(self):
            """Initialize mock resource allocator."""
            self.allocated_resources = {}


logger = logging.getLogger(__name__)


class PowerMode(Enum):
    """Device power management modes"""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    CRITICAL = "critical"


class TransportPreference(Enum):
    """Transport selection preferences based on constraints"""

    BITCHAT_ONLY = "bitchat_only"
    BITCHAT_PREFERRED = "bitchat_preferred"
    BALANCED = "balanced"
    BETANET_PREFERRED = "betanet_preferred"
    BETANET_ONLY = "betanet_only"


DEFAULT_CHUNK_SIZE = 1024
LOW_RESOURCE_CHUNK_SIZE = 256


def evaluate_policy_from_env() -> dict[str, Any]:
    """Evaluate resource policy using environment variables."""
    battery_str = os.getenv("BATTERY", "100")
    thermal_str = os.getenv("THERMAL", "")
    profile_name = os.getenv("AIV_MOBILE_PROFILE", "normal").lower()

    try:
        battery = int(battery_str)
    except ValueError:
        battery = 100

    low_ram = profile_name == "low_ram"
    low_power = battery < 20
    thermal_hot = False
    if thermal_str:
        try:
            thermal_hot = float(thermal_str) >= 60
        except ValueError:
            thermal_hot = thermal_str.lower() in {"hot", "critical"}

    constrained = low_ram or low_power or thermal_hot
    chunk = LOW_RESOURCE_CHUNK_SIZE if constrained else DEFAULT_CHUNK_SIZE
    transport = (
        TransportPreference.BITCHAT_PREFERRED
        if constrained
        else TransportPreference.BALANCED
    )
    rag_mode = "local" if constrained else "default"
    policy = {"chunk_size": chunk, "transport": transport, "rag_mode": rag_mode}
    logger.info(
        "Env policy: battery=%s, thermal=%s, profile=%s -> %s",
        battery_str,
        thermal_str or "n/a",
        profile_name,
        policy,
    )
    return policy


@dataclass
class ResourcePolicy:
    """Resource management policy configuration"""

    # Battery thresholds
    battery_critical: int = 10  # %
    battery_low: int = 20  # %
    battery_conservative: int = 40  # %

    # Thermal thresholds (Celsius)
    thermal_normal: float = 35.0
    thermal_warm: float = 45.0
    thermal_hot: float = 55.0
    thermal_critical: float = 65.0

    # Memory thresholds for chunk sizing
    memory_low_gb: float = 2.0
    memory_medium_gb: float = 4.0
    memory_high_gb: float = 8.0

    # Network data cost thresholds (MB/day)
    data_cost_low: int = 100
    data_cost_medium: int = 500
    data_cost_high: int = 1000


@dataclass
class ChunkingConfig:
    """Dynamic tensor/chunk sizing configuration"""

    base_chunk_size: int = 512
    max_chunk_size: int = 2048
    min_chunk_size: int = 64
    overlap_ratio: float = 0.1

    # Memory-aware scaling factors
    memory_scale_factor: float = 1.0
    thermal_scale_factor: float = 1.0
    battery_scale_factor: float = 1.0

    def effective_chunk_size(self) -> int:
        """Calculate effective chunk size based on scaling factors"""
        effective_size = int(
            self.base_chunk_size
            * self.memory_scale_factor
            * self.thermal_scale_factor
            * self.battery_scale_factor
        )
        return max(self.min_chunk_size, min(self.max_chunk_size, effective_size))


@dataclass
class ResourceState:
    """Current resource management state"""

    power_mode: PowerMode = PowerMode.BALANCED
    transport_preference: TransportPreference = TransportPreference.BALANCED
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig)
    active_policies: list[str] = field(default_factory=list)
    policy_changes: int = 0
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            "power_mode": self.power_mode.value,
            "transport_preference": self.transport_preference.value,
            "chunk_size": self.chunking_config.effective_chunk_size(),
            "active_policies": self.active_policies,
            "policy_changes": self.policy_changes,
            "last_update": self.last_update,
        }


class BatteryThermalResourceManager:
    """Battery/thermal-aware resource manager for mobile optimization

    Core Features:
    - BitChat-first routing under low battery conditions
    - Progressive thermal throttling with chunk size reduction
    - Memory-aware tensor chunking for 2-4GB devices
    - Network cost-aware transport selection
    - Real-time policy adaptation
    - Environment-driven simulation support

    Environment Variables for Testing:
    - AIV_MOBILE_PROFILE: low_ram, battery_save, thermal_throttle, balanced, performance
    - BATTERY: battery percentage (0-100)
    - THERMAL: thermal state (normal, warm, hot, critical) or temperature in Celsius
    - MEMORY_GB: available memory in GB
    - NETWORK_TYPE: wifi, cellular, 3g, 4g, 5g
    """

    def __init__(self, policy: ResourcePolicy | None = None):
        self.policy = policy or ResourcePolicy()
        self.allocator = ResourceAllocator()
        self.state = ResourceState()

        # Environment-driven simulation support
        self.env_simulation_mode = self._check_env_simulation()
        if self.env_simulation_mode:
            logger.info("Environment-driven simulation mode enabled")
            self._log_env_config()

        # Monitoring and statistics
        self.stats = {
            "policy_adaptations": 0,
            "transport_switches": 0,
            "thermal_throttles": 0,
            "battery_saves": 0,
            "chunk_adjustments": 0,
        }

        # History for trend analysis
        self.profile_history: list[DeviceProfile] = []
        self.decision_history: list[dict] = []

        logger.info("BatteryThermalResourceManager initialized")

    def _check_env_simulation(self) -> bool:
        """Check if environment-driven simulation mode is enabled"""
        env_vars = [
            "AIV_MOBILE_PROFILE",
            "BATTERY",
            "THERMAL",
            "MEMORY_GB",
            "NETWORK_TYPE",
        ]
        return any(os.getenv(var) is not None for var in env_vars)

    def _log_env_config(self) -> None:
        """Log current environment configuration"""
        config = {
            "AIV_MOBILE_PROFILE": os.getenv("AIV_MOBILE_PROFILE"),
            "BATTERY": os.getenv("BATTERY"),
            "THERMAL": os.getenv("THERMAL"),
            "MEMORY_GB": os.getenv("MEMORY_GB"),
            "NETWORK_TYPE": os.getenv("NETWORK_TYPE"),
        }
        active_vars = {k: v for k, v in config.items() if v is not None}
        logger.info(f"Environment simulation config: {active_vars}")

    def _create_env_profile(self) -> DeviceProfile:
        """Create DeviceProfile from environment variables for testing"""
        # Default values - use the correct DeviceProfile signature from device_profiler.py
        profile = DeviceProfile(
            timestamp=time.time(),
            cpu_percent=50.0,
            cpu_freq_mhz=2400.0,
            cpu_temp_celsius=35.0,
            cpu_cores=4,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            battery_percent=80,
            battery_charging=True,
            battery_temp_celsius=None,
            battery_health=None,
            network_type="wifi",
            network_bandwidth_mbps=None,
            network_latency_ms=50.0,
            storage_available_gb=50.0,
            storage_total_gb=100.0,
            gpu_available=False,
            gpu_memory_mb=None,
            thermal_state="normal",
            power_mode="balanced",
            screen_brightness=None,
            device_type="laptop",
        )

        # Override with environment variables
        if os.getenv("BATTERY"):
            try:
                battery_val = int(os.getenv("BATTERY"))
                profile.battery_percent = max(0, min(100, battery_val))
                # Assume not charging if battery specified (typical test scenario)
                profile.battery_charging = False
            except ValueError:
                logger.warning(f"Invalid BATTERY value: {os.getenv('BATTERY')}")

        if os.getenv("THERMAL"):
            thermal_val = os.getenv("THERMAL").lower()
            if thermal_val in ["normal", "warm", "hot", "critical"]:
                thermal_temps = {
                    "normal": 35.0,
                    "warm": 45.0,
                    "hot": 58.0,
                    "critical": 68.0,
                }
                profile.cpu_temp_celsius = thermal_temps[thermal_val]
            else:
                try:
                    # Try to parse as temperature in Celsius
                    temp_val = float(os.getenv("THERMAL"))
                    profile.cpu_temp_celsius = max(20.0, min(100.0, temp_val))
                except ValueError:
                    logger.warning(f"Invalid THERMAL value: {os.getenv('THERMAL')}")

        if os.getenv("MEMORY_GB"):
            try:
                memory_gb = float(os.getenv("MEMORY_GB"))
                profile.ram_total_mb = int(memory_gb * 1024)
                profile.ram_available_mb = int(memory_gb * 1024 * 0.7)  # 70% available
                profile.ram_used_mb = int(memory_gb * 1024 * 0.3)  # 30% used
            except ValueError:
                logger.warning(f"Invalid MEMORY_GB value: {os.getenv('MEMORY_GB')}")

        if os.getenv("NETWORK_TYPE"):
            network_type = os.getenv("NETWORK_TYPE").lower()
            if network_type in ["wifi", "cellular", "3g", "4g", "5g", "ethernet"]:
                profile.network_type = network_type
                # Set appropriate latency based on network type
                latency_map = {
                    "wifi": 20.0,
                    "ethernet": 10.0,
                    "cellular": 100.0,
                    "3g": 200.0,
                    "4g": 80.0,
                    "5g": 40.0,
                }
                profile.network_latency_ms = latency_map.get(network_type, 50.0)

        # Apply mobile profile presets
        mobile_profile = os.getenv("AIV_MOBILE_PROFILE", "").lower()
        if mobile_profile == "low_ram":
            profile.ram_total_mb = 2048
            profile.ram_available_mb = 1024
            profile.ram_used_mb = 1024
            profile.device_type = "phone"
        elif mobile_profile == "battery_save":
            profile.battery_percent = 15
            profile.battery_charging = False
            profile.device_type = "phone"
        elif mobile_profile == "thermal_throttle":
            profile.cpu_temp_celsius = 65.0
            profile.cpu_percent = 85.0
        elif mobile_profile == "performance":
            profile.battery_percent = 90
            profile.battery_charging = True
            profile.ram_total_mb = 8192
            profile.ram_available_mb = 6144
            profile.ram_used_mb = 2048
            profile.cpu_temp_celsius = 30.0

        return profile

    async def evaluate_and_adapt(
        self, profile: DeviceProfile | None = None
    ) -> ResourceState:
        """Main entry point: evaluate device state and adapt policies

        Args:
            profile: Current device profile with battery, thermal, memory state.
                    If None and env simulation is enabled, creates profile from env vars.

        Returns:
            Updated resource management state
        """
        # Use environment-driven profile if simulation mode is enabled and no profile provided
        if profile is None and self.env_simulation_mode:
            profile = self._create_env_profile()
        elif profile is None:
            # Create a default profile if none provided
            profile = DeviceProfile(
                timestamp=time.time(),
                cpu_percent=50.0,
                cpu_freq_mhz=2400.0,
                cpu_temp_celsius=35.0,
                cpu_cores=4,
                ram_used_mb=2048,
                ram_available_mb=2048,
                ram_total_mb=4096,
                battery_percent=80,
                battery_charging=True,
                battery_temp_celsius=None,
                battery_health=None,
                network_type="wifi",
                network_bandwidth_mbps=None,
                network_latency_ms=50.0,
                storage_available_gb=50.0,
                storage_total_gb=100.0,
                gpu_available=False,
                gpu_memory_mb=None,
                thermal_state="normal",
                power_mode="balanced",
                screen_brightness=None,
                device_type="laptop",
            )
        logger.debug(
            f"Evaluating resource state: battery={profile.battery_percent}%, "
            f"temp={profile.cpu_temp_celsius}°C, "
            f"memory={profile.ram_used_mb}/{profile.ram_total_mb}MB"
        )

        # Store profile for trend analysis
        self.profile_history.append(profile)
        if len(self.profile_history) > 50:
            self.profile_history = self.profile_history[-40:]  # Keep last 40

        # Evaluate power mode
        new_power_mode = self._evaluate_power_mode(profile)

        # Evaluate transport preference
        new_transport_pref = self._evaluate_transport_preference(profile)

        # Calculate chunking configuration
        new_chunking_config = self._calculate_chunking_config(profile)

        # Determine active policies
        active_policies = self._determine_active_policies(profile)

        # Check if state changed
        state_changed = (
            new_power_mode != self.state.power_mode
            or new_transport_pref != self.state.transport_preference
            or new_chunking_config.effective_chunk_size()
            != self.state.chunking_config.effective_chunk_size()
        )

        if state_changed:
            logger.info(
                f"Resource policy adaptation: {self.state.power_mode.value} → "
                f"{new_power_mode.value}, transport: "
                f"{self.state.transport_preference.value} → "
                f"{new_transport_pref.value}"
            )

            self.state.policy_changes += 1
            self.stats["policy_adaptations"] += 1

            if new_transport_pref != self.state.transport_preference:
                self.stats["transport_switches"] += 1

        # Update state
        self.state.power_mode = new_power_mode
        self.state.transport_preference = new_transport_pref
        self.state.chunking_config = new_chunking_config
        self.state.active_policies = active_policies
        self.state.last_update = time.time()

        # Record decision for analysis
        self.decision_history.append(
            {
                "timestamp": time.time(),
                "profile": profile.to_dict() if hasattr(profile, "to_dict") else {},
                "state": self.state.to_dict(),
                "reason": self._get_decision_reason(profile),
            }
        )

        # Limit decision history
        if len(self.decision_history) > 100:
            self.decision_history = self.decision_history[-80:]

        return self.state

    def _evaluate_power_mode(self, profile: DeviceProfile) -> PowerMode:
        """Evaluate appropriate power mode based on device state"""
        # Critical battery - always use critical mode
        if (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_critical
        ):
            self.stats["battery_saves"] += 1
            return PowerMode.CRITICAL

        # Critical thermal - always use critical mode
        if (
            profile.cpu_temp_celsius
            and profile.cpu_temp_celsius >= self.policy.thermal_critical
        ):
            self.stats["thermal_throttles"] += 1
            return PowerMode.CRITICAL

        # Hot thermal or low battery (not charging) - power save
        thermal_hot = (
            profile.cpu_temp_celsius
            and profile.cpu_temp_celsius >= self.policy.thermal_hot
        )
        battery_low_not_charging = (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_low
            and not profile.battery_charging
        )

        if thermal_hot or battery_low_not_charging:
            return PowerMode.POWER_SAVE

        # Warm thermal or conservative battery - balanced
        thermal_warm = (
            profile.cpu_temp_celsius
            and profile.cpu_temp_celsius >= self.policy.thermal_warm
        )
        battery_conservative = (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_conservative
            and not profile.battery_charging
        )

        if thermal_warm or battery_conservative:
            return PowerMode.BALANCED

        # Otherwise, performance mode is fine
        return PowerMode.PERFORMANCE

    def _evaluate_transport_preference(
        self, profile: DeviceProfile
    ) -> TransportPreference:
        """Evaluate transport preference based on battery, network costs, and performance needs"""
        # Critical battery - BitChat only (offline-first)
        if (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_critical
        ):
            return TransportPreference.BITCHAT_ONLY

        # Low battery not charging - prefer BitChat
        if (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_low
            and not profile.battery_charging
        ):
            return TransportPreference.BITCHAT_PREFERRED

        # Cellular network with potential data costs - prefer BitChat
        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            return TransportPreference.BITCHAT_PREFERRED

        # High latency network - prefer BitChat for better offline tolerance
        if profile.network_latency_ms and profile.network_latency_ms > 300:
            return TransportPreference.BITCHAT_PREFERRED

        # Good conditions - can use both transports
        if (
            profile.battery_percent
            and profile.battery_percent > self.policy.battery_conservative
            and profile.network_type in ["wifi", "ethernet"]
        ):
            return TransportPreference.BALANCED

        # Default to BitChat-preferred for mobile optimization
        return TransportPreference.BITCHAT_PREFERRED

    def _calculate_chunking_config(self, profile: DeviceProfile) -> ChunkingConfig:
        """Calculate optimal chunking configuration for current device state"""
        config = ChunkingConfig()

        # Memory-based scaling
        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.policy.memory_low_gb:
            config.memory_scale_factor = 0.5  # Smaller chunks for low memory
        elif available_gb <= self.policy.memory_medium_gb:
            config.memory_scale_factor = 0.75  # Moderate chunks
        elif available_gb <= self.policy.memory_high_gb:
            config.memory_scale_factor = 1.0  # Standard chunks
        else:
            config.memory_scale_factor = 1.25  # Larger chunks for high memory

        # Thermal-based scaling
        if profile.cpu_temp_celsius:
            if profile.cpu_temp_celsius >= self.policy.thermal_critical:
                config.thermal_scale_factor = 0.3  # Very small chunks
            elif profile.cpu_temp_celsius >= self.policy.thermal_hot:
                config.thermal_scale_factor = 0.5  # Small chunks
            elif profile.cpu_temp_celsius >= self.policy.thermal_warm:
                config.thermal_scale_factor = 0.75  # Moderate chunks
            else:
                config.thermal_scale_factor = 1.0  # Normal chunks

        # Battery-based scaling
        if profile.battery_percent:
            if profile.battery_percent <= self.policy.battery_critical:
                config.battery_scale_factor = 0.3  # Minimize processing
            elif profile.battery_percent <= self.policy.battery_low:
                config.battery_scale_factor = 0.6  # Reduce processing
            elif profile.battery_percent <= self.policy.battery_conservative:
                config.battery_scale_factor = 0.8  # Conservative processing
            else:
                config.battery_scale_factor = 1.0  # Normal processing

        # Track chunk adjustments
        old_size = self.state.chunking_config.effective_chunk_size()
        new_size = config.effective_chunk_size()
        if abs(new_size - old_size) > 32:  # Significant change
            self.stats["chunk_adjustments"] += 1

        return config

    def _determine_active_policies(self, profile: DeviceProfile) -> list[str]:
        """Determine which policies are currently active"""
        policies = []

        # Battery policies
        if profile.battery_percent:
            if profile.battery_percent <= self.policy.battery_critical:
                policies.append("battery_critical")
            elif profile.battery_percent <= self.policy.battery_low:
                policies.append("battery_low")
            elif profile.battery_percent <= self.policy.battery_conservative:
                policies.append("battery_conservative")

        # Thermal policies
        if profile.cpu_temp_celsius:
            if profile.cpu_temp_celsius >= self.policy.thermal_critical:
                policies.append("thermal_critical")
            elif profile.cpu_temp_celsius >= self.policy.thermal_hot:
                policies.append("thermal_hot")
            elif profile.cpu_temp_celsius >= self.policy.thermal_warm:
                policies.append("thermal_warm")

        # Memory policies
        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.policy.memory_low_gb:
            policies.append("memory_constrained")

        # Network policies
        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            policies.append("data_cost_aware")

        if profile.network_latency_ms and profile.network_latency_ms > 300:
            policies.append("high_latency_network")

        return policies

    def _get_decision_reason(self, profile: DeviceProfile) -> str:
        """Get human-readable reason for current policy decisions"""
        reasons = []

        if (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_critical
        ):
            reasons.append(f"critical_battery_{profile.battery_percent}%")
        elif (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_low
        ):
            reasons.append(f"low_battery_{profile.battery_percent}%")

        if (
            profile.cpu_temp_celsius
            and profile.cpu_temp_celsius >= self.policy.thermal_hot
        ):
            reasons.append(f"thermal_throttle_{profile.cpu_temp_celsius:.1f}C")

        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.policy.memory_low_gb:
            reasons.append(f"memory_constrained_{available_gb:.1f}GB")

        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            reasons.append("cellular_network")

        return "|".join(reasons) if reasons else "normal_operation"

    async def get_transport_routing_decision(
        self, message_size_bytes: int, priority: int = 5
    ) -> dict[str, Any]:
        """Get routing decision for a specific message

        Args:
            message_size_bytes: Size of message to route
            priority: Message priority (1=low, 10=urgent)

        Returns:
            Routing decision with transport preference and rationale
        """
        decision = {
            "primary_transport": "bitchat",
            "fallback_transport": "betanet",
            "chunk_size": self.state.chunking_config.effective_chunk_size(),
            "rationale": [],
            "estimated_cost": "low",
            "estimated_latency": "medium",
        }

        # Apply transport preference
        if self.state.transport_preference == TransportPreference.BITCHAT_ONLY:
            decision["primary_transport"] = "bitchat"
            decision["fallback_transport"] = None
            decision["rationale"].append("battery_critical_bitchat_only")

        elif self.state.transport_preference == TransportPreference.BITCHAT_PREFERRED:
            decision["primary_transport"] = "bitchat"
            decision["fallback_transport"] = "betanet"
            decision["rationale"].append("battery_aware_bitchat_preferred")

        elif self.state.transport_preference == TransportPreference.BALANCED:
            # For balanced, consider message size and priority
            if message_size_bytes > 10 * 1024:  # > 10KB
                decision["primary_transport"] = "betanet"
                decision["fallback_transport"] = "bitchat"
                decision["rationale"].append("large_message_betanet")
            elif priority >= 8:  # High priority
                decision["primary_transport"] = "betanet"
                decision["fallback_transport"] = "bitchat"
                decision["rationale"].append("high_priority_betanet")
            else:
                decision["primary_transport"] = "bitchat"
                decision["fallback_transport"] = "betanet"
                decision["rationale"].append("balanced_default_bitchat")

        # Adjust for power mode
        if self.state.power_mode in [PowerMode.CRITICAL, PowerMode.POWER_SAVE]:
            if decision["primary_transport"] == "betanet":
                decision["primary_transport"] = "bitchat"
                decision["rationale"].append("power_save_override")
                decision["estimated_cost"] = "very_low"
                decision["estimated_latency"] = "high"

        return decision

    def get_chunking_recommendations(self, data_type: str = "tensor") -> dict[str, Any]:
        """Get current chunking recommendations for different data types"""
        base_chunk_size = self.state.chunking_config.effective_chunk_size()

        recommendations = {
            "tensor": {
                "chunk_size": base_chunk_size,
                "overlap": int(
                    base_chunk_size * self.state.chunking_config.overlap_ratio
                ),
                "batch_size": max(1, base_chunk_size // 128),
            },
            "text": {
                "chunk_size": base_chunk_size,
                "overlap": max(16, base_chunk_size // 10),
                "max_tokens": base_chunk_size * 2,
            },
            "embedding": {
                "batch_size": max(8, base_chunk_size // 64),
                "chunk_size": base_chunk_size,
                "dimension_limit": 768 if base_chunk_size < 256 else 1536,
            },
        }

        # Apply power mode adjustments
        if self.state.power_mode == PowerMode.CRITICAL:
            for data_type_rec in recommendations.values():
                for key in data_type_rec:
                    if isinstance(data_type_rec[key], int):
                        data_type_rec[key] = max(1, data_type_rec[key] // 2)

        return recommendations.get(data_type, recommendations["tensor"])

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of resource management system"""
        status = {
            "state": self.state.to_dict(),
            "policy": {
                "battery_thresholds": {
                    "critical": self.policy.battery_critical,
                    "low": self.policy.battery_low,
                    "conservative": self.policy.battery_conservative,
                },
                "thermal_thresholds": {
                    "normal": self.policy.thermal_normal,
                    "warm": self.policy.thermal_warm,
                    "hot": self.policy.thermal_hot,
                    "critical": self.policy.thermal_critical,
                },
            },
            "statistics": self.stats.copy(),
            "profile_history_size": len(self.profile_history),
            "decision_history_size": len(self.decision_history),
        }
        status["power_mode"] = self.state.power_mode.value
        if self.profile_history:
            status["battery"] = self.profile_history[-1].battery_percent
        elif os.getenv("BATTERY") is not None:
            try:
                status["battery"] = int(os.getenv("BATTERY", "0"))
            except ValueError:
                status["battery"] = None
        return status

    async def reset_policies(self) -> None:
        """Reset to default policies (for testing/recovery)"""
        logger.info("Resetting resource management policies to defaults")
        self.state = ResourceState()
        self.stats = dict.fromkeys(self.stats, 0)
        self.profile_history.clear()
        self.decision_history.clear()
