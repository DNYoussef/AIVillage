"""
Mobile Resource Manager - Battery/Thermal-Aware Resource Optimization

Consolidates mobile optimization functionality from:
- src/production/monitoring/mobile/resource_management.py (primary implementation)
- Edge device profiling and policy adaptation systems

This provides comprehensive mobile device optimization:
- Battery-aware transport selection (BitChat-first under low power)
- Thermal throttling with progressive limits
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Network cost-aware routing decisions
- Real-time policy adaptation
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import time
from typing import Any

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


@dataclass
class MobileDeviceProfile:
    """Mobile device state profile for optimization decisions"""

    timestamp: float
    device_id: str

    # Power state (required)
    battery_percent: int | None
    battery_charging: bool

    # Thermal state (required)
    cpu_temp_celsius: float | None

    # System resources (required)
    cpu_percent: float
    ram_used_mb: int
    ram_available_mb: int
    ram_total_mb: int

    # Fields with defaults
    power_mode: str = "balanced"
    thermal_state: str = "normal"
    network_type: str = "wifi"  # wifi, cellular, 3g, 4g, 5g
    network_latency_ms: float | None = None
    has_internet: bool = True
    is_metered_connection: bool = False
    is_foreground: bool = True
    screen_brightness: int | None = None
    device_type: str = "smartphone"


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
            self.base_chunk_size * self.memory_scale_factor * self.thermal_scale_factor * self.battery_scale_factor
        )
        return max(self.min_chunk_size, min(self.max_chunk_size, effective_size))


@dataclass
class ResourceOptimization:
    """Resource optimization recommendations"""

    power_mode: PowerMode
    transport_preference: TransportPreference
    chunking_config: ChunkingConfig

    # Compute limits
    cpu_limit_percent: float = 50.0
    memory_limit_mb: int = 1024
    max_concurrent_tasks: int = 2

    # Active policies
    active_policies: list[str] = field(default_factory=list)
    reasoning: str = ""

    # Performance predictions
    estimated_battery_impact: str = "medium"  # low, medium, high
    estimated_performance: str = "medium"  # low, medium, high
    estimated_latency: str = "medium"  # low, medium, high

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "power_mode": self.power_mode.value,
            "transport_preference": self.transport_preference.value,
            "chunk_size": self.chunking_config.effective_chunk_size(),
            "cpu_limit_percent": self.cpu_limit_percent,
            "memory_limit_mb": self.memory_limit_mb,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "active_policies": self.active_policies,
            "reasoning": self.reasoning,
            "estimated_battery_impact": self.estimated_battery_impact,
            "estimated_performance": self.estimated_performance,
            "estimated_latency": self.estimated_latency,
        }


class MobileResourceManager:
    """
    Battery/thermal-aware resource manager for mobile optimization

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

    def __init__(
        self, policy: ResourcePolicy | None = None, harvest_enabled: bool = True, token_rewards_enabled: bool = True
    ):
        self.policy = policy or ResourcePolicy()

        # Fog computing configuration
        self.harvest_enabled = harvest_enabled
        self.token_rewards_enabled = token_rewards_enabled

        # Harvesting state tracking
        self.harvesting_sessions: dict[str, dict[str, Any]] = {}
        self.contribution_metrics: dict[str, dict[str, float]] = {}

        # P2P and marketplace integration
        self.p2p_coordinator = None
        self.marketplace_client = None

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
            "decisions_made": 0,
        }

        # History for trend analysis
        self.profile_history: list[MobileDeviceProfile] = []
        self.optimization_history: list[ResourceOptimization] = []

        logger.info(
            f"Mobile Resource Manager initialized: "
            f"harvest_enabled={harvest_enabled}, token_rewards={token_rewards_enabled}"
        )

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

    def create_device_profile_from_env(self, device_id: str = "test_device") -> MobileDeviceProfile:
        """Create MobileDeviceProfile from environment variables for testing"""

        # Default values
        profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id=device_id,
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            cpu_percent=50.0,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            network_type="wifi",
            network_latency_ms=50.0,
            device_type="smartphone",
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
                profile.thermal_state = thermal_val
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
                profile.is_metered_connection = network_type in ["cellular", "3g", "4g", "5g"]

        # Apply mobile profile presets
        mobile_profile = os.getenv("AIV_MOBILE_PROFILE", "").lower()
        if mobile_profile == "low_ram":
            profile.ram_total_mb = 2048
            profile.ram_available_mb = 1024
            profile.ram_used_mb = 1024
            profile.device_type = "smartphone"
        elif mobile_profile == "battery_save":
            profile.battery_percent = 15
            profile.battery_charging = False
            profile.power_mode = "power_save"
            profile.device_type = "smartphone"
        elif mobile_profile == "thermal_throttle":
            profile.cpu_temp_celsius = 65.0
            profile.thermal_state = "hot"
            profile.cpu_percent = 85.0
        elif mobile_profile == "performance":
            profile.battery_percent = 90
            profile.battery_charging = True
            profile.ram_total_mb = 8192
            profile.ram_available_mb = 6144
            profile.ram_used_mb = 2048
            profile.cpu_temp_celsius = 30.0
            profile.power_mode = "performance"

        return profile

    async def optimize_for_device(self, profile: MobileDeviceProfile | None = None) -> ResourceOptimization:
        """
        Main optimization entry point

        Args:
            profile: Current device profile. If None and env simulation is enabled,
                    creates profile from environment variables.

        Returns:
            Optimized resource configuration
        """

        # Use environment-driven profile if simulation mode is enabled and no profile provided
        if profile is None and self.env_simulation_mode:
            profile = self.create_device_profile_from_env()
        elif profile is None:
            # Create a reasonable default profile
            profile = MobileDeviceProfile(
                timestamp=time.time(),
                device_id="default_device",
                battery_percent=80,
                battery_charging=True,
                cpu_temp_celsius=35.0,
                cpu_percent=50.0,
                ram_used_mb=2048,
                ram_available_mb=2048,
                ram_total_mb=4096,
                network_type="wifi",
                network_latency_ms=50.0,
                device_type="smartphone",
            )

        logger.debug(
            f"Optimizing for device: battery={profile.battery_percent}%, "
            f"temp={profile.cpu_temp_celsius}°C, "
            f"memory={profile.ram_used_mb}/{profile.ram_total_mb}MB"
        )

        # Store profile for trend analysis
        self.profile_history.append(profile)
        if len(self.profile_history) > 50:
            self.profile_history = self.profile_history[-40:]  # Keep last 40

        # Evaluate power mode
        power_mode = self._evaluate_power_mode(profile)

        # Evaluate transport preference
        transport_preference = self._evaluate_transport_preference(profile)

        # Calculate chunking configuration
        chunking_config = self._calculate_chunking_config(profile)

        # Calculate compute limits
        cpu_limit, memory_limit, max_tasks = self._calculate_compute_limits(profile)

        # Determine active policies
        active_policies = self._determine_active_policies(profile)

        # Generate reasoning
        reasoning = self._generate_reasoning(profile, active_policies)

        # Estimate performance characteristics
        battery_impact, performance, latency = self._estimate_performance(power_mode, transport_preference, profile)

        # Create optimization result
        optimization = ResourceOptimization(
            power_mode=power_mode,
            transport_preference=transport_preference,
            chunking_config=chunking_config,
            cpu_limit_percent=cpu_limit,
            memory_limit_mb=memory_limit,
            max_concurrent_tasks=max_tasks,
            active_policies=active_policies,
            reasoning=reasoning,
            estimated_battery_impact=battery_impact,
            estimated_performance=performance,
            estimated_latency=latency,
        )

        # Store for analysis
        self.optimization_history.append(optimization)
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-80:]

        self.stats["decisions_made"] += 1

        return optimization

    def _evaluate_power_mode(self, profile: MobileDeviceProfile) -> PowerMode:
        """Evaluate appropriate power mode based on device state"""

        # Critical battery - always use critical mode
        if profile.battery_percent is not None and profile.battery_percent <= self.policy.battery_critical:
            self.stats["battery_saves"] += 1
            return PowerMode.CRITICAL

        # Critical thermal - always use critical mode
        if profile.cpu_temp_celsius is not None and profile.cpu_temp_celsius >= self.policy.thermal_critical:
            self.stats["thermal_throttles"] += 1
            return PowerMode.CRITICAL

        # Hot thermal or low battery (not charging) - power save
        thermal_hot = profile.cpu_temp_celsius is not None and profile.cpu_temp_celsius >= self.policy.thermal_hot
        battery_low_not_charging = (
            profile.battery_percent is not None
            and profile.battery_percent <= self.policy.battery_low
            and not profile.battery_charging
        )

        if thermal_hot or battery_low_not_charging:
            return PowerMode.POWER_SAVE

        # Warm thermal or conservative battery - balanced
        thermal_warm = profile.cpu_temp_celsius is not None and profile.cpu_temp_celsius >= self.policy.thermal_warm
        battery_conservative = (
            profile.battery_percent is not None
            and profile.battery_percent <= self.policy.battery_conservative
            and not profile.battery_charging
        )

        if thermal_warm or battery_conservative:
            return PowerMode.BALANCED

        # Otherwise, performance mode is fine
        return PowerMode.PERFORMANCE

    def _evaluate_transport_preference(self, profile: MobileDeviceProfile) -> TransportPreference:
        """Evaluate transport preference based on battery, network costs, and performance needs"""

        # Critical battery - BitChat only (offline-first)
        if profile.battery_percent is not None and profile.battery_percent <= self.policy.battery_critical:
            return TransportPreference.BITCHAT_ONLY

        # Low battery not charging - prefer BitChat
        if (
            profile.battery_percent is not None
            and profile.battery_percent <= self.policy.battery_low
            and not profile.battery_charging
        ):
            return TransportPreference.BITCHAT_PREFERRED

        # Cellular network with potential data costs - prefer BitChat
        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            return TransportPreference.BITCHAT_PREFERRED

        # High latency network - prefer BitChat for better offline tolerance
        if profile.network_latency_ms is not None and profile.network_latency_ms > 300:
            return TransportPreference.BITCHAT_PREFERRED

        # Good conditions - can use both transports
        if (
            profile.battery_percent is not None
            and profile.battery_percent > self.policy.battery_conservative
            and profile.network_type in ["wifi", "ethernet"]
        ):
            return TransportPreference.BALANCED

        # Default to BitChat-preferred for mobile optimization
        return TransportPreference.BITCHAT_PREFERRED

    def _calculate_chunking_config(self, profile: MobileDeviceProfile) -> ChunkingConfig:
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
        if profile.cpu_temp_celsius is not None:
            if profile.cpu_temp_celsius >= self.policy.thermal_critical:
                config.thermal_scale_factor = 0.3  # Very small chunks
            elif profile.cpu_temp_celsius >= self.policy.thermal_hot:
                config.thermal_scale_factor = 0.5  # Small chunks
            elif profile.cpu_temp_celsius >= self.policy.thermal_warm:
                config.thermal_scale_factor = 0.75  # Moderate chunks
            else:
                config.thermal_scale_factor = 1.0  # Normal chunks

        # Battery-based scaling
        if profile.battery_percent is not None:
            if profile.battery_percent <= self.policy.battery_critical:
                config.battery_scale_factor = 0.3  # Minimize processing
            elif profile.battery_percent <= self.policy.battery_low:
                config.battery_scale_factor = 0.6  # Reduce processing
            elif profile.battery_percent <= self.policy.battery_conservative:
                config.battery_scale_factor = 0.8  # Conservative processing
            else:
                config.battery_scale_factor = 1.0  # Normal processing

        # Track chunk adjustments
        if self.optimization_history:
            old_size = self.optimization_history[-1].chunking_config.effective_chunk_size()
            new_size = config.effective_chunk_size()
            if abs(new_size - old_size) > 32:  # Significant change
                self.stats["chunk_adjustments"] += 1

        return config

    def _calculate_compute_limits(self, profile: MobileDeviceProfile) -> tuple[float, int, int]:
        """Calculate CPU/memory limits and max concurrent tasks"""

        # Base limits
        cpu_limit = 50.0
        memory_limit = min(1024, profile.ram_available_mb // 2)
        max_tasks = 2

        # Battery adjustments
        if profile.battery_percent is not None:
            if profile.battery_percent <= self.policy.battery_critical:
                cpu_limit = 20.0
                memory_limit = min(256, memory_limit)
                max_tasks = 1
            elif profile.battery_percent <= self.policy.battery_low:
                cpu_limit = 35.0
                memory_limit = int(memory_limit * 0.7)
                max_tasks = 1

        # Thermal adjustments
        if profile.cpu_temp_celsius is not None:
            if profile.cpu_temp_celsius >= self.policy.thermal_critical:
                cpu_limit = min(cpu_limit, 15.0)
                memory_limit = min(memory_limit, 256)
                max_tasks = 1
            elif profile.cpu_temp_celsius >= self.policy.thermal_hot:
                cpu_limit = min(cpu_limit, 30.0)
                memory_limit = int(memory_limit * 0.6)

        # Memory constraints
        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.policy.memory_low_gb:
            memory_limit = min(memory_limit, 512)
            max_tasks = 1

        return cpu_limit, memory_limit, max_tasks

    def _determine_active_policies(self, profile: MobileDeviceProfile) -> list[str]:
        """Determine which policies are currently active"""

        policies = []

        # Battery policies
        if profile.battery_percent is not None:
            if profile.battery_percent <= self.policy.battery_critical:
                policies.append("battery_critical")
            elif profile.battery_percent <= self.policy.battery_low:
                policies.append("battery_low")
            elif profile.battery_percent <= self.policy.battery_conservative:
                policies.append("battery_conservative")

        # Thermal policies
        if profile.cpu_temp_celsius is not None:
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

        if profile.network_latency_ms is not None and profile.network_latency_ms > 300:
            policies.append("high_latency_network")

        return policies

    def _generate_reasoning(self, profile: MobileDeviceProfile, policies: list[str]) -> str:
        """Generate human-readable reasoning for optimization decisions"""

        reasons = []

        if "battery_critical" in policies:
            reasons.append(f"critical_battery_{profile.battery_percent}%")
        elif "battery_low" in policies:
            reasons.append(f"low_battery_{profile.battery_percent}%")

        if "thermal_critical" in policies or "thermal_hot" in policies:
            reasons.append(f"thermal_throttle_{profile.cpu_temp_celsius:.1f}C")

        if "memory_constrained" in policies:
            available_gb = profile.ram_available_mb / 1024.0
            reasons.append(f"memory_constrained_{available_gb:.1f}GB")

        if "data_cost_aware" in policies:
            reasons.append("cellular_network")

        return " | ".join(reasons) if reasons else "normal_operation"

    def _estimate_performance(
        self, power_mode: PowerMode, transport_preference: TransportPreference, profile: MobileDeviceProfile
    ) -> tuple[str, str, str]:
        """Estimate battery impact, performance, and latency characteristics"""

        # Battery impact estimation
        if power_mode in [PowerMode.CRITICAL, PowerMode.POWER_SAVE]:
            battery_impact = "low"
        elif transport_preference == TransportPreference.BITCHAT_PREFERRED:
            battery_impact = "low"
        elif profile.battery_charging:
            battery_impact = "medium"
        else:
            battery_impact = "medium"

        # Performance estimation
        if power_mode == PowerMode.PERFORMANCE:
            performance = "high"
        elif power_mode == PowerMode.CRITICAL:
            performance = "low"
        else:
            performance = "medium"

        # Latency estimation
        if transport_preference == TransportPreference.BITCHAT_ONLY:
            latency = "high"  # Store-and-forward has higher latency
        elif profile.network_type in ["5g", "wifi"]:
            latency = "low"
        else:
            latency = "medium"

        return battery_impact, performance, latency

    async def get_transport_routing_decision(
        self, message_size_bytes: int, priority: int = 5, profile: MobileDeviceProfile | None = None
    ) -> dict[str, Any]:
        """Get routing decision for a specific message"""

        # Get current optimization
        optimization = await self.optimize_for_device(profile)

        decision = {
            "primary_transport": "bitchat",
            "fallback_transport": "betanet",
            "chunk_size": optimization.chunking_config.effective_chunk_size(),
            "rationale": [],
            "estimated_cost": "low",
            "estimated_latency": optimization.estimated_latency,
        }

        # Apply transport preference
        if optimization.transport_preference == TransportPreference.BITCHAT_ONLY:
            decision["primary_transport"] = "bitchat"
            decision["fallback_transport"] = None
            decision["rationale"].append("battery_critical_bitchat_only")

        elif optimization.transport_preference == TransportPreference.BITCHAT_PREFERRED:
            decision["primary_transport"] = "bitchat"
            decision["fallback_transport"] = "betanet"
            decision["rationale"].append("battery_aware_bitchat_preferred")

        elif optimization.transport_preference == TransportPreference.BALANCED:
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

        # Override for critical power mode
        if optimization.power_mode in [PowerMode.CRITICAL, PowerMode.POWER_SAVE]:
            if decision["primary_transport"] == "betanet":
                decision["primary_transport"] = "bitchat"
                decision["rationale"].append("power_save_override")
                decision["estimated_cost"] = "very_low"

        return decision

    def get_chunking_recommendations(self, data_type: str = "tensor") -> dict[str, Any]:
        """Get current chunking recommendations for different data types"""

        if not self.optimization_history:
            # Return default recommendations
            base_chunk_size = 512
        else:
            base_chunk_size = self.optimization_history[-1].chunking_config.effective_chunk_size()

        recommendations = {
            "tensor": {
                "chunk_size": base_chunk_size,
                "overlap": int(base_chunk_size * 0.1),
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

        # Apply critical mode adjustments
        if self.optimization_history and self.optimization_history[-1].power_mode == PowerMode.CRITICAL:
            for data_type_rec in recommendations.values():
                for key in data_type_rec:
                    if isinstance(data_type_rec[key], int):
                        data_type_rec[key] = max(1, data_type_rec[key] // 2)

        return recommendations.get(data_type, recommendations["tensor"])

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive status of mobile resource management"""

        current_optimization = None
        if self.optimization_history:
            current_optimization = self.optimization_history[-1].to_dict()

        return {
            "current_optimization": current_optimization,
            "policy_thresholds": {
                "battery": {
                    "critical": self.policy.battery_critical,
                    "low": self.policy.battery_low,
                    "conservative": self.policy.battery_conservative,
                },
                "thermal": {
                    "normal": self.policy.thermal_normal,
                    "warm": self.policy.thermal_warm,
                    "hot": self.policy.thermal_hot,
                    "critical": self.policy.thermal_critical,
                },
                "memory": {
                    "low_gb": self.policy.memory_low_gb,
                    "medium_gb": self.policy.memory_medium_gb,
                    "high_gb": self.policy.memory_high_gb,
                },
            },
            "statistics": self.stats.copy(),
            "history_size": {
                "profiles": len(self.profile_history),
                "optimizations": len(self.optimization_history),
            },
            "environment_simulation": self.env_simulation_mode,
        }

    async def reset(self) -> None:
        """Reset to default state (for testing/recovery)"""
        logger.info("Resetting mobile resource manager to defaults")
        self.stats = dict.fromkeys(self.stats, 0)
        self.profile_history.clear()
        self.optimization_history.clear()

        # Reset fog computing state
        self.harvesting_sessions.clear()
        self.contribution_metrics.clear()

    # ============================================================================
    # FOG COMPUTING METHODS - Idle Resource Harvesting
    # ============================================================================

    async def evaluate_harvest_eligibility(self, device_id: str, profile: MobileDeviceProfile) -> bool:
        """Evaluate if device is eligible for compute harvesting"""

        if not self.harvest_enabled:
            return False

        # Check battery level and charging status
        if profile.battery_percent is None or profile.battery_percent < 20:
            logger.debug(f"Device {device_id} battery too low: {profile.battery_percent}%")
            return False

        if not profile.battery_charging:
            logger.debug(f"Device {device_id} not charging")
            return False

        # Check thermal state
        if profile.cpu_temp_celsius and profile.cpu_temp_celsius > 45.0:
            logger.debug(f"Device {device_id} too hot: {profile.cpu_temp_celsius}°C")
            return False

        # Check network - prefer WiFi for fog computing
        if profile.network_type not in ["wifi", "ethernet"]:
            logger.debug(f"Device {device_id} on metered network: {profile.network_type}")
            return False

        # Check if user is actively using device
        if profile.is_foreground and profile.screen_brightness and profile.screen_brightness > 10:
            logger.debug(f"Device {device_id} in active use")
            return False

        logger.info(f"Device {device_id} eligible for compute harvesting")
        return True

    async def start_harvest_session(self, device_id: str, profile: MobileDeviceProfile) -> str | None:
        """Start a compute harvesting session"""

        if not await self.evaluate_harvest_eligibility(device_id, profile):
            return None

        # Don't start if already harvesting
        if device_id in self.harvesting_sessions:
            return self.harvesting_sessions[device_id]["session_id"]

        session_id = f"harvest_{device_id}_{int(time.time())}"

        session = {
            "session_id": session_id,
            "device_id": device_id,
            "start_time": time.time(),
            "initial_profile": profile.to_dict() if hasattr(profile, "to_dict") else vars(profile),
            "cpu_cycles_contributed": 0,
            "memory_mb_hours_contributed": 0.0,
            "bandwidth_gb_contributed": 0.0,
            "tasks_completed": 0,
            "tokens_earned": 0,
            "status": "active",
        }

        self.harvesting_sessions[device_id] = session

        # Initialize contribution metrics if not exists
        if device_id not in self.contribution_metrics:
            self.contribution_metrics[device_id] = {
                "total_sessions": 0,
                "total_hours": 0.0,
                "total_cpu_cycles": 0,
                "total_tokens_earned": 0,
                "average_performance": 0.0,
            }

        logger.info(
            f"Started harvest session {session_id} for device {device_id}: "
            f"battery {profile.battery_percent}%, temp {profile.cpu_temp_celsius}°C"
        )

        return session_id

    async def update_harvest_metrics(self, device_id: str, metrics: dict[str, Any]) -> bool:
        """Update metrics for an active harvesting session"""

        if device_id not in self.harvesting_sessions:
            logger.warning(f"No active harvest session for device {device_id}")
            return False

        session = self.harvesting_sessions[device_id]

        # Update session metrics
        session["cpu_cycles_contributed"] += metrics.get("cpu_cycles", 0)
        session["memory_mb_hours_contributed"] += metrics.get("memory_mb_hours", 0.0)
        session["bandwidth_gb_contributed"] += metrics.get("bandwidth_gb", 0.0)
        session["tasks_completed"] += metrics.get("tasks_completed", 0)

        # Calculate token rewards based on contribution
        duration_hours = (time.time() - session["start_time"]) / 3600
        base_tokens = int(duration_hours * 10)  # 10 tokens per hour base rate

        # Bonus for completing tasks
        task_bonus = session["tasks_completed"] * 5  # 5 tokens per task

        # Performance multiplier based on device capabilities
        performance_multiplier = 1.0
        if metrics.get("cpu_cores", 0) > 4:
            performance_multiplier += 0.2  # 20% bonus for multi-core devices
        if metrics.get("has_gpu", False):
            performance_multiplier += 0.3  # 30% bonus for GPU devices

        session["tokens_earned"] = int((base_tokens + task_bonus) * performance_multiplier)

        logger.debug(
            f"Updated harvest session {session['session_id']}: "
            f"{session['tasks_completed']} tasks, {session['tokens_earned']} tokens"
        )

        return True

    async def stop_harvest_session(self, device_id: str, reason: str = "conditions_changed") -> dict[str, Any] | None:
        """Stop a harvesting session and finalize contributions"""

        if device_id not in self.harvesting_sessions:
            return None

        session = self.harvesting_sessions[device_id]
        session["end_time"] = time.time()
        session["duration_hours"] = (session["end_time"] - session["start_time"]) / 3600
        session["stop_reason"] = reason
        session["status"] = "completed"

        # Update cumulative contribution metrics
        metrics = self.contribution_metrics[device_id]
        metrics["total_sessions"] += 1
        metrics["total_hours"] += session["duration_hours"]
        metrics["total_cpu_cycles"] += session["cpu_cycles_contributed"]
        metrics["total_tokens_earned"] += session["tokens_earned"]

        # Calculate average performance
        if metrics["total_hours"] > 0:
            metrics["average_performance"] = (
                metrics["total_cpu_cycles"] / metrics["total_hours"] / 1000000000  # GHz equivalent
            )

        # Archive session data
        final_session = session.copy()
        del self.harvesting_sessions[device_id]

        logger.info(
            f"Stopped harvest session {session['session_id']}: "
            f"{session['duration_hours']:.2f} hours, "
            f"{session['tasks_completed']} tasks, "
            f"{session['tokens_earned']} tokens earned"
        )

        return final_session

    def get_harvest_stats(self, device_id: str | None = None) -> dict[str, Any]:
        """Get harvesting statistics for device or all devices"""

        if device_id:
            # Stats for specific device
            if device_id not in self.contribution_metrics:
                return {"error": "Device not found"}

            stats = self.contribution_metrics[device_id].copy()

            # Add current session info if active
            if device_id in self.harvesting_sessions:
                session = self.harvesting_sessions[device_id]
                stats["current_session"] = {
                    "session_id": session["session_id"],
                    "duration_hours": (time.time() - session["start_time"]) / 3600,
                    "tasks_completed": session["tasks_completed"],
                    "tokens_earned": session["tokens_earned"],
                }

            return stats

        else:
            # Aggregate stats for all devices
            total_devices = len(self.contribution_metrics)
            active_sessions = len(self.harvesting_sessions)

            total_hours = sum(m["total_hours"] for m in self.contribution_metrics.values())
            total_tokens = sum(m["total_tokens_earned"] for m in self.contribution_metrics.values())
            total_sessions = sum(m["total_sessions"] for m in self.contribution_metrics.values())

            return {
                "total_devices": total_devices,
                "active_sessions": active_sessions,
                "total_contribution_hours": total_hours,
                "total_tokens_earned": total_tokens,
                "total_sessions": total_sessions,
                "average_tokens_per_hour": total_tokens / max(total_hours, 1),
                "harvest_enabled": self.harvest_enabled,
            }

    async def set_p2p_coordinator(self, coordinator):
        """Set P2P coordinator for task distribution"""
        self.p2p_coordinator = coordinator
        logger.info("P2P coordinator connected for fog task distribution")

    async def set_marketplace_client(self, client):
        """Set marketplace client for service offerings"""
        self.marketplace_client = client
        logger.info("Marketplace client connected for fog services")

    async def register_as_fog_provider(self, device_capabilities: dict[str, Any]) -> bool:
        """Register device as fog computing provider in marketplace"""

        if not self.marketplace_client:
            logger.warning("No marketplace client available")
            return False

        try:
            # Create service offering based on device capabilities
            offering = {
                "service_type": "compute_instance",
                "service_tier": "basic",
                "cpu_cores": device_capabilities.get("cpu_cores", 2),
                "memory_gb": device_capabilities.get("ram_total_mb", 4096) / 1024,
                "gpu_available": device_capabilities.get("has_gpu", False),
                "base_price": 0.1,  # 0.1 tokens per hour
                "regions": ["mobile_fog"],
                "uptime_guarantee": 95.0,  # 95% uptime for mobile devices
            }

            success = await self.marketplace_client.register_offering(offering)

            if success:
                logger.info("Successfully registered as fog computing provider")
            else:
                logger.error("Failed to register as fog computing provider")

            return success

        except Exception as e:
            logger.error(f"Error registering as fog provider: {e}")
            return False
