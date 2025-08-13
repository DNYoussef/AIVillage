#!/usr/bin/env python3
"""Simplified mobile policy tests using mock DeviceProfile.

Tests the P2 Mobile Resource Optimization features:
- Battery-aware transport selection (BitChat-first under low power)
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Thermal throttling with progressive limits
- Network cost-aware routing decisions
- Real-time policy adaptation
"""

import asyncio
import os
import sys
from dataclasses import dataclass

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from src.production.monitoring.mobile.resource_management import (
    ChunkingConfig,
    PowerMode,
    ResourcePolicy,
    ResourceState,
    TransportPreference,
)


@dataclass
class MockDeviceProfile:
    """Mock DeviceProfile that matches our needs"""

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


class SimpleBatteryThermalResourceManager:
    """Simplified resource manager for testing"""

    def __init__(self, policy: ResourcePolicy | None = None):
        self.policy = policy or ResourcePolicy()
        self.state = ResourceState()

        # Environment-driven simulation support
        self.env_simulation_mode = self._check_env_simulation()
        if self.env_simulation_mode:
            self._log_env_config()

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
        print(f"Environment simulation config: {active_vars}")

    def _create_env_profile(self) -> MockDeviceProfile:
        """Create DeviceProfile from environment variables for testing"""
        import time

        # Default values
        profile = MockDeviceProfile(
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
        )

        # Override with environment variables
        if os.getenv("BATTERY"):
            try:
                battery_val = int(os.getenv("BATTERY"))
                profile.battery_percent = max(0, min(100, battery_val))
                profile.battery_charging = (
                    False  # Assume not charging if battery specified
                )
            except ValueError:
                pass

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
                    temp_val = float(os.getenv("THERMAL"))
                    profile.cpu_temp_celsius = max(20.0, min(100.0, temp_val))
                except ValueError:
                    pass

        if os.getenv("MEMORY_GB"):
            try:
                memory_gb = float(os.getenv("MEMORY_GB"))
                profile.ram_total_mb = int(memory_gb * 1024)
                profile.ram_available_mb = int(memory_gb * 1024 * 0.7)  # 70% available
                profile.ram_used_mb = int(memory_gb * 1024 * 0.3)  # 30% used
            except ValueError:
                pass

        if os.getenv("NETWORK_TYPE"):
            network_type = os.getenv("NETWORK_TYPE").lower()
            if network_type in ["wifi", "cellular", "3g", "4g", "5g", "ethernet"]:
                profile.network_type = network_type
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
        self, profile: MockDeviceProfile | None = None
    ) -> ResourceState:
        """Evaluate device state and adapt policies"""
        # Use environment-driven profile if simulation mode is enabled and no profile provided
        if profile is None and self.env_simulation_mode:
            profile = self._create_env_profile()
        elif profile is None:
            import time

            profile = MockDeviceProfile(
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
            )

        # Evaluate power mode
        new_power_mode = self._evaluate_power_mode(profile)

        # Evaluate transport preference
        new_transport_pref = self._evaluate_transport_preference(profile)

        # Calculate chunking configuration
        new_chunking_config = self._calculate_chunking_config(profile)

        # Determine active policies
        active_policies = self._determine_active_policies(profile)

        # Update state
        self.state.power_mode = new_power_mode
        self.state.transport_preference = new_transport_pref
        self.state.chunking_config = new_chunking_config
        self.state.active_policies = active_policies

        return self.state

    def _evaluate_power_mode(self, profile: MockDeviceProfile) -> PowerMode:
        """Evaluate appropriate power mode based on device state"""
        # Critical battery - always use critical mode
        if (
            profile.battery_percent
            and profile.battery_percent <= self.policy.battery_critical
        ):
            return PowerMode.CRITICAL

        # Critical thermal - always use critical mode
        if (
            profile.cpu_temp_celsius
            and profile.cpu_temp_celsius >= self.policy.thermal_critical
        ):
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
        self, profile: MockDeviceProfile
    ) -> TransportPreference:
        """Evaluate transport preference based on battery, network costs, and performance needs"""
        # Critical battery - BitChat only
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

        # High latency network - prefer BitChat
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

    def _calculate_chunking_config(self, profile: MockDeviceProfile) -> ChunkingConfig:
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

        return config

    def _determine_active_policies(self, profile: MockDeviceProfile) -> list[str]:
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

    async def get_transport_routing_decision(
        self, message_size_bytes: int, priority: int = 5
    ) -> dict[str, any]:
        """Get routing decision for a specific message"""
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

        return decision

    def get_chunking_recommendations(self, data_type: str = "tensor") -> dict[str, any]:
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


if __name__ == "__main__":
    print("Testing SimpleBatteryThermalResourceManager...")

    async def test_simple():
        # Test environment-driven mode
        os.environ["BATTERY"] = "15"
        os.environ["THERMAL"] = "hot"

        manager = SimpleBatteryThermalResourceManager()
        print(f"Env simulation mode: {manager.env_simulation_mode}")

        state = await manager.evaluate_and_adapt()
        print(f"Power mode: {state.power_mode.value}")
        print(f"Transport preference: {state.transport_preference.value}")
        print(f"Chunk size: {state.chunking_config.effective_chunk_size()}")
        print(f"Active policies: {state.active_policies}")

        decision = await manager.get_transport_routing_decision(1024, 5)
        print(f"Transport decision: {decision['primary_transport']}")

    asyncio.run(test_simple())
