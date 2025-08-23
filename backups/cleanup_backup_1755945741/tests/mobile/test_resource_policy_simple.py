"""Simple test for battery/thermal-aware resource policy without dependencies

Validates the core mobile optimization logic:
- Battery-aware transport selection
- Dynamic chunk sizing
- Transport preference logic
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest


class PowerMode(Enum):
    """Device power management modes"""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    CRITICAL = "critical"


class TransportPreference(Enum):
    """Transport selection preferences"""

    BITCHAT_ONLY = "bitchat_only"
    BITCHAT_PREFERRED = "bitchat_preferred"
    BALANCED = "balanced"
    BETANET_PREFERRED = "betanet_preferred"
    BETANET_ONLY = "betanet_only"


@dataclass
class MockDeviceProfile:
    """Mock device profile for testing"""

    battery_percent: int = 80
    battery_charging: bool = False
    cpu_temp_celsius: float = 35.0
    ram_total_mb: int = 4096
    ram_used_mb: int = 2048
    ram_available_mb: int = 2048
    network_type: str = "wifi"
    device_type: str = "phone"


@dataclass
class ChunkingConfig:
    """Mock chunking configuration"""

    base_chunk_size: int = 512
    memory_scale_factor: float = 1.0
    thermal_scale_factor: float = 1.0
    battery_scale_factor: float = 1.0

    def effective_chunk_size(self) -> int:
        effective_size = int(
            self.base_chunk_size * self.memory_scale_factor * self.thermal_scale_factor * self.battery_scale_factor
        )
        return max(64, min(2048, effective_size))


@dataclass
class ResourceState:
    """Mock resource state"""

    power_mode: PowerMode = PowerMode.BALANCED
    transport_preference: TransportPreference = TransportPreference.BALANCED
    chunking_config: ChunkingConfig = None
    active_policies: list = None

    def __post_init__(self):
        if self.chunking_config is None:
            self.chunking_config = ChunkingConfig()
        if self.active_policies is None:
            self.active_policies = []


class SimpleBatteryThermalResourceManager:
    """Simplified battery/thermal resource manager for testing"""

    def __init__(self):
        self.state = ResourceState()
        self.stats = {
            "policy_adaptations": 0,
            "transport_switches": 0,
            "thermal_throttles": 0,
            "battery_saves": 0,
            "chunk_adjustments": 0,
        }

        # Policy thresholds
        self.battery_critical = 10
        self.battery_low = 20
        self.thermal_hot = 55.0
        self.thermal_critical = 65.0
        self.memory_low_gb = 2.0

    async def evaluate_and_adapt(self, profile: MockDeviceProfile) -> ResourceState:
        """Evaluate device state and adapt policies"""

        # Determine power mode
        new_power_mode = self._evaluate_power_mode(profile)

        # Determine transport preference
        new_transport_pref = self._evaluate_transport_preference(profile)

        # Calculate chunking config
        new_chunking_config = self._calculate_chunking_config(profile)

        # Determine active policies
        active_policies = self._determine_active_policies(profile)

        # Update state
        old_power_mode = self.state.power_mode
        old_transport_pref = self.state.transport_preference

        self.state.power_mode = new_power_mode
        self.state.transport_preference = new_transport_pref
        self.state.chunking_config = new_chunking_config
        self.state.active_policies = active_policies

        # Track changes
        if old_power_mode != new_power_mode:
            self.stats["policy_adaptations"] += 1
        if old_transport_pref != new_transport_pref:
            self.stats["transport_switches"] += 1

        return self.state

    def _evaluate_power_mode(self, profile: MockDeviceProfile) -> PowerMode:
        """Evaluate power mode based on device state"""
        if profile.battery_percent <= self.battery_critical:
            self.stats["battery_saves"] += 1
            return PowerMode.CRITICAL

        if profile.cpu_temp_celsius >= self.thermal_critical:
            self.stats["thermal_throttles"] += 1
            return PowerMode.CRITICAL

        if profile.cpu_temp_celsius >= self.thermal_hot or (
            profile.battery_percent <= self.battery_low and not profile.battery_charging
        ):
            return PowerMode.POWER_SAVE

        if profile.device_type in ["phone", "tablet"]:
            return PowerMode.BALANCED

        return PowerMode.PERFORMANCE

    def _evaluate_transport_preference(self, profile: MockDeviceProfile) -> TransportPreference:
        """Evaluate transport preference"""
        if profile.battery_percent <= self.battery_critical:
            return TransportPreference.BITCHAT_ONLY

        if profile.battery_percent <= self.battery_low and not profile.battery_charging:
            return TransportPreference.BITCHAT_PREFERRED

        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            return TransportPreference.BITCHAT_PREFERRED

        return TransportPreference.BALANCED

    def _calculate_chunking_config(self, profile: MockDeviceProfile) -> ChunkingConfig:
        """Calculate chunking configuration"""
        config = ChunkingConfig()

        # Memory-based scaling
        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.memory_low_gb:
            config.memory_scale_factor = 0.5
        else:
            config.memory_scale_factor = 1.0

        # Thermal-based scaling
        if profile.cpu_temp_celsius >= self.thermal_critical:
            config.thermal_scale_factor = 0.3
        elif profile.cpu_temp_celsius >= self.thermal_hot:
            config.thermal_scale_factor = 0.5
        else:
            config.thermal_scale_factor = 1.0

        # Battery-based scaling
        if profile.battery_percent <= self.battery_critical:
            config.battery_scale_factor = 0.3
        elif profile.battery_percent <= self.battery_low:
            config.battery_scale_factor = 0.6
        else:
            config.battery_scale_factor = 1.0

        return config

    def _determine_active_policies(self, profile: MockDeviceProfile) -> list[str]:
        """Determine active policies"""
        policies = []

        if profile.battery_percent <= self.battery_critical:
            policies.append("battery_critical")
        elif profile.battery_percent <= self.battery_low:
            policies.append("battery_low")

        if profile.cpu_temp_celsius >= self.thermal_critical:
            policies.append("thermal_critical")
        elif profile.cpu_temp_celsius >= self.thermal_hot:
            policies.append("thermal_hot")

        available_gb = profile.ram_available_mb / 1024.0
        if available_gb <= self.memory_low_gb:
            policies.append("memory_constrained")

        if profile.network_type in ["cellular", "3g", "4g", "5g"]:
            policies.append("data_cost_aware")

        return policies

    async def get_transport_routing_decision(self, message_size_bytes: int, priority: int) -> dict[str, Any]:
        """Get routing decision for a message"""
        decision = {
            "primary_transport": "bitchat",
            "fallback_transport": "betanet",
            "chunk_size": self.state.chunking_config.effective_chunk_size(),
            "rationale": [],
            "estimated_cost": "low",
        }

        if self.state.transport_preference == TransportPreference.BITCHAT_ONLY:
            decision["primary_transport"] = "bitchat"
            decision["fallback_transport"] = None
            decision["rationale"].append("battery_critical_bitchat_only")
        elif self.state.transport_preference == TransportPreference.BITCHAT_PREFERRED:
            decision["primary_transport"] = "bitchat"
            decision["rationale"].append("battery_aware_bitchat_preferred")
        elif self.state.transport_preference == TransportPreference.BALANCED:
            if message_size_bytes > 10 * 1024:
                decision["primary_transport"] = "betanet"
                decision["rationale"].append("large_message_betanet")
            elif priority >= 8:
                decision["primary_transport"] = "betanet"
                decision["rationale"].append("high_priority_betanet")

        return decision

    def get_status(self) -> dict[str, Any]:
        """Get status"""
        return {
            "state": {
                "power_mode": self.state.power_mode.value,
                "transport_preference": self.state.transport_preference.value,
                "chunk_size": self.state.chunking_config.effective_chunk_size(),
                "active_policies": self.state.active_policies,
            },
            "statistics": self.stats.copy(),
        }


class TestSimpleResourcePolicy:
    """Test simplified resource policy implementation"""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing"""
        return SimpleBatteryThermalResourceManager()

    @pytest.mark.asyncio
    async def test_normal_conditions_balanced_mode(self, resource_manager):
        """Test resource management under normal conditions"""
        profile = MockDeviceProfile(
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            ram_available_mb=3072,
        )

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should use balanced or performance mode
        assert state.power_mode in [PowerMode.BALANCED, PowerMode.PERFORMANCE]

        # Should allow balanced transport usage
        assert state.transport_preference in [
            TransportPreference.BALANCED,
            TransportPreference.BITCHAT_PREFERRED,
        ]

        # Chunk size should be reasonable
        chunk_size = state.chunking_config.effective_chunk_size()
        assert 256 <= chunk_size <= 1024

        # Should not have critical policies active
        assert "battery_critical" not in state.active_policies
        assert "thermal_critical" not in state.active_policies

    @pytest.mark.asyncio
    async def test_low_battery_bitchat_preferred(self, resource_manager):
        """Test BitChat preference under low battery conditions"""
        profile = MockDeviceProfile(
            battery_percent=15,
            battery_charging=False,
            cpu_temp_celsius=35.0,
            ram_available_mb=3072,  # Low battery
        )

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should prefer power save mode
        assert state.power_mode in [PowerMode.POWER_SAVE, PowerMode.BALANCED]

        # Should prefer BitChat
        assert state.transport_preference == TransportPreference.BITCHAT_PREFERRED

        # Should have battery conservation policies
        assert "battery_low" in state.active_policies

        # Should reduce chunk sizes
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 512

    @pytest.mark.asyncio
    async def test_critical_battery_bitchat_only(self, resource_manager):
        """Test BitChat-only mode under critical battery"""
        profile = MockDeviceProfile(
            battery_percent=8,
            battery_charging=False,
            cpu_temp_celsius=35.0,
            ram_available_mb=3072,  # Critical battery
        )

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should use critical power mode
        assert state.power_mode == PowerMode.CRITICAL

        # Should use BitChat only
        assert state.transport_preference == TransportPreference.BITCHAT_ONLY

        # Should have critical battery policy
        assert "battery_critical" in state.active_policies

        # Should use very small chunks
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 256

    @pytest.mark.asyncio
    async def test_thermal_throttling(self, resource_manager):
        """Test thermal throttling reduces performance"""
        profile = MockDeviceProfile(
            battery_percent=80,
            cpu_temp_celsius=60.0,
            ram_available_mb=3072,  # High temperature
        )

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should throttle performance
        assert state.power_mode in [PowerMode.POWER_SAVE, PowerMode.CRITICAL]

        # Should have thermal policies
        assert any("thermal" in policy for policy in state.active_policies)

        # Should reduce chunk sizes
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 512

    @pytest.mark.asyncio
    async def test_memory_constrained_chunk_sizing(self, resource_manager):
        """Test chunk size reduction for memory-constrained devices"""
        profile = MockDeviceProfile(
            battery_percent=80,
            cpu_temp_celsius=35.0,
            ram_total_mb=2048,  # 2GB device
            ram_available_mb=512,  # Low available memory
        )

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should have memory constraint policies
        assert "memory_constrained" in state.active_policies

        # Should use smaller chunks
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size <= 256

    @pytest.mark.asyncio
    async def test_cellular_network_cost_awareness(self, resource_manager):
        """Test BitChat preference on cellular networks"""
        profile = MockDeviceProfile(battery_percent=50, network_type="cellular")

        state = await resource_manager.evaluate_and_adapt(profile)

        # Should prefer BitChat on cellular
        assert state.transport_preference == TransportPreference.BITCHAT_PREFERRED

        # Should have data cost awareness
        assert "data_cost_aware" in state.active_policies

    @pytest.mark.asyncio
    async def test_transport_routing_decision_small_message(self, resource_manager):
        """Test transport routing for small messages"""
        profile = MockDeviceProfile(battery_percent=80)
        await resource_manager.evaluate_and_adapt(profile)

        decision = await resource_manager.get_transport_routing_decision(1024, 5)

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert "rationale" in decision
        assert decision["chunk_size"] > 0

    @pytest.mark.asyncio
    async def test_transport_routing_decision_large_message(self, resource_manager):
        """Test transport routing for large messages"""
        profile = MockDeviceProfile(battery_percent=80)
        await resource_manager.evaluate_and_adapt(profile)

        decision = await resource_manager.get_transport_routing_decision(50 * 1024, 5)

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert "fallback_transport" in decision

    @pytest.mark.asyncio
    async def test_battery_critical_forces_bitchat_only(self, resource_manager):
        """Test critical battery forces BitChat-only routing"""
        profile = MockDeviceProfile(battery_percent=5)
        await resource_manager.evaluate_and_adapt(profile)

        decision = await resource_manager.get_transport_routing_decision(100 * 1024, 10)

        assert decision["primary_transport"] == "bitchat"
        assert decision["fallback_transport"] is None
        assert "battery_critical" in " ".join(decision["rationale"])

    @pytest.mark.asyncio
    async def test_policy_adaptation_tracking(self, resource_manager):
        """Test policy change tracking"""
        # Normal conditions
        profile1 = MockDeviceProfile(battery_percent=80)
        await resource_manager.evaluate_and_adapt(profile1)
        initial_adaptations = resource_manager.stats["policy_adaptations"]

        # Critical battery
        profile2 = MockDeviceProfile(battery_percent=5)
        await resource_manager.evaluate_and_adapt(profile2)

        # Should track the change
        assert resource_manager.stats["policy_adaptations"] > initial_adaptations

    def test_status_reporting(self, resource_manager):
        """Test status reporting"""
        status = resource_manager.get_status()

        assert "state" in status
        assert "statistics" in status
        assert "power_mode" in status["state"]
        assert "transport_preference" in status["state"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
