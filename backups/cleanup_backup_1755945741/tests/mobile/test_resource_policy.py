"""Test battery/thermal-aware resource policy for mobile optimization

Tests P2 Mobile Resource Optimization implementation:
- Battery-aware transport selection (BitChat-first under low power)
- Thermal throttling with progressive limits
- Dynamic tensor/chunk size tuning for 2-4GB devices
- Network cost-aware routing decisions
- Real-time policy adaptation

Verification Commands:
pytest tests/mobile/test_resource_policy.py -v -q
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

# Mock device profiler if not available
try:
    from src.production.monitoring.mobile.device_profiler import DeviceProfile
except ImportError:

    @dataclass
    class DeviceProfile:
        battery_percent: int = 80
        battery_charging: bool = False
        cpu_temp_celsius: float = 35.0
        ram_total_mb: int = 4096
        ram_used_mb: int = 2048
        ram_available_mb: int = 2048
        cpu_percent: float = 50.0
        network_type: str = "wifi"
        network_latency_ms: int = 50
        device_type: str = "phone"

        def to_dict(self):
            return {
                "battery_percent": self.battery_percent,
                "battery_charging": self.battery_charging,
                "cpu_temp_celsius": self.cpu_temp_celsius,
                "ram_total_mb": self.ram_total_mb,
                "ram_used_mb": self.ram_used_mb,
                "ram_available_mb": self.ram_available_mb,
                "cpu_percent": self.cpu_percent,
                "network_type": self.network_type,
                "network_latency_ms": self.network_latency_ms,
                "device_type": self.device_type,
            }


try:
    from src.production.monitoring.mobile.resource_management import (
        BatteryThermalResourceManager,
        PowerMode,
        TransportPreference,
    )

    RESOURCE_MANAGEMENT_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGEMENT_AVAILABLE = False


class TestResourcePolicy:
    """Test battery/thermal-aware resource policy"""

    @pytest.fixture
    def resource_manager(self):
        """Create resource manager for testing"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")
        return BatteryThermalResourceManager()

    @pytest.fixture
    def normal_profile(self):
        """Normal device profile - good conditions"""
        return DeviceProfile(
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            ram_total_mb=4096,
            ram_used_mb=1024,
            ram_available_mb=3072,
            cpu_percent=30.0,
            network_type="wifi",
            network_latency_ms=25,
            device_type="phone",
        )

    @pytest.fixture
    def low_battery_profile(self):
        """Low battery profile - should prefer BitChat"""
        return DeviceProfile(
            battery_percent=15,  # Below 20% threshold
            battery_charging=False,
            cpu_temp_celsius=35.0,
            ram_total_mb=4096,
            ram_used_mb=1024,
            ram_available_mb=3072,
            cpu_percent=30.0,
            network_type="wifi",
            network_latency_ms=25,
            device_type="phone",
        )

    @pytest.fixture
    def critical_battery_profile(self):
        """Critical battery profile - should use BitChat only"""
        return DeviceProfile(
            battery_percent=8,  # Below 10% threshold
            battery_charging=False,
            cpu_temp_celsius=35.0,
            ram_total_mb=4096,
            ram_used_mb=1024,
            ram_available_mb=3072,
            cpu_percent=30.0,
            network_type="wifi",
            network_latency_ms=25,
            device_type="phone",
        )

    @pytest.fixture
    def high_thermal_profile(self):
        """High thermal profile - should throttle performance"""
        return DeviceProfile(
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=60.0,  # High temperature
            ram_total_mb=4096,
            ram_used_mb=1024,
            ram_available_mb=3072,
            cpu_percent=85.0,  # High CPU usage
            network_type="wifi",
            network_latency_ms=25,
            device_type="phone",
        )

    @pytest.fixture
    def low_memory_profile(self):
        """Low memory profile - should reduce chunk sizes"""
        return DeviceProfile(
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            ram_total_mb=2048,  # 2GB device
            ram_used_mb=1800,  # High memory usage
            ram_available_mb=248,
            cpu_percent=30.0,
            network_type="wifi",
            network_latency_ms=25,
            device_type="phone",
        )

    @pytest.fixture
    def cellular_profile(self):
        """Cellular network profile - should prefer BitChat for cost savings"""
        return DeviceProfile(
            battery_percent=50,
            battery_charging=False,
            cpu_temp_celsius=35.0,
            ram_total_mb=4096,
            ram_used_mb=1024,
            ram_available_mb=3072,
            cpu_percent=30.0,
            network_type="cellular",  # Cellular network
            network_latency_ms=150,  # Higher latency
            device_type="phone",
        )

    @pytest.mark.asyncio
    async def test_normal_conditions_balanced_mode(self, resource_manager, normal_profile):
        """Test resource management under normal conditions"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(normal_profile)

        # Should use balanced or performance mode
        assert state.power_mode in [PowerMode.BALANCED, PowerMode.PERFORMANCE]

        # Should allow balanced transport usage
        assert state.transport_preference in [
            TransportPreference.BALANCED,
            TransportPreference.BITCHAT_PREFERRED,
        ]

        # Chunk size should be reasonable for 4GB device
        chunk_size = state.chunking_config.effective_chunk_size()
        assert 256 <= chunk_size <= 1024

        # Should not have critical policies active
        assert "battery_critical" not in state.active_policies
        assert "thermal_critical" not in state.active_policies

    @pytest.mark.asyncio
    async def test_low_battery_bitchat_preferred(self, resource_manager, low_battery_profile):
        """Test BitChat preference under low battery conditions"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(low_battery_profile)

        # Should prefer power save mode
        assert state.power_mode in [PowerMode.POWER_SAVE, PowerMode.BALANCED]

        # Should prefer BitChat for offline-first operation
        assert state.transport_preference in [
            TransportPreference.BITCHAT_PREFERRED,
            TransportPreference.BITCHAT_ONLY,
        ]

        # Should have battery conservation policies
        assert "battery_low" in state.active_policies

        # Should reduce chunk sizes to save processing power
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 512  # Smaller chunks for battery saving

    @pytest.mark.asyncio
    async def test_critical_battery_bitchat_only(self, resource_manager, critical_battery_profile):
        """Test BitChat-only mode under critical battery"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(critical_battery_profile)

        # Should use critical power mode
        assert state.power_mode == PowerMode.CRITICAL

        # Should use BitChat only for maximum power savings
        assert state.transport_preference == TransportPreference.BITCHAT_ONLY

        # Should have critical battery policy active
        assert "battery_critical" in state.active_policies

        # Should use very small chunks to minimize processing
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 256  # Very small chunks for critical battery

    @pytest.mark.asyncio
    async def test_thermal_throttling(self, resource_manager, high_thermal_profile):
        """Test thermal throttling reduces performance"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(high_thermal_profile)

        # Should throttle performance due to high temperature
        assert state.power_mode in [PowerMode.POWER_SAVE, PowerMode.CRITICAL]

        # Should have thermal throttling policies
        assert any("thermal" in policy for policy in state.active_policies)

        # Should reduce chunk sizes to reduce processing load
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size < 512  # Smaller chunks for thermal management

    @pytest.mark.asyncio
    async def test_memory_constrained_chunk_sizing(self, resource_manager, low_memory_profile):
        """Test chunk size reduction for memory-constrained devices"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(low_memory_profile)

        # Should have memory constraint policies
        assert "memory_constrained" in state.active_policies

        # Should use smaller chunks for 2GB device with high memory usage
        chunk_size = state.chunking_config.effective_chunk_size()
        assert chunk_size <= 256  # Small chunks for memory-constrained device

        # Chunking recommendations should reflect memory constraints
        recommendations = resource_manager.get_chunking_recommendations("tensor")
        assert recommendations["chunk_size"] <= 256
        assert recommendations["batch_size"] <= 4

    @pytest.mark.asyncio
    async def test_cellular_network_cost_awareness(self, resource_manager, cellular_profile):
        """Test BitChat preference on cellular networks for cost savings"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        state = await resource_manager.evaluate_and_adapt(cellular_profile)

        # Should prefer BitChat on cellular to save data costs
        assert state.transport_preference == TransportPreference.BITCHAT_PREFERRED

        # Should have data cost awareness policies
        assert "data_cost_aware" in state.active_policies

        # May also have high latency policy if latency is high
        if cellular_profile.network_latency_ms > 300:
            assert "high_latency_network" in state.active_policies

    @pytest.mark.asyncio
    async def test_transport_routing_decision_small_message(self, resource_manager, normal_profile):
        """Test transport routing decision for small messages"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(normal_profile)

        # Small message (1KB) should typically prefer BitChat
        decision = await resource_manager.get_transport_routing_decision(message_size_bytes=1024, priority=5)

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert "rationale" in decision
        assert isinstance(decision["chunk_size"], int)
        assert decision["chunk_size"] > 0

    @pytest.mark.asyncio
    async def test_transport_routing_decision_large_message(self, resource_manager, normal_profile):
        """Test transport routing decision for large messages"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(normal_profile)

        # Large message (50KB) under normal conditions
        decision = await resource_manager.get_transport_routing_decision(message_size_bytes=50 * 1024, priority=5)

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert "fallback_transport" in decision
        assert decision["chunk_size"] > 0

    @pytest.mark.asyncio
    async def test_transport_routing_decision_urgent_priority(self, resource_manager, normal_profile):
        """Test transport routing decision for urgent messages"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(normal_profile)

        # Urgent message (priority 9) should prefer fastest transport
        decision = await resource_manager.get_transport_routing_decision(
            message_size_bytes=5 * 1024,
            priority=9,  # High priority
        )

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert "high_priority" in " ".join(decision["rationale"])

    @pytest.mark.asyncio
    async def test_battery_critical_forces_bitchat_only(self, resource_manager, critical_battery_profile):
        """Test that critical battery forces BitChat-only routing"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(critical_battery_profile)

        # Even large, urgent messages should use BitChat only under critical battery
        decision = await resource_manager.get_transport_routing_decision(
            message_size_bytes=100 * 1024,
            priority=10,  # Large message  # Maximum priority
        )

        assert decision["primary_transport"] == "bitchat"
        assert decision["fallback_transport"] is None  # No fallback, BitChat only
        assert "battery_critical" in " ".join(decision["rationale"])

    @pytest.mark.asyncio
    async def test_chunking_recommendations_data_types(self, resource_manager, normal_profile):
        """Test chunking recommendations for different data types"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(normal_profile)

        # Test different data types
        tensor_rec = resource_manager.get_chunking_recommendations("tensor")
        text_rec = resource_manager.get_chunking_recommendations("text")
        embedding_rec = resource_manager.get_chunking_recommendations("embedding")

        # All should have required fields
        for rec in [tensor_rec, text_rec, embedding_rec]:
            assert "chunk_size" in rec
            assert rec["chunk_size"] > 0

        # Text should have token limits
        assert "max_tokens" in text_rec
        assert text_rec["max_tokens"] > 0

        # Embeddings should have batch sizes and dimension limits
        assert "batch_size" in embedding_rec
        assert "dimension_limit" in embedding_rec

    @pytest.mark.asyncio
    async def test_resource_status_reporting(self, resource_manager, normal_profile):
        """Test resource management status reporting"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        await resource_manager.evaluate_and_adapt(normal_profile)

        status = resource_manager.get_status()

        # Should have all required status fields
        assert "state" in status
        assert "policy" in status
        assert "statistics" in status

        # State should have current configuration
        state = status["state"]
        assert "power_mode" in state
        assert "transport_preference" in state
        assert "chunk_size" in state

        # Statistics should track adaptations
        stats = status["statistics"]
        assert "policy_adaptations" in stats
        assert "transport_switches" in stats

    @pytest.mark.asyncio
    async def test_policy_adaptation_tracking(self, resource_manager):
        """Test that policy changes are tracked correctly"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        # Start with normal conditions
        normal_profile = DeviceProfile(battery_percent=80, cpu_temp_celsius=35.0)
        await resource_manager.evaluate_and_adapt(normal_profile)
        initial_adaptations = resource_manager.stats["policy_adaptations"]

        # Switch to critical battery
        critical_profile = DeviceProfile(battery_percent=5, cpu_temp_celsius=35.0)
        await resource_manager.evaluate_and_adapt(critical_profile)

        # Should have tracked the adaptation
        assert resource_manager.stats["policy_adaptations"] > initial_adaptations


class TestDualPathTransportIntegration:
    """Test integration of resource management with dual-path transport"""

    @pytest.fixture
    def mock_dual_path_transport(self):
        """Create mock dual-path transport for testing"""
        # Mock the transport imports
        with patch("src.core.p2p.dual_path_transport.RESOURCE_MANAGEMENT_AVAILABLE", True):
            try:
                from packages.p2p.core.dual_path_transport import DualPathTransport

                return DualPathTransport(enable_bitchat=True, enable_betanet=True)
            except ImportError:
                pytest.skip("DualPathTransport not available")

    @pytest.mark.asyncio
    async def test_device_profile_update_integration(self, mock_dual_path_transport):
        """Test device profile updates in dual-path transport"""
        if not hasattr(mock_dual_path_transport, "update_device_profile"):
            pytest.skip("Device profile update not available")

        profile = DeviceProfile(battery_percent=15, cpu_temp_celsius=50.0)

        # Should not raise an exception
        await mock_dual_path_transport.update_device_profile(profile)

        # Should track resource adaptations
        assert "resource_adaptations" in mock_dual_path_transport.routing_stats

    def test_resource_status_integration(self, mock_dual_path_transport):
        """Test resource status reporting in dual-path transport"""
        if not hasattr(mock_dual_path_transport, "get_resource_status"):
            pytest.skip("Resource status not available")

        status = mock_dual_path_transport.get_resource_status()

        # Should have enabled flag
        assert "enabled" in status

        # If enabled, should have status details
        if status["enabled"]:
            assert "status" in status
            assert "chunking_recommendations" in status


# Integration test fixtures for protocol shift verification
@pytest.fixture
def battery_scenarios():
    """Battery level scenarios for testing protocol shifts"""
    return [
        {"level": 100, "charging": True, "expected_pref": "balanced"},
        {"level": 50, "charging": False, "expected_pref": "bitchat_preferred"},
        {"level": 15, "charging": False, "expected_pref": "bitchat_preferred"},
        {"level": 8, "charging": False, "expected_pref": "bitchat_only"},
    ]


@pytest.fixture
def memory_scenarios():
    """Memory scenarios for testing chunk size adaptation"""
    return [
        {"total_gb": 8, "used_percent": 50, "expected_scale": ">=1.0"},
        {"total_gb": 4, "used_percent": 60, "expected_scale": "~1.0"},
        {"total_gb": 2, "used_percent": 80, "expected_scale": "<=0.5"},
        {"total_gb": 1, "used_percent": 90, "expected_scale": "<=0.5"},
    ]


class TestResourcePolicyIntegration:
    """Integration tests for resource policy scenarios"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_index", [0, 1, 2, 3])
    async def test_battery_protocol_shift(self, battery_scenarios, scenario_index):
        """Test protocol shift based on battery levels"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        scenario = battery_scenarios[scenario_index]
        manager = BatteryThermalResourceManager()

        profile = DeviceProfile(
            battery_percent=scenario["level"],
            battery_charging=scenario["charging"],
            cpu_temp_celsius=35.0,
            ram_total_mb=4096,
            ram_available_mb=2048,
        )

        state = await manager.evaluate_and_adapt(profile)

        # Verify expected transport preference
        if scenario["expected_pref"] == "balanced":
            assert state.transport_preference in [
                TransportPreference.BALANCED,
                TransportPreference.BITCHAT_PREFERRED,
            ]
        elif scenario["expected_pref"] == "bitchat_preferred":
            assert state.transport_preference == TransportPreference.BITCHAT_PREFERRED
        elif scenario["expected_pref"] == "bitchat_only":
            assert state.transport_preference == TransportPreference.BITCHAT_ONLY

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_index", [0, 1, 2, 3])
    async def test_memory_chunk_adaptation(self, memory_scenarios, scenario_index):
        """Test chunk size adaptation based on memory constraints"""
        if not RESOURCE_MANAGEMENT_AVAILABLE:
            pytest.skip("Resource management not available")

        scenario = memory_scenarios[scenario_index]
        manager = BatteryThermalResourceManager()

        total_mb = int(scenario["total_gb"] * 1024)
        used_mb = int(total_mb * scenario["used_percent"] / 100)
        available_mb = total_mb - used_mb

        profile = DeviceProfile(
            battery_percent=80,
            cpu_temp_celsius=35.0,
            ram_total_mb=total_mb,
            ram_used_mb=used_mb,
            ram_available_mb=available_mb,
        )

        state = await manager.evaluate_and_adapt(profile)
        chunk_size = state.chunking_config.effective_chunk_size()

        # Verify chunk size adaptation based on memory constraints
        if scenario["expected_scale"] == ">=1.0":
            assert chunk_size >= 512  # Larger chunks for high memory
        elif scenario["expected_scale"] == "~1.0":
            assert 256 <= chunk_size <= 768  # Standard chunks
        elif scenario["expected_scale"] == "<=0.5":
            assert chunk_size <= 256  # Smaller chunks for low memory


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-q"])
