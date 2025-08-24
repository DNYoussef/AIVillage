"""
Integration tests for unified edge device consolidation

Tests the consolidated edge device architecture including:
- Edge device registration and management
- Mobile resource optimization
- Fog computing coordination
- Cross-platform compatibility
- Legacy system migration
"""

import asyncio
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest

# Add packages to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))

from edge.core.edge_manager import DeviceCapabilities, DeviceType, EdgeManager, EdgeState
from edge.fog_compute.fog_coordinator import ComputeCapacity, FogCoordinator, TaskPriority, TaskType
from edge.mobile.resource_manager import MobileDeviceProfile, MobileResourceManager, PowerMode, TransportPreference


class TestEdgeDeviceConsolidation:
    """Test suite for unified edge device system"""

    @pytest.fixture
    async def edge_manager(self):
        """Create edge manager for testing"""
        manager = EdgeManager()
        yield manager
        await manager.shutdown()

    @pytest.fixture
    async def mobile_resource_manager(self):
        """Create mobile resource manager for testing"""
        manager = MobileResourceManager()
        yield manager
        await manager.reset()

    @pytest.fixture
    async def fog_coordinator(self):
        """Create fog coordinator for testing"""
        coordinator = FogCoordinator()
        yield coordinator
        await coordinator.shutdown()

    async def test_edge_device_registration(self, edge_manager):
        """Test edge device registration with auto-detection"""

        # Test device registration
        device = await edge_manager.register_device(
            device_id="test_device_001", device_name="Test Smartphone", auto_detect=True
        )

        assert device.device_id == "test_device_001"
        assert device.device_name == "Test Smartphone"
        assert device.state == EdgeState.ONLINE
        assert device.device_type in [DeviceType.SMARTPHONE, DeviceType.LAPTOP, DeviceType.DESKTOP]

        # Check device is registered
        assert "test_device_001" in edge_manager.devices
        assert edge_manager.stats["devices_registered"] == 1

        # Test device status
        status = edge_manager.get_device_status("test_device_001")
        assert status["device_info"]["device_id"] == "test_device_001"
        assert "capabilities" in status
        assert "deployments" in status

    async def test_mobile_resource_optimization(self, mobile_resource_manager):
        """Test mobile resource optimization with various conditions"""

        # Test normal conditions
        normal_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="normal_device",
            battery_percent=80,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            cpu_percent=30.0,
            ram_used_mb=1024,
            ram_available_mb=3072,
            ram_total_mb=4096,
            network_type="wifi",
            device_type="smartphone",
        )

        optimization = await mobile_resource_manager.optimize_for_device(normal_profile)
        assert optimization.power_mode == PowerMode.PERFORMANCE
        assert optimization.transport_preference == TransportPreference.BALANCED
        assert optimization.cpu_limit_percent >= 40.0
        assert optimization.memory_limit_mb >= 1024

        # Test critical battery conditions
        critical_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="critical_device",
            battery_percent=8,
            battery_charging=False,
            cpu_temp_celsius=35.0,
            cpu_percent=30.0,
            ram_used_mb=1024,
            ram_available_mb=1024,
            ram_total_mb=2048,
            network_type="cellular",
            device_type="smartphone",
        )

        optimization = await mobile_resource_manager.optimize_for_device(critical_profile)
        assert optimization.power_mode == PowerMode.CRITICAL
        assert optimization.transport_preference == TransportPreference.BITCHAT_ONLY
        assert optimization.cpu_limit_percent <= 25.0
        assert optimization.memory_limit_mb <= 512
        assert "battery_critical" in optimization.active_policies

        # Test thermal throttling
        hot_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="hot_device",
            battery_percent=60,
            battery_charging=True,
            cpu_temp_celsius=68.0,
            cpu_percent=85.0,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            network_type="wifi",
            device_type="smartphone",
        )

        optimization = await mobile_resource_manager.optimize_for_device(hot_profile)
        assert optimization.power_mode == PowerMode.CRITICAL
        assert "thermal_critical" in optimization.active_policies
        assert optimization.chunking_config.thermal_scale_factor <= 0.5

    async def test_environment_simulation(self, mobile_resource_manager):
        """Test environment variable simulation support"""

        # Set environment variables for testing
        with patch.dict(
            os.environ,
            {
                "BATTERY": "15",
                "THERMAL": "hot",
                "MEMORY_GB": "2.0",
                "NETWORK_TYPE": "cellular",
                "AIV_MOBILE_PROFILE": "battery_save",
            },
        ):
            # Create profile from environment
            profile = mobile_resource_manager.create_device_profile_from_env()

            assert profile.battery_percent == 15
            assert profile.battery_charging is False
            assert profile.cpu_temp_celsius == 58.0
            assert profile.ram_total_mb == 2048
            assert profile.network_type == "cellular"
            assert profile.power_mode == "power_save"

            # Test optimization with environment profile
            optimization = await mobile_resource_manager.optimize_for_device(profile)
            assert optimization.power_mode == PowerMode.POWER_SAVE
            assert optimization.transport_preference == TransportPreference.BITCHAT_PREFERRED

    async def test_fog_computing_coordination(self, fog_coordinator):
        """Test fog computing task coordination"""

        # Register fog nodes with different capabilities
        charging_capacity = ComputeCapacity(
            cpu_cores=4,
            cpu_utilization=0.2,
            memory_mb=8192,
            memory_used_mb=2048,
            gpu_available=False,
            gpu_memory_mb=0,
            battery_powered=True,
            battery_percent=85,
            is_charging=True,
            thermal_state="normal",
            network_bandwidth_mbps=50.0,
        )

        battery_capacity = ComputeCapacity(
            cpu_cores=2,
            cpu_utilization=0.1,
            memory_mb=4096,
            memory_used_mb=1024,
            gpu_available=False,
            gpu_memory_mb=0,
            battery_powered=True,
            battery_percent=25,
            is_charging=False,
            thermal_state="normal",
            network_bandwidth_mbps=20.0,
        )

        desktop_capacity = ComputeCapacity(
            cpu_cores=8,
            cpu_utilization=0.3,
            memory_mb=16384,
            memory_used_mb=4096,
            gpu_available=True,
            gpu_memory_mb=8192,
            battery_powered=False,
            thermal_state="normal",
            network_bandwidth_mbps=100.0,
        )

        # Register nodes
        await fog_coordinator.register_node("charging_phone", charging_capacity)
        await fog_coordinator.register_node("battery_phone", battery_capacity)
        await fog_coordinator.register_node("desktop_pc", desktop_capacity)

        assert len(fog_coordinator.nodes) == 3
        assert fog_coordinator.stats["nodes_registered"] == 3

        # Submit tasks with different requirements
        task1_id = await fog_coordinator.submit_task(
            task_type=TaskType.INFERENCE,
            priority=TaskPriority.HIGH,
            cpu_cores=1.0,
            memory_mb=512,
            estimated_duration=60.0,
        )

        task2_id = await fog_coordinator.submit_task(
            task_type=TaskType.TRAINING,
            priority=TaskPriority.NORMAL,
            cpu_cores=4.0,
            memory_mb=4096,
            estimated_duration=300.0,
            requires_gpu=True,
        )

        # Allow scheduler to run
        await asyncio.sleep(1)

        # Check task assignments
        task1_status = fog_coordinator.get_task_status(task1_id)
        task2_status = fog_coordinator.get_task_status(task2_id)

        assert task1_status is not None
        assert task2_status is not None

        # GPU task should go to desktop, inference task should prefer charging device
        if task2_status["assigned_node"]:
            assert task2_status["assigned_node"] == "desktop_pc"  # GPU requirement

        # Check system status
        system_status = fog_coordinator.get_system_status()
        assert system_status["nodes"]["total"] == 3
        assert system_status["tasks"]["pending"] >= 0
        assert system_status["tasks"]["active"] >= 0

    async def test_workload_deployment(self, edge_manager):
        """Test AI workload deployment with optimization"""

        # Register a mobile device
        await edge_manager.register_device(
            device_id="mobile_001",
            device_name="Test Mobile Device",
            auto_detect=False,
            capabilities=DeviceCapabilities(
                cpu_cores=4,
                ram_total_mb=3072,
                ram_available_mb=2048,
                storage_available_gb=64.0,
                gpu_available=False,
                gpu_memory_mb=0,
                battery_powered=True,
                battery_percent=30,
                battery_charging=False,
                cpu_temp_celsius=42.0,
                thermal_state="warm",
                network_type="wifi",
                has_internet=True,
                supports_python=True,
                supports_bitchat=True,
                supports_nearby=True,
                supports_ble=True,
            ),
        )

        # Deploy a workload
        deployment_id = await edge_manager.deploy_workload(
            device_id="mobile_001", model_id="mobile_tutor_v1", deployment_type="tutor", config={"priority": 7}
        )

        assert deployment_id is not None
        assert deployment_id in edge_manager.deployments

        deployment = edge_manager.deployments[deployment_id]
        assert deployment.device_id == "mobile_001"
        assert deployment.model_id == "mobile_tutor_v1"
        assert deployment.deployment_type == "tutor"

        # Check mobile optimizations were applied
        assert deployment.cpu_limit_percent <= 40.0  # Battery optimization
        assert deployment.memory_limit_mb <= 1024  # Thermal optimization
        assert deployment.offline_capable is True  # Mobile optimization
        assert deployment.chunk_size_bytes <= 1024  # Resource optimization

        # Allow deployment to complete
        await asyncio.sleep(1)

        # Check device status includes deployment
        status = edge_manager.get_device_status("mobile_001")
        assert len(status["deployments"]) == 1
        assert status["deployments"][0]["deployment_id"] == deployment_id

    async def test_transport_routing_decisions(self, mobile_resource_manager):
        """Test transport routing decision making"""

        # Test normal conditions - should prefer balanced
        normal_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="normal_device",
            battery_percent=70,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            cpu_percent=40.0,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            network_type="wifi",
            device_type="smartphone",
        )

        decision = await mobile_resource_manager.get_transport_routing_decision(
            message_size_bytes=1024, priority=5, profile=normal_profile
        )

        assert decision["primary_transport"] in ["bitchat", "betanet"]
        assert decision["fallback_transport"] in ["bitchat", "betanet", None]
        assert isinstance(decision["chunk_size"], int)
        assert decision["chunk_size"] > 0

        # Test low battery - should prefer BitChat
        low_battery_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="low_battery_device",
            battery_percent=18,
            battery_charging=False,
            cpu_temp_celsius=35.0,
            cpu_percent=40.0,
            ram_used_mb=2048,
            ram_available_mb=2048,
            ram_total_mb=4096,
            network_type="cellular",
            device_type="smartphone",
        )

        decision = await mobile_resource_manager.get_transport_routing_decision(
            message_size_bytes=5120, priority=3, profile=low_battery_profile
        )

        assert decision["primary_transport"] == "bitchat"
        assert "battery_aware" in " ".join(decision["rationale"])

        # Test large message with good conditions - should consider BetaNet
        large_message_decision = await mobile_resource_manager.get_transport_routing_decision(
            message_size_bytes=50000, priority=8, profile=normal_profile  # 50KB
        )

        # Should consider BetaNet for large high-priority messages
        assert large_message_decision["primary_transport"] in ["betanet", "bitchat"]

    async def test_chunking_recommendations(self, mobile_resource_manager):
        """Test dynamic chunking recommendations"""

        # Create low memory profile
        with patch.dict(os.environ, {"MEMORY_GB": "1.5", "BATTERY": "25", "THERMAL": "warm"}):
            profile = mobile_resource_manager.create_device_profile_from_env()
            await mobile_resource_manager.optimize_for_device(profile)

            # Get chunking recommendations
            tensor_rec = mobile_resource_manager.get_chunking_recommendations("tensor")
            text_rec = mobile_resource_manager.get_chunking_recommendations("text")
            embedding_rec = mobile_resource_manager.get_chunking_recommendations("embedding")

            # Should have reduced chunk sizes for constrained device
            assert tensor_rec["chunk_size"] <= 512
            assert tensor_rec["batch_size"] >= 1
            assert text_rec["chunk_size"] <= 512
            assert embedding_rec["chunk_size"] <= 512
            assert embedding_rec["dimension_limit"] <= 768  # Reduced for low memory

    async def test_system_integration(self, edge_manager, mobile_resource_manager, fog_coordinator):
        """Test integration between all edge device systems"""

        # Register device in edge manager
        device = await edge_manager.register_device(
            device_id="integrated_device",
            device_name="Integration Test Device",
            auto_detect=False,
            capabilities=DeviceCapabilities(
                cpu_cores=4,
                ram_total_mb=4096,
                ram_available_mb=3072,
                storage_available_gb=128.0,
                gpu_available=False,
                gpu_memory_mb=0,
                battery_powered=True,
                battery_percent=60,
                battery_charging=True,
                cpu_temp_celsius=38.0,
                thermal_state="normal",
                network_type="wifi",
                has_internet=True,
                supports_bitchat=True,
            ),
        )

        # Create mobile profile for resource optimization
        mobile_profile = MobileDeviceProfile(
            timestamp=time.time(),
            device_id="integrated_device",
            battery_percent=device.capabilities.battery_percent,
            battery_charging=device.capabilities.battery_charging,
            cpu_temp_celsius=device.capabilities.cpu_temp_celsius,
            cpu_percent=30.0,
            ram_used_mb=device.capabilities.ram_total_mb - device.capabilities.ram_available_mb,
            ram_available_mb=device.capabilities.ram_available_mb,
            ram_total_mb=device.capabilities.ram_total_mb,
            network_type=device.capabilities.network_type,
            device_type="smartphone",
        )

        # Get mobile optimization
        optimization = await mobile_resource_manager.optimize_for_device(mobile_profile)

        # Register device as fog node
        fog_capacity = ComputeCapacity(
            cpu_cores=device.capabilities.cpu_cores,
            cpu_utilization=0.3,
            memory_mb=device.capabilities.ram_total_mb,
            memory_used_mb=device.capabilities.ram_total_mb - device.capabilities.ram_available_mb,
            gpu_available=device.capabilities.gpu_available,
            gpu_memory_mb=device.capabilities.gpu_memory_mb,
            battery_powered=device.capabilities.battery_powered,
            battery_percent=device.capabilities.battery_percent,
            is_charging=device.capabilities.battery_charging,
            thermal_state=device.capabilities.thermal_state,
        )

        await fog_coordinator.register_node("integrated_device", fog_capacity)

        # Deploy workload using optimization parameters
        deployment_id = await edge_manager.deploy_workload(
            device_id="integrated_device",
            model_id="optimized_model",
            deployment_type="inference",
            config={
                "cpu_limit": optimization.cpu_limit_percent,
                "memory_limit": optimization.memory_limit_mb,
                "priority": 6,
            },
        )

        # Submit fog task
        await fog_coordinator.submit_task(
            task_type=TaskType.INFERENCE,
            priority=TaskPriority.NORMAL,
            cpu_cores=1.0,
            memory_mb=optimization.memory_limit_mb,
            estimated_duration=120.0,
        )

        # Allow systems to process
        await asyncio.sleep(1)

        # Verify integration
        edge_status = edge_manager.get_device_status("integrated_device")
        mobile_status = mobile_resource_manager.get_status()
        fog_status = fog_coordinator.get_system_status()

        assert edge_status["device_info"]["device_id"] == "integrated_device"
        assert len(edge_status["deployments"]) == 1
        assert edge_status["deployments"][0]["deployment_id"] == deployment_id

        assert mobile_status["current_optimization"] is not None
        assert mobile_status["current_optimization"]["power_mode"] in ["performance", "balanced"]

        assert fog_status["nodes"]["total"] >= 1
        assert "integrated_device" in fog_coordinator.nodes


@pytest.mark.asyncio
async def test_edge_consolidation_integration():
    """Run comprehensive integration test"""
    test_suite = TestEdgeDeviceConsolidation()

    # Test individual components
    edge_manager = EdgeManager()
    mobile_manager = MobileResourceManager()
    fog_coordinator = FogCoordinator()

    try:
        # Test basic functionality
        await test_suite.test_edge_device_registration(edge_manager)
        await test_suite.test_mobile_resource_optimization(mobile_manager)
        await test_suite.test_fog_computing_coordination(fog_coordinator)

        # Test integration
        await test_suite.test_system_integration(edge_manager, mobile_manager, fog_coordinator)

        print("✅ All edge device consolidation tests passed!")
        return True

    except Exception as e:
        print(f"❌ Edge consolidation test failed: {e}")
        return False

    finally:
        await edge_manager.shutdown()
        await mobile_manager.reset()
        await fog_coordinator.shutdown()


if __name__ == "__main__":
    # Run integration test if executed directly
    result = asyncio.run(test_edge_consolidation_integration())
    exit(0 if result else 1)
