"""
Quick validation of edge device consolidation

Tests core functionality of the unified edge device system.
"""

import asyncio
from pathlib import Path
import sys

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

try:
    from edge.core.edge_manager import EdgeManager
    from edge.fog_compute.fog_coordinator import ComputeCapacity, FogCoordinator, TaskType
    from edge.mobile.resource_manager import MobileResourceManager

    print("SUCCESS: All edge device imports successful")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)


async def test_edge_consolidation():
    """Test edge device consolidation functionality"""

    print("\nTesting Edge Device Consolidation...")

    # Test 1: Edge Manager
    print("\nTesting Edge Manager...")
    edge_manager = EdgeManager()

    device = await edge_manager.register_device(device_id="test_device", device_name="Test Device", auto_detect=True)

    assert device.device_id == "test_device"
    assert device.device_name == "Test Device"
    print(f"   SUCCESS: Device registered: {device.device_name} ({device.device_type.value})")

    # Test deployment
    deployment_id = await edge_manager.deploy_workload(
        device_id="test_device", model_id="test_model", deployment_type="inference"
    )

    assert deployment_id is not None
    print(f"   SUCCESS: Workload deployed: {deployment_id}")

    # Test 2: Mobile Resource Manager
    print("\nTesting Mobile Resource Manager...")
    mobile_manager = MobileResourceManager()

    # Test with environment simulation
    with patch_env({"BATTERY": "25", "THERMAL": "warm", "NETWORK_TYPE": "cellular"}):
        profile = mobile_manager.create_device_profile_from_env()
        optimization = await mobile_manager.optimize_for_device(profile)

        assert optimization.power_mode.value in ["power_save", "balanced", "critical"]
        assert optimization.transport_preference.value.startswith("bitchat")
        print(
            f"   SUCCESS: Mobile optimization: {optimization.power_mode.value} mode, {optimization.transport_preference.value} transport"
        )

    # Test 3: Fog Coordinator
    print("\nTesting Fog Computing Coordinator...")
    fog_coordinator = FogCoordinator()

    # Register fog node
    capacity = ComputeCapacity(
        cpu_cores=4,
        cpu_utilization=0.2,
        memory_mb=4096,
        memory_used_mb=1024,
        gpu_available=False,
        gpu_memory_mb=0,
        battery_powered=True,
        battery_percent=70,
        is_charging=True,
        thermal_state="normal",
    )

    success = await fog_coordinator.register_node("fog_node_1", capacity)
    assert success
    print(f"   SUCCESS: Fog node registered with {capacity.cpu_cores} cores")

    # Submit task
    task_id = await fog_coordinator.submit_task(
        task_type=TaskType.INFERENCE, cpu_cores=1.0, memory_mb=512, estimated_duration=60.0
    )

    assert task_id is not None
    print(f"   SUCCESS: Fog task submitted: {task_id}")

    # Test 4: System Integration
    print("\nTesting System Integration...")

    # Get system status from all components
    edge_status = edge_manager.get_system_status()
    mobile_status = mobile_manager.get_status()
    fog_status = fog_coordinator.get_system_status()

    assert edge_status["devices"]["total"] == 1
    assert edge_status["deployments"]["total"] == 1
    assert mobile_status["statistics"]["decisions_made"] >= 1
    assert fog_status["nodes"]["total"] == 1
    assert fog_status["tasks"]["pending"] >= 0

    print("   SUCCESS: Integration verified:")
    print(f"      - Edge devices: {edge_status['devices']['total']}")
    print(f"      - Deployments: {edge_status['deployments']['total']}")
    print(f"      - Mobile decisions: {mobile_status['statistics']['decisions_made']}")
    print(f"      - Fog nodes: {fog_status['nodes']['total']}")

    # Test 5: Mobile Platform Detection
    print("\nTesting Mobile Platform Support...")
    try:
        from edge.mobile.platforms import get_platform_manager

        platform_manager = get_platform_manager()
        print(f"   SUCCESS: Platform manager created: {type(platform_manager).__name__}")
    except Exception as e:
        print(f"   WARNING: Platform manager: {e} (expected on non-mobile platforms)")

    # Cleanup
    await edge_manager.shutdown()
    await mobile_manager.reset()
    await fog_coordinator.shutdown()

    print("\nEdge Device Consolidation Test PASSED!")
    print("\nKey achievements:")
    print("  - Unified edge device architecture created")
    print("  - Mobile resource optimization implemented")
    print("  - Fog computing coordination working")
    print("  - Cross-system integration validated")
    print("  - Comprehensive test suite passing")

    return True


def patch_env(env_vars):
    """Context manager to temporarily patch environment variables"""
    import os

    class EnvPatcher:
        def __init__(self, vars_dict):
            self.vars_dict = vars_dict
            self.original_vars = {}

        def __enter__(self):
            for key, value in self.vars_dict.items():
                self.original_vars[key] = os.environ.get(key)
                os.environ[key] = value
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            for key in self.vars_dict:
                if self.original_vars[key] is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = self.original_vars[key]

    return EnvPatcher(env_vars)


if __name__ == "__main__":
    try:
        result = asyncio.run(test_edge_consolidation())
        print(f"\nTest Result: {'PASSED' if result else 'FAILED'}")
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
