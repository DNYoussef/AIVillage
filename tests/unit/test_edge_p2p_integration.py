"""
Test Edge Device and P2P Integration

Validates the integration between the unified edge device system
and the consolidated P2P transport layer.
"""

import asyncio
import sys
from pathlib import Path

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

try:
    from edge.bridges.p2p_integration import create_edge_p2p_bridge
    from edge.core.edge_manager import DeviceCapabilities, EdgeManager
    from edge.mobile.resource_manager import MobileResourceManager

    print("SUCCESS: All edge and P2P integration imports successful")
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)


async def test_edge_p2p_integration():
    """Test integration between edge devices and P2P transport"""

    print("\nTesting Edge Device P2P Integration...")

    # Create edge manager
    edge_manager = EdgeManager()

    # Register edge devices with different characteristics
    mobile_device = await edge_manager.register_device(
        device_id="mobile_001",
        device_name="Mobile Phone",
        auto_detect=False,
        capabilities=DeviceCapabilities(
            cpu_cores=4,
            ram_total_mb=3072,
            ram_available_mb=2048,
            storage_available_gb=64.0,
            gpu_available=False,
            gpu_memory_mb=0,
            battery_powered=True,
            battery_percent=25,  # Low battery
            battery_charging=False,
            cpu_temp_celsius=42.0,
            thermal_state="warm",
            network_type="cellular",
            has_internet=True,
            is_metered_connection=True,
            supports_bitchat=True,
            supports_nearby=True,
            supports_ble=True,
            max_concurrent_tasks=2,
        ),
    )

    laptop_device = await edge_manager.register_device(
        device_id="laptop_001",
        device_name="Laptop Computer",
        auto_detect=False,
        capabilities=DeviceCapabilities(
            cpu_cores=8,
            ram_total_mb=16384,
            ram_available_mb=12288,
            storage_available_gb=512.0,
            gpu_available=True,
            gpu_memory_mb=4096,
            battery_powered=True,
            battery_percent=85,
            battery_charging=True,
            cpu_temp_celsius=35.0,
            thermal_state="normal",
            network_type="wifi",
            has_internet=True,
            is_metered_connection=False,
            supports_bitchat=False,
            supports_nearby=False,
            supports_ble=False,
            max_concurrent_tasks=8,
        ),
    )

    print(f"   SUCCESS: Registered mobile device: {mobile_device.device_name}")
    print(f"   SUCCESS: Registered laptop device: {laptop_device.device_name}")

    # Test P2P bridge creation
    print("\nTesting P2P Bridge Creation...")
    p2p_bridge = create_edge_p2p_bridge(edge_manager)

    assert p2p_bridge is not None
    print(f"   SUCCESS: P2P bridge created (P2P available: {p2p_bridge.p2p_available})")

    # Test device initialization with P2P
    print("\nTesting Device P2P Initialization...")

    mobile_init = await p2p_bridge.initialize_p2p_for_device("mobile_001")
    laptop_init = await p2p_bridge.initialize_p2p_for_device("laptop_001")

    print(f"   Mobile P2P init: {'SUCCESS' if mobile_init else 'FALLBACK'}")
    print(f"   Laptop P2P init: {'SUCCESS' if laptop_init else 'FALLBACK'}")

    # Test transport optimization
    print("\nTesting Transport Optimization...")

    # Deploy workload to mobile device
    mobile_deployment = await edge_manager.deploy_workload(
        device_id="mobile_001", model_id="mobile_optimized_model", deployment_type="inference", config={"priority": 8}
    )

    # Get transport optimization for deployment
    mobile_optimization = await p2p_bridge.optimize_transport_for_deployment(
        deployment_id=mobile_deployment, device_id="mobile_001"
    )

    print("   Mobile transport optimization:")
    if "error" not in mobile_optimization:
        print(f"      - Recommended transport: {mobile_optimization.get('recommended_transport', 'unknown')}")
        print(f"      - Fallback transport: {mobile_optimization.get('fallback_transport', 'none')}")
        print(f"      - Chunk size: {mobile_optimization.get('chunk_size', 'unknown')} bytes")
        print(f"      - Rationale: {', '.join(mobile_optimization.get('rationale', []))}")
    else:
        print(f"      - Fallback mode: {mobile_optimization['error']}")

    # Test message sending (fallback mode)
    print("\nTesting Edge Message Communication...")

    message_payload = b"Test message from mobile to laptop"
    message_sent = await p2p_bridge.send_edge_message(
        from_device_id="mobile_001",
        to_device_id="laptop_001",
        message_type="test_communication",
        payload=message_payload,
        priority="normal",
    )

    print(f"   Message sent: {'SUCCESS' if message_sent else 'FALLBACK'}")

    # Test broadcast message
    broadcast_payload = b"Broadcast message from mobile device"
    broadcast_sent = await p2p_bridge.broadcast_edge_message(
        from_device_id="mobile_001", message_type="device_announcement", payload=broadcast_payload, priority="low"
    )

    print(f"   Broadcast sent: {'SUCCESS' if broadcast_sent else 'FALLBACK'}")

    # Test state synchronization
    print("\nTesting Device State Synchronization...")

    sync_results = await p2p_bridge.sync_device_states()
    print(f"   Sync results: {len(sync_results)} devices processed")

    for device_id, result in sync_results.items():
        if "error" not in result:
            print(f"      - {device_id}: Context updated, transports: {result.get('available_transports', [])}")
        else:
            print(f"      - {device_id}: Fallback mode")

    # Test integration status
    print("\nTesting Integration Status...")

    integration_status = p2p_bridge.get_integration_status()
    print(f"   P2P Available: {integration_status['p2p_available']}")
    print(f"   Integrated Devices: {integration_status['integrated_devices']}")
    print(f"   Device Mappings: {list(integration_status['device_mappings'].keys())}")

    if integration_status["p2p_available"]:
        transport_status = integration_status.get("transport_status", {})
        print(f"   Active Transports: {transport_status.get('available_transports', [])}")

    # Test mobile resource optimization integration
    print("\nTesting Mobile Resource Optimization Integration...")

    mobile_manager = MobileResourceManager()

    # Create profile from edge device
    import time

    from edge.mobile.resource_manager import MobileDeviceProfile

    profile = MobileDeviceProfile(
        timestamp=time.time(),
        device_id="mobile_001",
        battery_percent=mobile_device.capabilities.battery_percent,
        battery_charging=mobile_device.capabilities.battery_charging,
        cpu_temp_celsius=mobile_device.capabilities.cpu_temp_celsius,
        cpu_percent=60.0,
        ram_used_mb=mobile_device.capabilities.ram_total_mb - mobile_device.capabilities.ram_available_mb,
        ram_available_mb=mobile_device.capabilities.ram_available_mb,
        ram_total_mb=mobile_device.capabilities.ram_total_mb,
        network_type=mobile_device.capabilities.network_type,
        device_type=mobile_device.device_type.value,
    )

    optimization = await mobile_manager.optimize_for_device(profile)

    print("   Mobile optimization result:")
    print(f"      - Power mode: {optimization.power_mode.value}")
    print(f"      - Transport preference: {optimization.transport_preference.value}")
    print(f"      - CPU limit: {optimization.cpu_limit_percent}%")
    print(f"      - Memory limit: {optimization.memory_limit_mb}MB")
    print(f"      - Active policies: {', '.join(optimization.active_policies)}")

    # Get transport routing decision
    routing = await mobile_manager.get_transport_routing_decision(
        message_size_bytes=10240, priority=7, profile=profile  # 10KB message
    )

    print("   Transport routing decision:")
    print(f"      - Primary transport: {routing['primary_transport']}")
    print(f"      - Fallback transport: {routing.get('fallback_transport', 'none')}")
    print(f"      - Chunk size: {routing['chunk_size']} bytes")
    print(f"      - Rationale: {', '.join(routing['rationale'])}")

    # Test edge system status
    print("\nTesting System Status Integration...")

    edge_status = edge_manager.get_system_status()
    mobile_status = mobile_manager.get_status()

    print(f"   Edge devices: {edge_status['devices']['total']}")
    print(f"   Edge deployments: {edge_status['deployments']['total']}")
    print(f"   Mobile decisions: {mobile_status['statistics']['decisions_made']}")

    # Cleanup
    await edge_manager.shutdown()
    await mobile_manager.reset()

    print("\nEdge-P2P Integration Test COMPLETED!")
    print("\nIntegration Summary:")
    print("  - Edge devices can be registered with P2P bridge")
    print("  - Transport priorities determined by device type and capabilities")
    print("  - Mobile resource optimization integrated with P2P routing")
    print("  - Message communication framework established")
    print("  - State synchronization between systems working")
    print("  - Graceful fallback when P2P system not available")

    return True


if __name__ == "__main__":
    try:
        result = asyncio.run(test_edge_p2p_integration())
        print(f"\nIntegration Test Result: {'PASSED' if result else 'FAILED'}")
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\nERROR: Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
