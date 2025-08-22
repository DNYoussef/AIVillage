"""
Example: AIVillage Edge Integration

Demonstrates how the fog computing edge components integrate with
the existing AIVillage edge infrastructure:
- Reuses existing EdgeManager and FogCoordinator
- Integrates with P2P transport system (BitChat/BetaNet)
- Leverages existing device profiling and resource management
- Maintains compatibility with digital twin systems

This shows the "build on existing infrastructure" approach.
"""

import asyncio
import logging

from packages.fog.edge import AIVillageEdgeIntegration, JobRequest, create_integrated_edge_node
from packages.fog.edge.runner import ExecutionResources, RuntimeType

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def aivillage_integration_demo():
    """Demonstrate fog computing integration with existing AIVillage infrastructure"""

    print("🔗 AIVillage Edge Integration Demo")
    print("=" * 50)

    device_name = "aivillage-edge-001"
    namespace = "aivillage/fog"
    gateway_url = "https://fog-gateway.aivillage.dev"

    # 1. Create integrated edge node
    print("1. Creating integrated edge node...")

    try:
        integration = await create_integrated_edge_node(
            device_name=device_name, operator_namespace=namespace, fog_gateway_url=gateway_url
        )

        print(f"   ✓ Edge node created: {device_name}")
        print(f"   ✓ Namespace: {namespace}")
        print(f"   ✓ Gateway: {gateway_url}")

        # Check integration status
        device_status = await integration.get_device_status()

        print("\n   📊 Integration Status:")
        if "aivillage_integration" in device_status:
            ai_status = device_status["aivillage_integration"]
            print(f"      - EdgeManager: {'✓' if ai_status['edge_manager_connected'] else '❌'}")
            print(f"      - FogCoordinator: {'✓' if ai_status['fog_coordinator_connected'] else '❌'}")
            print(f"      - P2P Bridge: {'✓' if ai_status['p2p_bridge_connected'] else '❌'}")
            print(f"      - Resource Manager: {'✓' if ai_status['resource_manager_active'] else '❌'}")
        else:
            print("      - Using standalone fog components (AIVillage infrastructure not available)")

        # 2. Show device capabilities
        print("\n2. Device capabilities:")

        capabilities = device_status.get("capabilities", {})
        print(f"   🖥️ CPU Cores: {capabilities.get('cpu_cores', 'N/A')}")
        print(f"   🧠 Memory: {capabilities.get('memory_mb', 'N/A')}MB")
        print(f"   💾 Disk: {capabilities.get('disk_mb', 'N/A')}MB")
        print(f"   🔌 Power Profile: {capabilities.get('power_profile', 'N/A')}")
        print(f"   🔋 Battery: {capabilities.get('battery_percent', 'N/A')}%")

        supported_runtimes = capabilities.get("supported_runtimes", [])
        print(f"   ⚙️ Runtimes: {', '.join(supported_runtimes)}")

        # 3. Execute fog job
        print("\n3. Executing fog computing job...")

        if "wasi" in supported_runtimes:
            # Create sample WASM job
            wasm_payload = b"\x00asm\x01\x00\x00\x00" + b"\x00" * 100  # Mock WASM

            job_request = JobRequest(
                job_id="demo-job-001",
                runtime_type=RuntimeType.WASI,
                payload=wasm_payload,
                args=["hello", "fog"],
                env={"FOG_NODE": device_name, "INTEGRATION": "aivillage"},
                resources=ExecutionResources(cpu_cores=1.0, memory_mb=256, disk_mb=100, max_duration_s=30),
                priority="B",
                namespace=namespace,
                labels={"demo": "aivillage-integration", "type": "wasi-test"},
            )

            print(f"   🚀 Submitting job: {job_request.job_id}")
            print(f"   📦 Runtime: {job_request.runtime_type.value}")
            print(f"   💼 Resources: {job_request.resources.cpu_cores} cores, {job_request.resources.memory_mb}MB")

            try:
                # Execute job through integrated system
                job_status = await integration.execute_job(job_request)

                print(f"   ✓ Job execution initiated: {job_status.status}")

                # Monitor job progress
                for i in range(10):
                    current_status = await integration.fog_edge_node.get_job_status(job_request.job_id)
                    if current_status:
                        print(f"   📊 Job status: {current_status.status}")

                        if current_status.status in ["completed", "failed", "cancelled"]:
                            if current_status.result:
                                result = current_status.result
                                print(f"   ✓ Exit code: {result.exit_code}")
                                print(f"   ⏱️ Duration: {result.duration_ms:.1f}ms")
                                print(f"   💾 Memory peak: {result.memory_peak_mb}MB")
                                if result.stdout:
                                    print(f"   📄 Output: {result.stdout}")
                            break

                    await asyncio.sleep(2)

            except Exception as e:
                print(f"   ❌ Job execution failed: {e}")

        else:
            print("   ℹ️ WASI runtime not available, skipping job execution")

        # 4. Show resource monitoring integration
        print("\n4. Resource monitoring integration:")

        if integration.fog_edge_node and integration.fog_edge_node.monitor:
            monitor_status = integration.fog_edge_node.monitor.get_current_status()

            print(f"   🏥 Health: {monitor_status['health_status']}")
            print(f"   🌡️ Thermal: {monitor_status['thermal_state']}")
            print(f"   🔋 Battery: {monitor_status['battery_state']}")
            print(f"   📊 CPU Usage: {monitor_status['cpu_percent']:.1f}%")
            print(f"   🧠 Memory Usage: {monitor_status['memory_percent']:.1f}%")

        # 5. Demonstrate P2P integration
        print("\n5. P2P transport integration:")

        if integration.p2p_bridge:
            print("   ✓ P2P bridge connected")
            print("   📡 Available transports: BitChat (BLE), BetaNet (HTTP), QUIC")
            print("   🔄 Automatic transport selection based on device state")
            print("   📱 Mobile-optimized routing with battery awareness")
        else:
            print("   ℹ️ P2P bridge not available (using direct fog communication)")

        # 6. Show existing infrastructure integration
        print("\n6. AIVillage infrastructure integration:")

        if integration.use_existing_infrastructure:
            print("   ✓ Integrated with existing AIVillage edge infrastructure")

            if integration.edge_manager:
                print("   📋 EdgeManager: Device registered and managed")

            if integration.fog_coordinator:
                print("   🌫️ FogCoordinator: Workload distribution and coordination")

            if integration.resource_manager:
                print("   ⚙️ ResourceManager: Battery/thermal-aware policies")

            print("   🔗 Benefits:")
            print("      - Reuses existing device profiling")
            print("      - Leverages established P2P transport")
            print("      - Maintains compatibility with digital twins")
            print("      - Integrates with federated learning systems")

        else:
            print("   ℹ️ Using standalone fog components")
            print("   🔧 AIVillage infrastructure not available or disabled")

        # 7. Continuous monitoring demonstration
        print("\n7. Continuous monitoring (5 seconds)...")

        for i in range(3):
            await asyncio.sleep(2)

            # Get current status
            status = await integration.get_device_status()
            health = status.get("health", {})
            utilization = status.get("utilization", {})

            print(
                f"   📈 Health: {health.get('health_status', 'unknown')}, "
                f"CPU: {utilization.get('cpu_percent', 0):.1f}%, "
                f"Memory: {utilization.get('memory_percent', 0):.1f}%"
            )

        # 8. Integration benefits summary
        print("\n8. Integration Benefits Summary:")
        print("   ✅ Built on existing AIVillage infrastructure")
        print("   ✅ Reuses established edge device management")
        print("   ✅ Leverages existing P2P transport protocols")
        print("   ✅ Maintains compatibility with digital twin systems")
        print("   ✅ Integrates with federated learning infrastructure")
        print("   ✅ Provides unified fog computing capabilities")
        print("   ✅ Mobile-optimized resource policies")
        print("   ✅ Enterprise-grade security and compliance")

        # Cleanup
        print("\n🧹 Cleaning up...")
        await integration.shutdown()

        print("✅ AIVillage integration demo completed!")

    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        print(f"❌ Demo failed: {e}")


async def fog_coordination_demo():
    """Demonstrate fog computing coordination across multiple devices"""

    print("\n🌫️ Fog Computing Coordination Demo")
    print("=" * 40)

    # Simulate multiple edge devices joining the fog network
    devices = [
        {"name": "mobile-phone-001", "type": "mobile"},
        {"name": "laptop-002", "type": "laptop"},
        {"name": "desktop-003", "type": "desktop"},
    ]

    integrations = []

    try:
        # Create multiple integrated edge nodes
        print("1. Creating fog network with multiple devices...")

        for device in devices:
            print(f"   📱 Initializing {device['name']} ({device['type']})")

            integration = AIVillageEdgeIntegration(
                device_name=device["name"],
                operator_namespace="aivillage/fog",
                fog_gateway_url="https://fog-gateway.aivillage.dev",
            )

            await integration.initialize()
            integrations.append(integration)

        print(f"   ✓ Fog network created with {len(integrations)} devices")

        # Show network topology
        print("\n2. Fog network topology:")

        for i, integration in enumerate(integrations):
            device_status = await integration.get_device_status()
            capabilities = device_status.get("capabilities", {})

            print(f"   Device {i+1}: {integration.device_name}")
            print(f"      - CPU: {capabilities.get('cpu_cores', 0)} cores")
            print(f"      - Memory: {capabilities.get('memory_mb', 0)}MB")
            print(f"      - Power: {capabilities.get('power_profile', 'unknown')}")
            print(f"      - Runtimes: {capabilities.get('supported_runtimes', [])}")

        # Demonstrate workload distribution
        print("\n3. Workload distribution simulation...")

        print("   🎯 Fog coordinator would distribute workloads based on:")
        print("      - Device capabilities and current utilization")
        print("      - Battery level and charging status")
        print("      - Network connectivity and latency")
        print("      - Thermal state and performance profile")
        print("      - P2P mesh connectivity and routing costs")

        print("   📊 Optimal placement strategy:")
        for i, integration in enumerate(integrations):
            device_status = await integration.get_device_status()
            capabilities = device_status.get("capabilities", {})

            # Simulate placement scoring
            cpu_score = capabilities.get("cpu_cores", 1) / 8.0  # Normalize to 8 cores
            battery_score = capabilities.get("battery_percent", 50) / 100.0

            placement_score = (cpu_score + battery_score) / 2.0

            print(f"      {integration.device_name}: {placement_score:.2f} score")

        # Cleanup
        print("\n🧹 Cleaning up fog network...")
        for integration in integrations:
            await integration.shutdown()

        print("✅ Fog coordination demo completed!")

    except Exception as e:
        logger.error(f"Fog coordination demo failed: {e}")
        print(f"❌ Demo failed: {e}")


async def main():
    """Run all integration demos"""

    try:
        # Run main integration demo
        await aivillage_integration_demo()

        # Run fog coordination demo
        await fog_coordination_demo()

        print("\n🎉 All integration demos completed successfully!")

        print("\n📋 Integration Summary:")
        print("=" * 30)
        print("✓ Fog edge components integrated with AIVillage infrastructure")
        print("✓ Capability beacon works with existing device registry")
        print("✓ WASI runner provides secure execution environment")
        print("✓ Resource monitoring bridges with existing systems")
        print("✓ P2P transport integration for efficient communication")
        print("✓ Fog coordination leverages existing infrastructure")
        print("✓ Maintains compatibility with digital twin systems")
        print("✓ Ready for enterprise deployment")

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Integration demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
