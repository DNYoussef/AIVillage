"""
Example: Edge Device Integration

Demonstrates the complete edge device integration including:
- Capability beacon advertising device resources
- WASI runner executing WebAssembly jobs
- Resource monitoring and health assessment
- BetaNet integration for secure communication

This shows how an edge device joins the fog network and executes workloads.
"""

import asyncio
import logging

from packages.fog.edge import (
    CapabilityBeacon,
    DeviceType,
    ExecutionFabric,
    ExecutionResources,
    ResourceMonitor,
    RuntimeType,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_sample_wasm_module() -> bytes:
    """Create a simple WASM module for testing"""

    # This is a simple "Hello World" WASM module compiled from:
    # ```c
    # #include <stdio.h>
    # int main() {
    #     printf("Hello from WASI!\n");
    #     return 0;
    # }
    # ```

    # For this example, we'll create a mock WASM module
    # In practice, this would be real WebAssembly bytecode
    wasm_header = b"\x00asm\x01\x00\x00\x00"  # WASM magic number and version
    mock_wasm = wasm_header + b"\x00" * 100  # Placeholder content

    return mock_wasm


async def edge_device_simulation():
    """Simulate an edge device joining the fog network and executing jobs"""

    print("🌫️ Edge Device Fog Integration Demo")
    print("=" * 50)

    device_name = "dev-laptop-001"
    namespace = "aivillage/edge"

    # 1. Initialize Resource Monitor
    print("1. Initializing resource monitor...")

    monitor = ResourceMonitor(device_id=device_name, monitoring_interval=2.0, enable_profiling=True)

    # Set up callbacks for health changes
    async def on_health_change(old_status, new_status):
        print(f"   🏥 Health changed: {old_status} → {new_status}")

    async def on_thermal_change(old_state, new_state):
        print(f"   🌡️ Thermal state: {old_state} → {new_state}")

    async def on_critical_resource(conditions, snapshot):
        print(f"   ⚠️ Critical resources: {conditions}")

    monitor.on_health_change = on_health_change
    monitor.on_thermal_change = on_thermal_change
    monitor.on_critical_resource = on_critical_resource

    await monitor.start_monitoring()

    # Give monitor time to collect baseline
    await asyncio.sleep(3)

    status = monitor.get_current_status()
    print(f"   ✓ Device status: {status['health_status']}")
    print(f"   ✓ CPU: {status['cpu_percent']:.1f}%, Memory: {status['memory_percent']:.1f}%")
    print(
        f"   ✓ Available: {status['available_resources']['cpu_cores']} cores, "
        f"{status['available_resources']['memory_mb']}MB RAM"
    )

    # 2. Initialize Capability Beacon
    print("\n2. Starting capability beacon...")

    beacon = CapabilityBeacon(
        device_name=device_name,
        operator_namespace=namespace,
        device_type=DeviceType.LAPTOP,
        betanet_endpoint="betanet://dev-laptop-001.local:7337",
        advertisement_interval=10.0,
    )

    # Set up beacon callbacks
    async def on_capability_changed(capability):
        print(
            f"   📡 Capability updated: {capability.cpu_cores} cores, "
            f"{capability.memory_mb}MB, power: {capability.power_profile}"
        )

    beacon.on_capability_changed = on_capability_changed

    # Add mock fog gateway
    beacon.add_gateway("https://fog-gateway.aivillage.dev")

    await beacon.start()

    capability = beacon.get_capability()
    print(f"   ✓ Advertising: {len(capability.supported_runtimes)} runtimes")
    print(f"   ✓ Runtimes: {', '.join(r.value for r in capability.supported_runtimes)}")
    print(f"   ✓ Resources: {capability.cpu_cores} cores, {capability.memory_mb}MB")

    # 3. Initialize Execution Fabric
    print("\n3. Setting up execution environment...")

    execution_fabric = ExecutionFabric()

    supported_runtimes = await execution_fabric.get_supported_runtimes()
    print(f"   ✓ Supported runtimes: {[r.value for r in supported_runtimes]}")

    # Update beacon with actual runtime capabilities
    beacon.capability.supported_runtimes = set(supported_runtimes)

    # 4. Simulate Job Execution
    print("\n4. Executing sample WASI job...")

    if RuntimeType.WASI in supported_runtimes:
        # Create sample WASM module
        wasm_module = await create_sample_wasm_module()

        # Define job resources
        job_resources = ExecutionResources(
            cpu_cores=1.0, memory_mb=256, disk_mb=100, max_duration_s=30, network_egress=False
        )

        # Check if device is suitable for this workload
        suitable, issues = monitor.is_suitable_for_workload(cpu_requirement=1.0, memory_mb=256, duration_s=30)

        if suitable:
            print("   ✓ Device suitable for workload")

            # Execute WASM job
            print("   🔄 Executing WASM module...")

            # Update active job count
            beacon.update_job_count(1)

            try:
                result = await execution_fabric.execute(
                    runtime_type=RuntimeType.WASI,
                    payload=wasm_module,
                    args=["hello", "world"],
                    env={"FOG_NODE": device_name},
                    resources=job_resources,
                )

                print(f"   ✓ Execution completed: {result.status}")
                print(f"   ✓ Exit code: {result.exit_code}")
                print(f"   ✓ Duration: {result.duration_ms:.1f}ms")
                print(f"   ✓ Memory peak: {result.memory_peak_mb}MB")
                print(f"   ✓ CPU time: {result.cpu_time_s:.2f}s")

                if result.stdout:
                    print(f"   📄 Output: {result.stdout}")

                if result.resource_violations:
                    print(f"   ⚠️ Resource violations: {result.resource_violations}")

            except Exception as e:
                print(f"   ❌ Execution failed: {e}")

            finally:
                # Reset active job count
                beacon.update_job_count(0)

        else:
            print(f"   ❌ Device not suitable: {issues}")

    else:
        print("   ℹ️ WASI runtime not available, skipping job execution")

    # 5. Demonstrate Resource Monitoring
    print("\n5. Resource monitoring demonstration...")

    # Get recent CPU usage history
    cpu_history = monitor.get_historical_data(hours=1, metric="cpu_percent")
    if cpu_history:
        recent_cpu = [point[1] for point in cpu_history[-10:]]  # Last 10 samples
        avg_cpu = sum(recent_cpu) / len(recent_cpu)
        print(f"   📊 Average CPU (last 10 samples): {avg_cpu:.1f}%")

    # Get performance profile
    profile = monitor.get_performance_profile()
    if profile:
        print("   🏃 Performance profile:")
        print(f"      - CPU single-core score: {profile.cpu_single_core_score:.1f}")
        print(f"      - CPU multi-core score: {profile.cpu_multi_core_score:.1f}")
        print(f"      - Memory bandwidth: {profile.memory_bandwidth_mb_s:.0f} MB/s")

    # 6. Simulate Workload Scaling
    print("\n6. Workload scaling simulation...")

    # Check different workload sizes
    workloads = [
        {"name": "Light", "cpu": 0.5, "memory": 128, "duration": 60},
        {"name": "Medium", "cpu": 2.0, "memory": 512, "duration": 300},
        {"name": "Heavy", "cpu": 4.0, "memory": 2048, "duration": 600},
    ]

    for workload in workloads:
        suitable, issues = monitor.is_suitable_for_workload(
            cpu_requirement=workload["cpu"], memory_mb=workload["memory"], duration_s=workload["duration"]
        )

        status_icon = "✓" if suitable else "❌"
        print(
            f"   {status_icon} {workload['name']} workload "
            f"({workload['cpu']} cores, {workload['memory']}MB): "
            f"{'Suitable' if suitable else f'Issues: {issues}'}"
        )

    # 7. Peer Discovery Simulation
    print("\n7. Peer discovery simulation...")

    # Simulate discovering other edge devices
    discovered_peers = beacon.get_discovered_peers()
    print(f"   📡 Discovered peers: {len(discovered_peers)}")

    # In a real implementation, the beacon would discover actual peers
    # via mDNS and BetaNet mesh networking
    print("   ℹ️ Peer discovery via mDNS and BetaNet (simulated)")
    print("   ℹ️ Would discover other fog nodes in the mesh network")

    # 8. Gateway Integration
    print("\n8. Gateway integration status...")

    print(f"   🌐 Registered gateways: {len(beacon.fog_gateways)}")
    for gateway in beacon.fog_gateways:
        print(f"      - {gateway}")

    print("   ℹ️ In production, device would register with gateway")
    print("   ℹ️ Gateway would assign jobs based on capability and location")

    # 9. Monitoring Summary
    print("\n9. Final monitoring summary...")

    final_status = monitor.get_current_status()

    print("   📊 Final device status:")
    print(f"      - Health: {final_status['health_status']}")
    print(f"      - Thermal: {final_status['thermal_state']}")
    print(f"      - Battery: {final_status['battery_state']}")
    print(f"      - CPU usage: {final_status['cpu_percent']:.1f}%")
    print(f"      - Memory usage: {final_status['memory_percent']:.1f}%")
    print(f"      - Available resources: {final_status['available_resources']}")

    # Keep running for a bit to show continuous monitoring
    print("\n10. Continuous monitoring (10 seconds)...")

    for i in range(5):
        await asyncio.sleep(2)
        current_status = monitor.get_current_status()
        print(f"    📈 CPU: {current_status['cpu_percent']:.1f}%, " f"Memory: {current_status['memory_percent']:.1f}%")

    # Cleanup
    print("\n🧹 Cleaning up...")

    await beacon.stop()
    await monitor.stop_monitoring()

    print("✅ Edge device demo completed!")

    # Summary
    print("\n📋 Integration Summary:")
    print("=" * 30)
    print("✓ Resource monitoring with health assessment")
    print("✓ Capability beacon with runtime discovery")
    print("✓ WASI execution environment")
    print("✓ BetaNet integration points (simulated)")
    print("✓ Gateway registration (simulated)")
    print("✓ Workload suitability assessment")
    print("✓ Continuous monitoring and adaptation")

    print("\n🔗 Integration Points:")
    print("- BetaNet transport for secure job delivery")
    print("- P2P mesh for peer discovery")
    print("- Gateway API for job assignment")
    print("- Agent system for intelligent workload placement")
    print("- RAG system for knowledge sharing")


async def stress_test_simulation():
    """Simulate stress testing the edge device"""

    print("\n🔥 Stress Test Simulation")
    print("=" * 30)

    execution_fabric = ExecutionFabric()

    if RuntimeType.WASI not in await execution_fabric.get_supported_runtimes():
        print("WASI runtime not available for stress test")
        return

    # Create stress test WASM module (CPU intensive)
    stress_wasm = await create_sample_wasm_module()

    # Run multiple concurrent jobs
    print("Starting 3 concurrent WASI jobs...")

    tasks = []
    for i in range(3):
        task = asyncio.create_task(
            execution_fabric.execute(
                runtime_type=RuntimeType.WASI,
                payload=stress_wasm,
                args=[f"worker-{i}"],
                resources=ExecutionResources(cpu_cores=0.5, memory_mb=128, max_duration_s=10),
            )
        )
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    print("Stress test results:")
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Job {i}: Failed - {result}")
        else:
            print(f"  Job {i}: {result.status} (exit: {result.exit_code}, " f"duration: {result.duration_ms:.1f}ms)")


async def main():
    """Run the complete edge integration demo"""

    try:
        # Run main edge device simulation
        await edge_device_simulation()

        # Run stress test
        await stress_test_simulation()

    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
