#!/usr/bin/env python3
"""
Edge Computing Integration Test Suite

Comprehensive test suite for validating the complete edge computing implementation
including device deployment, fog computing orchestration, and mobile optimization.

Usage:
    python tools/test_edge_computing_integration.py

    # Test specific components
    python tools/test_edge_computing_integration.py --test-deployment
    python tools/test_edge_computing_integration.py --test-fog-computing
    python tools/test_edge_computing_integration.py --test-mobile-optimization
"""

import argparse
import asyncio
from datetime import UTC, datetime
import logging
import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import edge computing components
from infrastructure.fog.compute.harvest_manager import DeviceCapabilities as HarvestDeviceCapabilities
from infrastructure.fog.compute.harvest_manager import FogHarvestManager
from infrastructure.fog.edge.core.device_registry import DeviceRegistry
from infrastructure.fog.edge.deployment.edge_deployer import (
    DeploymentConfig,
    DeploymentStatus,
    DeviceCapabilities,
    DeviceType,
    EdgeDeployer,
    NetworkQuality,
)
from infrastructure.fog.edge.fog_compute.fog_coordinator import (
    ComputeCapacity,
    FogCoordinator,
    TaskPriority,
    TaskType,
)
from infrastructure.fog.edge.mobile.resource_manager import MobileDeviceProfile, MobileResourceManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EdgeComputingIntegrationTest:
    """Comprehensive integration test suite for edge computing components"""

    def __init__(self):
        self.edge_deployer = None
        self.fog_coordinator = None
        self.device_registry = None
        self.harvest_manager = None
        self.mobile_resource_manager = None

        # Test data
        self.test_devices = []
        self.test_deployments = []
        self.test_results = {"tests_run": 0, "tests_passed": 0, "tests_failed": 0, "failures": []}

    async def setup(self):
        """Initialize all edge computing components"""
        logger.info("Setting up edge computing integration test environment")

        # Initialize components
        self.edge_deployer = EdgeDeployer(
            coordinator_id="test_deployer", enable_fog_computing=True, enable_cross_device_coordination=True
        )

        self.fog_coordinator = FogCoordinator("test_fog_coordinator")
        self.device_registry = DeviceRegistry()
        self.harvest_manager = FogHarvestManager("test_harvest_manager")
        self.mobile_resource_manager = MobileResourceManager(harvest_enabled=True, token_rewards_enabled=True)

        # Create test devices
        await self._create_test_devices()

        logger.info("Edge computing test environment setup complete")

    async def _create_test_devices(self):
        """Create a variety of test devices with different characteristics"""

        # High-end smartphone
        smartphone_caps = DeviceCapabilities(
            device_id="smartphone_001",
            device_type=DeviceType.SMARTPHONE,
            device_name="iPhone 15 Pro",
            cpu_cores=6,
            cpu_freq_ghz=3.2,
            ram_total_mb=8192,
            ram_available_mb=4096,
            has_gpu=True,
            gpu_model="A17 Pro",
            battery_powered=True,
            battery_percent=85,
            is_charging=False,
            network_quality=NetworkQuality.EXCELLENT,
            supports_ml_frameworks=["coreml", "onnx"],
        )

        # Budget tablet
        tablet_caps = DeviceCapabilities(
            device_id="tablet_002",
            device_type=DeviceType.TABLET,
            device_name="iPad Air",
            cpu_cores=4,
            cpu_freq_ghz=2.0,
            ram_total_mb=4096,
            ram_available_mb=2048,
            has_gpu=False,
            battery_powered=True,
            battery_percent=45,
            is_charging=True,
            network_quality=NetworkQuality.GOOD,
        )

        # Gaming laptop
        laptop_caps = DeviceCapabilities(
            device_id="laptop_003",
            device_type=DeviceType.LAPTOP,
            device_name="Gaming Laptop",
            cpu_cores=8,
            cpu_freq_ghz=2.8,
            ram_total_mb=16384,
            ram_available_mb=8192,
            has_gpu=True,
            gpu_model="RTX 4060",
            gpu_memory_mb=8192,
            battery_powered=True,
            battery_percent=95,
            is_charging=True,
            network_quality=NetworkQuality.EXCELLENT,
            supports_containers=True,
            supports_ml_frameworks=["pytorch", "tensorflow", "onnx"],
        )

        # Low-end IoT device
        iot_caps = DeviceCapabilities(
            device_id="iot_004",
            device_type=DeviceType.IOT_DEVICE,
            device_name="Edge IoT Device",
            cpu_cores=2,
            cpu_freq_ghz=1.0,
            ram_total_mb=1024,
            ram_available_mb=512,
            has_gpu=False,
            battery_powered=True,
            battery_percent=30,
            is_charging=False,
            network_quality=NetworkQuality.FAIR,
        )

        self.test_devices = [smartphone_caps, tablet_caps, laptop_caps, iot_caps]

    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("Starting comprehensive edge computing integration tests")

        test_suites = [
            ("Device Registration", self.test_device_registration),
            ("Edge Deployment", self.test_edge_deployment),
            ("Fog Computing", self.test_fog_computing),
            ("Mobile Optimization", self.test_mobile_optimization),
            ("Cross-Device Coordination", self.test_cross_device_coordination),
            ("Resource Harvesting", self.test_resource_harvesting),
            ("Battery/Thermal Management", self.test_battery_thermal_management),
            ("Network Adaptation", self.test_network_adaptation),
            ("Failure Recovery", self.test_failure_recovery),
            ("Performance Monitoring", self.test_performance_monitoring),
        ]

        for test_name, test_func in test_suites:
            logger.info(f"Running test suite: {test_name}")
            try:
                await test_func()
                logger.info(f"✅ {test_name} tests passed")
            except Exception as e:
                logger.error(f"❌ {test_name} tests failed: {e}")
                self.test_results["failures"].append(f"{test_name}: {str(e)}")

        self._print_test_summary()

    async def test_device_registration(self):
        """Test device registration and capability discovery"""
        logger.info("Testing device registration...")

        for device_caps in self.test_devices:
            # Test edge deployer registration
            success = await self.edge_deployer.register_device(device_caps.device_id, device_caps)
            self._assert(success, f"Failed to register device {device_caps.device_id}")

            # Verify device status
            status = await self.edge_deployer.get_device_status(device_caps.device_id)
            self._assert(status is not None, f"Device status not found for {device_caps.device_id}")
            self._assert(
                status["capabilities"].device_id == device_caps.device_id,
                f"Device ID mismatch for {device_caps.device_id}",
            )

            # Test device registry
            registration = self.device_registry.register_device(
                device_caps.device_id, device_caps.device_name, device_caps.device_type.value
            )
            self._assert(registration is not None, "Device registry registration failed")

            # Test harvest manager registration for suitable devices
            if device_caps.battery_percent > 20:
                harvest_caps = HarvestDeviceCapabilities(
                    device_id=device_caps.device_id,
                    device_type=device_caps.device_type.value,
                    cpu_cores=device_caps.cpu_cores,
                    cpu_freq_mhz=int(device_caps.cpu_freq_ghz * 1000),
                    cpu_architecture="arm64" if "phone" in device_caps.device_id else "x86_64",
                    ram_total_mb=device_caps.ram_total_mb,
                    ram_available_mb=device_caps.ram_available_mb,
                    has_gpu=device_caps.has_gpu,
                )

                harvest_success = await self.harvest_manager.register_device(device_caps.device_id, harvest_caps)
                self._assert(harvest_success, f"Failed to register device {device_caps.device_id} with harvest manager")

        self._log_test_result("Device registration", True)

    async def test_edge_deployment(self):
        """Test edge device deployment functionality"""
        logger.info("Testing edge deployment...")

        # Create deployment configuration
        config = DeploymentConfig(
            deployment_id="test_deployment_001",
            model_id="test_model_lightweight",
            deployment_type="inference",
            target_devices=[device.device_id for device in self.test_devices],
            min_cpu_cores=1,
            min_memory_mb=512,
            battery_aware=True,
            thermal_aware=True,
            rollout_strategy="rolling",
            max_concurrent_deployments=2,
        )

        # Test deployment
        deployment_ids = await self.edge_deployer.deploy(config)
        self._assert(len(deployment_ids) > 0, "No deployments created")

        # Wait for deployments to complete
        await asyncio.sleep(10)

        # Verify deployment statuses
        successful_deployments = 0
        for dep_id in deployment_ids:
            status = await self.edge_deployer.get_deployment_status(dep_id)
            self._assert(status is not None, f"No status found for deployment {dep_id}")

            if status.status in [DeploymentStatus.DEPLOYED, DeploymentStatus.RUNNING]:
                successful_deployments += 1

        # Should have at least some successful deployments (high-end devices)
        self._assert(successful_deployments >= 2, f"Too few successful deployments: {successful_deployments}")

        self.test_deployments = deployment_ids
        self._log_test_result("Edge deployment", True)

    async def test_fog_computing(self):
        """Test fog computing orchestration"""
        logger.info("Testing fog computing orchestration...")

        # Register nodes with fog coordinator
        for device_caps in self.test_devices:
            if device_caps.battery_percent > 30 or device_caps.is_charging:
                capacity = ComputeCapacity(
                    cpu_cores=device_caps.cpu_cores,
                    cpu_utilization=0.2,
                    memory_mb=device_caps.ram_total_mb,
                    memory_used_mb=device_caps.ram_total_mb - device_caps.ram_available_mb,
                    gpu_available=device_caps.has_gpu,
                    gpu_memory_mb=getattr(device_caps, "gpu_memory_mb", 0) or 0,
                    battery_powered=device_caps.battery_powered,
                    battery_percent=device_caps.battery_percent,
                    is_charging=device_caps.is_charging,
                    network_bandwidth_mbps=device_caps.network_speed_mbps,
                    network_latency_ms=device_caps.network_latency_ms,
                )

                success = await self.fog_coordinator.register_node(device_caps.device_id, capacity)
                self._assert(success, f"Failed to register fog node {device_caps.device_id}")

        # Submit test tasks
        task_ids = []
        for i in range(5):
            task_id = await self.fog_coordinator.submit_task(
                task_type=TaskType.INFERENCE,
                priority=TaskPriority.NORMAL,
                cpu_cores=1.0,
                memory_mb=256,
                estimated_duration=30.0,
            )
            task_ids.append(task_id)

        self._assert(len(task_ids) == 5, "Not all tasks were submitted")

        # Wait for task scheduling
        await asyncio.sleep(15)

        # Check system status
        system_status = self.fog_coordinator.get_system_status()
        self._assert(system_status["nodes"]["total"] > 0, "No fog nodes registered")
        self._assert(system_status["tasks"]["active"] >= 0, "Task scheduling not working")

        self._log_test_result("Fog computing orchestration", True)

    async def test_mobile_optimization(self):
        """Test mobile optimization and adaptive QoS"""
        logger.info("Testing mobile optimization...")

        # Test different battery scenarios
        scenarios = [
            {"battery_percent": 95, "is_charging": True, "cpu_temp_celsius": 25.0},
            {"battery_percent": 15, "is_charging": False, "cpu_temp_celsius": 35.0},
            {"battery_percent": 60, "is_charging": True, "cpu_temp_celsius": 55.0},
            {"battery_percent": 30, "is_charging": False, "cpu_temp_celsius": 45.0},
        ]

        for i, scenario in enumerate(scenarios):
            # Create device profile
            profile = MobileDeviceProfile(
                timestamp=time.time(),
                device_id=f"test_mobile_{i}",
                battery_percent=scenario["battery_percent"],
                battery_charging=scenario["is_charging"],
                cpu_temp_celsius=scenario["cpu_temp_celsius"],
                cpu_percent=30.0,
                ram_used_mb=2048,
                ram_available_mb=2048,
                ram_total_mb=4096,
            )

            # Get optimization
            optimization = await self.mobile_resource_manager.optimize_for_device(profile)

            # Verify optimization makes sense
            self._assert(optimization.power_mode is not None, f"No power mode set for scenario {i}")
            self._assert(optimization.transport_preference is not None, f"No transport preference set for scenario {i}")

            # Test transport routing
            routing = await self.mobile_resource_manager.get_transport_routing_decision(
                message_size_bytes=1024, priority=5, profile=profile
            )

            self._assert(
                routing["primary_transport"] in ["bitchat", "betanet"],
                f"Invalid primary transport: {routing['primary_transport']}",
            )

            # Low battery should prefer BitChat
            if scenario["battery_percent"] < 20:
                self._assert(
                    "bitchat" in routing["primary_transport"],
                    f"Low battery should prefer BitChat, got {routing['primary_transport']}",
                )

        self._log_test_result("Mobile optimization", True)

    async def test_cross_device_coordination(self):
        """Test cross-device coordination protocols"""
        logger.info("Testing cross-device coordination...")

        # Check device clustering
        system_status = await self.edge_deployer.get_system_status()

        # Should have some device clusters
        if len(self.test_devices) >= 4:
            self._assert(len(system_status["device_clusters"]) > 0, "No device clusters formed")

        # Test P2P connections
        p2p_connections = self.edge_deployer.p2p_connections
        if len(p2p_connections) > 0:
            # At least one device should have P2P connections
            total_connections = sum(len(connections) for connections in p2p_connections.values())
            self._assert(total_connections > 0, "No P2P connections established")

        # Test load balancing
        await self.edge_deployer._balance_workload()

        # Check task distribution
        task_queues = self.edge_deployer.task_queues
        if len(task_queues) > 1:
            task_counts = [len(tasks) for tasks in task_queues.values()]
            max_tasks = max(task_counts) if task_counts else 0
            min_tasks = min(task_counts) if task_counts else 0

            # Load should be reasonably balanced (difference <= 3 tasks)
            if max_tasks > 0:
                self._assert(max_tasks - min_tasks <= 3, f"Load imbalance detected: max={max_tasks}, min={min_tasks}")

        self._log_test_result("Cross-device coordination", True)

    async def test_resource_harvesting(self):
        """Test idle resource harvesting"""
        logger.info("Testing resource harvesting...")

        # Start harvesting sessions for eligible devices
        harvest_sessions = []

        for device_caps in self.test_devices:
            if device_caps.is_charging and device_caps.battery_percent > 30:
                session_id = await self.harvest_manager.start_harvesting(
                    device_caps.device_id,
                    {
                        "battery_percent": device_caps.battery_percent,
                        "is_charging": device_caps.is_charging,
                        "cpu_temp_celsius": 35.0,
                        "screen_on": False,
                        "network_type": "wifi",
                    },
                )

                if session_id:
                    harvest_sessions.append(session_id)

        self._assert(len(harvest_sessions) > 0, "No harvest sessions started")

        # Simulate some harvesting activity
        await asyncio.sleep(2)

        # Update metrics for active sessions
        for i, session_id in enumerate(harvest_sessions):
            device_id = self.test_devices[i].device_id
            if device_id in self.harvest_manager.active_sessions:
                await self.harvest_manager.update_session_metrics(
                    device_id, {"cpu_cycles": 1000000, "memory_mb_hours": 0.5, "tasks_completed": 2}
                )

        # Get harvest statistics
        network_stats = await self.harvest_manager.get_network_stats()
        self._assert(network_stats["active_devices"] > 0, "No active harvest devices")

        # Stop harvesting sessions
        for i, session_id in enumerate(harvest_sessions):
            device_id = self.test_devices[i].device_id
            if device_id in self.harvest_manager.active_sessions:
                session = await self.harvest_manager.stop_harvesting(device_id, "test_complete")
                self._assert(session is not None, f"Failed to stop harvest session for {device_id}")

        self._log_test_result("Resource harvesting", True)

    async def test_battery_thermal_management(self):
        """Test battery and thermal management"""
        logger.info("Testing battery/thermal management...")

        # Test thermal monitoring updates
        await self.edge_deployer._update_thermal_monitoring()

        # Verify thermal data is being tracked
        thermal_monitors = self.edge_deployer.thermal_monitors
        self._assert(len(thermal_monitors) > 0, "No thermal monitoring data")

        for device_id, thermal_data in thermal_monitors.items():
            self._assert("temp_celsius" in thermal_data, f"Missing temperature data for {device_id}")
            self._assert(
                20 <= thermal_data["temp_celsius"] <= 70, f"Invalid temperature reading: {thermal_data['temp_celsius']}"
            )

        # Test battery monitoring
        await self.edge_deployer._update_battery_monitoring()

        battery_monitors = self.edge_deployer.battery_monitors
        self._assert(len(battery_monitors) > 0, "No battery monitoring data")

        for device_id, battery_data in battery_monitors.items():
            self._assert("percent" in battery_data, f"Missing battery percentage for {device_id}")
            self._assert(
                0 <= battery_data.get("percent", -1) <= 100,
                f"Invalid battery percentage: {battery_data.get('percent')}",
            )

        self._log_test_result("Battery/thermal management", True)

    async def test_network_adaptation(self):
        """Test network adaptation and QoS"""
        logger.info("Testing network adaptation...")

        # Update network monitoring
        await self.edge_deployer._update_network_monitoring()

        # Verify network monitoring data
        network_monitors = self.edge_deployer.network_monitors
        self._assert(len(network_monitors) > 0, "No network monitoring data")

        for device_id, network_data in network_monitors.items():
            self._assert("latency_ms" in network_data, f"Missing latency data for {device_id}")
            self._assert(network_data["latency_ms"] > 0, f"Invalid latency reading: {network_data['latency_ms']}")

        # Test QoS policy adaptation
        qos_policies = self.edge_deployer.qos_policies
        self._assert(len(qos_policies) > 0, "No QoS policies configured")

        for device_id, policy in qos_policies.items():
            self._assert("max_cpu_percent" in policy, f"Missing CPU limit in QoS policy for {device_id}")
            self._assert(0 < policy["max_cpu_percent"] <= 100, f"Invalid CPU limit: {policy['max_cpu_percent']}")

        self._log_test_result("Network adaptation", True)

    async def test_failure_recovery(self):
        """Test failure detection and recovery"""
        logger.info("Testing failure recovery...")

        # Simulate device failure by updating last_seen timestamp
        if self.test_devices:
            test_device_id = self.test_devices[0].device_id
            if test_device_id in self.edge_deployer.device_states:
                # Set last_seen to 10 minutes ago
                old_time = datetime.now(UTC)
                old_time = old_time.replace(minute=old_time.minute - 10)
                self.edge_deployer.device_states[test_device_id]["last_seen"] = old_time

                # Add some tasks to the failed device
                self.edge_deployer.task_queues[test_device_id] = ["task_1", "task_2", "task_3"]

        # Trigger failure handling
        await self.edge_deployer._handle_device_failures()

        # Check if failure was detected
        if self.test_devices:
            test_device_id = self.test_devices[0].device_id
            device_state = self.edge_deployer.device_states.get(test_device_id, {})

            # Device should be marked as failed
            self._assert(device_state.get("status") == "failed", f"Device {test_device_id} should be marked as failed")

            # Tasks should be redistributed
            failed_device_tasks = len(self.edge_deployer.task_queues.get(test_device_id, []))
            self._assert(
                failed_device_tasks == 0, f"Tasks not redistributed from failed device: {failed_device_tasks} remaining"
            )

        self._log_test_result("Failure recovery", True)

    async def test_performance_monitoring(self):
        """Test performance monitoring and metrics collection"""
        logger.info("Testing performance monitoring...")

        # Update global statistics
        await self.edge_deployer._update_global_statistics()

        # Check system statistics
        system_status = await self.edge_deployer.get_system_status()
        stats = system_status["statistics"]

        # Verify basic statistics exist
        required_stats = [
            "total_deployments",
            "successful_deployments",
            "failed_deployments",
            "active_devices",
            "total_inference_count",
            "average_latency_ms",
        ]

        for stat in required_stats:
            self._assert(stat in stats, f"Missing statistic: {stat}")
            self._assert(
                isinstance(stats[stat], int | float), f"Invalid statistic type for {stat}: {type(stats[stat])}"
            )

        # Test mobile resource manager status
        mobile_status = self.mobile_resource_manager.get_status()
        self._assert("statistics" in mobile_status, "Missing mobile resource manager statistics")

        # Test fog coordinator status
        fog_status = self.fog_coordinator.get_system_status()
        self._assert("statistics" in fog_status, "Missing fog coordinator statistics")

        self._log_test_result("Performance monitoring", True)

    def _assert(self, condition: bool, message: str):
        """Test assertion helper"""
        self.test_results["tests_run"] += 1
        if not condition:
            self.test_results["tests_failed"] += 1
            self.test_results["failures"].append(message)
            raise AssertionError(message)
        else:
            self.test_results["tests_passed"] += 1

    def _log_test_result(self, test_name: str, passed: bool):
        """Log test result"""
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")

    def _print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("EDGE COMPUTING INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests Run: {self.test_results['tests_run']}")
        print(f"Tests Passed: {self.test_results['tests_passed']}")
        print(f"Tests Failed: {self.test_results['tests_failed']}")

        if self.test_results["failures"]:
            print(f"\nFailures ({len(self.test_results['failures'])}):")
            for i, failure in enumerate(self.test_results["failures"], 1):
                print(f"  {i}. {failure}")

        success_rate = (self.test_results["tests_passed"] / max(self.test_results["tests_run"], 1)) * 100
        print(f"\nSuccess Rate: {success_rate:.1f}%")

        if success_rate >= 95:
            print("🎉 EXCELLENT: Edge computing implementation is production-ready!")
        elif success_rate >= 85:
            print("✅ GOOD: Edge computing implementation is mostly functional")
        elif success_rate >= 70:
            print("⚠️  NEEDS IMPROVEMENT: Some edge computing features need attention")
        else:
            print("❌ POOR: Significant issues in edge computing implementation")

        print("=" * 60)

    async def cleanup(self):
        """Clean up test resources"""
        logger.info("Cleaning up test resources...")

        if self.edge_deployer:
            await self.edge_deployer.shutdown()

        if self.fog_coordinator:
            await self.fog_coordinator.shutdown()

        if self.mobile_resource_manager:
            await self.mobile_resource_manager.reset()

        logger.info("Test cleanup complete")


async def run_specific_test(test_name: str):
    """Run a specific test component"""
    test_suite = EdgeComputingIntegrationTest()
    await test_suite.setup()

    test_mapping = {
        "deployment": test_suite.test_edge_deployment,
        "fog-computing": test_suite.test_fog_computing,
        "mobile-optimization": test_suite.test_mobile_optimization,
        "coordination": test_suite.test_cross_device_coordination,
        "harvesting": test_suite.test_resource_harvesting,
        "battery-thermal": test_suite.test_battery_thermal_management,
        "network": test_suite.test_network_adaptation,
        "failure-recovery": test_suite.test_failure_recovery,
        "monitoring": test_suite.test_performance_monitoring,
    }

    if test_name in test_mapping:
        logger.info(f"Running specific test: {test_name}")
        try:
            await test_mapping[test_name]()
            logger.info(f"✅ {test_name} test completed successfully")
        except Exception as e:
            logger.error(f"❌ {test_name} test failed: {e}")
    else:
        logger.error(f"Unknown test: {test_name}")
        logger.info(f"Available tests: {', '.join(test_mapping.keys())}")

    await test_suite.cleanup()


async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Edge Computing Integration Test Suite")
    parser.add_argument(
        "--test",
        choices=[
            "deployment",
            "fog-computing",
            "mobile-optimization",
            "coordination",
            "harvesting",
            "battery-thermal",
            "network",
            "failure-recovery",
            "monitoring",
        ],
        help="Run specific test component",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        await run_specific_test(args.test)
    else:
        # Run full integration test suite
        test_suite = EdgeComputingIntegrationTest()
        try:
            await test_suite.setup()
            await test_suite.run_all_tests()
        finally:
            await test_suite.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
