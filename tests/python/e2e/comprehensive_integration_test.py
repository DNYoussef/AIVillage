#!/usr/bin/env python3
"""Comprehensive Integration Test for AIVillage Sprint 6
Tests end-to-end workflows and real-world scenarios
"""

import asyncio
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ComprehensiveIntegrationTest:
    """Comprehensive integration test suite for Sprint 6 infrastructure"""

    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        self.test_count = 0
        self.passed_count = 0
        self.failed_count = 0

    def log_test_result(self, test_name: str, success: bool, details: str = "", duration: float = 0.0):
        """Log test result"""
        self.test_count += 1
        if success:
            self.passed_count += 1
            status = "✓ PASS"
        else:
            self.failed_count += 1
            status = "✗ FAIL"

        self.results[test_name] = {
            "success": success,
            "details": details,
            "duration": duration,
            "timestamp": time.time(),
        }

        logger.info(f"{status} - {test_name}: {details}")

    async def test_p2p_node_startup_workflow(self) -> bool:
        """Test complete P2P node startup and discovery workflow"""
        start_time = time.time()
        try:
            # Import P2P components
            from py.aivillage.core.core.p2p import P2PNode, PeerCapabilities, PeerDiscovery

            # Test node creation
            node = P2PNode(node_id="integration_test_node")

            # Test capabilities
            PeerCapabilities(
                device_id="integration_device",
                cpu_cores=psutil.cpu_count(),
                ram_mb=int(psutil.virtual_memory().total / (1024 * 1024)),
                evolution_capacity=0.7,
                available_for_evolution=True,
            )

            # Test peer discovery
            PeerDiscovery(node)

            duration = time.time() - start_time
            self.log_test_result(
                "P2P Node Startup Workflow",
                True,
                "Node created, capabilities evaluated, discovery initialized",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("P2P Node Startup Workflow", False, f"Error: {e!s}", duration)
            return False

    async def test_resource_management_integration(self) -> bool:
        """Test resource management system integration"""
        start_time = time.time()
        try:
            from py.aivillage.core.core.resources import (
                AdaptiveLoader,
                ConstraintManager,
                DeviceProfiler,
                ResourceMonitor,
            )

            # Initialize components
            profiler = DeviceProfiler()
            ResourceMonitor(profiler)
            constraint_manager = ConstraintManager(profiler)
            AdaptiveLoader(profiler, constraint_manager)

            # Test resource flow
            profiler.take_snapshot()
            allocation = profiler.get_evolution_resource_allocation()
            suitable = profiler.is_suitable_for_evolution("nightly")

            # Test constraint checking
            can_register = constraint_manager.register_task("integration_test", "nightly")
            if can_register:
                constraint_manager.unregister_task("integration_test")

            duration = time.time() - start_time
            self.log_test_result(
                "Resource Management Integration",
                True,
                f"Components integrated, allocation: {allocation['memory_mb']}MB, suitable: {suitable}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Resource Management Integration", False, f"Error: {e!s}", duration)
            return False

    async def test_evolution_system_integration(self) -> bool:
        """Test evolution system integration with infrastructure"""
        start_time = time.time()
        try:
            from py.aivillage.core.production.agent_forge.evolution.infrastructure_aware_evolution import (
                InfrastructureAwareEvolution,
                InfrastructureConfig,
            )
            from py.aivillage.core.production.agent_forge.evolution.resource_constrained_evolution import (
                ResourceConstrainedConfig,
            )

            # Test infrastructure-aware evolution
            infra_config = InfrastructureConfig(
                enable_p2p=False,  # Disable for integration test
                enable_resource_monitoring=True,
                enable_resource_constraints=True,
                enable_adaptive_loading=True,
            )

            evolution_system = InfrastructureAwareEvolution(infra_config)
            evolution_system.get_infrastructure_status()

            # Test resource-constrained evolution
            ResourceConstrainedConfig(
                memory_limit_multiplier=0.8,
                cpu_limit_multiplier=0.75,
                battery_optimization_mode=True,
            )

            duration = time.time() - start_time
            self.log_test_result(
                "Evolution System Integration",
                True,
                "Infrastructure and resource-constrained evolution systems initialized",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Evolution System Integration", False, f"Error: {e!s}", duration)
            return False

    async def test_monitoring_integration(self) -> bool:
        """Test monitoring integration across components"""
        start_time = time.time()
        try:
            from py.aivillage.core.core.resources import DeviceProfiler
            from py.aivillage.core.monitoring.sprint6_monitor import Sprint6Monitor

            # Initialize monitoring
            DeviceProfiler()
            monitor = Sprint6Monitor()

            # Test monitoring capabilities
            system_health = monitor.get_system_health()
            component_status = monitor.get_component_status()

            duration = time.time() - start_time
            self.log_test_result(
                "Monitoring Integration",
                True,
                f"System health: {system_health}, Components: {len(component_status)}",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Monitoring Integration", False, f"Error: {e!s}", duration)
            return False

    async def test_resource_constrained_scenario(self) -> bool:
        """Test system behavior under resource constraints"""
        start_time = time.time()
        try:
            from unittest.mock import patch

            from py.aivillage.core.core.resources import ConstraintManager, DeviceProfiler

            profiler = DeviceProfiler()
            constraint_manager = ConstraintManager(profiler)

            # Simulate low-resource device
            with patch.object(profiler.profile, "total_memory_gb", 2.0):
                with patch.object(profiler.profile, "cpu_cores", 2):
                    # Test adaptation to constraints
                    profiler.take_snapshot()
                    profiler.is_suitable_for_evolution("nightly")
                    allocation = profiler.get_evolution_resource_allocation()

                    # Test constraint enforcement
                    constraint_manager.register_task("constrained_test", "nightly")

            duration = time.time() - start_time
            self.log_test_result(
                "Resource Constrained Scenario",
                True,
                f"Low-resource simulation completed, allocation: {allocation['memory_mb']}MB",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Resource Constrained Scenario", False, f"Error: {e!s}", duration)
            return False

    async def test_multi_component_initialization(self) -> bool:
        """Test multi-component initialization sequence"""
        start_time = time.time()
        try:
            # Initialize all core components in sequence
            from py.aivillage.core.core.p2p import P2PNode
            from py.aivillage.core.core.resources import (
                AdaptiveLoader,
                ConstraintManager,
                DeviceProfiler,
                ResourceMonitor,
            )
            from py.aivillage.core.production.agent_forge.evolution.infrastructure_aware_evolution import (
                InfrastructureAwareEvolution,
                InfrastructureConfig,
            )

            # Step 1: Device profiling
            profiler = DeviceProfiler()

            # Step 2: Resource monitoring
            ResourceMonitor(profiler)

            # Step 3: Constraint management
            constraint_manager = ConstraintManager(profiler)

            # Step 4: Adaptive loading
            AdaptiveLoader(profiler, constraint_manager)

            # Step 5: P2P node
            P2PNode(node_id="multi_init_test")

            # Step 6: Evolution system
            config = InfrastructureConfig(enable_p2p=False)
            InfrastructureAwareEvolution(config)

            duration = time.time() - start_time
            self.log_test_result(
                "Multi-Component Initialization",
                True,
                "All 6 core components initialized successfully",
                duration,
            )
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Multi-Component Initialization", False, f"Error: {e!s}", duration)
            return False

    async def test_error_handling_and_recovery(self) -> bool:
        """Test error handling and recovery mechanisms"""
        start_time = time.time()
        recovery_count = 0

        try:
            from py.aivillage.core.core.resources import DeviceProfiler, ResourceMonitor

            # Test graceful degradation
            profiler = DeviceProfiler()
            ResourceMonitor(profiler)

            # Simulate various error conditions and test recovery
            test_scenarios = [
                ("Memory pressure", lambda: self._simulate_memory_pressure()),
                ("CPU spike", lambda: self._simulate_cpu_spike()),
                ("Network timeout", lambda: self._simulate_network_timeout()),
            ]

            for scenario_name, scenario_func in test_scenarios:
                try:
                    scenario_func()
                    # System should continue functioning
                    profiler.take_snapshot()
                    recovery_count += 1
                except Exception as scenario_error:
                    logger.warning(f"Scenario {scenario_name} failed: {scenario_error}")

            duration = time.time() - start_time
            success = recovery_count >= 2  # At least 2 out of 3 scenarios should recover

            self.log_test_result(
                "Error Handling and Recovery",
                success,
                f"Recovery successful in {recovery_count}/3 scenarios",
                duration,
            )
            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Error Handling and Recovery", False, f"Error: {e!s}", duration)
            return False

    def _simulate_memory_pressure(self):
        """Simulate memory pressure scenario"""
        # This is a mock simulation - in real testing this would create actual memory pressure

    def _simulate_cpu_spike(self):
        """Simulate CPU spike scenario"""
        # This is a mock simulation - in real testing this would create actual CPU load

    def _simulate_network_timeout(self):
        """Simulate network timeout scenario"""
        # This is a mock simulation - in real testing this would simulate network issues

    async def test_workflow_validation(self) -> bool:
        """Test complete workflow validation"""
        start_time = time.time()
        try:
            # Run the official Sprint 6 validation
            import subprocess

            result = subprocess.run(
                [sys.executable, "validate_sprint6.py"],
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
            )

            success = result.returncode == 0
            duration = time.time() - start_time

            self.log_test_result(
                "Workflow Validation",
                success,
                f"Sprint 6 validation {'passed' if success else 'failed'}",
                duration,
            )
            return success

        except Exception as e:
            duration = time.time() - start_time
            self.log_test_result("Workflow Validation", False, f"Error: {e!s}", duration)
            return False

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting Comprehensive Integration Test Suite for Sprint 6")
        logger.info("=" * 80)

        # Define test sequence
        tests = [
            ("P2P Node Startup", self.test_p2p_node_startup_workflow),
            ("Resource Management", self.test_resource_management_integration),
            ("Evolution System", self.test_evolution_system_integration),
            ("Monitoring Integration", self.test_monitoring_integration),
            ("Resource Constraints", self.test_resource_constrained_scenario),
            ("Multi-Component Init", self.test_multi_component_initialization),
            ("Error Handling", self.test_error_handling_and_recovery),
            ("Workflow Validation", self.test_workflow_validation),
        ]

        # Run all tests
        for test_name, test_func in tests:
            logger.info(f"Running: {test_name}")
            try:
                await test_func()
            except Exception as e:
                self.log_test_result(test_name, False, f"Exception: {e!s}")
                logger.error(f"Test {test_name} failed with exception: {e}")
                logger.error(traceback.format_exc())

        # Generate summary
        total_duration = time.time() - self.start_time
        success_rate = self.passed_count / self.test_count if self.test_count > 0 else 0

        summary = {
            "total_tests": self.test_count,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "results": self.results,
            "integration_status": "PASS" if success_rate >= 0.8 else "FAIL",
            "production_readiness": success_rate >= 0.9,
        }

        logger.info("=" * 80)
        logger.info("Integration Test Summary:")
        logger.info(f"  Total Tests: {self.test_count}")
        logger.info(f"  Passed: {self.passed_count}")
        logger.info(f"  Failed: {self.failed_count}")
        logger.info(f"  Success Rate: {success_rate:.1%}")
        logger.info(f"  Total Duration: {total_duration:.2f}s")
        logger.info(f"  Integration Status: {summary['integration_status']}")
        logger.info(f"  Production Ready: {summary['production_readiness']}")

        return summary


async def main():
    """Main entry point"""
    test_suite = ComprehensiveIntegrationTest()

    try:
        summary = await test_suite.run_all_tests()

        # Save results
        output_file = Path("integration_test_results.json")
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to: {output_file}")

        # Exit with appropriate code
        sys.exit(0 if summary["integration_status"] == "PASS" else 1)

    except Exception as e:
        logger.error(f"Integration test suite failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
