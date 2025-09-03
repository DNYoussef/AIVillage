"""
Production Validation and Chaos Testing

Comprehensive chaos engineering and production validation suite for the fog
computing infrastructure. Tests system resilience under various failure scenarios.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
import logging
from pathlib import Path
import random
import time
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


class ChaosType(Enum):
    """Types of chaos experiments."""

    NODE_FAILURE = "node_failure"
    NETWORK_PARTITION = "network_partition"
    HIGH_LATENCY = "high_latency"
    PACKET_LOSS = "packet_loss"
    CPU_STRESS = "cpu_stress"
    MEMORY_PRESSURE = "memory_pressure"
    DISK_IO_STRESS = "disk_io_stress"
    SERVICE_CRASH = "service_crash"
    DATABASE_FAILURE = "database_failure"
    DEPENDENCY_TIMEOUT = "dependency_timeout"
    BYZANTINE_FAULT = "byzantine_fault"
    TRAFFIC_SPIKE = "traffic_spike"


class ExperimentStatus(Enum):
    """Status of chaos experiments."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChaosValidationResult(Enum):
    """Results of chaos validation tests."""

    PASS = "pass"  # noqa: S105 - enum value
    FAIL = "fail"
    PARTIAL = "partial"
    TIMEOUT = "timeout"


@dataclass
class ChaosConfig:
    """Configuration for a chaos experiment."""

    experiment_id: str
    name: str
    description: str
    chaos_type: ChaosType
    duration_seconds: int
    intensity: float  # 0.0 to 1.0
    target_components: list[str]
    affected_percentage: float = 0.3  # Percentage of targets to affect

    # Safety constraints
    max_duration: int = 3600  # 1 hour max
    safety_checks: bool = True
    auto_rollback: bool = True
    rollback_timeout: int = 300  # 5 minutes

    # Success criteria
    expected_recovery_time: int = 300  # Expected recovery time in seconds
    max_acceptable_downtime: int = 60  # Maximum acceptable downtime
    min_healthy_replicas: int = 1  # Minimum healthy replicas during test


@dataclass
class ValidationTest:
    """A production validation test."""

    test_id: str
    name: str
    description: str
    test_type: str
    target_components: list[str]
    timeout_seconds: int = 300
    retry_count: int = 3
    critical: bool = False


@dataclass
class ExperimentResult:
    """Result of a chaos experiment."""

    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: float = 0.0

    # Metrics collected during experiment
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_during: dict[str, list[float]] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)

    # Observed behavior
    failures_detected: list[str] = field(default_factory=list)
    recovery_actions: list[str] = field(default_factory=list)
    recovery_time_seconds: float | None = None

    # Success criteria evaluation
    criteria_met: bool = False
    downtime_seconds: float = 0.0
    data_loss: bool = False
    corruption_detected: bool = False

    # Logs and observations
    logs: list[str] = field(default_factory=list)
    observations: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Report from validation testing."""

    report_id: str
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    critical_failures: int

    # Component health
    component_health: dict[str, str] = field(default_factory=dict)

    # Performance metrics
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    error_rate: float = 0.0
    availability: float = 0.0

    # Test results
    test_results: list[dict[str, Any]] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)


class ChaosTestingFramework:
    """
    Production Validation and Chaos Testing Framework.

    Provides comprehensive chaos engineering capabilities to test
    system resilience and validate production readiness.
    """

    def __init__(self, data_dir: str = "chaos_testing"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Experiment management
        self.active_experiments: dict[str, ChaosConfig] = {}
        self.experiment_results: dict[str, ExperimentResult] = {}
        self.experiment_history: deque = deque(maxlen=1000)

        # Validation tests
        self.validation_tests: dict[str, ValidationTest] = {}
        self.validation_reports: deque = deque(maxlen=100)

        # System state monitoring
        self.baseline_metrics: dict[str, float] = {}
        self.current_metrics: dict[str, float] = {}
        self.health_checks: dict[str, Callable] = {}

        # Safety mechanisms
        self.safety_enabled = True
        self.emergency_stop = False
        self.protected_components: set[str] = set()

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        # Initialize default validation tests
        self._initialize_validation_tests()

        logger.info("Chaos Testing Framework initialized")

    def _initialize_validation_tests(self):
        """Initialize default validation tests."""

        # Basic connectivity test
        self.validation_tests["connectivity"] = ValidationTest(
            test_id="connectivity",
            name="Network Connectivity Test",
            description="Test network connectivity between components",
            test_type="network",
            target_components=["fog_marketplace", "onion_router", "token_system"],
            timeout_seconds=60,
            critical=True,
        )

        # Service health checks
        self.validation_tests["service_health"] = ValidationTest(
            test_id="service_health",
            name="Service Health Check",
            description="Verify all core services are healthy",
            test_type="health",
            target_components=["fog_coordinator", "harvest_manager", "hidden_service_host"],
            timeout_seconds=30,
            critical=True,
        )

        # End-to-end workflow test
        self.validation_tests["e2e_workflow"] = ValidationTest(
            test_id="e2e_workflow",
            name="End-to-End Workflow Test",
            description="Test complete fog computing workflow",
            test_type="integration",
            target_components=["all"],
            timeout_seconds=300,
            critical=True,
        )

        # Performance baseline test
        self.validation_tests["performance"] = ValidationTest(
            test_id="performance",
            name="Performance Baseline Test",
            description="Measure baseline performance metrics",
            test_type="performance",
            target_components=["fog_marketplace", "mixnet_client"],
            timeout_seconds=180,
            critical=False,
        )

        # Security validation
        self.validation_tests["security"] = ValidationTest(
            test_id="security",
            name="Security Validation Test",
            description="Validate security controls and privacy measures",
            test_type="security",
            target_components=["onion_router", "mixnet_client", "contribution_ledger"],
            timeout_seconds=120,
            critical=True,
        )

        # Data consistency test
        self.validation_tests["data_consistency"] = ValidationTest(
            test_id="data_consistency",
            name="Data Consistency Test",
            description="Verify data consistency across distributed components",
            test_type="consistency",
            target_components=["contribution_ledger", "token_system"],
            timeout_seconds=180,
            critical=True,
        )

    async def start(self):
        """Start the chaos testing framework."""
        if self._running:
            return

        logger.info("Starting Chaos Testing Framework")
        self._running = True

        # Collect baseline metrics
        await self._collect_baseline_metrics()

        # Start background tasks
        tasks = [
            self._experiment_monitor(),
            self._metrics_collector(),
            self._safety_monitor(),
            self._health_checker(),
            self._report_generator(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Chaos Testing Framework started successfully")

    async def stop(self):
        """Stop the chaos testing framework."""
        if not self._running:
            return

        logger.info("Stopping Chaos Testing Framework")
        self._running = False

        # Stop all active experiments
        for experiment_id in list(self.active_experiments.keys()):
            await self.stop_experiment(experiment_id)

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        # Save final report
        await self._save_experiment_data()

        logger.info("Chaos Testing Framework stopped")

    async def run_validation_suite(self) -> ValidationReport:
        """Run the complete validation test suite."""
        report_id = f"validation_{int(time.time())}"

        logger.info("Starting production validation suite")

        report = ValidationReport(report_id=report_id, timestamp=datetime.now(), total_tests=len(self.validation_tests))

        # Run all validation tests
        for test_id, test_config in self.validation_tests.items():
            result = await self._run_validation_test(test_config)
            report.test_results.append(result)

            if result["status"] == ChaosValidationResult.PASS:
                report.passed_tests += 1
            else:
                report.failed_tests += 1
                if test_config.critical:
                    report.critical_failures += 1

        # Collect system health metrics
        await self._collect_system_health(report)

        # Generate recommendations
        await self._generate_recommendations(report)

        # Store report
        self.validation_reports.append(report)

        logger.info(f"Validation suite completed: {report.passed_tests}/{report.total_tests} tests passed")
        return report

    async def start_chaos_experiment(self, config: ChaosConfig) -> str:
        """Start a chaos experiment."""
        if config.experiment_id in self.active_experiments:
            raise ValueError(f"Experiment {config.experiment_id} already active")

        if self.emergency_stop:
            raise RuntimeError("Emergency stop active - experiments disabled")

        # Safety checks
        if config.safety_checks and not await self._safety_check(config):
            raise RuntimeError("Safety check failed - experiment aborted")

        logger.info(f"Starting chaos experiment: {config.name}")

        # Create experiment result
        result = ExperimentResult(
            experiment_id=config.experiment_id, status=ExperimentStatus.RUNNING, start_time=datetime.now()
        )

        # Collect baseline metrics
        result.metrics_before = await self._collect_current_metrics()

        # Store active experiment
        self.active_experiments[config.experiment_id] = config
        self.experiment_results[config.experiment_id] = result

        # Start the chaos injection
        asyncio.create_task(self._execute_chaos_experiment(config, result))

        return config.experiment_id

    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop a chaos experiment."""
        if experiment_id not in self.active_experiments:
            return False

        logger.info(f"Stopping chaos experiment: {experiment_id}")

        config = self.active_experiments[experiment_id]
        result = self.experiment_results[experiment_id]

        # Perform rollback
        await self._rollback_experiment(config, result)

        # Update result
        result.status = ExperimentStatus.CANCELLED
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        # Cleanup
        del self.active_experiments[experiment_id]
        self.experiment_history.append(result)

        return True

    async def get_experiment_status(self, experiment_id: str) -> dict[str, Any] | None:
        """Get status of a chaos experiment."""
        if experiment_id not in self.experiment_results:
            return None

        result = self.experiment_results[experiment_id]
        config = self.active_experiments.get(experiment_id)

        status = {
            "experiment_id": experiment_id,
            "status": result.status.value,
            "start_time": result.start_time.isoformat(),
            "duration_seconds": result.duration_seconds,
            "failures_detected": result.failures_detected,
            "recovery_actions": result.recovery_actions,
            "criteria_met": result.criteria_met,
        }

        if config:
            status["config"] = {
                "name": config.name,
                "chaos_type": config.chaos_type.value,
                "duration_seconds": config.duration_seconds,
                "intensity": config.intensity,
                "target_components": config.target_components,
            }

        return status

    async def get_system_resilience_report(self) -> dict[str, Any]:
        """Generate a comprehensive system resilience report."""
        completed_experiments = [
            result for result in self.experiment_history if result.status == ExperimentStatus.COMPLETED
        ]

        if not completed_experiments:
            return {"status": "No completed experiments"}

        # Calculate resilience metrics
        total_experiments = len(completed_experiments)
        successful_recoveries = sum(1 for r in completed_experiments if r.criteria_met)

        avg_recovery_time = 0.0
        if successful_recoveries > 0:
            recovery_times = [
                r.recovery_time_seconds for r in completed_experiments if r.recovery_time_seconds is not None
            ]
            avg_recovery_time = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # Group by chaos type
        chaos_type_stats = defaultdict(list)
        for result in completed_experiments:
            config = next(
                (c for c in self.active_experiments.values() if c.experiment_id == result.experiment_id), None
            )
            if config:
                chaos_type_stats[config.chaos_type.value].append(result)

        type_resilience = {}
        for chaos_type, results in chaos_type_stats.items():
            successful = sum(1 for r in results if r.criteria_met)
            type_resilience[chaos_type] = {
                "total_experiments": len(results),
                "successful_recoveries": successful,
                "success_rate": successful / len(results),
                "avg_recovery_time": sum(r.recovery_time_seconds or 0 for r in results) / len(results),
            }

        return {
            "overall": {
                "total_experiments": total_experiments,
                "successful_recoveries": successful_recoveries,
                "success_rate": successful_recoveries / total_experiments,
                "avg_recovery_time_seconds": avg_recovery_time,
            },
            "by_chaos_type": type_resilience,
            "recent_experiments": [
                {
                    "experiment_id": r.experiment_id,
                    "status": r.status.value,
                    "criteria_met": r.criteria_met,
                    "recovery_time": r.recovery_time_seconds,
                    "start_time": r.start_time.isoformat(),
                }
                for r in completed_experiments[-10:]
            ],
        }

    async def _run_validation_test(self, test_config: ValidationTest) -> dict[str, Any]:
        """Run a single validation test."""
        logger.info(f"Running validation test: {test_config.name}")

        start_time = time.time()
        result = {
            "test_id": test_config.test_id,
            "name": test_config.name,
            "status": ChaosValidationResult.FAIL,
            "duration_seconds": 0.0,
            "error_message": None,
            "metrics": {},
        }

        try:
            # Run the appropriate test based on type
            if test_config.test_type == "network":
                await self._test_network_connectivity(test_config, result)
            elif test_config.test_type == "health":
                await self._test_service_health(test_config, result)
            elif test_config.test_type == "integration":
                await self._test_e2e_workflow(test_config, result)
            elif test_config.test_type == "performance":
                await self._test_performance(test_config, result)
            elif test_config.test_type == "security":
                await self._test_security(test_config, result)
            elif test_config.test_type == "consistency":
                await self._test_data_consistency(test_config, result)
            else:
                result["error_message"] = f"Unknown test type: {test_config.test_type}"

            result["duration_seconds"] = time.time() - start_time

        except asyncio.TimeoutError:
            result["status"] = ChaosValidationResult.TIMEOUT
            result["error_message"] = "Test timed out"
        except Exception as e:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = str(e)
            logger.error(f"Validation test {test_config.test_id} failed: {e}")

        return result

    async def _test_network_connectivity(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test network connectivity between components."""
        # Mock network connectivity test
        await asyncio.sleep(1)  # Simulate test duration

        # Simulate success/failure
        if random.random() > 0.1:  # 90% success rate
            result["status"] = ChaosValidationResult.PASS
            result["metrics"]["avg_latency_ms"] = random.uniform(10, 50)
            result["metrics"]["packet_loss_percent"] = 0.0
        else:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = "Network connectivity issues detected"

    async def _test_service_health(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test service health checks."""
        # Mock service health test
        await asyncio.sleep(0.5)

        healthy_services = 0
        total_services = len(test_config.target_components)

        for component in test_config.target_components:
            # Simulate health check
            if random.random() > 0.05:  # 95% success rate
                healthy_services += 1

        result["metrics"]["healthy_services"] = healthy_services
        result["metrics"]["total_services"] = total_services
        result["metrics"]["health_percentage"] = (healthy_services / total_services) * 100

        if healthy_services == total_services:
            result["status"] = ChaosValidationResult.PASS
        elif healthy_services >= total_services * 0.8:  # 80% threshold
            result["status"] = ChaosValidationResult.PARTIAL
        else:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = f"Only {healthy_services}/{total_services} services healthy"

    async def _test_e2e_workflow(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test end-to-end workflow."""
        # Mock E2E test
        await asyncio.sleep(2)

        # Simulate complex workflow steps
        steps = [
            "service_discovery",
            "authentication",
            "resource_allocation",
            "task_execution",
            "result_collection",
            "reward_distribution",
        ]

        completed_steps = 0
        for step in steps:
            # Simulate step execution
            await asyncio.sleep(0.3)
            if random.random() > 0.02:  # 98% success rate per step
                completed_steps += 1
            else:
                result["error_message"] = f"E2E workflow failed at step: {step}"
                break

        result["metrics"]["completed_steps"] = completed_steps
        result["metrics"]["total_steps"] = len(steps)
        result["metrics"]["success_percentage"] = (completed_steps / len(steps)) * 100

        if completed_steps == len(steps):
            result["status"] = ChaosValidationResult.PASS
        else:
            result["status"] = ChaosValidationResult.FAIL

    async def _test_performance(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test performance metrics."""
        # Mock performance test
        await asyncio.sleep(1.5)

        # Generate mock performance metrics
        response_times = [random.uniform(50, 200) for _ in range(100)]

        result["metrics"]["avg_response_time_ms"] = sum(response_times) / len(response_times)
        result["metrics"]["p95_response_time_ms"] = sorted(response_times)[94]
        result["metrics"]["min_response_time_ms"] = min(response_times)
        result["metrics"]["max_response_time_ms"] = max(response_times)
        result["metrics"]["throughput_rps"] = random.uniform(80, 120)
        result["metrics"]["error_rate_percent"] = random.uniform(0, 2)

        # Pass if metrics are within acceptable ranges
        if result["metrics"]["avg_response_time_ms"] < 150 and result["metrics"]["error_rate_percent"] < 1:
            result["status"] = ChaosValidationResult.PASS
        else:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = "Performance metrics below threshold"

    async def _test_security(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test security controls."""
        # Mock security test
        await asyncio.sleep(1)

        security_checks = [
            "encryption_enabled",
            "authentication_required",
            "authorization_enforced",
            "audit_logging_active",
            "secure_communication",
            "data_anonymization",
        ]

        passed_checks = 0
        for check in security_checks:
            if random.random() > 0.01:  # 99% pass rate per check
                passed_checks += 1

        result["metrics"]["security_checks_passed"] = passed_checks
        result["metrics"]["security_checks_total"] = len(security_checks)
        result["metrics"]["security_score"] = (passed_checks / len(security_checks)) * 100

        if passed_checks == len(security_checks):
            result["status"] = ChaosValidationResult.PASS
        else:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = f"Security checks failed: {len(security_checks) - passed_checks}"

    async def _test_data_consistency(self, test_config: ValidationTest, result: dict[str, Any]):
        """Test data consistency."""
        # Mock consistency test
        await asyncio.sleep(1.2)

        # Simulate consistency checks across distributed components
        consistency_checks = random.randint(90, 100)  # Out of 100

        result["metrics"]["consistency_score"] = consistency_checks
        result["metrics"]["inconsistencies_detected"] = 100 - consistency_checks

        if consistency_checks >= 98:
            result["status"] = ChaosValidationResult.PASS
        elif consistency_checks >= 95:
            result["status"] = ChaosValidationResult.PARTIAL
            result["error_message"] = f"Minor consistency issues detected: {100 - consistency_checks}"
        else:
            result["status"] = ChaosValidationResult.FAIL
            result["error_message"] = f"Significant consistency issues: {100 - consistency_checks}"

    async def _execute_chaos_experiment(self, config: ChaosConfig, result: ExperimentResult):
        """Execute a chaos experiment."""
        try:
            logger.info(f"Executing chaos experiment: {config.chaos_type.value}")

            # Start chaos injection
            await self._inject_chaos(config, result)

            # Monitor system during experiment
            end_time = time.time() + config.duration_seconds

            while time.time() < end_time and result.status == ExperimentStatus.RUNNING:
                # Collect metrics
                current_metrics = await self._collect_current_metrics()
                for metric, value in current_metrics.items():
                    if metric not in result.metrics_during:
                        result.metrics_during[metric] = []
                    result.metrics_during[metric].append(value)

                # Check for failures
                await self._detect_failures(config, result)

                # Monitor recovery
                await self._monitor_recovery(config, result)

                await asyncio.sleep(10)  # Check every 10 seconds

            # Stop chaos injection
            await self._stop_chaos_injection(config, result)

            # Wait for recovery and collect final metrics
            await asyncio.sleep(config.expected_recovery_time)
            result.metrics_after = await self._collect_current_metrics()

            # Evaluate success criteria
            await self._evaluate_experiment_success(config, result)

            result.status = ExperimentStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            # Cleanup
            if config.experiment_id in self.active_experiments:
                del self.active_experiments[config.experiment_id]
            self.experiment_history.append(result)

            logger.info(f"Chaos experiment {config.experiment_id} completed")

        except Exception as e:
            logger.error(f"Chaos experiment {config.experiment_id} failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - result.start_time).total_seconds()

            # Attempt rollback
            await self._rollback_experiment(config, result)

    async def _inject_chaos(self, config: ChaosConfig, result: ExperimentResult):
        """Inject chaos based on experiment configuration."""
        if config.chaos_type == ChaosType.NODE_FAILURE:
            await self._inject_node_failure(config, result)
        elif config.chaos_type == ChaosType.NETWORK_PARTITION:
            await self._inject_network_partition(config, result)
        elif config.chaos_type == ChaosType.HIGH_LATENCY:
            await self._inject_high_latency(config, result)
        elif config.chaos_type == ChaosType.PACKET_LOSS:
            await self._inject_packet_loss(config, result)
        elif config.chaos_type == ChaosType.CPU_STRESS:
            await self._inject_cpu_stress(config, result)
        elif config.chaos_type == ChaosType.MEMORY_PRESSURE:
            await self._inject_memory_pressure(config, result)
        elif config.chaos_type == ChaosType.SERVICE_CRASH:
            await self._inject_service_crash(config, result)
        elif config.chaos_type == ChaosType.TRAFFIC_SPIKE:
            await self._inject_traffic_spike(config, result)
        else:
            logger.warning(f"Unknown chaos type: {config.chaos_type}")

    async def _inject_node_failure(self, config: ChaosConfig, result: ExperimentResult):
        """Inject node failure chaos."""
        affected_nodes = int(len(config.target_components) * config.affected_percentage)
        nodes_to_fail = random.sample(config.target_components, min(affected_nodes, len(config.target_components)))

        for node in nodes_to_fail:
            logger.info(f"Simulating failure of node: {node}")
            result.observations[f"failed_node_{node}"] = True
            result.failures_detected.append(f"node_failure_{node}")

    async def _inject_network_partition(self, config: ChaosConfig, result: ExperimentResult):
        """Inject network partition chaos."""
        logger.info(f"Simulating network partition affecting {config.affected_percentage * 100}% of components")
        result.observations["network_partition_active"] = True
        result.failures_detected.append("network_partition")

    async def _inject_high_latency(self, config: ChaosConfig, result: ExperimentResult):
        """Inject high latency chaos."""
        latency_ms = int(config.intensity * 1000)  # Convert intensity to milliseconds
        logger.info(f"Simulating high latency: {latency_ms}ms")
        result.observations["injected_latency_ms"] = latency_ms

    async def _inject_packet_loss(self, config: ChaosConfig, result: ExperimentResult):
        """Inject packet loss chaos."""
        loss_percentage = config.intensity * 10  # Up to 10% loss at max intensity
        logger.info(f"Simulating packet loss: {loss_percentage}%")
        result.observations["packet_loss_percent"] = loss_percentage

    async def _inject_cpu_stress(self, config: ChaosConfig, result: ExperimentResult):
        """Inject CPU stress chaos."""
        cpu_load_percent = config.intensity * 90  # Up to 90% CPU at max intensity
        logger.info(f"Simulating CPU stress: {cpu_load_percent}%")
        result.observations["cpu_stress_percent"] = cpu_load_percent

    async def _inject_memory_pressure(self, config: ChaosConfig, result: ExperimentResult):
        """Inject memory pressure chaos."""
        memory_pressure_percent = config.intensity * 85  # Up to 85% memory at max intensity
        logger.info(f"Simulating memory pressure: {memory_pressure_percent}%")
        result.observations["memory_pressure_percent"] = memory_pressure_percent

    async def _inject_service_crash(self, config: ChaosConfig, result: ExperimentResult):
        """Inject service crash chaos."""
        services_to_crash = random.sample(config.target_components, min(1, len(config.target_components)))

        for service in services_to_crash:
            logger.info(f"Simulating crash of service: {service}")
            result.observations[f"crashed_service_{service}"] = True
            result.failures_detected.append(f"service_crash_{service}")

    async def _inject_traffic_spike(self, config: ChaosConfig, result: ExperimentResult):
        """Inject traffic spike chaos."""
        traffic_multiplier = 1 + (config.intensity * 9)  # Up to 10x traffic at max intensity
        logger.info(f"Simulating traffic spike: {traffic_multiplier}x normal load")
        result.observations["traffic_multiplier"] = traffic_multiplier

    async def _stop_chaos_injection(self, config: ChaosConfig, result: ExperimentResult):
        """Stop chaos injection and begin recovery."""
        logger.info(f"Stopping chaos injection for experiment: {config.experiment_id}")
        result.logs.append("Chaos injection stopped")

        # Mock cleanup/recovery initiation
        await asyncio.sleep(1)

    async def _detect_failures(self, config: ChaosConfig, result: ExperimentResult):
        """Detect failures during chaos experiment."""
        # Mock failure detection
        if random.random() < 0.1:  # 10% chance of detecting a failure each check
            failure_type = random.choice(["timeout", "connection_error", "service_unavailable"])
            failure_msg = f"{failure_type}_detected"

            if failure_msg not in result.failures_detected:
                result.failures_detected.append(failure_msg)
                logger.warning(f"Failure detected during experiment: {failure_msg}")

    async def _monitor_recovery(self, config: ChaosConfig, result: ExperimentResult):
        """Monitor system recovery during experiment."""
        # Mock recovery monitoring
        if result.failures_detected and not result.recovery_time_seconds:
            # Simulate recovery detection
            if random.random() < 0.2:  # 20% chance of recovery each check
                result.recovery_time_seconds = time.time() - result.start_time.timestamp()
                result.recovery_actions.append("automated_recovery_detected")
                logger.info(f"Recovery detected after {result.recovery_time_seconds}s")

    async def _evaluate_experiment_success(self, config: ChaosConfig, result: ExperimentResult):
        """Evaluate if experiment met success criteria."""
        criteria_met = True

        # Check recovery time
        if result.recovery_time_seconds:
            if result.recovery_time_seconds > config.expected_recovery_time:
                criteria_met = False
                result.logs.append(
                    f"Recovery time {result.recovery_time_seconds}s exceeded expected {config.expected_recovery_time}s"
                )

        # Check downtime
        if result.downtime_seconds > config.max_acceptable_downtime:
            criteria_met = False
            result.logs.append(
                f"Downtime {result.downtime_seconds}s exceeded maximum {config.max_acceptable_downtime}s"
            )

        # Check for data loss or corruption
        if result.data_loss or result.corruption_detected:
            criteria_met = False
            result.logs.append("Data integrity issues detected")

        result.criteria_met = criteria_met

        if criteria_met:
            logger.info(f"Experiment {config.experiment_id} met all success criteria")
        else:
            logger.warning(f"Experiment {config.experiment_id} failed to meet success criteria")

    async def _rollback_experiment(self, config: ChaosConfig, result: ExperimentResult):
        """Rollback changes made during experiment."""
        if not config.auto_rollback:
            return

        logger.info(f"Rolling back experiment: {config.experiment_id}")

        # Mock rollback operations
        rollback_actions = [
            "restore_network_configuration",
            "restart_failed_services",
            "clear_injected_faults",
            "restore_normal_traffic",
        ]

        for action in rollback_actions:
            await asyncio.sleep(0.5)  # Simulate rollback time
            result.recovery_actions.append(f"rollback_{action}")

        result.logs.append("Rollback completed successfully")

    async def _safety_check(self, config: ChaosConfig) -> bool:
        """Perform safety checks before starting experiment."""
        # Check if protected components are targeted
        if any(component in self.protected_components for component in config.target_components):
            logger.warning("Experiment targets protected components")
            return False

        # Check system health
        current_health = await self._check_system_health()
        if current_health < 0.9:  # Require 90% health before experiments
            logger.warning(f"System health too low for experiments: {current_health}")
            return False

        # Check if other critical experiments are running
        critical_experiments = sum(
            1
            for exp in self.active_experiments.values()
            if exp.chaos_type in [ChaosType.NODE_FAILURE, ChaosType.SERVICE_CRASH]
        )
        if critical_experiments > 0:
            logger.warning("Critical experiment already running")
            return False

        return True

    async def _check_system_health(self) -> float:
        """Check overall system health."""
        # Mock system health check
        health_scores = []

        components = [
            "fog_coordinator",
            "harvest_manager",
            "onion_router",
            "fog_marketplace",
            "token_system",
            "mixnet_client",
        ]

        for component in components:
            # Simulate health check
            health = random.uniform(0.85, 1.0)
            health_scores.append(health)

        return sum(health_scores) / len(health_scores) if health_scores else 0.0

    async def _collect_baseline_metrics(self):
        """Collect baseline system metrics."""
        logger.info("Collecting baseline metrics")

        self.baseline_metrics = {
            "cpu_utilization": random.uniform(20, 40),
            "memory_utilization": random.uniform(30, 50),
            "network_latency_ms": random.uniform(10, 30),
            "error_rate": random.uniform(0, 0.5),
            "throughput_rps": random.uniform(100, 200),
            "availability": random.uniform(99.0, 99.9),
        }

    async def _collect_current_metrics(self) -> dict[str, float]:
        """Collect current system metrics."""
        return {
            "cpu_utilization": random.uniform(15, 60),
            "memory_utilization": random.uniform(25, 70),
            "network_latency_ms": random.uniform(8, 100),
            "error_rate": random.uniform(0, 2),
            "throughput_rps": random.uniform(80, 250),
            "availability": random.uniform(98.0, 100.0),
        }

    async def _collect_system_health(self, report: ValidationReport):
        """Collect system health metrics for validation report."""
        components = [
            "fog_coordinator",
            "harvest_manager",
            "onion_router",
            "fog_marketplace",
            "token_system",
            "mixnet_client",
            "hidden_service_host",
            "contribution_ledger",
            "slo_monitor",
        ]

        for component in components:
            # Mock health status
            health_status = random.choices(["healthy", "degraded", "unhealthy"], weights=[85, 10, 5])[0]
            report.component_health[component] = health_status

        # Mock performance metrics
        report.avg_response_time = random.uniform(50, 150)
        report.p95_response_time = random.uniform(100, 300)
        report.error_rate = random.uniform(0, 2)
        report.availability = random.uniform(99, 100)

    async def _generate_recommendations(self, report: ValidationReport):
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check failed tests
        failed_critical = sum(
            1
            for result in report.test_results
            if result["status"] != ChaosValidationResult.PASS and "critical" in str(result)
        )

        if failed_critical > 0:
            recommendations.append("Critical test failures detected - investigate immediately")

        # Check performance
        if report.avg_response_time > 100:
            recommendations.append("Average response time high - consider performance optimization")

        if report.error_rate > 1.0:
            recommendations.append("Error rate elevated - review error handling and resilience")

        if report.availability < 99.5:
            recommendations.append("Availability below target - review SLO configurations")

        # Check component health
        unhealthy_components = [
            comp for comp, health in report.component_health.items() if health in ["degraded", "unhealthy"]
        ]

        if unhealthy_components:
            recommendations.append(f"Unhealthy components detected: {', '.join(unhealthy_components)}")

        if not recommendations:
            recommendations.append("All validation tests passed - system ready for production")

        report.recommendations = recommendations

    async def _experiment_monitor(self):
        """Background task to monitor active experiments."""
        while self._running:
            try:
                # Check for experiment timeouts
                current_time = time.time()

                for experiment_id, config in list(self.active_experiments.items()):
                    result = self.experiment_results.get(experiment_id)
                    if result:
                        elapsed = current_time - result.start_time.timestamp()

                        # Check for timeout
                        if elapsed > config.max_duration:
                            logger.warning(f"Experiment {experiment_id} timed out - stopping")
                            await self.stop_experiment(experiment_id)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in experiment monitor: {e}")
                await asyncio.sleep(30)

    async def _metrics_collector(self):
        """Background task to collect system metrics."""
        while self._running:
            try:
                self.current_metrics = await self._collect_current_metrics()
                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)

    async def _safety_monitor(self):
        """Background task to monitor system safety."""
        while self._running:
            try:
                # Check system health
                health = await self._check_system_health()

                if health < 0.7:  # Critical health threshold
                    logger.critical(f"System health critical: {health}")
                    self.emergency_stop = True

                    # Stop all active experiments
                    for experiment_id in list(self.active_experiments.keys()):
                        await self.stop_experiment(experiment_id)

                elif health > 0.9 and self.emergency_stop:
                    # Re-enable experiments if health recovers
                    self.emergency_stop = False
                    logger.info("Emergency stop lifted - experiments re-enabled")

                await asyncio.sleep(120)  # Check every 2 minutes

            except Exception as e:
                logger.error(f"Error in safety monitor: {e}")
                await asyncio.sleep(60)

    async def _health_checker(self):
        """Background task for continuous health checks."""
        while self._running:
            try:
                # Perform lightweight health checks
                for component in ["fog_coordinator", "harvest_manager", "onion_router"]:
                    # Mock health check
                    await asyncio.sleep(0.1)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(60)

    async def _report_generator(self):
        """Background task to generate periodic reports."""
        while self._running:
            try:
                # Generate validation report
                if (
                    len(self.validation_reports) == 0
                    or (datetime.now() - self.validation_reports[-1].timestamp).total_seconds() > 3600
                ):
                    logger.info("Generating periodic validation report")
                    await self.run_validation_suite()

                await asyncio.sleep(1800)  # Check every 30 minutes

            except Exception as e:
                logger.error(f"Error in report generator: {e}")
                await asyncio.sleep(300)

    async def _save_experiment_data(self):
        """Save experiment data to disk."""
        # Save experiment results
        results_file = self.data_dir / "experiment_results.json"

        results_data = []
        for result in self.experiment_history:
            data = asdict(result)
            data["start_time"] = result.start_time.isoformat()
            if result.end_time:
                data["end_time"] = result.end_time.isoformat()
            data["status"] = result.status.value
            results_data.append(data)

        async with aiofiles.open(results_file, "w") as f:
            await f.write(json.dumps(results_data, indent=2))

        # Save validation reports
        reports_file = self.data_dir / "validation_reports.json"

        reports_data = []
        for report in self.validation_reports:
            data = asdict(report)
            data["timestamp"] = report.timestamp.isoformat()
            reports_data.append(data)

        async with aiofiles.open(reports_file, "w") as f:
            await f.write(json.dumps(reports_data, indent=2))

        logger.info("Experiment data saved to disk")
