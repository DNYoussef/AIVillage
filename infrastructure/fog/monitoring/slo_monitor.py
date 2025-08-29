"""
SLO Monitoring and Recovery Loops

Provides comprehensive Service Level Objective monitoring with automated recovery
loops for the fog computing infrastructure. Ensures system reliability and performance.
"""
import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Any

import aiofiles

logger = logging.getLogger(__name__)


class SLOType(Enum):
    """Types of Service Level Objectives."""

    AVAILABILITY = "availability"  # System uptime percentage
    LATENCY = "latency"  # Response time percentiles
    THROUGHPUT = "throughput"  # Requests/operations per second
    ERROR_RATE = "error_rate"  # Error percentage
    RESOURCE_UTILIZATION = "resource_util"  # CPU/Memory/Storage usage
    DATA_FRESHNESS = "data_freshness"  # Data staleness
    RECOVERY_TIME = "recovery_time"  # Mean time to recovery
    SECURITY_COMPLIANCE = "security"  # Security metrics


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"  # Service down or major degradation
    HIGH = "high"  # SLO breach imminent
    MEDIUM = "medium"  # Performance degradation
    LOW = "low"  # Minor issues
    INFO = "info"  # Informational alerts


class RecoveryAction(Enum):
    """Types of automated recovery actions."""

    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE_WORKLOAD = "migrate_workload"
    FAILOVER = "failover"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMIT = "rate_limit"
    CLEAR_CACHE = "clear_cache"
    REFRESH_CONFIG = "refresh_config"
    NOTIFY_ADMIN = "notify_admin"


@dataclass
class SLOTarget:
    """Definition of an SLO target."""

    name: str
    description: str
    slo_type: SLOType
    target_value: float
    comparison: str  # "<=", ">=", "==", "!=", "<", ">"
    measurement_window: int  # seconds
    evaluation_frequency: int  # seconds
    breach_threshold: int = 3  # consecutive breaches before alert
    enabled: bool = True

    # Alert configuration
    alert_severity: AlertSeverity = AlertSeverity.MEDIUM
    escalation_time: int = 300  # seconds before escalating

    # Recovery configuration
    auto_recovery_enabled: bool = True
    recovery_actions: list[RecoveryAction] = field(default_factory=list)
    recovery_cooldown: int = 300  # seconds between recovery attempts


@dataclass
class MetricPoint:
    """A single metric measurement point."""

    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class SLOBreach:
    """Record of an SLO breach."""

    breach_id: str
    slo_name: str
    timestamp: datetime
    measured_value: float
    target_value: float
    severity: AlertSeverity
    duration_seconds: int = 0
    recovery_actions_taken: list[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: datetime | None = None


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    attempt_id: str
    breach_id: str
    action: RecoveryAction
    timestamp: datetime
    parameters: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: str | None = None
    duration_seconds: float = 0.0


class SLOMonitor:
    """
    Comprehensive SLO Monitoring and Recovery System.

    Monitors service level objectives across the fog infrastructure,
    detects breaches, and triggers automated recovery actions.
    """

    def __init__(self, data_dir: str = "slo_monitoring"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # SLO configuration
        self.slo_targets: dict[str, SLOTarget] = {}

        # Metrics storage (time series data)
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Breach tracking
        self.active_breaches: dict[str, SLOBreach] = {}
        self.breach_history: deque = deque(maxlen=1000)
        self.breach_counters: dict[str, int] = defaultdict(int)

        # Recovery system
        self.recovery_handlers: dict[RecoveryAction, Callable] = {}
        self.recovery_attempts: deque = deque(maxlen=1000)
        self.recovery_cooldowns: dict[str, float] = {}

        # Monitoring state
        self.last_evaluations: dict[str, float] = {}
        self.circuit_breakers: dict[str, bool] = defaultdict(bool)

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        # Initialize default SLO targets
        self._initialize_default_slos()

        logger.info("SLO Monitor initialized")

    def _initialize_default_slos(self):
        """Initialize default SLO targets for fog infrastructure."""

        # Fog compute service availability
        self.slo_targets["fog_compute_availability"] = SLOTarget(
            name="fog_compute_availability",
            description="Fog compute service availability",
            slo_type=SLOType.AVAILABILITY,
            target_value=99.5,  # 99.5% uptime
            comparison=">=",
            measurement_window=300,  # 5 minutes
            evaluation_frequency=60,  # Check every minute
            breach_threshold=2,
            alert_severity=AlertSeverity.CRITICAL,
            recovery_actions=[RecoveryAction.RESTART_SERVICE, RecoveryAction.FAILOVER],
        )

        # Response latency
        self.slo_targets["fog_response_latency_p95"] = SLOTarget(
            name="fog_response_latency_p95",
            description="95th percentile response latency",
            slo_type=SLOType.LATENCY,
            target_value=500.0,  # 500ms
            comparison="<=",
            measurement_window=300,
            evaluation_frequency=60,
            alert_severity=AlertSeverity.HIGH,
            recovery_actions=[RecoveryAction.SCALE_UP, RecoveryAction.MIGRATE_WORKLOAD],
        )

        # Error rate
        self.slo_targets["fog_error_rate"] = SLOTarget(
            name="fog_error_rate",
            description="Service error rate",
            slo_type=SLOType.ERROR_RATE,
            target_value=1.0,  # 1% error rate
            comparison="<=",
            measurement_window=300,
            evaluation_frequency=60,
            alert_severity=AlertSeverity.HIGH,
            recovery_actions=[RecoveryAction.CIRCUIT_BREAKER, RecoveryAction.RESTART_SERVICE],
        )

        # Throughput
        self.slo_targets["fog_min_throughput"] = SLOTarget(
            name="fog_min_throughput",
            description="Minimum service throughput",
            slo_type=SLOType.THROUGHPUT,
            target_value=100.0,  # 100 requests/second
            comparison=">=",
            measurement_window=300,
            evaluation_frequency=120,
            alert_severity=AlertSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.SCALE_UP],
        )

        # Resource utilization
        self.slo_targets["cpu_utilization_max"] = SLOTarget(
            name="cpu_utilization_max",
            description="Maximum CPU utilization",
            slo_type=SLOType.RESOURCE_UTILIZATION,
            target_value=85.0,  # 85% max CPU usage
            comparison="<=",
            measurement_window=180,
            evaluation_frequency=30,
            alert_severity=AlertSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.SCALE_UP, RecoveryAction.MIGRATE_WORKLOAD],
        )

        # Memory utilization
        self.slo_targets["memory_utilization_max"] = SLOTarget(
            name="memory_utilization_max",
            description="Maximum memory utilization",
            slo_type=SLOType.RESOURCE_UTILIZATION,
            target_value=90.0,  # 90% max memory usage
            comparison="<=",
            measurement_window=180,
            evaluation_frequency=30,
            alert_severity=AlertSeverity.HIGH,
            recovery_actions=[RecoveryAction.CLEAR_CACHE, RecoveryAction.SCALE_UP],
        )

        # Hidden service availability
        self.slo_targets["hidden_service_availability"] = SLOTarget(
            name="hidden_service_availability",
            description="Hidden service availability",
            slo_type=SLOType.AVAILABILITY,
            target_value=99.0,  # 99% uptime
            comparison=">=",
            measurement_window=600,  # 10 minutes
            evaluation_frequency=120,
            alert_severity=AlertSeverity.HIGH,
            recovery_actions=[RecoveryAction.FAILOVER, RecoveryAction.MIGRATE_WORKLOAD],
        )

        # Mixnet node availability
        self.slo_targets["mixnet_node_availability"] = SLOTarget(
            name="mixnet_node_availability",
            description="Mixnet node availability",
            slo_type=SLOType.AVAILABILITY,
            target_value=95.0,  # 95% uptime
            comparison=">=",
            measurement_window=300,
            evaluation_frequency=60,
            alert_severity=AlertSeverity.MEDIUM,
            recovery_actions=[RecoveryAction.RESTART_SERVICE],
        )

    async def start(self):
        """Start the SLO monitoring system."""
        if self._running:
            return

        logger.info("Starting SLO Monitor")
        self._running = True

        # Load configuration and history
        await self._load_configuration()

        # Register default recovery handlers
        self._register_recovery_handlers()

        # Start background tasks
        tasks = [
            self._slo_evaluator(),
            self._breach_detector(),
            self._recovery_executor(),
            self._metrics_collector(),
            self._alert_manager(),
            self._health_reporter(),
            self._data_persister(),
        ]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("SLO Monitor started successfully")

    async def stop(self):
        """Stop the SLO monitoring system."""
        if not self._running:
            return

        logger.info("Stopping SLO Monitor")
        self._running = False

        # Save configuration and data
        await self._save_configuration()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("SLO Monitor stopped")

    async def record_metric(
        self, metric_name: str, value: float, timestamp: float | None = None, labels: dict[str, str] = None
    ):
        """Record a metric measurement."""
        if timestamp is None:
            timestamp = time.time()

        point = MetricPoint(timestamp=timestamp, value=value, labels=labels or {})

        self.metrics[metric_name].append(point)

        logger.debug(f"Recorded metric {metric_name}: {value} at {timestamp}")

    async def add_slo_target(self, target: SLOTarget):
        """Add a new SLO target."""
        self.slo_targets[target.name] = target
        logger.info(f"Added SLO target: {target.name}")

    async def remove_slo_target(self, slo_name: str):
        """Remove an SLO target."""
        if slo_name in self.slo_targets:
            del self.slo_targets[slo_name]
            logger.info(f"Removed SLO target: {slo_name}")

    def register_recovery_handler(self, action: RecoveryAction, handler: Callable):
        """Register a recovery action handler."""
        self.recovery_handlers[action] = handler
        logger.info(f"Registered recovery handler for {action.value}")

    async def get_slo_status(self) -> dict[str, Any]:
        """Get current SLO status."""
        status = {}

        for slo_name, target in self.slo_targets.items():
            if not target.enabled:
                continue

            # Get recent metrics
            current_value = await self._evaluate_slo(target)
            is_breached = await self._is_slo_breached(target, current_value)

            status[slo_name] = {
                "target_value": target.target_value,
                "current_value": current_value,
                "comparison": target.comparison,
                "is_breached": is_breached,
                "breach_count": self.breach_counters.get(slo_name, 0),
                "last_evaluation": self.last_evaluations.get(slo_name, 0),
                "enabled": target.enabled,
            }

        return status

    async def get_breach_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get breach summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_breaches = [breach for breach in self.breach_history if breach.timestamp > cutoff_time]

        # Group by SLO
        breaches_by_slo = defaultdict(list)
        for breach in recent_breaches:
            breaches_by_slo[breach.slo_name].append(breach)

        # Calculate statistics
        total_breaches = len(recent_breaches)
        resolved_breaches = sum(1 for b in recent_breaches if b.resolved)
        critical_breaches = sum(1 for b in recent_breaches if b.severity == AlertSeverity.CRITICAL)

        avg_resolution_time = 0.0
        if resolved_breaches > 0:
            resolution_times = [
                (b.resolution_time - b.timestamp).total_seconds()
                for b in recent_breaches
                if b.resolved and b.resolution_time
            ]
            if resolution_times:
                avg_resolution_time = statistics.mean(resolution_times)

        return {
            "time_period_hours": hours,
            "total_breaches": total_breaches,
            "resolved_breaches": resolved_breaches,
            "critical_breaches": critical_breaches,
            "avg_resolution_time_seconds": avg_resolution_time,
            "breaches_by_slo": {slo: len(breaches) for slo, breaches in breaches_by_slo.items()},
            "active_breaches": len(self.active_breaches),
        }

    async def get_recovery_stats(self) -> dict[str, Any]:
        """Get recovery action statistics."""
        total_attempts = len(self.recovery_attempts)
        successful_attempts = sum(1 for a in self.recovery_attempts if a.success)

        # Group by action type
        attempts_by_action = defaultdict(list)
        for attempt in self.recovery_attempts:
            attempts_by_action[attempt.action].append(attempt)

        action_stats = {}
        for action, attempts in attempts_by_action.items():
            successful = sum(1 for a in attempts if a.success)
            avg_duration = statistics.mean([a.duration_seconds for a in attempts])

            action_stats[action.value] = {
                "total_attempts": len(attempts),
                "successful_attempts": successful,
                "success_rate": successful / len(attempts) if attempts else 0,
                "avg_duration_seconds": avg_duration,
            }

        return {
            "total_recovery_attempts": total_attempts,
            "successful_recovery_attempts": successful_attempts,
            "overall_success_rate": successful_attempts / total_attempts if total_attempts else 0,
            "by_action": action_stats,
        }

    async def _slo_evaluator(self):
        """Background task to evaluate SLO targets."""
        while self._running:
            try:
                current_time = time.time()

                for slo_name, target in self.slo_targets.items():
                    if not target.enabled:
                        continue

                    # Check if it's time to evaluate
                    last_eval = self.last_evaluations.get(slo_name, 0)
                    if current_time - last_eval >= target.evaluation_frequency:
                        await self._evaluate_and_check_slo(target)
                        self.last_evaluations[slo_name] = current_time

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in SLO evaluator: {e}")
                await asyncio.sleep(60)

    async def _evaluate_and_check_slo(self, target: SLOTarget):
        """Evaluate an SLO target and check for breaches."""
        current_value = await self._evaluate_slo(target)
        is_breached = await self._is_slo_breached(target, current_value)

        if is_breached:
            self.breach_counters[target.name] += 1

            # Check if we should trigger a breach
            if self.breach_counters[target.name] >= target.breach_threshold:
                await self._trigger_slo_breach(target, current_value)
        else:
            # Reset counter on successful evaluation
            if target.name in self.breach_counters:
                self.breach_counters[target.name] = 0

            # Resolve active breach if exists
            if target.name in self.active_breaches:
                await self._resolve_slo_breach(target.name)

    async def _evaluate_slo(self, target: SLOTarget) -> float | None:
        """Evaluate current value for an SLO target."""
        # Get metrics for the measurement window
        cutoff_time = time.time() - target.measurement_window

        # For now, use the target name as metric name
        # In production, this would map to appropriate metrics
        metric_name = target.name.replace("_slo", "").replace("_target", "")

        if metric_name not in self.metrics:
            # Generate mock metric for demonstration
            await self._generate_mock_metric(metric_name, target)

        recent_points = [point for point in self.metrics[metric_name] if point.timestamp >= cutoff_time]

        if not recent_points:
            return None

        values = [point.value for point in recent_points]

        # Calculate value based on SLO type
        if target.slo_type == SLOType.AVAILABILITY:
            # Availability: percentage of successful measurements
            successful = sum(1 for v in values if v > 0)
            return (successful / len(values)) * 100 if values else 0

        elif target.slo_type == SLOType.LATENCY:
            # Latency: use 95th percentile
            if "p95" in target.name:
                return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values) if values else 0
            elif "p99" in target.name:
                return statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values) if values else 0
            else:
                return statistics.mean(values) if values else 0

        elif target.slo_type == SLOType.ERROR_RATE:
            # Error rate: percentage of errors
            errors = sum(1 for v in values if v > 0)
            return (errors / len(values)) * 100 if values else 0

        elif target.slo_type in [SLOType.THROUGHPUT, SLOType.RESOURCE_UTILIZATION]:
            # Use mean for throughput and resource utilization
            return statistics.mean(values) if values else 0

        else:
            return statistics.mean(values) if values else 0

    async def _is_slo_breached(self, target: SLOTarget, current_value: float | None) -> bool:
        """Check if an SLO is currently breached."""
        if current_value is None:
            return True  # No data is considered a breach

        if target.comparison == "<=":
            return current_value > target.target_value
        elif target.comparison == ">=":
            return current_value < target.target_value
        elif target.comparison == "==":
            return current_value != target.target_value
        elif target.comparison == "!=":
            return current_value == target.target_value
        elif target.comparison == "<":
            return current_value >= target.target_value
        elif target.comparison == ">":
            return current_value <= target.target_value

        return False

    async def _trigger_slo_breach(self, target: SLOTarget, measured_value: float):
        """Trigger an SLO breach alert and recovery."""
        breach_id = f"breach_{target.name}_{int(time.time())}"

        breach = SLOBreach(
            breach_id=breach_id,
            slo_name=target.name,
            timestamp=datetime.now(),
            measured_value=measured_value,
            target_value=target.target_value,
            severity=target.alert_severity,
        )

        # Store active breach
        self.active_breaches[target.name] = breach
        self.breach_history.append(breach)

        logger.warning(
            f"SLO breach detected: {target.name} = {measured_value} (target: {target.comparison} {target.target_value})"
        )

        # Trigger recovery actions if enabled
        if target.auto_recovery_enabled:
            await self._execute_recovery_actions(breach, target.recovery_actions)

    async def _resolve_slo_breach(self, slo_name: str):
        """Resolve an active SLO breach."""
        if slo_name in self.active_breaches:
            breach = self.active_breaches[slo_name]
            breach.resolved = True
            breach.resolution_time = datetime.now()
            breach.duration_seconds = (breach.resolution_time - breach.timestamp).total_seconds()

            del self.active_breaches[slo_name]

            logger.info(f"SLO breach resolved: {slo_name} (duration: {breach.duration_seconds}s)")

    async def _execute_recovery_actions(self, breach: SLOBreach, actions: list[RecoveryAction]):
        """Execute recovery actions for an SLO breach."""
        for action in actions:
            # Check cooldown
            cooldown_key = f"{breach.slo_name}_{action.value}"
            if cooldown_key in self.recovery_cooldowns:
                if time.time() - self.recovery_cooldowns[cooldown_key] < 300:  # 5 minute cooldown
                    logger.debug(f"Recovery action {action.value} in cooldown for {breach.slo_name}")
                    continue

            # Execute recovery action
            await self._execute_single_recovery_action(breach, action)

            # Set cooldown
            self.recovery_cooldowns[cooldown_key] = time.time()

    async def _execute_single_recovery_action(self, breach: SLOBreach, action: RecoveryAction):
        """Execute a single recovery action."""
        attempt_id = f"recovery_{breach.breach_id}_{action.value}_{int(time.time())}"

        attempt = RecoveryAttempt(
            attempt_id=attempt_id, breach_id=breach.breach_id, action=action, timestamp=datetime.now()
        )

        start_time = time.time()

        try:
            # Execute the recovery action
            if action in self.recovery_handlers:
                handler = self.recovery_handlers[action]
                await handler(breach, attempt)
                attempt.success = True
            else:
                # Default actions
                await self._default_recovery_action(action, breach, attempt)

            attempt.duration_seconds = time.time() - start_time

            # Record successful action
            breach.recovery_actions_taken.append(action.value)

            logger.info(f"Recovery action {action.value} executed successfully for breach {breach.breach_id}")

        except Exception as e:
            attempt.success = False
            attempt.error_message = str(e)
            attempt.duration_seconds = time.time() - start_time

            logger.error(f"Recovery action {action.value} failed for breach {breach.breach_id}: {e}")

        finally:
            self.recovery_attempts.append(attempt)

    async def _default_recovery_action(self, action: RecoveryAction, breach: SLOBreach, attempt: RecoveryAttempt):
        """Execute default recovery actions."""
        if action == RecoveryAction.RESTART_SERVICE:
            # Mock service restart
            logger.info(f"Restarting service for SLO: {breach.slo_name}")
            await asyncio.sleep(2)  # Simulate restart time

        elif action == RecoveryAction.SCALE_UP:
            # Mock scaling up
            logger.info(f"Scaling up resources for SLO: {breach.slo_name}")
            await asyncio.sleep(1)

        elif action == RecoveryAction.FAILOVER:
            # Mock failover
            logger.info(f"Initiating failover for SLO: {breach.slo_name}")
            await asyncio.sleep(3)

        elif action == RecoveryAction.CIRCUIT_BREAKER:
            # Enable circuit breaker
            self.circuit_breakers[breach.slo_name] = True
            logger.info(f"Circuit breaker enabled for SLO: {breach.slo_name}")

        elif action == RecoveryAction.CLEAR_CACHE:
            # Mock cache clearing
            logger.info(f"Clearing cache for SLO: {breach.slo_name}")
            await asyncio.sleep(0.5)

        elif action == RecoveryAction.NOTIFY_ADMIN:
            # Mock admin notification
            logger.warning(f"ADMIN NOTIFICATION: SLO breach {breach.slo_name} requires attention")

        else:
            logger.warning(f"Unknown recovery action: {action.value}")

    def _register_recovery_handlers(self):
        """Register default recovery handlers."""
        # Handlers would be registered here for specific recovery actions
        # For now, we use default implementations
        pass

    async def _breach_detector(self):
        """Background task to detect and manage breaches."""
        while self._running:
            try:
                # Check for escalations
                current_time = datetime.now()

                for slo_name, breach in list(self.active_breaches.items()):
                    # Check if breach should be escalated
                    target = self.slo_targets.get(slo_name)
                    if target and not breach.resolved:
                        duration = (current_time - breach.timestamp).total_seconds()

                        if duration > target.escalation_time:
                            # Escalate breach
                            await self._escalate_breach(breach, target)

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in breach detector: {e}")
                await asyncio.sleep(30)

    async def _escalate_breach(self, breach: SLOBreach, target: SLOTarget):
        """Escalate an SLO breach."""
        # Escalate severity
        if breach.severity == AlertSeverity.LOW:
            breach.severity = AlertSeverity.MEDIUM
        elif breach.severity == AlertSeverity.MEDIUM:
            breach.severity = AlertSeverity.HIGH
        elif breach.severity == AlertSeverity.HIGH:
            breach.severity = AlertSeverity.CRITICAL

        logger.warning(f"Escalated breach {breach.breach_id} to {breach.severity.value}")

        # Trigger additional recovery actions for escalated breaches
        if breach.severity == AlertSeverity.CRITICAL:
            await self._execute_recovery_actions(breach, [RecoveryAction.NOTIFY_ADMIN])

    async def _recovery_executor(self):
        """Background task to manage ongoing recovery operations."""
        while self._running:
            try:
                # Check circuit breakers
                for slo_name, enabled in list(self.circuit_breakers.items()):
                    if enabled:
                        # Check if we should disable circuit breaker
                        target = self.slo_targets.get(slo_name)
                        if target:
                            current_value = await self._evaluate_slo(target)
                            if current_value is not None and not await self._is_slo_breached(target, current_value):
                                self.circuit_breakers[slo_name] = False
                                logger.info(f"Circuit breaker disabled for {slo_name}")

                await asyncio.sleep(120)  # Check every 2 minutes

            except Exception as e:
                logger.error(f"Error in recovery executor: {e}")
                await asyncio.sleep(60)

    async def _metrics_collector(self):
        """Background task to collect system metrics."""
        while self._running:
            try:
                # Generate mock metrics for demonstration
                await self._collect_system_metrics()

                await asyncio.sleep(30)  # Collect every 30 seconds

            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)

    async def _collect_system_metrics(self):
        """Collect system metrics."""
        import random

        current_time = time.time()

        # Mock availability metrics
        await self.record_metric("fog_compute_availability", random.choice([1, 1, 1, 0]) * 100, current_time)
        await self.record_metric("hidden_service_availability", random.choice([1, 1, 1, 1, 0]) * 100, current_time)
        await self.record_metric("mixnet_node_availability", random.choice([1, 1, 1, 0]) * 100, current_time)

        # Mock latency metrics
        await self.record_metric("fog_response_latency_p95", random.gauss(400, 100), current_time)

        # Mock error rate
        await self.record_metric("fog_error_rate", random.choice([0, 0, 0, 0, 1]) * 100, current_time)

        # Mock throughput
        await self.record_metric("fog_min_throughput", random.gauss(120, 20), current_time)

        # Mock resource utilization
        await self.record_metric("cpu_utilization_max", random.gauss(70, 15), current_time)
        await self.record_metric("memory_utilization_max", random.gauss(75, 10), current_time)

    async def _generate_mock_metric(self, metric_name: str, target: SLOTarget):
        """Generate mock metric data for testing."""
        import random

        current_time = time.time()

        # Generate values around the target threshold
        if target.slo_type == SLOType.AVAILABILITY:
            # Generate availability data (0 or 100)
            value = random.choice([100, 100, 100, 100, 0])  # 80% availability
        elif target.slo_type == SLOType.LATENCY:
            # Generate latency around target
            value = random.gauss(target.target_value * 0.8, target.target_value * 0.2)
        elif target.slo_type == SLOType.ERROR_RATE:
            # Generate error rate (0 or 1 for individual requests)
            value = random.choice([0, 0, 0, 0, 1])  # 20% error rate
        else:
            # Default: generate around target value
            value = random.gauss(target.target_value, target.target_value * 0.1)

        await self.record_metric(metric_name, value, current_time)

    async def _alert_manager(self):
        """Background task to manage alerts and notifications."""
        while self._running:
            try:
                # Process active breaches for alerts
                for breach in self.active_breaches.values():
                    if not breach.resolved:
                        # Send alert (mock implementation)
                        await self._send_alert(breach)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in alert manager: {e}")
                await asyncio.sleep(60)

    async def _send_alert(self, breach: SLOBreach):
        """Send alert for SLO breach."""
        # Mock alert sending
        logger.warning(
            f"ALERT [{breach.severity.value.upper()}]: SLO {breach.slo_name} breached - "
            f"Value: {breach.measured_value}, Target: {breach.target_value}"
        )

    async def _health_reporter(self):
        """Background task to report system health."""
        while self._running:
            try:
                # Generate health report
                total_slos = len(self.slo_targets)
                active_breaches = len(self.active_breaches)

                health_percentage = ((total_slos - active_breaches) / total_slos * 100) if total_slos else 100

                logger.info(
                    f"System Health: {health_percentage:.1f}% ({total_slos - active_breaches}/{total_slos} SLOs healthy)"
                )

                await asyncio.sleep(900)  # Report every 15 minutes

            except Exception as e:
                logger.error(f"Error in health reporter: {e}")
                await asyncio.sleep(300)

    async def _data_persister(self):
        """Background task to persist monitoring data."""
        while self._running:
            try:
                await self._save_monitoring_data()
                await asyncio.sleep(1800)  # Save every 30 minutes

            except Exception as e:
                logger.error(f"Error in data persister: {e}")
                await asyncio.sleep(300)

    async def _save_configuration(self):
        """Save SLO configuration to disk."""
        config_file = self.data_dir / "slo_config.json"

        config_data = {}
        for slo_name, target in self.slo_targets.items():
            config_data[slo_name] = {
                "name": target.name,
                "description": target.description,
                "slo_type": target.slo_type.value,
                "target_value": target.target_value,
                "comparison": target.comparison,
                "measurement_window": target.measurement_window,
                "evaluation_frequency": target.evaluation_frequency,
                "breach_threshold": target.breach_threshold,
                "enabled": target.enabled,
                "alert_severity": target.alert_severity.value,
                "escalation_time": target.escalation_time,
                "auto_recovery_enabled": target.auto_recovery_enabled,
                "recovery_actions": [action.value for action in target.recovery_actions],
                "recovery_cooldown": target.recovery_cooldown,
            }

        async with aiofiles.open(config_file, "w") as f:
            await f.write(json.dumps(config_data, indent=2))

    async def _load_configuration(self):
        """Load SLO configuration from disk."""
        config_file = self.data_dir / "slo_config.json"

        if not config_file.exists():
            return

        try:
            async with aiofiles.open(config_file, "r") as f:
                config_data = json.loads(await f.read())

            for slo_name, data in config_data.items():
                target = SLOTarget(
                    name=data["name"],
                    description=data["description"],
                    slo_type=SLOType(data["slo_type"]),
                    target_value=data["target_value"],
                    comparison=data["comparison"],
                    measurement_window=data["measurement_window"],
                    evaluation_frequency=data["evaluation_frequency"],
                    breach_threshold=data["breach_threshold"],
                    enabled=data["enabled"],
                    alert_severity=AlertSeverity(data["alert_severity"]),
                    escalation_time=data["escalation_time"],
                    auto_recovery_enabled=data["auto_recovery_enabled"],
                    recovery_actions=[RecoveryAction(action) for action in data["recovery_actions"]],
                    recovery_cooldown=data["recovery_cooldown"],
                )
                self.slo_targets[slo_name] = target

            logger.info(f"Loaded {len(config_data)} SLO targets from configuration")

        except Exception as e:
            logger.error(f"Error loading SLO configuration: {e}")

    async def _save_monitoring_data(self):
        """Save monitoring data to disk."""
        # Save breach history
        breaches_file = self.data_dir / "breach_history.json"

        breaches_data = []
        for breach in self.breach_history:
            data = asdict(breach)
            data["timestamp"] = breach.timestamp.isoformat()
            if breach.resolution_time:
                data["resolution_time"] = breach.resolution_time.isoformat()
            data["severity"] = breach.severity.value
            breaches_data.append(data)

        async with aiofiles.open(breaches_file, "w") as f:
            await f.write(json.dumps(breaches_data, indent=2))

        # Save recovery attempts
        recovery_file = self.data_dir / "recovery_attempts.json"

        recovery_data = []
        for attempt in self.recovery_attempts:
            data = asdict(attempt)
            data["timestamp"] = attempt.timestamp.isoformat()
            data["action"] = attempt.action.value
            recovery_data.append(data)

        async with aiofiles.open(recovery_file, "w") as f:
            await f.write(json.dumps(recovery_data, indent=2))
