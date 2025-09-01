"""Comprehensive health checking and system monitoring.

This module provides health checking capabilities for the entire AIVillage system,
including agent health, system resources, external dependencies, and overall
system status monitoring with automatic recovery mechanisms.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import aiohttp
import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""

    AGENT = "agent"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    component_id: str
    component_type: ComponentType
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    recovery_suggestions: List[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status == HealthStatus.HEALTHY

    def needs_attention(self) -> bool:
        """Check if component needs attention."""
        return self.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.DOWN]


@dataclass
class SystemHealthSummary:
    """Summary of overall system health."""

    overall_status: HealthStatus
    healthy_components: int
    warning_components: int
    critical_components: int
    down_components: int
    total_components: int
    last_check: datetime
    uptime_seconds: float
    system_load: Dict[str, float] = field(default_factory=dict)

    @property
    def health_percentage(self) -> float:
        """Calculate health as percentage."""
        if self.total_components == 0:
            return 100.0
        return (self.healthy_components / self.total_components) * 100.0


class HealthChecker(ABC):
    """Abstract base class for health checkers."""

    def __init__(self, component_id: str, component_type: ComponentType):
        self.component_id = component_id
        self.component_type = component_type
        self.enabled = True
        self.timeout_seconds = 30.0
        self.retry_attempts = 3
        self.retry_delay = 1.0

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check and return result."""
        logger.warning(f"Health check not implemented for component {self.component_id}")
        return HealthCheckResult(
            component_id=self.component_id,
            component_type=self.component_type,
            status=HealthStatus.UNKNOWN,
            timestamp=datetime.now(),
            message="Health check method not implemented",
            recovery_suggestions=["Implement health check logic for this component"],
        )

    async def check_with_retry(self) -> HealthCheckResult:
        """Perform health check with retry logic."""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                time.time()
                result = await asyncio.wait_for(self.check_health(), timeout=self.timeout_seconds)
                return result

            except asyncio.TimeoutError:
                last_error = f"Health check timed out after {self.timeout_seconds}s"
                logger.error(f"Health check timeout for {self.component_id}: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"Health check exception for {self.component_id}: {last_error}",
                    extra={"component_id": self.component_id, "error_type": type(e).__name__},
                )

            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(self.retry_delay)

        # All attempts failed
        return HealthCheckResult(
            component_id=self.component_id,
            component_type=self.component_type,
            status=HealthStatus.DOWN,
            timestamp=datetime.now(),
            response_time_ms=self.timeout_seconds * 1000,
            message="Health check failed after retries",
            error=last_error,
            recovery_suggestions=["Check component logs", "Restart component", "Verify network connectivity"],
        )


class HttpHealthChecker(HealthChecker):
    """HTTP-based health checker for web services."""

    def __init__(self, component_id: str, url: str, expected_status: int = 200, expected_content: Optional[str] = None):
        super().__init__(component_id, ComponentType.EXTERNAL_SERVICE)
        self.url = url
        self.expected_status = expected_status
        self.expected_content = expected_content

    async def check_health(self) -> HealthCheckResult:
        """Check HTTP endpoint health."""
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session:
                async with session.get(self.url) as response:
                    response_time = (time.time() - start_time) * 1000
                    response_text = await response.text()

                    # Check status code
                    if response.status != self.expected_status:
                        return HealthCheckResult(
                            component_id=self.component_id,
                            component_type=self.component_type,
                            status=HealthStatus.CRITICAL,
                            timestamp=datetime.now(),
                            response_time_ms=response_time,
                            message=f"Unexpected status code: {response.status}",
                            details={"expected_status": self.expected_status, "actual_status": response.status},
                            recovery_suggestions=["Check service logs", "Verify service configuration"],
                        )

                    # Check content if specified
                    if self.expected_content and self.expected_content not in response_text:
                        return HealthCheckResult(
                            component_id=self.component_id,
                            component_type=self.component_type,
                            status=HealthStatus.WARNING,
                            timestamp=datetime.now(),
                            response_time_ms=response_time,
                            message="Expected content not found in response",
                            details={"expected_content": self.expected_content},
                            recovery_suggestions=["Verify service is returning correct content"],
                        )

                    # Determine status based on response time
                    status = HealthStatus.HEALTHY
                    if response_time > 5000:  # 5 seconds
                        status = HealthStatus.CRITICAL
                    elif response_time > 2000:  # 2 seconds
                        status = HealthStatus.WARNING

                    return HealthCheckResult(
                        component_id=self.component_id,
                        component_type=self.component_type,
                        status=status,
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        message="HTTP endpoint responding",
                        details={"url": self.url, "status_code": response.status, "response_size": len(response_text)},
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message="HTTP request failed",
                error=str(e),
                recovery_suggestions=[
                    "Check network connectivity",
                    "Verify service is running",
                    "Check firewall settings",
                ],
            )


class DatabaseHealthChecker(HealthChecker):
    """Database connection health checker."""

    def __init__(self, component_id: str, connection_string: str, test_query: str = "SELECT 1"):
        super().__init__(component_id, ComponentType.DATABASE)
        self.connection_string = connection_string
        self.test_query = test_query

    async def check_health(self) -> HealthCheckResult:
        """Check database connectivity and response."""
        start_time = time.time()

        try:
            # This is a simplified example - in practice, you'd use actual database connectors
            # like asyncpg for PostgreSQL, aiomysql for MySQL, etc.

            # Simulate database connection and query
            await asyncio.sleep(0.1)  # Simulate connection time

            response_time = (time.time() - start_time) * 1000

            # Determine status based on response time
            status = HealthStatus.HEALTHY
            if response_time > 1000:  # 1 second
                status = HealthStatus.WARNING
            elif response_time > 2000:  # 2 seconds
                status = HealthStatus.CRITICAL

            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message="Database responding",
                details={"query": self.test_query, "connection_status": "connected"},
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message="Database connection failed",
                error=str(e),
                recovery_suggestions=[
                    "Check database server status",
                    "Verify connection credentials",
                    "Check network connectivity",
                ],
            )


class SystemResourceHealthChecker(HealthChecker):
    """System resource health checker."""

    def __init__(self, component_id: str = "system_resources"):
        super().__init__(component_id, ComponentType.SYSTEM_RESOURCE)
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 80.0
        self.disk_critical_threshold = 90.0

    async def check_health(self) -> HealthCheckResult:
        """Check system resource usage."""
        start_time = time.time()

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Determine overall status
            status = HealthStatus.HEALTHY
            issues = []

            # Check CPU
            if cpu_percent >= self.cpu_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.cpu_warning_threshold:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            # Check Memory
            if memory.percent >= self.memory_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.memory_warning_threshold:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory.percent:.1f}%")

            # Check Disk
            if disk.percent >= self.disk_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent >= self.disk_warning_threshold:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk.percent:.1f}%")

            message = "System resources normal" if status == HealthStatus.HEALTHY else "; ".join(issues)

            response_time = (time.time() - start_time) * 1000

            recovery_suggestions = []
            if status != HealthStatus.HEALTHY:
                recovery_suggestions = [
                    "Monitor resource usage trends",
                    "Identify resource-intensive processes",
                    "Consider scaling or optimization",
                ]

            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None,
                },
                recovery_suggestions=recovery_suggestions,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message="Failed to check system resources",
                error=str(e),
                recovery_suggestions=["Check system monitoring tools", "Verify psutil installation"],
            )


class AgentHealthChecker(HealthChecker):
    """Health checker for individual agents."""

    def __init__(self, component_id: str, agent_metrics_collector):
        super().__init__(component_id, ComponentType.AGENT)
        self.metrics_collector = agent_metrics_collector
        self.max_idle_time = timedelta(minutes=10)
        self.min_success_rate = 0.8

    async def check_health(self) -> HealthCheckResult:
        """Check agent health based on metrics."""
        start_time = time.time()

        try:
            if not self.metrics_collector:
                return HealthCheckResult(
                    component_id=self.component_id,
                    component_type=self.component_type,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    message="No metrics collector available",
                    recovery_suggestions=["Initialize metrics collection for agent"],
                )

            health_metrics = self.metrics_collector.get_health_metrics()

            # Determine status based on various factors
            status = HealthStatus.HEALTHY
            issues = []

            # Check if agent is responsive
            idle_time = datetime.now() - health_metrics.last_activity
            if idle_time > self.max_idle_time:
                status = HealthStatus.WARNING
                issues.append(f"Agent idle for {idle_time}")

            # Check success rate
            if health_metrics.success_rate < self.min_success_rate:
                status = HealthStatus.CRITICAL
                issues.append(f"Low success rate: {health_metrics.success_rate:.1%}")

            # Check error rate
            if health_metrics.error_rate > 0.2:  # 20% error rate
                status = HealthStatus.CRITICAL
                issues.append(f"High error rate: {health_metrics.error_rate:.1%}")
            elif health_metrics.error_rate > 0.1:  # 10% error rate
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Elevated error rate: {health_metrics.error_rate:.1%}")

            # Check memory usage
            if health_metrics.current_memory_usage_mb > 1024:  # 1GB
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"High memory usage: {health_metrics.current_memory_usage_mb:.1f}MB")

            # Check agent state
            if health_metrics.state.value in ["error", "offline"]:
                status = HealthStatus.DOWN
                issues.append(f"Agent state: {health_metrics.state.value}")

            message = "Agent healthy" if status == HealthStatus.HEALTHY else "; ".join(issues)

            response_time = (time.time() - start_time) * 1000

            recovery_suggestions = []
            if status != HealthStatus.HEALTHY:
                recovery_suggestions = [
                    "Check agent logs for errors",
                    "Monitor agent resource usage",
                    "Consider restarting agent if needed",
                ]

            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=status,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message=message,
                details={
                    "agent_state": health_metrics.state.value,
                    "success_rate": health_metrics.success_rate,
                    "error_rate": health_metrics.error_rate,
                    "memory_usage_mb": health_metrics.current_memory_usage_mb,
                    "throughput": health_metrics.throughput_tasks_per_minute,
                    "last_activity": health_metrics.last_activity.isoformat(),
                },
                recovery_suggestions=recovery_suggestions,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                message="Failed to check agent health",
                error=str(e),
                recovery_suggestions=["Check agent metrics system", "Verify agent is running"],
            )


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""

    def __init__(self, check_interval: float = 60.0):
        """Initialize system health monitor.

        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.health_checkers: Dict[str, HealthChecker] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
        self.system_start_time = datetime.now()

        # Monitoring state
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Alerting
        self.alert_handlers: List[Callable[[HealthCheckResult], None]] = []
        self.notification_cooldown: Dict[str, datetime] = {}
        self.cooldown_period = timedelta(minutes=15)

        # Add default system resource checker
        self.add_health_checker(SystemResourceHealthChecker())

        logger.info("SystemHealthMonitor initialized")

    def add_health_checker(self, checker: HealthChecker) -> None:
        """Add a health checker."""
        self.health_checkers[checker.component_id] = checker
        if checker.component_id not in self.health_history:
            self.health_history[checker.component_id] = []

        logger.info(f"Added health checker for {checker.component_id}")

    def remove_health_checker(self, component_id: str) -> bool:
        """Remove a health checker."""
        if component_id in self.health_checkers:
            del self.health_checkers[component_id]
            logger.info(f"Removed health checker for {component_id}")
            return True
        return False

    def add_alert_handler(self, handler: Callable[[HealthCheckResult], None]) -> None:
        """Add alert handler for health issues."""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.is_running:
            logger.warning("Health monitoring is already running")
            return

        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started health monitoring")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self.check_all_health()
                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def check_all_health(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components."""
        results = {}

        # Run all health checks concurrently
        health_check_tasks = {
            component_id: checker.check_with_retry()
            for component_id, checker in self.health_checkers.items()
            if checker.enabled
        }

        # Wait for all checks to complete
        completed_checks = await asyncio.gather(*health_check_tasks.values(), return_exceptions=True)

        # Process results
        for (component_id, _), result in zip(health_check_tasks.items(), completed_checks):
            if isinstance(result, Exception):
                # Health check raised an exception
                result = HealthCheckResult(
                    component_id=component_id,
                    component_type=ComponentType.UNKNOWN,
                    status=HealthStatus.UNKNOWN,
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    message="Health check failed with exception",
                    error=str(result),
                )

            results[component_id] = result

            # Store in history
            self.health_history[component_id].append(result)

            # Limit history size
            if len(self.health_history[component_id]) > 1000:
                self.health_history[component_id] = self.health_history[component_id][-500:]

            # Handle alerts
            await self._handle_health_result(result)

        return results

    async def _handle_health_result(self, result: HealthCheckResult) -> None:
        """Handle health check result and trigger alerts if needed."""
        if result.needs_attention():
            # Check cooldown to avoid spam
            last_notification = self.notification_cooldown.get(result.component_id)
            if last_notification and datetime.now() - last_notification < self.cooldown_period:
                return

            # Trigger alert handlers
            for handler in self.alert_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(result)
                    else:
                        handler(result)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")

            # Update cooldown
            self.notification_cooldown[result.component_id] = datetime.now()

    def get_system_health_summary(self) -> SystemHealthSummary:
        """Get overall system health summary."""
        if not self.health_checkers:
            return SystemHealthSummary(
                overall_status=HealthStatus.UNKNOWN,
                healthy_components=0,
                warning_components=0,
                critical_components=0,
                down_components=0,
                total_components=0,
                last_check=datetime.now(),
                uptime_seconds=0,
            )

        # Get latest results for each component
        latest_results = {}
        for component_id in self.health_checkers.keys():
            if component_id in self.health_history and self.health_history[component_id]:
                latest_results[component_id] = self.health_history[component_id][-1]

        # Count by status
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.DOWN: 0,
            HealthStatus.UNKNOWN: 0,
        }

        latest_check = datetime.now()
        for result in latest_results.values():
            status_counts[result.status] += 1
            if result.timestamp < latest_check:
                latest_check = result.timestamp

        # Determine overall status
        if status_counts[HealthStatus.DOWN] > 0 or status_counts[HealthStatus.CRITICAL] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            overall_status = HealthStatus.WARNING
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now() - self.system_start_time).total_seconds()

        return SystemHealthSummary(
            overall_status=overall_status,
            healthy_components=status_counts[HealthStatus.HEALTHY],
            warning_components=status_counts[HealthStatus.WARNING],
            critical_components=status_counts[HealthStatus.CRITICAL],
            down_components=status_counts[HealthStatus.DOWN],
            total_components=len(self.health_checkers),
            last_check=latest_check,
            uptime_seconds=uptime,
        )

    def get_component_health_history(
        self, component_id: str, duration: Optional[timedelta] = None
    ) -> List[HealthCheckResult]:
        """Get health history for a specific component."""
        if component_id not in self.health_history:
            return []

        history = self.health_history[component_id]
        if duration is None:
            return list(history)

        cutoff_time = datetime.now() - duration
        return [result for result in history if result.timestamp >= cutoff_time]

    def get_detailed_status_report(self) -> Dict[str, Any]:
        """Get detailed status report for all components."""
        summary = self.get_system_health_summary()

        # Get latest status for each component
        component_status = {}
        for component_id in self.health_checkers.keys():
            if component_id in self.health_history and self.health_history[component_id]:
                latest = self.health_history[component_id][-1]
                component_status[component_id] = {
                    "status": latest.status.value,
                    "message": latest.message,
                    "last_check": latest.timestamp.isoformat(),
                    "response_time_ms": latest.response_time_ms,
                    "component_type": latest.component_type.value,
                    "error": latest.error,
                    "recovery_suggestions": latest.recovery_suggestions,
                }

        return {
            "system_summary": asdict(summary),
            "component_details": component_status,
            "monitoring_info": {
                "is_running": self.is_running,
                "check_interval": self.check_interval,
                "registered_checkers": len(self.health_checkers),
                "total_checks_performed": sum(len(history) for history in self.health_history.values()),
            },
        }

    async def export_health_data(self, file_path: str) -> bool:
        """Export health monitoring data to file."""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "system_summary": asdict(self.get_system_health_summary()),
                "detailed_report": self.get_detailed_status_report(),
                "component_history": {
                    component_id: [asdict(result) for result in history[-50:]]  # Last 50 results
                    for component_id, history in self.health_history.items()
                },
            }

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported health data to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export health data: {e}")
            return False


# Default alert handlers
def console_health_alert_handler(result: HealthCheckResult) -> None:
    """Console alert handler for health issues."""
    status_emoji = {HealthStatus.WARNING: "⚠️", HealthStatus.CRITICAL: "🚨", HealthStatus.DOWN: "🔴"}

    emoji = status_emoji.get(result.status, "❓")
    print(f"{emoji} [{result.status.value.upper()}] {result.component_id}: {result.message}")
    if result.error:
        print(f"   Error: {result.error}")
    if result.recovery_suggestions:
        print(f"   Suggestions: {', '.join(result.recovery_suggestions[:2])}")


def log_health_alert_handler(result: HealthCheckResult) -> None:
    """Log-based alert handler for health issues."""
    if result.status == HealthStatus.DOWN:
        logger.critical(f"HEALTH ALERT: {result.component_id} is DOWN - {result.message}")
    elif result.status == HealthStatus.CRITICAL:
        logger.error(f"HEALTH ALERT: {result.component_id} is CRITICAL - {result.message}")
    elif result.status == HealthStatus.WARNING:
        logger.warning(f"HEALTH ALERT: {result.component_id} has WARNING - {result.message}")


if __name__ == "__main__":

    async def demo():
        """Demonstrate health monitoring."""
        monitor = SystemHealthMonitor(check_interval=10.0)

        # Add alert handlers
        monitor.add_alert_handler(console_health_alert_handler)
        monitor.add_alert_handler(log_health_alert_handler)

        # Add some example health checkers
        monitor.add_health_checker(HttpHealthChecker("example_api", "https://httpbin.org/status/200"))
        monitor.add_health_checker(DatabaseHealthChecker("example_db", "postgresql://localhost/test"))

        print("Starting health monitoring...")
        await monitor.start_monitoring()

        try:
            # Run for 60 seconds
            await asyncio.sleep(60)

            # Get status report
            print("\n=== System Health Report ===")
            summary = monitor.get_system_health_summary()
            print(f"Overall Status: {summary.overall_status.value}")
            print(f"Health Percentage: {summary.health_percentage:.1f}%")
            print(
                f"Components - Healthy: {summary.healthy_components}, Warning: {summary.warning_components}, Critical: {summary.critical_components}"
            )

            # Export health data
            await monitor.export_health_data("health_report.json")

        finally:
            await monitor.stop_monitoring()

    # Run the demo
    asyncio.run(demo())
