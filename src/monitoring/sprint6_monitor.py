#!/usr/bin/env python3
"""Sprint 6 Infrastructure Monitor.

Real-time monitoring and alerting for Sprint 6 infrastructure components.
Integrates with existing monitoring infrastructure and provides specialized
monitoring for P2P, Resource Management, and Evolution systems.
"""

import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InfrastructureHealth:
    """Health status for Sprint 6 infrastructure components."""

    timestamp: str
    p2p_status: str
    resource_management_status: str
    evolution_system_status: str
    overall_health: str
    active_connections: int
    resource_utilization: float
    evolution_tasks_active: int
    last_validation_time: str | None = None
    last_validation_success: bool = False


@dataclass
class PerformanceMetrics:
    """Performance metrics for Sprint 6 systems."""

    timestamp: str
    avg_p2p_latency_ms: float
    resource_allocation_efficiency: float
    evolution_throughput: float
    system_load: float
    memory_pressure: float
    disk_io_rate: float


@dataclass
class AlertInfo:
    """Alert information structure."""

    alert_id: str
    timestamp: str
    severity: str  # critical, warning, info
    component: str
    message: str
    details: dict[str, Any]
    resolved: bool = False


class Sprint6Monitor:
    """Real-time monitor for Sprint 6 infrastructure."""

    def __init__(self, data_dir: str = "monitoring_data") -> None:
        """Initialize the monitor."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_interval = 30  # seconds

        # Data storage
        self.health_history: list[InfrastructureHealth] = []
        self.performance_history: list[PerformanceMetrics] = []
        self.active_alerts: list[AlertInfo] = []

        # Health thresholds
        self.thresholds = {
            "p2p_latency_warning": 100.0,  # ms
            "p2p_latency_critical": 500.0,  # ms
            "resource_utilization_warning": 80.0,  # %
            "resource_utilization_critical": 95.0,  # %
            "memory_pressure_warning": 75.0,  # %
            "memory_pressure_critical": 90.0,  # %
            "validation_age_warning": 3600,  # seconds (1 hour)
            "validation_age_critical": 7200,  # seconds (2 hours)
        }

        # Load existing data
        self.load_monitoring_data()

    def load_monitoring_data(self) -> None:
        """Load existing monitoring data."""
        try:
            # Load health history
            health_file = self.data_dir / "health_history.json"
            if health_file.exists():
                with open(health_file) as f:
                    data = json.load(f)
                    self.health_history = [
                        InfrastructureHealth(**item) for item in data[-100:]  # Keep last 100 records
                    ]

            # Load performance history
            perf_file = self.data_dir / "performance_history.json"
            if perf_file.exists():
                with open(perf_file) as f:
                    data = json.load(f)
                    self.performance_history = [
                        PerformanceMetrics(**item) for item in data[-100:]  # Keep last 100 records
                    ]

            # Load active alerts
            alerts_file = self.data_dir / "active_alerts.json"
            if alerts_file.exists():
                with open(alerts_file) as f:
                    data = json.load(f)
                    self.active_alerts = [AlertInfo(**item) for item in data if not item.get("resolved", False)]

            logger.info(
                f"Loaded monitoring data: {len(self.health_history)} health records, {len(self.performance_history)} performance records, {len(self.active_alerts)} active alerts"
            )

        except Exception as e:
            logger.exception(f"Failed to load monitoring data: {e}")

    def save_monitoring_data(self) -> None:
        """Save monitoring data to files."""
        try:
            # Save health history
            health_file = self.data_dir / "health_history.json"
            with open(health_file, "w") as f:
                json.dump([asdict(h) for h in self.health_history[-100:]], f, indent=2)

            # Save performance history
            perf_file = self.data_dir / "performance_history.json"
            with open(perf_file, "w") as f:
                json.dump([asdict(p) for p in self.performance_history[-100:]], f, indent=2)

            # Save all alerts (resolved and active)
            alerts_file = self.data_dir / "active_alerts.json"
            with open(alerts_file, "w") as f:
                json.dump([asdict(a) for a in self.active_alerts], f, indent=2)

        except Exception as e:
            logger.exception(f"Failed to save monitoring data: {e}")

    async def check_infrastructure_health(self) -> InfrastructureHealth:
        """Check the health of Sprint 6 infrastructure components."""
        timestamp = datetime.now().isoformat()

        # Check P2P system health
        p2p_status = await self._check_p2p_health()

        # Check resource management health
        resource_status = await self._check_resource_management_health()

        # Check evolution system health
        evolution_status = await self._check_evolution_system_health()

        # Determine overall health
        status_scores = {"healthy": 3, "degraded": 2, "warning": 1, "critical": 0}

        min_score = min(
            status_scores.get(p2p_status, 0),
            status_scores.get(resource_status, 0),
            status_scores.get(evolution_status, 0),
        )

        overall_health = next(status for status, score in status_scores.items() if score == min_score)

        # Get validation status
        validation_time, validation_success = await self._check_last_validation()

        health = InfrastructureHealth(
            timestamp=timestamp,
            p2p_status=p2p_status,
            resource_management_status=resource_status,
            evolution_system_status=evolution_status,
            overall_health=overall_health,
            active_connections=await self._get_active_connections(),
            resource_utilization=await self._get_resource_utilization(),
            evolution_tasks_active=await self._get_active_evolution_tasks(),
            last_validation_time=validation_time,
            last_validation_success=validation_success,
        )

        return health

    async def _check_p2p_health(self) -> str:
        """Check P2P system health."""
        try:
            # Try to import and basic check P2P components
            from src.core.p2p.p2p_node import P2PNode

            # Create a test node (without starting it)
            test_node = P2PNode(node_id="health_check_node")

            # Basic health indicators
            if hasattr(test_node, "node_id") and test_node.node_id:
                return "healthy"
            return "warning"

        except ImportError as e:
            logger.warning(f"P2P import failed: {e}")
            return "critical"
        except Exception as e:
            logger.warning(f"P2P health check failed: {e}")
            return "degraded"

    async def _check_resource_management_health(self) -> str:
        """Check resource management system health."""
        try:
            from src.core.resources.device_profiler import DeviceProfiler
            from src.core.resources.resource_monitor import ResourceMonitor

            # Test device profiler
            profiler = DeviceProfiler()
            if not profiler.profile or not profiler.profile.evolution_capable:
                return "warning"

            # Test resource monitor
            monitor = ResourceMonitor(profiler)
            if not monitor.device_profiler:
                return "degraded"

            return "healthy"

        except ImportError as e:
            logger.warning(f"Resource management import failed: {e}")
            return "critical"
        except Exception as e:
            logger.warning(f"Resource management health check failed: {e}")
            return "degraded"

    async def _check_evolution_system_health(self) -> str:
        """Check evolution system health."""
        try:
            from src.production.agent_forge.evolution.infrastructure_aware_evolution import (
                InfrastructureAwareEvolution,
                InfrastructureConfig,
            )

            # Test basic evolution system
            config = InfrastructureConfig(enable_p2p=False)
            system = InfrastructureAwareEvolution(config)

            if hasattr(system, "config") and system.config:
                return "healthy"
            return "warning"

        except ImportError as e:
            logger.warning(f"Evolution system import failed: {e}")
            return "critical"
        except Exception as e:
            logger.warning(f"Evolution system health check failed: {e}")
            return "degraded"

    async def _check_last_validation(self) -> tuple:
        """Check when Sprint 6 validation was last run."""
        try:
            validation_file = Path("sprint6_test_results.json")
            if validation_file.exists():
                with open(validation_file) as f:
                    data = json.load(f)
                    return data.get("timestamp"), data.get("overall_success", False)
            return None, False
        except Exception:
            return None, False

    async def _get_active_connections(self) -> int:
        """Get number of active P2P connections."""
        # Mock implementation - would integrate with actual P2P system
        return 0

    async def _get_resource_utilization(self) -> float:
        """Get current resource utilization percentage."""
        try:
            import psutil

            return psutil.cpu_percent(interval=0.1)
        except ImportError:
            return 0.0

    async def _get_active_evolution_tasks(self) -> int:
        """Get number of active evolution tasks."""
        # Mock implementation - would integrate with actual evolution system
        return 0

    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics for Sprint 6 systems."""
        timestamp = datetime.now().isoformat()

        try:
            import psutil

            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()

            metrics = PerformanceMetrics(
                timestamp=timestamp,
                avg_p2p_latency_ms=await self._measure_p2p_latency(),
                resource_allocation_efficiency=await self._measure_resource_efficiency(),
                evolution_throughput=await self._measure_evolution_throughput(),
                system_load=cpu_percent,
                memory_pressure=memory.percent,
                disk_io_rate=disk_io.read_bytes + disk_io.write_bytes if disk_io else 0.0,
            )

            return metrics

        except ImportError:
            # Fallback metrics if psutil not available
            return PerformanceMetrics(
                timestamp=timestamp,
                avg_p2p_latency_ms=0.0,
                resource_allocation_efficiency=0.0,
                evolution_throughput=0.0,
                system_load=0.0,
                memory_pressure=0.0,
                disk_io_rate=0.0,
            )

    async def _measure_p2p_latency(self) -> float:
        """Measure average P2P latency."""
        # Mock implementation - would ping actual P2P nodes
        return 50.0

    async def _measure_resource_efficiency(self) -> float:
        """Measure resource allocation efficiency."""
        # Mock implementation - would calculate based on actual resource usage
        return 85.0

    async def _measure_evolution_throughput(self) -> float:
        """Measure evolution system throughput."""
        # Mock implementation - would measure actual evolution operations
        return 1.5

    def check_alerts(self, health: InfrastructureHealth, metrics: PerformanceMetrics) -> None:
        """Check for alert conditions and create alerts."""
        current_alerts = []

        # P2P latency alerts
        if metrics.avg_p2p_latency_ms > self.thresholds["p2p_latency_critical"]:
            current_alerts.append(
                self._create_alert(
                    "p2p_latency_critical",
                    "critical",
                    "P2P Communication",
                    f"Critical P2P latency: {metrics.avg_p2p_latency_ms:.1f}ms",
                    {
                        "latency_ms": metrics.avg_p2p_latency_ms,
                        "threshold": self.thresholds["p2p_latency_critical"],
                    },
                )
            )
        elif metrics.avg_p2p_latency_ms > self.thresholds["p2p_latency_warning"]:
            current_alerts.append(
                self._create_alert(
                    "p2p_latency_warning",
                    "warning",
                    "P2P Communication",
                    f"High P2P latency: {metrics.avg_p2p_latency_ms:.1f}ms",
                    {
                        "latency_ms": metrics.avg_p2p_latency_ms,
                        "threshold": self.thresholds["p2p_latency_warning"],
                    },
                )
            )

        # Resource utilization alerts
        if health.resource_utilization > self.thresholds["resource_utilization_critical"]:
            current_alerts.append(
                self._create_alert(
                    "resource_utilization_critical",
                    "critical",
                    "Resource Management",
                    f"Critical resource utilization: {health.resource_utilization:.1f}%",
                    {
                        "utilization": health.resource_utilization,
                        "threshold": self.thresholds["resource_utilization_critical"],
                    },
                )
            )
        elif health.resource_utilization > self.thresholds["resource_utilization_warning"]:
            current_alerts.append(
                self._create_alert(
                    "resource_utilization_warning",
                    "warning",
                    "Resource Management",
                    f"High resource utilization: {health.resource_utilization:.1f}%",
                    {
                        "utilization": health.resource_utilization,
                        "threshold": self.thresholds["resource_utilization_warning"],
                    },
                )
            )

        # Memory pressure alerts
        if metrics.memory_pressure > self.thresholds["memory_pressure_critical"]:
            current_alerts.append(
                self._create_alert(
                    "memory_pressure_critical",
                    "critical",
                    "System Resources",
                    f"Critical memory pressure: {metrics.memory_pressure:.1f}%",
                    {
                        "memory_pressure": metrics.memory_pressure,
                        "threshold": self.thresholds["memory_pressure_critical"],
                    },
                )
            )
        elif metrics.memory_pressure > self.thresholds["memory_pressure_warning"]:
            current_alerts.append(
                self._create_alert(
                    "memory_pressure_warning",
                    "warning",
                    "System Resources",
                    f"High memory pressure: {metrics.memory_pressure:.1f}%",
                    {
                        "memory_pressure": metrics.memory_pressure,
                        "threshold": self.thresholds["memory_pressure_warning"],
                    },
                )
            )

        # Validation age alerts
        if health.last_validation_time:
            try:
                last_validation = datetime.fromisoformat(health.last_validation_time)
                age_seconds = (datetime.now() - last_validation).total_seconds()

                if age_seconds > self.thresholds["validation_age_critical"]:
                    current_alerts.append(
                        self._create_alert(
                            "validation_age_critical",
                            "critical",
                            "System Validation",
                            f"Sprint 6 validation is {age_seconds / 3600:.1f} hours old",
                            {
                                "age_hours": age_seconds / 3600,
                                "threshold_hours": self.thresholds["validation_age_critical"] / 3600,
                            },
                        )
                    )
                elif age_seconds > self.thresholds["validation_age_warning"]:
                    current_alerts.append(
                        self._create_alert(
                            "validation_age_warning",
                            "warning",
                            "System Validation",
                            f"Sprint 6 validation is {age_seconds / 3600:.1f} hours old",
                            {
                                "age_hours": age_seconds / 3600,
                                "threshold_hours": self.thresholds["validation_age_warning"] / 3600,
                            },
                        )
                    )
            except Exception:
                pass

        # Update active alerts
        self._update_alerts(current_alerts)

    def _create_alert(
        self,
        alert_id: str,
        severity: str,
        component: str,
        message: str,
        details: dict[str, Any],
    ) -> AlertInfo:
        """Create a new alert."""
        return AlertInfo(
            alert_id=alert_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            component=component,
            message=message,
            details=details,
        )

    def _update_alerts(self, current_alerts: list[AlertInfo]) -> None:
        """Update the active alerts list."""
        # Resolve alerts that are no longer active
        current_alert_ids = {alert.alert_id for alert in current_alerts}

        for alert in self.active_alerts:
            if alert.alert_id not in current_alert_ids and not alert.resolved:
                alert.resolved = True
                logger.info(f"Resolved alert: {alert.alert_id}")

        # Add new alerts
        existing_alert_ids = {alert.alert_id for alert in self.active_alerts if not alert.resolved}

        for alert in current_alerts:
            if alert.alert_id not in existing_alert_ids:
                self.active_alerts.append(alert)
                logger.warning(f"New {alert.severity} alert: {alert.message}")

    async def run_sprint6_validation(self) -> bool:
        """Run Sprint 6 validation and return success status."""
        try:
            logger.info("Running Sprint 6 validation...")

            result = subprocess.run(
                [sys.executable, "validate_sprint6.py"],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=Path.cwd(),
            )

            success = result.returncode == 0
            logger.info(f"Sprint 6 validation {'passed' if success else 'failed'}")

            return success

        except subprocess.TimeoutExpired:
            logger.exception("Sprint 6 validation timed out")
            return False
        except Exception as e:
            logger.exception(f"Sprint 6 validation failed: {e}")
            return False

    async def monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting Sprint 6 monitoring loop...")

        while self.is_monitoring:
            try:
                # Collect health and performance data
                health = await self.check_infrastructure_health()
                metrics = await self.collect_performance_metrics()

                # Store data
                self.health_history.append(health)
                self.performance_history.append(metrics)

                # Keep only recent data in memory
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                if len(self.performance_history) > 100:
                    self.performance_history = self.performance_history[-100:]

                # Check for alerts
                self.check_alerts(health, metrics)

                # Log current status
                logger.info(
                    f"Health: {health.overall_health}, CPU: {metrics.system_load:.1f}%, Memory: {metrics.memory_pressure:.1f}%, Active alerts: {len([a for a in self.active_alerts if not a.resolved])}"
                )

                # Save data periodically
                self.save_monitoring_data()

                # Run validation periodically (every 30 minutes)
                if len(self.health_history) % 60 == 0:  # Every 60 checks = ~30 minutes at 30sec interval
                    await self.run_sprint6_validation()

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.exception(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)

    def start_monitoring(self) -> None:
        """Start monitoring."""
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return

        self.is_monitoring = True
        logger.info("Starting Sprint 6 infrastructure monitoring...")

        # Run monitoring loop
        asyncio.create_task(self.monitoring_loop())

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        self.save_monitoring_data()
        logger.info("Sprint 6 monitoring stopped")

    def get_status_summary(self) -> dict[str, Any]:
        """Get current status summary."""
        latest_health = self.health_history[-1] if self.health_history else None
        latest_metrics = self.performance_history[-1] if self.performance_history else None
        active_alerts = [a for a in self.active_alerts if not a.resolved]

        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring_active": self.is_monitoring,
            "overall_health": latest_health.overall_health if latest_health else "unknown",
            "components": {
                "p2p": latest_health.p2p_status if latest_health else "unknown",
                "resources": latest_health.resource_management_status if latest_health else "unknown",
                "evolution": latest_health.evolution_system_status if latest_health else "unknown",
            },
            "metrics": {
                "system_load": latest_metrics.system_load if latest_metrics else 0,
                "memory_pressure": latest_metrics.memory_pressure if latest_metrics else 0,
                "p2p_latency": latest_metrics.avg_p2p_latency_ms if latest_metrics else 0,
            },
            "alerts": {
                "critical": len([a for a in active_alerts if a.severity == "critical"]),
                "warning": len([a for a in active_alerts if a.severity == "warning"]),
                "total": len(active_alerts),
            },
            "validation": {
                "last_run": latest_health.last_validation_time if latest_health else None,
                "success": latest_health.last_validation_success if latest_health else False,
            },
        }


async def main() -> None:
    """Main monitoring function."""
    monitor = Sprint6Monitor()

    try:
        monitor.start_monitoring()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        monitor.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
