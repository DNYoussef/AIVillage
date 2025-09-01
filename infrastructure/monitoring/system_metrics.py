"""System-wide metrics collection and monitoring infrastructure.

This module provides comprehensive system monitoring capabilities including:
- Resource usage tracking (CPU, memory, network, disk)
- Performance metrics collection
- Health status monitoring
- Alert generation and notification
- Metrics aggregation and reporting
"""

import asyncio
import json
import logging
import psutil
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """System alert representation."""

    id: str
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MetricThreshold:
    """Metric threshold configuration."""

    metric_name: str
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    comparison_operator: str = "gt"  # gt, lt, eq, gte, lte
    window_duration: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    min_samples: int = 3


@dataclass
class SystemSnapshot:
    """Point-in-time system state snapshot."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    process_count: int
    load_average: Optional[List[float]] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""

    @abstractmethod
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics and return as dictionary."""
        pass

    @abstractmethod
    def get_metric_definitions(self) -> Dict[str, MetricType]:
        """Return metric definitions."""
        pass


class SystemResourcesCollector(MetricsCollector):
    """Collects system resource metrics using psutil."""

    def __init__(self):
        self.last_network_io = None
        self.last_disk_io = None
        self.last_timestamp = None

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect system resource metrics."""
        current_time = time.time()

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk_usage = psutil.disk_usage("/")
        disk_io = psutil.disk_io_counters()

        # Network metrics
        network_io = psutil.net_io_counters()

        # Process metrics
        process_count = len(psutil.pids())

        # Calculate rates for network and disk I/O
        network_send_rate = 0
        network_recv_rate = 0
        disk_read_rate = 0
        disk_write_rate = 0

        if self.last_network_io and self.last_timestamp:
            time_delta = current_time - self.last_timestamp
            if time_delta > 0:
                network_send_rate = (network_io.bytes_sent - self.last_network_io.bytes_sent) / time_delta
                network_recv_rate = (network_io.bytes_recv - self.last_network_io.bytes_recv) / time_delta

                if self.last_disk_io:
                    disk_read_rate = (disk_io.read_bytes - self.last_disk_io.read_bytes) / time_delta
                    disk_write_rate = (disk_io.write_bytes - self.last_disk_io.write_bytes) / time_delta

        self.last_network_io = network_io
        self.last_disk_io = disk_io
        self.last_timestamp = current_time

        # Load average (Unix-like systems only)
        load_average = None
        if hasattr(psutil, "getloadavg"):
            try:
                load_average = list(psutil.getloadavg())
            except (AttributeError, OSError):
                pass

        return {
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "swap_percent": swap.percent,
            "disk_percent": disk_usage.percent,
            "disk_free_gb": disk_usage.free / (1024**3),
            "disk_used_gb": disk_usage.used / (1024**3),
            "disk_total_gb": disk_usage.total / (1024**3),
            "network_bytes_sent_total": network_io.bytes_sent,
            "network_bytes_recv_total": network_io.bytes_recv,
            "network_send_rate_bps": network_send_rate,
            "network_recv_rate_bps": network_recv_rate,
            "disk_read_rate_bps": disk_read_rate,
            "disk_write_rate_bps": disk_write_rate,
            "process_count": process_count,
            "load_average": load_average,
        }

    def get_metric_definitions(self) -> Dict[str, MetricType]:
        """Return metric definitions."""
        return {
            "cpu_percent": MetricType.GAUGE,
            "memory_percent": MetricType.GAUGE,
            "memory_available_gb": MetricType.GAUGE,
            "disk_percent": MetricType.GAUGE,
            "network_send_rate_bps": MetricType.GAUGE,
            "network_recv_rate_bps": MetricType.GAUGE,
            "process_count": MetricType.GAUGE,
        }


class SystemMetricsManager:
    """Centralized system metrics management and monitoring."""

    def __init__(self, collection_interval: float = 30.0):
        """Initialize system metrics manager.

        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.collectors: List[MetricsCollector] = []
        self.thresholds: Dict[str, MetricThreshold] = {}
        self.alerts: Dict[str, Alert] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.snapshots: deque = deque(maxlen=1000)

        # State management
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        self.alert_handlers: List[Callable[[Alert], None]] = []

        # Performance tracking
        self.collection_count = 0
        self.last_collection_duration = 0.0
        self.average_collection_duration = 0.0

        # Add default system collector
        self.add_collector(SystemResourcesCollector())

        logger.info(f"SystemMetricsManager initialized with {collection_interval}s interval")

    def add_collector(self, collector: MetricsCollector) -> None:
        """Add a metrics collector."""
        self.collectors.append(collector)
        logger.info(f"Added metrics collector: {collector.__class__.__name__}")

    def remove_collector(self, collector: MetricsCollector) -> None:
        """Remove a metrics collector."""
        if collector in self.collectors:
            self.collectors.remove(collector)
            logger.info(f"Removed metrics collector: {collector.__class__.__name__}")

    def set_threshold(self, threshold: MetricThreshold) -> None:
        """Set threshold for a metric."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Set threshold for metric {threshold.metric_name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add alert handler function."""
        self.alert_handlers.append(handler)
        logger.info("Added alert handler")

    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_running:
            logger.warning("Metrics collection is already running")
            return

        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Started metrics collection")

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        if not self.is_running:
            return

        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped metrics collection")

    async def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.is_running:
            try:
                start_time = time.time()

                # Collect metrics from all collectors
                all_metrics = {}
                for collector in self.collectors:
                    try:
                        metrics = await collector.collect_metrics()
                        all_metrics.update(metrics)
                    except Exception as e:
                        logger.error(f"Error collecting metrics from {collector.__class__.__name__}: {e}")

                # Store metrics in history
                timestamp = datetime.now()
                for metric_name, value in all_metrics.items():
                    self.metrics_history[metric_name].append((timestamp, value))

                # Create system snapshot
                snapshot = self._create_snapshot(timestamp, all_metrics)
                self.snapshots.append(snapshot)

                # Check thresholds and generate alerts
                await self._check_thresholds(all_metrics, timestamp)

                # Update collection metrics
                collection_duration = time.time() - start_time
                self.last_collection_duration = collection_duration
                self.collection_count += 1

                # Update average collection duration
                alpha = 0.1  # Smoothing factor
                if self.average_collection_duration == 0:
                    self.average_collection_duration = collection_duration
                else:
                    self.average_collection_duration = (
                        alpha * collection_duration + (1 - alpha) * self.average_collection_duration
                    )

                # Log collection statistics periodically
                if self.collection_count % 100 == 0:
                    logger.info(
                        f"Collected {self.collection_count} metric samples. "
                        f"Avg duration: {self.average_collection_duration:.3f}s"
                    )

                # Wait for next collection
                await asyncio.sleep(self.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    def _create_snapshot(self, timestamp: datetime, metrics: Dict[str, Any]) -> SystemSnapshot:
        """Create system snapshot from metrics."""
        return SystemSnapshot(
            timestamp=timestamp,
            cpu_percent=metrics.get("cpu_percent", 0.0),
            memory_percent=metrics.get("memory_percent", 0.0),
            disk_percent=metrics.get("disk_percent", 0.0),
            network_bytes_sent=metrics.get("network_bytes_sent_total", 0),
            network_bytes_recv=metrics.get("network_bytes_recv_total", 0),
            active_connections=metrics.get("active_connections", 0),
            process_count=metrics.get("process_count", 0),
            load_average=metrics.get("load_average"),
            custom_metrics={
                k: v
                for k, v in metrics.items()
                if k
                not in [
                    "cpu_percent",
                    "memory_percent",
                    "disk_percent",
                    "network_bytes_sent_total",
                    "network_bytes_recv_total",
                    "active_connections",
                    "process_count",
                    "load_average",
                ]
            },
        )

    async def _check_thresholds(self, metrics: Dict[str, Any], timestamp: datetime) -> None:
        """Check metrics against configured thresholds."""
        for metric_name, threshold in self.thresholds.items():
            if metric_name not in metrics:
                continue

            metrics[metric_name]

            # Get recent values for the metric
            recent_values = []
            cutoff_time = timestamp - threshold.window_duration

            for ts, value in reversed(list(self.metrics_history[metric_name])):
                if ts >= cutoff_time:
                    recent_values.append(value)
                else:
                    break

            if len(recent_values) < threshold.min_samples:
                continue

            # Check thresholds
            avg_value = sum(recent_values) / len(recent_values)

            # Determine if threshold is breached
            is_critical = self._is_threshold_breached(
                avg_value, threshold.critical_threshold, threshold.comparison_operator
            )
            is_warning = self._is_threshold_breached(
                avg_value, threshold.warning_threshold, threshold.comparison_operator
            )

            alert_id = f"{metric_name}_threshold"
            existing_alert = self.alerts.get(alert_id)

            if is_critical:
                if not existing_alert or existing_alert.severity != AlertSeverity.CRITICAL:
                    alert = Alert(
                        id=alert_id,
                        severity=AlertSeverity.CRITICAL,
                        title=f"Critical threshold exceeded for {metric_name}",
                        description=f"Metric {metric_name} has value {avg_value:.2f}, exceeding critical threshold {threshold.critical_threshold}",
                        metric_name=metric_name,
                        current_value=avg_value,
                        threshold=threshold.critical_threshold,
                        timestamp=timestamp,
                    )
                    self.alerts[alert_id] = alert
                    await self._trigger_alert(alert)

            elif is_warning:
                if not existing_alert or existing_alert.severity not in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
                    alert = Alert(
                        id=alert_id,
                        severity=AlertSeverity.WARNING,
                        title=f"Warning threshold exceeded for {metric_name}",
                        description=f"Metric {metric_name} has value {avg_value:.2f}, exceeding warning threshold {threshold.warning_threshold}",
                        metric_name=metric_name,
                        current_value=avg_value,
                        threshold=threshold.warning_threshold,
                        timestamp=timestamp,
                    )
                    self.alerts[alert_id] = alert
                    await self._trigger_alert(alert)

            else:
                # Resolve existing alert if value is back within thresholds
                if existing_alert and not existing_alert.resolved:
                    existing_alert.resolved = True
                    existing_alert.resolved_at = timestamp
                    await self._resolve_alert(existing_alert)

    def _is_threshold_breached(self, value: float, threshold: Optional[float], operator: str) -> bool:
        """Check if value breaches threshold based on operator."""
        if threshold is None:
            return False

        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        elif operator == "eq":
            return value == threshold
        else:
            return False

    async def _trigger_alert(self, alert: Alert) -> None:
        """Trigger alert handlers."""
        logger.warning(f"Alert triggered: {alert.title}")

        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    async def _resolve_alert(self, alert: Alert) -> None:
        """Handle alert resolution."""
        logger.info(f"Alert resolved: {alert.title}")

        # Could trigger resolution handlers here if needed

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        current_metrics = {}

        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = history[-1][1]  # Get latest value

        return current_metrics

    def get_metric_history(self, metric_name: str, duration: Optional[timedelta] = None) -> List[tuple]:
        """Get metric history for specified duration."""
        if metric_name not in self.metrics_history:
            return []

        if duration is None:
            return list(self.metrics_history[metric_name])

        cutoff_time = datetime.now() - duration
        return [(ts, value) for ts, value in self.metrics_history[metric_name] if ts >= cutoff_time]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        if not self.snapshots:
            return 100.0

        latest = self.snapshots[-1]

        # Calculate health score based on resource usage
        cpu_score = max(0, 100 - latest.cpu_percent)
        memory_score = max(0, 100 - latest.memory_percent)
        disk_score = max(0, 100 - latest.disk_percent)

        # Weight the scores
        health_score = cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2

        # Penalize for active critical alerts
        critical_alerts = len([a for a in self.get_active_alerts() if a.severity == AlertSeverity.CRITICAL])
        health_score = max(0, health_score - (critical_alerts * 20))

        return min(100.0, max(0.0, health_score))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            "collection_count": self.collection_count,
            "last_collection_duration": self.last_collection_duration,
            "average_collection_duration": self.average_collection_duration,
            "active_collectors": len(self.collectors),
            "tracked_metrics": len(self.metrics_history),
            "configured_thresholds": len(self.thresholds),
            "active_alerts": len(self.get_active_alerts()),
            "system_health_score": self.get_system_health_score(),
            "is_running": self.is_running,
        }

    async def export_metrics(self, file_path: str, format_type: str = "json") -> bool:
        """Export metrics data to file."""
        try:
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "collection_count": self.collection_count,
                "current_metrics": self.get_current_metrics(),
                "alerts": [asdict(alert) for alert in self.alerts.values()],
                "performance_summary": self.get_performance_summary(),
                "snapshots": [asdict(snapshot) for snapshot in list(self.snapshots)[-100:]],  # Last 100 snapshots
            }

            # Convert datetime objects to ISO strings for JSON serialization
            def datetime_converter(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            if format_type.lower() == "json":
                with open(file_path, "w") as f:
                    json.dump(export_data, f, indent=2, default=datetime_converter)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"Exported metrics to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False


# Default alert handlers
def console_alert_handler(alert: Alert) -> None:
    """Default console alert handler."""
    severity_emoji = {
        AlertSeverity.LOW: "ℹ️",
        AlertSeverity.MEDIUM: "⚠️",
        AlertSeverity.HIGH: "🚨",
        AlertSeverity.CRITICAL: "🔴",
    }

    print(f"{severity_emoji[alert.severity]} [{alert.severity.value.upper()}] {alert.title}")
    print(f"   {alert.description}")
    print(f"   Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")


def log_alert_handler(alert: Alert) -> None:
    """Log-based alert handler."""
    if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
        logger.error(f"ALERT: {alert.title} - {alert.description}")
    elif alert.severity == AlertSeverity.MEDIUM:
        logger.warning(f"ALERT: {alert.title} - {alert.description}")
    else:
        logger.info(f"ALERT: {alert.title} - {alert.description}")


if __name__ == "__main__":

    async def demo():
        """Demonstrate system metrics manager."""
        manager = SystemMetricsManager(collection_interval=5.0)

        # Add alert handlers
        manager.add_alert_handler(console_alert_handler)
        manager.add_alert_handler(log_alert_handler)

        # Configure thresholds
        cpu_threshold = MetricThreshold(
            metric_name="cpu_percent", warning_threshold=70.0, critical_threshold=90.0, comparison_operator="gt"
        )
        manager.set_threshold(cpu_threshold)

        memory_threshold = MetricThreshold(
            metric_name="memory_percent", warning_threshold=80.0, critical_threshold=95.0, comparison_operator="gt"
        )
        manager.set_threshold(memory_threshold)

        print("Starting system metrics collection...")
        await manager.start_collection()

        try:
            # Run for 30 seconds
            await asyncio.sleep(30)

            # Print current status
            print("\n=== System Status ===")
            metrics = manager.get_current_metrics()
            for name, value in metrics.items():
                print(f"{name}: {value}")

            print(f"\nSystem Health Score: {manager.get_system_health_score():.1f}/100")

            active_alerts = manager.get_active_alerts()
            if active_alerts:
                print(f"\nActive Alerts: {len(active_alerts)}")
                for alert in active_alerts:
                    print(f"  - {alert.title}")

            # Export metrics
            await manager.export_metrics("system_metrics_export.json")

        finally:
            await manager.stop_collection()

    # Run the demo
    asyncio.run(demo())
