"""Production monitoring and alerting system for Frontier Curriculum Engine.

Provides comprehensive monitoring, metrics collection, alerting, and
observability for curriculum engine operations in production environments.
"""

import asyncio
import json
import logging
import sqlite3
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Represents a monitoring alert."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    component: str
    metric_name: str | None
    metric_value: float | None
    threshold: float | None
    timestamp: datetime
    resolved: bool = False
    resolved_at: datetime | None = None


@dataclass
class Metric:
    """Represents a collected metric."""

    name: str
    type: MetricType
    value: float | int
    timestamp: datetime
    labels: dict[str, str]
    component: str


@dataclass
class HealthCheck:
    """Represents a component health check."""

    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: datetime
    response_time_ms: float
    details: dict[str, Any]


class CurriculumMonitor:
    """Main monitoring system for curriculum engine."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.alerts: list[Alert] = []
        self.alert_handlers: list[Callable] = []
        self.thresholds = self._load_default_thresholds()
        self.health_checks: dict[str, HealthCheck] = {}

        # Performance tracking
        self.operation_timers: dict[str, float] = {}
        self.component_stats: dict[str, dict[str, Any]] = defaultdict(dict)

        # Database for persistent storage
        self.db_path = self.config.get("db_path", ".forge/monitoring.db")
        self._init_database()

        logger.info("Curriculum monitoring system initialized")

    def _load_default_thresholds(self) -> dict[str, dict[str, float]]:
        """Load default alert thresholds."""
        return {
            "api_response_time": {
                "warning": 5.0,  # seconds
                "error": 10.0,
                "critical": 30.0,
            },
            "success_rate": {
                "warning": 0.9,  # 90%
                "error": 0.8,  # 80%
                "critical": 0.7,  # 70%
            },
            "cache_hit_rate": {
                "warning": 0.8,  # 80%
                "error": 0.6,  # 60%
                "critical": 0.4,  # 40%
            },
            "queue_depth": {"warning": 100, "error": 500, "critical": 1000},
            "memory_usage_mb": {
                "warning": 1000,  # 1GB
                "error": 2000,  # 2GB
                "critical": 4000,  # 4GB
            },
            "error_rate": {
                "warning": 0.05,  # 5%
                "error": 0.10,  # 10%
                "critical": 0.20,  # 20%
            },
        }

    def _init_database(self):
        """Initialize monitoring database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                component TEXT NOT NULL,
                labels TEXT,
                INDEX idx_name_timestamp (name, timestamp),
                INDEX idx_component (component)
            );

            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                component TEXT NOT NULL,
                metric_name TEXT,
                metric_value REAL,
                threshold REAL,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                resolved_at TEXT,
                INDEX idx_severity (severity),
                INDEX idx_component (component),
                INDEX idx_timestamp (timestamp)
            );

            CREATE TABLE IF NOT EXISTS health_checks (
                component TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                response_time_ms REAL NOT NULL,
                details TEXT
            );
        """
        )
        conn.commit()
        conn.close()

    def record_metric(
        self,
        name: str,
        value: float | int,
        metric_type: MetricType = MetricType.GAUGE,
        component: str = "curriculum",
        labels: dict[str, str] | None = None,
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            type=metric_type,
            value=float(value),
            timestamp=datetime.utcnow(),
            component=component,
            labels=labels or {},
        )

        self.metrics.append(metric)
        self._persist_metric(metric)
        self._check_thresholds(metric)

    def start_timer(self, operation: str) -> str:
        """Start timing an operation."""
        timer_id = f"{operation}_{int(time.time() * 1000000)}"
        self.operation_timers[timer_id] = time.time()
        return timer_id

    def end_timer(self, timer_id: str, component: str = "curriculum") -> float:
        """End timing and record duration metric."""
        if timer_id not in self.operation_timers:
            logger.warning(f"Timer {timer_id} not found")
            return 0.0

        duration = time.time() - self.operation_timers[timer_id]
        del self.operation_timers[timer_id]

        # Extract operation name from timer_id
        operation = timer_id.rsplit("_", 1)[0]

        self.record_metric(
            name=f"{operation}_duration",
            value=duration,
            metric_type=MetricType.TIMER,
            component=component,
            labels={"operation": operation},
        )

        return duration

    async def health_check(self, component: str, check_func: Callable) -> HealthCheck:
        """Perform health check on a component."""
        start_time = time.time()

        try:
            result = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
            response_time = (time.time() - start_time) * 1000  # ms

            if isinstance(result, dict):
                status = result.get("status", "healthy")
                message = result.get("message", "Component operational")
                details = result.get("details", {})
            else:
                status = "healthy" if result else "unhealthy"
                message = "Health check passed" if result else "Health check failed"
                details = {}

            health = HealthCheck(
                component=component,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details=details,
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health = HealthCheck(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time,
                details={"error": str(e)},
            )

            logger.error(f"Health check failed for {component}: {e}")

        self.health_checks[component] = health
        self._persist_health_check(health)

        # Record metrics
        self.record_metric(
            name="health_check_response_time",
            value=health.response_time_ms,
            metric_type=MetricType.TIMER,
            component=component,
            labels={"status": health.status},
        )

        return health

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        component: str,
        metric_name: str | None = None,
        metric_value: float | None = None,
        threshold: float | None = None,
    ) -> Alert:
        """Create and process an alert."""
        alert = Alert(
            id=f"{component}_{metric_name or 'general'}_{int(time.time())}",
            severity=severity,
            title=title,
            message=message,
            component=component,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            timestamp=datetime.utcnow(),
        )

        self.alerts.append(alert)
        self._persist_alert(alert)

        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"Alert created: [{severity.value}] {title} - {message}")
        return alert

    def resolve_alert(self, alert_id: str, message: str = ""):
        """Resolve an alert."""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()

                self._update_alert_resolution(alert_id, message)
                logger.info(f"Alert resolved: {alert_id} - {message}")
                break

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        now = datetime.utcnow()

        # Recent metrics (last 5 minutes)
        recent_metrics = [m for m in self.metrics if (now - m.timestamp).total_seconds() < 300]

        # Active alerts
        active_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]

        # Component health
        unhealthy_components = [name for name, health in self.health_checks.items() if health.status == "unhealthy"]

        # Overall status
        if critical_alerts or unhealthy_components:
            overall_status = "critical"
        elif any(a.severity in [AlertSeverity.ERROR, AlertSeverity.WARNING] for a in active_alerts):
            overall_status = "warning"
        else:
            overall_status = "healthy"

        return {
            "overall_status": overall_status,
            "timestamp": now.isoformat(),
            "metrics": {
                "recent_count": len(recent_metrics),
                "components_monitored": len(set(m.component for m in recent_metrics)),
            },
            "alerts": {
                "active_total": len(active_alerts),
                "critical": len(critical_alerts),
                "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            },
            "health_checks": {
                "total_components": len(self.health_checks),
                "healthy": len([h for h in self.health_checks.values() if h.status == "healthy"]),
                "degraded": len([h for h in self.health_checks.values() if h.status == "degraded"]),
                "unhealthy": len(unhealthy_components),
            },
        }

    def get_metrics_summary(self, component: str | None = None, hours: int = 24) -> dict[str, Any]:
        """Get metrics summary for analysis."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        if component:
            metrics = [m for m in self.metrics if m.component == component and m.timestamp > cutoff]
        else:
            metrics = [m for m in self.metrics if m.timestamp > cutoff]

        if not metrics:
            return {"error": "No metrics found for criteria"}

        # Group by metric name
        by_name = defaultdict(list)
        for metric in metrics:
            by_name[metric.name].append(metric.value)

        summary = {}
        for name, values in by_name.items():
            summary[name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1] if values else 0,
            }

        return {
            "timespan_hours": hours,
            "component": component or "all",
            "total_metrics": len(metrics),
            "metrics": summary,
        }

    def _check_thresholds(self, metric: Metric):
        """Check metric against thresholds and create alerts."""
        if metric.name not in self.thresholds:
            return

        thresholds = self.thresholds[metric.name]

        # Check in order of severity
        for severity_name in ["critical", "error", "warning"]:
            if severity_name in thresholds:
                threshold = thresholds[severity_name]

                # Different comparison logic for different metrics
                if self._threshold_exceeded(metric.name, metric.value, threshold):
                    severity = AlertSeverity(severity_name.upper())

                    self.create_alert(
                        severity=severity,
                        title=f"{metric.component.title()} {severity_name.title()}: {metric.name}",
                        message=f"Metric {metric.name} value {metric.value} exceeded threshold {threshold}",
                        component=metric.component,
                        metric_name=metric.name,
                        metric_value=metric.value,
                        threshold=threshold,
                    )
                    break

    def _threshold_exceeded(self, metric_name: str, value: float, threshold: float) -> bool:
        """Check if threshold is exceeded based on metric type."""
        # Metrics where lower values are bad
        if metric_name in ["success_rate", "cache_hit_rate"]:
            return value < threshold

        # Metrics where higher values are bad
        else:
            return value > threshold

    def _persist_metric(self, metric: Metric):
        """Persist metric to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO metrics (name, type, value, timestamp, component, labels)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    metric.name,
                    metric.type.value,
                    metric.value,
                    metric.timestamp.isoformat(),
                    metric.component,
                    json.dumps(metric.labels),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist metric: {e}")

    def _persist_alert(self, alert: Alert):
        """Persist alert to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT INTO alerts (id, severity, title, message, component, metric_name,
                                  metric_value, threshold, timestamp, resolved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.id,
                    alert.severity.value,
                    alert.title,
                    alert.message,
                    alert.component,
                    alert.metric_name,
                    alert.metric_value,
                    alert.threshold,
                    alert.timestamp.isoformat(),
                    alert.resolved,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist alert: {e}")

    def _persist_health_check(self, health: HealthCheck):
        """Persist health check to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                INSERT OR REPLACE INTO health_checks
                (component, status, message, timestamp, response_time_ms, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    health.component,
                    health.status,
                    health.message,
                    health.timestamp.isoformat(),
                    health.response_time_ms,
                    json.dumps(health.details),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to persist health check: {e}")

    def _update_alert_resolution(self, alert_id: str, message: str):
        """Update alert resolution in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """
                UPDATE alerts
                SET resolved = 1, resolved_at = ?
                WHERE id = ?
            """,
                (datetime.utcnow().isoformat(), alert_id),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update alert resolution: {e}")


class CurriculumHealthChecks:
    """Predefined health checks for curriculum components."""

    def __init__(self, curriculum_orchestrator):
        self.orchestrator = curriculum_orchestrator

    async def check_llm_client_health(self) -> dict[str, Any]:
        """Check LLM client connectivity and response time."""
        try:
            start_time = time.time()

            # Simple test query
            response = await self.orchestrator.llm_client.invoke(
                "Test connectivity. Respond with 'OK'.", max_tokens=10, temperature=0.1
            )

            response_time = time.time() - start_time

            if "OK" in response.upper() or len(response.strip()) > 0:
                return {
                    "status": "healthy",
                    "message": f"LLM client responding in {response_time:.2f}s",
                    "details": {
                        "response_time": response_time,
                        "response_length": len(response),
                    },
                }
            else:
                return {
                    "status": "degraded",
                    "message": "LLM client responding but with unexpected output",
                    "details": {"response": response[:100]},
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"LLM client health check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def check_curriculum_state_health(self) -> dict[str, Any]:
        """Check curriculum state and queue health."""
        try:
            status = await self.orchestrator.get_curriculum_status()

            status.get("queues", {})
            health_status = status.get("system_health", "unknown")

            if health_status == "critical - no problems queued":
                return {
                    "status": "unhealthy",
                    "message": "Critical: No problems queued",
                    "details": status,
                }
            elif "warning" in health_status.lower():
                return {
                    "status": "degraded",
                    "message": f"Warning: {health_status}",
                    "details": status,
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"Curriculum state: {health_status}",
                    "details": status,
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Curriculum state check failed: {str(e)}",
                "details": {"error": str(e)},
            }

    async def check_cache_health(self) -> dict[str, Any]:
        """Check cache system health."""
        try:
            if hasattr(self.orchestrator.llm_client, "get_cache_stats"):
                stats = self.orchestrator.llm_client.get_cache_stats()

                if "error" in stats:
                    return {
                        "status": "degraded",
                        "message": f"Cache error: {stats['error']}",
                        "details": stats,
                    }

                hit_rate = 0.0
                if stats.get("total_entries", 0) > 0:
                    # Calculate approximate hit rate from recent activity
                    hit_rate = 0.8  # Placeholder - would need actual hit/miss tracking

                if hit_rate < 0.4:
                    status = "degraded"
                    message = f"Low cache hit rate: {hit_rate:.1%}"
                else:
                    status = "healthy"
                    message = f"Cache operational, {stats['total_entries']} entries"

                return {"status": status, "message": message, "details": stats}
            else:
                return {
                    "status": "unknown",
                    "message": "Cache stats not available",
                    "details": {},
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Cache health check failed: {str(e)}",
                "details": {"error": str(e)},
            }


def create_default_alert_handlers() -> list[Callable]:
    """Create default alert handlers."""

    def console_alert_handler(alert: Alert):
        """Log alerts to console."""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }

        level = level_map.get(alert.severity, logging.INFO)
        logger.log(
            level,
            f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}",
        )

    def file_alert_handler(alert: Alert):
        """Write alerts to file."""
        try:
            alert_file = Path(".forge/alerts.jsonl")
            alert_file.parent.mkdir(parents=True, exist_ok=True)

            with alert_file.open("a") as f:
                f.write(json.dumps(asdict(alert), default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    return [console_alert_handler, file_alert_handler]


# Global monitoring instance (can be configured externally)
_global_monitor: CurriculumMonitor | None = None


def get_monitor() -> CurriculumMonitor:
    """Get global monitoring instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = CurriculumMonitor()

        # Add default alert handlers
        for handler in create_default_alert_handlers():
            _global_monitor.add_alert_handler(handler)

    return _global_monitor


def init_monitoring(config: dict[str, Any] | None = None) -> CurriculumMonitor:
    """Initialize global monitoring system."""
    global _global_monitor
    _global_monitor = CurriculumMonitor(config)

    # Add default alert handlers
    for handler in create_default_alert_handlers():
        _global_monitor.add_alert_handler(handler)

    return _global_monitor


# Monitoring decorators for easy instrumentation
def monitor_operation(component: str = "curriculum", metric_name: str | None = None):
    """Decorator to monitor function execution."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            operation_name = metric_name or func.__name__

            timer_id = monitor.start_timer(operation_name)

            try:
                if asyncio.iscoroutinefunction(func):

                    async def async_wrapper():
                        try:
                            result = await func(*args, **kwargs)
                            monitor.record_metric(
                                f"{operation_name}_success",
                                1,
                                MetricType.COUNTER,
                                component,
                            )
                            return result
                        except Exception:
                            monitor.record_metric(
                                f"{operation_name}_error",
                                1,
                                MetricType.COUNTER,
                                component,
                            )
                            raise
                        finally:
                            monitor.end_timer(timer_id, component)

                    return async_wrapper()
                else:
                    result = func(*args, **kwargs)
                    monitor.record_metric(f"{operation_name}_success", 1, MetricType.COUNTER, component)
                    return result

            except Exception:
                monitor.record_metric(f"{operation_name}_error", 1, MetricType.COUNTER, component)
                raise
            finally:
                if not asyncio.iscoroutinefunction(func):
                    monitor.end_timer(timer_id, component)

        return wrapper

    return decorator
