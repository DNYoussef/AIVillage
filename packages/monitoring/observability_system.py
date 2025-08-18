"""Monitoring & Observability System - Prompt I

Comprehensive monitoring and observability framework including:
- Metrics collection and aggregation
- Distributed tracing and logging
- Performance monitoring and alerting
- System health dashboards
- Real-time monitoring and anomaly detection
- OpenTelemetry-compatible instrumentation

Integration Point: Observability layer for Phase 4 testing
"""

import json
import logging
import sqlite3
import statistics
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SpanKind(Enum):
    """Trace span types."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class LogLevel(Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """A metric measurement."""

    name: str
    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class LogEntry:
    """A log entry."""

    timestamp: float
    level: LogLevel
    message: str
    service: str
    trace_id: str | None = None
    span_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A distributed trace span."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    service_name: str
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    kind: SpanKind = SpanKind.INTERNAL
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Alert:
    """An alert/notification."""

    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    triggered_at: float
    resolved_at: float | None = None
    condition: str = ""
    value: float | None = None
    threshold: float | None = None
    labels: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """System health check result."""

    name: str
    status: str
    timestamp: float
    response_time_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates system metrics."""

    def __init__(self, storage_backend: str = ":memory:"):
        """Initialize metrics collector.

        Args:
            storage_backend: Storage backend (SQLite database path)
        """
        self.storage_backend = storage_backend
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, list[float]] = defaultdict(list)

        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        """Initialize metrics storage."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    labels TEXT DEFAULT '{}'
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp
                ON metrics(name, timestamp)
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def record_counter(self, name: str, value: float = 1.0, labels: dict[str, str] = None):
        """Record a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Metric labels
        """
        with self._lock:
            key = self._metric_key(name, labels or {})
            self.counters[key] += value

            metric = MetricValue(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.COUNTER,
            )

            self.metrics_buffer.append(metric)

    def record_gauge(self, name: str, value: float, labels: dict[str, str] = None):
        """Record a gauge metric.

        Args:
            name: Metric name
            value: Current value
            labels: Metric labels
        """
        with self._lock:
            key = self._metric_key(name, labels or {})
            self.gauges[key] = value

            metric = MetricValue(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.GAUGE,
            )

            self.metrics_buffer.append(metric)

    def record_histogram(self, name: str, value: float, labels: dict[str, str] = None):
        """Record a histogram metric.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        with self._lock:
            key = self._metric_key(name, labels or {})
            self.histograms[key].append(value)

            # Keep only recent values (last 1000)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

            metric = MetricValue(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {},
                metric_type=MetricType.HISTOGRAM,
            )

            self.metrics_buffer.append(metric)

    def record_timer(self, name: str, duration_ms: float, labels: dict[str, str] = None):
        """Record a timer metric.

        Args:
            name: Metric name
            duration_ms: Duration in milliseconds
            labels: Metric labels
        """
        self.record_histogram(f"{name}_duration_ms", duration_ms, labels)

    def flush_to_storage(self) -> int:
        """Flush buffered metrics to storage.

        Returns:
            Number of metrics flushed
        """
        with self._lock:
            if not self.metrics_buffer:
                return 0

            metrics_to_flush = list(self.metrics_buffer)
            self.metrics_buffer.clear()

        with self._get_db() as conn:
            for metric in metrics_to_flush:
                conn.execute(
                    """
                    INSERT INTO metrics (name, value, timestamp, metric_type, labels)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        metric.name,
                        metric.value,
                        metric.timestamp,
                        metric.metric_type.value,
                        json.dumps(metric.labels),
                    ),
                )
            conn.commit()

        return len(metrics_to_flush)

    def get_metric_summary(self, name: str, start_time: float = None, end_time: float = None) -> dict[str, Any]:
        """Get metric summary statistics.

        Args:
            name: Metric name
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            Metric summary
        """
        query = "SELECT * FROM metrics WHERE name = ?"
        params = [name]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)

        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp"

        values = []
        with self._get_db() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                values.append(row["value"])

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sum": sum(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    def _metric_key(self, name: str, labels: dict[str, str]) -> str:
        """Generate metric key including labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


class DistributedTracer:
    """Distributed tracing system."""

    def __init__(self, service_name: str, storage_backend: str = ":memory:"):
        """Initialize distributed tracer.

        Args:
            service_name: Name of this service
            storage_backend: Storage backend for traces
        """
        self.service_name = service_name
        self.storage_backend = storage_backend
        self.active_spans: dict[str, Span] = {}
        self.completed_spans: deque = deque(maxlen=1000)

        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        """Initialize trace storage."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    service_name TEXT NOT NULL,
                    operation_name TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration_ms REAL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attributes TEXT DEFAULT '{}',
                    events TEXT DEFAULT '[]',
                    PRIMARY KEY (trace_id, span_id)
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_spans_trace_id
                ON spans(trace_id)
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def start_span(
        self,
        operation_name: str,
        parent_span_id: str | None = None,
        trace_id: str | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: dict[str, Any] = None,
    ) -> Span:
        """Start a new trace span.

        Args:
            operation_name: Name of the operation
            parent_span_id: Parent span ID
            trace_id: Trace ID (generates new if None)
            kind: Span kind
            attributes: Span attributes

        Returns:
            Started span
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span_id,
            service_name=self.service_name,
            operation_name=operation_name,
            start_time=time.time(),
            kind=kind,
            attributes=attributes or {},
        )

        with self._lock:
            self.active_spans[span.span_id] = span

        return span

    def finish_span(self, span: Span, status: str = "ok"):
        """Finish a trace span.

        Args:
            span: Span to finish
            status: Final span status
        """
        span.end_time = time.time()
        span.duration_ms = (span.end_time - span.start_time) * 1000
        span.status = status

        with self._lock:
            if span.span_id in self.active_spans:
                del self.active_spans[span.span_id]
            self.completed_spans.append(span)

    def add_span_event(self, span: Span, name: str, attributes: dict[str, Any] = None):
        """Add event to span.

        Args:
            span: Target span
            name: Event name
            attributes: Event attributes
        """
        event = {"name": name, "timestamp": time.time(), "attributes": attributes or {}}
        span.events.append(event)

    def set_span_attribute(self, span: Span, key: str, value: Any):
        """Set span attribute.

        Args:
            span: Target span
            key: Attribute key
            value: Attribute value
        """
        span.attributes[key] = value

    def flush_to_storage(self) -> int:
        """Flush completed spans to storage.

        Returns:
            Number of spans flushed
        """
        with self._lock:
            if not self.completed_spans:
                return 0

            spans_to_flush = list(self.completed_spans)
            self.completed_spans.clear()

        with self._get_db() as conn:
            for span in spans_to_flush:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO spans (
                        trace_id, span_id, parent_span_id, service_name,
                        operation_name, start_time, end_time, duration_ms,
                        kind, status, attributes, events
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        span.trace_id,
                        span.span_id,
                        span.parent_span_id,
                        span.service_name,
                        span.operation_name,
                        span.start_time,
                        span.end_time,
                        span.duration_ms,
                        span.kind.value,
                        span.status,
                        json.dumps(span.attributes),
                        json.dumps(span.events),
                    ),
                )
            conn.commit()

        return len(spans_to_flush)

    def get_trace(self, trace_id: str) -> list[Span]:
        """Get all spans for a trace.

        Args:
            trace_id: Trace identifier

        Returns:
            List of spans in trace
        """
        spans = []

        with self._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM spans WHERE trace_id = ?
                ORDER BY start_time
            """,
                (trace_id,),
            )

            for row in cursor.fetchall():
                span = Span(
                    trace_id=row["trace_id"],
                    span_id=row["span_id"],
                    parent_span_id=row["parent_span_id"],
                    service_name=row["service_name"],
                    operation_name=row["operation_name"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    duration_ms=row["duration_ms"],
                    kind=SpanKind(row["kind"]),
                    status=row["status"],
                    attributes=json.loads(row["attributes"]),
                    events=json.loads(row["events"]),
                )
                spans.append(span)

        return spans


@contextmanager
def traced_operation(
    tracer: DistributedTracer,
    operation_name: str,
    parent_span_id: str | None = None,
    attributes: dict[str, Any] = None,
):
    """Context manager for traced operations.

    Args:
        tracer: Distributed tracer
        operation_name: Operation name
        parent_span_id: Parent span ID
        attributes: Span attributes

    Yields:
        Active span
    """
    span = tracer.start_span(
        operation_name=operation_name,
        parent_span_id=parent_span_id,
        attributes=attributes,
    )

    try:
        yield span
        tracer.finish_span(span, "ok")
    except Exception as e:
        tracer.add_span_event(
            span,
            "exception",
            {"exception.type": type(e).__name__, "exception.message": str(e)},
        )
        tracer.finish_span(span, "error")
        raise


class LogHandler:
    """Structured logging handler."""

    def __init__(self, service_name: str, storage_backend: str = ":memory:"):
        """Initialize log handler.

        Args:
            service_name: Service name
            storage_backend: Storage backend for logs
        """
        self.service_name = service_name
        self.storage_backend = storage_backend
        self.log_buffer: deque = deque(maxlen=5000)

        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        """Initialize log storage."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    service TEXT NOT NULL,
                    trace_id TEXT,
                    span_id TEXT,
                    attributes TEXT DEFAULT '{}'
                )
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp_level
                ON logs(timestamp, level)
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def log(
        self,
        level: LogLevel,
        message: str,
        trace_id: str | None = None,
        span_id: str | None = None,
        **attributes,
    ):
        """Log a message.

        Args:
            level: Log level
            message: Log message
            trace_id: Associated trace ID
            span_id: Associated span ID
            **attributes: Additional attributes
        """
        log_entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            service=self.service_name,
            trace_id=trace_id,
            span_id=span_id,
            attributes=attributes,
        )

        with self._lock:
            self.log_buffer.append(log_entry)

        # Also log to standard logger
        python_logger = logging.getLogger(self.service_name)

        log_level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }

        python_logger.log(log_level_map[level], message, extra=attributes)

    def debug(self, message: str, **attributes):
        """Log debug message."""
        self.log(LogLevel.DEBUG, message, **attributes)

    def info(self, message: str, **attributes):
        """Log info message."""
        self.log(LogLevel.INFO, message, **attributes)

    def warning(self, message: str, **attributes):
        """Log warning message."""
        self.log(LogLevel.WARNING, message, **attributes)

    def error(self, message: str, **attributes):
        """Log error message."""
        self.log(LogLevel.ERROR, message, **attributes)

    def critical(self, message: str, **attributes):
        """Log critical message."""
        self.log(LogLevel.CRITICAL, message, **attributes)

    def flush_to_storage(self) -> int:
        """Flush buffered logs to storage.

        Returns:
            Number of logs flushed
        """
        with self._lock:
            if not self.log_buffer:
                return 0

            logs_to_flush = list(self.log_buffer)
            self.log_buffer.clear()

        with self._get_db() as conn:
            for log_entry in logs_to_flush:
                conn.execute(
                    """
                    INSERT INTO logs (
                        timestamp, level, message, service,
                        trace_id, span_id, attributes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        log_entry.timestamp,
                        log_entry.level.value,
                        log_entry.message,
                        log_entry.service,
                        log_entry.trace_id,
                        log_entry.span_id,
                        json.dumps(log_entry.attributes),
                    ),
                )
            conn.commit()

        return len(logs_to_flush)


class AlertManager:
    """Alert and notification manager."""

    def __init__(self, storage_backend: str = ":memory:"):
        """Initialize alert manager.

        Args:
            storage_backend: Storage backend for alerts
        """
        self.storage_backend = storage_backend
        self.active_alerts: dict[str, Alert] = {}
        self.alert_rules: list[dict[str, Any]] = []

        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self):
        """Initialize alert storage."""
        with self._get_db() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    triggered_at REAL NOT NULL,
                    resolved_at REAL,
                    condition TEXT DEFAULT '',
                    value REAL,
                    threshold REAL,
                    labels TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}'
                )
            """
            )

    @contextmanager
    def _get_db(self):
        """Get database connection."""
        conn = sqlite3.connect(self.storage_backend)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def add_alert_rule(
        self,
        name: str,
        condition: str,
        threshold: float,
        severity: AlertSeverity,
        description: str = "",
    ):
        """Add alert rule.

        Args:
            name: Alert rule name
            condition: Alert condition
            threshold: Threshold value
            severity: Alert severity
            description: Alert description
        """
        rule = {
            "name": name,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
            "description": description,
        }

        self.alert_rules.append(rule)

    def trigger_alert(
        self,
        name: str,
        description: str,
        severity: AlertSeverity,
        value: float | None = None,
        threshold: float | None = None,
        labels: dict[str, str] = None,
        metadata: dict[str, Any] = None,
    ) -> Alert:
        """Trigger an alert.

        Args:
            name: Alert name
            description: Alert description
            severity: Alert severity
            value: Current value
            threshold: Threshold value
            labels: Alert labels
            metadata: Additional metadata

        Returns:
            Created alert
        """
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=name,
            description=description,
            severity=severity,
            triggered_at=time.time(),
            value=value,
            threshold=threshold,
            labels=labels or {},
            metadata=metadata or {},
        )

        with self._lock:
            self.active_alerts[alert.alert_id] = alert

        # Store in database
        with self._get_db() as conn:
            conn.execute(
                """
                INSERT INTO alerts (
                    alert_id, name, description, severity, triggered_at,
                    value, threshold, labels, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.name,
                    alert.description,
                    alert.severity.value,
                    alert.triggered_at,
                    alert.value,
                    alert.threshold,
                    json.dumps(alert.labels),
                    json.dumps(alert.metadata),
                ),
            )
            conn.commit()

        return alert

    def resolve_alert(self, alert_id: str):
        """Resolve an alert.

        Args:
            alert_id: Alert ID to resolve
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = time.time()
                del self.active_alerts[alert_id]

                # Update in database
                with self._get_db() as conn:
                    conn.execute(
                        """
                        UPDATE alerts SET resolved_at = ? WHERE alert_id = ?
                    """,
                        (alert.resolved_at, alert_id),
                    )
                    conn.commit()

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts.

        Returns:
            List of active alerts
        """
        with self._lock:
            return list(self.active_alerts.values())

    def check_alert_conditions(self, metrics: dict[str, float]):
        """Check alert conditions against current metrics.

        Args:
            metrics: Current metric values
        """
        for rule in self.alert_rules:
            # Simple condition checking (extensible)
            condition = rule["condition"]
            threshold = rule["threshold"]

            if condition in metrics:
                value = metrics[condition]

                # Check if threshold is exceeded
                if rule["severity"] in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                    if value > threshold:
                        self.trigger_alert(
                            name=rule["name"],
                            description=rule["description"],
                            severity=rule["severity"],
                            value=value,
                            threshold=threshold,
                            metadata={"rule": rule},
                        )


class HealthMonitor:
    """System health monitoring."""

    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: dict[str, Callable] = {}
        self.health_status: dict[str, HealthCheck] = {}
        self.monitoring_active = False
        self._monitor_thread = None

    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check.

        Args:
            name: Health check name
            check_func: Function that returns health status
        """
        self.health_checks[name] = check_func

    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start health monitoring.

        Args:
            interval_seconds: Check interval
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)

    def _monitoring_loop(self, interval: float):
        """Health monitoring loop."""
        while self.monitoring_active:
            self.run_health_checks()
            time.sleep(interval)

    def run_health_checks(self):
        """Run all registered health checks."""
        for name, check_func in self.health_checks.items():
            start_time = time.time()

            try:
                result = check_func()
                response_time = (time.time() - start_time) * 1000

                if isinstance(result, bool):
                    status = "healthy" if result else "unhealthy"
                    details = {}
                elif isinstance(result, dict):
                    status = result.get("status", "unknown")
                    details = result.get("details", {})
                else:
                    status = str(result)
                    details = {}

                health_check = HealthCheck(
                    name=name,
                    status=status,
                    timestamp=time.time(),
                    response_time_ms=response_time,
                    details=details,
                )

                self.health_status[name] = health_check

            except Exception as e:
                health_check = HealthCheck(
                    name=name,
                    status="error",
                    timestamp=time.time(),
                    response_time_ms=(time.time() - start_time) * 1000,
                    details={"error": str(e)},
                )

                self.health_status[name] = health_check

    def get_health_status(self) -> dict[str, HealthCheck]:
        """Get current health status.

        Returns:
            Dictionary of health check results
        """
        return dict(self.health_status)


class ObservabilitySystem:
    """Main observability system coordinator."""

    def __init__(
        self,
        service_name: str,
        storage_backend: str = ":memory:",
        flush_interval: float = 30.0,
    ):
        """Initialize observability system.

        Args:
            service_name: Service name
            storage_backend: Storage backend
            flush_interval: Auto-flush interval in seconds
        """
        self.service_name = service_name
        self.storage_backend = storage_backend
        self.flush_interval = flush_interval

        # Initialize components
        self.metrics = MetricsCollector(storage_backend)
        self.tracer = DistributedTracer(service_name, storage_backend)
        self.logger = LogHandler(service_name, storage_backend)
        self.alerts = AlertManager(storage_backend)
        self.health = HealthMonitor()

        # Auto-flush thread
        self._flush_active = False
        self._flush_thread = None

        # Setup default health checks
        self._setup_default_health_checks()

    def _setup_default_health_checks(self):
        """Setup default health checks."""

        def memory_check():
            """Check memory usage."""
            try:
                import psutil

                memory = psutil.virtual_memory()
                return {
                    "status": "healthy" if memory.percent < 90 else "unhealthy",
                    "details": {
                        "percent_used": memory.percent,
                        "available_gb": memory.available / (1024**3),
                    },
                }
            except ImportError:
                return {
                    "status": "unknown",
                    "details": {"error": "psutil not available"},
                }

        def disk_check():
            """Check disk space."""
            try:
                import psutil

                disk = psutil.disk_usage("/")
                return {
                    "status": "healthy" if disk.percent < 95 else "unhealthy",
                    "details": {
                        "percent_used": disk.percent,
                        "free_gb": disk.free / (1024**3),
                    },
                }
            except ImportError:
                return {
                    "status": "unknown",
                    "details": {"error": "psutil not available"},
                }

        self.health.register_health_check("memory", memory_check)
        self.health.register_health_check("disk", disk_check)

    def start(self):
        """Start observability system."""
        # Start health monitoring
        self.health.start_monitoring()

        # Start auto-flush
        self._flush_active = True
        self._flush_thread = threading.Thread(target=self._auto_flush_loop, daemon=True)
        self._flush_thread.start()

        self.logger.info("Observability system started", service=self.service_name)

    def stop(self):
        """Stop observability system."""
        # Stop health monitoring
        self.health.stop_monitoring()

        # Stop auto-flush
        self._flush_active = False
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)

        # Final flush
        self.flush_all()

        self.logger.info("Observability system stopped", service=self.service_name)

    def _auto_flush_loop(self):
        """Auto-flush loop."""
        while self._flush_active:
            time.sleep(self.flush_interval)
            if self._flush_active:
                self.flush_all()

    def flush_all(self):
        """Flush all buffers to storage."""
        metrics_flushed = self.metrics.flush_to_storage()
        spans_flushed = self.tracer.flush_to_storage()
        logs_flushed = self.logger.flush_to_storage()

        if metrics_flushed + spans_flushed + logs_flushed > 0:
            self.logger.debug(
                "Flushed observability data",
                metrics=metrics_flushed,
                spans=spans_flushed,
                logs=logs_flushed,
            )

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data for monitoring dashboard.

        Returns:
            Dashboard data
        """
        # Get recent metrics summary
        current_time = time.time()

        # Get health status
        health_status = self.health.get_health_status()

        # Get active alerts
        active_alerts = self.alerts.get_active_alerts()

        # Get system overview
        with self.metrics._get_db() as conn:
            cursor = conn.execute(
                """
                SELECT name, COUNT(*) as count, AVG(value) as avg_value
                FROM metrics
                WHERE timestamp > ?
                GROUP BY name
                ORDER BY count DESC
                LIMIT 10
            """,
                (current_time - 3600,),
            )  # Last hour

            top_metrics = [
                {
                    "name": row["name"],
                    "count": row["count"],
                    "avg_value": row["avg_value"],
                }
                for row in cursor.fetchall()
            ]

        return {
            "service_name": self.service_name,
            "timestamp": current_time,
            "health_status": {
                name: {
                    "status": check.status,
                    "response_time_ms": check.response_time_ms,
                    "details": check.details,
                }
                for name, check in health_status.items()
            },
            "active_alerts": [
                {
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "triggered_at": alert.triggered_at,
                    "description": alert.description,
                }
                for alert in active_alerts
            ],
            "top_metrics": top_metrics,
            "system_stats": {
                "metrics_collected": len(self.metrics.metrics_buffer),
                "active_spans": len(self.tracer.active_spans),
                "logs_buffered": len(self.logger.log_buffer),
            },
        }


# Instrumentation decorators and context managers
def timed_operation(observability: ObservabilitySystem, operation_name: str):
    """Decorator for timing operations.

    Args:
        observability: Observability system
        operation_name: Operation name
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()

            with traced_operation(observability.tracer, operation_name) as span:
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    observability.metrics.record_timer(f"{operation_name}_duration", duration_ms, {"status": "success"})

                    observability.logger.info(
                        f"Operation {operation_name} completed",
                        duration_ms=duration_ms,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                    )

                    return result

                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000

                    observability.metrics.record_timer(f"{operation_name}_duration", duration_ms, {"status": "error"})

                    observability.logger.error(
                        f"Operation {operation_name} failed: {e}",
                        duration_ms=duration_ms,
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        error=str(e),
                    )

                    raise

        return wrapper

    return decorator


@contextmanager
def monitored_operation(
    observability: ObservabilitySystem,
    operation_name: str,
    attributes: dict[str, Any] = None,
):
    """Context manager for monitored operations.

    Args:
        observability: Observability system
        operation_name: Operation name
        attributes: Additional attributes

    Yields:
        Monitoring context
    """
    start_time = time.time()

    with traced_operation(observability.tracer, operation_name, attributes=attributes) as span:
        context = {
            "span": span,
            "start_time": start_time,
            "operation_name": operation_name,
        }

        try:
            yield context

            duration_ms = (time.time() - start_time) * 1000
            observability.metrics.record_timer(f"{operation_name}_duration", duration_ms, {"status": "success"})

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            observability.metrics.record_timer(f"{operation_name}_duration", duration_ms, {"status": "error"})

            observability.logger.error(
                f"Monitored operation failed: {e}",
                operation=operation_name,
                duration_ms=duration_ms,
                trace_id=span.trace_id,
                span_id=span.span_id,
            )

            raise
