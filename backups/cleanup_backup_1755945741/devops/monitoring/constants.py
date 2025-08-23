"""Monitoring and observability constants for AIVillage.

This module centralizes all monitoring, metrics, and alerting
magic literals to eliminate connascence of meaning.
"""

from enum import Enum
from typing import Final

# Metrics collection
METRICS_COLLECTION_INTERVAL_SECONDS: Final[int] = 30
METRICS_RETENTION_DAYS: Final[int] = 30
METRICS_BATCH_SIZE: Final[int] = 1000
METRICS_FLUSH_INTERVAL_SECONDS: Final[int] = 60
HIGH_FREQUENCY_METRICS_INTERVAL_SECONDS: Final[int] = 5

# Performance monitoring
CPU_USAGE_WARNING_THRESHOLD: Final[float] = 70.0
CPU_USAGE_CRITICAL_THRESHOLD: Final[float] = 90.0
MEMORY_USAGE_WARNING_THRESHOLD: Final[float] = 75.0
MEMORY_USAGE_CRITICAL_THRESHOLD: Final[float] = 90.0
DISK_USAGE_WARNING_THRESHOLD: Final[float] = 80.0
DISK_USAGE_CRITICAL_THRESHOLD: Final[float] = 95.0

# Network monitoring
NETWORK_LATENCY_WARNING_MS: Final[int] = 100
NETWORK_LATENCY_CRITICAL_MS: Final[int] = 500
NETWORK_PACKET_LOSS_WARNING_PERCENT: Final[float] = 1.0
NETWORK_PACKET_LOSS_CRITICAL_PERCENT: Final[float] = 5.0
BANDWIDTH_UTILIZATION_WARNING_PERCENT: Final[float] = 80.0
BANDWIDTH_UTILIZATION_CRITICAL_PERCENT: Final[float] = 95.0

# Application monitoring
RESPONSE_TIME_WARNING_MS: Final[int] = 1000
RESPONSE_TIME_CRITICAL_MS: Final[int] = 5000
ERROR_RATE_WARNING_PERCENT: Final[float] = 1.0
ERROR_RATE_CRITICAL_PERCENT: Final[float] = 5.0
THROUGHPUT_WARNING_RPS: Final[int] = 100
THROUGHPUT_CRITICAL_RPS: Final[int] = 10

# Health checks
HEALTH_CHECK_INTERVAL_SECONDS: Final[int] = 30
HEALTH_CHECK_TIMEOUT_SECONDS: Final[int] = 5
HEALTH_CHECK_RETRY_ATTEMPTS: Final[int] = 3
HEALTH_CHECK_FAILURE_THRESHOLD: Final[int] = 3
HEALTH_CHECK_SUCCESS_THRESHOLD: Final[int] = 2

# Alerting configuration
ALERT_EVALUATION_INTERVAL_SECONDS: Final[int] = 60
ALERT_GROUPING_WINDOW_SECONDS: Final[int] = 300
ALERT_COOLDOWN_PERIOD_SECONDS: Final[int] = 600
MAX_ALERTS_PER_HOUR: Final[int] = 100
ALERT_ESCALATION_TIMEOUT_SECONDS: Final[int] = 1800

# Dashboard configuration
DASHBOARD_REFRESH_INTERVAL_SECONDS: Final[int] = 30
DASHBOARD_DATA_POINTS_LIMIT: Final[int] = 1000
DASHBOARD_QUERY_TIMEOUT_SECONDS: Final[int] = 30
DASHBOARD_CACHE_TTL_SECONDS: Final[int] = 300

# Log monitoring
LOG_RETENTION_DAYS: Final[int] = 30
LOG_ROTATION_SIZE_MB: Final[int] = 100
LOG_ROTATION_COUNT: Final[int] = 10
LOG_ANALYSIS_BATCH_SIZE: Final[int] = 10000
ERROR_LOG_ALERT_THRESHOLD: Final[int] = 10

# Distributed tracing
TRACE_SAMPLING_RATE: Final[float] = 0.1
TRACE_RETENTION_HOURS: Final[int] = 72
MAX_TRACE_SPAN_COUNT: Final[int] = 1000
TRACE_EXPORT_BATCH_SIZE: Final[int] = 512
TRACE_EXPORT_TIMEOUT_SECONDS: Final[int] = 30

# Service discovery monitoring
SERVICE_DISCOVERY_CHECK_INTERVAL_SECONDS: Final[int] = 60
SERVICE_HEALTH_TIMEOUT_SECONDS: Final[int] = 10
SERVICE_REGISTRATION_TIMEOUT_SECONDS: Final[int] = 30
MAX_SERVICE_INSTANCES: Final[int] = 100


class MetricType(Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertState(Enum):
    """Alert states."""

    INACTIVE = "inactive"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    SILENCED = "silenced"


class HealthStatus(Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class MonitoringBackend(Enum):
    """Supported monitoring backends."""

    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    NEW_RELIC = "new_relic"
    ELASTIC_APM = "elastic_apm"
    JAEGER = "jaeger"
    ZIPKIN = "zipkin"


# Storage configuration
METRICS_STORAGE_PATH: Final[str] = "data/metrics"
LOGS_STORAGE_PATH: Final[str] = "data/logs"
TRACES_STORAGE_PATH: Final[str] = "data/traces"
BACKUP_STORAGE_PATH: Final[str] = "data/monitoring_backups"

# API configuration
MONITORING_API_PORT: Final[int] = 9090
METRICS_ENDPOINT: Final[str] = "/metrics"
HEALTH_ENDPOINT: Final[str] = "/health"
ALERTS_ENDPOINT: Final[str] = "/alerts"
DASHBOARD_ENDPOINT: Final[str] = "/dashboard"

# Security monitoring
SECURITY_EVENT_RETENTION_DAYS: Final[int] = 90
FAILED_LOGIN_THRESHOLD: Final[int] = 5
SUSPICIOUS_ACTIVITY_THRESHOLD: Final[int] = 10
SECURITY_ALERT_PRIORITY: Final[str] = "high"
AUDIT_LOG_BACKUP_INTERVAL_HOURS: Final[int] = 24

# Performance benchmarks
BENCHMARK_CPU_THRESHOLD_PERCENT: Final[float] = 50.0
BENCHMARK_MEMORY_THRESHOLD_MB: Final[int] = 1024
BENCHMARK_DISK_IO_THRESHOLD_IOPS: Final[int] = 1000
BENCHMARK_NETWORK_THRESHOLD_MBPS: Final[float] = 100.0

# Agent monitoring
AGENT_HEARTBEAT_TIMEOUT_SECONDS: Final[int] = 60
AGENT_PERFORMANCE_SAMPLE_RATE: Final[float] = 0.1
AGENT_ERROR_THRESHOLD_PER_HOUR: Final[int] = 10
AGENT_MEMORY_LEAK_THRESHOLD_MB: Final[int] = 100

# Cost monitoring
COST_TRACKING_INTERVAL_HOURS: Final[int] = 1
COST_ALERT_THRESHOLD_USD: Final[float] = 100.0
COST_FORECAST_HORIZON_DAYS: Final[int] = 30
RESOURCE_EFFICIENCY_TARGET_PERCENT: Final[int] = 80


class MonitoringMessages:
    """Standardized monitoring system messages."""

    METRIC_COLLECTED: Final[str] = "Metric '{metric_name}' collected: {value}"
    ALERT_TRIGGERED: Final[str] = "Alert '{alert_name}' triggered: {message}"
    ALERT_RESOLVED: Final[str] = "Alert '{alert_name}' resolved after {duration}"
    HEALTH_CHECK_FAILED: Final[str] = "Health check failed for service '{service}': {error}"
    THRESHOLD_EXCEEDED: Final[str] = "Threshold exceeded for '{metric}': {value} > {threshold}"
    SERVICE_DISCOVERED: Final[str] = "Service '{service}' discovered at {endpoint}"
    DASHBOARD_UPDATED: Final[str] = "Dashboard '{dashboard}' updated with {panels} panels"
    LOG_ANOMALY_DETECTED: Final[str] = "Log anomaly detected: {pattern} occurred {count} times"
    TRACE_ANALYSIS_COMPLETE: Final[str] = "Trace analysis complete: {spans} spans processed"
    MONITORING_STARTED: Final[str] = "Monitoring started for {targets} targets"


# Data aggregation
AGGREGATION_WINDOW_SECONDS: Final[int] = 300  # 5 minutes
AGGREGATION_FUNCTIONS: Final[tuple[str, ...]] = ("avg", "min", "max", "sum", "count")
DOWNSAMPLING_RESOLUTION_SECONDS: Final[int] = 3600  # 1 hour
RETENTION_POLICY_DAYS: Final[int] = 365

# Notification configuration
EMAIL_NOTIFICATION_TIMEOUT_SECONDS: Final[int] = 30
SLACK_NOTIFICATION_TIMEOUT_SECONDS: Final[int] = 10
WEBHOOK_NOTIFICATION_TIMEOUT_SECONDS: Final[int] = 15
SMS_NOTIFICATION_TIMEOUT_SECONDS: Final[int] = 60
MAX_NOTIFICATION_RETRIES: Final[int] = 3

# Compliance and auditing
COMPLIANCE_CHECK_INTERVAL_HOURS: Final[int] = 24
AUDIT_LOG_ENCRYPTION_ENABLED: Final[bool] = True
GDPR_DATA_RETENTION_DAYS: Final[int] = 1095  # 3 years
SOX_AUDIT_TRAIL_RETENTION_YEARS: Final[int] = 7
HIPAA_LOG_RETENTION_YEARS: Final[int] = 6

# Scalability configuration
MAX_CONCURRENT_MONITORS: Final[int] = 1000
MONITOR_POOL_SIZE: Final[int] = 50
QUEUE_SIZE_LIMIT: Final[int] = 10000
BATCH_PROCESSING_SIZE: Final[int] = 100
PARALLEL_PROCESSING_WORKERS: Final[int] = 10

# Integration settings
PROMETHEUS_SCRAPE_INTERVAL_SECONDS: Final[int] = 15
GRAFANA_DATASOURCE_TIMEOUT_SECONDS: Final[int] = 60
ELASTICSEARCH_BULK_SIZE: Final[int] = 1000
KAFKA_BATCH_SIZE: Final[int] = 16384
REDIS_CACHE_TTL_SECONDS: Final[int] = 3600

# AI/ML monitoring
MODEL_PERFORMANCE_THRESHOLD: Final[float] = 0.8
MODEL_DRIFT_DETECTION_WINDOW_HOURS: Final[int] = 24
PREDICTION_LATENCY_WARNING_MS: Final[int] = 500
PREDICTION_ACCURACY_WARNING_THRESHOLD: Final[float] = 0.7
DATA_QUALITY_CHECK_INTERVAL_HOURS: Final[int] = 6
