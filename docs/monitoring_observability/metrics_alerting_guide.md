# Metrics and Alerting Guide

## Introduction

This guide provides comprehensive coverage of the AIVillage metrics collection and alerting system. The platform implements a sophisticated observability framework that captures, processes, and analyzes metrics from across the entire system, enabling proactive monitoring and intelligent alerting.

## Metrics Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Metrics & Alerting Platform                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Metrics         │    │ Processing &    │    │ Alerting &      │  │
│  │ Collection      │    │ Aggregation     │    │ Notification    │  │
│  │                 │    │                 │    │                 │  │
│  │ • Counters      │    │ • Time-series   │    │ • Thresholds    │  │
│  │ • Gauges        │    │ • Aggregation   │    │ • Multi-channel │  │
│  │ • Histograms    │    │ • Correlation   │    │ • Escalation    │  │
│  │ • Timers        │    │ • Analytics     │    │ • Automation    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
│           │                       │                       │          │
│           └───────────────────────┼───────────────────────┘          │
│                                   │                                  │
│  ┌─────────────────────────────────┼─────────────────────────────────┐  │
│  │                    Data Storage & Visualization                  │  │
│  │                                                                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │
│  │  │ SQLite      │  │ Time-series │  │ Dashboards  │  │ Reports   │  │
│  │  │ Storage     │  │ Analysis    │  │             │  │           │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Pipeline

```
Application → Metrics Collection → Processing → Storage → Analysis → Alerting
     ↓              ↓                 ↓         ↓          ↓         ↓
  Instrumentation  Buffering      Aggregation SQLite   Thresholds Multi-channel
  Code Points     Auto-flush      Correlation  Time    Evaluation Notification
                                             Series
```

## Metrics Collection Framework

### MetricsCollector Implementation

**Location**: `packages/monitoring/observability_system.py:116-245`

```python
class MetricsCollector:
    """High-performance metrics collection with SQLite persistence."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.buffer = []
        self.buffer_size = 1000
        self.db_connection = None
        self._initialize_storage()

    def record_counter(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record counter metric with automatic increment."""
        metric = {
            'timestamp': time.time(),
            'name': name,
            'type': 'counter',
            'value': value,
            'labels': labels or {}
        }
        self._add_to_buffer(metric)

    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric (point-in-time value)."""
        metric = {
            'timestamp': time.time(),
            'name': name,
            'type': 'gauge',
            'value': value,
            'labels': labels or {}
        }
        self._add_to_buffer(metric)

    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram metric for distribution analysis."""
        metric = {
            'timestamp': time.time(),
            'name': name,
            'type': 'histogram',
            'value': value,
            'labels': labels or {}
        }
        self._add_to_buffer(metric)

    async def flush(self):
        """Flush buffered metrics to persistent storage."""
        if not self.buffer:
            return

        try:
            async with aiosqlite.connect(self.storage_path) as db:
                for metric in self.buffer:
                    await db.execute(
                        "INSERT INTO metrics (timestamp, name, type, value, labels, service_name) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            metric['timestamp'],
                            metric['name'],
                            metric['type'],
                            metric['value'],
                            json.dumps(metric['labels']),
                            self.service_name
                        )
                    )
                await db.commit()

            self.buffer.clear()
            logger.debug(f"Flushed {len(self.buffer)} metrics to storage")

        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
```

### Metric Types and Usage

#### 1. Counter Metrics

Counters represent cumulative values that only increase over time.

```python
# Request counting
observability.metrics.record_counter(
    "http_requests_total",
    1.0,
    {
        "method": "GET",
        "endpoint": "/api/users",
        "status": "200"
    }
)

# Error tracking
observability.metrics.record_counter(
    "errors_total",
    1.0,
    {
        "error_type": "validation_error",
        "component": "user_service",
        "severity": "medium"
    }
)

# Agent Forge phase completion
observability.metrics.record_counter(
    "agent_forge_phases_completed",
    1.0,
    {
        "phase": "evomerge",
        "model_size": "1.5B",
        "success": "true"
    }
)
```

**Common Counter Patterns**:
- `*_total` suffix for cumulative counts
- Include rate-of-change context in labels
- Use for events, requests, errors, completions

#### 2. Gauge Metrics

Gauges represent point-in-time values that can increase or decrease.

```python
# System resource monitoring
observability.metrics.record_gauge(
    "cpu_usage_percent",
    cpu_percent,
    {"host": "worker-01", "core": "all"}
)

observability.metrics.record_gauge(
    "memory_usage_bytes",
    memory_used,
    {"host": "worker-01", "type": "physical"}
)

# Active connection tracking
observability.metrics.record_gauge(
    "active_connections",
    connection_count,
    {"service": "mcp_server", "protocol": "websocket"}
)

# Queue depth monitoring
observability.metrics.record_gauge(
    "queue_depth",
    queue_size,
    {
        "queue_name": "security_events",
        "priority": "high"
    }
)

# Model performance metrics
observability.metrics.record_gauge(
    "model_accuracy",
    accuracy_score,
    {
        "model": "compression_model",
        "dataset": "validation",
        "version": "v2.1"
    }
)
```

**Common Gauge Patterns**:
- Current resource utilization
- Active item counts (connections, sessions)
- Performance scores and ratios
- Configuration values

#### 3. Histogram Metrics

Histograms capture distributions of values for statistical analysis.

```python
# Response time tracking
observability.metrics.record_histogram(
    "response_time_ms",
    request_duration_ms,
    {
        "endpoint": "/api/agent_forge/run",
        "method": "POST",
        "user_type": "authenticated"
    }
)

# Model inference latency
observability.metrics.record_histogram(
    "inference_latency_ms",
    inference_time,
    {
        "model": "compression_model",
        "input_size": "large",
        "device": "cpu"
    }
)

# Data processing times
observability.metrics.record_histogram(
    "data_processing_duration_seconds",
    processing_duration,
    {
        "operation": "rag_query",
        "data_size": "medium",
        "complexity": "high"
    }
)

# Agent Forge phase durations
observability.metrics.record_histogram(
    "phase_duration_seconds",
    phase_duration,
    {
        "phase": "prompt_baking",
        "model_size": "1.5B",
        "iterations": "50"
    }
)
```

**Histogram Analysis**:
- Automatic percentile calculation (p50, p90, p95, p99)
- Distribution analysis for performance optimization
- SLA/SLO compliance monitoring

### Advanced Metric Instrumentation

#### Decorator-Based Instrumentation

```python
from functools import wraps
import time

def timed_operation(observability: ObservabilitySystem, operation_name: str):
    """Decorator for automatic timing instrumentation."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_type = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000

                # Record timing histogram
                observability.metrics.record_histogram(
                    f"{operation_name}_duration_ms",
                    duration_ms,
                    {
                        "success": str(success),
                        "error_type": error_type or "none"
                    }
                )

                # Record operation counter
                observability.metrics.record_counter(
                    f"{operation_name}_operations_total",
                    1.0,
                    {
                        "success": str(success),
                        "error_type": error_type or "none"
                    }
                )

        return wrapper
    return decorator

# Usage example
@timed_operation(observability, "model_compression")
async def compress_model(model_path: str, compression_ratio: float):
    """Compress model with automatic timing metrics."""
    compressed_model = await run_compression(model_path, compression_ratio)
    return compressed_model
```

#### Context Manager Instrumentation

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def monitored_operation(observability: ObservabilitySystem, operation_name: str, **labels):
    """Context manager for detailed operation monitoring."""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss

    # Record operation start
    observability.metrics.record_counter(
        f"{operation_name}_started_total",
        1.0,
        labels
    )

    try:
        yield

        # Record successful completion
        observability.metrics.record_counter(
            f"{operation_name}_completed_total",
            1.0,
            {**labels, "status": "success"}
        )

    except Exception as e:
        # Record failure
        observability.metrics.record_counter(
            f"{operation_name}_failed_total",
            1.0,
            {**labels, "status": "failed", "error": type(e).__name__}
        )
        raise

    finally:
        # Record timing and resource usage
        duration_ms = (time.time() - start_time) * 1000
        end_memory = psutil.Process().memory_info().rss
        memory_delta_mb = (end_memory - start_memory) / 1024 / 1024

        observability.metrics.record_histogram(
            f"{operation_name}_duration_ms",
            duration_ms,
            labels
        )

        observability.metrics.record_histogram(
            f"{operation_name}_memory_delta_mb",
            memory_delta_mb,
            labels
        )

# Usage example
async with monitored_operation(
    observability,
    "rag_query_processing",
    query_type="complex",
    user_type="premium"
):
    results = await process_rag_query(query)
```

### System Metrics Collection

#### Resource Monitoring

```python
class SystemMetricsCollector:
    """Collect system-level metrics for monitoring."""

    def __init__(self, observability: ObservabilitySystem):
        self.observability = observability
        self.collection_interval = 10  # seconds

    async def start_collection(self):
        """Start continuous system metrics collection."""
        while True:
            try:
                await self.collect_cpu_metrics()
                await self.collect_memory_metrics()
                await self.collect_disk_metrics()
                await self.collect_network_metrics()
                await self.collect_gpu_metrics()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"System metrics collection failed: {e}")
                await asyncio.sleep(self.collection_interval)

    async def collect_cpu_metrics(self):
        """Collect CPU utilization metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()

        self.observability.metrics.record_gauge(
            "cpu_usage_percent",
            cpu_percent,
            {"host": socket.gethostname()}
        )

        self.observability.metrics.record_gauge(
            "cpu_count",
            cpu_count,
            {"host": socket.gethostname()}
        )

        # Per-core utilization
        per_cpu = psutil.cpu_percent(percpu=True)
        for i, cpu_usage in enumerate(per_cpu):
            self.observability.metrics.record_gauge(
                "cpu_core_usage_percent",
                cpu_usage,
                {"host": socket.gethostname(), "core": str(i)}
            )

    async def collect_memory_metrics(self):
        """Collect memory utilization metrics."""
        memory = psutil.virtual_memory()

        metrics = {
            "memory_total_bytes": memory.total,
            "memory_used_bytes": memory.used,
            "memory_available_bytes": memory.available,
            "memory_usage_percent": memory.percent
        }

        for metric_name, value in metrics.items():
            self.observability.metrics.record_gauge(
                metric_name,
                value,
                {"host": socket.gethostname()}
            )

    async def collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()

                for i in range(device_count):
                    # Memory metrics
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory

                    labels = {
                        "host": socket.gethostname(),
                        "device": str(i),
                        "name": torch.cuda.get_device_name(i)
                    }

                    self.observability.metrics.record_gauge(
                        "gpu_memory_allocated_bytes",
                        memory_allocated,
                        labels
                    )

                    self.observability.metrics.record_gauge(
                        "gpu_memory_reserved_bytes",
                        memory_reserved,
                        labels
                    )

                    self.observability.metrics.record_gauge(
                        "gpu_memory_total_bytes",
                        memory_total,
                        labels
                    )

                    # Utilization percentage
                    memory_percent = (memory_allocated / memory_total) * 100
                    self.observability.metrics.record_gauge(
                        "gpu_memory_usage_percent",
                        memory_percent,
                        labels
                    )

        except ImportError:
            logger.debug("PyTorch not available for GPU metrics")
```

## Alert Management System

### AlertManager Architecture

**Location**: `packages/monitoring/alert_manager.py:72-487`

```python
class AlertManager:
    """Intelligent alert management with multi-channel dispatch."""

    def __init__(self, config_path: Path):
        self.config = AlertConfig.from_yaml(config_path)
        self.active_alerts: List[Alert] = []
        self.channels = []  # Multi-channel configuration

        # Alert correlation and deduplication
        self.alert_correlation = AlertCorrelationEngine()
        self.escalation_manager = EscalationManager()

    async def evaluate_metrics(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Evaluate metrics against configured thresholds."""
        triggered_alerts = []

        # Check all configured alert rules
        for rule in self.config.alert_rules:
            try:
                if await self._evaluate_rule(rule, metrics):
                    alert = await self._create_alert(rule, metrics)
                    triggered_alerts.append(alert)

            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule.name}: {e}")

        return triggered_alerts

    async def _evaluate_rule(self, rule: AlertRule, metrics: Dict[str, Any]) -> bool:
        """Evaluate individual alert rule against metrics."""
        metric_value = self._extract_metric_value(metrics, rule.metric_path)

        if metric_value is None:
            return False

        # Apply threshold comparison
        if rule.condition == "greater_than":
            return metric_value > rule.threshold
        elif rule.condition == "less_than":
            return metric_value < rule.threshold
        elif rule.condition == "equals":
            return metric_value == rule.threshold
        elif rule.condition == "not_equals":
            return metric_value != rule.threshold

        return False
```

### Alert Configuration Framework

#### AlertConfig Structure

```python
@dataclass
class AlertRule:
    """Configuration for individual alert rule."""
    name: str
    description: str
    metric_path: str          # Dot notation path to metric
    condition: str            # greater_than, less_than, equals, not_equals
    threshold: float
    severity: str             # critical, high, medium, low
    evaluation_window: int    # seconds
    cooldown_period: int      # seconds between same alerts
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class AlertConfig:
    """Complete alerting configuration."""
    alert_rules: List[AlertRule]
    notification_channels: List[Dict[str, Any]]
    escalation_policies: List[Dict[str, Any]]
    global_settings: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AlertConfig":
        """Load alert configuration from YAML file."""
        if not config_path.exists():
            return cls._create_default_config()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        alert_rules = [
            AlertRule(**rule_data)
            for rule_data in data.get("alert_rules", [])
        ]

        return cls(
            alert_rules=alert_rules,
            notification_channels=data.get("channels", []),
            escalation_policies=data.get("escalation", []),
            global_settings=data.get("settings", {})
        )
```

#### Alert Configuration Example

```yaml
# alert_config.yaml
alert_rules:
  # System Performance Alerts
  - name: "high_cpu_usage"
    description: "CPU usage above 80%"
    metric_path: "cpu_usage_percent"
    condition: "greater_than"
    threshold: 80.0
    severity: "medium"
    evaluation_window: 300  # 5 minutes
    cooldown_period: 600    # 10 minutes
    labels:
      component: "system"
      category: "performance"

  - name: "critical_memory_usage"
    description: "Memory usage above 90%"
    metric_path: "memory_usage_percent"
    condition: "greater_than"
    threshold: 90.0
    severity: "critical"
    evaluation_window: 60   # 1 minute
    cooldown_period: 300    # 5 minutes

  # Application Performance Alerts
  - name: "slow_response_times"
    description: "95th percentile response time above 2000ms"
    metric_path: "response_time_ms.p95"
    condition: "greater_than"
    threshold: 2000.0
    severity: "high"
    evaluation_window: 600  # 10 minutes
    cooldown_period: 1800   # 30 minutes

  - name: "high_error_rate"
    description: "Error rate above 5%"
    metric_path: "errors_total.rate"
    condition: "greater_than"
    threshold: 0.05
    severity: "high"
    evaluation_window: 300  # 5 minutes
    cooldown_period: 900    # 15 minutes

  # Agent Forge Specific Alerts
  - name: "agent_forge_phase_failure"
    description: "Agent Forge phase failure rate above 10%"
    metric_path: "agent_forge_phases_failed.rate"
    condition: "greater_than"
    threshold: 0.10
    severity: "critical"
    evaluation_window: 1800 # 30 minutes
    cooldown_period: 3600   # 1 hour
    labels:
      service: "agent_forge"

  - name: "model_compression_slow"
    description: "Model compression taking longer than 30 minutes"
    metric_path: "model_compression_duration_ms.p90"
    condition: "greater_than"
    threshold: 1800000  # 30 minutes in ms
    severity: "medium"
    evaluation_window: 3600 # 1 hour
    cooldown_period: 7200   # 2 hours

# Notification Channels
channels:
  - type: "webhook"
    name: "slack_alerts"
    url: "${SLACK_WEBHOOK_URL}"
    timeout: 10
    filters:
      - severity: ["critical", "high"]

  - type: "email"
    name: "engineering_team"
    smtp_server: "smtp.company.com"
    smtp_port: 587
    username: "alerts@company.com"
    to_emails: ["engineering@company.com", "devops@company.com"]
    filters:
      - severity: ["critical"]
      - component: ["agent_forge"]

  - type: "github"
    name: "issue_tracker"
    repo: "company/aivillage"
    labels: ["alert", "automated"]
    filters:
      - severity: ["critical", "high"]
      - category: ["performance", "reliability"]

  - type: "sentry"
    name: "error_tracking"
    project: "aivillage-monitoring"
    environment: "production"

# Escalation Policies
escalation:
  - name: "critical_escalation"
    triggers:
      - severity: "critical"
    steps:
      - delay: 0      # Immediate
        channels: ["slack_alerts", "email"]
      - delay: 900    # 15 minutes
        channels: ["email", "github"]
      - delay: 3600   # 1 hour
        channels: ["email"]
        recipients: ["on-call@company.com"]

  - name: "performance_escalation"
    triggers:
      - category: "performance"
      - severity: ["high", "medium"]
    steps:
      - delay: 300    # 5 minutes
        channels: ["slack_alerts"]
      - delay: 1800   # 30 minutes
        channels: ["email", "github"]

# Global Settings
settings:
  evaluation_interval: 60    # seconds
  batch_size: 100           # alerts per batch
  max_alerts_per_minute: 50 # rate limiting
  alert_retention_days: 30  # alert history retention
  enable_correlation: true  # alert correlation
  enable_deduplication: true # duplicate suppression
```

### Advanced Alerting Features

#### 1. Alert Correlation and Deduplication

```python
class AlertCorrelationEngine:
    """Intelligent alert correlation and deduplication."""

    def __init__(self):
        self.correlation_rules = []
        self.active_correlations = {}

    def add_correlation_rule(self, rule: CorrelationRule):
        """Add alert correlation rule."""
        self.correlation_rules.append(rule)

    async def correlate_alert(self, alert: Alert) -> Optional[str]:
        """Correlate alert with existing incidents."""

        # Check for exact duplicates
        if self._is_duplicate(alert):
            logger.debug(f"Suppressing duplicate alert: {alert.name}")
            return "duplicate"

        # Check for related alerts
        correlation_id = self._find_correlation(alert)
        if correlation_id:
            logger.info(f"Correlated alert {alert.name} with incident {correlation_id}")
            return correlation_id

        # Create new correlation group
        correlation_id = self._create_correlation_group(alert)
        logger.info(f"Created new correlation group {correlation_id} for {alert.name}")
        return correlation_id

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate of recent alert."""
        recent_window = 3600  # 1 hour
        current_time = time.time()

        for existing_alert in self.recent_alerts:
            if (
                existing_alert.name == alert.name and
                existing_alert.labels == alert.labels and
                (current_time - existing_alert.timestamp) < recent_window
            ):
                return True
        return False

    def _find_correlation(self, alert: Alert) -> Optional[str]:
        """Find existing correlation group for alert."""
        for rule in self.correlation_rules:
            if self._matches_correlation_rule(alert, rule):
                return self._get_active_correlation(rule.group_key)
        return None

@dataclass
class CorrelationRule:
    """Rule for correlating related alerts."""
    name: str
    group_key: str              # Key for grouping related alerts
    time_window: int            # Correlation time window in seconds
    alert_patterns: List[str]   # Alert name patterns to correlate
    label_matchers: Dict[str, str]  # Label-based correlation
```

#### 2. Intelligent Escalation

```python
class EscalationManager:
    """Manage alert escalation policies."""

    def __init__(self, config: AlertConfig):
        self.escalation_policies = config.escalation_policies
        self.active_escalations = {}

    async def start_escalation(self, alert: Alert) -> str:
        """Start escalation process for alert."""
        policy = self._find_matching_policy(alert)
        if not policy:
            logger.debug(f"No escalation policy for alert {alert.name}")
            return None

        escalation_id = f"esc_{alert.id}_{int(time.time())}"

        escalation = {
            "id": escalation_id,
            "alert": alert,
            "policy": policy,
            "current_step": 0,
            "start_time": time.time(),
            "next_escalation": time.time() + policy["steps"][0]["delay"]
        }

        self.active_escalations[escalation_id] = escalation

        # Schedule first notification
        await self._execute_escalation_step(escalation)

        return escalation_id

    async def _execute_escalation_step(self, escalation: Dict[str, Any]):
        """Execute current escalation step."""
        step = escalation["policy"]["steps"][escalation["current_step"]]

        # Send notifications through configured channels
        for channel_name in step["channels"]:
            try:
                await self._send_escalation_notification(
                    escalation["alert"],
                    channel_name,
                    escalation["current_step"]
                )
            except Exception as e:
                logger.error(f"Failed to send escalation notification: {e}")

        # Schedule next escalation step
        if escalation["current_step"] + 1 < len(escalation["policy"]["steps"]):
            escalation["current_step"] += 1
            next_step = escalation["policy"]["steps"][escalation["current_step"]]
            escalation["next_escalation"] = time.time() + next_step["delay"]
        else:
            # Escalation complete
            logger.info(f"Escalation complete for {escalation['alert'].name}")
            del self.active_escalations[escalation["id"]]
```

#### 3. Dynamic Threshold Adjustment

```python
class DynamicThresholdManager:
    """Automatically adjust alert thresholds based on historical data."""

    def __init__(self, observability: ObservabilitySystem):
        self.observability = observability
        self.baseline_window = 7 * 24 * 3600  # 7 days
        self.adjustment_factor = 1.2  # 20% above baseline

    async def calculate_dynamic_threshold(
        self, metric_name: str, percentile: float = 95.0
    ) -> float:
        """Calculate dynamic threshold based on historical data."""

        # Query historical data
        end_time = time.time()
        start_time = end_time - self.baseline_window

        historical_data = await self._query_historical_metrics(
            metric_name, start_time, end_time
        )

        if not historical_data:
            logger.warning(f"No historical data for {metric_name}")
            return None

        # Calculate percentile-based threshold
        values = [point["value"] for point in historical_data]
        threshold = np.percentile(values, percentile)

        # Apply adjustment factor
        dynamic_threshold = threshold * self.adjustment_factor

        logger.info(
            f"Dynamic threshold for {metric_name}: {dynamic_threshold:.2f} "
            f"(p{percentile}: {threshold:.2f}, factor: {self.adjustment_factor})"
        )

        return dynamic_threshold

    async def update_alert_thresholds(self):
        """Update alert thresholds based on recent data patterns."""

        for alert_rule in self.alert_manager.config.alert_rules:
            if alert_rule.labels.get("dynamic_threshold") == "true":
                try:
                    new_threshold = await self.calculate_dynamic_threshold(
                        alert_rule.metric_path
                    )

                    if new_threshold and abs(new_threshold - alert_rule.threshold) > 0.1:
                        old_threshold = alert_rule.threshold
                        alert_rule.threshold = new_threshold

                        logger.info(
                            f"Updated threshold for {alert_rule.name}: "
                            f"{old_threshold:.2f} -> {new_threshold:.2f}"
                        )

                except Exception as e:
                    logger.error(f"Failed to update threshold for {alert_rule.name}: {e}")
```

### Notification Channels

#### 1. Webhook Notifications

```python
class WebhookNotificationChannel:
    """Send alerts via webhook with retry logic."""

    def __init__(self, config: Dict[str, Any]):
        self.url = config["url"]
        self.timeout = config.get("timeout", 10)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.retry_delay = config.get("retry_delay", 5)

    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook with exponential backoff retry."""

        payload = {
            "timestamp": alert.timestamp,
            "alert_name": alert.name,
            "severity": alert.severity,
            "description": alert.description,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "labels": alert.labels,
            "escalation_step": getattr(alert, "escalation_step", 0)
        }

        for attempt in range(self.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook alert sent successfully: {alert.name}")
                            return True
                        else:
                            logger.warning(
                                f"Webhook returned status {response.status}: {alert.name}"
                            )

            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))

        logger.error(f"All webhook attempts failed for alert: {alert.name}")
        return False
```

#### 2. GitHub Issue Creation

```python
class GitHubNotificationChannel:
    """Create GitHub issues for alerts."""

    def __init__(self, config: Dict[str, Any]):
        self.repo = config["repo"]
        self.labels = config.get("labels", [])
        self.token = os.getenv("GITHUB_TOKEN")

    async def send_alert(self, alert: Alert) -> bool:
        """Create GitHub issue for alert."""

        if not self.token:
            logger.error("GitHub token not configured")
            return False

        # Create issue title and body
        title = f"Alert: {alert.name} - {alert.severity.upper()}"

        body = f"""# Alert Details

**Alert Name**: {alert.name}
**Severity**: {alert.severity.upper()}
**Timestamp**: {alert.timestamp}
**Metric Value**: {alert.metric_value}
**Threshold**: {alert.threshold}

## Description
{alert.description}

## Labels
{json.dumps(alert.labels, indent=2)}

## Recommended Actions
{self._get_recommended_actions(alert)}

---
*This issue was automatically created by the AIVillage monitoring system.*
"""

        payload = {
            "title": title,
            "body": body,
            "labels": self.labels + [f"severity-{alert.severity}"]
        }

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json"
                }

                async with session.post(
                    f"https://api.github.com/repos/{self.repo}/issues",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 201:
                        issue_data = await response.json()
                        logger.info(f"Created GitHub issue: {issue_data['html_url']}")
                        return True
                    else:
                        logger.error(f"GitHub API error: {response.status}")
                        return False

        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            return False

    def _get_recommended_actions(self, alert: Alert) -> str:
        """Get recommended actions based on alert type."""
        action_map = {
            "high_cpu_usage": "Check for runaway processes and optimize resource usage",
            "critical_memory_usage": "Investigate memory leaks and consider scaling",
            "slow_response_times": "Analyze performance bottlenecks and optimize queries",
            "high_error_rate": "Check application logs and fix underlying issues",
            "agent_forge_phase_failure": "Review Agent Forge logs and model configurations"
        }

        return action_map.get(
            alert.name,
            "Investigate the underlying cause and take appropriate remediation steps"
        )
```

### Performance Monitoring Patterns

#### 1. SLI/SLO Monitoring

```python
class SLIMonitor:
    """Service Level Indicator monitoring."""

    def __init__(self, observability: ObservabilitySystem):
        self.observability = observability

    async def track_availability_sli(
        self, service_name: str, success: bool, duration_ms: float
    ):
        """Track availability SLI (successful requests / total requests)."""

        # Record request outcome
        self.observability.metrics.record_counter(
            "sli_requests_total",
            1.0,
            {
                "service": service_name,
                "success": str(success)
            }
        )

        # Record response time
        self.observability.metrics.record_histogram(
            "sli_request_duration_ms",
            duration_ms,
            {
                "service": service_name,
                "success": str(success)
            }
        )

    async def track_latency_sli(
        self, service_name: str, duration_ms: float, threshold_ms: float = 2000
    ):
        """Track latency SLI (fast requests / total requests)."""

        fast_request = duration_ms < threshold_ms

        self.observability.metrics.record_counter(
            "sli_latency_requests_total",
            1.0,
            {
                "service": service_name,
                "fast": str(fast_request)
            }
        )

    async def calculate_slo_compliance(
        self, service_name: str, time_window: int = 3600
    ) -> Dict[str, float]:
        """Calculate SLO compliance for service."""

        end_time = time.time()
        start_time = end_time - time_window

        # Query metrics for time window
        availability_metrics = await self._query_sli_metrics(
            "sli_requests_total", service_name, start_time, end_time
        )

        latency_metrics = await self._query_sli_metrics(
            "sli_latency_requests_total", service_name, start_time, end_time
        )

        # Calculate availability SLO
        total_requests = sum(m["value"] for m in availability_metrics)
        successful_requests = sum(
            m["value"] for m in availability_metrics
            if m["labels"].get("success") == "true"
        )

        availability_slo = (
            (successful_requests / total_requests * 100)
            if total_requests > 0 else 100.0
        )

        # Calculate latency SLO
        total_latency_requests = sum(m["value"] for m in latency_metrics)
        fast_requests = sum(
            m["value"] for m in latency_metrics
            if m["labels"].get("fast") == "true"
        )

        latency_slo = (
            (fast_requests / total_latency_requests * 100)
            if total_latency_requests > 0 else 100.0
        )

        return {
            "availability_slo": availability_slo,
            "latency_slo": latency_slo,
            "total_requests": total_requests,
            "time_window": time_window
        }

# Usage example for Agent Forge monitoring
@timed_operation(observability, "agent_forge_phase")
async def run_agent_forge_phase(phase_name: str):
    """Run Agent Forge phase with SLI tracking."""
    start_time = time.time()
    success = False

    try:
        result = await execute_phase(phase_name)
        success = True
        return result

    except Exception as e:
        logger.error(f"Phase {phase_name} failed: {e}")
        raise

    finally:
        duration_ms = (time.time() - start_time) * 1000

        # Track availability SLI
        await sli_monitor.track_availability_sli(
            "agent_forge", success, duration_ms
        )

        # Track latency SLI (phases should complete within 30 minutes)
        await sli_monitor.track_latency_sli(
            "agent_forge", duration_ms, threshold_ms=30*60*1000
        )
```

#### 2. Error Budget Monitoring

```python
class ErrorBudgetMonitor:
    """Monitor error budget consumption for services."""

    def __init__(self, observability: ObservabilitySystem):
        self.observability = observability
        self.error_budgets = {
            "agent_forge": {"slo": 99.5, "window": 30*24*3600},  # 99.5% over 30 days
            "rag_system": {"slo": 99.9, "window": 7*24*3600},    # 99.9% over 7 days
            "mcp_server": {"slo": 99.95, "window": 24*3600},     # 99.95% over 1 day
        }

    async def track_error_budget_consumption(self):
        """Track error budget consumption for all services."""

        for service_name, budget_config in self.error_budgets.items():
            try:
                consumption = await self._calculate_error_budget_consumption(
                    service_name, budget_config
                )

                self.observability.metrics.record_gauge(
                    "error_budget_consumption_percent",
                    consumption["consumption_percent"],
                    {
                        "service": service_name,
                        "slo": str(budget_config["slo"]),
                        "window": str(budget_config["window"])
                    }
                )

                # Alert if error budget consumption is high
                if consumption["consumption_percent"] > 80:
                    await self._send_error_budget_alert(service_name, consumption)

            except Exception as e:
                logger.error(f"Failed to calculate error budget for {service_name}: {e}")

    async def _calculate_error_budget_consumption(
        self, service_name: str, budget_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate error budget consumption for service."""

        slo_compliance = await sli_monitor.calculate_slo_compliance(
            service_name, budget_config["window"]
        )

        current_availability = slo_compliance["availability_slo"]
        target_slo = budget_config["slo"]

        # Calculate error budget consumption
        if current_availability >= target_slo:
            consumption_percent = 0.0
        else:
            error_rate = 100 - current_availability
            max_error_rate = 100 - target_slo
            consumption_percent = (error_rate / max_error_rate) * 100

        return {
            "current_availability": current_availability,
            "target_slo": target_slo,
            "consumption_percent": min(consumption_percent, 100.0),
            "time_window": budget_config["window"]
        }
```

### Best Practices and Optimization

#### 1. Metric Naming Conventions

```python
# Good metric naming patterns
METRIC_NAMING_PATTERNS = {
    "counters": [
        "{component}_{action}_total",           # "http_requests_total"
        "{component}_{action}_{outcome}_total", # "db_queries_failed_total"
    ],
    "gauges": [
        "{resource}_usage_{unit}",              # "memory_usage_bytes"
        "{component}_active_{items}",           # "connections_active_count"
    ],
    "histograms": [
        "{action}_duration_{unit}",             # "request_duration_ms"
        "{component}_{metric}_size_{unit}",     # "response_body_size_bytes"
    ]
}

# Label best practices
LABEL_GUIDELINES = {
    "use_consistent_names": ["method", "status", "endpoint"],
    "avoid_high_cardinality": ["user_id", "request_id", "timestamp"],
    "include_service_context": ["service", "version", "environment"],
    "group_related_metrics": ["component", "subsystem", "operation"]
}
```

#### 2. Performance Optimization

```python
class MetricsOptimizer:
    """Optimize metrics collection performance."""

    def __init__(self):
        self.batch_size = 1000
        self.flush_interval = 30  # seconds
        self.compression_enabled = True

    async def optimize_metric_storage(self):
        """Optimize SQLite storage for metrics."""

        # Create efficient indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_name_time ON metrics(name, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_labels ON metrics(labels)",
        ]

        async with aiosqlite.connect(self.storage_path) as db:
            for index_sql in indices:
                await db.execute(index_sql)
            await db.commit()

    async def implement_metric_sampling(self, sample_rate: float = 0.1):
        """Implement sampling for high-volume metrics."""

        def should_sample() -> bool:
            return random.random() < sample_rate

        # Apply sampling to high-cardinality metrics
        if self.metric_name in HIGH_CARDINALITY_METRICS:
            if not should_sample():
                return  # Skip this metric

        # Record metric normally
        await self._record_metric(metric_data)

    def setup_metric_aggregation(self):
        """Set up metric aggregation for efficiency."""

        # Pre-aggregate common queries
        aggregation_rules = [
            {
                "source": "http_requests_total",
                "target": "http_requests_rate_5m",
                "aggregation": "rate",
                "window": 300  # 5 minutes
            },
            {
                "source": "response_time_ms",
                "target": "response_time_p95_1h",
                "aggregation": "percentile",
                "percentile": 95,
                "window": 3600  # 1 hour
            }
        ]

        return aggregation_rules
```

This comprehensive metrics and alerting guide provides the foundation for robust observability and proactive monitoring within the AIVillage platform. The system combines high-performance data collection, intelligent alerting, and multi-channel notification capabilities to ensure optimal system performance and rapid incident response.
