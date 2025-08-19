# Integration Guide

## Introduction

This guide provides comprehensive instructions for integrating the AIVillage Monitoring & Observability system with external tools, services, and platforms. The system is designed with extensive integration capabilities to work seamlessly with existing monitoring infrastructure and development workflows.

## Integration Architecture Overview

### External System Integrations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AIVillage Observability Platform                  │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │
│  │ Data Collection │    │ Processing      │    │ Export &        │  │
│  │ & Ingestion     │    │ & Analysis      │    │ Integration     │  │
│  │                 │    │                 │    │                 │  │
│  │ • OpenTelemetry │    │ • Correlation   │    │ • Prometheus    │  │
│  │ • Custom APIs   │    │ • Aggregation   │    │ • Grafana       │  │
│  │ • Webhooks      │    │ • Alerting      │    │ • Sentry        │  │
│  │ • Log Streams   │    │ • ML Analysis   │    │ • GitHub        │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
┌─────────────────────────────────────────────────────────────────────┐
│                      External Integrations                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Monitoring  │  │ Alerting    │  │ Development │  │ Analytics   │  │
│  │ Systems     │  │ Platforms   │  │ Tools       │  │ Platforms   │  │
│  │             │  │             │  │             │  │             │  │
│  │ • Prometheus│  │ • Slack     │  │ • GitHub    │  │ • Weights & │  │
│  │ • Grafana   │  │ • PagerDuty │  │ • GitLab    │  │   Biases    │  │
│  │ • DataDog   │  │ • OpsGenie  │  │ • Jenkins   │  │ • TensorBoard│  │
│  │ • New Relic │  │ • Email     │  │ • Docker    │  │ • Jupyter   │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration Patterns

The AIVillage observability system supports several integration patterns:

1. **Push-Based Exports**: Send data to external systems via APIs/webhooks
2. **Pull-Based Scraping**: Expose metrics endpoints for external scraping
3. **Event-Driven Integrations**: Trigger actions based on events and alerts
4. **Bidirectional Sync**: Two-way data exchange with external platforms

## Prometheus Integration

### Metrics Export Configuration

**Location**: `packages/monitoring/security_monitor.py:61-93`

```python
class SecurityMetrics:
    """Security metrics with Prometheus integration."""

    def __init__(self):
        if PROMETHEUS_AVAILABLE:
            # Authentication metrics
            self.auth_failures = Counter(
                "auth_failures_total",
                "Total authentication failures",
                ["user_id", "source_ip", "reason"]
            )

            # Security event metrics
            self.security_events = Counter(
                "security_events_total",
                "Total security events",
                ["event_type", "severity"]
            )

            # Real-time threat scores
            self.threat_score = Gauge(
                "threat_score_current",
                "Current threat score for users",
                ["user_id"]
            )

            # Performance metrics
            self.detection_latency = Histogram(
                "security_detection_duration_seconds",
                "Time taken for threat detection",
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            )

            # Start Prometheus metrics server
            start_http_server(8090)
            logger.info("Prometheus metrics server started on port 8090")
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "aivillage_rules.yml"

scrape_configs:
  # AIVillage monitoring system
  - job_name: 'aivillage-observability'
    static_configs:
      - targets: ['localhost:8089']  # ObservabilitySystem metrics
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'aivillage-security'
    static_configs:
      - targets: ['localhost:8090']  # SecurityMonitor metrics
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'aivillage-dashboard'
    static_configs:
      - targets: ['localhost:8091']  # Dashboard metrics
    scrape_interval: 30s
    metrics_path: /metrics

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Custom Prometheus rules for AIVillage
# aivillage_rules.yml
groups:
  - name: aivillage.security
    rules:
      - alert: HighThreatScore
        expr: threat_score_current > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High threat score detected for user {{ $labels.user_id }}"
          description: "User {{ $labels.user_id }} has a threat score of {{ $value }}"

      - alert: AuthenticationFailureSpike
        expr: rate(auth_failures_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Authentication failure spike detected"
          description: "Auth failure rate is {{ $value }} failures/second"

  - name: aivillage.performance
    rules:
      - alert: SlowSecurityDetection
        expr: histogram_quantile(0.95, security_detection_duration_seconds_bucket) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Security detection is slow"
          description: "95th percentile detection time is {{ $value }}s"
```

### Prometheus Integration Implementation

```python
class PrometheusExporter:
    """Enhanced Prometheus metrics exporter for AIVillage."""

    def __init__(self, observability: ObservabilitySystem):
        self.observability = observability
        self.prometheus_metrics = {}
        self.metrics_server_port = 8089

    def start_metrics_server(self):
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.metrics_server_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_server_port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def register_custom_metrics(self):
        """Register custom AIVillage metrics with Prometheus."""

        # Agent Forge metrics
        self.prometheus_metrics.update({
            "agent_forge_phases_total": Counter(
                "agent_forge_phases_total",
                "Total Agent Forge phases executed",
                ["phase", "status", "model_size"]
            ),

            "agent_forge_phase_duration": Histogram(
                "agent_forge_phase_duration_seconds",
                "Agent Forge phase execution duration",
                ["phase", "model_size"],
                buckets=[60, 300, 600, 1800, 3600, 7200, 14400]  # 1m to 4h
            ),

            "model_compression_ratio": Gauge(
                "model_compression_ratio",
                "Model compression ratio achieved",
                ["model", "compression_type"]
            ),

            # RAG system metrics
            "rag_queries_total": Counter(
                "rag_queries_total",
                "Total RAG queries processed",
                ["query_type", "success"]
            ),

            "rag_query_latency": Histogram(
                "rag_query_latency_seconds",
                "RAG query processing latency",
                ["query_type"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
            ),

            # P2P network metrics
            "p2p_messages_total": Counter(
                "p2p_messages_total",
                "Total P2P messages sent/received",
                ["direction", "protocol", "status"]
            ),

            "p2p_peers_connected": Gauge(
                "p2p_peers_connected",
                "Number of connected P2P peers",
                ["protocol"]
            )
        })

    async def export_metrics_batch(self, metrics_batch: List[Dict[str, Any]]):
        """Export batch of metrics to Prometheus."""

        for metric in metrics_batch:
            try:
                metric_name = metric["name"]
                metric_type = metric["type"]
                value = metric["value"]
                labels = metric.get("labels", {})

                if metric_name in self.prometheus_metrics:
                    prom_metric = self.prometheus_metrics[metric_name]

                    if metric_type == "counter":
                        prom_metric.labels(**labels).inc(value)
                    elif metric_type == "gauge":
                        prom_metric.labels(**labels).set(value)
                    elif metric_type == "histogram":
                        prom_metric.labels(**labels).observe(value)

            except Exception as e:
                logger.error(f"Failed to export metric {metric_name}: {e}")

# Usage example
async def setup_prometheus_integration():
    """Set up Prometheus integration for AIVillage."""

    exporter = PrometheusExporter(observability)
    exporter.register_custom_metrics()
    exporter.start_metrics_server()

    # Export metrics periodically
    while True:
        try:
            metrics_batch = await observability.get_recent_metrics(limit=1000)
            await exporter.export_metrics_batch(metrics_batch)
            await asyncio.sleep(15)  # Export every 15 seconds
        except Exception as e:
            logger.error(f"Prometheus export failed: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute
```

## Grafana Integration

### Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "AIVillage System Monitoring",
    "tags": ["aivillage", "monitoring"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Security Threat Score",
        "type": "stat",
        "targets": [
          {
            "expr": "max(threat_score_current)",
            "legendFormat": "Max Threat Score"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 0.5},
                {"color": "red", "value": 0.8}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Authentication Failures Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(auth_failures_total[5m])",
            "legendFormat": "Failures/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Agent Forge Phase Duration",
        "type": "heatmap",
        "targets": [
          {
            "expr": "agent_forge_phase_duration_seconds_bucket",
            "legendFormat": "{{phase}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "RAG Query Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rag_query_latency_seconds_bucket)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rag_query_latency_seconds_bucket)",
            "legendFormat": "50th percentile"
          }
        ]
      }
    ]
  }
}
```

### Grafana Dashboard Provisioning

```yaml
# grafana/provisioning/dashboards/aivillage.yml
apiVersion: 1

providers:
  - name: 'aivillage'
    orgId: 1
    folder: 'AIVillage'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards

# grafana/provisioning/datasources/prometheus.yml
apiVersion: 1

datasources:
  - name: AIVillage Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: AIVillage Logs
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true
```

## Sentry Integration

### Error Tracking Configuration

```python
class SentryIntegration:
    """Sentry integration for error tracking and performance monitoring."""

    def __init__(self, dsn: str, environment: str = "production"):
        self.dsn = dsn
        self.environment = environment
        self.initialized = False

    def initialize(self):
        """Initialize Sentry with AIVillage-specific configuration."""

        try:
            import sentry_sdk
            from sentry_sdk.integrations.aiohttp import AioHttpIntegration
            from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

            sentry_sdk.init(
                dsn=self.dsn,
                environment=self.environment,
                traces_sample_rate=0.1,  # 10% performance monitoring
                profiles_sample_rate=0.1,  # 10% profiling
                integrations=[
                    AioHttpIntegration(),
                    SqlalchemyIntegration(),
                ],
                before_send=self.filter_sensitive_data,
                before_send_transaction=self.filter_sensitive_transactions
            )

            self.initialized = True
            logger.info("Sentry integration initialized")

        except ImportError:
            logger.warning("Sentry SDK not available")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")

    def filter_sensitive_data(self, event, hint):
        """Filter sensitive data from Sentry events."""

        # Remove sensitive user data
        if 'user' in event:
            event['user'] = {
                'id': event['user'].get('id', 'unknown'),
                # Remove other sensitive fields
            }

        # Filter sensitive request data
        if 'request' in event:
            headers = event['request'].get('headers', {})
            # Remove authorization headers
            headers.pop('Authorization', None)
            headers.pop('Cookie', None)

        return event

    def filter_sensitive_transactions(self, event, hint):
        """Filter sensitive data from performance transactions."""

        # Skip health check transactions
        if event.get('transaction', '').endswith('/health'):
            return None

        return event

    def capture_security_event(self, security_event: SecurityEvent):
        """Capture security events in Sentry."""

        if not self.initialized:
            return

        try:
            import sentry_sdk

            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("event_type", security_event.event_type)
                scope.set_tag("severity", security_event.severity)
                scope.set_tag("source_ip", security_event.source_ip)
                scope.set_context("security_event", {
                    "threat_score": security_event.threat_score,
                    "timestamp": security_event.timestamp.isoformat(),
                    "details": security_event.details
                })

                sentry_sdk.capture_message(
                    f"Security Event: {security_event.event_type}",
                    level="error" if security_event.severity in ["CRITICAL", "HIGH"] else "warning"
                )

        except Exception as e:
            logger.error(f"Failed to send security event to Sentry: {e}")

    def capture_performance_issue(self, component: str, metric_name: str, value: float, threshold: float):
        """Capture performance issues in Sentry."""

        if not self.initialized:
            return

        try:
            import sentry_sdk

            with sentry_sdk.configure_scope() as scope:
                scope.set_tag("component", component)
                scope.set_tag("metric", metric_name)
                scope.set_context("performance", {
                    "value": value,
                    "threshold": threshold,
                    "deviation": ((value - threshold) / threshold) * 100
                })

                sentry_sdk.capture_message(
                    f"Performance Issue: {component} {metric_name} exceeded threshold",
                    level="warning"
                )

        except Exception as e:
            logger.error(f"Failed to send performance issue to Sentry: {e}")

# Integration with existing monitoring
class SentryObservabilityIntegration:
    """Integrate Sentry with AIVillage observability system."""

    def __init__(self, observability: ObservabilitySystem, sentry: SentryIntegration):
        self.observability = observability
        self.sentry = sentry

    async def start_sentry_monitoring(self):
        """Start monitoring for Sentry-worthy events."""

        while True:
            try:
                # Check for performance issues
                await self._check_performance_issues()

                # Check for error spikes
                await self._check_error_spikes()

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Sentry monitoring loop failed: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes

    async def _check_performance_issues(self):
        """Check for performance degradation."""

        # Get recent performance metrics
        recent_metrics = await self.observability.get_recent_metrics(
            metric_names=["response_time_ms", "agent_forge_phase_duration_seconds"],
            time_window=300  # Last 5 minutes
        )

        for metric in recent_metrics:
            if metric["name"] == "response_time_ms" and metric["value"] > 5000:  # 5 seconds
                self.sentry.capture_performance_issue(
                    "web_server", "response_time", metric["value"], 2000
                )
            elif metric["name"] == "agent_forge_phase_duration_seconds" and metric["value"] > 3600:  # 1 hour
                self.sentry.capture_performance_issue(
                    "agent_forge", "phase_duration", metric["value"], 1800
                )
```

## GitHub Integration

### Automated Issue Creation

**Location**: `packages/monitoring/alert_manager.py:318-372`

```python
class GitHubIntegration:
    """Enhanced GitHub integration for automated issue management."""

    def __init__(self, repo: str, token: str):
        self.repo = repo
        self.token = token
        self.api_base = "https://api.github.com"

    async def create_alert_issue(self, alert: Alert) -> Optional[str]:
        """Create GitHub issue for alert with enhanced metadata."""

        title = f"🚨 {alert.severity.upper()}: {alert.name}"

        # Enhanced issue body with runbook links
        body = f"""# Alert Details

**Alert Name**: `{alert.name}`
**Severity**: {self._get_severity_emoji(alert.severity)} {alert.severity.upper()}
**Timestamp**: {alert.timestamp}
**Metric Value**: {alert.metric_value}
**Threshold**: {alert.threshold}

## Description
{alert.description}

## Impact Assessment
{self._get_impact_assessment(alert)}

## Troubleshooting Steps
{self._get_troubleshooting_steps(alert)}

## Labels and Metadata
```json
{json.dumps(alert.labels, indent=2)}
```

## Monitoring Links
- [Grafana Dashboard](https://grafana.aivillage.ai/d/system-overview)
- [Prometheus Metrics](https://prometheus.aivillage.ai)
- [Logs](https://logs.aivillage.ai)

## Related Documentation
{self._get_related_docs(alert)}

---
*🤖 This issue was automatically created by the AIVillage monitoring system.*
*Use the `/resolve` command to mark this alert as resolved.*
"""

        labels = [
            f"severity-{alert.severity.lower()}",
            f"component-{alert.labels.get('component', 'unknown')}",
            "automated-alert",
            "monitoring"
        ]

        payload = {
            "title": title,
            "body": body,
            "labels": labels,
            "assignees": self._get_default_assignees(alert)
        }

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json"
                }

                async with session.post(
                    f"{self.api_base}/repos/{self.repo}/issues",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 201:
                        issue_data = await response.json()
                        issue_url = issue_data["html_url"]
                        logger.info(f"Created GitHub issue: {issue_url}")

                        # Add issue to alert metadata
                        alert.labels["github_issue"] = issue_url

                        return issue_url
                    else:
                        error_text = await response.text()
                        logger.error(f"GitHub API error {response.status}: {error_text}")

        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")

        return None

    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for alert severity."""
        emoji_map = {
            "critical": "🔥",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️"
        }
        return emoji_map.get(severity.lower(), "❓")

    def _get_impact_assessment(self, alert: Alert) -> str:
        """Generate impact assessment based on alert type."""

        impact_map = {
            "high_cpu_usage": "**HIGH**: System performance degradation, potential service slowdown",
            "critical_memory_usage": "**CRITICAL**: Risk of OOM kills, service interruption",
            "slow_response_times": "**MEDIUM**: User experience degradation, potential SLA breach",
            "agent_forge_phase_failure": "**HIGH**: Model training interruption, pipeline blockage",
            "security_threat_detected": "**CRITICAL**: Potential security breach, immediate action required"
        }

        return impact_map.get(alert.name, "**UNKNOWN**: Impact assessment needed")

    def _get_troubleshooting_steps(self, alert: Alert) -> str:
        """Generate troubleshooting steps for alert type."""

        steps_map = {
            "high_cpu_usage": """
1. Check `top` or `htop` for high CPU processes
2. Review recent deployments for performance regressions
3. Check for infinite loops or inefficient algorithms
4. Scale horizontally if CPU usage is consistently high
5. Consider CPU profiling for persistent issues
""",
            "critical_memory_usage": """
1. Check `free -h` and `ps aux --sort=-%mem` for memory usage
2. Look for memory leaks in recent code changes
3. Review garbage collection logs if applicable
4. Check for large data structures or caches
5. Consider increasing memory limits or scaling out
""",
            "agent_forge_phase_failure": """
1. Check Agent Forge logs for error details
2. Verify model input data quality and format
3. Check disk space for checkpoint storage
4. Review GPU memory usage if applicable
5. Restart failed phase with clean state
6. Check for dependency version conflicts
"""
        }

        return steps_map.get(alert.name, "1. Investigate alert details\n2. Check system logs\n3. Review recent changes")

    def _get_related_docs(self, alert: Alert) -> str:
        """Get links to related documentation."""

        docs_map = {
            "agent_forge": "- [Agent Forge Documentation](https://docs.aivillage.ai/agent-forge)\n- [Troubleshooting Guide](https://docs.aivillage.ai/troubleshooting)",
            "security": "- [Security Monitoring Guide](https://docs.aivillage.ai/security)\n- [Incident Response Playbook](https://docs.aivillage.ai/incident-response)",
            "performance": "- [Performance Optimization Guide](https://docs.aivillage.ai/performance)\n- [System Metrics Reference](https://docs.aivillage.ai/metrics)"
        }

        component = alert.labels.get("component", "general")
        return docs_map.get(component, "- [General Documentation](https://docs.aivillage.ai)")

    async def update_issue_status(self, issue_number: int, status: str, comment: str = None):
        """Update GitHub issue status based on alert resolution."""

        if comment:
            await self._add_issue_comment(issue_number, comment)

        if status == "resolved":
            await self._close_issue(issue_number)
        elif status == "investigating":
            await self._add_issue_label(issue_number, "investigating")

    async def _add_issue_comment(self, issue_number: int, comment: str):
        """Add comment to GitHub issue."""

        payload = {"body": comment}

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"token {self.token}",
                    "Accept": "application/vnd.github.v3+json"
                }

                async with session.post(
                    f"{self.api_base}/repos/{self.repo}/issues/{issue_number}/comments",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 201:
                        logger.info(f"Added comment to issue #{issue_number}")
                    else:
                        logger.error(f"Failed to add comment: {response.status}")

        except Exception as e:
            logger.error(f"Failed to add issue comment: {e}")

# GitHub Webhook Handler for issue updates
class GitHubWebhookHandler:
    """Handle GitHub webhooks for issue updates."""

    def __init__(self, alert_manager: AlertManager, webhook_secret: str):
        self.alert_manager = alert_manager
        self.webhook_secret = webhook_secret

    async def handle_webhook(self, request_body: bytes, signature: str):
        """Handle GitHub webhook for issue updates."""

        # Verify webhook signature
        if not self._verify_signature(request_body, signature):
            logger.warning("Invalid GitHub webhook signature")
            return

        try:
            payload = json.loads(request_body.decode())
            action = payload.get("action")
            issue = payload.get("issue", {})

            if action == "closed":
                await self._handle_issue_closed(issue)
            elif action == "labeled":
                await self._handle_issue_labeled(issue, payload.get("label", {}))

        except Exception as e:
            logger.error(f"Failed to process GitHub webhook: {e}")

    def _verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify GitHub webhook signature."""

        import hmac
        import hashlib

        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected_signature}", signature)

    async def _handle_issue_closed(self, issue: Dict[str, Any]):
        """Handle issue closed event."""

        issue_url = issue.get("html_url")
        if issue_url:
            # Find and resolve corresponding alert
            alert = self.alert_manager.find_alert_by_github_issue(issue_url)
            if alert:
                self.alert_manager.resolve_alert(alert.id)
                logger.info(f"Resolved alert {alert.id} via GitHub issue closure")
```

## Weights & Biases Integration

### ML Experiment Tracking

```python
class WandBIntegration:
    """Weights & Biases integration for Agent Forge monitoring."""

    def __init__(self, project_name: str = "aivillage-agent-forge"):
        self.project_name = project_name
        self.run = None
        self.enabled = False

    def initialize(self, config: Dict[str, Any] = None):
        """Initialize W&B tracking."""

        try:
            import wandb

            self.run = wandb.init(
                project=self.project_name,
                config=config or {},
                tags=["aivillage", "monitoring"],
                notes="Automated monitoring from AIVillage observability system"
            )

            self.enabled = True
            logger.info(f"W&B tracking initialized: {self.run.url}")

        except ImportError:
            logger.warning("W&B not available")
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")

    def log_agent_forge_metrics(self, phase: str, metrics: Dict[str, Any]):
        """Log Agent Forge phase metrics to W&B."""

        if not self.enabled:
            return

        try:
            import wandb

            # Prepare metrics for logging
            wandb_metrics = {
                f"phase_{phase}_{key}": value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }

            # Add phase completion status
            wandb_metrics[f"phase_{phase}_completed"] = 1
            wandb_metrics["current_phase"] = phase

            self.run.log(wandb_metrics)

            # Log artifacts if available
            if "model_path" in metrics:
                self.run.log_artifact(metrics["model_path"], type="model")

            logger.debug(f"Logged W&B metrics for phase {phase}")

        except Exception as e:
            logger.error(f"Failed to log W&B metrics: {e}")

    def log_system_metrics(self, metrics: Dict[str, Any]):
        """Log system performance metrics to W&B."""

        if not self.enabled:
            return

        try:
            import wandb

            # Filter and format system metrics
            system_metrics = {
                "system/cpu_usage": metrics.get("cpu", {}).get("percent", 0),
                "system/memory_usage": metrics.get("memory", {}).get("percent", 0),
                "system/gpu_usage": metrics.get("gpu", {}).get("memory", {}).get("percent", 0),
                "system/timestamp": time.time()
            }

            self.run.log(system_metrics)

        except Exception as e:
            logger.error(f"Failed to log system metrics to W&B: {e}")

    def log_security_events(self, event_counts: Dict[str, int]):
        """Log security event summary to W&B."""

        if not self.enabled:
            return

        try:
            import wandb

            security_metrics = {
                f"security/{event_type}_count": count
                for event_type, count in event_counts.items()
            }

            self.run.log(security_metrics)

        except Exception as e:
            logger.error(f"Failed to log security metrics to W&B: {e}")

# W&B Dashboard Setup
class WandBDashboardSetup:
    """Set up W&B dashboards for AIVillage monitoring."""

    def __init__(self, wandb_integration: WandBIntegration):
        self.wandb = wandb_integration

    def create_system_dashboard(self):
        """Create system monitoring dashboard in W&B."""

        if not self.wandb.enabled:
            return

        try:
            import wandb

            # System performance dashboard
            dashboard_config = {
                "name": "AIVillage System Monitoring",
                "description": "Real-time system performance and health metrics",
                "charts": [
                    {
                        "name": "CPU Usage",
                        "type": "line",
                        "query": "system/cpu_usage",
                        "smooth": True
                    },
                    {
                        "name": "Memory Usage",
                        "type": "line",
                        "query": "system/memory_usage",
                        "smooth": True
                    },
                    {
                        "name": "GPU Usage",
                        "type": "line",
                        "query": "system/gpu_usage",
                        "smooth": True
                    },
                    {
                        "name": "Agent Forge Phase Progress",
                        "type": "bar",
                        "query": "phase_*_completed",
                        "group_by": "phase"
                    }
                ]
            }

            # Note: W&B dashboard creation via API requires enterprise features
            logger.info("W&B dashboard configuration ready")

        except Exception as e:
            logger.error(f"Failed to create W&B dashboard: {e}")
```

## Slack Integration

### Real-Time Notifications

```python
class SlackIntegration:
    """Slack integration for real-time monitoring notifications."""

    def __init__(self, webhook_url: str, channel: str = "#monitoring"):
        self.webhook_url = webhook_url
        self.channel = channel

    async def send_alert_notification(self, alert: Alert):
        """Send alert notification to Slack."""

        color_map = {
            "critical": "#ff0000",  # Red
            "high": "#ff8c00",      # Orange
            "medium": "#ffd700",    # Yellow
            "low": "#00ff00"        # Green
        }

        color = color_map.get(alert.severity.lower(), "#808080")

        payload = {
            "channel": self.channel,
            "username": "AIVillage Monitor",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": f"{alert.severity.upper()}: {alert.name}",
                    "text": alert.description,
                    "fields": [
                        {
                            "title": "Metric Value",
                            "value": str(alert.metric_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": str(alert.threshold),
                            "short": True
                        },
                        {
                            "title": "Component",
                            "value": alert.labels.get("component", "Unknown"),
                            "short": True
                        },
                        {
                            "title": "Timestamp",
                            "value": alert.timestamp,
                            "short": True
                        }
                    ],
                    "actions": [
                        {
                            "type": "button",
                            "text": "View Dashboard",
                            "url": "https://dashboard.aivillage.ai"
                        },
                        {
                            "type": "button",
                            "text": "View Logs",
                            "url": "https://logs.aivillage.ai"
                        }
                    ],
                    "footer": "AIVillage Monitoring",
                    "ts": int(time.time())
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack notification sent successfully")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def send_system_status_update(self, status_data: Dict[str, Any]):
        """Send periodic system status update to Slack."""

        payload = {
            "channel": self.channel,
            "username": "AIVillage Monitor",
            "icon_emoji": ":bar_chart:",
            "text": "🏥 *System Health Report*",
            "attachments": [
                {
                    "color": "#36a64f",  # Green
                    "fields": [
                        {
                            "title": "Overall Health",
                            "value": f"{status_data['completion_percentage']:.1f}%",
                            "short": True
                        },
                        {
                            "title": "Healthy Components",
                            "value": f"{status_data['healthy_components']}/{status_data['total_components']}",
                            "short": True
                        },
                        {
                            "title": "Active Alerts",
                            "value": str(status_data.get('active_alerts', 0)),
                            "short": True
                        },
                        {
                            "title": "Last Updated",
                            "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True
                        }
                    ]
                }
            ]
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("Slack status update sent")

        except Exception as e:
            logger.error(f"Failed to send Slack status update: {e}")
```

## OpenTelemetry Integration

### Distributed Tracing Setup

```python
class OpenTelemetryIntegration:
    """OpenTelemetry integration for distributed tracing."""

    def __init__(self, service_name: str, jaeger_endpoint: str = None):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint
        self.tracer = None

    def initialize(self):
        """Initialize OpenTelemetry with Jaeger exporter."""

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.sdk.resources import Resource

            # Set up resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": "production"
            })

            # Set up tracer provider
            trace.set_tracer_provider(TracerProvider(resource=resource))

            # Set up Jaeger exporter
            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=self.jaeger_endpoint.split("://")[1].split(":")[0],
                    agent_port=int(self.jaeger_endpoint.split(":")[-1]),
                )

                span_processor = BatchSpanProcessor(jaeger_exporter)
                trace.get_tracer_provider().add_span_processor(span_processor)

            self.tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracing initialized")

        except ImportError:
            logger.warning("OpenTelemetry not available")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")

    def trace_agent_forge_phase(self, phase_name: str):
        """Create distributed trace for Agent Forge phase."""

        if not self.tracer:
            return

        return self.tracer.start_as_current_span(
            f"agent_forge.{phase_name}",
            attributes={
                "agent_forge.phase": phase_name,
                "service.name": self.service_name
            }
        )

    def trace_security_analysis(self, event_type: str, user_id: str):
        """Create trace for security analysis operations."""

        if not self.tracer:
            return

        return self.tracer.start_as_current_span(
            f"security.analyze.{event_type}",
            attributes={
                "security.event_type": event_type,
                "user.id": user_id,
                "service.name": self.service_name
            }
        )

# Usage example with existing monitoring
async def traced_agent_forge_execution():
    """Example of traced Agent Forge execution."""

    otel = OpenTelemetryIntegration("aivillage-agent-forge")
    otel.initialize()

    with otel.trace_agent_forge_phase("evomerge") as span:
        try:
            # Execute EvoMerge phase
            result = await execute_evomerge_phase()

            span.set_attribute("phase.duration", result.duration)
            span.set_attribute("phase.success", result.success)
            span.set_attribute("models.merged", result.models_count)

            return result

        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise
```

## Best Practices and Troubleshooting

### Integration Health Monitoring

```python
class IntegrationHealthMonitor:
    """Monitor health of external integrations."""

    def __init__(self):
        self.integration_status = {}
        self.last_check = {}

    async def check_all_integrations(self):
        """Check health of all configured integrations."""

        integrations = [
            ("prometheus", self._check_prometheus_health),
            ("grafana", self._check_grafana_health),
            ("sentry", self._check_sentry_health),
            ("github", self._check_github_health),
            ("slack", self._check_slack_health),
            ("wandb", self._check_wandb_health)
        ]

        for name, check_func in integrations:
            try:
                status = await check_func()
                self.integration_status[name] = status
                self.last_check[name] = time.time()

            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                self.integration_status[name] = {"healthy": False, "error": str(e)}

    async def _check_prometheus_health(self) -> Dict[str, Any]:
        """Check Prometheus metrics endpoint health."""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8090/metrics", timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        metric_count = len([line for line in content.split('\n') if line and not line.startswith('#')])

                        return {
                            "healthy": True,
                            "metrics_exported": metric_count,
                            "last_check": datetime.now().isoformat()
                        }
                    else:
                        return {"healthy": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def _check_github_health(self) -> Dict[str, Any]:
        """Check GitHub API connectivity."""

        try:
            token = os.getenv("GITHUB_TOKEN")
            if not token:
                return {"healthy": False, "error": "No GitHub token configured"}

            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"token {token}"}
                async with session.get("https://api.github.com/rate_limit", headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "healthy": True,
                            "rate_limit": data["rate"]["remaining"],
                            "reset_time": data["rate"]["reset"]
                        }
                    else:
                        return {"healthy": False, "error": f"HTTP {response.status}"}

        except Exception as e:
            return {"healthy": False, "error": str(e)}

### Common Integration Issues and Solutions

**Prometheus Metrics Not Appearing**:
```python
# Common fixes:
# 1. Check if metrics server is running
# 2. Verify port configuration
# 3. Check firewall rules
# 4. Validate metric naming conventions

def troubleshoot_prometheus_integration():
    """Troubleshooting steps for Prometheus integration."""

    issues = []

    # Check if metrics server is running
    try:
        response = requests.get("http://localhost:8090/metrics", timeout=5)
        if response.status_code != 200:
            issues.append(f"Metrics endpoint returned {response.status_code}")
    except requests.RequestException as e:
        issues.append(f"Cannot connect to metrics endpoint: {e}")

    # Check metric naming
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import CollectorRegistry, REGISTRY
        metric_names = [metric.describe()[0].name for metric in REGISTRY._collector_to_names.keys()]
        if not metric_names:
            issues.append("No metrics registered with Prometheus client")

    return issues
```

**Sentry Events Not Appearing**:
```python
def troubleshoot_sentry_integration():
    """Troubleshooting steps for Sentry integration."""

    issues = []

    # Check DSN configuration
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        issues.append("SENTRY_DSN environment variable not set")

    # Check Sentry SDK initialization
    try:
        import sentry_sdk
        if not sentry_sdk.Hub.current.client:
            issues.append("Sentry client not initialized")
    except ImportError:
        issues.append("Sentry SDK not installed")

    # Test Sentry connectivity
    try:
        sentry_sdk.capture_message("Test message from troubleshooting")
        logger.info("Test Sentry message sent")
    except Exception as e:
        issues.append(f"Failed to send test message: {e}")

    return issues
```

This comprehensive integration guide provides the foundation for connecting AIVillage monitoring with existing infrastructure and external services, enabling seamless observability across the entire technology stack.
