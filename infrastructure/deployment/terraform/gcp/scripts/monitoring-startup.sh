#!/bin/bash

# Startup script for monitoring stack instances (Prometheus, Grafana, AlertManager)
# This script runs on instance startup to configure and start the monitoring containers

set -euo pipefail

# Configuration
ENVIRONMENT="${environment}"
PROJECT_ID="${project_id}"
SERVICE_NAME="aivillage-monitoring"

# Logging setup
exec 1> >(logger -s -t ${SERVICE_NAME})
exec 2>&1

echo "Starting ${SERVICE_NAME} startup script..."
echo "Environment: ${ENVIRONMENT}"
echo "Project ID: ${PROJECT_ID}"

# Wait for Docker to be ready
echo "Waiting for Docker daemon to be ready..."
until docker info > /dev/null 2>&1; do
    echo "Docker daemon not ready, waiting..."
    sleep 5
done
echo "Docker daemon is ready"

# Create data directories
echo "Creating data directories..."
mkdir -p /opt/aivillage/{prometheus,grafana,alertmanager,config,logs}
chmod 755 /opt/aivillage/{prometheus,grafana,alertmanager,config,logs}

# Set proper permissions for Grafana
chown -R 472:472 /opt/aivillage/grafana

# Create Prometheus configuration
echo "Creating Prometheus configuration..."
cat > /opt/aivillage/config/prometheus.yml <<EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'aivillage-${ENVIRONMENT}'
    region: 'gcp-us-west1'

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9091']

  - job_name: 'aivillage-bridge-orchestrator'
    static_configs:
      - targets: ['aivillage-bridge-igm-${ENVIRONMENT}:9090']
    scrape_interval: 10s
    metrics_path: '/metrics'

  - job_name: 'aivillage-python-bridge'
    static_configs:
      - targets: ['aivillage-python-igm-${ENVIRONMENT}:9876']
    scrape_interval: 10s
    metrics_path: '/metrics'

  - job_name: 'gke-nodes'
    kubernetes_sd_configs:
      - role: node
        api_server: 'https://kubernetes.default.svc:443'
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__address__]
        regex: '(.*):10250'
        target_label: __address__
        replacement: '\${1}:9100'

  - job_name: 'constitutional-validation'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 30s
    metrics_path: '/metrics/constitutional'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']

remote_write:
  - url: 'https://monitoring.googleapis.com/v1/projects/${PROJECT_ID}/location/global/prometheus/api/v1/write'
    queue_config:
      capacity: 2500
      max_shards: 200
      min_shards: 1
      max_samples_per_send: 500
      batch_send_deadline: 5s
      min_backoff: 30ms
      max_backoff: 100ms
EOF

# Create Prometheus alerting rules
echo "Creating Prometheus alerting rules..."
mkdir -p /opt/aivillage/config/rules
cat > /opt/aivillage/config/rules/aivillage.yml <<EOF
groups:
  - name: aivillage.rules
    rules:
      - alert: HighP95Latency
        expr: aivillage_p95_latency_milliseconds > 75
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency detected"
          description: "P95 latency is {{ \$value }}ms, above the 75ms target"

      - alert: ConstitutionalViolation
        expr: increase(aivillage_constitutional_violations_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Constitutional violation detected"
          description: "{{ \$value }} constitutional violations in the last 5 minutes"

      - alert: BetaNetBridgeDown
        expr: up{job="aivillage-python-bridge"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "BetaNet bridge is down"
          description: "Python BetaNet bridge has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: rate(aivillage_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High error rate"
          description: "Error rate is {{ \$value }} per second"

      - alert: InstanceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Instance down"
          description: "{{ \$labels.instance }} has been down for more than 1 minute"
EOF

# Create AlertManager configuration
echo "Creating AlertManager configuration..."
cat > /opt/aivillage/config/alertmanager.yml <<EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alertmanager@aivillage.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
        send_resolved: true
        title: 'AIVillage Alert - {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
EOF

# Create Grafana provisioning configuration
echo "Creating Grafana configuration..."
mkdir -p /opt/aivillage/config/grafana/{provisioning/datasources,provisioning/dashboards,dashboards}

cat > /opt/aivillage/config/grafana/provisioning/datasources/prometheus.yml <<EOF
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9091
    isDefault: true
    editable: true
EOF

cat > /opt/aivillage/config/grafana/provisioning/dashboards/aivillage.yml <<EOF
apiVersion: 1
providers:
  - name: 'AIVillage'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create Grafana dashboard for AIVillage
cat > /opt/aivillage/config/grafana/dashboards/aivillage-dashboard.json <<'EOF'
{
  "dashboard": {
    "id": null,
    "title": "AIVillage Fog Computing Platform",
    "tags": ["aivillage", "fog-computing", "betanet"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "P95 Latency",
        "type": "stat",
        "targets": [
          {
            "expr": "aivillage_p95_latency_milliseconds",
            "legendFormat": "P95 Latency (ms)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "max": 100,
            "min": 0,
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 75}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "BetaNet Translations",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(aivillage_betanet_translations_total[5m])",
            "legendFormat": "Translations/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Constitutional Violations",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(aivillage_constitutional_violations_total[1h])",
            "legendFormat": "Violations (1h)"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "red", "value": 1}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "Instance Health",
        "type": "table",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{instance}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

# Start Prometheus
echo "Starting Prometheus..."
docker run -d \
    --name prometheus \
    --restart unless-stopped \
    -p 9091:9090 \
    -v /opt/aivillage/config/prometheus.yml:/etc/prometheus/prometheus.yml \
    -v /opt/aivillage/config/rules:/etc/prometheus/rules \
    -v /opt/aivillage/prometheus:/prometheus \
    prom/prometheus:v2.45.0 \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/etc/prometheus/console_libraries \
    --web.console.templates=/etc/prometheus/consoles \
    --storage.tsdb.retention.time=30d \
    --web.enable-lifecycle

# Start AlertManager
echo "Starting AlertManager..."
docker run -d \
    --name alertmanager \
    --restart unless-stopped \
    -p 9093:9093 \
    -v /opt/aivillage/config/alertmanager.yml:/etc/alertmanager/config.yml \
    -v /opt/aivillage/alertmanager:/alertmanager \
    prom/alertmanager:v0.26.0

# Start Grafana
echo "Starting Grafana..."
docker run -d \
    --name grafana \
    --restart unless-stopped \
    -p 3000:3000 \
    -e GF_SECURITY_ADMIN_PASSWORD=admin123 \
    -e GF_INSTALL_PLUGINS=grafana-piechart-panel,grafana-worldmap-panel \
    -e GF_SERVER_ROOT_URL=http://localhost:3000 \
    -v /opt/aivillage/grafana:/var/lib/grafana \
    -v /opt/aivillage/config/grafana/provisioning:/etc/grafana/provisioning \
    -v /opt/aivillage/config/grafana/dashboards:/var/lib/grafana/dashboards \
    grafana/grafana:10.1.0

# Wait for services to be ready
echo "Waiting for monitoring services to be ready..."
sleep 30

# Health checks
echo "Performing health checks..."

# Check Prometheus
if curl -f -s http://localhost:9091/-/healthy > /dev/null; then
    echo "✓ Prometheus health check passed"
else
    echo "✗ Prometheus health check failed"
    docker logs prometheus
fi

# Check AlertManager
if curl -f -s http://localhost:9093/-/healthy > /dev/null; then
    echo "✓ AlertManager health check passed"
else
    echo "✗ AlertManager health check failed"
    docker logs alertmanager
fi

# Check Grafana
if curl -f -s http://localhost:3000/api/health > /dev/null; then
    echo "✓ Grafana health check passed"
else
    echo "✗ Grafana health check failed"
    docker logs grafana
fi

# Set up monitoring script
echo "Setting up monitoring script..."
cat > /opt/aivillage/monitor.sh <<'EOF'
#!/bin/bash

# Monitoring script for the monitoring stack
SERVICES=("prometheus" "alertmanager" "grafana")

for service in "${SERVICES[@]}"; do
    if ! docker ps | grep -q $service; then
        echo "ERROR: $service container is not running"
        exit 1
    fi
done

# Check service endpoints
if ! curl -f -s http://localhost:9091/-/healthy > /dev/null; then
    echo "ERROR: Prometheus health check failed"
    exit 1
fi

if ! curl -f -s http://localhost:9093/-/healthy > /dev/null; then
    echo "ERROR: AlertManager health check failed"
    exit 1
fi

if ! curl -f -s http://localhost:3000/api/health > /dev/null; then
    echo "ERROR: Grafana health check failed"
    exit 1
fi

echo "All monitoring services are healthy"
EOF

chmod +x /opt/aivillage/monitor.sh

# Set up cron job for monitoring
echo "Setting up monitoring cron job..."
echo "*/5 * * * * root /opt/aivillage/monitor.sh > /var/log/aivillage-monitoring.log 2>&1" > /etc/cron.d/aivillage-monitoring

# Configure firewall rules
echo "Configuring firewall rules..."
iptables -I INPUT -p tcp --dport 3000 -j ACCEPT   # Grafana
iptables -I INPUT -p tcp --dport 9091 -j ACCEPT   # Prometheus
iptables -I INPUT -p tcp --dport 9093 -j ACCEPT   # AlertManager

# Save iptables rules
iptables-save > /etc/iptables.rules

echo "Monitoring stack startup completed successfully!"
echo "Access URLs:"
echo "  Grafana: http://localhost:3000 (admin/admin123)"
echo "  Prometheus: http://localhost:9091"
echo "  AlertManager: http://localhost:9093"

echo "Startup script completed at $(date)"