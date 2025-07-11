#!/usr/bin/env bash
set -euo pipefail

# Default service URLs
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GATEWAY_URL="${GATEWAY_URL:-http://localhost:8000}"
TWIN_URL="${TWIN_URL:-http://localhost:8001}"
PUSHGATEWAY_URL="${PUSHGATEWAY_URL:-localhost:9091}"

start() {
    echo "🚀 Starting AI Village with monitoring..."
    docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

    echo "Waiting for services..."
    sleep 10
    echo "✅ Services started"
    echo "📊 Grafana: ${GRAFANA_URL} (admin/changeme)"
    echo "📈 Prometheus: ${PROMETHEUS_URL}"
    echo "🔧 Gateway: ${GATEWAY_URL}"
    echo "🤖 Twin: ${TWIN_URL}"
}

soak_test() {
    echo "🧪 AI Village Soak Test Starting..."
    pip install locust prometheus-client psutil >/dev/null

    locust -f tests/soak/locustfile_advanced.py \
           --headless -u 200 -r 20 -t 8h \
           --host "${GATEWAY_URL}" \
           --html soak-report.html \
           --csv soak-stats

    echo "📊 Results: soak-report.html"
    echo "📈 Metrics pushed to Prometheus via ${PUSHGATEWAY_URL}"
}

case "${1:-}" in
    start)
        start
        ;;
    soak-test)
        soak_test
        ;;
    *)
        echo "Usage: $0 {start|soak-test}" >&2
        exit 1
        ;;
esac
