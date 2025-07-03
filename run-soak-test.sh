#!/usr/bin/env bash
set -euo pipefail

echo "🧪 AI Village Soak Test Starting..."
export PUSHGATEWAY_URL="localhost:9091"

pip install locust prometheus-client psutil

locust -f tests/soak/locustfile.py \
       --headless -u 200 -r 20 -t 8h \
       --host http://localhost:8000 \
       --html soak-report.html \
       --csv soak-stats

echo "📊 Results: soak-report.html"
echo "📈 Metrics pushed to Prometheus"
