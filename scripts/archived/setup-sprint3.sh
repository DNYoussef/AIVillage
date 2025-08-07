#!/bin/bash
echo "🚀 AI Village Sprint 3 Setup"

mkdir -p monitoring/grafana/{provisioning,dashboards} tests/soak docs/adr

pip install locust prometheus-client psutil pytest-cov

chmod +x run-*.sh

echo "✅ Sprint 3 setup complete!"
echo ""
echo "Next steps:"
echo "1. Start services:    ./scripts/manage.sh start"
echo "2. Run quick test:    ./scripts/manage.sh soak-test"
echo "3. Access Grafana:    http://localhost:3000"
