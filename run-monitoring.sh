#!/bin/bash
echo "🚀 Starting AI Village with monitoring..."
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d

echo "Waiting for services..."
sleep 10
echo "✅ Services started"
echo "📊 Grafana: http://localhost:3000 (admin/changeme)"
echo "📈 Prometheus: http://localhost:9090"
echo "🔧 Gateway: http://localhost:8000"
echo "🤖 Twin: http://localhost:8001"
