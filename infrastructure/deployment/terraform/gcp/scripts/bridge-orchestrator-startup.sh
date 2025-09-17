#!/bin/bash

# Startup script for TypeScript Bridge Orchestrator instances
# This script runs on instance startup to configure and start the Docker container

set -euo pipefail

# Configuration
DOCKER_IMAGE="${docker_image}"
ENVIRONMENT="${environment}"
PROJECT_ID="${project_id}"
SERVICE_NAME="aivillage-bridge-orchestrator"

# Logging setup
exec 1> >(logger -s -t ${SERVICE_NAME})
exec 2>&1

echo "Starting ${SERVICE_NAME} startup script..."
echo "Environment: ${ENVIRONMENT}"
echo "Project ID: ${PROJECT_ID}"
echo "Docker Image: ${DOCKER_IMAGE}"

# Wait for Docker to be ready
echo "Waiting for Docker daemon to be ready..."
until docker info > /dev/null 2>&1; do
    echo "Docker daemon not ready, waiting..."
    sleep 5
done
echo "Docker daemon is ready"

# Configure Docker for GCR access
echo "Configuring Docker for GCR access..."
gcloud auth configure-docker --quiet

# Set up environment variables
cat > /tmp/bridge-orchestrator.env <<EOF
NODE_ENV=${ENVIRONMENT}
PORT=8080
METRICS_PORT=9090
PYTHON_BRIDGE_HOST=localhost
PYTHON_BRIDGE_PORT=9876
CONSTITUTIONAL_TIER=Silver
PRIVACY_MODE=enhanced
TARGET_P95_LATENCY=75
CIRCUIT_BREAKER_ENABLED=true
LOG_LEVEL=info
LOG_FORMAT=json
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
ENABLE_STACKDRIVER=true
ENABLE_TRACE=true
ENABLE_METRICS=true
EOF

# Pull the latest Docker image
echo "Pulling Docker image: ${DOCKER_IMAGE}"
docker pull ${DOCKER_IMAGE}

# Stop and remove existing container if it exists
echo "Stopping existing container if running..."
docker stop ${SERVICE_NAME} 2>/dev/null || true
docker rm ${SERVICE_NAME} 2>/dev/null || true

# Create data directories
echo "Creating data directories..."
mkdir -p /opt/aivillage/{logs,data,config}
chmod 755 /opt/aivillage/{logs,data,config}

# Start the container
echo "Starting ${SERVICE_NAME} container..."
docker run -d \
    --name ${SERVICE_NAME} \
    --restart unless-stopped \
    --env-file /tmp/bridge-orchestrator.env \
    -p 8080:8080 \
    -p 8081:8081 \
    -p 9090:9090 \
    -v /opt/aivillage/logs:/app/logs \
    -v /opt/aivillage/data:/app/data \
    -v /opt/aivillage/config:/app/config \
    --health-cmd="node healthcheck.js" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-start-period=40s \
    --health-retries=3 \
    ${DOCKER_IMAGE}

# Wait for container to be healthy
echo "Waiting for container to be healthy..."
timeout=300
counter=0
while [ $counter -lt $timeout ]; do
    if docker inspect ${SERVICE_NAME} | grep -q '"Health".*"healthy"'; then
        echo "Container is healthy!"
        break
    fi

    if [ $counter -gt 0 ] && [ $((counter % 30)) -eq 0 ]; then
        echo "Still waiting for container to be healthy... (${counter}s elapsed)"
        # Show container logs for debugging
        echo "Recent container logs:"
        docker logs --tail 10 ${SERVICE_NAME}
    fi

    sleep 5
    counter=$((counter + 5))
done

if [ $counter -ge $timeout ]; then
    echo "ERROR: Container failed to become healthy within ${timeout} seconds"
    echo "Container logs:"
    docker logs ${SERVICE_NAME}
    exit 1
fi

# Set up log rotation
echo "Setting up log rotation..."
cat > /etc/logrotate.d/aivillage-bridge <<EOF
/opt/aivillage/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
EOF

# Set up monitoring script
echo "Setting up monitoring script..."
cat > /opt/aivillage/monitor.sh <<'EOF'
#!/bin/bash

# Simple monitoring script for the bridge orchestrator
SERVICE_NAME="aivillage-bridge-orchestrator"

# Check if container is running
if ! docker ps | grep -q ${SERVICE_NAME}; then
    echo "ERROR: ${SERVICE_NAME} container is not running"
    # Attempt to restart
    systemctl restart google-startup-scripts.service
    exit 1
fi

# Check if container is healthy
if ! docker inspect ${SERVICE_NAME} | grep -q '"Health".*"healthy"'; then
    echo "WARNING: ${SERVICE_NAME} container is not healthy"
    docker logs --tail 20 ${SERVICE_NAME}
    exit 1
fi

# Check if ports are listening
if ! netstat -tlpn | grep -q ":8080.*LISTEN"; then
    echo "ERROR: Port 8080 is not listening"
    exit 1
fi

if ! netstat -tlpn | grep -q ":9090.*LISTEN"; then
    echo "ERROR: Port 9090 is not listening"
    exit 1
fi

echo "All checks passed - ${SERVICE_NAME} is running correctly"
EOF

chmod +x /opt/aivillage/monitor.sh

# Set up cron job for monitoring
echo "Setting up monitoring cron job..."
echo "*/5 * * * * root /opt/aivillage/monitor.sh > /var/log/aivillage-monitor.log 2>&1" > /etc/cron.d/aivillage-bridge

# Configure firewall rules
echo "Configuring firewall rules..."
# Allow HTTP traffic on port 8080
iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
# Allow WebSocket traffic on port 8081
iptables -I INPUT -p tcp --dport 8081 -j ACCEPT
# Allow metrics traffic on port 9090
iptables -I INPUT -p tcp --dport 9090 -j ACCEPT

# Save iptables rules
iptables-save > /etc/iptables.rules

# Set up iptables restore on boot
cat > /etc/systemd/system/iptables-restore.service <<EOF
[Unit]
Description=Restore iptables rules
After=network.target

[Service]
Type=oneshot
ExecStart=/sbin/iptables-restore /etc/iptables.rules
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl enable iptables-restore.service

# Final health check
echo "Performing final health check..."
sleep 10

# Check HTTP endpoint
if curl -f -s http://localhost:8080/health > /dev/null; then
    echo "✓ HTTP health check passed"
else
    echo "✗ HTTP health check failed"
    docker logs ${SERVICE_NAME}
    exit 1
fi

# Check metrics endpoint
if curl -f -s http://localhost:9090/metrics > /dev/null; then
    echo "✓ Metrics endpoint check passed"
else
    echo "✗ Metrics endpoint check failed"
fi

echo "Bridge Orchestrator startup completed successfully!"

# Clean up temporary files
rm -f /tmp/bridge-orchestrator.env

echo "Startup script completed at $(date)"