#!/bin/bash

# Startup script for Python BetaNet Bridge instances
# This script runs on instance startup to configure and start the Docker container

set -euo pipefail

# Configuration
DOCKER_IMAGE="${docker_image}"
ENVIRONMENT="${environment}"
PROJECT_ID="${project_id}"
SERVICE_NAME="aivillage-python-bridge"

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
cat > /tmp/python-bridge.env <<EOF
PYTHONPATH=/app
HOST=0.0.0.0
PORT=9876
CONSTITUTIONAL_TIER=Silver
PRIVACY_MODE=enhanced
LOG_LEVEL=info
BETANET_BRIDGE_MODE=${ENVIRONMENT}
ENABLE_METRICS=true
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
ENABLE_STACKDRIVER=true
ENABLE_TRACE=true
MAX_WORKERS=4
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
mkdir -p /opt/aivillage/{logs,data,tmp}
chmod 755 /opt/aivillage/{logs,data,tmp}

# Start the container
echo "Starting ${SERVICE_NAME} container..."
docker run -d \
    --name ${SERVICE_NAME} \
    --restart unless-stopped \
    --env-file /tmp/python-bridge.env \
    -p 9876:9876 \
    -v /opt/aivillage/logs:/app/logs \
    -v /opt/aivillage/data:/app/data \
    -v /opt/aivillage/tmp:/app/tmp \
    --health-cmd="python healthcheck.py" \
    --health-interval=30s \
    --health-timeout=10s \
    --health-start-period=60s \
    --health-retries=3 \
    ${DOCKER_IMAGE}

# Wait for container to be healthy
echo "Waiting for container to be healthy..."
timeout=600  # Longer timeout for Python service
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
cat > /etc/logrotate.d/aivillage-python <<EOF
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

# Simple monitoring script for the Python bridge
SERVICE_NAME="aivillage-python-bridge"

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

# Check if port is listening
if ! netstat -tlpn | grep -q ":9876.*LISTEN"; then
    echo "ERROR: Port 9876 is not listening"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(docker stats --no-stream --format "{{.MemPerc}}" ${SERVICE_NAME} | sed 's/%//')
if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
    echo "WARNING: High memory usage: ${MEMORY_USAGE}%"
fi

echo "All checks passed - ${SERVICE_NAME} is running correctly"
EOF

chmod +x /opt/aivillage/monitor.sh

# Set up cron job for monitoring
echo "Setting up monitoring cron job..."
echo "*/5 * * * * root /opt/aivillage/monitor.sh > /var/log/aivillage-python-monitor.log 2>&1" > /etc/cron.d/aivillage-python

# Configure firewall rules
echo "Configuring firewall rules..."
# Allow JSON-RPC traffic on port 9876
iptables -I INPUT -p tcp --dport 9876 -j ACCEPT

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

# Install additional monitoring tools
echo "Installing monitoring tools..."
# Install netcat for port checking
apt-get update -qq
apt-get install -y netcat bc

# Set up BetaNet-specific monitoring
echo "Setting up BetaNet monitoring..."
cat > /opt/aivillage/betanet-monitor.sh <<'EOF'
#!/bin/bash

# BetaNet-specific monitoring script
SERVICE_NAME="aivillage-python-bridge"

# Check BetaNet JSON-RPC endpoint
check_jsonrpc() {
    local response=$(echo '{"jsonrpc":"2.0","id":"health_check","method":"get_health_status","params":{}}' | nc -w 5 localhost 9876)

    if [[ $response == *"healthy"* ]]; then
        echo "✓ BetaNet JSON-RPC health check passed"
        return 0
    else
        echo "✗ BetaNet JSON-RPC health check failed: $response"
        return 1
    fi
}

# Check constitutional validation
check_constitutional() {
    local response=$(echo '{"jsonrpc":"2.0","id":"const_check","method":"validate_constitutional_tier","params":{"tier":"Silver"}}' | nc -w 5 localhost 9876)

    if [[ $response == *"valid"* ]]; then
        echo "✓ Constitutional validation check passed"
        return 0
    else
        echo "✗ Constitutional validation check failed: $response"
        return 1
    fi
}

# Run checks
echo "Running BetaNet-specific health checks..."

if check_jsonrpc && check_constitutional; then
    echo "All BetaNet checks passed"
    exit 0
else
    echo "Some BetaNet checks failed"
    # Log container status for debugging
    docker logs --tail 50 ${SERVICE_NAME}
    exit 1
fi
EOF

chmod +x /opt/aivillage/betanet-monitor.sh

# Set up BetaNet monitoring cron job
echo "*/10 * * * * root /opt/aivillage/betanet-monitor.sh >> /var/log/aivillage-betanet-monitor.log 2>&1" >> /etc/cron.d/aivillage-python

# Final health check
echo "Performing final health check..."
sleep 15

# Check JSON-RPC endpoint
echo "Testing JSON-RPC endpoint..."
response=$(echo '{"jsonrpc":"2.0","id":"startup_check","method":"get_health_status","params":{}}' | nc -w 10 localhost 9876)

if [[ $response == *"healthy"* ]]; then
    echo "✓ JSON-RPC health check passed"
else
    echo "✗ JSON-RPC health check failed: $response"
    echo "Container logs:"
    docker logs ${SERVICE_NAME}
    exit 1
fi

# Check constitutional validation
echo "Testing constitutional validation..."
const_response=$(echo '{"jsonrpc":"2.0","id":"const_startup","method":"validate_constitutional_tier","params":{"tier":"Silver"}}' | nc -w 10 localhost 9876)

if [[ $const_response == *"valid"* ]]; then
    echo "✓ Constitutional validation check passed"
else
    echo "✗ Constitutional validation check failed: $const_response"
fi

echo "Python BetaNet Bridge startup completed successfully!"

# Clean up temporary files
rm -f /tmp/python-bridge.env

echo "Startup script completed at $(date)"