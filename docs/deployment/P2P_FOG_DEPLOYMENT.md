# P2P/Fog Computing Deployment Guide

## Overview

This guide covers the deployment of the AIVillage P2P/Fog computing system with integrated admin interfaces. The system provides comprehensive real-time monitoring and management capabilities for distributed peer-to-peer networks and fog computing resources.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB available disk space
- **Network**: Stable internet connection for P2P communication

### Software Requirements
- **Python**: 3.8+ with pip
- **Node.js**: 14+ (optional, for advanced development)
- **Web Browser**: Modern browser with WebSocket support
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 20.04+)

## Quick Start Deployment

### 1. System Preparation

```bash
# Clone the repository
git clone https://github.com/DNYoussef/AIVillage.git
cd AIVillage

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables for optimal performance
export PYTHONIOENCODING=utf-8
export PYTHONPATH=.:core:infrastructure
```

### 2. Start Core Services

```bash
# Terminal 1: Start the Unified Backend
cd infrastructure/gateway
python unified_agent_forge_backend.py

# Terminal 2: Start the Web Server (from root directory)
python -m http.server 3000 --directory .
```

### 3. Access Admin Interfaces

Once services are running, access the interfaces:

- **Integration Test**: http://localhost:3000/test_integration.html
- **Frontend Dashboard**: http://localhost:3000/ui/web/public/admin-dashboard.html
- **Backend Interface**: http://localhost:3000/infrastructure/gateway/admin_interface.html

## Production Deployment

### Docker Deployment (Recommended)

#### 1. Build Docker Images

```bash
# Create Dockerfile for backend
cat > Dockerfile.backend << EOF
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8083
CMD ["python", "infrastructure/gateway/unified_agent_forge_backend.py"]
EOF

# Create Dockerfile for frontend
cat > Dockerfile.frontend << EOF
FROM nginx:alpine

COPY ui/ /usr/share/nginx/html/ui/
COPY infrastructure/ /usr/share/nginx/html/infrastructure/
COPY test_integration.html /usr/share/nginx/html/

EXPOSE 80
EOF

# Build images
docker build -f Dockerfile.backend -t aivillage-backend .
docker build -f Dockerfile.frontend -t aivillage-frontend .
```

#### 2. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8083:8083"
    environment:
      - PYTHONIOENCODING=utf-8
      - PYTHONPATH=.:core:infrastructure
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8083/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
    restart: unless-stopped
```

#### 3. Deploy with Docker Compose

```bash
# Start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Kubernetes Deployment

#### 1. Backend Deployment

```yaml
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-backend
  labels:
    app: aivillage-backend
spec:
  replicas: 2
  selector:
    matchLabels:
      app: aivillage-backend
  template:
    metadata:
      labels:
        app: aivillage-backend
    spec:
      containers:
      - name: backend
        image: aivillage-backend:latest
        ports:
        - containerPort: 8083
        env:
        - name: PYTHONIOENCODING
          value: "utf-8"
        - name: PYTHONPATH
          value: ".:core:infrastructure"
        livenessProbe:
          httpGet:
            path: /health
            port: 8083
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8083
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: aivillage-backend-service
spec:
  selector:
    app: aivillage-backend
  ports:
    - protocol: TCP
      port: 8083
      targetPort: 8083
  type: ClusterIP
```

#### 2. Frontend Deployment

```yaml
# k8s/frontend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aivillage-frontend
  labels:
    app: aivillage-frontend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aivillage-frontend
  template:
    metadata:
      labels:
        app: aivillage-frontend
    spec:
      containers:
      - name: frontend
        image: aivillage-frontend:latest
        ports:
        - containerPort: 80

---
apiVersion: v1
kind: Service
metadata:
  name: aivillage-frontend-service
spec:
  selector:
    app: aivillage-frontend
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: LoadBalancer
```

#### 3. Deploy to Kubernetes

```bash
# Apply deployments
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# Check deployments
kubectl get deployments
kubectl get services
kubectl get pods

# Access logs
kubectl logs -f deployment/aivillage-backend
```

## Environment Configuration

### Environment Variables

```bash
# Core Configuration
export PYTHONIOENCODING=utf-8
export PYTHONPATH=.:core:infrastructure

# P2P Configuration
export P2P_ENABLE_BITCHAT=true
export P2P_ENABLE_BETANET=true
export P2P_DEFAULT_PORT=8080

# Fog Computing Configuration
export FOG_ENABLE_HARVESTING=true
export FOG_TOKEN_REWARDS=true
export FOG_COORDINATOR_NODE=true

# Security Configuration
export ENABLE_ONION_ROUTING=true
export PRIVACY_LEVEL=high
export SECURITY_AUDIT_MODE=false

# Performance Configuration
export WEBSOCKET_MAX_CONNECTIONS=100
export API_RATE_LIMIT=1000
export CACHE_TIMEOUT=300
```

### Configuration Files

#### Backend Configuration
```python
# config/backend_config.py
import os

BACKEND_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("BACKEND_PORT", 8083)),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "cors_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "websocket_max_connections": int(os.getenv("WEBSOCKET_MAX_CONNECTIONS", 100)),
    "api_rate_limit": int(os.getenv("API_RATE_LIMIT", 1000))
}

P2P_CONFIG = {
    "enable_bitchat": os.getenv("P2P_ENABLE_BITCHAT", "true").lower() == "true",
    "enable_betanet": os.getenv("P2P_ENABLE_BETANET", "true").lower() == "true",
    "default_port": int(os.getenv("P2P_DEFAULT_PORT", 8080))
}

FOG_CONFIG = {
    "enable_harvesting": os.getenv("FOG_ENABLE_HARVESTING", "true").lower() == "true",
    "token_rewards": os.getenv("FOG_TOKEN_REWARDS", "true").lower() == "true",
    "coordinator_node": os.getenv("FOG_COORDINATOR_NODE", "true").lower() == "true"
}
```

## Network Configuration

### Port Requirements
- **8083**: Backend API and WebSocket server
- **3000/80**: Frontend web server
- **443**: HTTPS (production)
- **8080**: P2P communication (configurable)

### Firewall Configuration

```bash
# Ubuntu/Debian
sudo ufw allow 8083/tcp  # Backend API
sudo ufw allow 3000/tcp  # Frontend (development)
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow 8080/tcp  # P2P communication

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=8083/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --permanent --add-port=80/tcp
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

### Reverse Proxy Configuration (Nginx)

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server 127.0.0.1:8083;
    }

    upstream frontend {
        server 127.0.0.1:3000;
    }

    server {
        listen 80;
        server_name yourdomain.com;

        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name yourdomain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Backend API routes
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket routes
        location /ws {
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

## Security Considerations

### HTTPS Configuration
```bash
# Generate SSL certificates (self-signed for development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# For production, use Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Authentication Setup (Production)
```python
# Add to backend configuration
SECURITY_CONFIG = {
    "enable_auth": True,
    "jwt_secret": "your-secure-jwt-secret",
    "session_timeout": 3600,
    "rate_limiting": {
        "api": "100/hour",
        "websocket": "10/minute"
    }
}
```

## Monitoring and Logging

### Health Monitoring
```bash
# Health check script
#!/bin/bash
# health_check.sh

# Check backend health
BACKEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8083/health)
if [ "$BACKEND_STATUS" != "200" ]; then
    echo "Backend health check failed: $BACKEND_STATUS"
    exit 1
fi

# Check WebSocket connectivity
python3 -c "
import asyncio
import websockets
import sys

async def test_ws():
    try:
        uri = 'ws://localhost:8083/ws'
        async with websockets.connect(uri) as websocket:
            await websocket.recv()
        print('WebSocket health check passed')
    except Exception as e:
        print(f'WebSocket health check failed: {e}')
        sys.exit(1)

asyncio.run(test_ws())
"

echo "All health checks passed"
```

### Log Configuration
```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            RotatingFileHandler(
                'logs/aivillage.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
```

## Troubleshooting

### Common Issues

#### Backend Won't Start
```bash
# Check port availability
netstat -an | grep :8083

# Check Python dependencies
pip check

# Check environment variables
echo $PYTHONPATH

# View detailed logs
python unified_agent_forge_backend.py --debug
```

#### WebSocket Connection Issues
```bash
# Test WebSocket directly
websocat ws://localhost:8083/ws

# Check firewall
sudo ufw status

# Check nginx proxy (if used)
nginx -t
sudo nginx -s reload
```

#### Frontend Loading Issues
```bash
# Check web server
curl -I http://localhost:3000/

# Check component library
curl -I http://localhost:3000/ui/components/p2p-fog-components.js

# Check browser console
# Open Developer Tools > Console for JavaScript errors
```

### Performance Tuning

#### Backend Optimization
```python
# Add to backend configuration
PERFORMANCE_CONFIG = {
    "worker_processes": 4,
    "worker_connections": 1000,
    "keepalive_timeout": 65,
    "client_max_body_size": "50M"
}
```

#### Database Optimization (if applicable)
```bash
# PostgreSQL tuning
# Add to postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

## Backup and Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/aivillage/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup logs
cp -r logs/ "$BACKUP_DIR/"

# Backup data (if any persistent data exists)
cp -r data/ "$BACKUP_DIR/" 2>/dev/null || true

# Create tarball
tar -czf "$BACKUP_DIR.tar.gz" -C "$(dirname "$BACKUP_DIR")" "$(basename "$BACKUP_DIR")"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
```

### Recovery Procedure
```bash
#!/bin/bash
# restore.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    exit 1
fi

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

# Stop services
docker-compose down  # or systemctl stop aivillage

# Restore files
cp -r /tmp/*/config/ ./
cp -r /tmp/*/logs/ ./
cp -r /tmp/*/data/ ./ 2>/dev/null || true

# Restart services
docker-compose up -d  # or systemctl start aivillage

echo "Restore completed from: $BACKUP_FILE"
```

## Support and Maintenance

### Regular Maintenance Tasks
1. **Log Rotation**: Configure logrotate for application logs
2. **Security Updates**: Regular system and dependency updates
3. **Performance Monitoring**: Monitor CPU, memory, and network usage
4. **Backup Verification**: Regular backup testing and validation
5. **Certificate Renewal**: Automate SSL certificate renewal

### Support Resources
- **Documentation**: `/docs` directory for detailed guides
- **GitHub Issues**: https://github.com/DNYoussef/AIVillage/issues
- **Health Endpoints**: Use `/health` for automated monitoring
- **Integration Tests**: Run test suite for validation

This deployment guide provides comprehensive instructions for deploying the P2P/Fog computing system in various environments, from development to production-ready containerized deployments.
