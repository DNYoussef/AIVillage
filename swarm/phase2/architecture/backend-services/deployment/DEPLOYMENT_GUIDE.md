# Agent Forge Microservices Deployment Guide

## Overview

This guide covers deploying the Agent Forge microservice architecture using Docker Compose for development and Kubernetes for production.

## Architecture Summary

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Service   │    │ WebSocket       │    │  Monitoring     │
│   Port: 8000    │    │ Service         │    │  Service        │
└─────────────────┘    │ Port: 8003      │    │  Port: 8004     │
                       └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Training        │    │    Model        │    │     Redis       │
│ Service         │    │   Service       │    │  (Event Bus)    │
│ Port: 8001      │    │  Port: 8002     │    │  Port: 6379     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    │   Port: 5432    │
                    └─────────────────┘
```

## Prerequisites

### For Docker Deployment
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- NVIDIA Docker runtime (for GPU support)

### For Kubernetes Deployment
- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+
- 16GB+ RAM cluster
- GPU nodes (optional, for training acceleration)

## Docker Compose Deployment

### Quick Start

1. **Clone and Setup**
```bash
cd swarm/phase2/architecture/backend-services
cp deployment/.env.example deployment/.env
# Edit .env file with your settings
```

2. **Start Infrastructure**
```bash
docker-compose up -d redis postgres
docker-compose logs -f redis postgres  # Wait for healthy status
```

3. **Start Services**
```bash
docker-compose up -d
docker-compose ps  # Check all services are running
```

4. **Verify Deployment**
```bash
# Check API health
curl http://localhost:8000/health

# Check service status
curl http://localhost:8000/phases/status

# WebSocket test (requires wscat: npm install -g wscat)
wscat -c ws://localhost:8003/ws
```

### Environment Configuration

Create `deployment/.env`:
```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# WebSocket Service  
WS_MAX_CONNECTIONS=1000
WS_PING_INTERVAL=30

# Monitoring
MONITORING_INTERVAL=30

# GPU Support (if available)
CUDA_VISIBLE_DEVICES=0

# Grafana
GRAFANA_PASSWORD=admin
```

### Development Mode

For development with hot reload:

```bash
# Override with development compose file
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f api-service training-service
```

### Scaling Services

Scale specific services:
```bash
# Scale API service for load handling
docker-compose up -d --scale api-service=3

# Scale training service for parallel jobs
docker-compose up -d --scale training-service=2
```

## Kubernetes Deployment

### 1. Setup Namespace

```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

### 2. Deploy Infrastructure

```bash
# Deploy PostgreSQL
kubectl apply -f deployment/kubernetes/postgres/

# Deploy Redis
kubectl apply -f deployment/kubernetes/redis/

# Wait for infrastructure to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n agentforge --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n agentforge --timeout=300s
```

### 3. Deploy Microservices

```bash
# Deploy in dependency order
kubectl apply -f deployment/kubernetes/model-service/
kubectl apply -f deployment/kubernetes/training-service/
kubectl apply -f deployment/kubernetes/websocket-service/
kubectl apply -f deployment/kubernetes/monitoring-service/
kubectl apply -f deployment/kubernetes/api-service/

# Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=api-service -n agentforge --timeout=300s
```

### 4. Deploy Ingress and Load Balancer

```bash
# Install nginx ingress controller (if not already installed)
helm upgrade --install ingress-nginx ingress-nginx \
  --repo https://kubernetes.github.io/ingress-nginx \
  --namespace ingress-nginx --create-namespace

# Deploy ingress
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 5. Deploy Observability Stack

```bash
# Deploy Prometheus and Grafana
kubectl apply -f deployment/kubernetes/monitoring/
```

## Service-Specific Configuration

### Training Service (GPU Support)

For GPU-accelerated training:

**Docker Compose:**
```yaml
training-service:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Kubernetes:**
```yaml
spec:
  containers:
  - name: training-service
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
```

### Model Service (Storage)

Configure persistent storage:

**Docker Compose:**
```yaml
volumes:
  model_storage:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /path/to/model/storage
```

**Kubernetes:**
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-storage
spec:
  capacity:
    storage: 100Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: fast-ssd
```

### WebSocket Service (Load Balancing)

For high availability WebSocket connections:

**NGINX Configuration:**
```nginx
upstream websocket {
    ip_hash;  # Sticky sessions for WebSocket
    server websocket-service-1:8003;
    server websocket-service-2:8003;
    server websocket-service-3:8003;
}
```

## Monitoring and Observability

### Accessing Dashboards

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **API Documentation**: http://localhost:8000/docs

### Key Metrics to Monitor

1. **API Service**
   - Request rate and response time
   - Error rate
   - Active connections

2. **Training Service**
   - GPU utilization
   - Training job queue length
   - Memory usage

3. **Model Service**
   - Storage usage
   - File operation latency
   - Model cache hit rate

4. **WebSocket Service**
   - Connection count
   - Message throughput
   - Connection duration

### Health Checks

Each service provides health endpoints:
```bash
# API Service
curl http://localhost:8000/health

# Training Service  
curl http://localhost:8001/health

# Model Service
curl http://localhost:8002/health

# WebSocket Service
curl http://localhost:8003/health

# Monitoring Service
curl http://localhost:8004/health
```

## Troubleshooting

### Common Issues

1. **Service Dependencies Not Ready**
```bash
# Check service dependencies
docker-compose logs redis postgres
kubectl describe pod <pod-name> -n agentforge
```

2. **Out of Memory Errors**
```bash
# Increase memory limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 4G
```

3. **GPU Not Available**
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Check Kubernetes GPU support
kubectl describe nodes | grep nvidia.com/gpu
```

4. **Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Use different ports in docker-compose.yml
ports:
  - "8080:8000"  # Map to different host port
```

### Service Logs

```bash
# Docker Compose
docker-compose logs -f <service-name>

# Kubernetes
kubectl logs -f deployment/<service-name> -n agentforge
```

### Performance Tuning

1. **Database Connections**
```bash
# Increase PostgreSQL max connections
POSTGRES_INITDB_ARGS="-c max_connections=200"
```

2. **Redis Memory**
```bash
# Set Redis max memory
command: redis-server --appendonly yes --maxmemory 2gb
```

3. **Service Resources**
```bash
# Adjust service memory/CPU limits based on load
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "2Gi" 
    cpu: "2"
```

## Production Considerations

### Security

1. **Use TLS/SSL**
   - Configure HTTPS endpoints
   - Use TLS for inter-service communication
   - Secure WebSocket connections (WSS)

2. **Authentication**
   - Implement JWT authentication
   - Use service meshes for mTLS
   - Secure admin interfaces

3. **Secrets Management**
   - Use Kubernetes secrets or Docker secrets
   - Integrate with HashiCorp Vault
   - Rotate secrets regularly

### High Availability

1. **Load Balancing**
   - Multiple replicas per service
   - Health check-based routing
   - Circuit breakers for failures

2. **Data Persistence**
   - Database clustering
   - Model storage replication
   - Backup strategies

### Scaling Strategies

1. **Horizontal Pod Autoscaling (HPA)**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

2. **Vertical Pod Autoscaling (VPA)**
   - Automatic resource adjustment
   - Right-sizing containers
   - Cost optimization

This deployment architecture ensures the Agent Forge microservices can scale from development through production with proper observability and resilience patterns.