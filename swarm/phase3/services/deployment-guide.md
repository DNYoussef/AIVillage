# Fog Coordinator Services Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the 6-service fog coordinator architecture, transitioning from the monolithic `fog_coordinator.py` to independent, scalable services.

## Prerequisites

### System Requirements
- **Memory**: 2GB minimum, 4GB recommended per service cluster
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 50GB minimum, 100GB recommended  
- **Network**: 1Gbps minimum bandwidth
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Software Dependencies
```bash
# Container runtime
docker --version          # >= 20.10
docker-compose --version  # >= 2.0

# Kubernetes (optional)
kubectl version --client  # >= 1.24
helm version              # >= 3.8

# Python environment
python --version          # >= 3.9
pip --version            # >= 21.0
```

### External Services
- **Redis**: Message queue and caching (>= 6.2)
- **PostgreSQL**: Primary database (>= 13)
- **TimescaleDB**: Metrics storage (>= 2.8)
- **Prometheus**: Monitoring (>= 2.35)

## Deployment Options

### Option 1: Docker Compose (Recommended for Development)

**1. Prepare Environment**
```bash
# Clone the repository
git clone <repository-url>
cd AIVillage

# Create deployment directory
mkdir -p deployment/fog-services
cd deployment/fog-services

# Copy configuration files
cp ../../swarm/phase3/services/service-configuration.yaml ./config.yaml
```

**2. Create Docker Compose File**
```yaml
# docker-compose.yml
version: '3.8'

services:
  # Service Discovery & Configuration
  consul:
    image: consul:1.15
    ports:
      - "8500:8500"
    command: agent -dev -client 0.0.0.0
    
  # Message Queue
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      
  # Primary Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: fog_coordinator
      POSTGRES_USER: fog_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  # Metrics Database  
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_DB: fog_metrics
      POSTGRES_USER: metrics_user
      POSTGRES_PASSWORD: metrics_password
    ports:
      - "5433:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

  # Fog Services
  fog-orchestration:
    build: 
      context: .
      dockerfile: services/orchestration/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - CONFIG_PATH=/app/config.yaml
      - SERVICE_NAME=fog-orchestration
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - postgres
      
  fog-harvesting:
    build:
      context: .
      dockerfile: services/harvesting/Dockerfile
    ports:
      - "8081:8081"
    environment:
      - CONFIG_PATH=/app/config.yaml
      - SERVICE_NAME=fog-harvesting
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - fog-orchestration
      
  fog-marketplace:
    build:
      context: .
      dockerfile: services/marketplace/Dockerfile
    ports:
      - "8082:8082"
    environment:
      - CONFIG_PATH=/app/config.yaml
      - SERVICE_NAME=fog-marketplace
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - postgres
      - fog-orchestration
      
  fog-privacy:
    build:
      context: .
      dockerfile: services/privacy/Dockerfile
    ports:
      - "8083:8083"
    environment:
      - CONFIG_PATH=/app/config.yaml  
      - SERVICE_NAME=fog-privacy
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - fog-orchestration
      
  fog-tokenomics:
    build:
      context: .
      dockerfile: services/tokenomics/Dockerfile
    ports:
      - "8084:8084"
    environment:
      - CONFIG_PATH=/app/config.yaml
      - SERVICE_NAME=fog-tokenomics
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - postgres
      - fog-orchestration
      
  fog-system-stats:
    build:
      context: .
      dockerfile: services/system-stats/Dockerfile
    ports:
      - "8085:8085"
    environment:
      - CONFIG_PATH=/app/config.yaml
      - SERVICE_NAME=fog-system-stats
    volumes:
      - ./config.yaml:/app/config.yaml
    depends_on:
      - consul
      - redis
      - timescaledb
      - fog-orchestration

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  redis_data:
  postgres_data:
  timescale_data:
  prometheus_data:
  grafana_data:
```

**3. Deploy Services**
```bash
# Build and start services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f fog-orchestration

# Health check
curl http://localhost:8080/health
```

### Option 2: Kubernetes Deployment (Production)

**1. Create Namespace**
```bash
kubectl create namespace fog-coordinator
kubectl config set-context --current --namespace=fog-coordinator
```

**2. Deploy Configuration**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fog-config
  namespace: fog-coordinator
data:
  config.yaml: |
    # Include service-configuration.yaml content here
```

**3. Deploy Services with Helm**
```bash
# Create Helm chart
helm create fog-coordinator

# Install services
helm install fog-coordinator ./fog-coordinator \
  --set environment=production \
  --set replicas=2 \
  --set resources.requests.memory="512Mi" \
  --set resources.requests.cpu="250m"
```

**4. Service Deployment Manifests**
```yaml
# k8s/fog-orchestration-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fog-orchestration
  namespace: fog-coordinator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fog-orchestration
  template:
    metadata:
      labels:
        app: fog-orchestration
    spec:
      containers:
      - name: fog-orchestration
        image: fog-coordinator/orchestration:latest
        ports:
        - containerPort: 8080
        env:
        - name: CONFIG_PATH
          value: "/app/config.yaml"
        - name: SERVICE_NAME  
          value: "fog-orchestration"
        volumeMounts:
        - name: config
          mountPath: /app/config.yaml
          subPath: config.yaml
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: fog-config
---
apiVersion: v1
kind: Service
metadata:
  name: fog-orchestration-service
  namespace: fog-coordinator
spec:
  selector:
    app: fog-orchestration
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

## Migration Strategy

### Phase 1: Parallel Deployment (Week 1)

**1. Deploy New Services Alongside Monolith**
```bash
# Keep existing fog_coordinator.py running
python infrastructure/fog/edge/legacy_src/federated_learning/federated_coordinator.py

# Deploy new services with different ports
docker-compose up -d fog-orchestration fog-harvesting
```

**2. Feature Flag Configuration**
```python
# services/common/feature_flags.py
class FeatureFlags:
    USE_NEW_HARVESTING_SERVICE = os.getenv('USE_NEW_HARVESTING', 'false').lower() == 'true'
    USE_NEW_MARKETPLACE_SERVICE = os.getenv('USE_NEW_MARKETPLACE', 'false').lower() == 'true'
    
    @classmethod
    def is_service_enabled(cls, service_name: str) -> bool:
        flag_name = f'USE_NEW_{service_name.upper()}_SERVICE'
        return getattr(cls, flag_name, False)
```

**3. Traffic Splitting**
```python
# infrastructure/fog/coordinator_proxy.py
class FogCoordinatorProxy:
    def __init__(self):
        self.old_coordinator = DistributedFederatedLearning(...)
        self.new_services = ServiceContainer()
        
    async def coordinate_training_round(self, request):
        if FeatureFlags.USE_NEW_HARVESTING_SERVICE:
            harvesting = await self.new_services.get_service('harvesting')
            participants = await harvesting.select_participants(request.requirements)
        else:
            participants = await self.old_coordinator._select_participants_for_round(...)
            
        # Continue with hybrid approach
```

### Phase 2: Gradual Migration (Week 2-3)

**1. Service-by-Service Migration Schedule**
```
Day 1-2:  Deploy FogHarvestingService (lowest risk)
Day 3-4:  Deploy FogSystemStatsService (monitoring only)
Day 5-6:  Deploy FogPrivacyService (critical but isolated)
Day 7-8:  Deploy FogTokenomicsService (financial operations)
Day 9-10: Deploy FogMarketplaceService (complex interactions)
Day 11-12: Full FogOrchestrationService (coordination layer)
```

**2. Data Migration Scripts**
```python
# scripts/migrate_participant_data.py
async def migrate_participant_data():
    """Migrate participant data from monolith to new services."""
    
    # Extract from old system
    old_participants = await old_coordinator.get_all_participants()
    
    # Transform to new format
    new_format_participants = []
    for participant in old_participants:
        new_participant = DeviceCapability(
            device_id=participant.device_id,
            compute_gflops=participant.capabilities.compute_gflops,
            # ... map all fields
        )
        new_format_participants.append(new_participant)
    
    # Load into new service
    harvesting_service = await service_container.get_service('harvesting')
    await harvesting_service.import_devices(new_format_participants)
```

**3. Validation & Rollback Procedures**
```bash
#!/bin/bash
# scripts/validate_migration.sh

echo "Validating service migration..."

# Health checks
for service in orchestration harvesting marketplace privacy tokenomics stats; do
    response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:808${service#fog-}/health)
    if [ $response -ne 200 ]; then
        echo "ERROR: $service health check failed"
        exit 1
    fi
done

# Data consistency checks
python scripts/validate_data_consistency.py

# Performance benchmarks  
python scripts/benchmark_services.py

echo "Migration validation completed successfully"
```

### Phase 3: Complete Cutover (Week 4)

**1. Final Traffic Switch**
```python
# Update feature flags to route all traffic to new services
export USE_NEW_HARVESTING_SERVICE=true
export USE_NEW_MARKETPLACE_SERVICE=true
export USE_NEW_PRIVACY_SERVICE=true
export USE_NEW_TOKENOMICS_SERVICE=true
export USE_NEW_STATS_SERVICE=true

# Restart services to pick up new flags
docker-compose restart
```

**2. Decommission Monolith**
```bash
# Stop old coordinator
systemctl stop fog-coordinator-monolith

# Archive old code
mv infrastructure/fog/edge/legacy_src/federated_learning/federated_coordinator.py \
   infrastructure/fog/edge/legacy_src/federated_learning/federated_coordinator.py.deprecated

# Clean up old dependencies
pip uninstall -y unused-monolith-packages
```

## Monitoring & Observability

### Service Health Monitoring
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fog-services'
    static_configs:
      - targets: 
        - 'fog-orchestration:8080'
        - 'fog-harvesting:8081' 
        - 'fog-marketplace:8082'
        - 'fog-privacy:8083'
        - 'fog-tokenomics:8084'
        - 'fog-system-stats:8085'
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'infrastructure'
    static_configs:
      - targets:
        - 'redis:6379'
        - 'postgres:5432'
        - 'consul:8500'
```

### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "Fog Coordinator Services",
    "panels": [
      {
        "title": "Service Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"fog-services\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules
```yaml
# alerting-rules.yml
groups:
  - name: fog-coordinator
    rules:
      - alert: ServiceDown
        expr: up{job="fog-services"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Fog service {{ $labels.instance }} is down"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.service }}"
```

## Performance Optimization

### Resource Allocation
```yaml
# k8s resource limits
resources:
  orchestration:
    requests:
      memory: "256Mi"
      cpu: "250m"
    limits:
      memory: "512Mi"
      cpu: "500m"
      
  harvesting:
    requests:
      memory: "384Mi"
      cpu: "300m"
    limits:
      memory: "768Mi"
      cpu: "600m"
      
  marketplace:
    requests:
      memory: "512Mi"
      cpu: "400m"
    limits:
      memory: "1Gi"
      cpu: "800m"
```

### Auto-scaling Configuration
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fog-services-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fog-marketplace
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

**1. Service Discovery Failures**
```bash
# Check Consul connectivity
curl http://localhost:8500/v1/catalog/services

# Verify service registration
consul catalog services
```

**2. Database Connection Issues**
```bash
# Test PostgreSQL connection
psql -h localhost -p 5432 -U fog_user -d fog_coordinator

# Check connection pool status
curl http://localhost:8080/debug/db-stats
```

**3. Event Bus Problems**
```bash
# Check Redis connectivity
redis-cli ping

# Monitor event queue
redis-cli MONITOR
```

**4. Performance Issues**
```bash
# Check service metrics
curl http://localhost:8080/metrics | grep -E "(cpu|memory|requests)"

# Profile service performance
curl http://localhost:8080/debug/pprof/profile > profile.pb.gz
```

### Recovery Procedures

**1. Service Restart**
```bash
# Restart specific service
docker-compose restart fog-harvesting

# Kubernetes restart
kubectl rollout restart deployment/fog-harvesting
```

**2. Rollback to Previous Version**
```bash
# Docker rollback
docker-compose down
docker-compose up -d --force-recreate

# Kubernetes rollback
kubectl rollout undo deployment/fog-harvesting
```

**3. Emergency Fallback to Monolith**
```bash
# Activate emergency feature flags
export EMERGENCY_FALLBACK_MODE=true
export USE_NEW_*_SERVICE=false

# Restart proxy with fallback config
systemctl restart fog-coordinator-proxy
```

This comprehensive deployment guide ensures a smooth transition from the monolithic fog coordinator to the new 6-service architecture while maintaining system reliability and performance.