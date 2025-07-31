# AIVillage Production Deployment System

This directory contains the complete production deployment automation pipeline for the AIVillage system, including CI/CD workflows, containerization, Kubernetes manifests, and deployment scripts.

## 🏗️ Architecture Overview

The AIVillage deployment system uses a modern, cloud-native architecture with:

- **Blue-Green Deployment Strategy**: Zero-downtime deployments with automatic rollback
- **Kubernetes Orchestration**: Container orchestration with auto-scaling and health monitoring
- **Helm Package Management**: Templated deployments with environment-specific configurations
- **Multi-Service Architecture**: Microservices with independent scaling and deployment
- **Comprehensive Monitoring**: Prometheus, Grafana, and custom health checks

## 📁 Directory Structure

```
deploy/
├── docker/                 # Docker images for all services
│   ├── Dockerfile.gateway
│   ├── Dockerfile.hyperag-mcp
│   ├── Dockerfile.twin
│   ├── Dockerfile.credits-api
│   ├── Dockerfile.credits-worker
│   ├── Dockerfile.agent-forge
│   ├── Dockerfile.compression-service
│   ├── Dockerfile.evolution-engine
│   └── Dockerfile.mesh-network
├── k8s/                    # Kubernetes manifests
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── postgres.yaml
│   ├── redis.yaml
│   ├── neo4j.yaml
│   ├── qdrant.yaml
│   ├── gateway.yaml
│   └── hyperag-mcp.yaml
├── helm/                   # Helm charts
│   └── aivillage/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── values-production.yaml
│       ├── values-staging.yaml
│       └── templates/
├── scripts/                # Deployment automation scripts
│   ├── deploy.py
│   ├── smoke_tests.py
│   ├── health_check.py
│   ├── readiness_tests.py
│   └── production_verification.py
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Required Tools**:
   - `kubectl` (v1.28+)
   - `helm` (v3.12+)
   - `docker` (v24.0+)
   - Python 3.11+ with `aiohttp`, `psutil`

2. **Access Requirements**:
   - Kubernetes cluster access (staging and production)
   - Container registry push permissions (GitHub Container Registry)
   - Environment secrets configured in CI/CD

### Deploy to Staging

```bash
# Using the orchestration script
python deploy/scripts/deploy.py --environment staging --image-tag latest

# Or using Helm directly
helm upgrade --install aivillage-staging ./deploy/helm/aivillage \
  --namespace aivillage-staging \
  --create-namespace \
  --values deploy/helm/aivillage/values-staging.yaml \
  --wait
```

### Deploy to Production

```bash
# Production deployment with blue-green strategy
python deploy/scripts/deploy.py --environment production --image-tag v1.2.3

# Or trigger via GitHub Actions
gh workflow run production-deploy.yml -f environment=production
```

## 🔄 CI/CD Pipeline

The deployment pipeline is defined in `.github/workflows/production-deploy.yml` with the following stages:

### 1. Security & Quality Gates
- **Bandit** security scanning
- **Safety** dependency vulnerability checks
- **Semgrep** SAST analysis
- Quality gate validation

### 2. Pre-Deployment Testing
- Comprehensive test suite execution
- Integration tests with real databases
- Performance benchmarks
- Coverage analysis

### 3. Container Image Building
- Multi-service Docker image builds
- Multi-architecture support (AMD64, ARM64)
- Layer caching optimization
- Security scanning of images

### 4. Staging Deployment
- Automatic deployment to staging
- Smoke tests execution
- Health check validation
- Performance verification

### 5. Production Deployment (Blue-Green)
- Blue-green deployment strategy
- Production readiness tests
- Traffic switching automation
- Post-deployment verification

### 6. Monitoring & Notifications
- Deployment status notifications (Slack)
- Automatic rollback on failure
- Monitoring integration
- Performance tracking

## 🏭 Production Services

### Core Application Services

1. **Gateway Service** (Port 8000)
   - API gateway and load balancer
   - Rate limiting and authentication
   - Request routing and transformation

2. **Digital Twin Service** (Port 8001)
   - AI agent interaction endpoint
   - Model serving and inference
   - Session management

3. **Credits API** (Port 8002)
   - Credit system management
   - User account tracking
   - Payment processing integration

4. **HyperRAG MCP Server** (Port 8765)
   - Model Context Protocol server
   - RAG system coordination
   - Knowledge graph integration

5. **Agent Forge** (Port 8003)
   - Agent creation and management
   - Training pipeline coordination
   - Model evolution tracking

6. **Compression Service** (Port 8004)
   - Model compression and optimization
   - Mobile deployment preparation
   - Performance benchmarking

7. **Evolution Engine** (Port 8005)
   - Model evolution and merging
   - Genetic algorithm coordination
   - Performance optimization

8. **Mesh Network Service** (Port 8006)
   - Distributed communication
   - Peer-to-peer coordination
   - Network resilience

### Database Services

1. **PostgreSQL** (Port 5432)
   - Primary application database
   - User data and transactions
   - Persistent storage

2. **Redis** (Port 6379)
   - Caching and session storage
   - Real-time data processing
   - Message queue

3. **Neo4j** (Port 7687/7474)
   - Knowledge graph database
   - Relationship modeling
   - Graph-based queries

4. **Qdrant** (Port 6333/6334)
   - Vector database for embeddings
   - Similarity search
   - AI model storage

### Monitoring Services

1. **Prometheus** (Port 9090)
   - Metrics collection and storage
   - Alert rule evaluation
   - Time-series database

2. **Grafana** (Port 3000)
   - Visualization and dashboards
   - Alert management
   - Performance monitoring

## 📊 Health Monitoring

### Health Check Endpoints

All services expose health check endpoints:

- `/health` or `/healthz` - Basic health status
- `/ready` - Readiness for traffic
- `/metrics` - Prometheus metrics

### Monitoring Dashboards

1. **System Overview**
   - Service status and uptime
   - Resource utilization
   - Error rates and latency

2. **Database Performance**
   - Connection pool status
   - Query performance metrics
   - Storage utilization

3. **Application Metrics**
   - Request rates and response times
   - Model inference performance
   - User activity metrics

## 🔒 Security Configuration

### Pod Security Standards

- **Non-root containers**: All containers run as non-root users
- **Read-only filesystems**: Root filesystems are read-only
- **No privilege escalation**: Containers cannot escalate privileges
- **Dropped capabilities**: All unnecessary Linux capabilities dropped

### Network Security

- **Network policies**: Traffic isolation between namespaces
- **TLS encryption**: All inter-service communication encrypted
- **Ingress control**: WAF and rate limiting at ingress
- **Secret management**: Kubernetes secrets with rotation

### Compliance

- **GDPR compliance**: Data protection and privacy controls
- **SOC 2 Type II**: Security and availability controls
- **Regular audits**: Automated security scanning and reporting

## 🛠️ Troubleshooting

### Common Issues

1. **Pod Startup Failures**
   ```bash
   kubectl describe pod <pod-name> -n aivillage-production
   kubectl logs <pod-name> -n aivillage-production
   ```

2. **Service Connectivity Issues**
   ```bash
   kubectl get svc -n aivillage-production
   kubectl port-forward svc/<service-name> 8080:8080 -n aivillage-production
   ```

3. **Database Connection Problems**
   ```bash
   kubectl exec -it statefulset/aivillage-postgres -n aivillage-production -- psql -U aivillage_user
   ```

4. **Resource Constraints**
   ```bash
   kubectl top pods -n aivillage-production
   kubectl describe nodes
   ```

### Log Analysis

```bash
# Application logs
kubectl logs -f deployment/aivillage-gateway -n aivillage-production

# Database logs
kubectl logs -f statefulset/aivillage-postgres -n aivillage-production

# System events
kubectl get events -n aivillage-production --sort-by='.lastTimestamp'
```

### Performance Debugging

```bash
# Run health checks
python deploy/scripts/health_check.py --environment production

# Run smoke tests
python deploy/scripts/smoke_tests.py --environment production --namespace aivillage-production

# Run full verification
python deploy/scripts/production_verification.py --environment production --slot blue
```

## 🔄 Rollback Procedures

### Automatic Rollback

The system includes automatic rollback triggers:
- Health check failures post-deployment
- Error rate thresholds exceeded
- Response time degradation

### Manual Rollback

```bash
# Using Helm
helm rollback aivillage-production -n aivillage-production

# Using the orchestration script
python deploy/scripts/deploy.py --environment production --action rollback

# Blue-green traffic switch
kubectl patch service aivillage-active -p '{"spec":{"selector":{"slot":"blue"}}}' -n aivillage-production
```

## 📈 Scaling Configuration

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aivillage-gateway-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aivillage-gateway
  minReplicas: 3
  maxReplicas: 20
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

### Vertical Pod Autoscaling

Enabled for all services with resource recommendations and automatic updates.

## 🔧 Configuration Management

### Environment Variables

Configuration is managed through:
- **ConfigMaps**: Non-sensitive configuration
- **Secrets**: Sensitive data (passwords, API keys)
- **Environment-specific values**: Helm values files

### Secret Rotation

Secrets are rotated automatically:
- Database passwords: Monthly
- API keys: Quarterly
- TLS certificates: Automatically via cert-manager

## 📞 Support and Maintenance

### Monitoring Alerts

Critical alerts are configured for:
- Service downtime (>1 minute)
- High error rates (>5%)
- Resource exhaustion (>90% CPU/Memory)
- Database connectivity issues
- Security policy violations

### Maintenance Windows

- **Staging**: No maintenance window (continuous deployment)
- **Production**: Weekly maintenance window (Sunday 2-4 AM UTC)
- **Emergency patches**: Immediate deployment capability

### Contact Information

- **Primary On-call**: Slack #aivillage-alerts
- **Secondary Escalation**: Email alerts@aivillage.com
- **Emergency Contact**: +1-XXX-XXX-XXXX

---

## 📋 Deployment Checklist

### Pre-Deployment

- [ ] All tests passing in CI/CD
- [ ] Security scans completed
- [ ] Performance benchmarks validated
- [ ] Change approval obtained
- [ ] Rollback plan confirmed

### During Deployment

- [ ] Monitor deployment progress
- [ ] Validate health checks
- [ ] Verify service connectivity
- [ ] Check resource utilization
- [ ] Confirm monitoring alerts

### Post-Deployment

- [ ] Run production verification tests
- [ ] Validate user-facing functionality
- [ ] Monitor error rates and performance
- [ ] Confirm monitoring and alerting
- [ ] Document any issues or changes

---

For detailed technical support or questions about the deployment system, please refer to the [AIVillage Documentation](../docs/) or contact the development team.