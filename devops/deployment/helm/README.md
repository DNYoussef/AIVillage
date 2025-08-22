# AIVillage Helm Chart

This Helm chart deploys the AIVillage distributed AI platform to Kubernetes, providing a complete stack including:

- **Core Services**: Gateway API, Digital Twin Server, HyperRAG MCP Server
- **Databases**: PostgreSQL, Redis, Neo4j, Qdrant
- **Monitoring**: Grafana, Prometheus (optional)
- **Security**: RBAC, Pod Security Contexts, Network Policies

## Prerequisites

- Kubernetes 1.19+
- Helm 3.8+
- PV provisioner support in the underlying infrastructure (for persistent storage)
- Ingress controller (nginx recommended)
- cert-manager (for TLS certificates)

## Installation

### Add Helm Dependencies

First, add the required Helm repositories and update dependencies:

```bash
# Add Bitnami repository for PostgreSQL and Redis
helm repo add bitnami https://charts.bitnami.com/bitnami

# Add Grafana repository for monitoring
helm repo add grafana https://grafana.github.io/helm-charts

# Update repositories
helm repo update

# Navigate to the chart directory
cd deploy/helm/aivillage

# Update dependencies
helm dependency update
```

### Quick Start (Development)

```bash
# Install with default values (development configuration)
helm install aivillage-dev . \
  --create-namespace \
  --namespace aivillage-dev \
  --set environment=development \
  --set debug=true

# Check deployment status
kubectl get pods -n aivillage-dev
```

### Staging Deployment

```bash
# Install staging environment
helm install aivillage-staging . \
  --create-namespace \
  --namespace aivillage-staging \
  --values values-staging.yaml \
  --set secrets.postgresPassword="$(openssl rand -base64 32)" \
  --set secrets.redisPassword="$(openssl rand -base64 32)" \
  --set secrets.neo4jPassword="$(openssl rand -base64 32)" \
  --set secrets.grafanaPassword="$(openssl rand -base64 32)"

# Verify deployment
kubectl get all -n aivillage-staging
```

### Production Deployment

```bash
# Create production namespace
kubectl create namespace aivillage-production

# Create secrets from environment variables or CI/CD system
kubectl create secret generic aivillage-secrets \
  --namespace aivillage-production \
  --from-literal=postgres-password="${POSTGRES_PASSWORD}" \
  --from-literal=redis-password="${REDIS_PASSWORD}" \
  --from-literal=neo4j-password="${NEO4J_PASSWORD}" \
  --from-literal=openai-api-key="${OPENAI_API_KEY}" \
  --from-literal=anthropic-api-key="${ANTHROPIC_API_KEY}" \
  --from-literal=grafana-password="${GRAFANA_PASSWORD}"

# Install production release
helm install aivillage-prod . \
  --namespace aivillage-production \
  --values values-production.yaml \
  --set ingress.hosts[0].host=api.yourdomain.com \
  --set ingress.tls[0].secretName=aivillage-prod-tls \
  --set ingress.tls[0].hosts[0]=api.yourdomain.com

# Wait for deployment to be ready
kubectl rollout status deployment/aivillage-prod-gateway -n aivillage-production
kubectl rollout status deployment/aivillage-prod-twin -n aivillage-production
```

## Configuration

### Values Files

The chart includes three values files for different environments:

- `values.yaml` - Base configuration with defaults
- `values-staging.yaml` - Staging environment overrides
- `values-production.yaml` - Production environment overrides

### Key Configuration Options

```yaml
# Environment settings
environment: "production"  # production, staging, development
debug: false
logLevel: "INFO"

# Scaling
gateway:
  replicaCount: 3
twin:
  replicaCount: 2

# Storage
global:
  storageClass: "fast-ssd"

postgresql:
  primary:
    persistence:
      size: 100Gi

# Security
security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 65534
    fsGroup: 65534

# Ingress
ingress:
  enabled: true
  hosts:
    - host: api.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
```

## Monitoring

### Grafana Dashboard

Access Grafana dashboard:

```bash
# Port-forward to access locally
kubectl port-forward svc/aivillage-prod-grafana 3000:80 -n aivillage-production

# Or access via ingress (if configured)
open https://grafana.yourdomain.com
```

Default credentials:
- Username: `admin`
- Password: Value from `secrets.grafanaPassword`

### Health Checks

Check service health:

```bash
# Check all services
kubectl get pods -n aivillage-production

# Check specific service logs
kubectl logs -l app.kubernetes.io/name=aivillage-gateway -n aivillage-production

# Port-forward to access health endpoints
kubectl port-forward svc/aivillage-prod-gateway 8080:8000 -n aivillage-production
curl http://localhost:8080/healthz
```

## Database Access

### PostgreSQL

```bash
# Connect to PostgreSQL
kubectl exec -it deployment/aivillage-prod-postgresql -n aivillage-production -- psql -U aivillage_user -d aivillage

# Port-forward for external access
kubectl port-forward svc/aivillage-prod-postgresql 5432:5432 -n aivillage-production
```

### Neo4j

```bash
# Access Neo4j browser
kubectl port-forward svc/aivillage-prod-neo4j 7474:7474 -n aivillage-production
open http://localhost:7474

# Connect via Bolt protocol
kubectl port-forward svc/aivillage-prod-neo4j 7687:7687 -n aivillage-production
```

### Redis

```bash
# Connect to Redis
kubectl exec -it deployment/aivillage-prod-redis-master -n aivillage-production -- redis-cli

# Port-forward for external access
kubectl port-forward svc/aivillage-prod-redis-master 6379:6379 -n aivillage-production
```

## Troubleshooting

### Common Issues

#### Pods Stuck in Pending State

Check resource quotas and node capacity:

```bash
kubectl describe nodes
kubectl get events -n aivillage-production --sort-by='.lastTimestamp'
```

#### Database Connection Issues

Verify database connectivity:

```bash
# Check database pod status
kubectl get pods -l app.kubernetes.io/component=database -n aivillage-production

# Check service endpoints
kubectl get endpoints -n aivillage-production

# Test database connection from gateway pod
kubectl exec -it deployment/aivillage-prod-gateway -n aivillage-production -- nc -zv aivillage-prod-postgresql 5432
```

#### Ingress Not Working

Check ingress controller and certificates:

```bash
# Check ingress status
kubectl get ingress -n aivillage-production
kubectl describe ingress aivillage-prod -n aivillage-production

# Check cert-manager certificates
kubectl get certificates -n aivillage-production
kubectl describe certificate aivillage-prod-tls -n aivillage-production
```

### Useful Commands

```bash
# View all resources
kubectl get all -n aivillage-production

# Check resource usage
kubectl top pods -n aivillage-production
kubectl top nodes

# View recent events
kubectl get events -n aivillage-production --sort-by='.lastTimestamp'

# Scale deployment
kubectl scale deployment aivillage-prod-gateway --replicas=5 -n aivillage-production

# Rolling restart
kubectl rollout restart deployment/aivillage-prod-gateway -n aivillage-production
```

## Upgrading

### Helm Upgrade

```bash
# Update dependencies
helm dependency update

# Upgrade staging
helm upgrade aivillage-staging . \
  --namespace aivillage-staging \
  --values values-staging.yaml

# Upgrade production with confirmation
helm upgrade aivillage-prod . \
  --namespace aivillage-production \
  --values values-production.yaml \
  --wait \
  --timeout=10m

# Rollback if needed
helm rollback aivillage-prod 1 -n aivillage-production
```

### Database Migrations

```bash
# Run database migrations (if needed)
kubectl create job aivillage-migrate-$(date +%s) \
  --from=deployment/aivillage-prod-gateway \
  -n aivillage-production \
  -- python manage.py migrate
```

## Backup and Recovery

### Database Backup

```bash
# PostgreSQL backup
kubectl exec deployment/aivillage-prod-postgresql -n aivillage-production -- \
  pg_dump -U aivillage_user aivillage > backup-$(date +%Y%m%d).sql

# Neo4j backup
kubectl exec deployment/aivillage-prod-neo4j -n aivillage-production -- \
  neo4j-admin dump --database=neo4j --to=/tmp/neo4j-backup-$(date +%Y%m%d).dump
```

### Persistent Volume Backup

```bash
# List persistent volumes
kubectl get pv

# Create volume snapshots (if supported by storage class)
kubectl create volumesnapshot aivillage-postgres-snap \
  --claim=aivillage-prod-postgresql \
  -n aivillage-production
```

## Security Considerations

### Network Policies

Consider implementing network policies to restrict pod-to-pod communication:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: aivillage-netpol
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/instance: aivillage-prod
  policyTypes:
  - Ingress
  - Egress
```

### Pod Security Standards

The chart implements Pod Security Standards:

```yaml
security:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 65534
    fsGroup: 65534
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
```

### Secret Management

For production, consider using external secret management:

- HashiCorp Vault
- AWS Secrets Manager
- Azure Key Vault
- GCP Secret Manager

## Performance Tuning

### Resource Limits

Adjust resource requests and limits based on workload:

```yaml
gateway:
  resources:
    requests:
      memory: "512Mi"
      cpu: "500m"
    limits:
      memory: "1Gi"
      cpu: "1"
```

### Horizontal Pod Autoscaling

Enable HPA for automatic scaling:

```bash
kubectl autoscale deployment aivillage-prod-gateway \
  --min=3 --max=10 --cpu-percent=70 \
  -n aivillage-production
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/DNYoussef/AIVillage/issues
- Documentation: https://docs.aivillage.io
