#!/bin/bash
set -euo pipefail

# AIVillage Helm Deployment Script - Production Environment
# Usage: ./deploy-production.sh [RELEASE_NAME]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(dirname "$SCRIPT_DIR")/aivillage"
RELEASE_NAME="${1:-aivillage-prod}"
NAMESPACE="aivillage-production"

echo "🚀 Deploying AIVillage to Production Environment"
echo "Release Name: $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Chart Directory: $CHART_DIR"
echo ""

# Production deployment warning
echo "⚠️  PRODUCTION DEPLOYMENT"
echo "This will deploy to the production environment."
echo "Make sure you have:"
echo "  - Reviewed the configuration"
echo "  - Tested in staging"
echo "  - Have database backups"
echo "  - Have rollback plan ready"
echo ""
read -p "Do you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Check prerequisites
if ! command -v helm &> /dev/null; then
    echo "❌ Error: helm command not found. Please install Helm 3.8+"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl command not found. Please install kubectl"
    exit 1
fi

# Check for required environment variables
required_vars=(
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "NEO4J_PASSWORD"
    "GRAFANA_PASSWORD"
    "HYPERAG_JWT_SECRET"
)

missing_vars=()
for var in "${required_vars[@]}"; do
    if [ -z "${!var:-}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    echo "❌ Error: Missing required environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Set these variables or load from your secrets management system:"
    echo "export POSTGRES_PASSWORD=\"your-secure-password\""
    echo "export REDIS_PASSWORD=\"your-secure-password\""
    echo "# ... etc"
    exit 1
fi

# Check Kubernetes connection
echo "🔍 Checking Kubernetes connection..."
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: Unable to connect to Kubernetes cluster"
    echo "Please check your kubeconfig and try again"
    exit 1
fi

# Verify we're connected to the right cluster
echo "📍 Current Kubernetes context:"
kubectl config current-context
echo ""
read -p "Is this the correct production cluster? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Please switch to the correct context and try again."
    echo "kubectl config use-context <production-context>"
    exit 1
fi

# Add Helm repositories
echo "📦 Adding Helm repositories..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Update chart dependencies
echo "🔄 Updating chart dependencies..."
cd "$CHART_DIR"
helm dependency update

# Create namespace if it doesn't exist
echo "🏗️  Creating namespace: $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Label namespace for monitoring and backup
kubectl label namespace "$NAMESPACE" \
  name="$NAMESPACE" \
  environment="production" \
  backup="required" \
  monitoring="enabled" \
  --overwrite

# Create production secrets
echo "🔐 Creating production secrets..."
kubectl create secret generic "$RELEASE_NAME-secrets" \
  --namespace="$NAMESPACE" \
  --from-literal=postgres-password="$POSTGRES_PASSWORD" \
  --from-literal=redis-password="$REDIS_PASSWORD" \
  --from-literal=neo4j-password="$NEO4J_PASSWORD" \
  --from-literal=neo4j-auth="neo4j/$NEO4J_PASSWORD" \
  --from-literal=grafana-password="$GRAFANA_PASSWORD" \
  --from-literal=hyperag-jwt-secret="$HYPERAG_JWT_SECRET" \
  ${OPENAI_API_KEY:+--from-literal=openai-api-key="$OPENAI_API_KEY"} \
  ${ANTHROPIC_API_KEY:+--from-literal=anthropic-api-key="$ANTHROPIC_API_KEY"} \
  --dry-run=client -o yaml | kubectl apply -f -

# Label secret for backup
kubectl label secret "$RELEASE_NAME-secrets" -n "$NAMESPACE" \
  backup="required" \
  environment="production"

# Dry-run first for validation
echo "🧪 Running dry-run validation..."
helm upgrade --install "$RELEASE_NAME" . \
  --namespace "$NAMESPACE" \
  --values values-production.yaml \
  --set environment="production" \
  --set debug="false" \
  --set logLevel="WARNING" \
  --dry-run

if [ $? -ne 0 ]; then
    echo "❌ Dry-run validation failed!"
    exit 1
fi

echo "✅ Dry-run validation passed!"
echo ""

# Final confirmation
echo "🎯 Ready to deploy to production with the following configuration:"
echo "  - Release: $RELEASE_NAME"
echo "  - Namespace: $NAMESPACE"
echo "  - Environment: production"
echo "  - Debug: false"
echo "  - Log Level: WARNING"
echo ""
read -p "Proceed with production deployment? (YES/no): " -r
if [[ ! $REPLY =~ ^YES$ ]]; then
    echo "Deployment cancelled. Type 'YES' to confirm production deployment."
    exit 0
fi

# Deploy the chart
echo "🎯 Deploying to production..."
helm upgrade --install "$RELEASE_NAME" . \
  --namespace "$NAMESPACE" \
  --values values-production.yaml \
  --set environment="production" \
  --set debug="false" \
  --set logLevel="WARNING" \
  --wait \
  --timeout=20m

if [ $? -eq 0 ]; then
    echo "✅ Production deployment completed successfully!"
else
    echo "❌ Production deployment failed!"

    # Show recent events for troubleshooting
    echo "Recent events:"
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10

    echo ""
    echo "🔄 Consider rolling back if needed:"
    echo "helm rollback $RELEASE_NAME -n $NAMESPACE"
    exit 1
fi

# Show deployment status
echo ""
echo "📊 Production Deployment Status:"
kubectl get pods -n "$NAMESPACE"
echo ""

# Show services and ingress
echo "🌐 Services:"
kubectl get svc -n "$NAMESPACE"
echo ""

echo "🔗 Ingress:"
kubectl get ingress -n "$NAMESPACE"
echo ""

# Production health check
echo "🏥 Performing comprehensive health check..."

# Wait for all deployments to be ready
echo "⏳ Waiting for all services to be ready..."
deployments=(
    "$RELEASE_NAME-gateway"
    "$RELEASE_NAME-twin"
    "$RELEASE_NAME-hyperrag-mcp"
)

for deployment in "${deployments[@]}"; do
    echo "  Waiting for $deployment..."
    kubectl wait --for=condition=available "deployment/$deployment" -n "$NAMESPACE" --timeout=600s

    if [ $? -eq 0 ]; then
        echo "  ✅ $deployment is ready"
    else
        echo "  ❌ $deployment failed to become ready"
        kubectl describe deployment "$deployment" -n "$NAMESPACE"
    fi
done

# Test health endpoints
echo ""
echo "🔍 Testing health endpoints..."
if kubectl get svc "$RELEASE_NAME-gateway" -n "$NAMESPACE" > /dev/null 2>&1; then
    echo "✅ Gateway service is available"
else
    echo "❌ Gateway service is not available"
fi

# Show resource usage
echo ""
echo "📈 Resource Usage:"
kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available yet"
echo ""

# Show useful production commands
echo "💡 Production Management Commands:"
echo ""
echo "# Monitor deployment:"
echo "kubectl get pods -n $NAMESPACE -w"
echo ""
echo "# View logs:"
echo "kubectl logs -l app.kubernetes.io/name=aivillage-gateway -n $NAMESPACE -f"
echo ""
echo "# Check resource usage:"
echo "kubectl top pods -n $NAMESPACE"
echo "kubectl top nodes"
echo ""
echo "# Scale services:"
echo "kubectl scale deployment $RELEASE_NAME-gateway --replicas=5 -n $NAMESPACE"
echo ""
echo "# Rollback if needed:"
echo "helm history $RELEASE_NAME -n $NAMESPACE"
echo "helm rollback $RELEASE_NAME [REVISION] -n $NAMESPACE"
echo ""
echo "# Access Grafana:"
echo "kubectl port-forward svc/$RELEASE_NAME-grafana 3000:80 -n $NAMESPACE"
echo "Username: admin"
echo "Password: [from environment variable]"
echo ""

# Production monitoring setup reminder
echo "📊 Post-Deployment Checklist:"
echo "  □ Verify all services are healthy"
echo "  □ Check application logs for errors"
echo "  □ Validate database connections"
echo "  □ Test API endpoints"
echo "  □ Configure monitoring alerts"
echo "  □ Setup backup schedules"
echo "  □ Update DNS/CDN configurations"
echo "  □ Notify stakeholders of deployment"
echo ""

echo "🎉 Production deployment complete!"
echo "🌍 Your production AIVillage is now running!"
echo ""
