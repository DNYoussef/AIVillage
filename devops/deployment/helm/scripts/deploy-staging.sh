#!/bin/bash
set -euo pipefail

# AIVillage Helm Deployment Script - Staging Environment
# Usage: ./deploy-staging.sh [RELEASE_NAME]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_DIR="$(dirname "$SCRIPT_DIR")/aivillage"
RELEASE_NAME="${1:-aivillage-staging}"
NAMESPACE="aivillage-staging"

echo "🚀 Deploying AIVillage to Staging Environment"
echo "Release Name: $RELEASE_NAME"
echo "Namespace: $NAMESPACE"
echo "Chart Directory: $CHART_DIR"
echo ""

# Check prerequisites
if ! command -v helm &> /dev/null; then
    echo "❌ Error: helm command not found. Please install Helm 3.8+"
    exit 1
fi

if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl command not found. Please install kubectl"
    exit 1
fi

# Check Kubernetes connection
echo "🔍 Checking Kubernetes connection..."
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: Unable to connect to Kubernetes cluster"
    echo "Please check your kubeconfig and try again"
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

# Generate secure passwords if not provided
echo "🔐 Generating secure passwords..."
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-$(openssl rand -base64 32)}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(openssl rand -base64 32)}"
HYPERAG_JWT_SECRET="${HYPERAG_JWT_SECRET:-$(openssl rand -base64 64)}"

# Create namespace if it doesn't exist
echo "🏗️  Creating namespace: $NAMESPACE"
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Label namespace for monitoring
kubectl label namespace "$NAMESPACE" name="$NAMESPACE" --overwrite

# Deploy the chart
echo "🎯 Deploying Helm chart..."
helm upgrade --install "$RELEASE_NAME" . \
  --namespace "$NAMESPACE" \
  --values values-staging.yaml \
  --set secrets.postgresPassword="$POSTGRES_PASSWORD" \
  --set secrets.redisPassword="$REDIS_PASSWORD" \
  --set secrets.neo4jPassword="$NEO4J_PASSWORD" \
  --set secrets.grafanaPassword="$GRAFANA_PASSWORD" \
  --set secrets.hyperhagJwtSecret="$HYPERAG_JWT_SECRET" \
  --set environment="staging" \
  --set debug="false" \
  --set logLevel="INFO" \
  --wait \
  --timeout=15m

if [ $? -eq 0 ]; then
    echo "✅ Deployment completed successfully!"
else
    echo "❌ Deployment failed!"
    exit 1
fi

# Show deployment status
echo ""
echo "📊 Deployment Status:"
kubectl get pods -n "$NAMESPACE"
echo ""

# Show services and ingress
echo "🌐 Services:"
kubectl get svc -n "$NAMESPACE"
echo ""

echo "🔗 Ingress:"
kubectl get ingress -n "$NAMESPACE"
echo ""

# Show useful commands
echo "💡 Useful commands:"
echo ""
echo "# Watch pod status:"
echo "kubectl get pods -n $NAMESPACE -w"
echo ""
echo "# View logs:"
echo "kubectl logs -l app.kubernetes.io/name=aivillage-gateway -n $NAMESPACE -f"
echo ""
echo "# Port-forward to access services locally:"
echo "kubectl port-forward svc/$RELEASE_NAME-gateway 8080:8000 -n $NAMESPACE"
echo "kubectl port-forward svc/$RELEASE_NAME-twin 8081:8001 -n $NAMESPACE"
echo ""
echo "# Access Grafana (if enabled):"
echo "kubectl port-forward svc/$RELEASE_NAME-grafana 3000:80 -n $NAMESPACE"
echo "Username: admin"
echo "Password: $GRAFANA_PASSWORD"
echo ""

# Save passwords to file for reference
SECRETS_FILE="$HOME/.aivillage-staging-secrets"
cat > "$SECRETS_FILE" << EOF
# AIVillage Staging Environment Secrets
# Generated on $(date)

export POSTGRES_PASSWORD="$POSTGRES_PASSWORD"
export REDIS_PASSWORD="$REDIS_PASSWORD"
export NEO4J_PASSWORD="$NEO4J_PASSWORD"
export GRAFANA_PASSWORD="$GRAFANA_PASSWORD"
export HYPERAG_JWT_SECRET="$HYPERAG_JWT_SECRET"

# To load these secrets:
# source $SECRETS_FILE
EOF

chmod 600 "$SECRETS_FILE"
echo "🔒 Secrets saved to: $SECRETS_FILE"
echo ""

# Health check
echo "🏥 Performing health check..."
sleep 30

# Wait for gateway to be ready
echo "⏳ Waiting for gateway to be ready..."
kubectl wait --for=condition=available deployment/$RELEASE_NAME-gateway -n "$NAMESPACE" --timeout=300s

if [ $? -eq 0 ]; then
    echo "✅ Gateway is ready!"

    # Try to access health endpoint
    echo "🔍 Testing health endpoint..."
    if kubectl port-forward svc/$RELEASE_NAME-gateway 8080:8000 -n "$NAMESPACE" > /dev/null 2>&1 &
    then
        PF_PID=$!
        sleep 5
        if curl -f http://localhost:8080/healthz > /dev/null 2>&1; then
            echo "✅ Health check passed!"
        else
            echo "⚠️  Health check failed - service may still be starting"
        fi
        kill $PF_PID > /dev/null 2>&1 || true
    fi
else
    echo "❌ Gateway failed to become ready"
    echo "Check pod logs:"
    kubectl logs -l app.kubernetes.io/name=aivillage-gateway -n "$NAMESPACE" --tail=50
fi

echo ""
echo "🎉 Staging deployment complete!"
echo "🌍 Access your staging environment at: https://staging-api.aivillage.com"
echo ""
