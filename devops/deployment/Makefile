# AIVillage Deployment Makefile

# Variables
ENVIRONMENT ?= staging
IMAGE_TAG ?= latest
NAMESPACE = aivillage-$(ENVIRONMENT)
HELM_CHART = ./helm/aivillage
DOCKER_REGISTRY = ghcr.io/aivillage

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

.PHONY: help build push deploy rollback test clean status

# Default target
help: ## Show this help message
	@echo "$(BLUE)AIVillage Deployment Commands$(NC)"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Prerequisites check
check-prereqs: ## Check if all required tools are installed
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v kubectl >/dev/null 2>&1 || { echo "$(RED)kubectl is required but not installed$(NC)"; exit 1; }
	@command -v helm >/dev/null 2>&1 || { echo "$(RED)helm is required but not installed$(NC)"; exit 1; }
	@command -v docker >/dev/null 2>&1 || { echo "$(RED)docker is required but not installed$(NC)"; exit 1; }
	@echo "$(GREEN)All prerequisites satisfied$(NC)"

# Build all Docker images
build: check-prereqs ## Build all Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@for service in gateway twin hyperag-mcp credits-api credits-worker agent-forge compression-service evolution-engine mesh-network; do \
		echo "$(YELLOW)Building $$service...$(NC)"; \
		docker build -f docker/Dockerfile.$$service -t $(DOCKER_REGISTRY)/$$service:$(IMAGE_TAG) .; \
	done
	@echo "$(GREEN)All images built successfully$(NC)"

# Push all Docker images
push: ## Push all Docker images to registry
	@echo "$(BLUE)Pushing Docker images...$(NC)"
	@for service in gateway twin hyperag-mcp credits-api credits-worker agent-forge compression-service evolution-engine mesh-network; do \
		echo "$(YELLOW)Pushing $$service...$(NC)"; \
		docker push $(DOCKER_REGISTRY)/$$service:$(IMAGE_TAG); \
	done
	@echo "$(GREEN)All images pushed successfully$(NC)"

# Build and push all images
build-push: build push ## Build and push all Docker images

# Deploy to staging
deploy-staging: ## Deploy to staging environment
	@$(MAKE) deploy ENVIRONMENT=staging

# Deploy to production
deploy-production: ## Deploy to production environment
	@$(MAKE) deploy ENVIRONMENT=production

# Deploy using orchestration script
deploy: check-prereqs ## Deploy to specified environment
	@echo "$(BLUE)Deploying to $(ENVIRONMENT) environment...$(NC)"
	@python scripts/deploy.py --environment $(ENVIRONMENT) --image-tag $(IMAGE_TAG)
	@echo "$(GREEN)Deployment completed$(NC)"

# Deploy using Helm directly
helm-deploy: check-prereqs ## Deploy using Helm directly
	@echo "$(BLUE)Deploying with Helm to $(ENVIRONMENT)...$(NC)"
	@helm upgrade --install aivillage-$(ENVIRONMENT) $(HELM_CHART) \
		--namespace $(NAMESPACE) \
		--create-namespace \
		--set image.tag=$(IMAGE_TAG) \
		--set environment=$(ENVIRONMENT) \
		--values $(HELM_CHART)/values-$(ENVIRONMENT).yaml \
		--wait \
		--timeout=15m
	@echo "$(GREEN)Helm deployment completed$(NC)"

# Run smoke tests
test-smoke: ## Run smoke tests
	@echo "$(BLUE)Running smoke tests for $(ENVIRONMENT)...$(NC)"
	@python scripts/smoke_tests.py --environment $(ENVIRONMENT) --namespace $(NAMESPACE)

# Run health checks
test-health: ## Run health checks
	@echo "$(BLUE)Running health checks for $(ENVIRONMENT)...$(NC)"
	@python scripts/health_check.py --environment $(ENVIRONMENT)

# Run readiness tests
test-readiness: ## Run production readiness tests
	@echo "$(BLUE)Running readiness tests for $(ENVIRONMENT)...$(NC)"
	@python scripts/readiness_tests.py --environment $(ENVIRONMENT) --namespace $(NAMESPACE) --slot blue

# Run production verification
test-production: ## Run production verification tests
	@echo "$(BLUE)Running production verification for $(ENVIRONMENT)...$(NC)"
	@python scripts/production_verification.py --environment $(ENVIRONMENT) --slot blue

# Run all tests
test-all: test-smoke test-health test-readiness ## Run all test suites

# Rollback deployment
rollback: check-prereqs ## Rollback to previous deployment
	@echo "$(YELLOW)Rolling back $(ENVIRONMENT) deployment...$(NC)"
	@python scripts/deploy.py --environment $(ENVIRONMENT) --action rollback
	@echo "$(GREEN)Rollback completed$(NC)"

# Rollback using Helm
helm-rollback: check-prereqs ## Rollback using Helm
	@echo "$(YELLOW)Rolling back with Helm...$(NC)"
	@helm rollback aivillage-$(ENVIRONMENT) --namespace $(NAMESPACE)
	@echo "$(GREEN)Helm rollback completed$(NC)"

# Get deployment status
status: ## Get deployment status
	@echo "$(BLUE)Getting deployment status for $(ENVIRONMENT)...$(NC)"
	@python scripts/deploy.py --environment $(ENVIRONMENT) --action status
	@kubectl get pods -n $(NAMESPACE)
	@helm status aivillage-$(ENVIRONMENT) -n $(NAMESPACE)

# Get service endpoints
endpoints: ## Get service endpoints
	@echo "$(BLUE)Service endpoints for $(ENVIRONMENT):$(NC)"
	@kubectl get services -n $(NAMESPACE)
	@kubectl get ingress -n $(NAMESPACE)

# View logs
logs: ## View logs for all services
	@echo "$(BLUE)Recent logs for $(ENVIRONMENT):$(NC)"
	@kubectl logs -l app.kubernetes.io/instance=aivillage-$(ENVIRONMENT) -n $(NAMESPACE) --tail=100

# Watch deployment progress
watch: ## Watch deployment progress
	@echo "$(BLUE)Watching deployment progress...$(NC)"
	@watch kubectl get pods -n $(NAMESPACE)

# Scale deployment
scale: ## Scale deployment (usage: make scale REPLICAS=5)
	@echo "$(BLUE)Scaling deployment to $(REPLICAS) replicas...$(NC)"
	@kubectl scale deployment aivillage-gateway --replicas=$(REPLICAS) -n $(NAMESPACE)
	@kubectl scale deployment aivillage-twin --replicas=$(REPLICAS) -n $(NAMESPACE)

# Port forward to gateway service
port-forward: ## Port forward to gateway service (localhost:8000)
	@echo "$(BLUE)Port forwarding to gateway service...$(NC)"
	@kubectl port-forward service/aivillage-gateway 8000:8000 -n $(NAMESPACE)

# Connect to database
db-connect: ## Connect to PostgreSQL database
	@echo "$(BLUE)Connecting to PostgreSQL...$(NC)"
	@kubectl exec -it statefulset/aivillage-postgres -n $(NAMESPACE) -- psql -U aivillage_user -d aivillage_$(ENVIRONMENT)

# Backup database
db-backup: ## Backup PostgreSQL database
	@echo "$(BLUE)Creating database backup...$(NC)"
	@kubectl exec statefulset/aivillage-postgres -n $(NAMESPACE) -- pg_dump -U aivillage_user aivillage_$(ENVIRONMENT) > backup-$(ENVIRONMENT)-$(shell date +%Y%m%d-%H%M%S).sql
	@echo "$(GREEN)Database backup created$(NC)"

# Clean up resources
clean: ## Clean up deployment resources
	@echo "$(YELLOW)Cleaning up $(ENVIRONMENT) resources...$(NC)"
	@helm uninstall aivillage-$(ENVIRONMENT) -n $(NAMESPACE) || true
	@kubectl delete namespace $(NAMESPACE) || true
	@echo "$(GREEN)Cleanup completed$(NC)"

# Emergency stop
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP: Scaling down all services...$(NC)"
	@kubectl scale deployment --all --replicas=0 -n $(NAMESPACE)
	@echo "$(RED)All services stopped$(NC)"

# Quick restart
restart: ## Quick restart of all services
	@echo "$(BLUE)Restarting all services...$(NC)"
	@kubectl rollout restart deployment -n $(NAMESPACE)
	@kubectl rollout status deployment -n $(NAMESPACE)
	@echo "$(GREEN)All services restarted$(NC)"

# Development helpers
dev-setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@kubectl create namespace aivillage-dev || true
	@helm upgrade --install aivillage-dev $(HELM_CHART) \
		--namespace aivillage-dev \
		--set environment=development \
		--set image.tag=dev \
		--set deployment.replicaCount=1 \
		--wait
	@echo "$(GREEN)Development environment ready$(NC)"

# Monitoring
monitor: ## Open monitoring dashboard
	@echo "$(BLUE)Opening monitoring dashboard...$(NC)"
	@kubectl port-forward service/aivillage-grafana 3000:3000 -n $(NAMESPACE) &
	@echo "$(GREEN)Grafana available at http://localhost:3000$(NC)"
	@echo "$(BLUE)Opening Prometheus...$(NC)"
	@kubectl port-forward service/aivillage-prometheus 9090:9090 -n $(NAMESPACE) &
	@echo "$(GREEN)Prometheus available at http://localhost:9090$(NC)"

# Security scan
security-scan: ## Run security scan on images
	@echo "$(BLUE)Running security scan...$(NC)"
	@for service in gateway twin hyperag-mcp credits-api; do \
		echo "$(YELLOW)Scanning $$service...$(NC)"; \
		docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
			aquasec/trivy image $(DOCKER_REGISTRY)/$$service:$(IMAGE_TAG); \
	done

# Performance test
perf-test: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	@kubectl run perf-test --rm -i --restart=Never --image=loadimpact/k6 -- \
		run --vus 10 --duration 30s https://api.aivillage.com/healthz

# Generate deployment report
report: ## Generate deployment report
	@echo "$(BLUE)Generating deployment report...$(NC)"
	@echo "# AIVillage Deployment Report" > deployment-report.md
	@echo "Generated: $(shell date)" >> deployment-report.md
	@echo "Environment: $(ENVIRONMENT)" >> deployment-report.md
	@echo "Image Tag: $(IMAGE_TAG)" >> deployment-report.md
	@echo "" >> deployment-report.md
	@echo "## Pod Status" >> deployment-report.md
	@kubectl get pods -n $(NAMESPACE) >> deployment-report.md
	@echo "" >> deployment-report.md
	@echo "## Service Status" >> deployment-report.md
	@kubectl get services -n $(NAMESPACE) >> deployment-report.md
	@echo "$(GREEN)Report saved to deployment-report.md$(NC)"

# Interactive shell
shell: ## Open interactive shell in gateway pod
	@echo "$(BLUE)Opening shell in gateway pod...$(NC)"
	@kubectl exec -it deployment/aivillage-gateway -n $(NAMESPACE) -- /bin/bash

# View resource usage
resources: ## View resource usage
	@echo "$(BLUE)Resource usage for $(ENVIRONMENT):$(NC)"
	@kubectl top nodes
	@echo ""
	@kubectl top pods -n $(NAMESPACE)

# Update dependencies
update-deps: ## Update Helm dependencies
	@echo "$(BLUE)Updating Helm dependencies...$(NC)"
	@cd $(HELM_CHART) && helm dependency update
	@echo "$(GREEN)Dependencies updated$(NC)"

# Validate configuration
validate: ## Validate Kubernetes manifests and Helm charts
	@echo "$(BLUE)Validating configurations...$(NC)"
	@kubeval k8s/*.yaml
	@helm lint $(HELM_CHART)
	@echo "$(GREEN)Configuration validation completed$(NC)"

# Examples and help
examples: ## Show usage examples
	@echo "$(BLUE)Common Usage Examples:$(NC)"
	@echo "  Deploy to staging:     make deploy-staging"
	@echo "  Deploy to production:  make deploy-production IMAGE_TAG=v1.2.3"
	@echo "  Run all tests:         make test-all ENVIRONMENT=staging"
	@echo "  Scale up:              make scale REPLICAS=5 ENVIRONMENT=production"
	@echo "  View logs:             make logs ENVIRONMENT=production"
	@echo "  Emergency stop:        make emergency-stop ENVIRONMENT=staging"
	@echo "  Monitor services:      make monitor ENVIRONMENT=production"
