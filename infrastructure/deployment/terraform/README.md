# AIVillage Infrastructure Deployment

This directory contains Terraform configurations for deploying the AIVillage Fog Computing Platform to cloud providers. The infrastructure supports the complete BetaNet protocol stack with constitutional AI validation, auto-scaling, and comprehensive monitoring.

## 🏗️ Infrastructure Overview

### Architecture Components

- **TypeScript Bridge Orchestrator**: HTTP/WebSocket API gateway with circuit breaker patterns
- **Python BetaNet Bridge**: JSON-RPC service for constitutional transport layer
- **Fog Coordinator**: Distributed node management with namespace isolation
- **Monitoring Stack**: Prometheus, Grafana, AlertManager with custom dashboards
- **Auto-scaling**: Dynamic scaling based on P95 latency and request rate
- **CDN**: CloudFront (AWS) / Cloud CDN (GCP) for global distribution
- **Security**: WAF, DDoS protection, SSL/TLS termination

### Performance Targets

- **P95 Latency**: <75ms for bridge translations
- **Uptime**: 99.9% availability with auto-healing
- **Throughput**: Auto-scale to handle traffic spikes
- **Constitutional Validation**: Zero tolerance for violations

## 🚀 Quick Start

### Prerequisites

1. **Terraform** >= 1.0
2. **Cloud CLI** (AWS CLI or gcloud)
3. **kubectl** for Kubernetes management
4. **Docker** for container registry access

### AWS Deployment

```bash
# Configure AWS credentials
aws configure

# Initialize Terraform
cd terraform/aws
terraform init

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
aws_region = "us-west-2"
environment = "dev"
domain_name = "your-domain.com"  # Optional
alert_email = "alerts@your-domain.com"
constitutional_tier = "Silver"
privacy_mode = "enhanced"
EOF

# Plan deployment
terraform plan

# Deploy infrastructure
terraform apply

# Get kubectl config
aws eks --region us-west-2 update-kubeconfig --name aivillage-cluster
```

### GCP Deployment

```bash
# Configure GCP credentials
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Initialize Terraform
cd terraform/gcp
terraform init

# Create terraform.tfvars
cat > terraform.tfvars <<EOF
project_id = "your-project-id"
region = "us-west1"
environment = "dev"
domain_name = "your-domain.com"  # Optional
alert_email = "alerts@your-domain.com"
constitutional_tier = "Silver"
privacy_mode = "enhanced"
enable_autopilot = false  # or true for GKE Autopilot
EOF

# Plan deployment
terraform plan

# Deploy infrastructure
terraform apply

# Get kubectl config
gcloud container clusters get-credentials aivillage-cluster --region=us-west1
```

## 📁 Directory Structure

```
terraform/
├── aws/                      # AWS-specific deployment
│   ├── main.tf              # Core EKS cluster and VPC
│   ├── iam.tf               # IAM roles and policies
│   ├── autoscaling.tf       # ALB and auto-scaling configuration
│   ├── cdn.tf               # CloudFront distribution
│   ├── ssl.tf               # Route53 and SSL certificates
│   ├── variables.tf         # Input variables
│   └── outputs.tf           # Output values
├── gcp/                     # GCP-specific deployment
│   ├── main.tf              # Core GKE cluster and VPC
│   ├── load-balancer.tf     # Global load balancer
│   ├── auto-scaling.tf      # Instance groups and auto-scaling
│   ├── variables.tf         # Input variables
│   ├── outputs.tf           # Output values
│   └── scripts/             # Instance startup scripts
│       ├── bridge-orchestrator-startup.sh
│       ├── python-bridge-startup.sh
│       └── monitoring-startup.sh
└── README.md               # This file
```

## ⚙️ Configuration Variables

### Common Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `environment` | Environment name (dev/staging/prod) | `dev` | No |
| `domain_name` | Custom domain name | `""` | No |
| `constitutional_tier` | AI validation tier | `Silver` | No |
| `privacy_mode` | Privacy level | `enhanced` | No |
| `target_p95_latency_ms` | Performance target | `75` | No |
| `alert_email` | Alert notifications | - | Yes |

### AWS-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-west-2` |
| `vpc_cidr` | VPC CIDR block | `10.0.0.0/16` |
| `cluster_name` | EKS cluster name | `aivillage-cluster` |
| `node_instance_types` | EC2 instance types | `["t3.medium", "t3.large"]` |
| `cloudfront_price_class` | CDN price class | `PriceClass_100` |

### GCP-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `project_id` | GCP project ID | - |
| `region` | GCP region | `us-west1` |
| `enable_autopilot` | Use GKE Autopilot | `false` |
| `instance_machine_type` | Compute instance type | `e2-standard-2` |
| `enable_cloud_armor` | Enable DDoS protection | `true` |

## 🔐 Security Features

### Network Security
- Private subnets for compute resources
- Network segmentation with security groups/firewall rules
- NAT gateways for outbound internet access
- VPN/bastion host access for management

### Application Security
- WAF with OWASP rule sets
- DDoS protection and rate limiting
- SSL/TLS termination with strong cipher suites
- Container image scanning and binary authorization

### Constitutional AI Security
- Multi-tier validation (Bronze/Silver/Gold/Platinum)
- Privacy modes (standard/enhanced/maximum)
- Zero-knowledge proof integration
- Audit logging for compliance

## 📊 Monitoring and Observability

### Metrics Collection
- Prometheus for metrics aggregation
- Custom metrics for BetaNet translations
- Constitutional validation tracking
- Performance and latency monitoring

### Visualization
- Grafana dashboards for real-time monitoring
- P95 latency tracking with alerting
- Constitutional violation detection
- Resource utilization and scaling metrics

### Alerting
- CloudWatch Alarms (AWS) / Cloud Monitoring (GCP)
- Email and Slack notifications
- Auto-scaling based on performance thresholds
- Health check failures and recovery

## 🔄 Auto-scaling Configuration

### Scaling Triggers
- CPU utilization (>70% scale up, <30% scale down)
- P95 latency (>75ms target)
- Request rate and queue depth
- Constitutional validation load

### Scaling Policies
- Gradual scale-up with 5-minute cooldown
- Conservative scale-down with 10-minute cooldown
- Maximum 10 instances per service
- Minimum 1 instance for availability

## 🌐 CDN and Global Distribution

### AWS CloudFront
- Global edge locations for low latency
- Static asset caching with TTL policies
- Dynamic content acceleration
- Security headers and WAF integration

### GCP Cloud CDN
- Google's global network infrastructure
- Intelligent caching with cache invalidation
- HTTP/2 and QUIC protocol support
- Cloud Armor integration for security

## 🚨 Disaster Recovery

### Backup Strategy
- Daily automated backups of persistent data
- Cross-region replication for critical components
- Point-in-time recovery capabilities
- Configuration versioning in Terraform state

### High Availability
- Multi-AZ/zone deployment for redundancy
- Auto-healing with health checks
- Rolling updates with zero downtime
- Circuit breaker patterns for fault tolerance

## 🔧 Maintenance and Updates

### Infrastructure Updates
```bash
# Update Terraform configurations
terraform plan
terraform apply

# Update Kubernetes cluster
# AWS EKS
aws eks update-cluster-version --name aivillage-cluster --version 1.28

# GCP GKE
gcloud container clusters upgrade aivillage-cluster --region us-west1
```

### Application Updates
```bash
# Update container images
kubectl set image deployment/bridge-orchestrator \
  bridge=gcr.io/project/aivillage-bridge:v2.0.0

# Rolling restart
kubectl rollout restart deployment/bridge-orchestrator
```

## 🐛 Troubleshooting

### Common Issues

**EKS/GKE Access Issues**
```bash
# Verify kubectl context
kubectl config current-context

# Check cluster access
kubectl get nodes
kubectl get pods --all-namespaces
```

**Performance Issues**
```bash
# Check metrics
kubectl top nodes
kubectl top pods

# View logs
kubectl logs -f deployment/bridge-orchestrator
```

**Constitutional Validation Failures**
```bash
# Check Python bridge logs
kubectl logs -f deployment/betanet-bridge

# Verify constitutional tier configuration
kubectl get configmap constitutional-config -o yaml
```

### Support Resources
- [AWS EKS Troubleshooting](https://docs.aws.amazon.com/eks/latest/userguide/troubleshooting.html)
- [GKE Troubleshooting](https://cloud.google.com/kubernetes-engine/docs/troubleshooting)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

## 📝 Cost Optimization

### AWS Cost Management
- Use Spot instances for non-critical workloads
- Enable AWS Cost Explorer and budgets
- Implement lifecycle policies for S3 storage
- Monitor CloudWatch usage and retention

### GCP Cost Management
- Use preemptible instances where appropriate
- Enable Cloud Billing alerts and quotas
- Optimize Cloud CDN caching policies
- Monitor Stackdriver usage and retention

## 🔄 Cleanup

### Destroy Infrastructure
```bash
# AWS
cd terraform/aws
terraform destroy

# GCP
cd terraform/gcp
terraform destroy
```

**Warning**: This will permanently delete all resources and data. Ensure you have backups before proceeding.

---

For additional support or questions, please refer to the [main AIVillage documentation](../../README.md) or open an issue in the repository.