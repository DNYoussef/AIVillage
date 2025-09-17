# Variables for AWS Infrastructure Deployment

# General Configuration
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "aivillage"
}

# VPC Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

# EKS Configuration
variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "aivillage-cluster"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "cluster_endpoint_public_access_cidrs" {
  description = "List of CIDR blocks that can access the Amazon EKS public API server endpoint"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Node Group Configuration
variable "node_instance_types" {
  description = "Instance types for EKS node group"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 3
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 10
}

# EKS Addon Versions
variable "vpc_cni_version" {
  description = "Version of the VPC CNI addon"
  type        = string
  default     = "v1.15.1-eksbuild.1"
}

variable "coredns_version" {
  description = "Version of the CoreDNS addon"
  type        = string
  default     = "v1.10.1-eksbuild.5"
}

variable "kube_proxy_version" {
  description = "Version of the kube-proxy addon"
  type        = string
  default     = "v1.28.2-eksbuild.2"
}

variable "ebs_csi_driver_version" {
  description = "Version of the EBS CSI driver addon"
  type        = string
  default     = "v1.25.0-eksbuild.1"
}

# Domain and SSL Configuration
variable "domain_name" {
  description = "Domain name for the application (leave empty for no custom domain)"
  type        = string
  default     = ""
}

variable "additional_domains" {
  description = "Additional domain names for SSL certificate"
  type        = list(string)
  default     = []
}

# CloudFront Configuration
variable "cloudfront_price_class" {
  description = "CloudFront distribution price class"
  type        = string
  default     = "PriceClass_100"

  validation {
    condition     = contains(["PriceClass_All", "PriceClass_200", "PriceClass_100"], var.cloudfront_price_class)
    error_message = "CloudFront price class must be one of: PriceClass_All, PriceClass_200, PriceClass_100."
  }
}

# Monitoring Configuration
variable "cloudwatch_log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
}

variable "alert_email" {
  description = "Email address for CloudWatch alarms"
  type        = string
  default     = "alerts@aivillage.com"
}

# Security Configuration
variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "enable_waf" {
  description = "Enable WAF protection"
  type        = bool
  default     = true
}

# Performance Configuration
variable "target_p95_latency_ms" {
  description = "Target P95 latency in milliseconds"
  type        = number
  default     = 75
}

variable "target_uptime_percentage" {
  description = "Target uptime percentage"
  type        = number
  default     = 99.9
}

# Constitutional AI Configuration
variable "constitutional_tier" {
  description = "Constitutional AI tier (Bronze, Silver, Gold, Platinum)"
  type        = string
  default     = "Silver"

  validation {
    condition     = contains(["Bronze", "Silver", "Gold", "Platinum"], var.constitutional_tier)
    error_message = "Constitutional tier must be one of: Bronze, Silver, Gold, Platinum."
  }
}

variable "privacy_mode" {
  description = "Privacy mode (standard, enhanced, maximum)"
  type        = string
  default     = "enhanced"

  validation {
    condition     = contains(["standard", "enhanced", "maximum"], var.privacy_mode)
    error_message = "Privacy mode must be one of: standard, enhanced, maximum."
  }
}

# BetaNet Configuration
variable "enable_betanet" {
  description = "Enable BetaNet protocol integration"
  type        = bool
  default     = true
}

variable "betanet_max_nodes" {
  description = "Maximum number of BetaNet nodes"
  type        = number
  default     = 50
}

# Fog Computing Configuration
variable "fog_coordinator_replicas" {
  description = "Number of fog coordinator replicas"
  type        = number
  default     = 3
}

variable "enable_fog_auto_scaling" {
  description = "Enable auto-scaling for fog coordinators"
  type        = bool
  default     = true
}

# Database Configuration
variable "enable_rds" {
  description = "Enable RDS PostgreSQL database"
  type        = bool
  default     = false
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
}

# Cache Configuration
variable "enable_elasticache" {
  description = "Enable ElastiCache Redis"
  type        = bool
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
}

# Load Balancer Configuration
variable "alb_idle_timeout" {
  description = "ALB idle timeout in seconds"
  type        = number
  default     = 60
}

variable "enable_cross_zone_load_balancing" {
  description = "Enable cross-zone load balancing"
  type        = bool
  default     = true
}

# Auto Scaling Configuration
variable "scale_up_threshold" {
  description = "CPU utilization threshold for scaling up"
  type        = number
  default     = 70
}

variable "scale_down_threshold" {
  description = "CPU utilization threshold for scaling down"
  type        = number
  default     = 30
}

variable "scale_up_cooldown" {
  description = "Cooldown period for scaling up in seconds"
  type        = number
  default     = 300
}

variable "scale_down_cooldown" {
  description = "Cooldown period for scaling down in seconds"
  type        = number
  default     = 300
}

# Backup Configuration
variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "enable_automated_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}
}