# Variables for GCP Infrastructure Deployment

# General Configuration
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for deployment"
  type        = string
  default     = "us-west1"
}

variable "zone" {
  description = "GCP zone for single-zone resources"
  type        = string
  default     = "us-west1-b"
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

# Network Configuration
variable "private_subnet_cidr" {
  description = "CIDR block for private subnet"
  type        = string
  default     = "10.0.0.0/24"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.0.1.0/24"
}

variable "pods_cidr_range" {
  description = "CIDR range for Kubernetes pods"
  type        = string
  default     = "10.1.0.0/16"
}

variable "services_cidr_range" {
  description = "CIDR range for Kubernetes services"
  type        = string
  default     = "10.2.0.0/16"
}

variable "master_ipv4_cidr_block" {
  description = "CIDR block for GKE master"
  type        = string
  default     = "10.3.0.0/28"
}

# GKE Configuration
variable "cluster_name" {
  description = "Name of the GKE cluster"
  type        = string
  default     = "aivillage-cluster"
}

variable "enable_autopilot" {
  description = "Enable GKE Autopilot mode"
  type        = bool
  default     = false
}

variable "release_channel" {
  description = "GKE release channel"
  type        = string
  default     = "STABLE"

  validation {
    condition     = contains(["RAPID", "REGULAR", "STABLE"], var.release_channel)
    error_message = "Release channel must be one of: RAPID, REGULAR, STABLE."
  }
}

# Node Configuration (for Standard GKE)
variable "node_machine_type" {
  description = "Machine type for GKE nodes"
  type        = string
  default     = "e2-standard-2"
}

variable "node_disk_size" {
  description = "Disk size for GKE nodes in GB"
  type        = number
  default     = 50
}

variable "initial_node_count" {
  description = "Initial number of nodes per zone"
  type        = number
  default     = 1
}

variable "node_count_per_zone" {
  description = "Number of nodes per zone"
  type        = number
  default     = 1
}

variable "min_node_count" {
  description = "Minimum number of nodes per zone"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes per zone"
  type        = number
  default     = 10
}

# Compute Instance Configuration
variable "instance_machine_type" {
  description = "Machine type for compute instances"
  type        = string
  default     = "e2-standard-2"
}

variable "instance_disk_size" {
  description = "Disk size for compute instances in GB"
  type        = number
  default     = 50
}

variable "monitoring_machine_type" {
  description = "Machine type for monitoring instances"
  type        = string
  default     = "e2-standard-4"
}

variable "monitoring_disk_size" {
  description = "Disk size for monitoring instances in GB"
  type        = number
  default     = 100
}

variable "container_optimized_os_image" {
  description = "Container-Optimized OS image for instances"
  type        = string
  default     = "projects/cos-cloud/global/images/family/cos-stable"
}

# Docker Images
variable "bridge_orchestrator_image" {
  description = "Docker image for TypeScript Bridge Orchestrator"
  type        = string
  default     = "gcr.io/PROJECT_ID/aivillage-bridge-orchestrator:latest"
}

variable "python_bridge_image" {
  description = "Docker image for Python BetaNet Bridge"
  type        = string
  default     = "gcr.io/PROJECT_ID/aivillage-python-bridge:latest"
}

# Auto-scaling Configuration
variable "bridge_orchestrator_target_size" {
  description = "Target size for bridge orchestrator instance group"
  type        = number
  default     = 2
}

variable "bridge_orchestrator_min_replicas" {
  description = "Minimum replicas for bridge orchestrator autoscaler"
  type        = number
  default     = 1
}

variable "bridge_orchestrator_max_replicas" {
  description = "Maximum replicas for bridge orchestrator autoscaler"
  type        = number
  default     = 10
}

variable "python_bridge_target_size" {
  description = "Target size for Python bridge instance group"
  type        = number
  default     = 2
}

variable "python_bridge_min_replicas" {
  description = "Minimum replicas for Python bridge autoscaler"
  type        = number
  default     = 1
}

variable "python_bridge_max_replicas" {
  description = "Maximum replicas for Python bridge autoscaler"
  type        = number
  default     = 10
}

variable "monitoring_target_size" {
  description = "Target size for monitoring instance group"
  type        = number
  default     = 1
}

# Domain and SSL Configuration
variable "domain_name" {
  description = "Domain name for the application (leave empty for no custom domain)"
  type        = string
  default     = ""
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

# Monitoring Configuration
variable "alert_email" {
  description = "Email address for monitoring alerts"
  type        = string
  default     = "alerts@aivillage.com"
}

variable "enable_monitoring" {
  description = "Enable monitoring and alerting"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "Log retention period in days"
  type        = number
  default     = 30
}

# Security Configuration
variable "enable_binary_authorization" {
  description = "Enable Binary Authorization for container image security"
  type        = bool
  default     = true
}

variable "enable_shielded_instances" {
  description = "Enable Shielded VM instances"
  type        = bool
  default     = true
}

variable "enable_workload_identity" {
  description = "Enable Workload Identity for GKE"
  type        = bool
  default     = true
}

# Storage Configuration
variable "enable_persistent_disk_csi" {
  description = "Enable Persistent Disk CSI driver"
  type        = bool
  default     = true
}

variable "disk_encryption_key" {
  description = "Customer-managed encryption key for disks"
  type        = string
  default     = ""
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

# Load Balancer Configuration
variable "enable_cdn" {
  description = "Enable Cloud CDN"
  type        = bool
  default     = true
}

variable "cdn_cache_mode" {
  description = "CDN cache mode"
  type        = string
  default     = "CACHE_ALL_STATIC"

  validation {
    condition     = contains(["CACHE_ALL_STATIC", "USE_ORIGIN_HEADERS", "FORCE_CACHE_ALL"], var.cdn_cache_mode)
    error_message = "CDN cache mode must be one of: CACHE_ALL_STATIC, USE_ORIGIN_HEADERS, FORCE_CACHE_ALL."
  }
}

variable "ssl_policy_profile" {
  description = "SSL policy profile"
  type        = string
  default     = "RESTRICTED"

  validation {
    condition     = contains(["COMPATIBLE", "MODERN", "RESTRICTED", "CUSTOM"], var.ssl_policy_profile)
    error_message = "SSL policy profile must be one of: COMPATIBLE, MODERN, RESTRICTED, CUSTOM."
  }
}

# Cloud Armor Configuration
variable "enable_cloud_armor" {
  description = "Enable Cloud Armor security policies"
  type        = bool
  default     = true
}

variable "rate_limit_threshold" {
  description = "Rate limit threshold (requests per minute)"
  type        = number
  default     = 100
}

variable "rate_limit_ban_duration" {
  description = "Rate limit ban duration in seconds"
  type        = number
  default     = 300
}

# Stackdriver Configuration
variable "enable_stackdriver_logging" {
  description = "Enable Stackdriver logging"
  type        = bool
  default     = true
}

variable "enable_stackdriver_monitoring" {
  description = "Enable Stackdriver monitoring"
  type        = bool
  default     = true
}

variable "enable_stackdriver_trace" {
  description = "Enable Stackdriver trace"
  type        = bool
  default     = true
}

# Tags
variable "additional_labels" {
  description = "Additional labels to apply to all resources"
  type        = map(string)
  default     = {}
}