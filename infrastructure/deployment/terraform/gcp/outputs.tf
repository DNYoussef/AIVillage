# Outputs for GCP Infrastructure

# Network Outputs
output "vpc_network_name" {
  description = "Name of the VPC network"
  value       = google_compute_network.aivillage.name
}

output "vpc_network_self_link" {
  description = "Self-link of the VPC network"
  value       = google_compute_network.aivillage.self_link
}

output "private_subnet_name" {
  description = "Name of the private subnet"
  value       = google_compute_subnetwork.private.name
}

output "public_subnet_name" {
  description = "Name of the public subnet"
  value       = google_compute_subnetwork.public.name
}

output "private_subnet_cidr" {
  description = "CIDR range of the private subnet"
  value       = google_compute_subnetwork.private.ip_cidr_range
}

output "public_subnet_cidr" {
  description = "CIDR range of the public subnet"
  value       = google_compute_subnetwork.public.ip_cidr_range
}

# GKE Cluster Outputs
output "cluster_name" {
  description = "Name of the GKE cluster"
  value       = google_container_cluster.aivillage.name
}

output "cluster_endpoint" {
  description = "Endpoint for GKE control plane"
  value       = google_container_cluster.aivillage.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "Cluster ca certificate (base64 encoded)"
  value       = google_container_cluster.aivillage.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "Location of the GKE cluster"
  value       = google_container_cluster.aivillage.location
}

output "cluster_master_version" {
  description = "Master version of the GKE cluster"
  value       = google_container_cluster.aivillage.master_version
}

output "cluster_node_version" {
  description = "Node version of the GKE cluster"
  value       = google_container_cluster.aivillage.node_version
}

output "cluster_service_account" {
  description = "Service account for GKE nodes"
  value       = google_service_account.gke_nodes.email
}

# Load Balancer Outputs
output "load_balancer_ip" {
  description = "External IP address of the load balancer"
  value       = google_compute_global_address.aivillage.address
}

output "load_balancer_name" {
  description = "Name of the load balancer"
  value       = google_compute_global_address.aivillage.name
}

output "https_forwarding_rule" {
  description = "HTTPS forwarding rule name"
  value       = google_compute_global_forwarding_rule.https.name
}

output "http_forwarding_rule" {
  description = "HTTP forwarding rule name"
  value       = google_compute_global_forwarding_rule.http.name
}

output "url_map_name" {
  description = "URL map name"
  value       = google_compute_url_map.aivillage.name
}

# SSL Certificate Outputs
output "ssl_certificate_name" {
  description = "Name of the SSL certificate"
  value       = var.domain_name != "" ? google_compute_managed_ssl_certificate.aivillage[0].name : google_compute_ssl_certificate.aivillage_self_signed[0].name
}

output "ssl_certificate_status" {
  description = "Status of the managed SSL certificate"
  value       = var.domain_name != "" ? google_compute_managed_ssl_certificate.aivillage[0].managed[0].status : "SELF_SIGNED"
}

# Backend Services Outputs
output "backend_services" {
  description = "Backend service information"
  value = {
    bridge_orchestrator = {
      name      = google_compute_backend_service.bridge_orchestrator.name
      self_link = google_compute_backend_service.bridge_orchestrator.self_link
    }
    python_bridge = {
      name      = google_compute_backend_service.python_bridge.name
      self_link = google_compute_backend_service.python_bridge.self_link
    }
    monitoring = {
      name      = google_compute_backend_service.monitoring.name
      self_link = google_compute_backend_service.monitoring.self_link
    }
  }
}

# Instance Group Managers Outputs
output "instance_group_managers" {
  description = "Instance group manager information"
  value = {
    bridge_orchestrator = {
      name           = google_compute_instance_group_manager.bridge_orchestrator.name
      instance_group = google_compute_instance_group_manager.bridge_orchestrator.instance_group
      target_size    = google_compute_instance_group_manager.bridge_orchestrator.target_size
    }
    python_bridge = {
      name           = google_compute_instance_group_manager.python_bridge.name
      instance_group = google_compute_instance_group_manager.python_bridge.instance_group
      target_size    = google_compute_instance_group_manager.python_bridge.target_size
    }
    monitoring = {
      name           = google_compute_instance_group_manager.monitoring.name
      instance_group = google_compute_instance_group_manager.monitoring.instance_group
      target_size    = google_compute_instance_group_manager.monitoring.target_size
    }
  }
}

# Autoscaler Outputs
output "autoscalers" {
  description = "Autoscaler information"
  value = {
    bridge_orchestrator = {
      name        = google_compute_autoscaler.bridge_orchestrator.name
      target      = google_compute_autoscaler.bridge_orchestrator.target
      min_replicas = var.bridge_orchestrator_min_replicas
      max_replicas = var.bridge_orchestrator_max_replicas
    }
    python_bridge = {
      name        = google_compute_autoscaler.python_bridge.name
      target      = google_compute_autoscaler.python_bridge.target
      min_replicas = var.python_bridge_min_replicas
      max_replicas = var.python_bridge_max_replicas
    }
  }
}

# Service Account Outputs
output "service_accounts" {
  description = "Service account emails"
  value = {
    gke_nodes         = google_service_account.gke_nodes.email
    compute_instances = google_service_account.compute_instances.email
    load_balancer     = google_service_account.load_balancer.email
  }
}

# KMS Outputs
output "kms_key_ring" {
  description = "KMS key ring name"
  value       = google_kms_key_ring.gke.name
}

output "kms_crypto_key" {
  description = "KMS crypto key name"
  value       = google_kms_crypto_key.gke.name
}

# Security Policy Outputs
output "security_policy_name" {
  description = "Cloud Armor security policy name"
  value       = google_compute_security_policy.aivillage.name
}

output "security_policy_self_link" {
  description = "Cloud Armor security policy self-link"
  value       = google_compute_security_policy.aivillage.self_link
}

# Application URLs
output "application_url" {
  description = "Application URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "https://${google_compute_global_address.aivillage.address}"
}

output "api_url" {
  description = "API URL"
  value       = var.domain_name != "" ? "https://api.${var.domain_name}" : "https://${google_compute_global_address.aivillage.address}/api"
}

output "monitoring_url" {
  description = "Monitoring dashboard URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}/monitoring" : "https://${google_compute_global_address.aivillage.address}/monitoring"
}

# Health Check URLs
output "health_check_urls" {
  description = "Health check URLs for services"
  value = {
    bridge_orchestrator = "https://${google_compute_global_address.aivillage.address}/health"
    python_bridge      = "https://${google_compute_global_address.aivillage.address}/betanet/health"
    monitoring         = "https://${google_compute_global_address.aivillage.address}/monitoring/api/health"
  }
}

# Service Endpoints
output "service_endpoints" {
  description = "Service endpoints for AIVillage components"
  value = {
    bridge_orchestrator = {
      http      = "https://${google_compute_global_address.aivillage.address}/api"
      websocket = "wss://${google_compute_global_address.aivillage.address}/bridge/ws"
      metrics   = "https://${google_compute_global_address.aivillage.address}/metrics"
    }
    python_bridge = {
      jsonrpc = "https://${google_compute_global_address.aivillage.address}/betanet"
    }
    monitoring = {
      grafana    = "https://${google_compute_global_address.aivillage.address}/monitoring"
      prometheus = "https://${google_compute_global_address.aivillage.address}/monitoring/prometheus"
    }
  }
}

# Monitoring Outputs
output "notification_channel" {
  description = "Monitoring notification channel"
  value       = google_monitoring_notification_channel.email.name
}

output "alert_policies" {
  description = "Alert policy names"
  value = {
    high_cpu        = google_monitoring_alert_policy.high_cpu.name
    high_p95_latency = google_monitoring_alert_policy.high_p95_latency.name
    instance_down   = google_monitoring_alert_policy.instance_down.name
  }
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    project_id          = var.project_id
    region              = var.region
    zone                = var.zone
    environment         = var.environment
    constitutional_tier = var.constitutional_tier
    privacy_mode       = var.privacy_mode
    enable_autopilot   = var.enable_autopilot
    deployment_time    = timestamp()
  }
}

# GKE Configuration Commands
output "gke_get_credentials_command" {
  description = "Command to get GKE credentials"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.aivillage.name} --region=${var.region} --project=${var.project_id}"
}

output "kubectl_context" {
  description = "kubectl context name"
  value       = "gke_${var.project_id}_${var.region}_${google_container_cluster.aivillage.name}"
}

# Performance Targets
output "performance_targets" {
  description = "Performance targets and thresholds"
  value = {
    target_p95_latency_ms    = var.target_p95_latency_ms
    target_uptime_percentage = var.target_uptime_percentage
    rate_limit_threshold     = var.rate_limit_threshold
    rate_limit_ban_duration  = var.rate_limit_ban_duration
  }
}

# Network Configuration
output "network_configuration" {
  description = "Network configuration details"
  value = {
    vpc_network           = google_compute_network.aivillage.name
    private_subnet_cidr   = var.private_subnet_cidr
    public_subnet_cidr    = var.public_subnet_cidr
    pods_cidr_range       = var.pods_cidr_range
    services_cidr_range   = var.services_cidr_range
    master_ipv4_cidr_block = var.master_ipv4_cidr_block
  }
}

# Security Configuration
output "security_configuration" {
  description = "Security configuration details"
  value = {
    enable_binary_authorization = var.enable_binary_authorization
    enable_shielded_instances   = var.enable_shielded_instances
    enable_workload_identity    = var.enable_workload_identity
    enable_cloud_armor         = var.enable_cloud_armor
    ssl_policy_profile         = var.ssl_policy_profile
  }
}

# Resource Names for Reference
output "resource_names" {
  description = "Important resource names for reference"
  value = {
    vpc_network                = google_compute_network.aivillage.name
    private_subnet             = google_compute_subnetwork.private.name
    public_subnet              = google_compute_subnetwork.public.name
    gke_cluster                = google_container_cluster.aivillage.name
    load_balancer_ip           = google_compute_global_address.aivillage.name
    ssl_certificate            = var.domain_name != "" ? google_compute_managed_ssl_certificate.aivillage[0].name : google_compute_ssl_certificate.aivillage_self_signed[0].name
    security_policy            = google_compute_security_policy.aivillage.name
    kms_key_ring              = google_kms_key_ring.gke.name
    kms_crypto_key            = google_kms_crypto_key.gke.name
  }
}