# Outputs for AWS Infrastructure

# VPC Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.aivillage.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.aivillage.cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.aivillage.id
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.aivillage[*].id
}

# EKS Cluster Outputs
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.aivillage.name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.aivillage.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.aivillage.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_eks_cluster.aivillage.role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.aivillage.certificate_authority[0].data
}

output "cluster_version" {
  description = "The Kubernetes version for the EKS cluster"
  value       = aws_eks_cluster.aivillage.version
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = aws_eks_cluster.aivillage.identity[0].oidc[0].issuer
}

# Node Group Outputs
output "node_group_arn" {
  description = "Amazon Resource Name (ARN) of the EKS Node Group"
  value       = aws_eks_node_group.general.arn
}

output "node_group_status" {
  description = "Status of the EKS Node Group"
  value       = aws_eks_node_group.general.status
}

# Load Balancer Outputs
output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.aivillage.dns_name
}

output "alb_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.aivillage.zone_id
}

output "alb_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.aivillage.arn
}

output "target_group_arns" {
  description = "ARNs of the target groups"
  value = {
    bridge_orchestrator = aws_lb_target_group.bridge_orchestrator.arn
    python_bridge      = aws_lb_target_group.python_bridge.arn
    monitoring         = aws_lb_target_group.monitoring.arn
  }
}

# CloudFront Outputs
output "cloudfront_distribution_id" {
  description = "ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.aivillage.id
}

output "cloudfront_distribution_arn" {
  description = "ARN of the CloudFront distribution"
  value       = aws_cloudfront_distribution.aivillage.arn
}

output "cloudfront_domain_name" {
  description = "Domain name of the CloudFront distribution"
  value       = aws_cloudfront_distribution.aivillage.domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "Hosted zone ID of the CloudFront distribution"
  value       = aws_cloudfront_distribution.aivillage.hosted_zone_id
}

# SSL Certificate Outputs
output "certificate_arn" {
  description = "ARN of the ACM certificate"
  value       = aws_acm_certificate.aivillage.arn
}

output "certificate_domain_validation_options" {
  description = "Domain validation options for the certificate"
  value       = aws_acm_certificate.aivillage.domain_validation_options
}

# Route53 Outputs
output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = var.domain_name != "" ? aws_route53_zone.aivillage[0].zone_id : null
}

output "route53_name_servers" {
  description = "Route53 name servers"
  value       = var.domain_name != "" ? aws_route53_zone.aivillage[0].name_servers : null
}

# Domain Outputs
output "application_url" {
  description = "Application URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}" : "https://${aws_cloudfront_distribution.aivillage.domain_name}"
}

output "api_url" {
  description = "API URL"
  value       = var.domain_name != "" ? "https://api.${var.domain_name}" : "https://${aws_lb.aivillage.dns_name}"
}

output "monitoring_url" {
  description = "Monitoring dashboard URL"
  value       = var.domain_name != "" ? "https://${var.domain_name}/monitoring" : "https://${aws_cloudfront_distribution.aivillage.domain_name}/monitoring"
}

# Security Outputs
output "security_group_ids" {
  description = "Security group IDs"
  value = {
    eks_cluster    = aws_security_group.eks_cluster.id
    eks_node_group = aws_security_group.eks_node_group.id
    alb           = aws_security_group.alb.id
  }
}

output "waf_web_acl_arn" {
  description = "ARN of the WAF Web ACL"
  value       = aws_wafv2_web_acl.aivillage.arn
}

# S3 Outputs
output "s3_bucket_names" {
  description = "Names of S3 buckets"
  value = {
    static_assets     = aws_s3_bucket.static_assets.bucket
    alb_logs         = aws_s3_bucket.alb_logs.bucket
    cloudfront_logs  = aws_s3_bucket.cloudfront_logs.bucket
  }
}

# IAM Outputs
output "iam_role_arns" {
  description = "ARNs of IAM roles"
  value = {
    eks_cluster        = aws_iam_role.eks_cluster.arn
    eks_node_group     = aws_iam_role.eks_node_group.arn
    alb_controller     = aws_iam_role.alb_controller.arn
    cluster_autoscaler = aws_iam_role.cluster_autoscaler.arn
    ebs_csi_driver     = aws_iam_role.ebs_csi_driver.arn
  }
}

# KMS Outputs
output "kms_key_arns" {
  description = "ARNs of KMS keys"
  value = {
    eks        = aws_kms_key.eks.arn
    cloudwatch = aws_kms_key.cloudwatch.arn
    s3         = aws_kms_key.s3.arn
  }
}

# CloudWatch Outputs
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.eks_cluster.name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.alerts.arn
}

# Service Endpoints
output "service_endpoints" {
  description = "Service endpoints for AIVillage components"
  value = {
    bridge_orchestrator = {
      http      = "http://${aws_lb.aivillage.dns_name}:8080"
      websocket = "ws://${aws_lb.aivillage.dns_name}:8081"
      metrics   = "http://${aws_lb.aivillage.dns_name}:9090"
    }
    python_bridge = {
      jsonrpc = "http://${aws_lb.aivillage.dns_name}:9876"
    }
    monitoring = {
      grafana    = "http://${aws_lb.aivillage.dns_name}:3000"
      prometheus = "http://${aws_lb.aivillage.dns_name}:9091"
    }
  }
}

# Deployment Information
output "deployment_info" {
  description = "Deployment information"
  value = {
    environment         = var.environment
    aws_region         = var.aws_region
    cluster_name       = var.cluster_name
    constitutional_tier = var.constitutional_tier
    privacy_mode       = var.privacy_mode
    deployment_time    = timestamp()
  }
}

# Configuration for kubectl
output "kubectl_config" {
  description = "kubectl configuration command"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${aws_eks_cluster.aivillage.name}"
}

# Health Check URLs
output "health_check_urls" {
  description = "Health check URLs for services"
  value = {
    alb_health         = "https://${aws_lb.aivillage.dns_name}/health"
    bridge_health      = "https://${aws_lb.aivillage.dns_name}:8080/health"
    python_bridge_health = "https://${aws_lb.aivillage.dns_name}:9876/health"
    cloudfront_health  = "https://${aws_cloudfront_distribution.aivillage.domain_name}"
  }
}

# Performance Targets
output "performance_targets" {
  description = "Performance targets and thresholds"
  value = {
    target_p95_latency_ms    = var.target_p95_latency_ms
    target_uptime_percentage = var.target_uptime_percentage
    scale_up_threshold      = var.scale_up_threshold
    scale_down_threshold    = var.scale_down_threshold
  }
}