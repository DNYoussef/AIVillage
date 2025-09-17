# Load Balancer and Auto-scaling Configuration for GCP

# Global static IP for load balancer
resource "google_compute_global_address" "aivillage" {
  name = "aivillage-lb-ip-${var.environment}"
}

# SSL Certificate
resource "google_compute_managed_ssl_certificate" "aivillage" {
  count = var.domain_name != "" ? 1 : 0
  name  = "aivillage-ssl-cert-${var.environment}"

  managed {
    domains = [var.domain_name, "www.${var.domain_name}", "api.${var.domain_name}"]
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Self-signed SSL certificate for development
resource "google_compute_ssl_certificate" "aivillage_self_signed" {
  count = var.domain_name == "" ? 1 : 0
  name  = "aivillage-ssl-cert-self-signed-${var.environment}"

  private_key = tls_private_key.aivillage[0].private_key_pem
  certificate = tls_self_signed_cert.aivillage[0].cert_pem

  lifecycle {
    create_before_destroy = true
  }
}

resource "tls_private_key" "aivillage" {
  count     = var.domain_name == "" ? 1 : 0
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_self_signed_cert" "aivillage" {
  count           = var.domain_name == "" ? 1 : 0
  private_key_pem = tls_private_key.aivillage[0].private_key_pem

  subject {
    common_name  = "aivillage.local"
    organization = "AIVillage"
  }

  validity_period_hours = 8760 # 1 year

  allowed_uses = [
    "key_encipherment",
    "digital_signature",
    "server_auth",
  ]
}

# Backend Services
resource "google_compute_backend_service" "bridge_orchestrator" {
  name        = "aivillage-bridge-backend-${var.environment}"
  description = "Backend service for TypeScript Bridge Orchestrator"

  port_name   = "http"
  protocol    = "HTTP"
  timeout_sec = 30

  health_checks = [google_compute_health_check.bridge_orchestrator.id]

  backend {
    group = google_compute_instance_group_manager.bridge_orchestrator.instance_group
    balancing_mode               = "RATE"
    max_rate_per_instance       = 100
    capacity_scaler             = 1.0
  }

  load_balancing_scheme = "EXTERNAL_MANAGED"

  # Enable CDN
  enable_cdn = true
  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 3600
    max_ttl                      = 86400
    negative_caching             = true
    serve_while_stale            = 86400
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

resource "google_compute_backend_service" "python_bridge" {
  name        = "aivillage-python-backend-${var.environment}"
  description = "Backend service for Python BetaNet Bridge"

  port_name   = "http"
  protocol    = "HTTP"
  timeout_sec = 30

  health_checks = [google_compute_health_check.python_bridge.id]

  backend {
    group = google_compute_instance_group_manager.python_bridge.instance_group
    balancing_mode               = "RATE"
    max_rate_per_instance       = 100
    capacity_scaler             = 1.0
  }

  load_balancing_scheme = "EXTERNAL_MANAGED"

  # Disable CDN for dynamic BetaNet content
  enable_cdn = false

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

resource "google_compute_backend_service" "monitoring" {
  name        = "aivillage-monitoring-backend-${var.environment}"
  description = "Backend service for monitoring stack"

  port_name   = "http"
  protocol    = "HTTP"
  timeout_sec = 30

  health_checks = [google_compute_health_check.monitoring.id]

  backend {
    group = google_compute_instance_group_manager.monitoring.instance_group
    balancing_mode               = "RATE"
    max_rate_per_instance       = 50
    capacity_scaler             = 1.0
  }

  load_balancing_scheme = "EXTERNAL_MANAGED"

  # Limited CDN for monitoring dashboards
  enable_cdn = true
  cdn_policy {
    cache_mode                   = "CACHE_ALL_STATIC"
    default_ttl                  = 300
    max_ttl                      = 3600
    negative_caching             = true
  }

  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# Health Checks
resource "google_compute_health_check" "bridge_orchestrator" {
  name = "aivillage-bridge-health-check-${var.environment}"

  timeout_sec        = 5
  check_interval_sec = 30
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 8080
    request_path = "/health"
  }

  log_config {
    enable = true
  }
}

resource "google_compute_health_check" "python_bridge" {
  name = "aivillage-python-health-check-${var.environment}"

  timeout_sec        = 5
  check_interval_sec = 30
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 9876
    request_path = "/health"
  }

  log_config {
    enable = true
  }
}

resource "google_compute_health_check" "monitoring" {
  name = "aivillage-monitoring-health-check-${var.environment}"

  timeout_sec        = 5
  check_interval_sec = 30
  healthy_threshold  = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 3000
    request_path = "/api/health"
  }

  log_config {
    enable = true
  }
}

# URL Map
resource "google_compute_url_map" "aivillage" {
  name        = "aivillage-url-map-${var.environment}"
  description = "URL map for AIVillage services"

  default_service = google_compute_backend_service.bridge_orchestrator.id

  host_rule {
    hosts        = var.domain_name != "" ? [var.domain_name, "www.${var.domain_name}"] : ["*"]
    path_matcher = "allpaths"
  }

  path_matcher {
    name            = "allpaths"
    default_service = google_compute_backend_service.bridge_orchestrator.id

    path_rule {
      paths   = ["/api/*", "/bridge/*"]
      service = google_compute_backend_service.bridge_orchestrator.id
    }

    path_rule {
      paths   = ["/betanet/*", "/python/*"]
      service = google_compute_backend_service.python_bridge.id
    }

    path_rule {
      paths   = ["/monitoring/*", "/grafana/*"]
      service = google_compute_backend_service.monitoring.id
    }
  }

  # Security headers
  response_headers_policy {
    response_headers_to_add {
      header_name  = "Strict-Transport-Security"
      header_value = "max-age=31536000; includeSubDomains"
      replace      = true
    }

    response_headers_to_add {
      header_name  = "X-Content-Type-Options"
      header_value = "nosniff"
      replace      = true
    }

    response_headers_to_add {
      header_name  = "X-Frame-Options"
      header_value = "DENY"
      replace      = true
    }

    response_headers_to_add {
      header_name  = "X-XSS-Protection"
      header_value = "1; mode=block"
      replace      = true
    }

    response_headers_to_remove = ["Server", "X-Powered-By"]
  }
}

# HTTPS Target Proxy
resource "google_compute_target_https_proxy" "aivillage" {
  name   = "aivillage-https-proxy-${var.environment}"
  url_map = google_compute_url_map.aivillage.id

  ssl_certificates = var.domain_name != "" ? [
    google_compute_managed_ssl_certificate.aivillage[0].id
  ] : [
    google_compute_ssl_certificate.aivillage_self_signed[0].id
  ]

  ssl_policy = google_compute_ssl_policy.aivillage.id
}

# HTTP Target Proxy (for redirect)
resource "google_compute_target_http_proxy" "aivillage" {
  name    = "aivillage-http-proxy-${var.environment}"
  url_map = google_compute_url_map.aivillage_redirect.id
}

# URL Map for HTTP to HTTPS redirect
resource "google_compute_url_map" "aivillage_redirect" {
  name = "aivillage-redirect-${var.environment}"

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

# SSL Policy
resource "google_compute_ssl_policy" "aivillage" {
  name            = "aivillage-ssl-policy-${var.environment}"
  profile         = "RESTRICTED"
  min_tls_version = "TLS_1_2"
}

# Global Forwarding Rules
resource "google_compute_global_forwarding_rule" "https" {
  name       = "aivillage-https-forwarding-rule-${var.environment}"
  target     = google_compute_target_https_proxy.aivillage.id
  port_range = "443"
  ip_address = google_compute_global_address.aivillage.address
}

resource "google_compute_global_forwarding_rule" "http" {
  name       = "aivillage-http-forwarding-rule-${var.environment}"
  target     = google_compute_target_http_proxy.aivillage.id
  port_range = "80"
  ip_address = google_compute_global_address.aivillage.address
}

# Cloud Armor Security Policy
resource "google_compute_security_policy" "aivillage" {
  name = "aivillage-security-policy-${var.environment}"

  # Default rule
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  # Rate limiting rule
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
      rate_limit_threshold {
        count        = 100
        interval_sec = 60
      }
      ban_duration_sec = 300
    }
    description = "Rate limiting rule"
  }

  # Block common attack patterns
  rule {
    action   = "deny(403)"
    priority = "2000"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('sqli-stable')"
      }
    }
    description = "Block SQL injection attempts"
  }

  rule {
    action   = "deny(403)"
    priority = "2001"
    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-stable')"
      }
    }
    description = "Block XSS attempts"
  }

  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
    }
  }
}

# Attach security policy to backend services
resource "google_compute_backend_service_signed_url_key" "bridge_orchestrator" {
  name            = "aivillage-bridge-key-${var.environment}"
  key_value       = base64encode(random_bytes.url_signature_key.result)
  backend_service = google_compute_backend_service.bridge_orchestrator.name
}

resource "random_bytes" "url_signature_key" {
  length = 16
}

# Update backend services with security policy
resource "google_compute_backend_service_iam_policy" "bridge_orchestrator" {
  backend_service = google_compute_backend_service.bridge_orchestrator.name
  policy_data     = data.google_iam_policy.backend_service.policy_data
}

data "google_iam_policy" "backend_service" {
  binding {
    role = "roles/compute.loadBalancerServiceUser"
    members = [
      "serviceAccount:${google_service_account.load_balancer.email}"
    ]
  }
}

resource "google_service_account" "load_balancer" {
  account_id   = "aivillage-lb-${var.environment}"
  display_name = "AIVillage Load Balancer Service Account"
}