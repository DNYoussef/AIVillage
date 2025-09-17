# Auto-scaling Configuration for GCP Compute Instances

# Instance Templates
resource "google_compute_instance_template" "bridge_orchestrator" {
  name_prefix  = "aivillage-bridge-template-${var.environment}-"
  description  = "Template for TypeScript Bridge Orchestrator instances"
  machine_type = var.instance_machine_type

  disk {
    source_image = var.container_optimized_os_image
    auto_delete  = true
    boot         = true
    disk_size_gb = var.instance_disk_size
    disk_type    = "pd-ssd"
  }

  network_interface {
    network    = google_compute_network.aivillage.self_link
    subnetwork = google_compute_subnetwork.private.self_link
  }

  service_account {
    email  = google_service_account.compute_instances.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "TRUE"
    startup-script = templatefile("${path.module}/scripts/bridge-orchestrator-startup.sh", {
      docker_image = var.bridge_orchestrator_image
      environment  = var.environment
      project_id   = var.project_id
    })
  }

  tags = ["aivillage-service", "bridge-orchestrator"]

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_instance_template" "python_bridge" {
  name_prefix  = "aivillage-python-template-${var.environment}-"
  description  = "Template for Python BetaNet Bridge instances"
  machine_type = var.instance_machine_type

  disk {
    source_image = var.container_optimized_os_image
    auto_delete  = true
    boot         = true
    disk_size_gb = var.instance_disk_size
    disk_type    = "pd-ssd"
  }

  network_interface {
    network    = google_compute_network.aivillage.self_link
    subnetwork = google_compute_subnetwork.private.self_link
  }

  service_account {
    email  = google_service_account.compute_instances.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "TRUE"
    startup-script = templatefile("${path.module}/scripts/python-bridge-startup.sh", {
      docker_image = var.python_bridge_image
      environment  = var.environment
      project_id   = var.project_id
    })
  }

  tags = ["aivillage-service", "python-bridge"]

  lifecycle {
    create_before_destroy = true
  }
}

resource "google_compute_instance_template" "monitoring" {
  name_prefix  = "aivillage-monitoring-template-${var.environment}-"
  description  = "Template for monitoring stack instances"
  machine_type = var.monitoring_machine_type

  disk {
    source_image = var.container_optimized_os_image
    auto_delete  = true
    boot         = true
    disk_size_gb = var.monitoring_disk_size
    disk_type    = "pd-ssd"
  }

  network_interface {
    network    = google_compute_network.aivillage.self_link
    subnetwork = google_compute_subnetwork.private.self_link
  }

  service_account {
    email  = google_service_account.compute_instances.email
    scopes = ["cloud-platform"]
  }

  metadata = {
    enable-oslogin = "TRUE"
    startup-script = templatefile("${path.module}/scripts/monitoring-startup.sh", {
      environment = var.environment
      project_id  = var.project_id
    })
  }

  tags = ["aivillage-service", "monitoring"]

  lifecycle {
    create_before_destroy = true
  }
}

# Instance Group Managers
resource "google_compute_instance_group_manager" "bridge_orchestrator" {
  name = "aivillage-bridge-igm-${var.environment}"
  zone = var.zone

  version {
    instance_template = google_compute_instance_template.bridge_orchestrator.id
  }

  base_instance_name = "aivillage-bridge"
  target_size        = var.bridge_orchestrator_target_size

  named_port {
    name = "http"
    port = 8080
  }

  named_port {
    name = "websocket"
    port = 8081
  }

  named_port {
    name = "metrics"
    port = 9090
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.bridge_orchestrator.id
    initial_delay_sec = 300
  }

  update_policy {
    type                         = "PROACTIVE"
    instance_redistribution_type = "PROACTIVE"
    minimal_action               = "REPLACE"
    max_surge_fixed              = 1
    max_unavailable_fixed        = 0
  }
}

resource "google_compute_instance_group_manager" "python_bridge" {
  name = "aivillage-python-igm-${var.environment}"
  zone = var.zone

  version {
    instance_template = google_compute_instance_template.python_bridge.id
  }

  base_instance_name = "aivillage-python"
  target_size        = var.python_bridge_target_size

  named_port {
    name = "http"
    port = 9876
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.python_bridge.id
    initial_delay_sec = 300
  }

  update_policy {
    type                         = "PROACTIVE"
    instance_redistribution_type = "PROACTIVE"
    minimal_action               = "REPLACE"
    max_surge_fixed              = 1
    max_unavailable_fixed        = 0
  }
}

resource "google_compute_instance_group_manager" "monitoring" {
  name = "aivillage-monitoring-igm-${var.environment}"
  zone = var.zone

  version {
    instance_template = google_compute_instance_template.monitoring.id
  }

  base_instance_name = "aivillage-monitoring"
  target_size        = var.monitoring_target_size

  named_port {
    name = "grafana"
    port = 3000
  }

  named_port {
    name = "prometheus"
    port = 9091
  }

  auto_healing_policies {
    health_check      = google_compute_health_check.monitoring.id
    initial_delay_sec = 300
  }

  update_policy {
    type                         = "PROACTIVE"
    instance_redistribution_type = "PROACTIVE"
    minimal_action               = "REPLACE"
    max_surge_fixed              = 1
    max_unavailable_fixed        = 0
  }
}

# Autoscalers
resource "google_compute_autoscaler" "bridge_orchestrator" {
  name   = "aivillage-bridge-autoscaler-${var.environment}"
  zone   = var.zone
  target = google_compute_instance_group_manager.bridge_orchestrator.id

  autoscaling_policy {
    max_replicas    = var.bridge_orchestrator_max_replicas
    min_replicas    = var.bridge_orchestrator_min_replicas
    cooldown_period = 60

    cpu_utilization {
      target = 0.7
    }

    load_balancing_utilization {
      target = 0.8
    }

    # Scale based on custom metrics
    metric {
      name   = "custom.googleapis.com/aivillage/p95_latency_milliseconds"
      target = var.target_p95_latency_ms
      type   = "GAUGE"
    }

    scale_in_control {
      max_scaled_in_replicas {
        fixed = 1
      }
      time_window_sec = 300
    }
  }
}

resource "google_compute_autoscaler" "python_bridge" {
  name   = "aivillage-python-autoscaler-${var.environment}"
  zone   = var.zone
  target = google_compute_instance_group_manager.python_bridge.id

  autoscaling_policy {
    max_replicas    = var.python_bridge_max_replicas
    min_replicas    = var.python_bridge_min_replicas
    cooldown_period = 60

    cpu_utilization {
      target = 0.7
    }

    load_balancing_utilization {
      target = 0.8
    }

    # Scale based on BetaNet translation rate
    metric {
      name   = "custom.googleapis.com/aivillage/betanet_translations_per_second"
      target = 50
      type   = "GAUGE"
    }

    scale_in_control {
      max_scaled_in_replicas {
        fixed = 1
      }
      time_window_sec = 300
    }
  }
}

# Service Account for Compute Instances
resource "google_service_account" "compute_instances" {
  account_id   = "aivillage-compute-${var.environment}"
  display_name = "AIVillage Compute Instances Service Account"
}

resource "google_project_iam_member" "compute_instances" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer",
    "roles/secretmanager.secretAccessor"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.compute_instances.email}"
}

# Cloud Monitoring Policies
resource "google_monitoring_alert_policy" "high_cpu" {
  display_name = "AIVillage High CPU Usage - ${var.environment}"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "CPU usage is above 80%"

    condition_threshold {
      filter         = "resource.type=\"gce_instance\" AND resource.labels.instance_name=~\"aivillage-.*\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = 0.8

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "high_p95_latency" {
  display_name = "AIVillage High P95 Latency - ${var.environment}"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "P95 latency is above target"

    condition_threshold {
      filter         = "resource.type=\"gce_instance\" AND metric.type=\"custom.googleapis.com/aivillage/p95_latency_milliseconds\""
      duration       = "300s"
      comparison     = "COMPARISON_GREATER_THAN"
      threshold_value = var.target_p95_latency_ms

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}

resource "google_monitoring_alert_policy" "instance_down" {
  display_name = "AIVillage Instance Down - ${var.environment}"
  combiner     = "OR"
  enabled      = true

  conditions {
    display_name = "Instance is down"

    condition_threshold {
      filter         = "resource.type=\"gce_instance\" AND resource.labels.instance_name=~\"aivillage-.*\""
      duration       = "300s"
      comparison     = "COMPARISON_EQUAL"
      threshold_value = 0

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_MEAN"
        cross_series_reducer = "REDUCE_COUNT_TRUE"
        group_by_fields      = ["resource.labels.instance_name"]
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s"
  }
}

# Notification Channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "AIVillage Email Alerts - ${var.environment}"
  type         = "email"

  labels = {
    email_address = var.alert_email
  }

  enabled = true
}

# Log-based Metrics
resource "google_logging_metric" "error_rate" {
  name   = "aivillage_error_rate_${var.environment}"
  filter = "resource.type=\"gce_instance\" AND resource.labels.instance_name=~\"aivillage-.*\" AND severity=\"ERROR\""

  metric_descriptor {
    metric_kind = "GAUGE"
    value_type  = "INT64"
    display_name = "AIVillage Error Rate"
  }

  value_extractor = "EXTRACT(jsonPayload.error_count)"
}

resource "google_logging_metric" "constitutional_violations" {
  name   = "aivillage_constitutional_violations_${var.environment}"
  filter = "resource.type=\"gce_instance\" AND jsonPayload.type=\"constitutional_violation\""

  metric_descriptor {
    metric_kind = "COUNTER"
    value_type  = "INT64"
    display_name = "Constitutional Violations"
  }
}