# GCP Infrastructure for AIVillage Fog Computing Platform
# Production-ready deployment with auto-scaling, load balancing, and CDN

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }

  backend "gcs" {
    bucket = "aivillage-terraform-state"
    prefix = "fog-infrastructure/terraform.tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Data sources
data "google_client_config" "default" {}

data "google_compute_zones" "available" {
  region = var.region
}

# VPC Network
resource "google_compute_network" "aivillage" {
  name                    = "aivillage-vpc-${var.environment}"
  auto_create_subnetworks = false
  mtu                     = 1460

  lifecycle {
    create_before_destroy = true
  }
}

# Subnets
resource "google_compute_subnetwork" "private" {
  name          = "aivillage-private-${var.environment}"
  ip_cidr_range = var.private_subnet_cidr
  region        = var.region
  network       = google_compute_network.aivillage.id

  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = var.pods_cidr_range
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = var.services_cidr_range
  }
}

resource "google_compute_subnetwork" "public" {
  name          = "aivillage-public-${var.environment}"
  ip_cidr_range = var.public_subnet_cidr
  region        = var.region
  network       = google_compute_network.aivillage.id
}

# Cloud Router for NAT
resource "google_compute_router" "aivillage" {
  name    = "aivillage-router-${var.environment}"
  region  = var.region
  network = google_compute_network.aivillage.id
}

# Cloud NAT
resource "google_compute_router_nat" "aivillage" {
  name                               = "aivillage-nat-${var.environment}"
  router                             = google_compute_router.aivillage.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "aivillage-allow-internal-${var.environment}"
  network = google_compute_network.aivillage.name

  allow {
    protocol = "icmp"
  }

  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }

  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }

  source_ranges = [
    var.private_subnet_cidr,
    var.public_subnet_cidr,
    var.pods_cidr_range,
    var.services_cidr_range
  ]
}

resource "google_compute_firewall" "allow_ssh" {
  name    = "aivillage-allow-ssh-${var.environment}"
  network = google_compute_network.aivillage.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh-allowed"]
}

resource "google_compute_firewall" "allow_aivillage_services" {
  name    = "aivillage-allow-services-${var.environment}"
  network = google_compute_network.aivillage.name

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "8080", "8081", "9876", "3000", "9090", "9091"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["aivillage-service"]
}

# GKE Cluster
resource "google_container_cluster" "aivillage" {
  name     = var.cluster_name
  location = var.region

  # Enable Autopilot for fully managed Kubernetes
  enable_autopilot = var.enable_autopilot

  # Network configuration
  network    = google_compute_network.aivillage.self_link
  subnetwork = google_compute_subnetwork.private.self_link

  # IP allocation policy for secondary ranges
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Master auth configuration
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  # Addons
  addons_config {
    http_load_balancing {
      disabled = false
    }

    horizontal_pod_autoscaling {
      disabled = false
    }

    network_policy_config {
      disabled = false
    }

    cloudrun_config {
      disabled = false
    }

    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
  }

  # Workload Identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Logging and monitoring
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  # Release channel
  release_channel {
    channel = var.release_channel
  }

  # Security configuration
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Database encryption
  database_encryption {
    state    = "ENCRYPTED"
    key_name = google_kms_crypto_key.gke.id
  }

  # Maintenance policy
  maintenance_policy {
    recurring_window {
      start_time = "2023-01-01T03:00:00Z"
      end_time   = "2023-01-01T07:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA"
    }
  }

  # Node configuration (for Standard GKE)
  dynamic "node_config" {
    for_each = var.enable_autopilot ? [] : [1]
    content {
      machine_type = var.node_machine_type
      disk_size_gb = var.node_disk_size

      service_account = google_service_account.gke_nodes.email
      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]

      tags = ["aivillage-service"]

      workload_metadata_config {
        mode = "GKE_METADATA"
      }

      shielded_instance_config {
        enable_secure_boot          = true
        enable_integrity_monitoring = true
      }
    }
  }

  # Initial node count (for Standard GKE)
  initial_node_count = var.enable_autopilot ? null : var.initial_node_count

  lifecycle {
    ignore_changes = [initial_node_count]
  }
}

# Node Pool (for Standard GKE only)
resource "google_container_node_pool" "general" {
  count      = var.enable_autopilot ? 0 : 1
  name       = "general-pool"
  location   = var.region
  cluster    = google_container_cluster.aivillage.name
  node_count = var.node_count_per_zone

  node_config {
    machine_type = var.node_machine_type
    disk_size_gb = var.node_disk_size
    disk_type    = "pd-ssd"

    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    tags = ["aivillage-service"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    metadata = {
      disable-legacy-endpoints = "true"
    }
  }

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 0
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# Service Account for GKE Nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "aivillage-gke-nodes-${var.environment}"
  display_name = "AIVillage GKE Nodes Service Account"
}

resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/storage.objectViewer"
  ])

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# KMS Key for GKE encryption
resource "google_kms_key_ring" "gke" {
  name     = "aivillage-gke-${var.environment}"
  location = var.region
}

resource "google_kms_crypto_key" "gke" {
  name     = "gke-encryption-key"
  key_ring = google_kms_key_ring.gke.id

  rotation_period = "7776000s" # 90 days

  lifecycle {
    prevent_destroy = true
  }
}

# Grant GKE service account access to KMS key
resource "google_kms_crypto_key_iam_member" "gke" {
  crypto_key_id = google_kms_crypto_key.gke.id
  role          = "roles/cloudkms.cryptoKeyEncrypterDecrypter"
  member        = "serviceAccount:service-${data.google_project.current.number}@container-engine-robot.iam.gserviceaccount.com"
}

data "google_project" "current" {
  project_id = var.project_id
}