# Qdrant development environment (placeholder)
terraform {
  required_version = ">= 1.0"
}

provider "docker" {}

resource "docker_container" "qdrant" {
  image = "qdrant/qdrant"
  name  = "qdrant-dev"
  ports {
    internal = 6333
    external = 6333
  }
}
