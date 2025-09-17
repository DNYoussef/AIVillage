# AWS Infrastructure for AIVillage Fog Computing Platform
# Production-ready deployment with auto-scaling, load balancing, and CDN

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
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

  backend "s3" {
    bucket         = "aivillage-terraform-state"
    key            = "fog-infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "aivillage-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "AIVillage"
      Environment = var.environment
      Component   = "FogComputing"
      ManagedBy   = "Terraform"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
resource "aws_vpc" "aivillage" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "aivillage-vpc-${var.environment}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "aivillage" {
  vpc_id = aws_vpc.aivillage.id

  tags = {
    Name = "aivillage-igw-${var.environment}"
  }
}

# Public Subnets
resource "aws_subnet" "public" {
  count = length(var.public_subnet_cidrs)

  vpc_id                  = aws_vpc.aivillage.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "aivillage-public-${count.index + 1}-${var.environment}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }
}

# Private Subnets
resource "aws_subnet" "private" {
  count = length(var.private_subnet_cidrs)

  vpc_id            = aws_vpc.aivillage.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "aivillage-private-${count.index + 1}-${var.environment}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# NAT Gateways
resource "aws_eip" "nat" {
  count = length(aws_subnet.public)

  domain = "vpc"
  depends_on = [aws_internet_gateway.aivillage]

  tags = {
    Name = "aivillage-nat-eip-${count.index + 1}-${var.environment}"
  }
}

resource "aws_nat_gateway" "aivillage" {
  count = length(aws_subnet.public)

  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "aivillage-nat-${count.index + 1}-${var.environment}"
  }

  depends_on = [aws_internet_gateway.aivillage]
}

# Route Tables
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.aivillage.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.aivillage.id
  }

  tags = {
    Name = "aivillage-public-rt-${var.environment}"
  }
}

resource "aws_route_table" "private" {
  count = length(aws_nat_gateway.aivillage)

  vpc_id = aws_vpc.aivillage.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.aivillage[count.index].id
  }

  tags = {
    Name = "aivillage-private-rt-${count.index + 1}-${var.environment}"
  }
}

# Route Table Associations
resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = length(aws_subnet.private)

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private[count.index].id
}

# Security Groups
resource "aws_security_group" "eks_cluster" {
  name_prefix = "aivillage-eks-cluster-"
  vpc_id      = aws_vpc.aivillage.id

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "aivillage-eks-cluster-sg-${var.environment}"
  }
}

resource "aws_security_group" "eks_node_group" {
  name_prefix = "aivillage-eks-node-"
  vpc_id      = aws_vpc.aivillage.id

  ingress {
    description = "Node to node"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Cluster to node"
    from_port   = 1025
    to_port     = 65535
    protocol    = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  ingress {
    description = "HTTPS from cluster"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "aivillage-eks-node-sg-${var.environment}"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "aivillage" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.kubernetes_version

  vpc_config {
    subnet_ids              = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    security_group_ids      = [aws_security_group.eks_cluster.id]
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = var.cluster_endpoint_public_access_cidrs
  }

  encryption_config {
    provider {
      key_arn = aws_kms_key.eks.arn
    }
    resources = ["secrets"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_iam_role_policy_attachment.eks_vpc_resource_controller,
    aws_cloudwatch_log_group.eks_cluster,
  ]

  tags = {
    Name = var.cluster_name
  }
}

# EKS Node Groups
resource "aws_eks_node_group" "general" {
  cluster_name    = aws_eks_cluster.aivillage.name
  node_group_name = "general"
  node_role_arn   = aws_iam_role.eks_node_group.arn
  subnet_ids      = aws_subnet.private[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = var.node_instance_types

  scaling_config {
    desired_size = var.node_group_desired_size
    max_size     = var.node_group_max_size
    min_size     = var.node_group_min_size
  }

  update_config {
    max_unavailable = 1
  }

  # Ensure that IAM Role permissions are created before and deleted after EKS Node Group handling.
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy,
  ]

  tags = {
    Name = "${var.cluster_name}-general-node-group"
  }
}

# EKS Addons
resource "aws_eks_addon" "vpc_cni" {
  cluster_name = aws_eks_cluster.aivillage.name
  addon_name   = "vpc-cni"
  addon_version = var.vpc_cni_version
  resolve_conflicts = "OVERWRITE"
}

resource "aws_eks_addon" "coredns" {
  cluster_name = aws_eks_cluster.aivillage.name
  addon_name   = "coredns"
  addon_version = var.coredns_version
  resolve_conflicts = "OVERWRITE"
}

resource "aws_eks_addon" "kube_proxy" {
  cluster_name = aws_eks_cluster.aivillage.name
  addon_name   = "kube-proxy"
  addon_version = var.kube_proxy_version
  resolve_conflicts = "OVERWRITE"
}

resource "aws_eks_addon" "ebs_csi_driver" {
  cluster_name = aws_eks_cluster.aivillage.name
  addon_name   = "aws-ebs-csi-driver"
  addon_version = var.ebs_csi_driver_version
  resolve_conflicts = "OVERWRITE"
}

# CloudWatch Log Group for EKS
resource "aws_cloudwatch_log_group" "eks_cluster" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = var.cloudwatch_log_retention_days
  kms_key_id        = aws_kms_key.cloudwatch.arn

  tags = {
    Name = "${var.cluster_name}-logs"
  }
}

# KMS Keys
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "aivillage-eks-secrets-key-${var.environment}"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/aivillage-eks-${var.environment}"
  target_key_id = aws_kms_key.eks.key_id
}

resource "aws_kms_key" "cloudwatch" {
  description             = "CloudWatch Logs Encryption Key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = {
    Name = "aivillage-cloudwatch-key-${var.environment}"
  }
}

resource "aws_kms_alias" "cloudwatch" {
  name          = "alias/aivillage-cloudwatch-${var.environment}"
  target_key_id = aws_kms_key.cloudwatch.key_id
}