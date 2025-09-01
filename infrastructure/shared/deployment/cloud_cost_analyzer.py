"""
Cloud Cost Analysis and Optimization for AIVillage.

Provides comprehensive cost analysis for cloud deployments with optimization
recommendations for different cloud providers and deployment scenarios.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITAL_OCEAN = "digitalocean"
    LINODE = "linode"
    HETZNER = "hetzner"
    VULTR = "vultr"


class DeploymentType(Enum):
    """Types of deployment configurations."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_AVAILABILITY = "high_availability"
    EDGE_DISTRIBUTED = "edge_distributed"


class ResourceType(Enum):
    """Types of cloud resources."""

    COMPUTE = "compute"
    STORAGE = "storage"
    DATABASE = "database"
    NETWORKING = "networking"
    LOAD_BALANCER = "load_balancer"
    CDN = "cdn"
    MONITORING = "monitoring"
    BACKUP = "backup"


@dataclass
class ResourceSpecification:
    """Specification for a cloud resource."""

    resource_type: ResourceType
    name: str
    description: str

    # Compute specs
    cpu_cores: int | None = None
    memory_gb: int | None = None
    storage_gb: int | None = None

    # Database specs
    database_type: str | None = None
    read_replicas: int = 0

    # Networking specs
    bandwidth_gb: int | None = None
    requests_per_month: int | None = None

    # Instance specs
    instance_count: int = 1
    uptime_hours_per_month: int = 744  # 24 * 31

    # Additional metadata
    tags: dict[str, str] = field(default_factory=dict)
    required: bool = True
    scaling_factor: float = 1.0


@dataclass
class CostEstimate:
    """Cost estimation for a resource."""

    resource_spec: ResourceSpecification
    provider: CloudProvider
    region: str

    # Cost breakdown
    hourly_cost: Decimal
    monthly_cost: Decimal
    annual_cost: Decimal

    # Cost components
    compute_cost: Decimal | None = None
    storage_cost: Decimal | None = None
    network_cost: Decimal | None = None
    other_costs: dict[str, Decimal] = field(default_factory=dict)

    # Additional info
    instance_type: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class DeploymentCostAnalysis:
    """Complete cost analysis for a deployment."""

    deployment_type: DeploymentType
    provider: CloudProvider
    region: str

    resource_costs: list[CostEstimate]
    total_monthly_cost: Decimal
    total_annual_cost: Decimal

    # Optimization recommendations
    potential_savings: dict[str, Decimal] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    # Analysis metadata
    analysis_date: datetime = field(default_factory=datetime.utcnow)
    confidence_level: float = 0.85  # 85% confidence in estimates

    def add_recommendation(self, category: str, description: str, potential_savings: Decimal = Decimal("0")):
        """Add optimization recommendation."""
        self.recommendations.append(f"{category}: {description}")
        if potential_savings > 0:
            self.potential_savings[category] = potential_savings


class CloudCostAnalyzer:
    """Cloud cost analyzer for AIVillage deployments."""

    def __init__(self):
        """Initialize cost analyzer."""
        self.pricing_data = self._load_pricing_data()
        self.deployment_configs = self._load_deployment_configs()

        # Regional cost multipliers (USD base pricing)
        self.region_multipliers = {
            "us-east-1": 1.0,  # AWS Virginia (baseline)
            "us-west-2": 1.05,  # AWS Oregon
            "eu-west-1": 1.15,  # AWS Ireland
            "ap-southeast-1": 1.2,  # AWS Singapore
            "ap-northeast-1": 1.25,  # AWS Tokyo
        }

        logger.info("Cloud cost analyzer initialized")

    def _load_pricing_data(self) -> dict[str, Any]:
        """Load cloud provider pricing data."""
        # In production, this would load from external pricing APIs
        # For now, we'll use estimated pricing based on recent data (August 2025)
        return {
            "aws": {
                "compute": {
                    "t3.micro": {"cpu": 2, "memory": 1, "hourly": 0.0104},
                    "t3.small": {"cpu": 2, "memory": 2, "hourly": 0.0208},
                    "t3.medium": {"cpu": 2, "memory": 4, "hourly": 0.0416},
                    "t3.large": {"cpu": 2, "memory": 8, "hourly": 0.0832},
                    "t3.xlarge": {"cpu": 4, "memory": 16, "hourly": 0.1664},
                    "t3.2xlarge": {"cpu": 8, "memory": 32, "hourly": 0.3328},
                    "m5.large": {"cpu": 2, "memory": 8, "hourly": 0.096},
                    "m5.xlarge": {"cpu": 4, "memory": 16, "hourly": 0.192},
                    "m5.2xlarge": {"cpu": 8, "memory": 32, "hourly": 0.384},
                    "m5.4xlarge": {"cpu": 16, "memory": 64, "hourly": 0.768},
                    "c5.large": {"cpu": 2, "memory": 4, "hourly": 0.085},
                    "c5.xlarge": {"cpu": 4, "memory": 8, "hourly": 0.17},
                    "c5.2xlarge": {"cpu": 8, "memory": 16, "hourly": 0.34},
                    "r5.large": {"cpu": 2, "memory": 16, "hourly": 0.126},
                    "r5.xlarge": {"cpu": 4, "memory": 32, "hourly": 0.252},
                },
                "storage": {
                    "gp3": 0.08,  # per GB/month
                    "gp2": 0.10,  # per GB/month
                    "io1": 0.125,  # per GB/month
                    "s3_standard": 0.023,  # per GB/month
                    "s3_infrequent": 0.0125,  # per GB/month
                },
                "database": {
                    "rds_postgres_small": {"instance": "db.t3.micro", "hourly": 0.017},
                    "rds_postgres_medium": {"instance": "db.t3.small", "hourly": 0.034},
                    "rds_postgres_large": {"instance": "db.t3.medium", "hourly": 0.068},
                    "redis_small": {"node": "cache.t3.micro", "hourly": 0.017},
                    "redis_medium": {"node": "cache.t3.small", "hourly": 0.034},
                },
                "networking": {
                    "alb": 0.0225,  # per hour
                    "nlb": 0.0225,  # per hour
                    "data_transfer_out": 0.09,  # per GB
                    "cloudfront": 0.085,  # per GB
                },
            },
            "azure": {
                "compute": {
                    "B1s": {"cpu": 1, "memory": 1, "hourly": 0.0104},
                    "B2s": {"cpu": 2, "memory": 4, "hourly": 0.0416},
                    "D2s_v3": {"cpu": 2, "memory": 8, "hourly": 0.096},
                    "D4s_v3": {"cpu": 4, "memory": 16, "hourly": 0.192},
                    "D8s_v3": {"cpu": 8, "memory": 32, "hourly": 0.384},
                },
                "storage": {
                    "standard_lrs": 0.045,  # per GB/month
                    "premium_lrs": 0.15,  # per GB/month
                    "blob_hot": 0.018,  # per GB/month
                },
                "database": {
                    "postgres_basic": 0.017,  # per hour
                    "postgres_general": 0.068,  # per hour
                    "redis_basic": 0.016,  # per hour
                },
            },
            "gcp": {
                "compute": {
                    "e2-micro": {"cpu": 1, "memory": 1, "hourly": 0.008},
                    "e2-small": {"cpu": 1, "memory": 2, "hourly": 0.016},
                    "e2-medium": {"cpu": 1, "memory": 4, "hourly": 0.032},
                    "n1-standard-1": {"cpu": 1, "memory": 3.75, "hourly": 0.0475},
                    "n1-standard-2": {"cpu": 2, "memory": 7.5, "hourly": 0.095},
                    "n1-standard-4": {"cpu": 4, "memory": 15, "hourly": 0.19},
                },
                "storage": {
                    "standard": 0.04,  # per GB/month
                    "ssd": 0.17,  # per GB/month
                    "gcs_standard": 0.02,  # per GB/month
                },
                "database": {
                    "postgres_micro": 0.017,  # per hour
                    "postgres_small": 0.068,  # per hour
                    "redis_basic": 0.049,  # per hour
                },
            },
            "digitalocean": {
                "compute": {
                    "basic-1vcpu-1gb": {"cpu": 1, "memory": 1, "monthly": 4},
                    "basic-2vcpu-2gb": {"cpu": 2, "memory": 2, "monthly": 12},
                    "basic-2vcpu-4gb": {"cpu": 2, "memory": 4, "monthly": 24},
                    "basic-4vcpu-8gb": {"cpu": 4, "memory": 8, "monthly": 48},
                    "general-2vcpu-8gb": {"cpu": 2, "memory": 8, "monthly": 56},
                    "general-4vcpu-16gb": {"cpu": 4, "memory": 16, "monthly": 112},
                },
                "storage": {
                    "volume": 0.10,  # per GB/month
                    "spaces": 0.02,  # per GB/month
                },
                "database": {
                    "postgres_basic": 15,  # per month
                    "postgres_professional": 60,  # per month
                    "redis_basic": 15,  # per month
                },
            },
        }

    def _load_deployment_configs(self) -> dict[DeploymentType, dict[str, Any]]:
        """Load deployment configurations."""
        return {
            DeploymentType.DEVELOPMENT: {
                "description": "Minimal development environment",
                "resources": [
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="app-server",
                        description="Main application server",
                        cpu_cores=2,
                        memory_gb=4,
                        storage_gb=20,
                        instance_count=1,
                        uptime_hours_per_month=200,  # Not 24/7
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="postgres",
                        description="PostgreSQL database",
                        database_type="postgres",
                        cpu_cores=1,
                        memory_gb=2,
                        storage_gb=20,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.STORAGE,
                        name="object-storage",
                        description="File and backup storage",
                        storage_gb=50,
                    ),
                ],
            },
            DeploymentType.STAGING: {
                "description": "Staging environment for testing",
                "resources": [
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="app-server",
                        description="Application servers",
                        cpu_cores=4,
                        memory_gb=8,
                        storage_gb=40,
                        instance_count=2,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="postgres",
                        description="PostgreSQL database",
                        database_type="postgres",
                        cpu_cores=2,
                        memory_gb=8,
                        storage_gb=100,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="redis",
                        description="Redis cache",
                        database_type="redis",
                        cpu_cores=1,
                        memory_gb=2,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.LOAD_BALANCER,
                        name="load-balancer",
                        description="Application load balancer",
                        bandwidth_gb=100,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.STORAGE,
                        name="object-storage",
                        description="File and backup storage",
                        storage_gb=200,
                    ),
                ],
            },
            DeploymentType.PRODUCTION: {
                "description": "Production environment",
                "resources": [
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="app-server",
                        description="Application servers",
                        cpu_cores=8,
                        memory_gb=32,
                        storage_gb=100,
                        instance_count=3,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="agent_forge",
                        description="Agent Forge training servers",
                        cpu_cores=16,
                        memory_gb=64,
                        storage_gb=500,
                        instance_count=2,
                        uptime_hours_per_month=400,  # Training workloads
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="postgres-primary",
                        description="Primary PostgreSQL database",
                        database_type="postgres",
                        cpu_cores=8,
                        memory_gb=32,
                        storage_gb=500,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="postgres-replica",
                        description="PostgreSQL read replica",
                        database_type="postgres",
                        cpu_cores=4,
                        memory_gb=16,
                        storage_gb=500,
                        read_replicas=2,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="redis-cluster",
                        description="Redis cluster",
                        database_type="redis",
                        cpu_cores=4,
                        memory_gb=16,
                        instance_count=3,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="neo4j",
                        description="Neo4j graph database",
                        database_type="neo4j",
                        cpu_cores=8,
                        memory_gb=32,
                        storage_gb=200,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.DATABASE,
                        name="qdrant",
                        description="Qdrant vector database",
                        database_type="qdrant",
                        cpu_cores=4,
                        memory_gb=16,
                        storage_gb=100,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.LOAD_BALANCER,
                        name="load-balancer",
                        description="Application load balancer",
                        bandwidth_gb=1000,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.CDN,
                        name="cdn",
                        description="Content delivery network",
                        bandwidth_gb=500,
                        requests_per_month=10000000,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.STORAGE,
                        name="object-storage",
                        description="File and backup storage",
                        storage_gb=2000,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.BACKUP,
                        name="backup-storage",
                        description="Automated backup storage",
                        storage_gb=5000,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.MONITORING,
                        name="monitoring",
                        description="Prometheus/Grafana monitoring",
                        cpu_cores=4,
                        memory_gb=16,
                        storage_gb=200,
                    ),
                ],
            },
            DeploymentType.HIGH_AVAILABILITY: {
                "description": "High availability production with multi-AZ",
                "resources": [
                    # Similar to production but with higher redundancy
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="app-server",
                        description="Application servers (multi-AZ)",
                        cpu_cores=8,
                        memory_gb=32,
                        storage_gb=100,
                        instance_count=6,  # 2x for HA
                        scaling_factor=1.5,  # Auto-scaling overhead
                    ),
                    # Add more HA-specific resources...
                ],
            },
            DeploymentType.EDGE_DISTRIBUTED: {
                "description": "Edge-distributed deployment with P2P nodes",
                "resources": [
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="edge-nodes",
                        description="Distributed edge computing nodes",
                        cpu_cores=4,
                        memory_gb=8,
                        storage_gb=100,
                        instance_count=10,  # Multiple regions
                        uptime_hours_per_month=744,
                    ),
                    ResourceSpecification(
                        resource_type=ResourceType.COMPUTE,
                        name="p2p-coordinators",
                        description="P2P network coordinators",
                        cpu_cores=2,
                        memory_gb=4,
                        storage_gb=50,
                        instance_count=5,
                    ),
                ],
            },
        }

    def estimate_resource_cost(
        self, resource_spec: ResourceSpecification, provider: CloudProvider, region: str = "us-east-1"
    ) -> CostEstimate:
        """Estimate cost for a single resource."""
        provider_pricing = self.pricing_data.get(provider.value, {})
        region_multiplier = self.region_multipliers.get(region, 1.0)

        hourly_cost = Decimal("0")
        compute_cost = Decimal("0")
        storage_cost = Decimal("0")
        network_cost = Decimal("0")
        other_costs = {}
        instance_type = None
        notes = []

        if resource_spec.resource_type == ResourceType.COMPUTE:
            # Find best matching instance type
            instance_type, instance_hourly = self._find_best_instance(
                resource_spec, provider_pricing.get("compute", {})
            )
            if instance_type:
                compute_cost = Decimal(str(instance_hourly)) * resource_spec.instance_count
                hourly_cost += compute_cost
                notes.append(f"Selected instance type: {instance_type}")

            # Add storage cost
            if resource_spec.storage_gb:
                storage_type = "gp3" if provider == CloudProvider.AWS else "standard"
                storage_price = Decimal(str(provider_pricing.get("storage", {}).get(storage_type, 0.08)))
                storage_cost = (storage_price * resource_spec.storage_gb * resource_spec.instance_count) / Decimal(
                    "744"
                )  # Monthly to hourly
                hourly_cost += storage_cost

        elif resource_spec.resource_type == ResourceType.DATABASE:
            # Database-specific pricing
            db_key = f"{resource_spec.database_type}_small"
            if resource_spec.memory_gb >= 16:
                db_key = f"{resource_spec.database_type}_large"
            elif resource_spec.memory_gb >= 8:
                db_key = f"{resource_spec.database_type}_medium"

            db_pricing = provider_pricing.get("database", {}).get(db_key, {"hourly": 0.068})
            if isinstance(db_pricing, dict):
                hourly_rate = db_pricing.get("hourly", 0.068)
            else:
                hourly_rate = db_pricing / 744  # Monthly to hourly

            compute_cost = Decimal(str(hourly_rate)) * (1 + resource_spec.read_replicas)
            hourly_cost += compute_cost

            # Database storage
            if resource_spec.storage_gb:
                storage_price = Decimal(str(provider_pricing.get("storage", {}).get("gp3", 0.08)))
                storage_cost = (storage_price * resource_spec.storage_gb * (1 + resource_spec.read_replicas)) / Decimal(
                    "744"
                )
                hourly_cost += storage_cost

        elif resource_spec.resource_type == ResourceType.STORAGE:
            storage_price = Decimal(str(provider_pricing.get("storage", {}).get("s3_standard", 0.023)))
            storage_cost = (storage_price * resource_spec.storage_gb) / Decimal("744")  # Monthly to hourly
            hourly_cost = storage_cost

        elif resource_spec.resource_type == ResourceType.LOAD_BALANCER:
            lb_hourly = Decimal(str(provider_pricing.get("networking", {}).get("alb", 0.0225)))
            hourly_cost = lb_hourly
            if resource_spec.bandwidth_gb:
                transfer_cost = Decimal(str(provider_pricing.get("networking", {}).get("data_transfer_out", 0.09)))
                network_cost = (transfer_cost * resource_spec.bandwidth_gb) / Decimal("744")
                hourly_cost += network_cost

        elif resource_spec.resource_type == ResourceType.CDN:
            if resource_spec.bandwidth_gb:
                cdn_price = Decimal(str(provider_pricing.get("networking", {}).get("cloudfront", 0.085)))
                network_cost = (cdn_price * resource_spec.bandwidth_gb) / Decimal("744")
                hourly_cost = network_cost

        elif resource_spec.resource_type in [ResourceType.MONITORING, ResourceType.BACKUP]:
            # Use compute pricing for monitoring/backup services
            instance_type, instance_hourly = self._find_best_instance(
                resource_spec, provider_pricing.get("compute", {})
            )
            if instance_type:
                hourly_cost = Decimal(str(instance_hourly))

        # Apply region multiplier and scaling factor
        hourly_cost *= Decimal(str(region_multiplier * resource_spec.scaling_factor))

        # Calculate monthly and annual costs
        monthly_hours = Decimal(str(resource_spec.uptime_hours_per_month))
        monthly_cost = hourly_cost * monthly_hours
        annual_cost = monthly_cost * Decimal("12")

        return CostEstimate(
            resource_spec=resource_spec,
            provider=provider,
            region=region,
            hourly_cost=hourly_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
            monthly_cost=monthly_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            annual_cost=annual_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            compute_cost=compute_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if compute_cost else None,
            storage_cost=storage_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if storage_cost else None,
            network_cost=network_cost.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) if network_cost else None,
            other_costs={k: v.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) for k, v in other_costs.items()},
            instance_type=instance_type,
            notes=notes,
        )

    def _find_best_instance(self, resource_spec: ResourceSpecification, instance_pricing: dict[str, Any]) -> tuple:
        """Find the best matching instance type for resource requirements."""
        best_match = None
        best_cost = float("inf")
        best_hourly = 0

        required_cpu = resource_spec.cpu_cores or 1
        required_memory = resource_spec.memory_gb or 1

        for instance_type, specs in instance_pricing.items():
            if specs["cpu"] >= required_cpu and specs["memory"] >= required_memory:
                hourly_cost = specs.get("hourly", specs.get("monthly", 0) / 744)
                if hourly_cost < best_cost:
                    best_match = instance_type
                    best_cost = hourly_cost
                    best_hourly = hourly_cost

        return best_match, best_hourly

    def analyze_deployment(
        self, deployment_type: DeploymentType, provider: CloudProvider, region: str = "us-east-1"
    ) -> DeploymentCostAnalysis:
        """Analyze costs for a complete deployment."""
        logger.info(f"Analyzing {deployment_type.value} deployment on {provider.value} in {region}")

        deployment_config = self.deployment_configs.get(deployment_type, {})
        resources = deployment_config.get("resources", [])

        resource_costs = []
        total_monthly = Decimal("0")

        for resource_spec in resources:
            cost_estimate = self.estimate_resource_cost(resource_spec, provider, region)
            resource_costs.append(cost_estimate)
            total_monthly += cost_estimate.monthly_cost

        total_annual = total_monthly * Decimal("12")

        analysis = DeploymentCostAnalysis(
            deployment_type=deployment_type,
            provider=provider,
            region=region,
            resource_costs=resource_costs,
            total_monthly_cost=total_monthly.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_annual_cost=total_annual.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        )

        # Generate optimization recommendations
        self._generate_optimization_recommendations(analysis)

        logger.info(f"Analysis complete: ${total_monthly}/month, ${total_annual}/year")
        return analysis

    def _generate_optimization_recommendations(self, analysis: DeploymentCostAnalysis):
        """Generate cost optimization recommendations."""
        monthly_cost = analysis.total_monthly_cost

        # Reserved Instance recommendations (for AWS/Azure)
        if analysis.provider in [CloudProvider.AWS, CloudProvider.AZURE]:
            compute_costs = sum(
                cost.monthly_cost
                for cost in analysis.resource_costs
                if cost.resource_spec.resource_type == ResourceType.COMPUTE
            )
            if compute_costs > Decimal("100"):
                ri_savings = compute_costs * Decimal("0.3")  # 30% savings with RIs
                analysis.add_recommendation(
                    "Reserved Instances", "Use 1-year reserved instances for stable workloads to save ~30%", ri_savings
                )

        # Spot instance recommendations
        training_costs = sum(
            cost.monthly_cost for cost in analysis.resource_costs if "agent_forge" in cost.resource_spec.name.lower()
        )
        if training_costs > Decimal("50"):
            spot_savings = training_costs * Decimal("0.7")  # 70% savings with spot
            analysis.add_recommendation(
                "Spot Instances", "Use spot instances for Agent Forge training workloads to save ~70%", spot_savings
            )

        # Storage optimization
        storage_costs = sum(cost.storage_cost or Decimal("0") for cost in analysis.resource_costs)
        if storage_costs > Decimal("20"):
            storage_savings = storage_costs * Decimal("0.2")  # 20% savings with lifecycle
            analysis.add_recommendation(
                "Storage Lifecycle",
                "Implement storage lifecycle policies to move old data to cheaper tiers",
                storage_savings,
            )

        # Database optimization
        db_costs = sum(
            cost.monthly_cost
            for cost in analysis.resource_costs
            if cost.resource_spec.resource_type == ResourceType.DATABASE
        )
        if db_costs > Decimal("100"):
            db_savings = db_costs * Decimal("0.15")  # 15% savings with optimization
            analysis.add_recommendation(
                "Database Optimization", "Right-size database instances and use read replicas efficiently", db_savings
            )

        # CDN optimization
        cdn_costs = sum(
            cost.monthly_cost
            for cost in analysis.resource_costs
            if cost.resource_spec.resource_type == ResourceType.CDN
        )
        if cdn_costs > Decimal("20"):
            analysis.add_recommendation(
                "CDN Optimization",
                "Optimize cache policies and use compression to reduce bandwidth costs",
                cdn_costs * Decimal("0.25"),
            )

        # Multi-cloud recommendations
        if monthly_cost > Decimal("500"):
            analysis.add_recommendation(
                "Multi-Cloud Strategy",
                "Consider multi-cloud deployment for better pricing and redundancy",
                monthly_cost * Decimal("0.1"),
            )

        # Auto-scaling recommendations
        if analysis.deployment_type == DeploymentType.PRODUCTION:
            analysis.add_recommendation(
                "Auto-Scaling", "Implement auto-scaling to match capacity with demand", monthly_cost * Decimal("0.2")
            )

        # Global South optimization
        analysis.add_recommendation(
            "Global South Deployment",
            "Consider edge nodes in Global South regions for reduced latency and costs",
            monthly_cost * Decimal("0.15"),
        )

    def compare_providers(
        self, deployment_type: DeploymentType, providers: list[CloudProvider], region: str = "us-east-1"
    ) -> dict[CloudProvider, DeploymentCostAnalysis]:
        """Compare costs across multiple cloud providers."""
        logger.info(f"Comparing providers for {deployment_type.value} deployment")

        comparisons = {}
        for provider in providers:
            analysis = self.analyze_deployment(deployment_type, provider, region)
            comparisons[provider] = analysis

        return comparisons

    def generate_cost_report(self, analysis: DeploymentCostAnalysis, output_format: str = "detailed") -> dict[str, Any]:
        """Generate comprehensive cost report."""
        report = {
            "summary": {
                "deployment_type": analysis.deployment_type.value,
                "provider": analysis.provider.value,
                "region": analysis.region,
                "analysis_date": analysis.analysis_date.isoformat(),
                "confidence_level": analysis.confidence_level,
                "total_monthly_cost": float(analysis.total_monthly_cost),
                "total_annual_cost": float(analysis.total_annual_cost),
            },
            "resource_breakdown": [],
            "cost_categories": {},
            "optimization": {
                "recommendations": analysis.recommendations,
                "potential_monthly_savings": float(sum(analysis.potential_savings.values())),
                "optimized_monthly_cost": float(analysis.total_monthly_cost - sum(analysis.potential_savings.values())),
            },
        }

        # Resource breakdown
        for cost_estimate in analysis.resource_costs:
            resource_info = {
                "name": cost_estimate.resource_spec.name,
                "type": cost_estimate.resource_spec.resource_type.value,
                "description": cost_estimate.resource_spec.description,
                "monthly_cost": float(cost_estimate.monthly_cost),
                "annual_cost": float(cost_estimate.annual_cost),
                "instance_count": cost_estimate.resource_spec.instance_count,
                "instance_type": cost_estimate.instance_type,
                "notes": cost_estimate.notes,
            }

            if output_format == "detailed":
                resource_info.update(
                    {
                        "hourly_cost": float(cost_estimate.hourly_cost),
                        "compute_cost": float(cost_estimate.compute_cost) if cost_estimate.compute_cost else None,
                        "storage_cost": float(cost_estimate.storage_cost) if cost_estimate.storage_cost else None,
                        "network_cost": float(cost_estimate.network_cost) if cost_estimate.network_cost else None,
                        "specs": {
                            "cpu_cores": cost_estimate.resource_spec.cpu_cores,
                            "memory_gb": cost_estimate.resource_spec.memory_gb,
                            "storage_gb": cost_estimate.resource_spec.storage_gb,
                        },
                    }
                )

            report["resource_breakdown"].append(resource_info)

        # Cost categories
        categories = {}
        for cost_estimate in analysis.resource_costs:
            category = cost_estimate.resource_spec.resource_type.value
            if category not in categories:
                categories[category] = Decimal("0")
            categories[category] += cost_estimate.monthly_cost

        report["cost_categories"] = {k: float(v) for k, v in categories.items()}

        return report

    def save_analysis(self, analysis: DeploymentCostAnalysis, output_path: str | Path):
        """Save cost analysis to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = self.generate_cost_report(analysis, output_format="detailed")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Cost analysis saved to {output_path}")


async def create_cost_analyzer() -> CloudCostAnalyzer:
    """Create and initialize cloud cost analyzer."""
    analyzer = CloudCostAnalyzer()
    logger.info("Cloud cost analyzer created successfully")
    return analyzer


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        analyzer = await create_cost_analyzer()

        # Analyze production deployment on AWS
        analysis = analyzer.analyze_deployment(DeploymentType.PRODUCTION, CloudProvider.AWS, "us-east-1")

        print(f"Monthly Cost: ${analysis.total_monthly_cost}")
        print(f"Annual Cost: ${analysis.total_annual_cost}")
        print(f"Potential Savings: ${sum(analysis.potential_savings.values())}/month")

        # Compare providers
        comparisons = analyzer.compare_providers(
            DeploymentType.PRODUCTION, [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP], "us-east-1"
        )

        print("\nProvider Comparison:")
        for provider, analysis in comparisons.items():
            print(f"{provider.value}: ${analysis.total_monthly_cost}/month")

        # Save detailed report
        report = analyzer.generate_cost_report(analysis)
        with open("cost_analysis.json", "w") as f:
            json.dump(report, f, indent=2)

    asyncio.run(main())
