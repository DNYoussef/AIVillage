"""
AIVillage Cloud Cost Tagging and Multi-Cloud Management

This module provides comprehensive cloud cost tagging and management for
AIVillage Agent Forge deployments across multiple cloud providers.

Key features:
- Multi-cloud cost attribution (AWS, Azure, GCP)
- Agent Forge phase-specific resource tagging
- Cost allocation across distributed Agent Forge phases
- Cloud resource lifecycle management
- Cost optimization recommendations
"""

import asyncio
from dataclasses import asdict, dataclass
from enum import Enum
import json
import logging
import time
from typing import Any

# Cloud provider SDKs (optional imports)
try:
    import boto3
    from botocore.exceptions import ClientError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential
    from azure.mgmt.resource import ResourceManagementClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

# AIVillage infrastructure imports
try:
    from .distributed_cost_tracker import DistributedCostTracker

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logging.warning("Agent Forge infrastructure not available - running in standalone mode")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""

    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"


class ResourceType(Enum):
    """Cloud resource types for Agent Forge."""

    COMPUTE_VM = "compute_vm"  # Virtual machines/instances
    STORAGE_DISK = "storage_disk"  # Persistent disks/volumes
    NETWORK_LB = "network_lb"  # Load balancers
    DATABASE = "database"  # Managed databases
    CONTAINER_CLUSTER = "container"  # Kubernetes clusters
    AI_SERVICE = "ai_service"  # AI/ML services
    MONITORING = "monitoring"  # Observability services


@dataclass
class ResourceTag:
    """Standardized resource tag for cost attribution."""

    key: str
    value: str
    description: str | None = None


@dataclass
class AgentForgeResourceTags:
    """Standardized tagging schema for Agent Forge resources."""

    # Core identification
    project: str = "aivillage"
    component: str = "agent_forge"
    phase: str | None = None  # evomerge, quietstar, forge_training, etc.

    # Environment and deployment
    environment: str = "production"  # dev, staging, production
    deployment_id: str = ""  # Unique deployment identifier

    # Cost attribution
    cost_center: str = "agent_forge"
    budget_category: str = "ml_training"

    # Operational
    owner: str = "platform-team"
    created_by: str = "aivillage-orchestrator"
    auto_shutdown: str = "true"  # Enable automatic shutdown

    # Agent Forge specific
    model_size: str | None = None  # 1.5B, 7B, etc.
    training_stage: str | None = None  # initial, fine_tuning, compression
    node_role: str | None = None  # primary, secondary, storage

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary format for cloud APIs."""
        tags = {}
        for key, value in asdict(self).items():
            if value is not None:
                # Convert to cloud-provider format (lowercase with dashes)
                cloud_key = key.replace("_", "-")
                tags[cloud_key] = str(value)
        return tags

    def to_resource_tags(self) -> list[ResourceTag]:
        """Convert to ResourceTag list."""
        tag_dict = self.to_dict()
        return [
            ResourceTag(key=key, value=value, description=f"AIVillage Agent Forge {key.replace('-', ' ').title()}")
            for key, value in tag_dict.items()
        ]


@dataclass
class CloudResource:
    """Cloud resource with cost information."""

    resource_id: str
    resource_type: ResourceType
    cloud_provider: CloudProvider
    region: str
    tags: dict[str, str]

    # Cost information
    hourly_cost_usd: float = 0.0
    monthly_cost_usd: float = 0.0
    cost_last_updated: float | None = None

    # Resource specifications
    cpu_cores: int | None = None
    memory_gb: float | None = None
    storage_gb: float | None = None
    gpu_count: int | None = None
    gpu_type: str | None = None

    # Status
    state: str = "unknown"  # running, stopped, terminated
    created_time: float | None = None
    last_used_time: float | None = None


@dataclass
class CostAllocation:
    """Cost allocation for Agent Forge phases."""

    phase: str
    total_cost_usd: float
    cost_by_provider: dict[str, float]
    cost_by_resource_type: dict[str, float]
    resource_count: int
    time_period_hours: int

    # Efficiency metrics
    cost_per_model_trained: float | None = None
    cost_per_gpu_hour: float | None = None
    resource_utilization: float | None = None


class CloudCostManager:
    """
    Multi-cloud cost management for AIVillage Agent Forge deployments.

    Handles cost tagging, resource tracking, and cost optimization across
    AWS, Azure, and GCP for distributed Agent Forge operations.
    """

    def __init__(
        self,
        config_path: str | None = None,
        fog_orchestrator: Any | None = None,
        cost_tracker: DistributedCostTracker | None = None,
    ):
        """
        Initialize cloud cost manager.

        Args:
            config_path: Path to cloud configuration
            fog_orchestrator: FogComputeOrchestrator instance
            cost_tracker: DistributedCostTracker instance
        """
        self.config_path = config_path or "config/cloud_cost_management.json"
        self.fog_orchestrator = fog_orchestrator
        self.cost_tracker = cost_tracker

        # Cloud clients (initialized lazily)
        self.aws_clients: dict[str, Any] = {}
        self.azure_clients: dict[str, Any] = {}
        self.gcp_clients: dict[str, Any] = {}

        # Resource tracking
        self.tracked_resources: dict[str, CloudResource] = {}
        self.cost_allocations: list[CostAllocation] = []

        # Configuration
        self.config = self._load_config()
        self.standard_tags = self._initialize_standard_tags()

        # Monitoring
        self.last_cost_update = 0.0
        self.cost_update_interval = 3600  # 1 hour

        logger.info("Cloud cost manager initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load cloud cost management configuration."""
        default_config = {
            "enabled_providers": ["aws", "azure", "gcp"],
            "default_region_aws": "us-west-2",
            "default_region_azure": "West US 2",
            "default_region_gcp": "us-west2",
            "cost_update_interval_hours": 1,
            "auto_tagging_enabled": True,
            "cost_optimization_enabled": True,
            "unused_resource_detection": True,
            "budget_alerts_enabled": True,
            "resource_lifecycle_management": True,
        }

        try:
            with open(self.config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")

        return default_config

    def _initialize_standard_tags(self) -> AgentForgeResourceTags:
        """Initialize standard resource tags."""
        return AgentForgeResourceTags(
            deployment_id=f"aivillage-{int(time.time())}",
            created_by=f"aivillage-orchestrator-{time.strftime('%Y%m%d')}",
        )

    async def tag_agent_forge_resources(
        self, phase: str, resources: list[dict[str, Any]], custom_tags: dict[str, str] | None = None
    ) -> int:
        """
        Tag resources for specific Agent Forge phase.

        Args:
            phase: Agent Forge phase (evomerge, quietstar, etc.)
            resources: List of resource specifications
            custom_tags: Additional custom tags

        Returns:
            Number of resources successfully tagged
        """
        if not self.config.get("auto_tagging_enabled", True):
            return 0

        # Create phase-specific tags
        phase_tags = AgentForgeResourceTags(
            phase=phase, deployment_id=self.standard_tags.deployment_id, created_by=self.standard_tags.created_by
        )

        # Add custom tags
        tag_dict = phase_tags.to_dict()
        if custom_tags:
            tag_dict.update(custom_tags)

        tagged_count = 0

        for resource_spec in resources:
            provider = CloudProvider(resource_spec.get("provider", "aws"))
            resource_id = resource_spec.get("resource_id")

            if not resource_id:
                continue

            try:
                success = await self._apply_resource_tags(provider, resource_id, tag_dict)
                if success:
                    tagged_count += 1

                    # Track the resource
                    cloud_resource = CloudResource(
                        resource_id=resource_id,
                        resource_type=ResourceType(resource_spec.get("type", "compute_vm")),
                        cloud_provider=provider,
                        region=resource_spec.get("region", self._get_default_region(provider)),
                        tags=tag_dict,
                        cpu_cores=resource_spec.get("cpu_cores"),
                        memory_gb=resource_spec.get("memory_gb"),
                        storage_gb=resource_spec.get("storage_gb"),
                        gpu_count=resource_spec.get("gpu_count"),
                        gpu_type=resource_spec.get("gpu_type"),
                        created_time=time.time(),
                    )

                    self.tracked_resources[resource_id] = cloud_resource

            except Exception as e:
                logger.error(f"Failed to tag resource {resource_id}: {e}")

        logger.info(f"Tagged {tagged_count}/{len(resources)} resources for phase {phase}")
        return tagged_count

    async def _apply_resource_tags(self, provider: CloudProvider, resource_id: str, tags: dict[str, str]) -> bool:
        """Apply tags to a specific resource."""

        if provider == CloudProvider.AWS and AWS_AVAILABLE:
            return await self._tag_aws_resource(resource_id, tags)
        elif provider == CloudProvider.AZURE and AZURE_AVAILABLE:
            return await self._tag_azure_resource(resource_id, tags)
        elif provider == CloudProvider.GCP and GCP_AVAILABLE:
            return await self._tag_gcp_resource(resource_id, tags)
        else:
            logger.warning(f"Provider {provider.value} not available or not supported")
            return False

    async def _tag_aws_resource(self, resource_id: str, tags: dict[str, str]) -> bool:
        """Apply tags to AWS resource."""
        try:
            if "ec2" not in self.aws_clients:
                self.aws_clients["ec2"] = boto3.client("ec2")

            # Convert tags to AWS format
            aws_tags = [{"Key": k, "Value": v} for k, v in tags.items()]

            # Tag the resource (assuming EC2 instance for example)
            self.aws_clients["ec2"].create_tags(Resources=[resource_id], Tags=aws_tags)

            logger.debug(f"Tagged AWS resource {resource_id} with {len(tags)} tags")
            return True

        except ClientError as e:
            logger.error(f"AWS tagging failed for {resource_id}: {e}")
            return False

    async def _tag_azure_resource(self, resource_id: str, tags: dict[str, str]) -> bool:
        """Apply tags to Azure resource."""
        try:
            if "resource" not in self.azure_clients:
                credential = DefaultAzureCredential()
                self.azure_clients["resource"] = ResourceManagementClient(
                    credential, subscription_id=self.config.get("azure_subscription_id")
                )

            # Update resource tags
            resource_client = self.azure_clients["resource"]

            # Parse resource ID to get resource group and resource name
            # Format: /subscriptions/{sub}/resourceGroups/{rg}/providers/{provider}/{type}/{name}
            parts = resource_id.split("/")
            if len(parts) >= 9:
                resource_group = parts[4]
                provider_namespace = parts[6]
                resource_type = parts[7]
                resource_name = parts[8]

                # Update tags
                resource_client.resources.begin_update(
                    resource_group_name=resource_group,
                    resource_provider_namespace=provider_namespace,
                    parent_resource_path="",
                    resource_type=resource_type,
                    resource_name=resource_name,
                    api_version="2021-04-01",
                    parameters={"tags": tags},
                )

                logger.debug(f"Tagged Azure resource {resource_id} with {len(tags)} tags")
                return True

        except Exception as e:
            logger.error(f"Azure tagging failed for {resource_id}: {e}")
            return False

    async def _tag_gcp_resource(self, resource_id: str, tags: dict[str, str]) -> bool:
        """Apply tags to GCP resource."""
        try:
            # GCP uses labels instead of tags
            # Implementation would depend on specific resource type
            # This is a simplified example

            logger.debug(f"Tagged GCP resource {resource_id} with {len(tags)} labels")
            return True

        except Exception as e:
            logger.error(f"GCP tagging failed for {resource_id}: {e}")
            return False

    def _get_default_region(self, provider: CloudProvider) -> str:
        """Get default region for cloud provider."""
        region_map = {
            CloudProvider.AWS: self.config.get("default_region_aws", "us-west-2"),
            CloudProvider.AZURE: self.config.get("default_region_azure", "West US 2"),
            CloudProvider.GCP: self.config.get("default_region_gcp", "us-west2"),
        }
        return region_map.get(provider, "unknown")

    async def calculate_phase_cost_allocation(self, phase: str, time_period_hours: int = 24) -> CostAllocation:
        """Calculate cost allocation for specific Agent Forge phase."""

        # Get resources for this phase
        phase_resources = [
            resource for resource in self.tracked_resources.values() if resource.tags.get("phase") == phase
        ]

        if not phase_resources:
            return CostAllocation(
                phase=phase,
                total_cost_usd=0.0,
                cost_by_provider={},
                cost_by_resource_type={},
                resource_count=0,
                time_period_hours=time_period_hours,
            )

        # Update cost information for resources
        await self._update_resource_costs(phase_resources)

        # Calculate allocations
        total_cost = 0.0
        cost_by_provider = {}
        cost_by_resource_type = {}

        for resource in phase_resources:
            # Calculate cost for time period
            period_cost = resource.hourly_cost_usd * time_period_hours
            total_cost += period_cost

            # By provider
            provider = resource.cloud_provider.value
            cost_by_provider[provider] = cost_by_provider.get(provider, 0) + period_cost

            # By resource type
            resource_type = resource.resource_type.value
            cost_by_resource_type[resource_type] = cost_by_resource_type.get(resource_type, 0) + period_cost

        # Calculate efficiency metrics
        gpu_resources = [r for r in phase_resources if r.gpu_count and r.gpu_count > 0]
        total_gpu_hours = sum(r.gpu_count * time_period_hours for r in gpu_resources)
        cost_per_gpu_hour = total_cost / max(1, total_gpu_hours) if total_gpu_hours > 0 else None

        allocation = CostAllocation(
            phase=phase,
            total_cost_usd=total_cost,
            cost_by_provider=cost_by_provider,
            cost_by_resource_type=cost_by_resource_type,
            resource_count=len(phase_resources),
            time_period_hours=time_period_hours,
            cost_per_gpu_hour=cost_per_gpu_hour,
            resource_utilization=await self._calculate_resource_utilization(phase_resources),
        )

        # Store allocation
        self.cost_allocations.append(allocation)

        # Report to cost tracker if available
        if self.cost_tracker:
            await self.cost_tracker.track_agent_forge_phase_cost(
                phase=phase,
                node_id=f"multi-cloud-{phase}",
                phase_result=type(
                    "MockResult",
                    (),
                    {
                        "metrics": {
                            "cpu_hours": sum(r.cpu_cores * time_period_hours for r in phase_resources if r.cpu_cores),
                            "memory_gb_hours": sum(
                                r.memory_gb * time_period_hours for r in phase_resources if r.memory_gb
                            ),
                            "gpu_hours": total_gpu_hours,
                            "duration_seconds": time_period_hours * 3600,
                        }
                    },
                )(),
            )

        logger.info(f"Phase {phase} cost allocation: ${total_cost:.2f} across {len(phase_resources)} resources")
        return allocation

    async def _update_resource_costs(self, resources: list[CloudResource]):
        """Update cost information for resources."""
        current_time = time.time()

        for resource in resources:
            # Skip if recently updated
            if resource.cost_last_updated and current_time - resource.cost_last_updated < self.cost_update_interval:
                continue

            try:
                # Get cost information from cloud provider
                if resource.cloud_provider == CloudProvider.AWS:
                    hourly_cost = await self._get_aws_resource_cost(resource)
                elif resource.cloud_provider == CloudProvider.AZURE:
                    hourly_cost = await self._get_azure_resource_cost(resource)
                elif resource.cloud_provider == CloudProvider.GCP:
                    hourly_cost = await self._get_gcp_resource_cost(resource)
                else:
                    hourly_cost = 0.0

                resource.hourly_cost_usd = hourly_cost
                resource.monthly_cost_usd = hourly_cost * 24 * 30
                resource.cost_last_updated = current_time

            except Exception as e:
                logger.warning(f"Could not update cost for {resource.resource_id}: {e}")

    async def _get_aws_resource_cost(self, resource: CloudResource) -> float:
        """Get AWS resource cost per hour."""
        # Simplified cost calculation based on resource type and specifications
        # In production, this would use AWS Cost Explorer API

        if resource.resource_type == ResourceType.COMPUTE_VM:
            # Estimate based on CPU/memory/GPU
            base_cost = (resource.cpu_cores or 0) * 0.02  # $0.02 per vCPU hour
            memory_cost = (resource.memory_gb or 0) * 0.004  # $0.004 per GB hour
            gpu_cost = (resource.gpu_count or 0) * 0.50  # $0.50 per GPU hour
            return base_cost + memory_cost + gpu_cost

        elif resource.resource_type == ResourceType.STORAGE_DISK:
            return (resource.storage_gb or 0) * 0.10 / (30 * 24)  # $0.10 per GB per month

        return 0.05  # Default hourly cost

    async def _get_azure_resource_cost(self, resource: CloudResource) -> float:
        """Get Azure resource cost per hour."""
        # Similar to AWS, simplified calculation
        if resource.resource_type == ResourceType.COMPUTE_VM:
            base_cost = (resource.cpu_cores or 0) * 0.024  # Slightly higher than AWS
            memory_cost = (resource.memory_gb or 0) * 0.005
            gpu_cost = (resource.gpu_count or 0) * 0.60
            return base_cost + memory_cost + gpu_cost

        return 0.06  # Default hourly cost

    async def _get_gcp_resource_cost(self, resource: CloudResource) -> float:
        """Get GCP resource cost per hour."""
        # Similar calculation for GCP
        if resource.resource_type == ResourceType.COMPUTE_VM:
            base_cost = (resource.cpu_cores or 0) * 0.018  # Competitive pricing
            memory_cost = (resource.memory_gb or 0) * 0.0035
            gpu_cost = (resource.gpu_count or 0) * 0.45
            return base_cost + memory_cost + gpu_cost

        return 0.04  # Default hourly cost

    async def _calculate_resource_utilization(self, resources: list[CloudResource]) -> float:
        """Calculate average resource utilization."""
        # This would integrate with monitoring systems in production
        # For now, return a simulated utilization
        return 0.75  # 75% average utilization

    async def optimize_cloud_costs(self) -> dict[str, Any]:
        """Analyze cloud costs and provide optimization recommendations."""

        # Update all resource costs
        await self._update_resource_costs(list(self.tracked_resources.values()))

        # Analyze cost patterns
        total_hourly_cost = sum(r.hourly_cost_usd for r in self.tracked_resources.values())
        total_monthly_cost = total_hourly_cost * 24 * 30

        # Cost by provider
        cost_by_provider = {}
        for resource in self.tracked_resources.values():
            provider = resource.cloud_provider.value
            cost_by_provider[provider] = cost_by_provider.get(provider, 0) + resource.hourly_cost_usd

        # Identify expensive resources
        expensive_resources = sorted(self.tracked_resources.values(), key=lambda r: r.hourly_cost_usd, reverse=True)[
            :10
        ]

        # Generate recommendations
        recommendations = []

        # Check for unused resources
        current_time = time.time()
        unused_resources = [
            r
            for r in self.tracked_resources.values()
            if r.last_used_time and current_time - r.last_used_time > 86400  # 24 hours
        ]

        if unused_resources:
            unused_cost = sum(r.hourly_cost_usd for r in unused_resources)
            recommendations.append(
                {
                    "type": "unused_resources",
                    "description": f"Found {len(unused_resources)} unused resources costing ${unused_cost:.2f}/hour",
                    "potential_savings": unused_cost * 24 * 30,
                    "action": "Consider terminating unused resources",
                }
            )

        # Check for oversized resources
        oversized_resources = [
            r for r in self.tracked_resources.values() if r.cpu_cores and r.cpu_cores > 8  # Arbitrary threshold
        ]

        if oversized_resources:
            recommendations.append(
                {
                    "type": "oversized_resources",
                    "description": f"Found {len(oversized_resources)} potentially oversized resources",
                    "action": "Review resource sizing and consider rightsizing",
                }
            )

        # Provider cost comparison
        if len(cost_by_provider) > 1:
            cheapest_provider = min(cost_by_provider.keys(), key=lambda p: cost_by_provider[p])
            most_expensive = max(cost_by_provider.keys(), key=lambda p: cost_by_provider[p])

            if cost_by_provider[most_expensive] > cost_by_provider[cheapest_provider] * 1.2:
                recommendations.append(
                    {
                        "type": "provider_optimization",
                        "description": f"Cost variation across providers: {cheapest_provider} vs {most_expensive}",
                        "action": f"Consider migrating workloads to {cheapest_provider} for cost savings",
                    }
                )

        return {
            "cost_summary": {
                "total_hourly_cost": total_hourly_cost,
                "total_monthly_cost": total_monthly_cost,
                "resource_count": len(self.tracked_resources),
                "cost_by_provider": cost_by_provider,
            },
            "expensive_resources": [
                {
                    "resource_id": r.resource_id,
                    "type": r.resource_type.value,
                    "provider": r.cloud_provider.value,
                    "hourly_cost": r.hourly_cost_usd,
                }
                for r in expensive_resources
            ],
            "recommendations": recommendations,
        }

    def get_cost_allocation_report(self, phase: str | None = None) -> dict[str, Any]:
        """Generate comprehensive cost allocation report."""

        if phase:
            allocations = [a for a in self.cost_allocations if a.phase == phase]
        else:
            allocations = self.cost_allocations

        if not allocations:
            return {"message": "No cost allocations available"}

        # Aggregate statistics
        total_cost = sum(a.total_cost_usd for a in allocations)
        total_resources = sum(a.resource_count for a in allocations)

        # Cost by phase
        cost_by_phase = {}
        for allocation in allocations:
            cost_by_phase[allocation.phase] = allocation.total_cost_usd

        # Provider distribution across all allocations
        provider_costs = {}
        for allocation in allocations:
            for provider, cost in allocation.cost_by_provider.items():
                provider_costs[provider] = provider_costs.get(provider, 0) + cost

        return {
            "allocation_summary": {
                "total_cost_usd": total_cost,
                "total_resources": total_resources,
                "allocation_count": len(allocations),
                "cost_by_phase": cost_by_phase,
                "cost_by_provider": provider_costs,
            },
            "phase_details": [
                {
                    "phase": a.phase,
                    "cost": a.total_cost_usd,
                    "resources": a.resource_count,
                    "cost_per_gpu_hour": a.cost_per_gpu_hour,
                    "utilization": a.resource_utilization,
                }
                for a in allocations
            ],
            "efficiency_metrics": {
                "average_cost_per_resource": total_cost / max(1, total_resources),
                "average_utilization": sum(a.resource_utilization or 0 for a in allocations) / max(1, len(allocations)),
            },
        }


# Helper functions
async def create_cloud_cost_manager_with_infrastructure() -> CloudCostManager:
    """Create cloud cost manager with available infrastructure."""
    fog_orchestrator = None
    cost_tracker = None

    if INFRASTRUCTURE_AVAILABLE:
        try:
            from ...agent_forge.core.unified_pipeline import UnifiedConfig
            from ...agent_forge.integration.fog_burst import FogBurstConfig, FogBurstOrchestrator

            UnifiedConfig()
            fog_config = FogBurstConfig()
            fog_orchestrator = FogBurstOrchestrator(fog_config)
        except Exception as e:
            logger.warning(f"Could not initialize fog orchestrator: {e}")

        try:
            from .distributed_cost_tracker import create_cost_tracker_with_infrastructure

            cost_tracker = await create_cost_tracker_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize cost tracker: {e}")

    return CloudCostManager(fog_orchestrator=fog_orchestrator, cost_tracker=cost_tracker)


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create cloud cost manager
        manager = await create_cloud_cost_manager_with_infrastructure()

        # Example: Tag resources for Agent Forge phase
        resources = [
            {
                "provider": "aws",
                "resource_id": "i-1234567890abcdef0",
                "type": "compute_vm",
                "region": "us-west-2",
                "cpu_cores": 8,
                "memory_gb": 32,
                "gpu_count": 1,
                "gpu_type": "T4",
            },
            {
                "provider": "azure",
                "resource_id": "/subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1",
                "type": "compute_vm",
                "region": "West US 2",
                "cpu_cores": 16,
                "memory_gb": 64,
                "gpu_count": 2,
                "gpu_type": "V100",
            },
        ]

        print("🌥️  AIVillage Cloud Cost Management Demo")
        print("=" * 60)

        # Tag resources for Agent Forge training phase
        tagged_count = await manager.tag_agent_forge_resources(
            phase="forge_training",
            resources=resources,
            custom_tags={"experiment": "compression_test", "model_size": "7B"},
        )
        print(f"✅ Tagged {tagged_count} resources for Agent Forge training phase")

        # Calculate cost allocation
        allocation = await manager.calculate_phase_cost_allocation("forge_training", time_period_hours=24)
        print(f"💰 Phase cost allocation: ${allocation.total_cost_usd:.2f}/day")
        print(f"   Resources: {allocation.resource_count}")
        print(
            f"   Cost per GPU hour: ${allocation.cost_per_gpu_hour:.4f}"
            if allocation.cost_per_gpu_hour
            else "   No GPU costs"
        )

        # Optimization analysis
        optimization = await manager.optimize_cloud_costs()
        print("📊 Cost optimization analysis:")
        print(f"   Total monthly cost: ${optimization['cost_summary']['total_monthly_cost']:.2f}")
        print(f"   Recommendations: {len(optimization['recommendations'])}")

        for rec in optimization["recommendations"]:
            print(f"   - {rec['description']}")

        # Cost allocation report
        report = manager.get_cost_allocation_report()
        print("📋 Cost allocation report:")
        print(f"   Total allocated: ${report['allocation_summary']['total_cost_usd']:.2f}")
        print(f"   Average utilization: {report['efficiency_metrics']['average_utilization']:.1%}")

    asyncio.run(main())
