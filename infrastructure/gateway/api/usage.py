"""
Fog Usage API

Provides REST endpoints for usage tracking and billing:
- GET /v1/fog/usage - Get usage metrics by namespace
- GET /v1/fog/usage/pricing - Get current pricing information
- GET /v1/fog/usage/quotas - Get namespace quotas and limits
- POST /v1/fog/usage/quotas - Update namespace quotas (admin)

Integrates with existing AIVillage billing and quota systems.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UsagePeriod(str, Enum):
    """Usage reporting periods"""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class ResourceType(str, Enum):
    """Billable resource types"""

    CPU_SECONDS = "cpu_seconds"
    MEMORY_MB_HOURS = "memory_mb_hours"
    DISK_MB_HOURS = "disk_mb_hours"
    NETWORK_EGRESS_BYTES = "network_egress_bytes"
    JOB_EXECUTIONS = "job_executions"
    SANDBOX_HOURS = "sandbox_hours"


@dataclass
class UsageMetric:
    """Usage metric data point"""

    namespace: str
    resource_type: ResourceType
    quantity: float
    unit_price: float
    cost: float
    timestamp: datetime
    period: UsagePeriod


class NamespaceUsage(BaseModel):
    """Usage summary for a namespace"""

    namespace: str
    period_start: datetime
    period_end: datetime

    # Resource usage
    cpu_seconds: float = Field(0.0, description="Total CPU seconds consumed")
    memory_mb_hours: float = Field(0.0, description="Memory MB-hours consumed")
    disk_mb_hours: float = Field(0.0, description="Disk MB-hours consumed")
    network_egress_bytes: int = Field(0, description="Network egress bytes")

    # Service usage
    job_executions: int = Field(0, description="Number of jobs executed")
    sandbox_hours: float = Field(0.0, description="Sandbox hours")

    # Cost breakdown
    cpu_cost: float = Field(0.0, description="CPU cost in USD")
    memory_cost: float = Field(0.0, description="Memory cost in USD")
    disk_cost: float = Field(0.0, description="Disk cost in USD")
    network_cost: float = Field(0.0, description="Network cost in USD")
    service_cost: float = Field(0.0, description="Service execution cost in USD")

    # Totals
    total_cost: float = Field(0.0, description="Total cost in USD")

    # Utilization metrics
    avg_cpu_utilization: float = Field(0.0, description="Average CPU utilization %")
    peak_memory_mb: int = Field(0, description="Peak memory usage MB")


class PricingInfo(BaseModel):
    """Current pricing information"""

    effective_date: datetime
    currency: str = "USD"

    # Resource pricing (per unit)
    cpu_second_price: float = Field(0.0001, description="Price per CPU second")
    memory_mb_hour_price: float = Field(0.00001, description="Price per MB-hour")
    disk_mb_hour_price: float = Field(0.000001, description="Price per disk MB-hour")
    network_egress_gb_price: float = Field(0.01, description="Price per GB egress")

    # Service pricing
    job_execution_price: float = Field(0.001, description="Price per job execution")
    sandbox_hour_price: float = Field(0.10, description="Price per sandbox hour")

    # Priority multipliers
    priority_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "B": 1.0,  # Best effort
            "A": 2.0,  # Standard (replicated)
            "S": 5.0,  # Premium (replicated + attested)
        },
        description="Priority class pricing multipliers",
    )


class NamespaceQuota(BaseModel):
    """Namespace resource quotas"""

    namespace: str

    # Resource limits
    max_concurrent_jobs: int = Field(100, description="Maximum concurrent jobs")
    max_concurrent_sandboxes: int = Field(10, description="Maximum concurrent sandboxes")
    max_cpu_cores: float = Field(100.0, description="Maximum CPU cores")
    max_memory_mb: int = Field(102400, description="Maximum memory MB (100GB)")
    max_disk_mb: int = Field(1048576, description="Maximum disk MB (1TB)")

    # Cost limits
    daily_cost_limit: float = Field(1000.0, description="Daily cost limit USD")
    monthly_cost_limit: float = Field(10000.0, description="Monthly cost limit USD")

    # Network limits
    daily_egress_gb_limit: int = Field(100, description="Daily egress limit GB")

    # Current usage against quotas
    current_jobs: int = Field(0, description="Current running jobs")
    current_sandboxes: int = Field(0, description="Current active sandboxes")
    current_cpu_cores: float = Field(0.0, description="Current CPU allocation")
    current_memory_mb: int = Field(0, description="Current memory allocation")

    # Cost tracking
    daily_cost_used: float = Field(0.0, description="Today's cost usage")
    monthly_cost_used: float = Field(0.0, description="This month's cost usage")


class UsageAPI:
    """Usage API implementation"""

    def __init__(self):
        """Initialize usage API"""
        self.router = APIRouter(prefix="/v1/fog/usage", tags=["usage"])
        self._setup_routes()

        # Reference data for demonstration
        self._sample_usage_data: dict[str, list[UsageMetric]] = {}
        self._sample_quotas: dict[str, NamespaceQuota] = {}
        self._current_pricing = PricingInfo(effective_date=datetime.now(UTC))

    def _setup_routes(self):
        """Setup API routes"""

        @self.router.get(
            "",
            response_model=list[NamespaceUsage],
            summary="Get usage metrics",
            description="Get resource usage and cost metrics for accessible namespaces",
        )
        async def get_usage(
            namespace: str | None = None,
            period: UsagePeriod = UsagePeriod.DAY,
            start_time: datetime | None = None,
            end_time: datetime | None = None,
            # Production RBAC integration required
            # current_user: User = Security(get_current_user, scopes=["fog.usage.read"])
        ) -> list[NamespaceUsage]:
            """Get usage metrics for namespaces"""

            # Default time range based on period
            if not end_time:
                end_time = datetime.now(UTC)

            if not start_time:
                if period == UsagePeriod.HOUR:
                    start_time = end_time - timedelta(hours=1)
                elif period == UsagePeriod.DAY:
                    start_time = end_time - timedelta(days=1)
                elif period == UsagePeriod.WEEK:
                    start_time = end_time - timedelta(weeks=1)
                else:  # MONTH
                    start_time = end_time - timedelta(days=30)

            # Reference usage data
            reference_namespaces = ["demo/team1", "prod/backend", "dev/frontend"]

            if namespace:
                reference_namespaces = [namespace] if namespace in reference_namespaces else []

            usage_reports = []
            for ns in reference_namespaces:
                # Generate mock usage data
                usage = NamespaceUsage(
                    namespace=ns,
                    period_start=start_time,
                    period_end=end_time,
                    cpu_seconds=3600.5,  # 1 hour of CPU
                    memory_mb_hours=2048.0,  # 2GB for 1 hour
                    disk_mb_hours=5120.0,  # 5GB for 1 hour
                    network_egress_bytes=1073741824,  # 1GB
                    job_executions=25,
                    sandbox_hours=2.5,
                    cpu_cost=0.36,
                    memory_cost=0.02,
                    disk_cost=0.005,
                    network_cost=0.01,
                    service_cost=0.275,  # Jobs + sandboxes
                    total_cost=0.67,
                    avg_cpu_utilization=45.2,
                    peak_memory_mb=4096,
                )
                usage_reports.append(usage)

            return usage_reports

        @self.router.get(
            "/pricing",
            response_model=PricingInfo,
            summary="Get pricing information",
            description="Get current pricing for all fog compute resources",
        )
        async def get_pricing() -> PricingInfo:
            """Get current pricing information"""
            return self._current_pricing

        @self.router.get(
            "/quotas",
            response_model=list[NamespaceQuota],
            summary="Get namespace quotas",
            description="Get resource quotas for accessible namespaces",
        )
        async def get_quotas(
            namespace: str | None = None,
            # Reference implementation: RBAC dependency injection
        ) -> list[NamespaceQuota]:
            """Get namespace quotas and current usage"""

            # Mock quota data
            mock_namespaces = ["demo/team1", "prod/backend", "dev/frontend"]

            if namespace:
                if namespace not in mock_namespaces:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND, detail=f"Namespace {namespace} not found"
                    )
                mock_namespaces = [namespace]

            quotas = []
            for ns in reference_namespaces:
                quota = NamespaceQuota(
                    namespace=ns,
                    max_concurrent_jobs=100 if "prod" in ns else 50,
                    max_concurrent_sandboxes=20 if "prod" in ns else 10,
                    max_cpu_cores=200.0 if "prod" in ns else 50.0,
                    max_memory_mb=204800 if "prod" in ns else 51200,  # 200GB vs 50GB
                    daily_cost_limit=2000.0 if "prod" in ns else 100.0,
                    monthly_cost_limit=50000.0 if "prod" in ns else 2000.0,
                    current_jobs=5,
                    current_sandboxes=2,
                    current_cpu_cores=8.5,
                    current_memory_mb=16384,
                    daily_cost_used=45.67,
                    monthly_cost_used=567.89,
                )
                quotas.append(quota)

            return quotas

        @self.router.post(
            "/quotas",
            response_model=NamespaceQuota,
            summary="Update namespace quota",
            description="Update resource quotas for a namespace (admin only)",
        )
        async def update_quota(
            quota_update: NamespaceQuota,
            # Reference implementation: admin RBAC dependency
            # current_user: User = Security(get_current_user, scopes=["fog.admin.quotas"])
        ) -> NamespaceQuota:
            """Update namespace quota (admin operation)"""

            # Reference implementation: admin permission validation
            # Reference implementation: quota persistence to database

            # For now, just return the updated quota
            self._mock_quotas[quota_update.namespace] = quota_update

            logger.info(f"Updated quota for namespace {quota_update.namespace}")

            return quota_update

        @self.router.get(
            "/report", summary="Generate usage report", description="Generate detailed usage report (CSV/JSON)"
        )
        async def generate_report(
            format: str = Query("json", regex="^(json|csv)$"),
            namespace: str | None = None,
            period: UsagePeriod = UsagePeriod.MONTH,
            # Reference implementation: RBAC dependency injection
        ):
            """Generate detailed usage report"""

            # Reference implementation: comprehensive report generation
            # For now, return summary

            if format == "csv":
                # Reference implementation: CSV report generation
                return {
                    "message": "CSV report generation implementation pending yet",
                    "format": format,
                    "namespace": namespace,
                    "period": period.value,
                }
            else:
                # Return JSON summary
                usage_data = await get_usage(namespace=namespace, period=period)

                return {
                    "report_type": "usage_summary",
                    "generated_at": datetime.now(UTC),
                    "period": period.value,
                    "namespace_filter": namespace,
                    "data": usage_data,
                    "total_namespaces": len(usage_data),
                    "total_cost": sum(u.total_cost for u in usage_data),
                }


# Factory function
def create_usage_api() -> UsageAPI:
    """Create usage API instance"""
    return UsageAPI()
