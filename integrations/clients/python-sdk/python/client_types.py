"""
Type definitions for Fog Client components.

Centralized type definitions to reduce connascence of meaning
and provide single source of truth for data structures.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class FogClientError(Exception):
    """Base exception for fog client errors"""

    pass


class JobNotFoundError(FogClientError):
    """Job not found error"""

    pass


class SandboxNotFoundError(FogClientError):
    """Sandbox not found error"""

    pass


class QuotaExceededError(FogClientError):
    """Namespace quota exceeded error"""

    pass


class AuthenticationError(FogClientError):
    """Authentication/authorization error"""

    pass


class JobResult(BaseModel):
    """Job execution result"""

    job_id: str
    status: str
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    cpu_seconds_used: float = 0.0
    memory_mb_peak: int = 0
    duration_ms: float | None = None
    error_message: str | None = None


class SandboxInfo(BaseModel):
    """Sandbox information"""

    sandbox_id: str
    status: str
    connection_url: str | None = None
    ssh_command: str | None = None
    created_at: datetime
    idle_minutes: int = 0


class NamespaceUsage(BaseModel):
    """Namespace usage summary"""

    namespace: str
    total_cost: float
    cpu_seconds: float
    memory_mb_hours: float
    job_executions: int
    period_start: datetime
    period_end: datetime


class PriceQuote(BaseModel):
    """Price quote for resource requirements"""

    available: bool
    min_price: float | None = None
    max_price: float | None = None
    avg_price: float | None = None
    market_estimate: float | None = None
    current_spot_rate: float | None = None
    current_on_demand_rate: float | None = None
    price_volatility: float | None = None
    available_providers: int | None = None
    suggested_max_price: float | None = None
    estimated_wait_time_minutes: int | None = None
    reason: str | None = None


class MarketplacePrices(BaseModel):
    """Current marketplace pricing information"""

    currency: str = "USD"
    last_updated: datetime
    spot_price_per_cpu_hour: float
    on_demand_price_per_cpu_hour: float
    price_volatility: float
    utilization_rate: float
    demand_supply_ratio: float
    available_providers: int
    pricing_tiers: dict[str, dict[str, Any]]


class CostEstimate(BaseModel):
    """Cost estimate for job execution"""

    estimated_cost: float
    cost_breakdown: dict[str, float]
    pricing_tier: str
    bid_type: str
    duration_hours: float
    confidence_level: float
    price_range: dict[str, float]
    recommendations: list[str]
