"""
Fog Gateway Billing API

Implements usage tracking, price quotes, and invoice generation:
- Namespace-scoped usage accrual
- Real-time price quotes with market rates
- Usage reports and cost breakdown
- Invoice generation with detailed line items

Billing Model:
- Usage accrues per namespace continuously
- Pricing based on marketplace rates
- Cost attribution to jobs and resources
- Comprehensive audit trail for compliance
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

# Import marketplace for pricing
from ..scheduler.marketplace import BidType, PricingTier, get_marketplace_engine

logger = logging.getLogger(__name__)


# Pydantic models for API
class UsageMetrics(BaseModel):
    """Resource usage metrics"""

    cpu_core_seconds: float = Field(0.0, description="Total CPU core-seconds consumed")
    memory_gb_seconds: float = Field(0.0, description="Total memory GB-seconds consumed")
    disk_gb_seconds: float = Field(0.0, description="Total disk GB-seconds consumed")
    network_ingress_bytes: int = Field(0, description="Network ingress bytes")
    network_egress_bytes: int = Field(0, description="Network egress bytes")
    job_count: int = Field(0, description="Total number of jobs executed")


class CostBreakdown(BaseModel):
    """Cost breakdown by resource type"""

    cpu_cost: float = Field(0.0, description="CPU usage cost in USD")
    memory_cost: float = Field(0.0, description="Memory usage cost in USD")
    disk_cost: float = Field(0.0, description="Disk usage cost in USD")
    network_cost: float = Field(0.0, description="Network usage cost in USD")
    premium_charges: float = Field(0.0, description="Premium charges (trust, SLA)")
    total_cost: float = Field(0.0, description="Total cost in USD")


class UsageReport(BaseModel):
    """Usage report for namespace"""

    namespace: str
    period_start: datetime
    period_end: datetime

    usage_metrics: UsageMetrics
    cost_breakdown: CostBreakdown

    job_summary: dict[str, Any] = Field(default_factory=dict)
    pricing_tier_breakdown: dict[str, float] = Field(default_factory=dict)


class PriceQuoteRequest(BaseModel):
    """Request for price quote"""

    cpu_cores: float = Field(1.0, ge=0.1, le=100.0, description="CPU cores required")
    memory_gb: float = Field(1.0, ge=0.1, le=1000.0, description="Memory GB required")
    disk_gb: float = Field(2.0, ge=0.1, le=10000.0, description="Disk GB required")
    estimated_duration_hours: float = Field(1.0, ge=0.1, le=168.0, description="Estimated duration")

    bid_type: BidType = Field(BidType.SPOT, description="Bidding type")
    pricing_tier: PricingTier = Field(PricingTier.BASIC, description="SLA pricing tier")
    min_trust_score: float = Field(0.3, ge=0.0, le=1.0, description="Minimum trust score")
    max_latency_ms: float = Field(500.0, ge=10.0, le=10000.0, description="Maximum latency")


class PriceQuote(BaseModel):
    """Price quote response"""

    available: bool
    quote: dict[str, float] | None = None
    market_conditions: dict[str, Any] | None = None
    recommendations: dict[str, Any] | None = None
    reason: str | None = None


class InvoiceLineItem(BaseModel):
    """Invoice line item"""

    item_id: str
    description: str
    resource_type: str  # cpu, memory, disk, network
    quantity: float
    unit_price: float
    total_price: float
    job_id: str | None = None
    period_start: datetime
    period_end: datetime


class Invoice(BaseModel):
    """Generated invoice"""

    invoice_id: str
    namespace: str
    period_start: datetime
    period_end: datetime

    line_items: list[InvoiceLineItem]

    subtotal: float
    tax_rate: float = 0.0
    tax_amount: float = 0.0
    total_amount: float

    generated_at: datetime
    due_date: datetime

    currency: str = "USD"
    payment_terms: str = "Net 30"


@dataclass
class NamespaceUsageTracker:
    """Tracks resource usage for a namespace"""

    namespace: str

    # Cumulative usage metrics
    cpu_core_seconds: float = 0.0
    memory_gb_seconds: float = 0.0
    disk_gb_seconds: float = 0.0
    network_ingress_bytes: int = 0
    network_egress_bytes: int = 0

    # Job tracking
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0

    # Cost tracking
    total_cost: float = 0.0
    cpu_cost: float = 0.0
    memory_cost: float = 0.0
    disk_cost: float = 0.0
    network_cost: float = 0.0
    premium_cost: float = 0.0

    # Pricing tier breakdown
    basic_tier_cost: float = 0.0
    standard_tier_cost: float = 0.0
    premium_tier_cost: float = 0.0

    # Usage by time period
    hourly_usage: dict[str, float] = field(default_factory=dict)
    daily_usage: dict[str, float] = field(default_factory=dict)
    monthly_usage: dict[str, float] = field(default_factory=dict)

    # Last update timestamp
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record_job_start(
        self,
        job_id: str,
        cpu_cores: float,
        memory_gb: float,
        disk_gb: float,
        pricing_tier: PricingTier = PricingTier.BASIC,
    ) -> None:
        """Record job start for usage tracking"""

        self.total_jobs += 1
        self.last_updated = datetime.now(UTC)

        logger.debug(f"Started job {job_id} in namespace {self.namespace}")

    def record_job_completion(
        self,
        job_id: str,
        cpu_cores: float,
        memory_gb: float,
        disk_gb: float,
        duration_seconds: float,
        network_ingress: int = 0,
        network_egress: int = 0,
        success: bool = True,
        pricing_tier: PricingTier = PricingTier.BASIC,
        marketplace_price: float | None = None,
    ) -> float:
        """Record job completion and calculate cost"""

        # Update usage metrics
        self.cpu_core_seconds += cpu_cores * duration_seconds
        self.memory_gb_seconds += memory_gb * duration_seconds
        self.disk_gb_seconds += disk_gb * duration_seconds
        self.network_ingress_bytes += network_ingress
        self.network_egress_bytes += network_egress

        # Update job counts
        if success:
            self.completed_jobs += 1
        else:
            self.failed_jobs += 1

        # Calculate cost
        job_cost = self._calculate_job_cost(
            cpu_cores, memory_gb, disk_gb, duration_seconds, pricing_tier, marketplace_price
        )

        # Update cost tracking
        self.total_cost += job_cost

        # Update time-based usage tracking
        self._update_time_based_usage(job_cost)

        self.last_updated = datetime.now(UTC)

        logger.info(
            f"Completed job {job_id} in namespace {self.namespace}: " f"${job_cost:.4f} for {duration_seconds}s"
        )

        return job_cost

    def get_usage_report(self, start_time: datetime, end_time: datetime) -> UsageReport:
        """Generate usage report for time period"""

        usage_metrics = UsageMetrics(
            cpu_core_seconds=self.cpu_core_seconds,
            memory_gb_seconds=self.memory_gb_seconds,
            disk_gb_seconds=self.disk_gb_seconds,
            network_ingress_bytes=self.network_ingress_bytes,
            network_egress_bytes=self.network_egress_bytes,
            job_count=self.total_jobs,
        )

        cost_breakdown = CostBreakdown(
            cpu_cost=self.cpu_cost,
            memory_cost=self.memory_cost,
            disk_cost=self.disk_cost,
            network_cost=self.network_cost,
            premium_charges=self.premium_cost,
            total_cost=self.total_cost,
        )

        job_summary = {
            "total_jobs": self.total_jobs,
            "completed_jobs": self.completed_jobs,
            "failed_jobs": self.failed_jobs,
            "success_rate": self.completed_jobs / max(1, self.total_jobs),
            "avg_cost_per_job": self.total_cost / max(1, self.total_jobs),
        }

        pricing_tier_breakdown = {
            "basic": self.basic_tier_cost,
            "standard": self.standard_tier_cost,
            "premium": self.premium_tier_cost,
        }

        return UsageReport(
            namespace=self.namespace,
            period_start=start_time,
            period_end=end_time,
            usage_metrics=usage_metrics,
            cost_breakdown=cost_breakdown,
            job_summary=job_summary,
            pricing_tier_breakdown=pricing_tier_breakdown,
        )

    def _calculate_job_cost(
        self,
        cpu_cores: float,
        memory_gb: float,
        disk_gb: float,
        duration_seconds: float,
        pricing_tier: PricingTier,
        marketplace_price: float | None = None,
    ) -> float:
        """Calculate cost for job execution"""

        duration_hours = duration_seconds / 3600.0

        if marketplace_price is not None:
            # Use marketplace pricing if available
            base_cost = marketplace_price
        else:
            # Default pricing model
            cpu_rate = 0.10  # $0.10 per CPU-hour
            memory_rate = 0.01  # $0.01 per GB-hour
            disk_rate = 0.001  # $0.001 per GB-hour

            cpu_cost = cpu_cores * duration_hours * cpu_rate
            memory_cost = memory_gb * duration_hours * memory_rate
            disk_cost = disk_gb * duration_hours * disk_rate

            base_cost = cpu_cost + memory_cost + disk_cost

        # Apply pricing tier multiplier
        tier_multipliers = {
            PricingTier.BASIC: 1.0,
            PricingTier.STANDARD: 1.5,  # 50% premium for A-class SLA
            PricingTier.PREMIUM: 2.0,  # 100% premium for S-class SLA
        }

        total_cost = base_cost * tier_multipliers[pricing_tier]

        # Update cost breakdown
        self.cpu_cost += total_cost * 0.7  # 70% attributed to CPU
        self.memory_cost += total_cost * 0.2  # 20% to memory
        self.disk_cost += total_cost * 0.05  # 5% to disk
        self.network_cost += total_cost * 0.05  # 5% to network

        # Update tier breakdown
        if pricing_tier == PricingTier.BASIC:
            self.basic_tier_cost += total_cost
        elif pricing_tier == PricingTier.STANDARD:
            self.standard_tier_cost += total_cost
        else:
            self.premium_tier_cost += total_cost

        return total_cost

    def _update_time_based_usage(self, cost: float) -> None:
        """Update hourly/daily/monthly usage tracking"""

        now = datetime.now(UTC)

        # Hourly tracking
        hour_key = now.strftime("%Y-%m-%d %H:00")
        self.hourly_usage[hour_key] = self.hourly_usage.get(hour_key, 0.0) + cost

        # Daily tracking
        day_key = now.strftime("%Y-%m-%d")
        self.daily_usage[day_key] = self.daily_usage.get(day_key, 0.0) + cost

        # Monthly tracking
        month_key = now.strftime("%Y-%m")
        self.monthly_usage[month_key] = self.monthly_usage.get(month_key, 0.0) + cost

        # Clean up old data (keep last 30 days)
        cutoff = now - timedelta(days=30)

        # Clean hourly data
        self.hourly_usage = {
            k: v
            for k, v in self.hourly_usage.items()
            if datetime.fromisoformat(k.replace(" ", "T") + ":00+00:00") > cutoff
        }


class FogBillingEngine:
    """
    Fog Computing Billing Engine

    Tracks resource usage and costs per namespace with:
    - Real-time usage accrual
    - Marketplace-based pricing
    - Comprehensive cost attribution
    - Invoice generation and reporting
    """

    def __init__(self):
        # Usage tracking per namespace
        self.namespace_trackers: dict[str, NamespaceUsageTracker] = {}

        # Invoice storage
        self.invoices: dict[str, Invoice] = {}

        # Billing configuration
        self.tax_rate = 0.0  # Default no tax
        self.payment_terms_days = 30

        logger.info("Billing engine initialized")

    async def get_or_create_tracker(self, namespace: str) -> NamespaceUsageTracker:
        """Get usage tracker for namespace, creating if needed"""

        if namespace not in self.namespace_trackers:
            self.namespace_trackers[namespace] = NamespaceUsageTracker(namespace=namespace)
            logger.info(f"Created usage tracker for namespace: {namespace}")

        return self.namespace_trackers[namespace]

    async def record_job_usage(
        self,
        namespace: str,
        job_id: str,
        cpu_cores: float,
        memory_gb: float,
        disk_gb: float,
        duration_seconds: float,
        network_ingress: int = 0,
        network_egress: int = 0,
        success: bool = True,
        pricing_tier: PricingTier = PricingTier.BASIC,
        marketplace_price: float | None = None,
    ) -> float:
        """Record job resource usage and calculate cost"""

        tracker = await self.get_or_create_tracker(namespace)

        cost = tracker.record_job_completion(
            job_id,
            cpu_cores,
            memory_gb,
            disk_gb,
            duration_seconds,
            network_ingress,
            network_egress,
            success,
            pricing_tier,
            marketplace_price,
        )

        logger.info(
            f"Recorded usage for job {job_id} in namespace {namespace}: "
            f"${cost:.4f} ({cpu_cores} cores × {duration_seconds}s)"
        )

        return cost

    async def get_usage_report(
        self, namespace: str, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> UsageReport:
        """Get usage report for namespace"""

        tracker = await self.get_or_create_tracker(namespace)

        if start_time is None:
            start_time = datetime.now(UTC) - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now(UTC)

        return tracker.get_usage_report(start_time, end_time)

    async def generate_invoice(self, namespace: str, period_start: datetime, period_end: datetime) -> Invoice:
        """Generate invoice for namespace usage"""

        usage_report = await self.get_usage_report(namespace, period_start, period_end)

        invoice_id = f"INV-{namespace}-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid4().hex[:6].upper()}"

        # Create line items
        line_items = []

        # CPU usage line item
        if usage_report.usage_metrics.cpu_core_seconds > 0:
            cpu_hours = usage_report.usage_metrics.cpu_core_seconds / 3600.0
            line_items.append(
                InvoiceLineItem(
                    item_id=f"{invoice_id}-CPU",
                    description=f"CPU usage ({cpu_hours:.2f} core-hours)",
                    resource_type="cpu",
                    quantity=cpu_hours,
                    unit_price=usage_report.cost_breakdown.cpu_cost / cpu_hours,
                    total_price=usage_report.cost_breakdown.cpu_cost,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        # Memory usage line item
        if usage_report.usage_metrics.memory_gb_seconds > 0:
            memory_hours = usage_report.usage_metrics.memory_gb_seconds / 3600.0
            line_items.append(
                InvoiceLineItem(
                    item_id=f"{invoice_id}-MEM",
                    description=f"Memory usage ({memory_hours:.2f} GB-hours)",
                    resource_type="memory",
                    quantity=memory_hours,
                    unit_price=usage_report.cost_breakdown.memory_cost / memory_hours,
                    total_price=usage_report.cost_breakdown.memory_cost,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        # Network usage line item
        if usage_report.cost_breakdown.network_cost > 0:
            total_bytes = (
                usage_report.usage_metrics.network_ingress_bytes + usage_report.usage_metrics.network_egress_bytes
            )
            line_items.append(
                InvoiceLineItem(
                    item_id=f"{invoice_id}-NET",
                    description=f"Network usage ({total_bytes:,} bytes)",
                    resource_type="network",
                    quantity=total_bytes / (1024**3),  # Convert to GB
                    unit_price=usage_report.cost_breakdown.network_cost,
                    total_price=usage_report.cost_breakdown.network_cost,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        # Premium charges line item
        if usage_report.cost_breakdown.premium_charges > 0:
            line_items.append(
                InvoiceLineItem(
                    item_id=f"{invoice_id}-PREMIUM",
                    description="Premium SLA charges",
                    resource_type="premium",
                    quantity=1.0,
                    unit_price=usage_report.cost_breakdown.premium_charges,
                    total_price=usage_report.cost_breakdown.premium_charges,
                    period_start=period_start,
                    period_end=period_end,
                )
            )

        # Calculate totals
        subtotal = sum(item.total_price for item in line_items)
        tax_amount = subtotal * self.tax_rate
        total_amount = subtotal + tax_amount

        # Create invoice
        invoice = Invoice(
            invoice_id=invoice_id,
            namespace=namespace,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items,
            subtotal=subtotal,
            tax_rate=self.tax_rate,
            tax_amount=tax_amount,
            total_amount=total_amount,
            generated_at=datetime.now(UTC),
            due_date=datetime.now(UTC) + timedelta(days=self.payment_terms_days),
        )

        # Store invoice
        self.invoices[invoice_id] = invoice

        logger.info(f"Generated invoice {invoice_id} for namespace {namespace}: ${total_amount:.2f}")

        return invoice

    async def get_all_namespace_usage(self) -> dict[str, dict[str, Any]]:
        """Get usage summary for all namespaces"""

        summary = {}

        for namespace, tracker in self.namespace_trackers.items():
            summary[namespace] = {
                "total_cost": tracker.total_cost,
                "total_jobs": tracker.total_jobs,
                "cpu_core_seconds": tracker.cpu_core_seconds,
                "memory_gb_seconds": tracker.memory_gb_seconds,
                "last_updated": tracker.last_updated.isoformat(),
            }

        return summary


# Global billing engine instance
_billing_engine: FogBillingEngine | None = None


async def get_billing_engine() -> FogBillingEngine:
    """Get global billing engine instance"""
    global _billing_engine

    if _billing_engine is None:
        _billing_engine = FogBillingEngine()

    return _billing_engine


# FastAPI router for billing endpoints
billing_router = APIRouter(prefix="/v1/fog", tags=["billing"])


@billing_router.get("/usage", response_model=UsageReport)
async def get_namespace_usage(
    namespace: str = Query(..., description="Namespace to get usage for"),
    start_date: datetime | None = Query(None, description="Start date for usage report"),
    end_date: datetime | None = Query(None, description="End date for usage report"),
    billing_engine: FogBillingEngine = Depends(get_billing_engine),
) -> UsageReport:
    """Get usage report for namespace"""

    try:
        return await billing_engine.get_usage_report(namespace, start_date, end_date)

    except Exception as e:
        logger.error(f"Error getting usage for namespace {namespace}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage report: {str(e)}")


@billing_router.post("/quotes", response_model=PriceQuote)
async def get_price_quote(quote_request: PriceQuoteRequest) -> PriceQuote:
    """Get price quote for resource requirements"""

    try:
        marketplace = await get_marketplace_engine()

        quote_data = await marketplace.get_price_quote(
            cpu_cores=quote_request.cpu_cores,
            memory_gb=quote_request.memory_gb,
            duration_hours=quote_request.estimated_duration_hours,
            bid_type=quote_request.bid_type,
            pricing_tier=quote_request.pricing_tier,
            min_trust_score=quote_request.min_trust_score,
        )

        return PriceQuote(**quote_data)

    except Exception as e:
        logger.error(f"Error getting price quote: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get price quote: {str(e)}")


@billing_router.get("/prices")
async def get_current_prices():
    """Get current market prices"""

    try:
        marketplace = await get_marketplace_engine()
        market_status = await marketplace.get_marketplace_status()

        return {
            "currency": "USD",
            "last_updated": datetime.now(UTC).isoformat(),
            "spot_price_per_cpu_hour": market_status["pricing"]["current_spot_price_per_cpu_hour"],
            "on_demand_price_per_cpu_hour": market_status["pricing"]["current_on_demand_price_per_cpu_hour"],
            "price_volatility": market_status["pricing"]["price_volatility_24h"],
            "market_conditions": {
                "utilization_rate": market_status["resource_supply"]["utilization_rate"],
                "demand_supply_ratio": market_status["resource_demand"]["demand_supply_ratio"],
                "available_providers": market_status["marketplace_summary"]["active_listings"],
            },
            "pricing_tiers": {
                "basic": {"multiplier": 1.0, "description": "Best effort (B-class SLA)"},
                "standard": {"multiplier": 1.5, "description": "Replicated (A-class SLA)"},
                "premium": {"multiplier": 2.0, "description": "Replicated + Attested (S-class SLA)"},
            },
        }

    except Exception as e:
        logger.error(f"Error getting current prices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current prices: {str(e)}")


@billing_router.post("/invoices", response_model=Invoice)
async def generate_invoice(
    namespace: str = Query(..., description="Namespace to generate invoice for"),
    period_start: datetime = Query(..., description="Invoice period start date"),
    period_end: datetime = Query(..., description="Invoice period end date"),
    billing_engine: FogBillingEngine = Depends(get_billing_engine),
) -> Invoice:
    """Generate invoice for namespace usage"""

    try:
        if period_end <= period_start:
            raise HTTPException(status_code=400, detail="End date must be after start date")

        if (period_end - period_start).days > 365:
            raise HTTPException(status_code=400, detail="Invoice period cannot exceed 1 year")

        return await billing_engine.generate_invoice(namespace, period_start, period_end)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating invoice for namespace {namespace}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate invoice: {str(e)}")


@billing_router.get("/invoices/{invoice_id}", response_model=Invoice)
async def get_invoice(invoice_id: str, billing_engine: FogBillingEngine = Depends(get_billing_engine)) -> Invoice:
    """Get existing invoice by ID"""

    try:
        if invoice_id not in billing_engine.invoices:
            raise HTTPException(status_code=404, detail=f"Invoice {invoice_id} not found")

        return billing_engine.invoices[invoice_id]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting invoice {invoice_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get invoice: {str(e)}")


@billing_router.get("/summary")
async def get_billing_summary(billing_engine: FogBillingEngine = Depends(get_billing_engine)):
    """Get billing summary for all namespaces"""

    try:
        namespace_usage = await billing_engine.get_all_namespace_usage()

        # Calculate overall statistics
        total_cost = sum(usage["total_cost"] for usage in namespace_usage.values())
        total_jobs = sum(usage["total_jobs"] for usage in namespace_usage.values())
        total_cpu_seconds = sum(usage["cpu_core_seconds"] for usage in namespace_usage.values())

        return {
            "summary": {
                "total_namespaces": len(namespace_usage),
                "total_cost_usd": total_cost,
                "total_jobs": total_jobs,
                "total_cpu_core_seconds": total_cpu_seconds,
                "avg_cost_per_job": total_cost / max(1, total_jobs),
                "total_invoices_generated": len(billing_engine.invoices),
            },
            "namespace_breakdown": namespace_usage,
            "generated_at": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting billing summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get billing summary: {str(e)}")


# Convenience functions for integration
async def record_job_billing(
    namespace: str, job_id: str, cpu_cores: float, memory_gb: float, duration_seconds: float, **kwargs
) -> float:
    """Record job usage for billing"""

    billing_engine = await get_billing_engine()
    return await billing_engine.record_job_usage(
        namespace, job_id, cpu_cores, memory_gb, 0.0, duration_seconds, **kwargs
    )


async def get_namespace_cost_summary(namespace: str) -> dict[str, Any]:
    """Get cost summary for namespace"""

    billing_engine = await get_billing_engine()
    usage_report = await billing_engine.get_usage_report(namespace)

    return {
        "namespace": namespace,
        "total_cost": usage_report.cost_breakdown.total_cost,
        "job_count": usage_report.usage_metrics.job_count,
        "cpu_hours": usage_report.usage_metrics.cpu_core_seconds / 3600.0,
        "memory_hours": usage_report.usage_metrics.memory_gb_seconds / 3600.0,
    }
