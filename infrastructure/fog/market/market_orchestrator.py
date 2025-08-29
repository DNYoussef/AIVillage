"""
Market-Based Resource Allocation Orchestrator

Integrates auction engine, dynamic pricing, and tokenomics for comprehensive market operations:
- Coordinates between auction engine and pricing manager
- Handles resource allocation based on market outcomes
- Integrates with existing fog task scheduling
- Provides unified API for market-based resource discovery
- Manages cross-system state synchronization

Key Features:
- Unified market operations interface
- Resource allocation workflow orchestration
- Integration with existing fog infrastructure
- Market-driven task scheduling
- Real-time price discovery and matching
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any
import uuid

from .auction_engine import (
    AuctionEngine,
    ResourceRequirement,
    create_reverse_auction,
    get_auction_engine,
)
from .pricing_manager import (
    DynamicPricingManager,
    ResourceLane,
    get_pricing_manager,
    update_resource_supply_demand,
)

logger = logging.getLogger(__name__)


class AllocationStrategy(str, Enum):
    """Resource allocation strategies"""

    LOWEST_COST = "lowest_cost"  # Minimize cost
    BEST_QUALITY = "best_quality"  # Maximize quality metrics
    BALANCED = "balanced"  # Balance cost and quality
    FASTEST_DELIVERY = "fastest_delivery"  # Minimize time to allocation
    MARKET_DRIVEN = "market_driven"  # Pure market mechanisms


class TaskPriority(str, Enum):
    """Task priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class AllocationStatus(str, Enum):
    """Resource allocation lifecycle states"""

    REQUESTED = "requested"  # Resource request submitted
    PRICING = "pricing"  # Getting pricing information
    AUCTIONING = "auctioning"  # Running auction
    MATCHING = "matching"  # Finding optimal matches
    ALLOCATED = "allocated"  # Resources reserved
    DEPLOYING = "deploying"  # Deploying to fog nodes
    ACTIVE = "active"  # Running on fog infrastructure
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"  # Allocation or execution failed
    CANCELLED = "cancelled"  # Request cancelled


@dataclass
class ResourceAllocationRequest:
    """Request for market-based resource allocation"""

    request_id: str
    requester_id: str
    task_spec: dict[str, Any]

    # Resource requirements
    cpu_cores: Decimal
    memory_gb: Decimal
    storage_gb: Decimal
    bandwidth_mbps: Decimal = Decimal("10")
    duration_hours: Decimal = Decimal("1")

    # Quality requirements
    min_trust_score: Decimal = Decimal("0.3")
    max_latency_ms: Decimal = Decimal("500")
    required_regions: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)

    # Allocation preferences
    allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED
    task_priority: TaskPriority = TaskPriority.NORMAL
    max_budget: Decimal = Decimal("100")
    deadline: datetime | None = None

    # Market preferences
    use_auctions: bool = True
    auction_duration_minutes: int = 15
    max_bid_attempts: int = 3

    # Status tracking
    status: AllocationStatus = AllocationStatus.REQUESTED
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Results
    allocated_resources: list[dict[str, Any]] = field(default_factory=list)
    total_cost: Decimal | None = None
    allocation_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure all numeric fields are Decimal"""
        for field_name in [
            "cpu_cores",
            "memory_gb",
            "storage_gb",
            "bandwidth_mbps",
            "duration_hours",
            "min_trust_score",
            "max_latency_ms",
            "max_budget",
        ]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

    def to_resource_requirement(self) -> ResourceRequirement:
        """Convert to ResourceRequirement for auction engine"""
        return ResourceRequirement(
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            storage_gb=self.storage_gb,
            bandwidth_mbps=self.bandwidth_mbps,
            duration_hours=self.duration_hours,
            min_trust_score=self.min_trust_score,
            max_latency_ms=self.max_latency_ms,
            required_regions=self.required_regions,
            required_capabilities=self.required_capabilities,
        )

    def calculate_priority_multiplier(self) -> Decimal:
        """Get priority multiplier for cost/timing calculations"""
        multipliers = {
            TaskPriority.LOW: Decimal("0.8"),
            TaskPriority.NORMAL: Decimal("1.0"),
            TaskPriority.HIGH: Decimal("1.2"),
            TaskPriority.URGENT: Decimal("1.5"),
            TaskPriority.CRITICAL: Decimal("2.0"),
        }
        return multipliers[self.task_priority]


@dataclass
class MarketAllocationResult:
    """Result of market-based resource allocation"""

    request_id: str
    allocation_successful: bool

    # Resource allocation details
    allocated_nodes: list[dict[str, Any]] = field(default_factory=list)
    total_cost: Decimal = Decimal("0")
    average_trust_score: Decimal = Decimal("0")
    average_latency_ms: Decimal = Decimal("0")

    # Market details
    auction_id: str | None = None
    winning_bids: list[dict[str, Any]] = field(default_factory=list)
    market_clearing_price: Decimal | None = None

    # Allocation metadata
    allocation_time_seconds: Decimal = Decimal("0")
    allocation_method: str = ""  # "auction", "direct_pricing", "hybrid"
    quality_score: Decimal = Decimal("0")

    # Performance tracking
    allocated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deployment_started_at: datetime | None = None
    deployment_completed_at: datetime | None = None

    # Error information
    error_message: str | None = None
    retry_suggestions: list[str] = field(default_factory=list)


class MarketOrchestrator:
    """
    Main orchestrator for market-based fog computing resource allocation

    Coordinates between auction engine, pricing manager, and fog infrastructure
    to provide efficient, market-driven resource allocation.
    """

    def __init__(self, token_system=None, fog_scheduler=None):
        self.token_system = token_system
        self.fog_scheduler = fog_scheduler

        # Market components
        self.auction_engine: AuctionEngine | None = None
        self.pricing_manager: DynamicPricingManager | None = None

        # Active requests
        self.active_requests: dict[str, ResourceAllocationRequest] = {}
        self.completed_requests: dict[str, ResourceAllocationRequest] = {}

        # Market state
        self.total_allocations = 0
        self.total_volume = Decimal("0")
        self.average_allocation_time = Decimal("0")

        # Configuration
        self.config = {
            "default_auction_duration": 15,  # minutes
            "pricing_update_interval": 30,  # seconds
            "max_concurrent_auctions": 50,
            "enable_hybrid_allocation": True,
            "quality_weight": Decimal("0.3"),
            "cost_weight": Decimal("0.7"),
        }

        # Background tasks
        self._orchestration_task: asyncio.Task | None = None
        self._market_sync_task: asyncio.Task | None = None

        logger.info("Market orchestrator initialized")

    async def start(self):
        """Start market orchestrator"""

        # Initialize market components
        self.auction_engine = await get_auction_engine()
        self.pricing_manager = await get_pricing_manager()

        # Start background tasks
        self._orchestration_task = asyncio.create_task(self._orchestration_loop())
        self._market_sync_task = asyncio.create_task(self._market_sync_loop())

        logger.info("Market orchestrator started")

    async def stop(self):
        """Stop market orchestrator"""

        if self._orchestration_task:
            self._orchestration_task.cancel()
        if self._market_sync_task:
            self._market_sync_task.cancel()

        logger.info("Market orchestrator stopped")

    async def request_resources(
        self,
        requester_id: str,
        task_spec: dict[str, Any],
        cpu_cores: float,
        memory_gb: float,
        duration_hours: float = 1.0,
        **kwargs,
    ) -> str:
        """Submit resource allocation request"""

        request_id = f"req_{uuid.uuid4().hex[:8]}"

        request = ResourceAllocationRequest(
            request_id=request_id,
            requester_id=requester_id,
            task_spec=task_spec,
            cpu_cores=Decimal(str(cpu_cores)),
            memory_gb=Decimal(str(memory_gb)),
            duration_hours=Decimal(str(duration_hours)),
            **kwargs,
        )

        self.active_requests[request_id] = request

        logger.info(
            f"Resource request {request_id} submitted: "
            f"{float(cpu_cores)} cores, {float(memory_gb)}GB memory, "
            f"{float(duration_hours)}h duration"
        )

        # Start allocation process asynchronously
        asyncio.create_task(self._process_allocation_request(request))

        return request_id

    async def get_price_estimate(
        self,
        cpu_cores: float,
        memory_gb: float,
        duration_hours: float = 1.0,
        strategy: AllocationStrategy = AllocationStrategy.BALANCED,
    ) -> dict[str, Any]:
        """Get price estimate for resource requirements"""

        if not self.pricing_manager:
            return {"error": "Pricing manager not available"}

        # Get estimates for different resource lanes
        estimates = {}
        total_estimate = Decimal("0")

        # CPU pricing
        cpu_quote = await self.pricing_manager.get_resource_price(
            ResourceLane.CPU, Decimal(str(cpu_cores)), Decimal(str(duration_hours))
        )
        estimates["cpu"] = cpu_quote
        total_estimate += Decimal(str(cpu_quote["final_price"]))

        # Memory pricing
        memory_quote = await self.pricing_manager.get_resource_price(
            ResourceLane.MEMORY, Decimal(str(memory_gb)), Decimal(str(duration_hours))
        )
        estimates["memory"] = memory_quote
        total_estimate += Decimal(str(memory_quote["final_price"]))

        # Get strategy-based adjustments
        strategy_multiplier = self._get_strategy_price_multiplier(strategy)
        adjusted_estimate = total_estimate * strategy_multiplier

        return {
            "lane_estimates": estimates,
            "base_total": float(total_estimate),
            "strategy": strategy.value,
            "strategy_multiplier": float(strategy_multiplier),
            "adjusted_total": float(adjusted_estimate),
            "currency": "USD",
            "valid_for_minutes": 15,
            "market_conditions": await self._get_current_market_conditions(),
        }

    async def get_allocation_status(self, request_id: str) -> dict[str, Any] | None:
        """Get status of resource allocation request"""

        request = self.active_requests.get(request_id) or self.completed_requests.get(request_id)
        if not request:
            return None

        # Get auction status if applicable
        auction_status = None
        if hasattr(request, "auction_id") and request.allocation_metadata.get("auction_id"):
            auction_status = await self.auction_engine.get_auction_status(request.allocation_metadata["auction_id"])

        return {
            "request_id": request_id,
            "status": request.status.value,
            "requester_id": request.requester_id,
            "resources": {
                "cpu_cores": float(request.cpu_cores),
                "memory_gb": float(request.memory_gb),
                "duration_hours": float(request.duration_hours),
            },
            "allocation_details": {
                "strategy": request.allocation_strategy.value,
                "priority": request.task_priority.value,
                "max_budget": float(request.max_budget),
                "allocated_nodes": len(request.allocated_resources),
                "total_cost": float(request.total_cost) if request.total_cost else None,
            },
            "timing": {
                "created_at": request.created_at.isoformat(),
                "updated_at": request.updated_at.isoformat(),
                "allocation_time": (request.updated_at - request.created_at).total_seconds()
                if request.status != AllocationStatus.REQUESTED
                else None,
            },
            "auction_details": auction_status,
            "metadata": request.allocation_metadata,
        }

    async def cancel_allocation_request(self, request_id: str, reason: str = "User cancelled") -> bool:
        """Cancel active allocation request"""

        if request_id not in self.active_requests:
            return False

        request = self.active_requests[request_id]

        # Cancel auction if running
        if request.allocation_metadata.get("auction_id"):
            # Would cancel auction in auction engine
            pass

        request.status = AllocationStatus.CANCELLED
        request.allocation_metadata["cancellation_reason"] = reason
        request.updated_at = datetime.now(UTC)

        # Move to completed requests
        self.completed_requests[request_id] = request
        del self.active_requests[request_id]

        logger.info(f"Allocation request {request_id} cancelled: {reason}")
        return True

    async def update_resource_availability(
        self, node_id: str, available_resources: dict[str, float], pricing_info: dict[str, float]
    ):
        """Update resource availability for market pricing"""

        # Convert to lane-based supply data
        supply_data = {
            "cpu": available_resources.get("cpu_cores", 0),
            "memory": available_resources.get("memory_gb", 0),
            "storage": available_resources.get("storage_gb", 0),
            "bandwidth": available_resources.get("bandwidth_mbps", 0),
        }

        # Update pricing manager (simplified - would aggregate from multiple nodes)
        await update_resource_supply_demand(
            supply_data=supply_data,
            demand_data={k: 0 for k in supply_data.keys()},  # Would track demand
            utilization_data={k: 0.5 for k in supply_data.keys()},  # Would track utilization
        )

        logger.debug(f"Updated resource availability for node {node_id}")

    async def get_market_statistics(self) -> dict[str, Any]:
        """Get comprehensive market statistics"""

        # Get statistics from components
        auction_stats = await self.auction_engine.get_market_statistics() if self.auction_engine else {}
        pricing_analytics = await self.pricing_manager.get_market_analytics() if self.pricing_manager else {}

        # Calculate orchestrator-specific metrics
        active_count = len(self.active_requests)
        completed_count = len(self.completed_requests)

        success_rate = 0.0
        if completed_count > 0:
            successful_requests = len(
                [req for req in self.completed_requests.values() if req.status == AllocationStatus.COMPLETED]
            )
            success_rate = successful_requests / completed_count

        return {
            "orchestrator_metrics": {
                "total_allocations": self.total_allocations,
                "active_requests": active_count,
                "completed_requests": completed_count,
                "success_rate": success_rate,
                "total_volume": float(self.total_volume),
                "average_allocation_time_seconds": float(self.average_allocation_time),
            },
            "auction_statistics": auction_stats,
            "pricing_analytics": pricing_analytics,
            "system_health": {
                "auction_engine_status": "active" if self.auction_engine else "inactive",
                "pricing_manager_status": "active" if self.pricing_manager else "inactive",
                "fog_scheduler_status": "active" if self.fog_scheduler else "inactive",
            },
        }

    # Private methods

    async def _process_allocation_request(self, request: ResourceAllocationRequest):
        """Process resource allocation request through market mechanisms"""

        start_time = datetime.now(UTC)

        try:
            # Update status
            request.status = AllocationStatus.PRICING
            request.updated_at = datetime.now(UTC)

            # Get pricing information
            price_estimate = await self.get_price_estimate(
                float(request.cpu_cores),
                float(request.memory_gb),
                float(request.duration_hours),
                request.allocation_strategy,
            )

            if price_estimate.get("error"):
                raise Exception(f"Pricing error: {price_estimate['error']}")

            estimated_cost = Decimal(str(price_estimate["adjusted_total"]))

            # Check budget
            if estimated_cost > request.max_budget:
                request.status = AllocationStatus.FAILED
                request.allocation_metadata["error"] = "Estimated cost exceeds budget"
                request.allocation_metadata["estimated_cost"] = float(estimated_cost)
                return

            # Choose allocation method based on configuration and request
            if request.use_auctions and self.config["enable_hybrid_allocation"]:
                result = await self._allocate_via_auction(request, estimated_cost)
            else:
                result = await self._allocate_via_direct_pricing(request, estimated_cost)

            # Process result
            if result.allocation_successful:
                request.status = AllocationStatus.ALLOCATED
                request.allocated_resources = result.allocated_nodes
                request.total_cost = result.total_cost
                request.allocation_metadata.update(
                    {
                        "allocation_method": result.allocation_method,
                        "auction_id": result.auction_id,
                        "market_clearing_price": float(result.market_clearing_price)
                        if result.market_clearing_price
                        else None,
                        "quality_score": float(result.quality_score),
                        "allocation_time_seconds": float(result.allocation_time_seconds),
                    }
                )

                # Deploy to fog infrastructure
                await self._deploy_to_fog_infrastructure(request)

                # Update statistics
                self.total_allocations += 1
                self.total_volume += result.total_cost
                allocation_time = (datetime.now(UTC) - start_time).total_seconds()
                self._update_average_allocation_time(Decimal(str(allocation_time)))

            else:
                request.status = AllocationStatus.FAILED
                request.allocation_metadata["error"] = result.error_message
                request.allocation_metadata["retry_suggestions"] = result.retry_suggestions

        except Exception as e:
            logger.error(f"Error processing allocation request {request.request_id}: {e}")
            request.status = AllocationStatus.FAILED
            request.allocation_metadata["error"] = str(e)

        finally:
            request.updated_at = datetime.now(UTC)

            # Move to completed requests if finished
            if request.status in [AllocationStatus.COMPLETED, AllocationStatus.FAILED, AllocationStatus.CANCELLED]:
                self.completed_requests[request.request_id] = request
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]

    async def _allocate_via_auction(
        self, request: ResourceAllocationRequest, estimated_cost: Decimal
    ) -> MarketAllocationResult:
        """Allocate resources using auction mechanism"""

        start_time = datetime.now(UTC)

        try:
            # Create auction
            auction_id = await create_reverse_auction(
                requester_id=request.requester_id,
                cpu_cores=float(request.cpu_cores),
                memory_gb=float(request.memory_gb),
                duration_hours=float(request.duration_hours),
                reserve_price=float(estimated_cost),
                duration_minutes=request.auction_duration_minutes,
            )

            request.allocation_metadata["auction_id"] = auction_id
            request.status = AllocationStatus.AUCTIONING

            # Wait for auction to complete (simplified - would use callbacks in production)
            await asyncio.sleep(request.auction_duration_minutes * 60 + 30)  # Auction time + settlement

            # Get auction result
            auction_status = await self.auction_engine.get_auction_status(auction_id)

            if auction_status and auction_status.get("result"):
                result_data = auction_status["result"]

                return MarketAllocationResult(
                    request_id=request.request_id,
                    allocation_successful=True,
                    auction_id=auction_id,
                    total_cost=Decimal(str(result_data["total_cost"])),
                    market_clearing_price=Decimal(str(result_data["clearing_price"])),
                    allocation_method="auction",
                    allocation_time_seconds=Decimal(str((datetime.now(UTC) - start_time).total_seconds())),
                    allocated_nodes=[{"node_id": "auction_winner", "cost": result_data["total_cost"]}],
                )
            else:
                return MarketAllocationResult(
                    request_id=request.request_id,
                    allocation_successful=False,
                    error_message="Auction failed to produce winners",
                    retry_suggestions=["Try direct pricing", "Increase budget", "Reduce requirements"],
                )

        except Exception as e:
            return MarketAllocationResult(
                request_id=request.request_id,
                allocation_successful=False,
                error_message=f"Auction allocation failed: {e}",
                retry_suggestions=["Try direct pricing", "Check network connectivity"],
            )

    async def _allocate_via_direct_pricing(
        self, request: ResourceAllocationRequest, estimated_cost: Decimal
    ) -> MarketAllocationResult:
        """Allocate resources using direct pricing mechanism"""

        start_time = datetime.now(UTC)

        try:
            # Simulate finding available resources at market prices
            # In production, this would integrate with fog scheduler

            # Calculate quality score based on requirements
            quality_score = Decimal("0.7")  # Simplified

            allocated_nodes = [
                {
                    "node_id": f"node_{uuid.uuid4().hex[:8]}",
                    "cpu_cores": float(request.cpu_cores),
                    "memory_gb": float(request.memory_gb),
                    "cost": float(estimated_cost),
                    "trust_score": 0.8,
                    "latency_ms": 150,
                }
            ]

            return MarketAllocationResult(
                request_id=request.request_id,
                allocation_successful=True,
                total_cost=estimated_cost,
                allocated_nodes=allocated_nodes,
                allocation_method="direct_pricing",
                quality_score=quality_score,
                allocation_time_seconds=Decimal(str((datetime.now(UTC) - start_time).total_seconds())),
            )

        except Exception as e:
            return MarketAllocationResult(
                request_id=request.request_id,
                allocation_successful=False,
                error_message=f"Direct pricing allocation failed: {e}",
                retry_suggestions=["Try auction mechanism", "Reduce resource requirements"],
            )

    async def _deploy_to_fog_infrastructure(self, request: ResourceAllocationRequest):
        """Deploy allocated resources to fog infrastructure"""

        request.status = AllocationStatus.DEPLOYING

        try:
            # Integration with fog scheduler would happen here
            if self.fog_scheduler:
                # deployment_result = await self.fog_scheduler.deploy_task(
                #     task_spec=request.task_spec,
                #     allocated_resources=request.allocated_resources
                # )
                pass

            # Simulate deployment
            await asyncio.sleep(2)  # Simulate deployment time

            request.status = AllocationStatus.ACTIVE
            request.allocation_metadata["deployment_completed_at"] = datetime.now(UTC).isoformat()

            logger.info(f"Request {request.request_id} deployed to fog infrastructure")

        except Exception as e:
            logger.error(f"Deployment failed for request {request.request_id}: {e}")
            request.status = AllocationStatus.FAILED
            request.allocation_metadata["deployment_error"] = str(e)

    async def _orchestration_loop(self):
        """Background task for orchestration management"""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Update request statuses
                await self._update_active_requests()

                # Clean up old completed requests
                await self._cleanup_old_requests()

            except Exception as e:
                logger.error(f"Error in orchestration loop: {e}")
                await asyncio.sleep(60)

    async def _market_sync_loop(self):
        """Background task for market synchronization"""

        while True:
            try:
                await asyncio.sleep(self.config["pricing_update_interval"])

                # Sync market conditions between components
                await self._synchronize_market_state()

            except Exception as e:
                logger.error(f"Error in market sync loop: {e}")
                await asyncio.sleep(120)

    async def _update_active_requests(self):
        """Update status of active requests"""

        for request in self.active_requests.values():
            if request.status == AllocationStatus.ACTIVE:
                # Check if task completed (simplified)
                if (datetime.now(UTC) - request.updated_at).total_seconds() > 300:  # 5 minutes for demo
                    request.status = AllocationStatus.COMPLETED
                    request.updated_at = datetime.now(UTC)

    async def _cleanup_old_requests(self):
        """Clean up old completed requests"""

        cutoff = datetime.now(UTC) - timedelta(days=7)

        old_requests = [
            req_id for req_id, req in self.completed_requests.items() if req.updated_at.replace(tzinfo=UTC) < cutoff
        ]

        for req_id in old_requests:
            del self.completed_requests[req_id]

        if old_requests:
            logger.info(f"Cleaned up {len(old_requests)} old allocation requests")

    async def _synchronize_market_state(self):
        """Synchronize market state between components"""

        # Update pricing manager with auction results
        if self.auction_engine and self.pricing_manager:
            # Get recent auction statistics
            auction_stats = await self.auction_engine.get_market_statistics()

            # Update supply/demand based on auction activity
            # This is simplified - production would be more sophisticated
            active_auctions = auction_stats.get("auction_statistics", {}).get("active_auctions", 0)

            if active_auctions > 10:  # High demand
                pass
            else:
                pass

            # Would update pricing manager with real market data

    def _get_strategy_price_multiplier(self, strategy: AllocationStrategy) -> Decimal:
        """Get price multiplier based on allocation strategy"""

        multipliers = {
            AllocationStrategy.LOWEST_COST: Decimal("1.0"),
            AllocationStrategy.BEST_QUALITY: Decimal("1.3"),
            AllocationStrategy.BALANCED: Decimal("1.1"),
            AllocationStrategy.FASTEST_DELIVERY: Decimal("1.2"),
            AllocationStrategy.MARKET_DRIVEN: Decimal("1.0"),
        }
        return multipliers[strategy]

    async def _get_current_market_conditions(self) -> dict[str, Any]:
        """Get current market conditions summary"""

        if not self.pricing_manager:
            return {"status": "pricing_unavailable"}

        analytics = await self.pricing_manager.get_market_analytics()

        return {
            "market_condition": analytics["market_overview"]["market_condition"],
            "average_price": analytics["market_overview"].get("average_price_across_lanes", 0),
            "volatility": analytics.get("pricing_metrics", {}).get("manipulation_risk_score", 0),
            "liquidity": analytics.get("market_overview", {}).get("health_score", 0),
        }

    def _update_average_allocation_time(self, new_time: Decimal):
        """Update rolling average allocation time"""

        if self.average_allocation_time == 0:
            self.average_allocation_time = new_time
        else:
            # Simple moving average
            self.average_allocation_time = (self.average_allocation_time + new_time) / Decimal("2")


# Global orchestrator instance
_market_orchestrator: MarketOrchestrator | None = None


async def get_market_orchestrator() -> MarketOrchestrator:
    """Get global market orchestrator instance"""
    global _market_orchestrator

    if _market_orchestrator is None:
        _market_orchestrator = MarketOrchestrator()
        await _market_orchestrator.start()

    return _market_orchestrator


# Convenience functions for integration
async def request_fog_resources(
    requester_id: str,
    task_spec: dict[str, Any],
    cpu_cores: float,
    memory_gb: float,
    duration_hours: float = 1.0,
    **kwargs,
) -> str:
    """Request fog computing resources through market mechanisms"""

    orchestrator = await get_market_orchestrator()
    return await orchestrator.request_resources(requester_id, task_spec, cpu_cores, memory_gb, duration_hours, **kwargs)


async def get_fog_resource_quote(
    cpu_cores: float, memory_gb: float, duration_hours: float = 1.0, strategy: str = "balanced"
) -> dict[str, Any]:
    """Get pricing quote for fog computing resources"""

    orchestrator = await get_market_orchestrator()
    return await orchestrator.get_price_estimate(cpu_cores, memory_gb, duration_hours, AllocationStrategy(strategy))


async def check_allocation_status(request_id: str) -> dict[str, Any] | None:
    """Check status of resource allocation request"""

    orchestrator = await get_market_orchestrator()
    return await orchestrator.get_allocation_status(request_id)
