"""
Fog Task Scheduling Integration with Market-Based Pricing

Integrates market-based pricing with existing fog task scheduling:
- Replace fixed pricing with dynamic market prices
- Connect auctions to task allocation
- Market-driven resource discovery and matching
- Integration with existing marketplace engine
- Seamless transition from legacy pricing to market mechanisms

Key Features:
- Backward compatibility with existing scheduler
- Market-enhanced resource matching
- Dynamic pricing integration
- Task priority-based market allocation
- Performance monitoring and optimization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import logging
from typing import Any
import uuid

from .market_orchestrator import (
    AllocationStatus,
    AllocationStrategy,
    MarketOrchestrator,
    ResourceAllocationRequest,
    TaskPriority,
    get_market_orchestrator,
)
from .pricing_manager import DynamicPricingManager, get_pricing_manager

logger = logging.getLogger(__name__)


class TaskSchedulingMode(str, Enum):
    """Task scheduling modes"""

    LEGACY = "legacy"  # Use original fixed pricing
    MARKET_HYBRID = "market_hybrid"  # Mix of market and fixed pricing
    FULL_MARKET = "full_market"  # Pure market-based allocation
    ADAPTIVE = "adaptive"  # Auto-switch based on conditions


class ResourceMatchStrategy(str, Enum):
    """Resource matching strategies for market integration"""

    PRICE_FIRST = "price_first"  # Minimize cost
    QUALITY_FIRST = "quality_first"  # Maximize quality
    SPEED_FIRST = "speed_first"  # Minimize allocation time
    BALANCED = "balanced"  # Balance multiple factors


@dataclass
class MarketSchedulingConfig:
    """Configuration for market-based scheduling"""

    scheduling_mode: TaskSchedulingMode = TaskSchedulingMode.MARKET_HYBRID
    default_match_strategy: ResourceMatchStrategy = ResourceMatchStrategy.BALANCED

    # Market thresholds
    market_activation_threshold: Decimal = Decimal("10.0")  # Min task value for market
    auction_timeout_minutes: int = 15
    max_price_premium: Decimal = Decimal("0.5")  # 50% premium over base price

    # Quality requirements
    min_trust_score: Decimal = Decimal("0.3")
    max_latency_ms: Decimal = Decimal("500")

    # Fallback behavior
    enable_fallback_to_legacy: bool = True
    fallback_timeout_seconds: int = 300  # 5 minutes

    # Performance optimization
    cache_pricing_seconds: int = 60
    batch_allocation_size: int = 10


@dataclass
class EnhancedTaskRequest:
    """Enhanced task request with market integration"""

    task_id: str
    original_request: dict[str, Any]  # Original fog scheduling request

    # Market enhancements
    market_budget: Decimal | None = None
    preferred_allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED
    task_priority: TaskPriority = TaskPriority.NORMAL

    # Resource requirements (extracted from original)
    cpu_cores: Decimal = Decimal("1.0")
    memory_gb: Decimal = Decimal("1.0")
    duration_hours: Decimal = Decimal("1.0")

    # Market status
    market_request_id: str | None = None
    market_allocated: bool = False
    fallback_to_legacy: bool = False

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    market_allocation_started_at: datetime | None = None
    allocated_at: datetime | None = None

    def to_market_request(self, requester_id: str) -> ResourceAllocationRequest:
        """Convert to market allocation request"""
        return ResourceAllocationRequest(
            request_id=f"market_{self.task_id}",
            requester_id=requester_id,
            task_spec=self.original_request,
            cpu_cores=self.cpu_cores,
            memory_gb=self.memory_gb,
            duration_hours=self.duration_hours,
            allocation_strategy=self.preferred_allocation_strategy,
            task_priority=self.task_priority,
            max_budget=self.market_budget or Decimal("100.0"),
        )


class MarketSchedulerIntegration:
    """
    Integration layer between market-based pricing and fog task scheduler

    Provides seamless integration with existing fog scheduler while adding
    market-based pricing and allocation capabilities.
    """

    def __init__(self, legacy_marketplace_engine=None, config: MarketSchedulingConfig | None = None):
        self.legacy_marketplace = legacy_marketplace_engine
        self.config = config or MarketSchedulingConfig()

        # Market components
        self.market_orchestrator: MarketOrchestrator | None = None
        self.pricing_manager: DynamicPricingManager | None = None

        # Task tracking
        self.active_tasks: dict[str, EnhancedTaskRequest] = {}
        self.completed_tasks: list[EnhancedTaskRequest] = []

        # Performance metrics
        self.allocation_stats = {
            "total_tasks": 0,
            "market_allocated": 0,
            "legacy_allocated": 0,
            "failed_allocations": 0,
            "average_allocation_time": 0.0,
            "cost_savings_percentage": 0.0,
        }

        # Pricing cache
        self.pricing_cache: dict[str, tuple[dict[str, Any], datetime]] = {}

        # Background tasks
        self._integration_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None

        logger.info("Market scheduler integration initialized")

    async def start(self):
        """Start market scheduler integration"""

        # Initialize market components
        self.market_orchestrator = await get_market_orchestrator()
        self.pricing_manager = await get_pricing_manager()

        # Start background tasks
        self._integration_task = asyncio.create_task(self._integration_loop())
        self._metrics_task = asyncio.create_task(self._metrics_loop())

        logger.info("Market scheduler integration started")

    async def stop(self):
        """Stop market scheduler integration"""

        if self._integration_task:
            self._integration_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()

        logger.info("Market scheduler integration stopped")

    async def submit_task_with_market_pricing(
        self, task_spec: dict[str, Any], requester_id: str = "system", **kwargs
    ) -> str:
        """Submit task with market-based pricing and allocation"""

        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Extract resource requirements from task spec
        cpu_cores = Decimal(str(task_spec.get("cpu_cores", 1.0)))
        memory_gb = Decimal(str(task_spec.get("memory_gb", 1.0)))
        duration_hours = Decimal(str(task_spec.get("duration_hours", 1.0)))

        # Create enhanced task request
        enhanced_request = EnhancedTaskRequest(
            task_id=task_id,
            original_request=task_spec,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            duration_hours=duration_hours,
            **kwargs,
        )

        self.active_tasks[task_id] = enhanced_request

        # Start allocation process
        asyncio.create_task(self._allocate_task_with_market(enhanced_request, requester_id))

        logger.info(f"Submitted task {task_id} for market-based allocation")
        return task_id

    async def get_market_price_quote(
        self, cpu_cores: float, memory_gb: float, duration_hours: float = 1.0, strategy: str = "balanced"
    ) -> dict[str, Any]:
        """Get price quote using market pricing (with caching)"""

        # Check cache
        cache_key = f"{cpu_cores}:{memory_gb}:{duration_hours}:{strategy}"

        if cache_key in self.pricing_cache:
            cached_quote, cached_at = self.pricing_cache[cache_key]

            # Check if cache is still valid
            if (datetime.now(UTC) - cached_at).total_seconds() < self.config.cache_pricing_seconds:
                return cached_quote

        # Get fresh quote
        if self.market_orchestrator:
            # Use market-based pricing
            quote = await self.market_orchestrator.get_price_estimate(
                cpu_cores, memory_gb, duration_hours, AllocationStrategy(strategy)
            )

            # Add comparison with legacy pricing if available
            if self.legacy_marketplace:
                legacy_quote = await self._get_legacy_price_quote(cpu_cores, memory_gb, duration_hours)
                quote["legacy_comparison"] = legacy_quote

                # Calculate savings
                market_price = quote.get("adjusted_total", 0)
                legacy_price = legacy_quote.get("avg_price", market_price)

                if legacy_price > 0:
                    savings_percentage = ((legacy_price - market_price) / legacy_price) * 100
                    quote["cost_savings_percentage"] = savings_percentage

        else:
            # Fallback to legacy pricing
            quote = await self._get_legacy_price_quote(cpu_cores, memory_gb, duration_hours)
            quote["pricing_source"] = "legacy_fallback"

        # Cache the result
        self.pricing_cache[cache_key] = (quote, datetime.now(UTC))

        return quote

    async def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get enhanced task status with market information"""

        if task_id not in self.active_tasks:
            # Check completed tasks
            completed_task = next((task for task in self.completed_tasks if task.task_id == task_id), None)
            if not completed_task:
                return None

            return self._format_task_status(completed_task)

        task = self.active_tasks[task_id]
        status = self._format_task_status(task)

        # Add market allocation status if applicable
        if task.market_request_id and self.market_orchestrator:
            market_status = await self.market_orchestrator.get_allocation_status(task.market_request_id)
            status["market_allocation"] = market_status

        return status

    async def switch_scheduling_mode(self, new_mode: TaskSchedulingMode) -> bool:
        """Switch scheduling mode dynamically"""

        old_mode = self.config.scheduling_mode
        self.config.scheduling_mode = new_mode

        logger.info(f"Switched scheduling mode: {old_mode.value} -> {new_mode.value}")

        # Update active tasks if needed
        if new_mode == TaskSchedulingMode.LEGACY:
            # Cancel market allocations and switch to legacy
            await self._migrate_to_legacy_allocation()

        elif old_mode == TaskSchedulingMode.LEGACY and new_mode != TaskSchedulingMode.LEGACY:
            # Initialize market components if not already done
            if not self.market_orchestrator:
                await self.start()

        return True

    async def get_integration_analytics(self) -> dict[str, Any]:
        """Get comprehensive integration analytics"""

        # Calculate performance metrics
        total_tasks = len(self.active_tasks) + len(self.completed_tasks)

        if total_tasks > 0:
            market_success_rate = self.allocation_stats["market_allocated"] / total_tasks
            legacy_usage_rate = self.allocation_stats["legacy_allocated"] / total_tasks
        else:
            market_success_rate = 0.0
            legacy_usage_rate = 0.0

        # Get market component status
        market_stats = {}
        if self.market_orchestrator:
            market_stats = await self.market_orchestrator.get_market_statistics()

        return {
            "integration_status": {
                "scheduling_mode": self.config.scheduling_mode.value,
                "market_orchestrator_active": self.market_orchestrator is not None,
                "pricing_manager_active": self.pricing_manager is not None,
                "legacy_marketplace_available": self.legacy_marketplace is not None,
            },
            "task_statistics": {
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "total_tasks": total_tasks,
                "market_success_rate": market_success_rate,
                "legacy_usage_rate": legacy_usage_rate,
                "failed_allocation_rate": self.allocation_stats["failed_allocations"] / max(1, total_tasks),
            },
            "performance_metrics": {
                "average_allocation_time_seconds": self.allocation_stats["average_allocation_time"],
                "cost_savings_percentage": self.allocation_stats["cost_savings_percentage"],
                "pricing_cache_hit_rate": len(self.pricing_cache) / max(1, total_tasks),
            },
            "market_integration": market_stats,
            "configuration": {
                "market_activation_threshold": float(self.config.market_activation_threshold),
                "auction_timeout_minutes": self.config.auction_timeout_minutes,
                "max_price_premium": float(self.config.max_price_premium),
                "cache_pricing_seconds": self.config.cache_pricing_seconds,
            },
        }

    # Private methods

    async def _allocate_task_with_market(self, task: EnhancedTaskRequest, requester_id: str):
        """Allocate task using market mechanisms with fallback"""

        allocation_start = datetime.now(UTC)
        task.market_allocation_started_at = allocation_start

        try:
            # Determine if task should use market allocation
            should_use_market = await self._should_use_market_allocation(task)

            if should_use_market and self.market_orchestrator:
                # Try market-based allocation
                success = await self._try_market_allocation(task, requester_id)

                if success:
                    task.market_allocated = True
                    self.allocation_stats["market_allocated"] += 1
                else:
                    # Fallback to legacy if enabled
                    if self.config.enable_fallback_to_legacy:
                        success = await self._fallback_to_legacy_allocation(task)
                        if success:
                            task.fallback_to_legacy = True
                            self.allocation_stats["legacy_allocated"] += 1
                        else:
                            self.allocation_stats["failed_allocations"] += 1
                    else:
                        self.allocation_stats["failed_allocations"] += 1

            else:
                # Use legacy allocation directly
                success = await self._use_legacy_allocation(task)
                if success:
                    self.allocation_stats["legacy_allocated"] += 1
                else:
                    self.allocation_stats["failed_allocations"] += 1

            # Update timing metrics
            allocation_time = (datetime.now(UTC) - allocation_start).total_seconds()
            task.allocated_at = datetime.now(UTC)

            # Update average allocation time
            total_allocations = self.allocation_stats["market_allocated"] + self.allocation_stats["legacy_allocated"]

            if total_allocations > 0:
                current_avg = self.allocation_stats["average_allocation_time"]
                self.allocation_stats["average_allocation_time"] = (
                    current_avg * (total_allocations - 1) + allocation_time
                ) / total_allocations

            self.allocation_stats["total_tasks"] += 1

        except Exception as e:
            logger.error(f"Task allocation failed for {task.task_id}: {e}")
            self.allocation_stats["failed_allocations"] += 1

        finally:
            # Move to completed tasks
            self.completed_tasks.append(task)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

    async def _should_use_market_allocation(self, task: EnhancedTaskRequest) -> bool:
        """Determine if task should use market allocation"""

        # Check scheduling mode
        if self.config.scheduling_mode == TaskSchedulingMode.LEGACY:
            return False
        elif self.config.scheduling_mode == TaskSchedulingMode.FULL_MARKET:
            return True

        # For hybrid and adaptive modes, use additional criteria

        # Task value threshold
        task_value = task.cpu_cores * task.memory_gb * task.duration_hours
        if task_value < self.config.market_activation_threshold:
            return False

        # Market component availability
        if not self.market_orchestrator or not self.pricing_manager:
            return False

        # Adaptive mode logic
        if self.config.scheduling_mode == TaskSchedulingMode.ADAPTIVE:
            # Use market conditions to decide
            try:
                market_stats = await self.market_orchestrator.get_market_statistics()
                market_health = market_stats.get("system_health", {})

                # Use market if it's healthy
                if (
                    market_health.get("auction_engine_status") == "active"
                    and market_health.get("pricing_manager_status") == "active"
                ):
                    return True
            except Exception:
                logging.exception("Failed to check market component health status")

        # Default to market for hybrid mode if components are available
        return self.config.scheduling_mode == TaskSchedulingMode.MARKET_HYBRID

    async def _try_market_allocation(self, task: EnhancedTaskRequest, requester_id: str) -> bool:
        """Try to allocate task using market mechanisms"""

        try:
            # Submit market request
            task.to_market_request(requester_id)
            task.market_request_id = await self.market_orchestrator.request_resources(
                requester_id=requester_id,
                task_spec=task.original_request,
                cpu_cores=float(task.cpu_cores),
                memory_gb=float(task.memory_gb),
                duration_hours=float(task.duration_hours),
                allocation_strategy=task.preferred_allocation_strategy,
                task_priority=task.task_priority,
                max_budget=float(task.market_budget) if task.market_budget else 100.0,
            )

            # Wait for allocation with timeout
            timeout_seconds = self.config.fallback_timeout_seconds
            start_time = datetime.now(UTC)

            while (datetime.now(UTC) - start_time).total_seconds() < timeout_seconds:
                status = await self.market_orchestrator.get_allocation_status(task.market_request_id)

                if not status:
                    break

                allocation_status = AllocationStatus(status["status"])

                if allocation_status in [AllocationStatus.ALLOCATED, AllocationStatus.ACTIVE]:
                    logger.info(f"Market allocation successful for task {task.task_id}")
                    return True
                elif allocation_status in [AllocationStatus.FAILED, AllocationStatus.CANCELLED]:
                    logger.warning(f"Market allocation failed for task {task.task_id}: {allocation_status}")
                    break

                await asyncio.sleep(5)  # Check every 5 seconds

            # Timeout or failure
            logger.warning(f"Market allocation timed out for task {task.task_id}")
            return False

        except Exception as e:
            logger.error(f"Market allocation error for task {task.task_id}: {e}")
            return False

    async def _fallback_to_legacy_allocation(self, task: EnhancedTaskRequest) -> bool:
        """Fallback to legacy allocation"""

        logger.info(f"Falling back to legacy allocation for task {task.task_id}")
        return await self._use_legacy_allocation(task)

    async def _use_legacy_allocation(self, task: EnhancedTaskRequest) -> bool:
        """Use legacy marketplace for allocation"""

        if not self.legacy_marketplace:
            logger.error(f"Legacy marketplace not available for task {task.task_id}")
            return False

        try:
            # Convert task to legacy format
            bid_id = await self.legacy_marketplace.submit_bid(
                namespace=task.task_id,
                cpu_cores=float(task.cpu_cores),
                memory_gb=float(task.memory_gb),
                max_price=50.0,  # Default max price
                estimated_duration_hours=float(task.duration_hours),
            )

            if bid_id:
                logger.info(f"Legacy allocation submitted for task {task.task_id}: {bid_id}")
                return True
            else:
                logger.error(f"Legacy allocation failed for task {task.task_id}")
                return False

        except Exception as e:
            logger.error(f"Legacy allocation error for task {task.task_id}: {e}")
            return False

    async def _get_legacy_price_quote(
        self, cpu_cores: float, memory_gb: float, duration_hours: float
    ) -> dict[str, Any]:
        """Get price quote from legacy marketplace"""

        if not self.legacy_marketplace:
            return {"error": "Legacy marketplace not available"}

        try:
            return await self.legacy_marketplace.get_price_quote(cpu_cores, memory_gb, duration_hours)
        except Exception as e:
            logger.error(f"Legacy pricing error: {e}")
            return {"error": str(e)}

    def _format_task_status(self, task: EnhancedTaskRequest) -> dict[str, Any]:
        """Format task status for API response"""

        return {
            "task_id": task.task_id,
            "status": "allocated" if task.allocated_at else "allocating",
            "market_allocation": {
                "used_market": task.market_allocated,
                "fallback_to_legacy": task.fallback_to_legacy,
                "market_request_id": task.market_request_id,
            },
            "resources": {
                "cpu_cores": float(task.cpu_cores),
                "memory_gb": float(task.memory_gb),
                "duration_hours": float(task.duration_hours),
            },
            "timing": {
                "created_at": task.created_at.isoformat(),
                "allocation_started_at": (
                    task.market_allocation_started_at.isoformat() if task.market_allocation_started_at else None
                ),
                "allocated_at": task.allocated_at.isoformat() if task.allocated_at else None,
                "allocation_time_seconds": (
                    (task.allocated_at - task.market_allocation_started_at).total_seconds()
                    if task.allocated_at and task.market_allocation_started_at
                    else None
                ),
            },
            "preferences": {
                "allocation_strategy": task.preferred_allocation_strategy.value,
                "task_priority": task.task_priority.value,
                "market_budget": float(task.market_budget) if task.market_budget else None,
            },
        }

    async def _migrate_to_legacy_allocation(self):
        """Migrate active market allocations to legacy"""

        tasks_to_migrate = [
            task for task in self.active_tasks.values() if task.market_request_id and not task.market_allocated
        ]

        for task in tasks_to_migrate:
            try:
                # Cancel market request
                if self.market_orchestrator:
                    await self.market_orchestrator.cancel_allocation_request(
                        task.market_request_id, "Switching to legacy mode"
                    )

                # Start legacy allocation
                await self._use_legacy_allocation(task)

            except Exception as e:
                logger.error(f"Migration to legacy failed for task {task.task_id}: {e}")

        logger.info(f"Migrated {len(tasks_to_migrate)} tasks to legacy allocation")

    async def _integration_loop(self):
        """Background task for integration maintenance"""

        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Clean up old completed tasks
                cutoff = datetime.now(UTC) - timedelta(hours=24)
                old_tasks = [
                    task
                    for task in self.completed_tasks
                    if task.allocated_at and task.allocated_at.replace(tzinfo=UTC) < cutoff
                ]

                for task in old_tasks:
                    self.completed_tasks.remove(task)

                # Clear old pricing cache
                cache_cutoff = datetime.now(UTC) - timedelta(seconds=self.config.cache_pricing_seconds * 2)
                expired_keys = [key for key, (_, cached_at) in self.pricing_cache.items() if cached_at < cache_cutoff]

                for key in expired_keys:
                    del self.pricing_cache[key]

                if old_tasks or expired_keys:
                    logger.debug(f"Cleaned up {len(old_tasks)} old tasks and {len(expired_keys)} cache entries")

            except Exception as e:
                logger.error(f"Error in integration loop: {e}")
                await asyncio.sleep(300)

    async def _metrics_loop(self):
        """Background task for metrics calculation"""

        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes

                # Calculate cost savings
                if self.allocation_stats["market_allocated"] > 0:
                    # This would calculate actual savings in production
                    # For now, use a placeholder calculation
                    estimated_savings = 15.0  # 15% average savings
                    self.allocation_stats["cost_savings_percentage"] = estimated_savings

            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(600)


# Global integration instance
_scheduler_integration: MarketSchedulerIntegration | None = None


async def get_scheduler_integration() -> MarketSchedulerIntegration:
    """Get global scheduler integration instance"""
    global _scheduler_integration

    if _scheduler_integration is None:
        _scheduler_integration = MarketSchedulerIntegration()
        await _scheduler_integration.start()

    return _scheduler_integration


# Convenience functions
async def submit_market_task(task_spec: dict[str, Any], requester_id: str = "system", **kwargs) -> str:
    """Submit task with market-based allocation"""

    integration = await get_scheduler_integration()
    return await integration.submit_task_with_market_pricing(task_spec, requester_id, **kwargs)


async def get_market_task_quote(cpu_cores: float, memory_gb: float, duration_hours: float = 1.0) -> dict[str, Any]:
    """Get pricing quote for task resources"""

    integration = await get_scheduler_integration()
    return await integration.get_market_price_quote(cpu_cores, memory_gb, duration_hours)


async def switch_to_market_mode() -> bool:
    """Switch scheduler to full market mode"""

    integration = await get_scheduler_integration()
    return await integration.switch_scheduling_mode(TaskSchedulingMode.FULL_MARKET)
