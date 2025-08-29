"""
Market-Based Pricing System for Fog Computing

Comprehensive market mechanisms for dynamic resource pricing and allocation:
- Reverse auction engine with second-price settlement
- Dynamic pricing manager with anti-manipulation safeguards
- Market orchestrator for unified resource allocation
- Integration with existing tokenomics and fog infrastructure

Key Components:
- AuctionEngine: Handles reverse auctions with deposits and quality scoring
- DynamicPricingManager: Lane-specific pricing with volatility protection
- MarketOrchestrator: Coordinates market operations and resource allocation

Usage:
    from infrastructure.fog.market import (
        request_fog_resources,
        get_fog_resource_quote,
        check_allocation_status
    )

    # Request resources through market mechanisms
    request_id = await request_fog_resources(
        requester_id="user123",
        task_spec={"type": "ml_training"},
        cpu_cores=4.0,
        memory_gb=16.0,
        duration_hours=2.0,
        max_budget=25.0
    )

    # Get pricing quote
    quote = await get_fog_resource_quote(
        cpu_cores=2.0,
        memory_gb=8.0,
        strategy="balanced"
    )

    # Check allocation status
    status = await check_allocation_status(request_id)
"""

from .auction_engine import (
    Auction,
    AuctionBid,
    AuctionEngine,
    AuctionResult,
    AuctionStatus,
    AuctionType,
    BidStatus,
    ResourceRequirement,
    create_reverse_auction,
    get_auction_engine,
    submit_provider_bid,
)
from .market_orchestrator import (
    AllocationStatus,
    AllocationStrategy,
    MarketAllocationResult,
    MarketOrchestrator,
    ResourceAllocationRequest,
    TaskPriority,
    check_allocation_status,
    get_fog_resource_quote,
    get_market_orchestrator,
    request_fog_resources,
)
from .pricing_manager import (
    DynamicPricingManager,
    MarketCondition,
    MarketMetrics,
    PriceBand,
    PricingAnomalyAlert,
    PricingStrategy,
    ResourceLane,
    get_current_resource_price,
    get_dynamic_reserve_price,
    get_pricing_manager,
    update_resource_supply_demand,
)

# Version information
__version__ = "1.0.0"
__author__ = "AIVillage Development Team"

# Export main interfaces
__all__ = [
    # Auction Engine
    "AuctionEngine",
    "AuctionType",
    "AuctionStatus",
    "BidStatus",
    "ResourceRequirement",
    "AuctionBid",
    "AuctionResult",
    "Auction",
    "get_auction_engine",
    "create_reverse_auction",
    "submit_provider_bid",
    # Pricing Manager
    "DynamicPricingManager",
    "ResourceLane",
    "PricingStrategy",
    "MarketCondition",
    "PriceBand",
    "PricingAnomalyAlert",
    "MarketMetrics",
    "get_pricing_manager",
    "get_current_resource_price",
    "update_resource_supply_demand",
    "get_dynamic_reserve_price",
    # Market Orchestrator
    "MarketOrchestrator",
    "AllocationStrategy",
    "TaskPriority",
    "AllocationStatus",
    "ResourceAllocationRequest",
    "MarketAllocationResult",
    "get_market_orchestrator",
    "request_fog_resources",
    "get_fog_resource_quote",
    "check_allocation_status",
]
