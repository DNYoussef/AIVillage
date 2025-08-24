"""
Fog Gateway Marketplace Engine

Implements minimal viable marketplace for fog compute resource renting:
- Spot & on-demand bidding with dynamic pricing
- Trust-based matching algorithms
- Resource reservation and allocation
- Real-time price discovery

Economic Model:
- Spot Pricing: Dynamic pricing based on supply/demand
- On-Demand Pricing: Fixed pricing for guaranteed availability
- Trust Premium: Higher trust nodes charge premium rates
- Bid Matching: max_price × trust optimization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class BidType(str, Enum):
    """Resource bidding types"""

    SPOT = "spot"  # Dynamic pricing, can be preempted
    ON_DEMAND = "on_demand"  # Fixed pricing, guaranteed execution
    RESERVED = "reserved"  # Pre-purchased capacity at discount


class BidStatus(str, Enum):
    """Bid processing status"""

    PENDING = "pending"  # Bid submitted, awaiting matching
    MATCHED = "matched"  # Matched with resource, awaiting confirmation
    ACTIVE = "active"  # Resources reserved and executing
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"  # Bid matching or execution failed
    CANCELLED = "cancelled"  # Bid cancelled by user
    PREEMPTED = "preempted"  # Spot instance preempted


class PricingTier(str, Enum):
    """Pricing tiers based on SLA classes"""

    BASIC = "basic"  # B-class: best effort, lowest cost
    STANDARD = "standard"  # A-class: replicated, moderate cost
    PREMIUM = "premium"  # S-class: replicated+attested, highest cost


@dataclass
class ResourceListing:
    """Available resources advertised by fog node"""

    listing_id: str
    node_id: str

    # Resource specification
    cpu_cores: float
    memory_gb: float
    disk_gb: float

    # Pricing
    spot_price_per_cpu_hour: float
    on_demand_price_per_cpu_hour: float
    pricing_tier: PricingTier = PricingTier.BASIC

    # Quality metrics
    trust_score: float = 0.5  # 0.0-1.0 trust rating
    reputation_score: float = 0.5  # Based on historical performance
    latency_ms: float = 100.0  # Network latency to gateway

    # Availability
    available_until: datetime | None = None
    min_duration_minutes: int = 5
    max_duration_hours: int = 24

    # Constraints
    min_trust_required: float = 0.0  # Minimum bidder trust required
    accepts_spot_bids: bool = True
    accepts_on_demand: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_available(self) -> bool:
        """Check if listing is still available"""
        if self.available_until is None:
            return True
        return datetime.now(UTC) < self.available_until

    def matches_requirements(
        self, cpu_required: float, memory_required: float, duration_hours: float, bidder_trust: float
    ) -> bool:
        """Check if listing can satisfy resource requirements"""

        return (
            self.is_available()
            and self.cpu_cores >= cpu_required
            and self.memory_gb >= memory_required
            and duration_hours >= (self.min_duration_minutes / 60.0)
            and duration_hours <= self.max_duration_hours
            and bidder_trust >= self.min_trust_required
        )

    def calculate_cost(self, cpu_cores: float, memory_gb: float, duration_hours: float, bid_type: BidType) -> float:
        """Calculate cost for resource usage"""

        if bid_type == BidType.SPOT:
            base_price = self.spot_price_per_cpu_hour
        else:
            base_price = self.on_demand_price_per_cpu_hour

        # Base cost calculation
        cpu_cost = cpu_cores * duration_hours * base_price
        memory_cost = memory_gb * duration_hours * (base_price * 0.1)  # Memory is 10% of CPU price

        # Trust premium: higher trust nodes charge more
        trust_multiplier = 1.0 + (self.trust_score * 0.5)  # Up to 50% premium

        # Pricing tier multiplier
        tier_multiplier = {PricingTier.BASIC: 1.0, PricingTier.STANDARD: 1.5, PricingTier.PREMIUM: 2.0}[
            self.pricing_tier
        ]

        total_cost = (cpu_cost + memory_cost) * trust_multiplier * tier_multiplier
        return round(total_cost, 4)


@dataclass
class ResourceBid:
    """Resource bid submitted by user"""

    bid_id: str
    namespace: str
    user_id: str | None = None

    # Resource requirements
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    disk_gb: float = 2.0
    estimated_duration_hours: float = 1.0

    # Pricing constraints
    bid_type: BidType = BidType.SPOT
    max_price: float = 1.0  # Maximum willing to pay
    pricing_tier: PricingTier = PricingTier.BASIC

    # Quality requirements
    min_trust_score: float = 0.3
    max_latency_ms: float = 500.0
    preferred_regions: list[str] = field(default_factory=list)

    # Job specification
    job_spec: dict[str, Any] = field(default_factory=dict)

    # Status tracking
    status: BidStatus = BidStatus.PENDING
    matched_listing_id: str | None = None
    allocated_node_id: str | None = None
    actual_cost: float | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    matched_at: datetime | None = None
    completed_at: datetime | None = None

    def is_expired(self, ttl_minutes: int = 10) -> bool:
        """Check if bid has expired"""
        expiry = self.created_at + timedelta(minutes=ttl_minutes)
        return datetime.now(UTC) > expiry

    def can_afford(self, quoted_price: float) -> bool:
        """Check if bid can afford quoted price"""
        return quoted_price <= self.max_price

    def calculate_score(self, listing: ResourceListing) -> float:
        """Calculate bid-listing matching score (higher is better)"""

        # Price affordability (0.0-1.0)
        quoted_price = listing.calculate_cost(
            self.cpu_cores, self.memory_gb, self.estimated_duration_hours, self.bid_type
        )

        if quoted_price > self.max_price:
            return 0.0  # Can't afford, no match

        price_score = 1.0 - (quoted_price / self.max_price)

        # Trust score (0.0-1.0)
        trust_score = min(1.0, listing.trust_score / max(0.1, self.min_trust_score))

        # Latency score (0.0-1.0)
        latency_score = max(0.0, 1.0 - (listing.latency_ms / self.max_latency_ms))

        # Pricing tier compatibility
        tier_score = 1.0 if listing.pricing_tier == self.pricing_tier else 0.8

        # Weighted composite score
        composite_score = (
            price_score * 0.4
            + trust_score * 0.3  # Price is most important
            + latency_score * 0.2  # Trust is critical
            + tier_score * 0.1  # Performance matters  # Tier preference
        )

        return composite_score


@dataclass
class MarketplaceTrade:
    """Completed marketplace trade record"""

    trade_id: str
    bid_id: str
    listing_id: str

    # Participants
    buyer_namespace: str
    seller_node_id: str

    # Resources traded
    cpu_cores: float
    memory_gb: float
    duration_hours: float

    # Financial details
    agreed_price: float
    bid_type: BidType
    pricing_tier: PricingTier

    # Performance metrics
    actual_duration_hours: float | None = None
    job_success: bool = False
    performance_score: float = 0.0  # 0.0-1.0 based on SLA compliance

    # Timestamps
    executed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None


class MarketplacePricingEngine:
    """Dynamic pricing engine for fog marketplace"""

    def __init__(self):
        self.base_spot_price = 0.10  # Base price per CPU-hour
        self.base_on_demand_price = 0.15
        self.demand_multiplier = 1.0
        self.supply_multiplier = 1.0

        # Price history for volatility calculation
        self.price_history: list[tuple[datetime, float]] = []

    def update_market_conditions(self, total_demand: float, total_supply: float, utilization_rate: float) -> None:
        """Update pricing based on market conditions"""

        # Demand pressure: more demand = higher prices
        if total_supply > 0:
            demand_ratio = total_demand / total_supply
            self.demand_multiplier = 1.0 + min(2.0, demand_ratio * 0.5)
        else:
            self.demand_multiplier = 2.0  # Scarcity pricing

        # Supply abundance: more supply = lower prices
        if total_demand > 0:
            supply_ratio = total_supply / total_demand
            self.supply_multiplier = max(0.5, 1.0 - (supply_ratio * 0.2))
        else:
            self.supply_multiplier = 0.5  # Abundant supply

        # Utilization pressure: high utilization = higher prices
        utilization_multiplier = 1.0 + (utilization_rate * 0.3)

        # Calculate new spot price
        new_spot_price = self.base_spot_price * self.demand_multiplier * self.supply_multiplier * utilization_multiplier

        # Store price history
        self.price_history.append((datetime.now(UTC), new_spot_price))

        # Keep only last 24 hours
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        self.price_history = [(ts, price) for ts, price in self.price_history if ts > cutoff]

        logger.info(
            f"Market pricing updated: spot=${new_spot_price:.4f}/cpu-hour "
            f"(demand={self.demand_multiplier:.2f}, supply={self.supply_multiplier:.2f})"
        )

    def get_current_spot_price(self) -> float:
        """Get current spot price per CPU-hour"""
        return self.base_spot_price * self.demand_multiplier * self.supply_multiplier

    def get_current_on_demand_price(self) -> float:
        """Get current on-demand price per CPU-hour"""
        return self.base_on_demand_price * self.demand_multiplier

    def get_price_volatility(self) -> float:
        """Calculate price volatility over last 24 hours"""
        if len(self.price_history) < 2:
            return 0.0

        prices = [price for _, price in self.price_history]
        mean_price = sum(prices) / len(prices)
        variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
        volatility = (variance**0.5) / mean_price

        return volatility


class MarketplaceEngine:
    """
    Core fog compute marketplace engine

    Handles resource bidding, matching, and allocation with:
    - Multi-unit sealed-bid auctions
    - Trust-based matching algorithms
    - Dynamic spot pricing
    - Resource reservation management
    """

    def __init__(self):
        # Market state
        self.active_listings: dict[str, ResourceListing] = {}
        self.pending_bids: dict[str, ResourceBid] = {}
        self.active_trades: dict[str, MarketplaceTrade] = {}
        self.trade_history: list[MarketplaceTrade] = []

        # Pricing engine
        self.pricing_engine = MarketplacePricingEngine()

        # Market statistics
        self.total_trades = 0
        self.total_volume = 0.0  # Total USD traded
        self.avg_utilization = 0.0

        # Background tasks
        self._matching_task: asyncio.Task | None = None
        self._pricing_task: asyncio.Task | None = None

        logger.info("Marketplace engine initialized")

    async def start(self) -> None:
        """Start marketplace background tasks"""

        self._matching_task = asyncio.create_task(self._bid_matching_loop())
        self._pricing_task = asyncio.create_task(self._pricing_update_loop())

        logger.info("Marketplace engine started")

    async def stop(self) -> None:
        """Stop marketplace background tasks"""

        if self._matching_task:
            self._matching_task.cancel()
        if self._pricing_task:
            self._pricing_task.cancel()

        logger.info("Marketplace engine stopped")

    async def add_resource_listing(
        self,
        node_id: str,
        cpu_cores: float,
        memory_gb: float,
        disk_gb: float,
        spot_price: float,
        on_demand_price: float,
        trust_score: float = 0.5,
        pricing_tier: PricingTier = PricingTier.BASIC,
        **kwargs,
    ) -> str:
        """Add resource listing from fog node"""

        listing_id = f"listing_{uuid4().hex[:8]}"

        listing = ResourceListing(
            listing_id=listing_id,
            node_id=node_id,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            spot_price_per_cpu_hour=spot_price,
            on_demand_price_per_cpu_hour=on_demand_price,
            trust_score=trust_score,
            pricing_tier=pricing_tier,
            **kwargs,
        )

        self.active_listings[listing_id] = listing

        logger.info(
            f"Added resource listing {listing_id}: {cpu_cores} cores, "
            f"{memory_gb}GB memory, spot=${spot_price:.4f}/cpu-hour"
        )

        return listing_id

    async def submit_bid(
        self,
        namespace: str,
        cpu_cores: float,
        memory_gb: float,
        max_price: float,
        bid_type: BidType = BidType.SPOT,
        estimated_duration_hours: float = 1.0,
        job_spec: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Submit resource bid"""

        bid_id = f"bid_{uuid4().hex[:8]}"

        bid = ResourceBid(
            bid_id=bid_id,
            namespace=namespace,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            max_price=max_price,
            bid_type=bid_type,
            estimated_duration_hours=estimated_duration_hours,
            job_spec=job_spec or {},
            **kwargs,
        )

        self.pending_bids[bid_id] = bid

        logger.info(
            f"Submitted bid {bid_id}: {cpu_cores} cores, {memory_gb}GB memory, "
            f"max_price=${max_price:.4f}, type={bid_type.value}"
        )

        return bid_id

    async def get_price_quote(
        self,
        cpu_cores: float,
        memory_gb: float,
        duration_hours: float,
        bid_type: BidType = BidType.SPOT,
        pricing_tier: PricingTier = PricingTier.BASIC,
        min_trust_score: float = 0.3,
    ) -> dict[str, Any]:
        """Get price quote for resource requirements"""

        # Find available listings that match requirements
        matching_listings = []

        for listing in self.active_listings.values():
            if listing.matches_requirements(cpu_cores, memory_gb, duration_hours, min_trust_score):
                cost = listing.calculate_cost(cpu_cores, memory_gb, duration_hours, bid_type)
                matching_listings.append((listing, cost))

        if not matching_listings:
            return {
                "available": False,
                "reason": "No matching resources available",
                "suggestions": await self._get_availability_suggestions(cpu_cores, memory_gb, duration_hours),
            }

        # Sort by cost (ascending)
        matching_listings.sort(key=lambda x: x[1])

        # Price statistics
        costs = [cost for _, cost in matching_listings]
        min_price = min(costs)
        max_price = max(costs)
        avg_price = sum(costs) / len(costs)

        # Market price (current spot/on-demand rates)
        if bid_type == BidType.SPOT:
            market_price_per_hour = self.pricing_engine.get_current_spot_price()
        else:
            market_price_per_hour = self.pricing_engine.get_current_on_demand_price()

        market_estimate = cpu_cores * duration_hours * market_price_per_hour

        return {
            "available": True,
            "quote": {
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": avg_price,
                "market_estimate": market_estimate,
                "currency": "USD",
            },
            "market_conditions": {
                "bid_type": bid_type.value,
                "pricing_tier": pricing_tier.value,
                "current_spot_rate": self.pricing_engine.get_current_spot_price(),
                "current_on_demand_rate": self.pricing_engine.get_current_on_demand_price(),
                "price_volatility": self.pricing_engine.get_price_volatility(),
                "available_providers": len(matching_listings),
            },
            "recommendations": {
                "suggested_max_price": avg_price * 1.1,  # 10% buffer
                "best_value_listing": matching_listings[0][0].listing_id,
                "estimated_wait_time_minutes": 2 if len(matching_listings) > 5 else 5,
            },
        }

    async def get_marketplace_status(self) -> dict[str, Any]:
        """Get comprehensive marketplace status"""

        # Active listings summary
        total_cpu = sum(l.cpu_cores for l in self.active_listings.values())
        total_memory = sum(l.memory_gb for l in self.active_listings.values())
        avg_trust = sum(l.trust_score for l in self.active_listings.values()) / max(1, len(self.active_listings))

        # Pending bids summary
        pending_bid_demand = sum(b.cpu_cores for b in self.pending_bids.values())
        avg_max_price = sum(b.max_price for b in self.pending_bids.values()) / max(1, len(self.pending_bids))

        # Pricing metrics
        current_spot = self.pricing_engine.get_current_spot_price()
        current_on_demand = self.pricing_engine.get_current_on_demand_price()
        volatility = self.pricing_engine.get_price_volatility()

        return {
            "marketplace_summary": {
                "active_listings": len(self.active_listings),
                "pending_bids": len(self.pending_bids),
                "active_trades": len(self.active_trades),
                "total_completed_trades": self.total_trades,
                "total_volume_usd": self.total_volume,
            },
            "resource_supply": {
                "total_cpu_cores": total_cpu,
                "total_memory_gb": total_memory,
                "avg_trust_score": avg_trust,
                "utilization_rate": self.avg_utilization,
            },
            "resource_demand": {
                "pending_bid_cpu_demand": pending_bid_demand,
                "avg_bid_max_price": avg_max_price,
                "demand_supply_ratio": pending_bid_demand / max(1.0, total_cpu),
            },
            "pricing": {
                "current_spot_price_per_cpu_hour": current_spot,
                "current_on_demand_price_per_cpu_hour": current_on_demand,
                "price_volatility_24h": volatility,
                "demand_multiplier": self.pricing_engine.demand_multiplier,
                "supply_multiplier": self.pricing_engine.supply_multiplier,
            },
            "market_health": {
                "liquidity_score": min(1.0, len(self.active_listings) / 10.0),
                "price_stability": max(0.0, 1.0 - volatility),
                "trust_quality": avg_trust,
                "matching_efficiency": self._calculate_matching_efficiency(),
            },
        }

    # Private methods

    async def _bid_matching_loop(self) -> None:
        """Background task for matching bids with listings"""

        while True:
            try:
                await asyncio.sleep(5)  # Match every 5 seconds

                # Remove expired bids
                expired_bids = [bid_id for bid_id, bid in self.pending_bids.items() if bid.is_expired()]

                for bid_id in expired_bids:
                    del self.pending_bids[bid_id]
                    logger.info(f"Removed expired bid: {bid_id}")

                # Remove unavailable listings
                unavailable_listings = [
                    listing_id for listing_id, listing in self.active_listings.items() if not listing.is_available()
                ]

                for listing_id in unavailable_listings:
                    del self.active_listings[listing_id]
                    logger.info(f"Removed unavailable listing: {listing_id}")

                # Attempt to match pending bids
                matched_count = await self._match_bids_to_listings()

                if matched_count > 0:
                    logger.info(f"Matched {matched_count} bids to listings")

            except Exception as e:
                logger.error(f"Error in bid matching loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _match_bids_to_listings(self) -> int:
        """Match pending bids to available listings"""

        matched_count = 0

        for bid_id, bid in list(self.pending_bids.items()):
            if bid.status != BidStatus.PENDING:
                continue

            # Find best matching listing
            best_listing = None
            best_score = 0.0

            for listing in self.active_listings.values():
                if listing.matches_requirements(
                    bid.cpu_cores, bid.memory_gb, bid.estimated_duration_hours, bid.min_trust_score
                ):
                    score = bid.calculate_score(listing)
                    if score > best_score:
                        best_score = score
                        best_listing = listing

            # Execute trade if good match found
            if best_listing and best_score > 0.5:  # Minimum score threshold
                await self._execute_trade(bid, best_listing)
                matched_count += 1

        return matched_count

    async def _execute_trade(self, bid: ResourceBid, listing: ResourceListing) -> None:
        """Execute trade between bid and listing"""

        # Calculate final price
        final_price = listing.calculate_cost(bid.cpu_cores, bid.memory_gb, bid.estimated_duration_hours, bid.bid_type)

        # Verify bid can afford the price
        if not bid.can_afford(final_price):
            logger.warning(f"Bid {bid.bid_id} cannot afford price ${final_price:.4f}")
            return

        # Create trade record
        trade_id = f"trade_{uuid4().hex[:8]}"

        trade = MarketplaceTrade(
            trade_id=trade_id,
            bid_id=bid.bid_id,
            listing_id=listing.listing_id,
            buyer_namespace=bid.namespace,
            seller_node_id=listing.node_id,
            cpu_cores=bid.cpu_cores,
            memory_gb=bid.memory_gb,
            duration_hours=bid.estimated_duration_hours,
            agreed_price=final_price,
            bid_type=bid.bid_type,
            pricing_tier=listing.pricing_tier,
        )

        # Update bid status
        bid.status = BidStatus.MATCHED
        bid.matched_listing_id = listing.listing_id
        bid.allocated_node_id = listing.node_id
        bid.actual_cost = final_price
        bid.matched_at = datetime.now(UTC)

        # Store trade
        self.active_trades[trade_id] = trade

        # Remove matched bid and listing
        del self.pending_bids[bid.bid_id]
        del self.active_listings[listing.listing_id]

        # Update statistics
        self.total_trades += 1
        self.total_volume += final_price

        logger.info(
            f"Executed trade {trade_id}: {bid.cpu_cores} cores for "
            f"${final_price:.4f} between {bid.namespace} and {listing.node_id}"
        )

    async def _pricing_update_loop(self) -> None:
        """Background task for updating market pricing"""

        while True:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Calculate market conditions
                total_supply = sum(l.cpu_cores for l in self.active_listings.values())
                total_demand = sum(b.cpu_cores for b in self.pending_bids.values())

                # Calculate utilization from active trades
                active_cpu = sum(t.cpu_cores for t in self.active_trades.values())
                total_capacity = total_supply + active_cpu

                if total_capacity > 0:
                    utilization_rate = active_cpu / total_capacity
                else:
                    utilization_rate = 0.0

                self.avg_utilization = utilization_rate

                # Update pricing engine
                self.pricing_engine.update_market_conditions(total_demand, total_supply, utilization_rate)

            except Exception as e:
                logger.error(f"Error in pricing update loop: {e}")
                await asyncio.sleep(300)  # Wait longer on error

    async def _get_availability_suggestions(
        self, cpu_cores: float, memory_gb: float, duration_hours: float
    ) -> dict[str, Any]:
        """Get suggestions when no resources are available"""

        suggestions = {"reduce_requirements": [], "increase_budget": [], "wait_for_availability": []}

        # Check if reducing requirements would help
        if cpu_cores > 1.0:
            suggestions["reduce_requirements"].append(f"Reduce CPU from {cpu_cores} to {cpu_cores * 0.5} cores")

        if memory_gb > 1.0:
            suggestions["reduce_requirements"].append(f"Reduce memory from {memory_gb} to {memory_gb * 0.5}GB")

        # Get price range for increased budget suggestions
        avg_listing_price = 0.15  # Default estimate
        if self.active_listings:
            prices = [l.spot_price_per_cpu_hour for l in self.active_listings.values()]
            avg_listing_price = sum(prices) / len(prices)

        estimated_budget = cpu_cores * duration_hours * avg_listing_price * 1.5
        suggestions["increase_budget"].append(f"Consider budget of ${estimated_budget:.2f}")

        # Estimate wait times
        suggestions["wait_for_availability"].append("Check again in 5-10 minutes for new listings")

        return suggestions

    def _calculate_matching_efficiency(self) -> float:
        """Calculate marketplace matching efficiency (0.0-1.0)"""

        total_bids = len(self.pending_bids) + self.total_trades
        if total_bids == 0:
            return 1.0

        return self.total_trades / total_bids


# Global marketplace engine instance
_marketplace_engine: MarketplaceEngine | None = None


async def get_marketplace_engine() -> MarketplaceEngine:
    """Get global marketplace engine instance"""
    global _marketplace_engine

    if _marketplace_engine is None:
        _marketplace_engine = MarketplaceEngine()
        await _marketplace_engine.start()

    return _marketplace_engine


# Convenience functions for integration
async def submit_resource_bid(namespace: str, cpu_cores: float, memory_gb: float, max_price: float, **kwargs) -> str:
    """Submit resource bid to marketplace"""

    engine = await get_marketplace_engine()
    return await engine.submit_bid(namespace, cpu_cores, memory_gb, max_price, **kwargs)


async def get_market_price_quote(
    cpu_cores: float, memory_gb: float, duration_hours: float = 1.0, **kwargs
) -> dict[str, Any]:
    """Get price quote from marketplace"""

    engine = await get_marketplace_engine()
    return await engine.get_price_quote(cpu_cores, memory_gb, duration_hours, **kwargs)


async def advertise_resources(
    node_id: str, cpu_cores: float, memory_gb: float, spot_price: float, on_demand_price: float, **kwargs
) -> str:
    """Advertise available resources in marketplace"""

    engine = await get_marketplace_engine()
    return await engine.add_resource_listing(node_id, cpu_cores, memory_gb, 0.0, spot_price, on_demand_price, **kwargs)
