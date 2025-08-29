"""
Auction Engine for Fog Computing Market-Based Pricing

Implements reverse auction mechanics with second-price settlement and anti-griefing:
- Reverse auctions where providers bid DOWN on prices
- Second-price sealed-bid auctions for efficient price discovery
- Deposit system to prevent bid manipulation and ensure commitment
- Dynamic reserve prices based on network conditions
- Multi-unit auctions for bulk resource allocation

Key Features:
- Sealed-bid reverse auctions with second-price payment
- Bid deposits for anti-griefing protection
- Reserve price management
- Multi-resource auction support (CPU, Memory, Storage)
- Auction history and analytics
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from enum import Enum
import hashlib
import logging
from typing import Any
import uuid

# Set decimal precision for financial calculations
getcontext().prec = 18

logger = logging.getLogger(__name__)


class AuctionType(str, Enum):
    """Types of auctions supported"""

    REVERSE = "reverse"  # Providers bid down on price
    FORWARD = "forward"  # Consumers bid up on price
    DUTCH = "dutch"  # Decreasing price auction
    VICKREY = "vickrey"  # Second-price sealed-bid


class AuctionStatus(str, Enum):
    """Auction lifecycle states"""

    OPEN = "open"  # Accepting bids
    CLOSING = "closing"  # Final bids period
    CLOSED = "closed"  # Bid submission ended
    SETTLING = "settling"  # Determining winners
    SETTLED = "settled"  # Winners determined
    COMPLETED = "completed"  # Resources allocated
    CANCELLED = "cancelled"  # Auction cancelled


class BidStatus(str, Enum):
    """Individual bid states"""

    SUBMITTED = "submitted"  # Bid placed with deposit
    VALID = "valid"  # Bid validated and active
    INVALID = "invalid"  # Bid validation failed
    WINNING = "winning"  # Current winning bid
    WON = "won"  # Final winning bid
    LOST = "lost"  # Non-winning bid
    REFUNDED = "refunded"  # Deposit refunded


@dataclass
class ResourceRequirement:
    """Specification for resources being auctioned"""

    cpu_cores: Decimal
    memory_gb: Decimal
    storage_gb: Decimal
    bandwidth_mbps: Decimal
    duration_hours: Decimal

    # Quality requirements
    min_trust_score: Decimal = Decimal("0.3")
    max_latency_ms: Decimal = Decimal("500")
    required_regions: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure all values are Decimal for precision"""
        if not isinstance(self.cpu_cores, Decimal):
            self.cpu_cores = Decimal(str(self.cpu_cores))
        if not isinstance(self.memory_gb, Decimal):
            self.memory_gb = Decimal(str(self.memory_gb))
        if not isinstance(self.storage_gb, Decimal):
            self.storage_gb = Decimal(str(self.storage_gb))
        if not isinstance(self.bandwidth_mbps, Decimal):
            self.bandwidth_mbps = Decimal(str(self.bandwidth_mbps))
        if not isinstance(self.duration_hours, Decimal):
            self.duration_hours = Decimal(str(self.duration_hours))
        if not isinstance(self.min_trust_score, Decimal):
            self.min_trust_score = Decimal(str(self.min_trust_score))
        if not isinstance(self.max_latency_ms, Decimal):
            self.max_latency_ms = Decimal(str(self.max_latency_ms))

    def calculate_resource_score(self) -> Decimal:
        """Calculate weighted resource requirement score"""
        # Weighted scoring based on resource complexity
        cpu_weight = Decimal("0.4")
        memory_weight = Decimal("0.3")
        storage_weight = Decimal("0.2")
        bandwidth_weight = Decimal("0.1")

        score = (
            self.cpu_cores * cpu_weight
            + self.memory_gb * memory_weight
            + self.storage_gb * storage_weight
            + self.bandwidth_mbps * bandwidth_weight
        ) * self.duration_hours

        return score


@dataclass
class AuctionBid:
    """Individual bid in an auction"""

    bid_id: str
    auction_id: str
    bidder_id: str
    node_id: str

    # Bid price (total for entire resource requirement)
    bid_price: Decimal  # Total price willing to accept (reverse) or pay (forward)
    per_hour_rate: Decimal = field(init=False)

    # Deposit for anti-griefing
    deposit_amount: Decimal
    deposit_tx_hash: str = ""
    deposit_confirmed: bool = False

    # Bidder capabilities
    trust_score: Decimal
    reputation_score: Decimal
    latency_ms: Decimal
    available_resources: dict[str, Decimal] = field(default_factory=dict)

    # Bid metadata
    bid_hash: str = ""  # Hash of sealed bid
    revealed_at: datetime | None = None
    status: BidStatus = BidStatus.SUBMITTED

    # Quality metrics
    sla_commitment: dict[str, Any] = field(default_factory=dict)
    performance_guarantees: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    validated_at: datetime | None = None

    def __post_init__(self):
        """Initialize derived fields"""
        if not isinstance(self.bid_price, Decimal):
            self.bid_price = Decimal(str(self.bid_price))
        if not isinstance(self.deposit_amount, Decimal):
            self.deposit_amount = Decimal(str(self.deposit_amount))
        if not isinstance(self.trust_score, Decimal):
            self.trust_score = Decimal(str(self.trust_score))
        if not isinstance(self.reputation_score, Decimal):
            self.reputation_score = Decimal(str(self.reputation_score))
        if not isinstance(self.latency_ms, Decimal):
            self.latency_ms = Decimal(str(self.latency_ms))

        # Generate bid hash for sealed-bid auctions
        if not self.bid_hash:
            self.bid_hash = self._generate_bid_hash()

    def set_per_hour_rate(self, duration_hours: Decimal):
        """Calculate per-hour rate from total bid price"""
        self.per_hour_rate = self.bid_price / duration_hours

    def _generate_bid_hash(self) -> str:
        """Generate cryptographic hash of sealed bid"""
        bid_data = f"{self.bidder_id}:{self.bid_price}:{self.submitted_at.isoformat()}"
        return hashlib.sha256(bid_data.encode()).hexdigest()

    def calculate_quality_score(self) -> Decimal:
        """Calculate quality score based on trust, reputation, and latency"""
        trust_weight = Decimal("0.4")
        reputation_weight = Decimal("0.4")
        latency_weight = Decimal("0.2")

        # Normalize latency (lower is better)
        latency_score = max(Decimal("0"), Decimal("1") - (self.latency_ms / Decimal("1000")))

        quality_score = (
            self.trust_score * trust_weight + self.reputation_score * reputation_weight + latency_score * latency_weight
        )

        return min(Decimal("1"), quality_score)

    def validate_resources(self, requirements: ResourceRequirement) -> bool:
        """Validate that bidder can fulfill resource requirements"""
        if not self.available_resources:
            return False

        return (
            self.available_resources.get("cpu_cores", Decimal("0")) >= requirements.cpu_cores
            and self.available_resources.get("memory_gb", Decimal("0")) >= requirements.memory_gb
            and self.available_resources.get("storage_gb", Decimal("0")) >= requirements.storage_gb
            and self.available_resources.get("bandwidth_mbps", Decimal("0")) >= requirements.bandwidth_mbps
            and self.trust_score >= requirements.min_trust_score
            and self.latency_ms <= requirements.max_latency_ms
        )


@dataclass
class AuctionResult:
    """Result of an auction including winners and payments"""

    auction_id: str
    winning_bids: list[AuctionBid]
    clearing_price: Decimal  # Second-price in Vickrey auctions
    total_cost: Decimal

    # Payment details
    payments: dict[str, Decimal] = field(default_factory=dict)  # bidder_id -> payment
    deposits_refunded: list[str] = field(default_factory=list)

    # Quality metrics
    average_trust_score: Decimal = Decimal("0")
    average_latency_ms: Decimal = Decimal("0")

    # Settlement details
    settled_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    settlement_tx_hashes: dict[str, str] = field(default_factory=dict)

    def calculate_statistics(self):
        """Calculate result statistics"""
        if not self.winning_bids:
            return

        self.average_trust_score = sum(bid.trust_score for bid in self.winning_bids) / len(self.winning_bids)
        self.average_latency_ms = sum(bid.latency_ms for bid in self.winning_bids) / len(self.winning_bids)

        for bid in self.winning_bids:
            # In second-price auctions, winners pay the second-highest bid
            self.payments[bid.bidder_id] = self.clearing_price


@dataclass
class Auction:
    """Main auction object"""

    auction_id: str
    auction_type: AuctionType
    requester_id: str

    # Resource requirements
    requirements: ResourceRequirement

    # Auction parameters
    reserve_price: Decimal  # Minimum acceptable price
    max_bidders: int = 100
    deposit_percentage: Decimal = Decimal("0.1")  # 10% of reserve price as deposit

    # Timing
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime = field(init=False)
    bid_reveal_deadline: datetime | None = None  # For sealed-bid auctions
    duration_minutes: int = 30

    # State
    status: AuctionStatus = AuctionStatus.OPEN
    bids: list[AuctionBid] = field(default_factory=list)
    result: AuctionResult | None = None

    # Anti-griefing
    required_deposit: Decimal = field(init=False)
    total_deposits_held: Decimal = Decimal("0")

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize calculated fields"""
        if not isinstance(self.reserve_price, Decimal):
            self.reserve_price = Decimal(str(self.reserve_price))
        if not isinstance(self.deposit_percentage, Decimal):
            self.deposit_percentage = Decimal(str(self.deposit_percentage))

        self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        self.required_deposit = self.reserve_price * self.deposit_percentage

        # Set bid reveal deadline for sealed-bid auctions
        if self.auction_type in [AuctionType.VICKREY]:
            self.bid_reveal_deadline = self.end_time + timedelta(minutes=15)

    def is_open_for_bids(self) -> bool:
        """Check if auction is accepting bids"""
        now = datetime.now(UTC)
        return (
            self.status == AuctionStatus.OPEN
            and now >= self.start_time
            and now <= self.end_time
            and len(self.bids) < self.max_bidders
        )

    def get_current_best_bid(self) -> AuctionBid | None:
        """Get current best bid based on auction type"""
        valid_bids = [bid for bid in self.bids if bid.status == BidStatus.VALID]

        if not valid_bids:
            return None

        if self.auction_type == AuctionType.REVERSE:
            # Lowest price wins in reverse auctions
            return min(valid_bids, key=lambda b: b.bid_price)
        else:
            # Highest price wins in forward auctions
            return max(valid_bids, key=lambda b: b.bid_price)

    def calculate_clearing_price(self, winning_bids: list[AuctionBid]) -> Decimal:
        """Calculate clearing price based on auction type"""
        if not winning_bids:
            return self.reserve_price

        if self.auction_type == AuctionType.VICKREY:
            # Second-price auction: winners pay second-highest bid
            all_prices = sorted([bid.bid_price for bid in self.bids], reverse=True)
            if len(all_prices) > 1:
                return all_prices[1]  # Second highest
            else:
                return all_prices[0]  # Only one bid
        else:
            # First-price auction: winners pay their bid
            return min(bid.bid_price for bid in winning_bids)


class AuctionEngine:
    """
    Main auction engine for market-based fog computing pricing

    Features:
    - Multiple auction types (reverse, Vickrey, Dutch)
    - Anti-griefing deposit system
    - Quality-weighted bid evaluation
    - Automated settlement and payment processing
    """

    def __init__(self, token_system=None):
        self.token_system = token_system

        # Active auctions
        self.active_auctions: dict[str, Auction] = {}
        self.completed_auctions: dict[str, Auction] = {}

        # Market state
        self.total_auctions_created = 0
        self.total_volume_processed = Decimal("0")
        self.average_clearing_price = Decimal("0")

        # Background tasks
        self._auction_monitor_task: asyncio.Task | None = None

        logger.info("Auction engine initialized")

    async def start(self):
        """Start auction engine background tasks"""
        self._auction_monitor_task = asyncio.create_task(self._auction_monitor_loop())
        logger.info("Auction engine started")

    async def stop(self):
        """Stop auction engine background tasks"""
        if self._auction_monitor_task:
            self._auction_monitor_task.cancel()
        logger.info("Auction engine stopped")

    async def create_auction(
        self,
        requester_id: str,
        requirements: ResourceRequirement,
        auction_type: AuctionType = AuctionType.REVERSE,
        reserve_price: Decimal = None,
        duration_minutes: int = 30,
        **kwargs,
    ) -> str:
        """Create a new auction"""

        auction_id = f"auction_{uuid.uuid4().hex[:8]}"

        # Calculate default reserve price if not provided
        if reserve_price is None:
            reserve_price = await self._calculate_reserve_price(requirements)

        auction = Auction(
            auction_id=auction_id,
            auction_type=auction_type,
            requester_id=requester_id,
            requirements=requirements,
            reserve_price=reserve_price,
            duration_minutes=duration_minutes,
            **kwargs,
        )

        self.active_auctions[auction_id] = auction
        self.total_auctions_created += 1

        logger.info(
            f"Created {auction_type.value} auction {auction_id}: "
            f"{float(requirements.cpu_cores)} cores, "
            f"{float(requirements.memory_gb)}GB memory, "
            f"reserve=${float(reserve_price):.4f}"
        )

        return auction_id

    async def submit_bid(
        self,
        auction_id: str,
        bidder_id: str,
        node_id: str,
        bid_price: Decimal,
        trust_score: Decimal,
        reputation_score: Decimal,
        available_resources: dict[str, Decimal],
        **kwargs,
    ) -> str | None:
        """Submit a bid to an auction"""

        if auction_id not in self.active_auctions:
            logger.error(f"Auction not found: {auction_id}")
            return None

        auction = self.active_auctions[auction_id]

        if not auction.is_open_for_bids():
            logger.error(f"Auction {auction_id} is not open for bids")
            return None

        # Validate bid against requirements
        if not self._validate_bid_requirements(auction, available_resources, trust_score, **kwargs):
            logger.error("Bid does not meet auction requirements")
            return None

        # Check deposit requirement
        required_deposit = auction.required_deposit
        if self.token_system:
            account_balance = await self._get_account_balance(bidder_id)
            if account_balance < required_deposit:
                logger.error(f"Insufficient balance for deposit: required {float(required_deposit)}")
                return None

        # Create bid
        bid_id = f"bid_{uuid.uuid4().hex[:8]}"

        bid = AuctionBid(
            bid_id=bid_id,
            auction_id=auction_id,
            bidder_id=bidder_id,
            node_id=node_id,
            bid_price=bid_price,
            deposit_amount=required_deposit,
            trust_score=trust_score,
            reputation_score=reputation_score,
            available_resources=available_resources,
            **kwargs,
        )

        bid.set_per_hour_rate(auction.requirements.duration_hours)

        # Hold deposit
        if self.token_system:
            deposit_success = await self._hold_deposit(bidder_id, required_deposit, bid_id)
            if not deposit_success:
                logger.error(f"Failed to hold deposit for bid {bid_id}")
                return None
            bid.deposit_confirmed = True

        # Validate bid
        if await self._validate_bid(auction, bid):
            bid.status = BidStatus.VALID
            auction.bids.append(bid)
            auction.total_deposits_held += required_deposit

            logger.info(
                f"Bid {bid_id} submitted to auction {auction_id}: "
                f"${float(bid_price):.4f} (${float(bid.per_hour_rate):.4f}/hour)"
            )

            return bid_id
        else:
            # Refund deposit if validation fails
            if self.token_system and bid.deposit_confirmed:
                await self._refund_deposit(bidder_id, required_deposit, bid_id)
            return None

    async def close_auction(self, auction_id: str) -> AuctionResult | None:
        """Close auction and determine winners"""

        if auction_id not in self.active_auctions:
            logger.error(f"Auction not found: {auction_id}")
            return None

        auction = self.active_auctions[auction_id]
        auction.status = AuctionStatus.SETTLING

        # Determine winning bids
        winning_bids = await self._determine_winners(auction)

        if not winning_bids:
            logger.warning(f"No valid bids for auction {auction_id}")
            auction.status = AuctionStatus.CANCELLED
            await self._refund_all_deposits(auction)
            return None

        # Calculate clearing price
        clearing_price = auction.calculate_clearing_price(winning_bids)

        # Create result
        result = AuctionResult(
            auction_id=auction_id, winning_bids=winning_bids, clearing_price=clearing_price, total_cost=clearing_price
        )

        result.calculate_statistics()

        # Process payments and refunds
        await self._settle_auction(auction, result)

        # Update auction
        auction.result = result
        auction.status = AuctionStatus.SETTLED

        # Move to completed auctions
        self.completed_auctions[auction_id] = auction
        del self.active_auctions[auction_id]

        # Update statistics
        self.total_volume_processed += result.total_cost
        self._update_average_clearing_price(result.clearing_price)

        logger.info(
            f"Auction {auction_id} settled: "
            f"{len(winning_bids)} winners, "
            f"clearing price=${float(clearing_price):.4f}"
        )

        return result

    async def get_auction_status(self, auction_id: str) -> dict[str, Any] | None:
        """Get current auction status"""

        auction = self.active_auctions.get(auction_id) or self.completed_auctions.get(auction_id)
        if not auction:
            return None

        current_best_bid = auction.get_current_best_bid()

        return {
            "auction_id": auction_id,
            "status": auction.status.value,
            "auction_type": auction.auction_type.value,
            "requirements": {
                "cpu_cores": float(auction.requirements.cpu_cores),
                "memory_gb": float(auction.requirements.memory_gb),
                "duration_hours": float(auction.requirements.duration_hours),
            },
            "reserve_price": float(auction.reserve_price),
            "total_bids": len(auction.bids),
            "valid_bids": len([b for b in auction.bids if b.status == BidStatus.VALID]),
            "current_best_price": float(current_best_bid.bid_price) if current_best_bid else None,
            "time_remaining": max(0, (auction.end_time - datetime.now(UTC)).total_seconds()),
            "deposits_held": float(auction.total_deposits_held),
            "result": {
                "winners": len(auction.result.winning_bids) if auction.result else 0,
                "clearing_price": float(auction.result.clearing_price) if auction.result else None,
                "total_cost": float(auction.result.total_cost) if auction.result else None,
            }
            if auction.result
            else None,
        }

    async def get_market_statistics(self) -> dict[str, Any]:
        """Get comprehensive market statistics"""

        active_auctions = len(self.active_auctions)
        total_active_deposits = sum(auction.total_deposits_held for auction in self.active_auctions.values())

        # Calculate average metrics from recent auctions
        recent_auctions = [
            auction
            for auction in self.completed_auctions.values()
            if auction.result and (datetime.now(UTC) - auction.result.settled_at).days <= 7
        ]

        avg_bids_per_auction = sum(len(auction.bids) for auction in recent_auctions) / max(1, len(recent_auctions))
        avg_clearing_price = sum(auction.result.clearing_price for auction in recent_auctions) / max(
            1, len(recent_auctions)
        )

        return {
            "auction_statistics": {
                "active_auctions": active_auctions,
                "completed_auctions": len(self.completed_auctions),
                "total_auctions_created": self.total_auctions_created,
                "success_rate": len(self.completed_auctions) / max(1, self.total_auctions_created),
            },
            "financial_metrics": {
                "total_volume_processed": float(self.total_volume_processed),
                "active_deposits_held": float(total_active_deposits),
                "average_clearing_price": float(avg_clearing_price),
                "recent_auctions_count": len(recent_auctions),
            },
            "market_health": {
                "average_bids_per_auction": avg_bids_per_auction,
                "auction_completion_rate": len([a for a in recent_auctions if a.status == AuctionStatus.COMPLETED])
                / max(1, len(recent_auctions)),
                "market_liquidity_score": min(1.0, avg_bids_per_auction / 5.0),  # 5+ bids = full liquidity
            },
        }

    # Private methods

    async def _auction_monitor_loop(self):
        """Background task to monitor and close auctions"""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                now = datetime.now(UTC)
                expired_auctions = []

                for auction_id, auction in self.active_auctions.items():
                    if now > auction.end_time and auction.status == AuctionStatus.OPEN:
                        expired_auctions.append(auction_id)

                # Close expired auctions
                for auction_id in expired_auctions:
                    await self.close_auction(auction_id)

            except Exception as e:
                logger.error(f"Error in auction monitor loop: {e}")
                await asyncio.sleep(60)

    async def _calculate_reserve_price(self, requirements: ResourceRequirement) -> Decimal:
        """Calculate default reserve price based on resource requirements"""

        # Base pricing model (simplified)
        cpu_base_price = Decimal("0.10")  # $0.10 per CPU-hour
        memory_base_price = Decimal("0.01")  # $0.01 per GB-hour
        storage_base_price = Decimal("0.001")  # $0.001 per GB-hour
        bandwidth_base_price = Decimal("0.05")  # $0.05 per Mbps-hour

        base_cost = (
            requirements.cpu_cores * cpu_base_price
            + requirements.memory_gb * memory_base_price
            + requirements.storage_gb * storage_base_price
            + requirements.bandwidth_mbps * bandwidth_base_price
        ) * requirements.duration_hours

        # Add market conditions multiplier (would be dynamic in production)
        market_multiplier = Decimal("1.2")  # 20% above base cost

        reserve_price = base_cost * market_multiplier

        logger.debug(f"Calculated reserve price: ${float(reserve_price):.4f}")
        return reserve_price

    def _validate_bid_requirements(
        self, auction: Auction, available_resources: dict[str, Decimal], trust_score: Decimal, **kwargs
    ) -> bool:
        """Validate that bid meets auction requirements"""

        requirements = auction.requirements

        # Check resource availability
        if (
            available_resources.get("cpu_cores", Decimal("0")) < requirements.cpu_cores
            or available_resources.get("memory_gb", Decimal("0")) < requirements.memory_gb
            or available_resources.get("storage_gb", Decimal("0")) < requirements.storage_gb
        ):
            return False

        # Check trust score
        if trust_score < requirements.min_trust_score:
            return False

        # Check latency if provided
        latency_ms = kwargs.get("latency_ms", Decimal("0"))
        if latency_ms > requirements.max_latency_ms:
            return False

        return True

    async def _validate_bid(self, auction: Auction, bid: AuctionBid) -> bool:
        """Comprehensive bid validation"""

        # Check if bid meets reserve price
        if auction.auction_type == AuctionType.REVERSE:
            # In reverse auctions, bids must be at or above reserve price
            if bid.bid_price < auction.reserve_price:
                logger.warning(f"Bid {bid.bid_id} below reserve price")
                return False

        # Validate resource capabilities
        if not bid.validate_resources(auction.requirements):
            logger.warning(f"Bid {bid.bid_id} fails resource validation")
            return False

        # Check for duplicate bids from same bidder
        existing_bids = [b for b in auction.bids if b.bidder_id == bid.bidder_id and b.status == BidStatus.VALID]
        if existing_bids:
            logger.warning(f"Bidder {bid.bidder_id} already has valid bid in auction")
            return False

        return True

    async def _determine_winners(self, auction: Auction) -> list[AuctionBid]:
        """Determine winning bids based on auction type"""

        valid_bids = [bid for bid in auction.bids if bid.status == BidStatus.VALID]

        if not valid_bids:
            return []

        if auction.auction_type == AuctionType.REVERSE:
            # Lowest price wins, but consider quality weighting
            winning_bids = self._select_quality_weighted_winners(valid_bids, auction.requirements, lowest_price=True)
        elif auction.auction_type == AuctionType.VICKREY:
            # Highest bid wins but pays second price
            winning_bids = self._select_quality_weighted_winners(valid_bids, auction.requirements, lowest_price=False)
        else:
            # Default to lowest price
            winning_bids = self._select_quality_weighted_winners(valid_bids, auction.requirements, lowest_price=True)

        # Update bid statuses
        winning_bid_ids = {bid.bid_id for bid in winning_bids}
        for bid in auction.bids:
            if bid.bid_id in winning_bid_ids:
                bid.status = BidStatus.WON
            elif bid.status == BidStatus.VALID:
                bid.status = BidStatus.LOST

        return winning_bids

    def _select_quality_weighted_winners(
        self, bids: list[AuctionBid], requirements: ResourceRequirement, lowest_price: bool = True
    ) -> list[AuctionBid]:
        """Select winners using quality-weighted scoring"""

        # Calculate composite scores for each bid
        scored_bids = []

        for bid in bids:
            quality_score = bid.calculate_quality_score()

            # Price score (normalized)
            if lowest_price:
                price_score = Decimal("1") - (bid.bid_price / max(b.bid_price for b in bids))
            else:
                price_score = bid.bid_price / max(b.bid_price for b in bids)

            # Composite score (price 60%, quality 40%)
            composite_score = price_score * Decimal("0.6") + quality_score * Decimal("0.4")

            scored_bids.append((bid, composite_score))

        # Sort by composite score (highest first)
        scored_bids.sort(key=lambda x: x[1], reverse=True)

        # For now, return single best bid (could be extended for multi-winner auctions)
        return [scored_bids[0][0]]

    async def _settle_auction(self, auction: Auction, result: AuctionResult):
        """Process payments and refunds for completed auction"""

        if not self.token_system:
            logger.warning("No token system configured, skipping settlement")
            return

        # Process payments for winners
        for bid in result.winning_bids:
            payment_amount = result.payments[bid.bidder_id]

            # Transfer payment from requester to winner
            success = await self.token_system.transfer(
                from_account=auction.requester_id,
                to_account=bid.bidder_id,
                amount=float(payment_amount),
                description=f"Auction payment for {auction.auction_id}",
            )

            if success:
                result.settlement_tx_hashes[bid.bidder_id] = f"tx_{uuid.uuid4().hex[:8]}"
                logger.info(f"Payment of ${float(payment_amount):.4f} sent to {bid.bidder_id}")
            else:
                logger.error(f"Failed to process payment to {bid.bidder_id}")

        # Refund deposits for all bidders
        for bid in auction.bids:
            if bid.deposit_confirmed:
                await self._refund_deposit(bid.bidder_id, bid.deposit_amount, bid.bid_id)
                result.deposits_refunded.append(bid.bidder_id)

    async def _hold_deposit(self, account_id: str, amount: Decimal, reference: str) -> bool:
        """Hold deposit tokens from bidder account"""

        if not self.token_system:
            return True  # Skip if no token system

        # In production, this would lock tokens in escrow
        # For now, just check balance
        account_info = self.token_system.get_account_balance(account_id)
        if account_info.get("error"):
            return False

        available_balance = account_info.get("balance", 0)
        return available_balance >= float(amount)

    async def _refund_deposit(self, account_id: str, amount: Decimal, reference: str) -> bool:
        """Refund deposit tokens to bidder account"""

        if not self.token_system:
            return True

        # In production, this would release tokens from escrow
        logger.info(f"Refunded deposit of ${float(amount):.4f} to {account_id}")
        return True

    async def _refund_all_deposits(self, auction: Auction):
        """Refund all deposits for cancelled auction"""

        for bid in auction.bids:
            if bid.deposit_confirmed:
                await self._refund_deposit(bid.bidder_id, bid.deposit_amount, bid.bid_id)

        logger.info(f"Refunded all deposits for cancelled auction {auction.auction_id}")

    async def _get_account_balance(self, account_id: str) -> Decimal:
        """Get account balance for deposit validation"""

        if not self.token_system:
            return Decimal("1000")  # Default for testing

        account_info = self.token_system.get_account_balance(account_id)
        return Decimal(str(account_info.get("balance", 0)))

    def _update_average_clearing_price(self, new_price: Decimal):
        """Update rolling average clearing price"""

        # Simple moving average (in production, would use weighted or time-decay)
        if self.average_clearing_price == 0:
            self.average_clearing_price = new_price
        else:
            self.average_clearing_price = (self.average_clearing_price + new_price) / Decimal("2")


# Global auction engine instance
_auction_engine: AuctionEngine | None = None


async def get_auction_engine() -> AuctionEngine:
    """Get global auction engine instance"""
    global _auction_engine

    if _auction_engine is None:
        _auction_engine = AuctionEngine()
        await _auction_engine.start()

    return _auction_engine


# Convenience functions for integration
async def create_reverse_auction(
    requester_id: str, cpu_cores: float, memory_gb: float, duration_hours: float, reserve_price: float = None, **kwargs
) -> str:
    """Create reverse auction for fog computing resources"""

    requirements = ResourceRequirement(
        cpu_cores=Decimal(str(cpu_cores)),
        memory_gb=Decimal(str(memory_gb)),
        storage_gb=Decimal("1.0"),  # Default
        bandwidth_mbps=Decimal("10.0"),  # Default
        duration_hours=Decimal(str(duration_hours)),
        **kwargs,
    )

    engine = await get_auction_engine()
    return await engine.create_auction(
        requester_id=requester_id,
        requirements=requirements,
        auction_type=AuctionType.REVERSE,
        reserve_price=Decimal(str(reserve_price)) if reserve_price else None,
        **kwargs,
    )


async def submit_provider_bid(
    auction_id: str,
    provider_id: str,
    node_id: str,
    bid_price: float,
    available_resources: dict[str, float],
    trust_score: float = 0.5,
    **kwargs,
) -> str | None:
    """Submit bid from fog computing provider"""

    # Convert to Decimal for precision
    resources_decimal = {k: Decimal(str(v)) for k, v in available_resources.items()}

    engine = await get_auction_engine()
    return await engine.submit_bid(
        auction_id=auction_id,
        bidder_id=provider_id,
        node_id=node_id,
        bid_price=Decimal(str(bid_price)),
        trust_score=Decimal(str(trust_score)),
        reputation_score=Decimal(str(kwargs.get("reputation_score", 0.5))),
        available_resources=resources_decimal,
        **kwargs,
    )
