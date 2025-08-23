"""
Integration tests for fog marketplace matching algorithms

Tests the marketplace bid matching engine with various scenarios:
- Trust-based matching with score optimization
- Price constraint enforcement (max_price × trust)
- Resource requirement satisfaction
- Bid expiration and cleanup
- Dynamic pricing under different market conditions
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from packages.fog.gateway.scheduler.marketplace import BidStatus, BidType, MarketplaceEngine, PricingTier


class TestMarketplaceMatching:
    """Test marketplace bid matching algorithms"""

    @pytest.fixture
    async def marketplace(self):
        """Fresh marketplace instance for each test"""
        engine = MarketplaceEngine()
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.fixture
    def sample_listings(self, marketplace):
        """Create sample resource listings"""

        async def _create_listings():
            listings = []

            # High-trust, premium server
            listings.append(
                await marketplace.add_resource_listing(
                    node_id="premium-server-001",
                    cpu_cores=8.0,
                    memory_gb=32.0,
                    disk_gb=500.0,
                    spot_price=0.05,
                    on_demand_price=0.08,
                    trust_score=0.95,
                    pricing_tier=PricingTier.PREMIUM,
                    latency_ms=50.0,
                    min_trust_required=0.8,
                )
            )

            # Medium-trust, standard server
            listings.append(
                await marketplace.add_resource_listing(
                    node_id="standard-server-002",
                    cpu_cores=4.0,
                    memory_gb=16.0,
                    disk_gb=200.0,
                    spot_price=0.08,
                    on_demand_price=0.12,
                    trust_score=0.7,
                    pricing_tier=PricingTier.STANDARD,
                    latency_ms=100.0,
                    min_trust_required=0.5,
                )
            )

            # Low-trust, basic server
            listings.append(
                await marketplace.add_resource_listing(
                    node_id="basic-server-003",
                    cpu_cores=2.0,
                    memory_gb=8.0,
                    disk_gb=100.0,
                    spot_price=0.12,
                    on_demand_price=0.18,
                    trust_score=0.4,
                    pricing_tier=PricingTier.BASIC,
                    latency_ms=200.0,
                    min_trust_required=0.2,
                )
            )

            # Mobile device (expensive, limited resources)
            listings.append(
                await marketplace.add_resource_listing(
                    node_id="mobile-device-004",
                    cpu_cores=1.0,
                    memory_gb=4.0,
                    disk_gb=50.0,
                    spot_price=0.25,
                    on_demand_price=0.35,
                    trust_score=0.6,
                    pricing_tier=PricingTier.BASIC,
                    latency_ms=150.0,
                    min_trust_required=0.3,
                    max_duration_hours=2,  # Short jobs only
                )
            )

            return listings

        return await _create_listings()

    @pytest.mark.asyncio
    async def test_trust_based_matching_priority(self, marketplace, sample_listings):
        """Test that higher trust nodes are preferred when price allows"""

        # Submit bid that can afford premium server but trusts all levels
        bid_id = await marketplace.submit_bid(
            namespace="test-org/trust-test",
            cpu_cores=2.0,
            memory_gb=8.0,
            max_price=2.0,  # High budget
            bid_type=BidType.SPOT,
            min_trust_score=0.3,  # Accept any trust level
            estimated_duration_hours=1.0,
        )

        # Allow matching to occur
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find the trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Should match with highest trust server that meets requirements
        # Given the pricing, it should prefer premium-server-001 (trust=0.95)
        assert trade.seller_node_id == "premium-server-001"

    @pytest.mark.asyncio
    async def test_price_constraint_enforcement(self, marketplace, sample_listings):
        """Test max_price constraint is strictly enforced"""

        # Submit bid with low budget that can't afford premium servers
        bid_id = await marketplace.submit_bid(
            namespace="test-org/budget-test",
            cpu_cores=1.0,
            memory_gb=2.0,
            max_price=0.15,  # Low budget
            bid_type=BidType.SPOT,
            min_trust_score=0.2,
            estimated_duration_hours=1.0,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Should match with basic server only (others too expensive)
        assert trade.seller_node_id == "basic-server-003"
        assert trade.agreed_price <= 0.15

    @pytest.mark.asyncio
    async def test_resource_requirement_matching(self, marketplace, sample_listings):
        """Test that resource requirements are satisfied"""

        # Submit bid requiring more resources than mobile device can provide
        bid_id = await marketplace.submit_bid(
            namespace="test-org/resource-test",
            cpu_cores=3.0,  # More than mobile device (1.0)
            memory_gb=12.0,  # More than mobile device (4.0)
            max_price=1.0,
            bid_type=BidType.SPOT,
            min_trust_score=0.3,
            estimated_duration_hours=1.0,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Should NOT match with mobile device due to insufficient resources
        assert trade.seller_node_id != "mobile-device-004"

        # Should match with server that has sufficient resources
        matched_listing = marketplace.active_listings.get(trade.listing_id)
        if matched_listing is None:
            # Listing was removed after matching, check the server node
            assert trade.seller_node_id in ["premium-server-001", "standard-server-002"]

    @pytest.mark.asyncio
    async def test_duration_constraint_matching(self, marketplace, sample_listings):
        """Test job duration constraints are respected"""

        # Submit long-running job (mobile device max is 2 hours)
        bid_id = await marketplace.submit_bid(
            namespace="test-org/duration-test",
            cpu_cores=0.5,  # Small enough for mobile
            memory_gb=2.0,  # Small enough for mobile
            max_price=1.0,  # High enough for mobile
            bid_type=BidType.SPOT,
            min_trust_score=0.3,
            estimated_duration_hours=5.0,  # Longer than mobile max (2 hours)
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Should NOT match with mobile device due to duration constraint
        assert trade.seller_node_id != "mobile-device-004"

    @pytest.mark.asyncio
    async def test_trust_requirement_filtering(self, marketplace, sample_listings):
        """Test minimum trust requirements filter out low-trust nodes"""

        # Submit bid requiring high trust
        bid_id = await marketplace.submit_bid(
            namespace="test-org/high-trust-test",
            cpu_cores=1.0,
            memory_gb=4.0,
            max_price=1.0,
            bid_type=BidType.SPOT,
            min_trust_score=0.8,  # High trust requirement
            estimated_duration_hours=1.0,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Should only match with premium server (trust=0.95)
        # Others have trust < 0.8
        assert trade.seller_node_id == "premium-server-001"

    @pytest.mark.asyncio
    async def test_bid_expiration_cleanup(self, marketplace, sample_listings):
        """Test expired bids are cleaned up"""

        # Submit bid
        bid_id = await marketplace.submit_bid(
            namespace="test-org/expiry-test",
            cpu_cores=1.0,
            memory_gb=2.0,
            max_price=0.05,  # Too low to match
            bid_type=BidType.SPOT,
            min_trust_score=0.3,
            estimated_duration_hours=1.0,
        )

        assert bid_id in marketplace.pending_bids

        # Manually expire the bid by setting old timestamp
        bid = marketplace.pending_bids[bid_id]
        bid.created_at = datetime.now(UTC) - timedelta(minutes=15)  # 15 minutes ago

        # Run matching loop which should clean up expired bids
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        # Bid should be removed due to expiration
        assert bid_id not in marketplace.pending_bids
        assert matched_count == 0  # No match due to removal

    @pytest.mark.asyncio
    async def test_listing_availability_filtering(self, marketplace, sample_listings):
        """Test unavailable listings are filtered out"""

        # Make one listing unavailable
        premium_listing_id = sample_listings[0]
        listing = marketplace.active_listings[premium_listing_id]
        listing.available_until = datetime.now(UTC) - timedelta(minutes=5)  # Expired

        # Submit bid that would prefer the premium server
        bid_id = await marketplace.submit_bid(
            namespace="test-org/availability-test",
            cpu_cores=2.0,
            memory_gb=8.0,
            max_price=2.0,
            bid_type=BidType.SPOT,
            min_trust_score=0.3,
            estimated_duration_hours=1.0,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Premium listing should be removed
        assert premium_listing_id not in marketplace.active_listings

        # Trade should match with available server
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None
        assert trade.seller_node_id != "premium-server-001"  # Should not match expired listing

    @pytest.mark.asyncio
    async def test_composite_scoring_algorithm(self, marketplace, sample_listings):
        """Test the composite scoring algorithm balances price, trust, latency"""

        # Submit bid where different listings optimize different factors
        bid_id = await marketplace.submit_bid(
            namespace="test-org/scoring-test",
            cpu_cores=1.5,
            memory_gb=6.0,
            max_price=0.30,  # Can afford basic and standard servers
            bid_type=BidType.SPOT,
            min_trust_score=0.3,
            max_latency_ms=250.0,  # Accept all latencies
            estimated_duration_hours=1.0,
        )

        # Calculate expected scores for each valid listing
        bid = marketplace.pending_bids[bid_id]

        scores = {}
        for listing_id, listing in marketplace.active_listings.items():
            if listing.matches_requirements(
                bid.cpu_cores, bid.memory_gb, bid.estimated_duration_hours, bid.min_trust_score
            ):
                score = bid.calculate_score(listing)
                scores[listing.node_id] = score

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 1

        # Find trade
        trade = None
        for t in marketplace.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None

        # Verify it matched with the highest scoring available option
        matched_node = trade.seller_node_id
        matched_score = scores.get(matched_node, 0.0)

        # Should be one of the top scores
        all_scores = list(scores.values())
        all_scores.sort(reverse=True)

        assert matched_score >= all_scores[0] * 0.9  # Within 10% of best score

    @pytest.mark.asyncio
    async def test_no_match_scenario(self, marketplace, sample_listings):
        """Test handling when no listings can satisfy bid requirements"""

        # Submit impossible bid
        bid_id = await marketplace.submit_bid(
            namespace="test-org/impossible-test",
            cpu_cores=100.0,  # More than any listing
            memory_gb=500.0,  # More than any listing
            max_price=0.01,  # Less than any listing costs
            bid_type=BidType.SPOT,
            min_trust_score=0.99,  # Higher than any listing
            estimated_duration_hours=1.0,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        assert matched_count == 0

        # Bid should remain pending (until expiration)
        assert bid_id in marketplace.pending_bids
        assert marketplace.pending_bids[bid_id].status == BidStatus.PENDING

        # No trades should be created
        assert len(marketplace.active_trades) == 0

    @pytest.mark.asyncio
    async def test_multiple_bids_single_listing(self, marketplace):
        """Test multiple bids competing for single listing"""

        # Add single small listing
        listing_id = await marketplace.add_resource_listing(
            node_id="small-server",
            cpu_cores=2.0,
            memory_gb=4.0,
            disk_gb=50.0,
            spot_price=0.10,
            on_demand_price=0.15,
            trust_score=0.6,
        )

        # Submit multiple competing bids
        bid_ids = []
        for i in range(3):
            bid_id = await marketplace.submit_bid(
                namespace=f"test-org/competitor-{i}",
                cpu_cores=1.0,
                memory_gb=2.0,
                max_price=0.20 + (i * 0.05),  # Different budgets
                bid_type=BidType.SPOT,
                min_trust_score=0.3,
                estimated_duration_hours=1.0,
            )
            bid_ids.append(bid_id)

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace._match_bids_to_listings()

        # Only one bid should match (highest scoring)
        assert matched_count == 1

        # Listing should be consumed
        assert listing_id not in marketplace.active_listings

        # Two bids should remain pending
        remaining_bids = len([bid_id for bid_id in bid_ids if bid_id in marketplace.pending_bids])
        assert remaining_bids == 2

        # One trade should be created
        assert len(marketplace.active_trades) == 1


class TestMarketplaceDynamicPricing:
    """Test dynamic pricing under different market conditions"""

    @pytest.fixture
    async def marketplace(self):
        """Fresh marketplace for pricing tests"""
        engine = MarketplaceEngine()
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_demand_supply_pricing_adjustment(self, marketplace):
        """Test pricing adjusts based on demand/supply ratio"""

        # Record initial pricing
        initial_spot = marketplace.pricing_engine.get_current_spot_price()
        initial_on_demand = marketplace.pricing_engine.get_current_on_demand_price()

        # Create high demand scenario
        marketplace.pricing_engine.update_market_conditions(
            total_demand=10.0, total_supply=2.0, utilization_rate=0.9  # High demand  # Low supply  # High utilization
        )

        high_demand_spot = marketplace.pricing_engine.get_current_spot_price()
        high_demand_on_demand = marketplace.pricing_engine.get_current_on_demand_price()

        # Prices should increase under high demand
        assert high_demand_spot > initial_spot
        assert high_demand_on_demand > initial_on_demand

        # Create low demand scenario
        marketplace.pricing_engine.update_market_conditions(
            total_demand=1.0, total_supply=10.0, utilization_rate=0.1  # Low demand  # High supply  # Low utilization
        )

        low_demand_spot = marketplace.pricing_engine.get_current_spot_price()
        low_demand_on_demand = marketplace.pricing_engine.get_current_on_demand_price()

        # Prices should decrease under low demand
        assert low_demand_spot < high_demand_spot
        assert low_demand_on_demand < high_demand_on_demand

    @pytest.mark.asyncio
    async def test_price_volatility_calculation(self, marketplace):
        """Test price volatility is calculated correctly"""

        # Initial volatility should be 0 (no history)
        initial_volatility = marketplace.pricing_engine.get_price_volatility()
        assert initial_volatility == 0.0

        # Create price changes over time
        price_points = [
            (5.0, 5.0, 0.5),  # Normal conditions
            (10.0, 2.0, 0.8),  # High demand
            (8.0, 3.0, 0.7),  # Medium demand
            (15.0, 1.0, 0.9),  # Very high demand
            (3.0, 8.0, 0.2),  # Low demand
        ]

        for demand, supply, utilization in price_points:
            marketplace.pricing_engine.update_market_conditions(demand, supply, utilization)
            await asyncio.sleep(0.01)  # Small delay between updates

        # Should now have non-zero volatility
        final_volatility = marketplace.pricing_engine.get_price_volatility()
        assert final_volatility > 0.0

        # Volatility should be reasonable (not extreme)
        assert final_volatility < 2.0  # Less than 200% volatility

    @pytest.mark.asyncio
    async def test_pricing_affects_matching(self, marketplace):
        """Test that dynamic pricing affects bid matching outcomes"""

        # Add listing with moderate pricing
        await marketplace.add_resource_listing(
            node_id="dynamic-pricing-server",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=100.0,
            spot_price=0.10,
            on_demand_price=0.15,
            trust_score=0.6,
        )

        # Submit bid with moderate budget
        await marketplace.submit_bid(
            namespace="test-org/pricing-test-1",
            cpu_cores=2.0,
            memory_gb=4.0,
            max_price=0.25,  # Moderate budget
            bid_type=BidType.SPOT,
            estimated_duration_hours=1.0,
        )

        # Set low demand conditions (should result in lower prices)
        marketplace.pricing_engine.update_market_conditions(total_demand=1.0, total_supply=10.0, utilization_rate=0.1)

        # Allow matching under low prices
        await asyncio.sleep(0.1)
        matched_count_1 = await marketplace._match_bids_to_listings()

        # Should match successfully with low pricing
        assert matched_count_1 == 1

        # Create another identical scenario but with high demand
        await marketplace.add_resource_listing(
            node_id="dynamic-pricing-server-2",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=100.0,
            spot_price=0.10,
            on_demand_price=0.15,
            trust_score=0.6,
        )

        bid_id_2 = await marketplace.submit_bid(
            namespace="test-org/pricing-test-2",
            cpu_cores=2.0,
            memory_gb=4.0,
            max_price=0.25,  # Same budget
            bid_type=BidType.SPOT,
            estimated_duration_hours=1.0,
        )

        # Set high demand conditions (should result in higher prices)
        marketplace.pricing_engine.update_market_conditions(total_demand=20.0, total_supply=2.0, utilization_rate=0.95)

        # Under high demand, the same budget might not be sufficient
        await asyncio.sleep(0.1)
        matched_count_2 = await marketplace._match_bids_to_listings()

        # Verify that market conditions affected the outcome
        trade_1_cost = None
        trade_2_cost = None

        for trade in marketplace.active_trades.values():
            if trade.buyer_namespace == "test-org/pricing-test-1":
                trade_1_cost = trade.agreed_price
            elif trade.buyer_namespace == "test-org/pricing-test-2":
                trade_2_cost = trade.agreed_price

        # If both matched, trade 2 should be more expensive due to market conditions
        if trade_1_cost and trade_2_cost:
            assert trade_2_cost > trade_1_cost
        elif trade_1_cost and not trade_2_cost:
            # High demand prevented second match - this is also valid
            assert bid_id_2 in marketplace.pending_bids or matched_count_2 == 0
