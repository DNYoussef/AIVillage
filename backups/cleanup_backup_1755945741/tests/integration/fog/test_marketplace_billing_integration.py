"""
Integration tests for fog marketplace and billing system

Tests the complete flow from marketplace bidding to billing tracking:
- Price quote generation and marketplace matching
- Job execution cost calculation
- Usage tracking and invoice generation
- Edge device pricing integration
- Multi-tier pricing validation
"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from packages.fog.edge.beacon import CapabilityBeacon, DeviceType, PowerProfile
from packages.fog.gateway.api.billing import PriceQuoteRequest, get_billing_engine

# Import fog marketplace and billing components
from packages.fog.gateway.scheduler.marketplace import BidType, PricingTier, get_marketplace_engine


class TestMarketplaceBillingIntegration:
    """Integration tests for marketplace and billing systems"""

    @pytest.fixture
    async def marketplace_engine(self):
        """Get marketplace engine instance"""
        engine = await get_marketplace_engine()
        # Clean slate for testing
        engine.active_listings.clear()
        engine.pending_bids.clear()
        engine.active_trades.clear()
        yield engine
        await engine.stop()

    @pytest.fixture
    async def billing_engine(self):
        """Get billing engine instance"""
        engine = await get_billing_engine()
        # Clean slate for testing
        engine.namespace_trackers.clear()
        engine.invoices.clear()
        yield engine

    @pytest.fixture
    def test_namespace(self):
        """Test namespace for isolation"""
        return "test-org/marketplace-billing"

    @pytest.mark.asyncio
    async def test_price_quote_to_billing_flow(self, marketplace_engine, billing_engine, test_namespace):
        """Test complete flow: price quote → bid → execution → billing"""

        # Step 1: Add resource listings to marketplace
        await marketplace_engine.add_resource_listing(
            node_id="edge-phone-001",
            cpu_cores=2.0,
            memory_gb=4.0,
            disk_gb=8.0,
            spot_price=0.12,
            on_demand_price=0.18,
            trust_score=0.8,
            pricing_tier=PricingTier.STANDARD,
        )

        await marketplace_engine.add_resource_listing(
            node_id="edge-laptop-002",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=16.0,
            spot_price=0.08,
            on_demand_price=0.15,
            trust_score=0.6,
            pricing_tier=PricingTier.BASIC,
        )

        assert len(marketplace_engine.active_listings) == 2

        # Step 2: Get price quote
        quote = await marketplace_engine.get_price_quote(
            cpu_cores=1.5,
            memory_gb=2.0,
            duration_hours=2.0,
            bid_type=BidType.SPOT,
            pricing_tier=PricingTier.BASIC,
            min_trust_score=0.5,
        )

        assert quote["available"] is True
        assert "quote" in quote
        assert quote["quote"]["min_price"] > 0
        assert quote["quote"]["avg_price"] > 0

        avg_price = quote["quote"]["avg_price"]

        # Step 3: Submit bid with quoted price + buffer
        bid_id = await marketplace_engine.submit_bid(
            namespace=test_namespace,
            cpu_cores=1.5,
            memory_gb=2.0,
            max_price=avg_price * 1.1,  # 10% buffer
            bid_type=BidType.SPOT,
            estimated_duration_hours=2.0,
            job_spec={"image": "test-job:latest", "args": ["python", "test.py"]},
        )

        assert bid_id in marketplace_engine.pending_bids

        # Step 4: Let marketplace matching run
        await asyncio.sleep(0.1)  # Allow matching to occur
        matched_count = await marketplace_engine._match_bids_to_listings()

        assert matched_count >= 1
        assert bid_id not in marketplace_engine.pending_bids  # Should be matched and removed

        # Step 5: Find the trade that was created
        trade = None
        for t in marketplace_engine.active_trades.values():
            if t.bid_id == bid_id:
                trade = t
                break

        assert trade is not None
        assert trade.buyer_namespace == test_namespace
        assert trade.cpu_cores == 1.5
        assert trade.memory_gb == 2.0
        assert trade.agreed_price > 0

        # Step 6: Record job usage in billing system
        job_cost = await billing_engine.record_job_usage(
            namespace=test_namespace,
            job_id="job-12345",
            cpu_cores=trade.cpu_cores,
            memory_gb=trade.memory_gb,
            disk_gb=4.0,
            duration_seconds=7200,  # 2 hours
            pricing_tier=PricingTier.BASIC,
            marketplace_price=trade.agreed_price,
        )

        assert job_cost > 0
        assert abs(job_cost - trade.agreed_price) < 0.01  # Should match marketplace price

        # Step 7: Verify usage tracking
        tracker = await billing_engine.get_or_create_tracker(test_namespace)
        assert tracker.total_jobs == 1
        assert tracker.completed_jobs == 1
        assert tracker.cpu_core_seconds == 1.5 * 7200  # 1.5 cores × 2 hours
        assert tracker.memory_gb_seconds == 2.0 * 7200  # 2GB × 2 hours
        assert abs(tracker.total_cost - job_cost) < 0.01

        # Step 8: Generate usage report
        usage_report = await billing_engine.get_usage_report(
            test_namespace,
            start_time=datetime.now(UTC) - timedelta(hours=1),
            end_time=datetime.now(UTC) + timedelta(hours=1),
        )

        assert usage_report.namespace == test_namespace
        assert usage_report.usage_metrics.job_count == 1
        assert usage_report.cost_breakdown.total_cost == job_cost

        # Step 9: Generate invoice
        invoice = await billing_engine.generate_invoice(
            test_namespace,
            period_start=datetime.now(UTC) - timedelta(hours=1),
            period_end=datetime.now(UTC) + timedelta(hours=1),
        )

        assert invoice.namespace == test_namespace
        assert invoice.total_amount == job_cost
        assert len(invoice.line_items) > 0
        assert any("CPU usage" in item.description for item in invoice.line_items)

    @pytest.mark.asyncio
    async def test_edge_device_pricing_integration(self, marketplace_engine, billing_engine, test_namespace):
        """Test integration with edge device pricing from capability beacon"""

        # Create capability beacon with marketplace pricing
        beacon = CapabilityBeacon(
            device_name="test-mobile-device",
            operator_namespace=test_namespace,
            device_type=DeviceType.MOBILE_PHONE,
            betanet_endpoint="htx://mobile-device:8080",
        )

        # Set up device state for pricing calculation
        beacon.capability.cpu_cores = 4.0
        beacon.capability.memory_mb = 6144  # 6GB
        beacon.capability.battery_percent = 85.0
        beacon.capability.power_profile = PowerProfile.BALANCED
        beacon.capability.thermal_state = "normal"
        beacon.capability.cpu_usage_percent = 30.0
        beacon.capability.memory_usage_percent = 45.0
        beacon.capability.trust_score = 0.75
        beacon.capability.device_type = DeviceType.MOBILE_PHONE

        # Update pricing based on device conditions
        await beacon._update_marketplace_pricing()

        # Get marketplace listing from beacon
        listing_data = beacon.get_marketplace_listing()

        assert listing_data["cpu_cores"] > 0
        assert listing_data["memory_gb"] > 0
        assert listing_data["spot_price_per_cpu_hour"] > 0
        assert listing_data["on_demand_price_per_cpu_hour"] > 0
        assert listing_data["pricing_tier"] in ["basic", "standard", "premium"]

        # Add listing to marketplace
        await marketplace_engine.add_resource_listing(
            node_id=listing_data["node_id"],
            cpu_cores=listing_data["cpu_cores"],
            memory_gb=listing_data["memory_gb"],
            disk_gb=listing_data["disk_gb"],
            spot_price=listing_data["spot_price_per_cpu_hour"],
            on_demand_price=listing_data["on_demand_price_per_cpu_hour"],
            trust_score=listing_data["trust_score"],
            pricing_tier=PricingTier.BASIC,  # Mobile device typically basic tier
        )

        # Test cost estimation from beacon
        estimated_cost = beacon.estimate_job_cost(
            cpu_cores=1.0, memory_gb=1.0, disk_gb=2.0, duration_hours=1.0, bid_type="spot"
        )

        assert estimated_cost > 0

        # Submit bid that should match the device listing
        await marketplace_engine.submit_bid(
            namespace=test_namespace,
            cpu_cores=1.0,
            memory_gb=1.0,
            max_price=estimated_cost * 1.2,  # 20% buffer
            bid_type=BidType.SPOT,
        )

        # Allow matching
        await asyncio.sleep(0.1)
        matched_count = await marketplace_engine._match_bids_to_listings()

        assert matched_count >= 1

        # Verify trade occurred with mobile device
        mobile_trade = None
        for trade in marketplace_engine.active_trades.values():
            if trade.seller_node_id == listing_data["node_id"]:
                mobile_trade = trade
                break

        assert mobile_trade is not None
        assert mobile_trade.buyer_namespace == test_namespace

        # Record billing for mobile device job
        mobile_job_cost = await billing_engine.record_job_usage(
            namespace=test_namespace,
            job_id="mobile-job-001",
            cpu_cores=mobile_trade.cpu_cores,
            memory_gb=mobile_trade.memory_gb,
            disk_gb=2.0,
            duration_seconds=3600,  # 1 hour
            marketplace_price=mobile_trade.agreed_price,
        )

        assert mobile_job_cost > 0

        # Verify mobile device premium is reflected in pricing
        # Mobile devices should have higher pricing than servers
        assert mobile_trade.agreed_price > estimated_cost * 0.8  # Should be reasonably close to estimate

    @pytest.mark.asyncio
    async def test_multi_tier_pricing_billing(self, marketplace_engine, billing_engine, test_namespace):
        """Test billing integration with different pricing tiers"""

        # Add listings for each pricing tier
        await marketplace_engine.add_resource_listing(
            node_id="basic-server",
            cpu_cores=8.0,
            memory_gb=16.0,
            disk_gb=100.0,
            spot_price=0.08,
            on_demand_price=0.12,
            trust_score=0.5,
            pricing_tier=PricingTier.BASIC,
        )

        await marketplace_engine.add_resource_listing(
            node_id="standard-server",
            cpu_cores=8.0,
            memory_gb=16.0,
            disk_gb=100.0,
            spot_price=0.12,
            on_demand_price=0.18,
            trust_score=0.7,
            pricing_tier=PricingTier.STANDARD,
        )

        await marketplace_engine.add_resource_listing(
            node_id="premium-server",
            cpu_cores=8.0,
            memory_gb=16.0,
            disk_gb=100.0,
            spot_price=0.16,
            on_demand_price=0.24,
            trust_score=0.9,
            pricing_tier=PricingTier.PREMIUM,
        )

        # Test quote for each tier
        tiers = [
            (PricingTier.BASIC, "basic-server"),
            (PricingTier.STANDARD, "standard-server"),
            (PricingTier.PREMIUM, "premium-server"),
        ]

        tier_costs = {}

        for pricing_tier, expected_node in tiers:
            # Get quote for this tier
            quote = await marketplace_engine.get_price_quote(
                cpu_cores=2.0, memory_gb=4.0, duration_hours=1.0, bid_type=BidType.ON_DEMAND, pricing_tier=pricing_tier
            )

            assert quote["available"] is True

            # Submit bid
            await marketplace_engine.submit_bid(
                namespace=f"{test_namespace}-{pricing_tier.value}",
                cpu_cores=2.0,
                memory_gb=4.0,
                max_price=quote["quote"]["max_price"],
                bid_type=BidType.ON_DEMAND,
                pricing_tier=pricing_tier,
            )

            # Match bid
            matched_count = await marketplace_engine._match_bids_to_listings()
            assert matched_count >= 1

            # Find the trade
            trade = None
            for t in marketplace_engine.active_trades.values():
                if t.buyer_namespace == f"{test_namespace}-{pricing_tier.value}":
                    trade = t
                    break

            assert trade is not None
            assert trade.pricing_tier == pricing_tier

            # Record usage with tier-specific billing
            job_cost = await billing_engine.record_job_usage(
                namespace=f"{test_namespace}-{pricing_tier.value}",
                job_id=f"job-{pricing_tier.value}",
                cpu_cores=2.0,
                memory_gb=4.0,
                disk_gb=8.0,
                duration_seconds=3600,
                pricing_tier=pricing_tier,
                marketplace_price=trade.agreed_price,
            )

            tier_costs[pricing_tier] = job_cost

        # Verify pricing tier hierarchy: basic < standard < premium
        assert tier_costs[PricingTier.BASIC] < tier_costs[PricingTier.STANDARD]
        assert tier_costs[PricingTier.STANDARD] < tier_costs[PricingTier.PREMIUM]

        # Verify tier breakdown in usage tracking
        for pricing_tier in [PricingTier.BASIC, PricingTier.STANDARD, PricingTier.PREMIUM]:
            tracker = await billing_engine.get_or_create_tracker(f"{test_namespace}-{pricing_tier.value}")

            usage_report = tracker.get_usage_report(
                start_time=datetime.now(UTC) - timedelta(hours=1), end_time=datetime.now(UTC) + timedelta(hours=1)
            )

            # Check that cost is attributed to correct tier
            if pricing_tier == PricingTier.BASIC:
                assert usage_report.pricing_tier_breakdown["basic"] > 0
                assert usage_report.pricing_tier_breakdown["standard"] == 0
                assert usage_report.pricing_tier_breakdown["premium"] == 0
            elif pricing_tier == PricingTier.STANDARD:
                assert usage_report.pricing_tier_breakdown["basic"] == 0
                assert usage_report.pricing_tier_breakdown["standard"] > 0
                assert usage_report.pricing_tier_breakdown["premium"] == 0
            else:  # PREMIUM
                assert usage_report.pricing_tier_breakdown["basic"] == 0
                assert usage_report.pricing_tier_breakdown["standard"] == 0
                assert usage_report.pricing_tier_breakdown["premium"] > 0

    @pytest.mark.asyncio
    async def test_marketplace_price_volatility_billing(self, marketplace_engine, billing_engine, test_namespace):
        """Test billing with price volatility and market conditions"""

        # Set up market conditions with high demand
        await marketplace_engine.add_resource_listing(
            node_id="volatile-server",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=50.0,
            spot_price=0.10,
            on_demand_price=0.15,
            trust_score=0.6,
        )

        # Create high demand scenario with multiple pending bids
        bid_ids = []
        for i in range(5):
            bid_id = await marketplace_engine.submit_bid(
                namespace=f"{test_namespace}-demand-{i}",
                cpu_cores=1.0,
                memory_gb=2.0,
                max_price=0.20,
                bid_type=BidType.SPOT,
            )
            bid_ids.append(bid_id)

        # Update market conditions manually
        marketplace_engine.pricing_engine.update_market_conditions(
            total_demand=5.0, total_supply=4.0, utilization_rate=0.8  # High demand  # Lower supply
        )

        # Check price volatility
        marketplace_engine.pricing_engine.get_price_volatility()
        current_spot = marketplace_engine.pricing_engine.get_current_spot_price()

        # Should be higher than base price due to demand
        assert current_spot > 0.10

        # Get quote under volatile conditions
        quote = await marketplace_engine.get_price_quote(
            cpu_cores=1.0, memory_gb=2.0, duration_hours=1.0, bid_type=BidType.SPOT
        )

        assert quote["available"] is True
        assert quote["market_conditions"]["price_volatility"] >= 0

        # Let some bids match
        matched_count = await marketplace_engine._match_bids_to_listings()
        assert matched_count >= 1  # At least one should match

        # Find matched trade
        volatile_trade = None
        for trade in marketplace_engine.active_trades.values():
            if trade.seller_node_id == "volatile-server":
                volatile_trade = trade
                break

        assert volatile_trade is not None

        # Record billing for volatile market job
        volatile_job_cost = await billing_engine.record_job_usage(
            namespace=volatile_trade.buyer_namespace,
            job_id="volatile-market-job",
            cpu_cores=volatile_trade.cpu_cores,
            memory_gb=volatile_trade.memory_gb,
            disk_gb=4.0,
            duration_seconds=3600,
            marketplace_price=volatile_trade.agreed_price,
        )

        # Cost should reflect market conditions
        assert volatile_job_cost > 0
        assert volatile_job_cost >= 0.10  # Should be at least base rate

        # Generate invoice that includes volatile pricing
        invoice = await billing_engine.generate_invoice(
            volatile_trade.buyer_namespace,
            period_start=datetime.now(UTC) - timedelta(hours=1),
            period_end=datetime.now(UTC) + timedelta(hours=1),
        )

        assert invoice.total_amount == volatile_job_cost
        assert len(invoice.line_items) > 0


class TestMarketplaceBillingAPI:
    """Test API integration between marketplace and billing"""

    @pytest.fixture
    def sample_price_quote_request(self):
        """Sample price quote request"""
        return PriceQuoteRequest(
            cpu_cores=2.0,
            memory_gb=4.0,
            disk_gb=10.0,
            estimated_duration_hours=3.0,
            bid_type=BidType.SPOT,
            pricing_tier=PricingTier.BASIC,
            min_trust_score=0.4,
            max_latency_ms=200.0,
        )

    @pytest.mark.asyncio
    async def test_price_quote_api_integration(self, sample_price_quote_request):
        """Test price quote API returns valid marketplace data"""

        # This would typically be called through FastAPI, but we'll test directly
        marketplace = await get_marketplace_engine()

        # Add a listing for the quote to find
        await marketplace.add_resource_listing(
            node_id="api-test-server",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=20.0,
            spot_price=0.08,
            on_demand_price=0.12,
            trust_score=0.6,
        )

        # Test the quote generation
        quote_data = await marketplace.get_price_quote(
            cpu_cores=sample_price_quote_request.cpu_cores,
            memory_gb=sample_price_quote_request.memory_gb,
            duration_hours=sample_price_quote_request.estimated_duration_hours,
            bid_type=sample_price_quote_request.bid_type,
            pricing_tier=sample_price_quote_request.pricing_tier,
            min_trust_score=sample_price_quote_request.min_trust_score,
        )

        assert quote_data["available"] is True
        assert "quote" in quote_data
        assert "market_conditions" in quote_data
        assert "recommendations" in quote_data

        assert quote_data["quote"]["min_price"] > 0
        assert quote_data["quote"]["avg_price"] > 0
        assert quote_data["quote"]["max_price"] > 0
        assert quote_data["market_conditions"]["available_providers"] > 0

        await marketplace.stop()

    @pytest.mark.asyncio
    async def test_usage_tracking_namespace_isolation(self):
        """Test that usage tracking properly isolates namespaces"""

        billing = await get_billing_engine()

        # Record usage for different namespaces
        namespaces = ["org-a/team-1", "org-b/team-2", "org-c/team-3"]
        job_costs = {}

        for i, namespace in enumerate(namespaces):
            cost = await billing.record_job_usage(
                namespace=namespace,
                job_id=f"job-{i}",
                cpu_cores=float(i + 1),
                memory_gb=float((i + 1) * 2),
                disk_gb=10.0,
                duration_seconds=3600,
                pricing_tier=PricingTier.BASIC,
            )
            job_costs[namespace] = cost

        # Verify isolation - each namespace should have different usage
        for namespace in namespaces:
            tracker = await billing.get_or_create_tracker(namespace)
            usage_report = tracker.get_usage_report(
                start_time=datetime.now(UTC) - timedelta(hours=1), end_time=datetime.now(UTC) + timedelta(hours=1)
            )

            assert usage_report.namespace == namespace
            assert usage_report.usage_metrics.job_count == 1
            assert usage_report.cost_breakdown.total_cost == job_costs[namespace]

        # Verify namespaces don't interfere with each other
        tracker_a = await billing.get_or_create_tracker("org-a/team-1")
        tracker_b = await billing.get_or_create_tracker("org-b/team-2")

        assert tracker_a.total_cost != tracker_b.total_cost
        assert tracker_a.cpu_core_seconds != tracker_b.cpu_core_seconds
        assert tracker_a.memory_gb_seconds != tracker_b.memory_gb_seconds
