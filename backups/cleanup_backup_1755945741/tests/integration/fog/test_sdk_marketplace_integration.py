"""
Integration tests for fog SDK marketplace functionality

Tests the complete SDK marketplace integration:
- Price quote requests and response validation
- Bid submission and status tracking
- Job execution with budget constraints
- Cost estimation and marketplace prices
- Error handling and edge cases
"""

from datetime import UTC, datetime

import pytest
from packages.fog.gateway.scheduler.marketplace import BidType, PricingTier, get_marketplace_engine
from packages.fog.sdk.python.fog_client import CostEstimate, FogClient, FogClientError, MarketplacePrices, PriceQuote


class TestSDKMarketplaceIntegration:
    """Test SDK marketplace functionality with real backend"""

    @pytest.fixture
    async def marketplace_backend(self):
        """Set up marketplace backend for SDK testing"""
        engine = await get_marketplace_engine()

        # Add test listings
        await engine.add_resource_listing(
            node_id="sdk-test-server-1",
            cpu_cores=4.0,
            memory_gb=8.0,
            disk_gb=100.0,
            spot_price=0.08,
            on_demand_price=0.12,
            trust_score=0.7,
            pricing_tier=PricingTier.STANDARD,
        )

        await engine.add_resource_listing(
            node_id="sdk-test-server-2",
            cpu_cores=2.0,
            memory_gb=4.0,
            disk_gb=50.0,
            spot_price=0.10,
            on_demand_price=0.15,
            trust_score=0.6,
            pricing_tier=PricingTier.BASIC,
        )

        yield engine
        await engine.stop()

    @pytest.fixture
    def fog_client(self):
        """Create fog client for testing"""
        return FogClient(base_url="http://localhost:8000", api_key="test-api-key", namespace="test-org/sdk-integration")

    @pytest.mark.asyncio
    async def test_get_price_quote_integration(self, marketplace_backend, fog_client):
        """Test price quote functionality end-to-end"""

        # Mock the HTTP request to return marketplace data
        async def mock_request(method, path, json_data=None, params=None):
            if path == "/v1/fog/quotes":
                # Use actual marketplace engine to generate quote
                quote_data = await marketplace_backend.get_price_quote(
                    cpu_cores=json_data["cpu_cores"],
                    memory_gb=json_data["memory_gb"],
                    duration_hours=json_data["estimated_duration_hours"],
                    bid_type=BidType(json_data["bid_type"]),
                    pricing_tier=PricingTier(json_data["pricing_tier"]),
                    min_trust_score=json_data["min_trust_score"],
                )

                return {
                    "available": quote_data["available"],
                    "quote": quote_data.get("quote", {}),
                    "market_conditions": quote_data.get("market_conditions", {}),
                    "recommendations": quote_data.get("recommendations", {}),
                    "reason": quote_data.get("reason"),
                }
            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Test successful price quote
            quote = await fog_client.get_price_quote(
                cpu_cores=2.0,
                memory_gb=4.0,
                estimated_duration_hours=3.0,
                bid_type="spot",
                pricing_tier="standard",
                min_trust_score=0.5,
            )

            assert isinstance(quote, PriceQuote)
            assert quote.available is True
            assert quote.min_price is not None
            assert quote.max_price is not None
            assert quote.avg_price is not None
            assert quote.current_spot_rate is not None
            assert quote.available_providers is not None
            assert quote.suggested_max_price is not None

    @pytest.mark.asyncio
    async def test_get_marketplace_prices_integration(self, marketplace_backend, fog_client):
        """Test marketplace pricing information retrieval"""

        async def mock_request(method, path, json_data=None, params=None):
            if path == "/v1/fog/prices":
                status = await marketplace_backend.get_marketplace_status()
                return {
                    "currency": "USD",
                    "last_updated": datetime.now(UTC).isoformat(),
                    "spot_price_per_cpu_hour": status["pricing"]["current_spot_price_per_cpu_hour"],
                    "on_demand_price_per_cpu_hour": status["pricing"]["current_on_demand_price_per_cpu_hour"],
                    "price_volatility": status["pricing"]["price_volatility_24h"],
                    "market_conditions": {
                        "utilization_rate": status["resource_supply"]["utilization_rate"],
                        "demand_supply_ratio": status["resource_demand"]["demand_supply_ratio"],
                        "available_providers": status["marketplace_summary"]["active_listings"],
                    },
                    "pricing_tiers": {
                        "basic": {"multiplier": 1.0, "description": "Best effort"},
                        "standard": {"multiplier": 1.5, "description": "Replicated"},
                        "premium": {"multiplier": 2.0, "description": "Replicated + Attested"},
                    },
                }
            return {}

        async with fog_client:
            fog_client._request = mock_request

            prices = await fog_client.get_marketplace_prices()

            assert isinstance(prices, MarketplacePrices)
            assert prices.currency == "USD"
            assert prices.spot_price_per_cpu_hour > 0
            assert prices.on_demand_price_per_cpu_hour > 0
            assert prices.price_volatility >= 0
            assert prices.available_providers >= 0
            assert "basic" in prices.pricing_tiers
            assert "standard" in prices.pricing_tiers
            assert "premium" in prices.pricing_tiers

    @pytest.mark.asyncio
    async def test_estimate_job_cost_integration(self, marketplace_backend, fog_client):
        """Test job cost estimation with marketplace data"""

        async def mock_request(method, path, json_data=None, params=None):
            if path == "/v1/fog/quotes":
                # Generate real quote from marketplace
                quote_data = await marketplace_backend.get_price_quote(
                    cpu_cores=json_data["cpu_cores"],
                    memory_gb=json_data["memory_gb"],
                    duration_hours=json_data["estimated_duration_hours"],
                    bid_type=BidType(json_data["bid_type"]),
                    pricing_tier=PricingTier(json_data["pricing_tier"]),
                    min_trust_score=json_data["min_trust_score"],
                )

                return {
                    "available": quote_data["available"],
                    "quote": quote_data.get("quote", {}),
                    "market_conditions": quote_data.get("market_conditions", {}),
                    "recommendations": quote_data.get("recommendations", {}),
                    "reason": quote_data.get("reason"),
                }
            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Test cost estimation
            estimate = await fog_client.estimate_job_cost(
                image="test-app:latest",
                cpu_cores=2.0,
                memory_gb=4.0,
                estimated_duration_hours=2.0,
                bid_type="spot",
                pricing_tier="basic",
            )

            assert isinstance(estimate, CostEstimate)
            assert estimate.estimated_cost > 0
            assert estimate.pricing_tier == "basic"
            assert estimate.bid_type == "spot"
            assert estimate.duration_hours == 2.0
            assert estimate.confidence_level > 0
            assert estimate.confidence_level <= 1.0

            # Verify cost breakdown
            assert "cpu_cost" in estimate.cost_breakdown
            assert "memory_cost" in estimate.cost_breakdown
            assert "disk_cost" in estimate.cost_breakdown
            assert "network_cost" in estimate.cost_breakdown

            # Verify price range
            assert "min" in estimate.price_range
            assert "max" in estimate.price_range
            assert "market" in estimate.price_range

    @pytest.mark.asyncio
    async def test_submit_bid_integration(self, marketplace_backend, fog_client):
        """Test bid submission through SDK"""

        async def mock_request(method, path, json_data=None, params=None):
            if path == "/v1/fog/marketplace/bids":
                # Use actual marketplace to submit bid
                bid_id = await marketplace_backend.submit_bid(
                    namespace=json_data["namespace"],
                    cpu_cores=json_data["resources"]["cpu_cores"],
                    memory_gb=json_data["resources"]["memory_gb"],
                    max_price=json_data["max_price"],
                    bid_type=BidType(json_data["bid_type"]),
                    estimated_duration_hours=json_data["estimated_duration_hours"],
                    job_spec={"image": json_data["image"], "args": json_data["args"], "env": json_data["env"]},
                )

                return {"bid_id": bid_id, "status": "pending", "message": "Bid submitted successfully"}
            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Submit bid
            result = await fog_client.submit_bid(
                image="test-job:latest",
                cpu_cores=1.5,
                memory_gb=3.0,
                max_price=0.50,
                args=["python", "app.py"],
                env={"ENV": "test"},
                bid_type="spot",
                pricing_tier="basic",
            )

            assert "bid_id" in result
            assert result["status"] == "pending"
            assert "bid_" in result["bid_id"]  # Should have proper bid ID format

    @pytest.mark.asyncio
    async def test_get_bid_status_integration(self, marketplace_backend, fog_client):
        """Test bid status tracking"""

        # First submit a bid through marketplace
        bid_id = await marketplace_backend.submit_bid(
            namespace="test-org/sdk-status-test",
            cpu_cores=1.0,
            memory_gb=2.0,
            max_price=0.25,
            bid_type=BidType.SPOT,
            estimated_duration_hours=1.0,
        )

        async def mock_request(method, path, json_data=None, params=None):
            if f"/v1/fog/marketplace/bids/{bid_id}" in path:
                # Get bid status from marketplace
                if bid_id in marketplace_backend.pending_bids:
                    bid = marketplace_backend.pending_bids[bid_id]
                    return {
                        "bid_id": bid_id,
                        "status": bid.status.value,
                        "cpu_cores": bid.cpu_cores,
                        "memory_gb": bid.memory_gb,
                        "max_price": bid.max_price,
                        "created_at": bid.created_at.isoformat(),
                        "matched_listing_id": bid.matched_listing_id,
                        "actual_cost": bid.actual_cost,
                    }
                return {"error": "Bid not found"}
            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Get bid status
            status = await fog_client.get_bid_status(bid_id)

            assert status["bid_id"] == bid_id
            assert status["status"] in ["pending", "matched", "active", "completed", "failed", "cancelled"]
            assert status["cpu_cores"] == 1.0
            assert status["memory_gb"] == 2.0
            assert status["max_price"] == 0.25

    @pytest.mark.asyncio
    async def test_run_job_with_budget_integration(self, marketplace_backend, fog_client):
        """Test budget-constrained job execution"""

        job_matched = False
        job_id = "test-job-12345"

        async def mock_request(method, path, json_data=None, params=None):
            nonlocal job_matched

            if path == "/v1/fog/marketplace/bids":
                # Submit bid and immediately mark as matched with job
                bid_id = await marketplace_backend.submit_bid(
                    namespace=json_data["namespace"],
                    cpu_cores=json_data["resources"]["cpu_cores"],
                    memory_gb=json_data["resources"]["memory_gb"],
                    max_price=json_data["max_price"],
                    bid_type=BidType(json_data["bid_type"]),
                )

                # Simulate immediate matching
                job_matched = True
                return {"bid_id": bid_id, "status": "matched", "job_id": job_id, "message": "Bid matched immediately"}

            elif path == f"/v1/fog/jobs/{job_id}":
                # Return job completion
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "exit_code": 0,
                    "cpu_seconds_used": 3600.0,  # 1 hour
                    "memory_mb_peak": 2048,
                    "execution_latency_ms": 3600000.0,  # 1 hour in ms
                }

            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Run job with budget constraint
            result = await fog_client.run_job_with_budget(
                image="test-budget-job:latest",
                max_price=0.30,
                args=["python", "budget_test.py"],
                resources={"cpu_cores": 1.0, "memory_gb": 2.0},
                bid_type="spot",
                timeout=10.0,  # Short timeout for test
            )

            assert result.job_id == job_id
            assert result.status == "completed"
            assert result.exit_code == 0
            assert result.cpu_seconds_used == 3600.0
            assert job_matched is True  # Verify bid was processed

    @pytest.mark.asyncio
    async def test_marketplace_error_handling(self, fog_client):
        """Test error handling in marketplace operations"""

        async def mock_request_error(method, path, json_data=None, params=None):
            if path == "/v1/fog/quotes":
                return {
                    "available": False,
                    "reason": "No resources available matching requirements",
                    "suggestions": {
                        "reduce_requirements": ["Reduce CPU from 100 to 50 cores"],
                        "increase_budget": ["Consider budget of $50.00"],
                        "wait_for_availability": ["Check again in 5-10 minutes"],
                    },
                }
            return {}

        async with fog_client:
            fog_client._request = mock_request_error

            # Test quote for impossible requirements
            quote = await fog_client.get_price_quote(
                cpu_cores=100.0, memory_gb=500.0, estimated_duration_hours=1.0  # Impossible requirement
            )

            assert quote.available is False
            assert quote.reason is not None
            assert "No resources available" in quote.reason

    @pytest.mark.asyncio
    async def test_price_comparison_across_bid_types(self, marketplace_backend, fog_client):
        """Test price differences between spot and on-demand bidding"""

        async def mock_request(method, path, json_data=None, params=None):
            if path == "/v1/fog/quotes":
                quote_data = await marketplace_backend.get_price_quote(
                    cpu_cores=json_data["cpu_cores"],
                    memory_gb=json_data["memory_gb"],
                    duration_hours=json_data["estimated_duration_hours"],
                    bid_type=BidType(json_data["bid_type"]),
                    pricing_tier=PricingTier(json_data["pricing_tier"]),
                    min_trust_score=json_data["min_trust_score"],
                )

                return {
                    "available": quote_data["available"],
                    "quote": quote_data.get("quote", {}),
                    "market_conditions": quote_data.get("market_conditions", {}),
                    "recommendations": quote_data.get("recommendations", {}),
                    "reason": quote_data.get("reason"),
                }
            return {}

        async with fog_client:
            fog_client._request = mock_request

            # Get spot quote
            spot_quote = await fog_client.get_price_quote(
                cpu_cores=2.0, memory_gb=4.0, estimated_duration_hours=2.0, bid_type="spot", pricing_tier="basic"
            )

            # Get on-demand quote
            on_demand_quote = await fog_client.get_price_quote(
                cpu_cores=2.0, memory_gb=4.0, estimated_duration_hours=2.0, bid_type="on_demand", pricing_tier="basic"
            )

            assert spot_quote.available is True
            assert on_demand_quote.available is True

            # On-demand should generally be more expensive than spot
            assert on_demand_quote.current_on_demand_rate >= spot_quote.current_spot_rate

            # Both should have valid pricing information
            assert spot_quote.avg_price > 0
            assert on_demand_quote.avg_price > 0


class TestSDKMarketplaceErrorHandling:
    """Test SDK error handling for marketplace operations"""

    @pytest.fixture
    def fog_client(self):
        return FogClient(base_url="http://localhost:8000", api_key="test-api-key", namespace="test-org/error-handling")

    @pytest.mark.asyncio
    async def test_price_quote_network_error(self, fog_client):
        """Test handling of network errors in price quotes"""

        async def mock_request_network_error(method, path, json_data=None, params=None):
            raise aiohttp.ClientError("Network connection failed")

        async with fog_client:
            fog_client._request = mock_request_network_error

            with pytest.raises(FogClientError) as exc_info:
                await fog_client.get_price_quote(cpu_cores=1.0, memory_gb=2.0, estimated_duration_hours=1.0)

            assert "Request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_bid_submission_validation_error(self, fog_client):
        """Test validation errors in bid submission"""

        async def mock_request_validation_error(method, path, json_data=None, params=None):
            if path == "/v1/fog/marketplace/bids":
                raise aiohttp.ClientResponseError(
                    request_info=None, history=None, status=400, message="Validation error: max_price must be positive"
                )
            return {}

        async with fog_client:
            fog_client._request = mock_request_validation_error

            with pytest.raises(FogClientError) as exc_info:
                await fog_client.submit_bid(
                    image="test:latest",
                    cpu_cores=1.0,
                    memory_gb=2.0,
                    max_price=-1.0,  # Invalid negative price
                )

            assert "400" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_job_budget_timeout_handling(self, fog_client):
        """Test timeout handling in budget-constrained jobs"""

        async def mock_request_slow_matching(method, path, json_data=None, params=None):
            if path == "/v1/fog/marketplace/bids":
                # Return bid that never gets matched
                return {"bid_id": "slow-bid-123", "status": "pending", "message": "Bid submitted, waiting for matching"}
            elif "bids/slow-bid-123" in path:
                # Always return pending status
                return {"bid_id": "slow-bid-123", "status": "pending", "message": "Still waiting for matching"}
            elif path.startswith("/v1/fog/marketplace/bids/") and method == "DELETE":
                # Simulate successful cancellation
                return {"message": "Bid cancelled successfully"}
            return {}

        async with fog_client:
            fog_client._request = mock_request_slow_matching

            with pytest.raises(FogClientError) as exc_info:
                await fog_client.run_job_with_budget(
                    image="slow-job:latest", max_price=0.10, timeout=0.5  # Very short timeout
                )

            assert "not matched within" in str(exc_info.value)
