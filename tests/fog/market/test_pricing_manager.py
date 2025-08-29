"""
Comprehensive test suite for the Dynamic Pricing Manager.

Tests dynamic pricing bands, resource lane pricing, market manipulation prevention,
and circuit breaker mechanisms.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from infrastructure.fog.market.pricing_manager import DynamicPricingManager, MarketCondition, PriceBand, ResourceLane


class TestDynamicPricingManager:
    """Test suite for DynamicPricingManager functionality."""

    @pytest.fixture
    async def pricing_manager(self):
        """Create a test pricing manager instance."""
        manager = DynamicPricingManager()
        await manager.initialize()
        return manager

    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        return {
            "cpu_utilization": Decimal("0.65"),
            "memory_utilization": Decimal("0.70"),
            "storage_utilization": Decimal("0.45"),
            "bandwidth_utilization": Decimal("0.80"),
            "gpu_utilization": Decimal("0.30"),
            "demand_multiplier": Decimal("1.2"),
            "supply_multiplier": Decimal("0.9"),
            "volatility_index": Decimal("0.15")
        }

    @pytest.mark.asyncio
    async def test_get_resource_price_basic(self, pricing_manager):
        """Test basic resource price calculation."""
        price_info = await pricing_manager.get_resource_price(
            lane=ResourceLane.CPU,
            quantity=Decimal("4"),
            duration_hours=Decimal("24")
        )

        assert price_info is not None
        assert "base_price" in price_info
        assert "final_price" in price_info
        assert "price_band" in price_info
        assert "market_condition" in price_info

        assert isinstance(price_info["base_price"], Decimal)
        assert isinstance(price_info["final_price"], Decimal)
        assert price_info["final_price"] > Decimal("0")

    @pytest.mark.asyncio
    async def test_price_band_transitions(self, pricing_manager, sample_market_data):
        """Test price band transitions based on market conditions."""
        # Test LOW demand scenario
        low_demand_data = sample_market_data.copy()
        low_demand_data["demand_multiplier"] = Decimal("0.6")
        low_demand_data["supply_multiplier"] = Decimal("1.4")

        with patch.object(pricing_manager, '_get_current_market_data', return_value=low_demand_data):
            price_info = await pricing_manager.get_resource_price(
                lane=ResourceLane.CPU,
                quantity=Decimal("2"),
                duration_hours=Decimal("1")
            )

            assert price_info["price_band"] == PriceBand.LOW
            assert price_info["market_condition"] == MarketCondition.BUYER_MARKET

        # Test HIGH demand scenario
        high_demand_data = sample_market_data.copy()
        high_demand_data["demand_multiplier"] = Decimal("2.1")
        high_demand_data["supply_multiplier"] = Decimal("0.5")

        with patch.object(pricing_manager, '_get_current_market_data', return_value=high_demand_data):
            price_info = await pricing_manager.get_resource_price(
                lane=ResourceLane.CPU,
                quantity=Decimal("2"),
                duration_hours=Decimal("1")
            )

            assert price_info["price_band"] == PriceBand.HIGH
            assert price_info["market_condition"] == MarketCondition.SELLER_MARKET

        # Test PREMIUM demand scenario
        premium_demand_data = sample_market_data.copy()
        premium_demand_data["demand_multiplier"] = Decimal("3.5")
        premium_demand_data["supply_multiplier"] = Decimal("0.3")
        premium_demand_data["volatility_index"] = Decimal("0.35")

        with patch.object(pricing_manager, '_get_current_market_data', return_value=premium_demand_data):
            price_info = await pricing_manager.get_resource_price(
                lane=ResourceLane.CPU,
                quantity=Decimal("2"),
                duration_hours=Decimal("1")
            )

            assert price_info["price_band"] == PriceBand.PREMIUM
            assert price_info["market_condition"] == MarketCondition.VOLATILE

    @pytest.mark.asyncio
    async def test_resource_lane_specific_pricing(self, pricing_manager):
        """Test that different resource lanes have appropriate pricing."""
        duration = Decimal("1")
        quantity = Decimal("1")

        # Get prices for different resource lanes
        cpu_price = await pricing_manager.get_resource_price(ResourceLane.CPU, quantity, duration)
        memory_price = await pricing_manager.get_resource_price(ResourceLane.MEMORY, quantity, duration)
        storage_price = await pricing_manager.get_resource_price(ResourceLane.STORAGE, quantity, duration)
        bandwidth_price = await pricing_manager.get_resource_price(ResourceLane.BANDWIDTH, quantity, duration)
        gpu_price = await pricing_manager.get_resource_price(ResourceLane.GPU, quantity, duration)
        specialized_price = await pricing_manager.get_resource_price(ResourceLane.SPECIALIZED, quantity, duration)

        # Verify all prices are valid
        prices = [cpu_price, memory_price, storage_price, bandwidth_price, gpu_price, specialized_price]
        for price_info in prices:
            assert price_info["final_price"] > Decimal("0")
            assert "lane" in price_info
            assert "pricing_factors" in price_info

        # GPU and Specialized should typically be more expensive
        assert gpu_price["final_price"] > cpu_price["final_price"]
        assert specialized_price["final_price"] >= gpu_price["final_price"]

    @pytest.mark.asyncio
    async def test_quantity_scaling(self, pricing_manager):
        """Test price scaling with quantity."""
        base_quantity = Decimal("1")
        large_quantity = Decimal("100")
        duration = Decimal("1")

        base_price_info = await pricing_manager.get_resource_price(
            ResourceLane.CPU, base_quantity, duration
        )
        large_price_info = await pricing_manager.get_resource_price(
            ResourceLane.CPU, large_quantity, duration
        )

        base_total = base_price_info["final_price"] * base_quantity
        large_total = large_price_info["final_price"] * large_quantity

        # Large quantities should have some bulk discount or scaling
        per_unit_base = base_total / base_quantity
        per_unit_large = large_total / large_quantity

        # Either bulk discount or at least not higher per-unit cost
        assert per_unit_large <= per_unit_base * Decimal("1.1")  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_duration_scaling(self, pricing_manager):
        """Test price scaling with duration."""
        quantity = Decimal("2")
        short_duration = Decimal("1")
        long_duration = Decimal("168")  # 1 week

        short_price_info = await pricing_manager.get_resource_price(
            ResourceLane.MEMORY, quantity, short_duration
        )
        long_price_info = await pricing_manager.get_resource_price(
            ResourceLane.MEMORY, quantity, long_duration
        )

        short_hourly = short_price_info["final_price"] / short_duration
        long_hourly = long_price_info["final_price"] / long_duration

        # Longer durations should have better hourly rates
        assert long_hourly <= short_hourly * Decimal("1.1")  # Allow 10% tolerance

    @pytest.mark.asyncio
    async def test_circuit_breaker_activation(self, pricing_manager):
        """Test circuit breaker activation under extreme conditions."""
        # Create extreme market conditions
        extreme_market_data = {
            "cpu_utilization": Decimal("0.95"),
            "memory_utilization": Decimal("0.98"),
            "storage_utilization": Decimal("0.92"),
            "bandwidth_utilization": Decimal("0.97"),
            "gpu_utilization": Decimal("0.89"),
            "demand_multiplier": Decimal("10.0"),  # Extreme demand
            "supply_multiplier": Decimal("0.1"),   # Very low supply
            "volatility_index": Decimal("0.85")    # High volatility
        }

        with patch.object(pricing_manager, '_get_current_market_data', return_value=extreme_market_data):
            price_info = await pricing_manager.get_resource_price(
                lane=ResourceLane.CPU,
                quantity=Decimal("1"),
                duration_hours=Decimal("1")
            )

            # Circuit breaker should be activated
            assert price_info.get("circuit_breaker_active", False) is True
            assert "max_price_cap" in price_info

            # Price should be capped
            max_cap = price_info["max_price_cap"]
            assert price_info["final_price"] <= max_cap

    @pytest.mark.asyncio
    async def test_market_manipulation_detection(self, pricing_manager):
        """Test detection of potential market manipulation."""
        # Simulate rapid price changes that could indicate manipulation
        manipulation_data = [
            {"demand_multiplier": Decimal("1.0"), "supply_multiplier": Decimal("1.0")},
            {"demand_multiplier": Decimal("5.0"), "supply_multiplier": Decimal("1.0")},  # Sudden spike
            {"demand_multiplier": Decimal("0.2"), "supply_multiplier": Decimal("1.0")},  # Sudden drop
            {"demand_multiplier": Decimal("4.8"), "supply_multiplier": Decimal("1.0")},  # Another spike
        ]

        manipulation_detected = False

        for data in manipulation_data:
            full_data = {
                "cpu_utilization": Decimal("0.70"),
                "memory_utilization": Decimal("0.70"),
                "storage_utilization": Decimal("0.70"),
                "bandwidth_utilization": Decimal("0.70"),
                "gpu_utilization": Decimal("0.70"),
                "volatility_index": Decimal("0.45"),
                **data
            }

            with patch.object(pricing_manager, '_get_current_market_data', return_value=full_data):
                await pricing_manager._update_market_conditions()

                # Check for manipulation flags
                market_status = await pricing_manager.get_market_status()
                if market_status.get("manipulation_risk", Decimal("0")) > Decimal("0.7"):
                    manipulation_detected = True
                    break

        # Should eventually detect manipulation pattern
        assert manipulation_detected is True

    @pytest.mark.asyncio
    async def test_historical_price_tracking(self, pricing_manager):
        """Test historical price data collection and analysis."""
        # Get initial price
        initial_price = await pricing_manager.get_resource_price(
            ResourceLane.CPU, Decimal("1"), Decimal("1")
        )

        # Simulate price history accumulation
        await pricing_manager._record_price_point(
            ResourceLane.CPU,
            initial_price["final_price"],
            datetime.utcnow()
        )

        # Get price history
        history = await pricing_manager.get_price_history(
            ResourceLane.CPU,
            hours_back=24
        )

        assert history is not None
        assert len(history) >= 1
        assert "timestamps" in history
        assert "prices" in history
        assert len(history["timestamps"]) == len(history["prices"])

    @pytest.mark.asyncio
    async def test_price_predictions(self, pricing_manager):
        """Test price prediction functionality."""
        # Add some historical data points
        current_time = datetime.utcnow()
        historical_prices = [
            (Decimal("0.10"), current_time - timedelta(hours=6)),
            (Decimal("0.12"), current_time - timedelta(hours=5)),
            (Decimal("0.11"), current_time - timedelta(hours=4)),
            (Decimal("0.13"), current_time - timedelta(hours=3)),
            (Decimal("0.12"), current_time - timedelta(hours=2)),
            (Decimal("0.14"), current_time - timedelta(hours=1)),
        ]

        for price, timestamp in historical_prices:
            await pricing_manager._record_price_point(ResourceLane.CPU, price, timestamp)

        # Get price prediction
        prediction = await pricing_manager.predict_price(
            ResourceLane.CPU,
            hours_ahead=2
        )

        assert prediction is not None
        assert "predicted_price" in prediction
        assert "confidence_level" in prediction
        assert "trend_direction" in prediction

        assert isinstance(prediction["predicted_price"], Decimal)
        assert Decimal("0") <= prediction["confidence_level"] <= Decimal("1")
        assert prediction["trend_direction"] in ["up", "down", "stable"]

    @pytest.mark.asyncio
    async def test_concurrent_pricing_requests(self, pricing_manager):
        """Test handling of concurrent pricing requests."""
        # Create multiple concurrent pricing requests
        tasks = []
        for i in range(20):
            lane = list(ResourceLane)[i % len(ResourceLane)]
            quantity = Decimal(str(1 + (i % 5)))
            duration = Decimal(str(1 + (i % 10)))

            task = asyncio.create_task(
                pricing_manager.get_resource_price(lane, quantity, duration)
            )
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all requests succeeded
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 20

        # Verify all results have valid structure
        for result in successful_results:
            assert "final_price" in result
            assert result["final_price"] > Decimal("0")

    @pytest.mark.asyncio
    async def test_pricing_under_stress(self, pricing_manager, sample_market_data):
        """Test pricing system under stress conditions."""
        # Simulate high-stress market conditions
        stress_data = sample_market_data.copy()
        stress_data.update({
            "cpu_utilization": Decimal("0.98"),
            "memory_utilization": Decimal("0.95"),
            "storage_utilization": Decimal("0.92"),
            "bandwidth_utilization": Decimal("0.97"),
            "gpu_utilization": Decimal("0.88"),
            "demand_multiplier": Decimal("8.5"),
            "supply_multiplier": Decimal("0.2"),
            "volatility_index": Decimal("0.75")
        })

        with patch.object(pricing_manager, '_get_current_market_data', return_value=stress_data):
            # Make rapid requests to stress test the system
            start_time = datetime.utcnow()
            requests_completed = 0

            while (datetime.utcnow() - start_time).seconds < 5:  # 5 second stress test
                try:
                    price_info = await pricing_manager.get_resource_price(
                        ResourceLane.CPU, Decimal("1"), Decimal("1")
                    )
                    assert price_info["final_price"] > Decimal("0")
                    requests_completed += 1
                except Exception as e:
                    pytest.fail(f"Pricing failed under stress: {e}")

            # Should handle multiple requests per second
            assert requests_completed > 10

    @pytest.mark.asyncio
    async def test_market_condition_analysis(self, pricing_manager):
        """Test comprehensive market condition analysis."""
        market_status = await pricing_manager.get_market_status()

        assert market_status is not None
        assert "overall_condition" in market_status
        assert "resource_conditions" in market_status
        assert "volatility_metrics" in market_status
        assert "manipulation_risk" in market_status
        assert "circuit_breaker_status" in market_status

        # Verify resource conditions for all lanes
        resource_conditions = market_status["resource_conditions"]
        for lane in ResourceLane:
            assert lane.value in resource_conditions
            lane_condition = resource_conditions[lane.value]
            assert "utilization" in lane_condition
            assert "price_trend" in lane_condition
            assert "availability" in lane_condition

    @pytest.mark.asyncio
    async def test_pricing_configuration_updates(self, pricing_manager):
        """Test dynamic pricing configuration updates."""
        # Get initial configuration
        await pricing_manager.get_pricing_configuration()

        # Update configuration
        new_config = {
            "base_prices": {
                ResourceLane.CPU.value: Decimal("0.05"),
                ResourceLane.MEMORY.value: Decimal("0.02")
            },
            "volatility_threshold": Decimal("0.25"),
            "circuit_breaker_threshold": Decimal("5.0")
        }

        success = await pricing_manager.update_configuration(new_config)
        assert success is True

        # Verify configuration was updated
        updated_config = await pricing_manager.get_pricing_configuration()
        assert updated_config["base_prices"][ResourceLane.CPU.value] == Decimal("0.05")
        assert updated_config["volatility_threshold"] == Decimal("0.25")

    @pytest.mark.asyncio
    async def test_bulk_pricing_discounts(self, pricing_manager):
        """Test bulk pricing discount mechanisms."""
        # Small quantity
        small_qty_price = await pricing_manager.get_resource_price(
            ResourceLane.CPU, Decimal("1"), Decimal("24")
        )

        # Large quantity (should trigger bulk discount)
        large_qty_price = await pricing_manager.get_resource_price(
            ResourceLane.CPU, Decimal("1000"), Decimal("24")
        )

        # Calculate per-unit costs
        small_per_unit = small_qty_price["final_price"]
        large_per_unit = large_qty_price["final_price"] / Decimal("1000")

        # Large quantities should have lower per-unit cost
        assert large_per_unit < small_per_unit

        # Should have bulk discount information
        assert "bulk_discount_applied" in large_qty_price
        if large_qty_price["bulk_discount_applied"]:
            assert "discount_percentage" in large_qty_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
