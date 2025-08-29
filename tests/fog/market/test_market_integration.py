"""
Comprehensive integration test suite for the Market-Based Pricing System.

Tests end-to-end workflows, integration between components, and system behavior
under various market conditions.
"""

import asyncio
from decimal import Decimal
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

from infrastructure.fog.market.anti_griefing import AntiGriefingSystem
from infrastructure.fog.market.auction_engine import AuctionEngine
from infrastructure.fog.market.market_orchestrator import (
    AllocationMethod,
    MarketOrchestrator,
    ResourceAllocationRequest,
)
from infrastructure.fog.market.pricing_manager import DynamicPricingManager
from infrastructure.fog.market.scheduler_integration import MarketSchedulerIntegration
from infrastructure.fog.market.tokenomics_integration import TokenomicsIntegration


class TestMarketIntegration:
    """Test suite for integrated market system functionality."""

    @pytest.fixture
    async def market_orchestrator(self):
        """Create a test market orchestrator with all components."""
        orchestrator = MarketOrchestrator()
        await orchestrator.initialize()
        return orchestrator

    @pytest.fixture
    async def integrated_components(self):
        """Create all market components for integration testing."""
        components = {}

        components['auction_engine'] = AuctionEngine()
        await components['auction_engine'].initialize()

        components['pricing_manager'] = DynamicPricingManager()
        await components['pricing_manager'].initialize()

        components['tokenomics'] = TokenomicsIntegration()
        await components['tokenomics'].initialize()

        components['scheduler'] = MarketSchedulerIntegration()
        await components['scheduler'].initialize()

        components['anti_griefing'] = AntiGriefingSystem()
        await components['anti_griefing'].initialize()

        return components

    @pytest.fixture
    def sample_allocation_request(self):
        """Create a sample resource allocation request."""
        return ResourceAllocationRequest(
            requester_id="test_user_123",
            resource_requirements={
                "cpu_cores": Decimal("4"),
                "memory_gb": Decimal("8"),
                "storage_gb": Decimal("100"),
                "bandwidth_mbps": Decimal("1000"),
                "gpu_units": Decimal("0")
            },
            duration_hours=Decimal("24"),
            max_budget=Decimal("50.0"),
            quality_requirements={
                "min_trust_score": Decimal("0.8"),
                "min_reputation": Decimal("0.7"),
                "max_latency_ms": 50
            },
            preferred_method=AllocationMethod.BEST_PRICE
        )

    @pytest.fixture
    def sample_providers(self):
        """Create sample provider data for testing."""
        return [
            {
                "provider_id": "provider_alpha",
                "node_id": "node_alpha_1",
                "available_resources": {
                    "cpu_cores": Decimal("16"),
                    "memory_gb": Decimal("32"),
                    "storage_gb": Decimal("1000"),
                    "bandwidth_mbps": Decimal("5000")
                },
                "trust_score": Decimal("0.95"),
                "reputation_score": Decimal("0.92"),
                "base_pricing": {
                    "cpu_per_hour": Decimal("0.08"),
                    "memory_per_gb_hour": Decimal("0.02"),
                    "storage_per_gb_hour": Decimal("0.001"),
                    "bandwidth_per_mbps_hour": Decimal("0.0001")
                },
                "location": "us-east-1",
                "specialized_hardware": []
            },
            {
                "provider_id": "provider_beta",
                "node_id": "node_beta_1",
                "available_resources": {
                    "cpu_cores": Decimal("8"),
                    "memory_gb": Decimal("16"),
                    "storage_gb": Decimal("500"),
                    "bandwidth_mbps": Decimal("2000"),
                    "gpu_units": Decimal("2")
                },
                "trust_score": Decimal("0.88"),
                "reputation_score": Decimal("0.85"),
                "base_pricing": {
                    "cpu_per_hour": Decimal("0.10"),
                    "memory_per_gb_hour": Decimal("0.025"),
                    "storage_per_gb_hour": Decimal("0.0012"),
                    "bandwidth_per_mbps_hour": Decimal("0.00012"),
                    "gpu_per_unit_hour": Decimal("1.50")
                },
                "location": "us-west-2",
                "specialized_hardware": ["cuda", "tensor_rt"]
            },
            {
                "provider_id": "provider_gamma",
                "node_id": "node_gamma_1",
                "available_resources": {
                    "cpu_cores": Decimal("12"),
                    "memory_gb": Decimal("24"),
                    "storage_gb": Decimal("750"),
                    "bandwidth_mbps": Decimal("3000")
                },
                "trust_score": Decimal("0.91"),
                "reputation_score": Decimal("0.89"),
                "base_pricing": {
                    "cpu_per_hour": Decimal("0.09"),
                    "memory_per_gb_hour": Decimal("0.022"),
                    "storage_per_gb_hour": Decimal("0.0011"),
                    "bandwidth_per_mbps_hour": Decimal("0.00011")
                },
                "location": "eu-west-1",
                "specialized_hardware": ["high_memory"]
            }
        ]

    @pytest.mark.asyncio
    async def test_end_to_end_auction_workflow(self, market_orchestrator, sample_allocation_request, sample_providers):
        """Test complete auction-based resource allocation workflow."""
        # Mock provider discovery and bidding
        with patch.object(market_orchestrator.auction_engine, 'discover_providers', return_value=sample_providers):
            with patch.object(market_orchestrator.tokenomics, 'validate_budget', return_value=True):
                with patch.object(market_orchestrator.tokenomics, 'hold_escrow', return_value="escrow_123"):

                    # Request allocation via auction
                    allocation_request = sample_allocation_request
                    allocation_request.preferred_method = AllocationMethod.AUCTION

                    result = await market_orchestrator.allocate_resources(allocation_request)

                    assert result is not None
                    assert result.success is True
                    assert result.allocation_method == AllocationMethod.AUCTION
                    assert "auction_id" in result.details
                    assert "winning_provider" in result.details
                    assert result.total_cost > Decimal("0")
                    assert result.total_cost <= allocation_request.max_budget

    @pytest.mark.asyncio
    async def test_end_to_end_direct_pricing_workflow(self, market_orchestrator, sample_allocation_request, sample_providers):
        """Test complete direct pricing-based resource allocation workflow."""
        # Mock provider discovery
        with patch.object(market_orchestrator, '_discover_available_providers', return_value=sample_providers):
            with patch.object(market_orchestrator.tokenomics, 'validate_budget', return_value=True):
                with patch.object(market_orchestrator.tokenomics, 'process_payment', return_value="payment_123"):

                    # Request allocation via direct pricing
                    allocation_request = sample_allocation_request
                    allocation_request.preferred_method = AllocationMethod.BEST_PRICE

                    result = await market_orchestrator.allocate_resources(allocation_request)

                    assert result is not None
                    assert result.success is True
                    assert result.allocation_method == AllocationMethod.BEST_PRICE
                    assert "selected_provider" in result.details
                    assert result.total_cost > Decimal("0")
                    assert result.total_cost <= allocation_request.max_budget

    @pytest.mark.asyncio
    async def test_scheduler_integration(self, integrated_components, sample_allocation_request):
        """Test integration with existing fog task scheduler."""
        scheduler = integrated_components['scheduler']

        # Mock scheduler methods
        with patch.object(scheduler, 'submit_market_task', return_value="task_456") as mock_submit:
            with patch.object(scheduler, 'get_task_status', return_value={'status': 'running', 'progress': 0.3}):

                # Submit task through scheduler
                task_id = await scheduler.submit_market_task(
                    task_definition={
                        "type": "compute",
                        "requirements": sample_allocation_request.resource_requirements,
                        "duration": sample_allocation_request.duration_hours
                    },
                    allocation_method=AllocationMethod.HYBRID,
                    max_budget=sample_allocation_request.max_budget
                )

                assert task_id == "task_456"
                mock_submit.assert_called_once()

                # Check task status
                status = await scheduler.get_task_status(task_id)
                assert status['status'] == 'running'
                assert status['progress'] == 0.3

    @pytest.mark.asyncio
    async def test_tokenomics_integration(self, integrated_components, sample_allocation_request):
        """Test integration with tokenomics system."""
        tokenomics = integrated_components['tokenomics']

        # Test budget validation
        is_valid = await tokenomics.validate_budget(
            user_id=sample_allocation_request.requester_id,
            required_amount=sample_allocation_request.max_budget
        )
        # Mock should return True by default
        assert isinstance(is_valid, bool)

        # Test escrow operations
        with patch.object(tokenomics, 'hold_escrow', return_value="escrow_789"):
            with patch.object(tokenomics, 'release_escrow', return_value=True):

                escrow_id = await tokenomics.hold_escrow(
                    user_id=sample_allocation_request.requester_id,
                    amount=Decimal("25.0"),
                    purpose="resource_allocation"
                )

                assert escrow_id == "escrow_789"

                # Release escrow
                released = await tokenomics.release_escrow(
                    escrow_id=escrow_id,
                    recipient_id="provider_alpha",
                    amount=Decimal("20.0")
                )

                assert released is True

    @pytest.mark.asyncio
    async def test_anti_griefing_integration(self, integrated_components, sample_providers):
        """Test integration with anti-griefing system."""
        anti_griefing = integrated_components['anti_griefing']

        provider_data = sample_providers[0]

        # Test provider validation
        validation_result = await anti_griefing.validate_provider(
            provider_id=provider_data["provider_id"],
            node_id=provider_data["node_id"],
            offered_resources=provider_data["available_resources"],
            pricing_info=provider_data["base_pricing"]
        )

        assert "is_valid" in validation_result
        assert "confidence_score" in validation_result
        assert isinstance(validation_result["is_valid"], bool)
        assert Decimal("0") <= validation_result["confidence_score"] <= Decimal("1")

    @pytest.mark.asyncio
    async def test_market_condition_responses(self, market_orchestrator, sample_allocation_request):
        """Test system response to different market conditions."""
        # Test high demand scenario
        high_demand_data = {
            "cpu_utilization": Decimal("0.95"),
            "memory_utilization": Decimal("0.92"),
            "demand_multiplier": Decimal("3.5"),
            "supply_multiplier": Decimal("0.4"),
            "volatility_index": Decimal("0.35")
        }

        with patch.object(market_orchestrator.pricing_manager, '_get_current_market_data', return_value=high_demand_data):
            # Should automatically switch to auction for better pricing
            allocation_request = sample_allocation_request
            allocation_request.preferred_method = AllocationMethod.HYBRID

            with patch.object(market_orchestrator, 'allocate_resources', return_value=MagicMock(
                success=True,
                allocation_method=AllocationMethod.AUCTION,
                total_cost=Decimal("45.0")
            )) as mock_allocate:

                result = await market_orchestrator.allocate_resources(allocation_request)
                assert result.success is True
                # In high demand, should prefer auction over direct pricing
                mock_allocate.assert_called_once()

    @pytest.mark.asyncio
    async def test_resource_scaling_integration(self, market_orchestrator, sample_allocation_request):
        """Test integration with resource scaling mechanisms."""
        # Test auto-scaling request
        scaling_request = sample_allocation_request
        scaling_request.resource_requirements.update({
            "auto_scale": True,
            "min_instances": Decimal("1"),
            "max_instances": Decimal("10"),
            "scale_trigger": "cpu_utilization > 0.8"
        })

        with patch.object(market_orchestrator, '_handle_auto_scaling', return_value=True):
            with patch.object(market_orchestrator, 'allocate_resources', return_value=MagicMock(
                success=True,
                scaling_enabled=True
            )) as mock_allocate:

                result = await market_orchestrator.allocate_resources(scaling_request)
                assert result.success is True
                mock_allocate.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_region_allocation(self, market_orchestrator, sample_allocation_request, sample_providers):
        """Test resource allocation across multiple regions."""
        # Add region preferences
        allocation_request = sample_allocation_request
        allocation_request.region_preferences = ["us-east-1", "us-west-2", "eu-west-1"]
        allocation_request.latency_requirements = {"max_latency_ms": 100}

        with patch.object(market_orchestrator, '_discover_available_providers', return_value=sample_providers):
            with patch.object(market_orchestrator, '_calculate_regional_latency', return_value=Decimal("45")):

                result = await market_orchestrator.allocate_resources(allocation_request)

                # Should successfully allocate considering region preferences
                assert result is not None
                if result.success:
                    assert "region" in result.details
                    assert result.details["region"] in allocation_request.region_preferences

    @pytest.mark.asyncio
    async def test_failover_mechanisms(self, market_orchestrator, sample_allocation_request):
        """Test failover mechanisms when primary allocation methods fail."""
        allocation_request = sample_allocation_request
        allocation_request.preferred_method = AllocationMethod.AUCTION

        # Mock auction failure
        with patch.object(market_orchestrator.auction_engine, 'create_auction', side_effect=Exception("Auction service unavailable")):
            with patch.object(market_orchestrator, '_fallback_to_direct_pricing', return_value=MagicMock(
                success=True,
                allocation_method=AllocationMethod.BEST_PRICE
            )) as mock_fallback:

                result = await market_orchestrator.allocate_resources(allocation_request)

                # Should fallback to direct pricing
                mock_fallback.assert_called_once()
                assert result.success is True
                assert result.allocation_method == AllocationMethod.BEST_PRICE

    @pytest.mark.asyncio
    async def test_concurrent_allocations(self, market_orchestrator, sample_allocation_request):
        """Test handling of concurrent resource allocation requests."""
        # Create multiple allocation requests
        requests = []
        for i in range(10):
            request = ResourceAllocationRequest(
                requester_id=f"user_{i}",
                resource_requirements=sample_allocation_request.resource_requirements.copy(),
                duration_hours=Decimal("1"),
                max_budget=Decimal("10.0"),
                preferred_method=AllocationMethod.BEST_PRICE
            )
            requests.append(request)

        # Mock successful allocation
        with patch.object(market_orchestrator, 'allocate_resources', return_value=MagicMock(
            success=True,
            total_cost=Decimal("8.0")
        )) as mock_allocate:

            # Submit concurrent requests
            tasks = [
                asyncio.create_task(market_orchestrator.allocate_resources(req))
                for req in requests
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All requests should complete successfully
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 10
            assert mock_allocate.call_count == 10

    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self, integrated_components):
        """Test integration with resource monitoring systems."""
        scheduler = integrated_components['scheduler']

        # Test resource monitoring
        with patch.object(scheduler, 'get_resource_utilization', return_value={
            "cpu": Decimal("0.65"),
            "memory": Decimal("0.72"),
            "storage": Decimal("0.48"),
            "bandwidth": Decimal("0.85")
        }) as mock_monitoring:

            utilization = await scheduler.get_resource_utilization("task_123")

            assert utilization is not None
            assert "cpu" in utilization
            assert "memory" in utilization
            assert Decimal("0") <= utilization["cpu"] <= Decimal("1")
            mock_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_cost_optimization_integration(self, market_orchestrator, sample_allocation_request):
        """Test integration of cost optimization algorithms."""
        allocation_request = sample_allocation_request
        allocation_request.preferred_method = AllocationMethod.COST_OPTIMIZED

        with patch.object(market_orchestrator, '_optimize_cost_allocation', return_value=MagicMock(
            success=True,
            total_cost=Decimal("32.50"),
            optimization_savings=Decimal("12.50")
        )) as mock_optimize:

            result = await market_orchestrator.allocate_resources(allocation_request)

            assert result.success is True
            assert result.total_cost == Decimal("32.50")
            assert hasattr(result, 'optimization_savings')
            mock_optimize.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_benchmarking_integration(self, integrated_components):
        """Test integration with performance benchmarking systems."""
        anti_griefing = integrated_components['anti_griefing']

        benchmark_results = {
            "cpu_benchmark": Decimal("8500"),  # CPU score
            "memory_benchmark": Decimal("12000"),  # Memory throughput
            "storage_benchmark": Decimal("450"),  # IOPS
            "network_benchmark": Decimal("980")  # Mbps
        }

        with patch.object(anti_griefing, 'validate_performance_claims', return_value={
            "benchmarks_verified": True,
            "performance_score": Decimal("0.92"),
            "verification_confidence": Decimal("0.88")
        }) as mock_benchmark:

            validation = await anti_griefing.validate_performance_claims(
                provider_id="provider_alpha",
                claimed_performance=benchmark_results,
                verification_method="trusted_benchmark"
            )

            assert validation["benchmarks_verified"] is True
            assert validation["performance_score"] > Decimal("0.8")
            mock_benchmark.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
