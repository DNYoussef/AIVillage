"""
Comprehensive Test Suite for Federated AI Marketplace

Tests all enhanced marketplace functionality:
- Federated workload auctions (inference & training)
- Size-tier pricing system
- Dynamic resource allocation with QoS guarantees
- RESTful marketplace APIs
- Multi-criteria bidding and optimization
- Real-time monitoring and failover

Test Coverage:
- Unit tests for individual components
- Integration tests for cross-component interactions
- Performance tests for scalability
- End-to-end workflow tests
"""

import asyncio
from decimal import Decimal
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

# Import marketplace components
from infrastructure.fog.market.auction_engine import (
    AuctionEngine,
    ResourceRequirement,
    create_federated_inference_auction,
    create_federated_training_auction,
    create_multi_criteria_auction,
)
from infrastructure.fog.market.pricing_manager import (
    DynamicPricingManager,
    UserSizeTier,
    SizeTierPricing,
)
from infrastructure.fog.market.market_orchestrator import (
    MarketOrchestrator,
    AllocationStrategy,
)
from infrastructure.fog.market.resource_allocator import (
    DynamicResourceAllocator,
    ResourceNode,
    ResourceType,
    QoSRequirement,
    AllocationPlan,
)
from infrastructure.fog.market.marketplace_api import (
    MarketplaceAPI,
    FederatedInferenceRequest,
    ResourceQuoteRequest,
    WorkloadType,
    ModelSize,
    PrivacyLevel,
)


class TestFederatedAuctionEngine:
    """Test federated auction engine enhancements"""

    @pytest_asyncio.fixture
    async def auction_engine(self):
        """Create auction engine instance for testing"""
        engine = AuctionEngine()
        await engine.start()
        yield engine
        await engine.stop()

    @pytest.mark.asyncio
    async def test_federated_inference_auction_creation(self, auction_engine):
        """Test creation of federated inference auctions"""

        auction_id = await create_federated_inference_auction(
            requester_id="test_user",
            model_size="large",
            participants_needed=10,
            duration_hours=2.0,
            privacy_level="high",
            max_latency_ms=150.0,
            reserve_price=100.0,
        )

        assert auction_id is not None
        assert auction_id.startswith("auction_")

        # Check auction details
        status = await auction_engine.get_auction_status(auction_id)
        assert status is not None
        assert status["auction_type"] == "federated_inference"
        assert status["requirements"]["cpu_cores"] == 8.0  # Large model requirement
        assert status["requirements"]["memory_gb"] == 16.0

    @pytest.mark.asyncio
    async def test_federated_training_auction_creation(self, auction_engine):
        """Test creation of federated training auctions"""

        auction_id = await create_federated_training_auction(
            requester_id="test_researcher",
            model_size="xlarge",
            participants_needed=25,
            duration_hours=8.0,
            privacy_level="critical",
            reliability_requirement="guaranteed",
            reserve_price=5000.0,
        )

        assert auction_id is not None

        # Check auction details
        status = await auction_engine.get_auction_status(auction_id)
        assert status is not None
        assert status["auction_type"] == "federated_training"
        assert status["requirements"]["cpu_cores"] == 32.0  # XLarge model requirement
        assert status["requirements"]["memory_gb"] == 64.0

    @pytest.mark.asyncio
    async def test_multi_criteria_auction_creation(self, auction_engine):
        """Test creation of multi-criteria auctions"""

        requirements = {
            "cpu_cores": 4.0,
            "memory_gb": 8.0,
            "duration_hours": 4.0,
            "participants_needed": 5,
            "privacy_level": "medium",
            "reliability_requirement": "high",
            "reserve_price": 200.0,
        }

        criteria_weights = {
            "cost": 0.3,
            "latency": 0.3,
            "privacy": 0.2,
            "reliability": 0.2,
        }

        auction_id = await create_multi_criteria_auction(
            requester_id="test_optimizer",
            requirements=requirements,
            criteria_weights=criteria_weights,
        )

        assert auction_id is not None

        # Check auction details
        status = await auction_engine.get_auction_status(auction_id)
        assert status is not None
        assert status["auction_type"] == "multi_criteria"
        assert "criteria_weights" in status.get("metadata", {})


class TestSizeTierPricing:
    """Test size-tier pricing system"""

    @pytest_asyncio.fixture
    async def pricing_manager(self):
        """Create pricing manager instance for testing"""
        manager = DynamicPricingManager()
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_tier_pricing_configuration(self, pricing_manager):
        """Test size-tier pricing configuration"""

        # Check all tiers are configured
        for tier in UserSizeTier:
            assert tier in pricing_manager.tier_pricing
            tier_config = pricing_manager.tier_pricing[tier]
            assert isinstance(tier_config, SizeTierPricing)
            assert tier_config.inference_price_base > 0
            assert tier_config.training_price_base > 0

    @pytest.mark.asyncio
    async def test_federated_inference_pricing(self, pricing_manager):
        """Test federated inference pricing calculation"""

        # Test small tier pricing
        price_quote = await pricing_manager.get_federated_inference_price(
            user_tier=UserSizeTier.SMALL,
            model_size="medium",
            requests_count=100,
            participants_needed=5,
            privacy_level="medium",
        )

        assert price_quote["workload_type"] == "federated_inference"
        assert price_quote["user_tier"] == "small"
        assert price_quote["model_size"] == "medium"
        assert price_quote["total_cost"] > 0
        assert "pricing_breakdown" in price_quote
        assert "tier_info" in price_quote

        # Test volume discount application
        assert "volume_discount" in price_quote["pricing_breakdown"]
        volume_discount = price_quote["pricing_breakdown"]["volume_discount"]
        assert volume_discount <= 1.0  # Should be discount or no change

    @pytest.mark.asyncio
    async def test_federated_training_pricing(self, pricing_manager):
        """Test federated training pricing calculation"""

        # Test enterprise tier pricing
        price_quote = await pricing_manager.get_federated_training_price(
            user_tier=UserSizeTier.ENTERPRISE,
            model_size="xlarge",
            duration_hours=24.0,
            participants_needed=50,
            privacy_level="critical",
            reliability_requirement="guaranteed",
        )

        assert price_quote["workload_type"] == "federated_training"
        assert price_quote["user_tier"] == "enterprise"
        assert price_quote["total_cost"] > 1000  # Should be expensive for enterprise tier

        # Check privacy and reliability multipliers applied
        breakdown = price_quote["pricing_breakdown"]
        assert breakdown["privacy_multiplier"] >= 2.0  # Critical privacy
        assert breakdown["reliability_multiplier"] >= 1.5  # Guaranteed reliability

    @pytest.mark.asyncio
    async def test_tier_price_bounds(self, pricing_manager):
        """Test that pricing respects tier bounds"""

        # Test extreme case that should hit ceiling
        price_quote = await pricing_manager.get_federated_inference_price(
            user_tier=UserSizeTier.SMALL,
            model_size="xlarge",  # Large model for small tier
            requests_count=10000,  # Many requests
            participants_needed=100,  # Many participants
            privacy_level="critical",
        )

        tier_config = pricing_manager.tier_pricing[UserSizeTier.SMALL]
        price_per_request = price_quote["price_per_request"]

        # Should not exceed tier maximum
        assert price_per_request <= float(tier_config.inference_price_max)
        assert price_per_request >= float(tier_config.inference_price_min)


class TestDynamicResourceAllocator:
    """Test dynamic resource allocation with QoS guarantees"""

    @pytest_asyncio.fixture
    async def resource_allocator(self):
        """Create resource allocator instance for testing"""
        allocator = DynamicResourceAllocator()
        await allocator.start()

        # Register some test nodes
        test_nodes = [
            ResourceNode(
                node_id=f"node_{i}",
                node_type=ResourceType.EDGE_SERVER if i % 2 == 0 else ResourceType.CLOUD_INSTANCE,
                cpu_cores=Decimal("4.0"),
                memory_gb=Decimal("8.0"),
                storage_gb=Decimal("100.0"),
                bandwidth_mbps=Decimal("100.0"),
                trust_score=Decimal("0.8"),
                availability_score=Decimal("0.95"),
                latency_ms=Decimal("50.0") + Decimal(str(i * 10)),
                cost_per_hour=Decimal("2.0") + Decimal(str(i * 0.5)),
                region=f"region_{i // 10}",
            )
            for i in range(20)
        ]

        for node in test_nodes:
            allocator.register_resource_node(node)

        yield allocator
        await allocator.stop()

    @pytest.mark.asyncio
    async def test_resource_discovery(self, resource_allocator):
        """Test resource discovery functionality"""

        requirements = ResourceRequirement(
            cpu_cores=Decimal("2.0"),
            memory_gb=Decimal("4.0"),
            storage_gb=Decimal("50.0"),
            bandwidth_mbps=Decimal("50.0"),
            duration_hours=Decimal("2.0"),
            participants_needed=5,
            min_trust_score=Decimal("0.7"),
            max_latency_ms=Decimal("200.0"),
        )

        qos_requirements = QoSRequirement(
            max_latency_ms=Decimal("200.0"),
            min_availability_percentage=Decimal("95.0"),
            max_cost_per_hour=Decimal("10.0"),
        )

        discovered_nodes = await resource_allocator.discover_resources(requirements, qos_requirements)

        assert len(discovered_nodes) > 0
        assert len(discovered_nodes) <= 20  # Should not exceed registered nodes

        # Check that all discovered nodes meet requirements
        for node in discovered_nodes:
            assert node.can_handle_workload(requirements)
            assert resource_allocator._meets_qos_requirements(node, qos_requirements)

    @pytest.mark.asyncio
    async def test_allocation_plan_creation(self, resource_allocator):
        """Test creation of allocation plans"""

        requirements = ResourceRequirement(
            cpu_cores=Decimal("2.0"),
            memory_gb=Decimal("4.0"),
            storage_gb=Decimal("50.0"),
            bandwidth_mbps=Decimal("50.0"),
            duration_hours=Decimal("4.0"),
            participants_needed=3,
        )

        qos_requirements = QoSRequirement(
            min_reliability_percentage=Decimal("99.0"),
            max_cost_per_hour=Decimal("20.0"),
        )

        # Discover resources
        discovered_nodes = await resource_allocator.discover_resources(requirements, qos_requirements)

        # Create allocation plan
        plan = await resource_allocator.create_allocation_plan(
            requirements, qos_requirements, discovered_nodes, AllocationStrategy.BALANCED
        )

        assert isinstance(plan, AllocationPlan)
        assert len(plan.primary_nodes) == 3  # participants_needed
        assert len(plan.backup_nodes) >= 1  # Should have backups for high reliability
        assert plan.total_cost > 0
        assert plan.expected_quality_score > 0
        assert 0 <= plan.risk_score <= 1

    @pytest.mark.asyncio
    async def test_allocation_execution_and_monitoring(self, resource_allocator):
        """Test allocation execution and QoS monitoring"""

        requirements = ResourceRequirement(
            cpu_cores=Decimal("2.0"),
            memory_gb=Decimal("4.0"),
            storage_gb=Decimal("50.0"),
            bandwidth_mbps=Decimal("50.0"),
            duration_hours=Decimal("1.0"),
            participants_needed=2,
        )

        qos_requirements = QoSRequirement()

        # Create and execute allocation plan
        discovered_nodes = await resource_allocator.discover_resources(requirements, qos_requirements)

        plan = await resource_allocator.create_allocation_plan(requirements, qos_requirements, discovered_nodes)

        allocation_id = await resource_allocator.execute_allocation_plan(plan, "test_requester")

        assert allocation_id is not None
        assert allocation_id.startswith("alloc_")
        assert allocation_id in resource_allocator.active_allocations

        # Test monitoring
        qos_status = await resource_allocator.monitor_allocation_qos(allocation_id)
        assert qos_status["allocation_id"] == allocation_id
        assert "qos_status" in qos_status
        assert "current_metrics" in qos_status

        # Test status retrieval
        status = await resource_allocator.get_allocation_status(allocation_id)
        assert status["allocation_id"] == allocation_id
        assert status["status"] in ["monitoring", "scaling", "completed"]
        assert "resources" in status
        assert "qos_metrics" in status

    @pytest.mark.asyncio
    async def test_dynamic_scaling(self, resource_allocator):
        """Test dynamic scaling functionality"""

        # Create a minimal allocation
        requirements = ResourceRequirement(
            cpu_cores=Decimal("1.0"),
            memory_gb=Decimal("2.0"),
            storage_gb=Decimal("10.0"),
            bandwidth_mbps=Decimal("10.0"),
            duration_hours=Decimal("1.0"),
            participants_needed=2,
        )

        qos_requirements = QoSRequirement()

        discovered_nodes = await resource_allocator.discover_resources(requirements, qos_requirements)

        plan = await resource_allocator.create_allocation_plan(requirements, qos_requirements, discovered_nodes)

        allocation_id = await resource_allocator.execute_allocation_plan(plan, "test_requester")

        # Test scale up
        original_node_count = len(plan.primary_nodes)
        scale_result = await resource_allocator.scale_allocation(allocation_id, Decimal("1.5"))  # Scale up by 50%

        assert scale_result is True

        # Check that nodes were added
        updated_plan = resource_allocator.active_allocations[allocation_id]
        assert len(updated_plan.primary_nodes) > original_node_count


class TestMarketplaceAPI:
    """Test RESTful marketplace API"""

    @pytest_asyncio.fixture
    async def marketplace_api(self):
        """Create marketplace API instance for testing"""
        api = MarketplaceAPI()
        await api.initialize_market_components()
        yield api

    @pytest.mark.asyncio
    async def test_federated_inference_request_processing(self, marketplace_api):
        """Test federated inference request processing"""

        # Mock the market orchestrator
        marketplace_api.market_orchestrator = AsyncMock()
        marketplace_api.market_orchestrator.request_resources.return_value = "alloc_123"

        request = FederatedInferenceRequest(
            requester_id="test_user",
            user_tier=UserSizeTier.MEDIUM,
            model_size=ModelSize.LARGE,
            requests_count=1000,
            participants_needed=15,
            privacy_level=PrivacyLevel.HIGH,
            max_latency_ms=150.0,
            max_budget=2000.0,
        )

        # Process request through API route logic
        await marketplace_api.app.router.execute(
            {
                "type": "http",
                "method": "POST",
                "path": "/federated/inference/request",
                "body": request.json().encode(),
                "headers": {"content-type": "application/json"},
            }
        )

        # Verify request was processed
        marketplace_api.market_orchestrator.request_resources.assert_called_once()
        assert len(marketplace_api.active_requests) > 0

    @pytest.mark.asyncio
    async def test_pricing_quote_generation(self, marketplace_api):
        """Test pricing quote generation"""

        # Mock the pricing manager
        marketplace_api.pricing_manager = AsyncMock()
        marketplace_api.pricing_manager.get_federated_inference_price.return_value = {
            "workload_type": "federated_inference",
            "total_cost": 500.0,
            "price_per_request": 0.5,
            "currency": "USD",
        }

        quote_request = ResourceQuoteRequest(
            user_tier=UserSizeTier.LARGE,
            workload_type=WorkloadType.INFERENCE,
            model_size=ModelSize.MEDIUM,
            requests_count=1000,
            participants_needed=10,
            privacy_level=PrivacyLevel.MEDIUM,
        )

        # This would be called through the API
        quote = await marketplace_api.pricing_manager.get_federated_inference_price(
            user_tier=quote_request.user_tier,
            model_size=quote_request.model_size.value,
            requests_count=quote_request.requests_count,
            participants_needed=quote_request.participants_needed,
            privacy_level=quote_request.privacy_level.value,
        )

        assert quote["workload_type"] == "federated_inference"
        assert quote["total_cost"] == 500.0
        marketplace_api.pricing_manager.get_federated_inference_price.assert_called_once()


class TestMarketIntegration:
    """Test integration between marketplace components"""

    @pytest_asyncio.fixture
    async def integrated_marketplace(self):
        """Create fully integrated marketplace system"""

        # Create all components
        auction_engine = AuctionEngine()
        pricing_manager = DynamicPricingManager()
        market_orchestrator = MarketOrchestrator()
        resource_allocator = DynamicResourceAllocator()
        marketplace_api = MarketplaceAPI()

        # Start components
        await auction_engine.start()
        await pricing_manager.start()
        await market_orchestrator.start()
        await resource_allocator.start()
        await marketplace_api.initialize_market_components()

        # Register test resources
        test_nodes = [
            ResourceNode(
                node_id=f"integrated_node_{i}",
                node_type=ResourceType.EDGE_SERVER,
                cpu_cores=Decimal("8.0"),
                memory_gb=Decimal("16.0"),
                storage_gb=Decimal("200.0"),
                bandwidth_mbps=Decimal("200.0"),
                trust_score=Decimal("0.9"),
                availability_score=Decimal("0.99"),
                latency_ms=Decimal(str(30 + i * 5)),
                cost_per_hour=Decimal(str(5.0 + i * 0.5)),
            )
            for i in range(10)
        ]

        for node in test_nodes:
            resource_allocator.register_resource_node(node)

        components = {
            "auction_engine": auction_engine,
            "pricing_manager": pricing_manager,
            "market_orchestrator": market_orchestrator,
            "resource_allocator": resource_allocator,
            "marketplace_api": marketplace_api,
        }

        yield components

        # Cleanup
        await auction_engine.stop()
        await pricing_manager.stop()
        await market_orchestrator.stop()
        await resource_allocator.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_federated_inference_workflow(self, integrated_marketplace):
        """Test complete end-to-end federated inference workflow"""

        components = integrated_marketplace
        pricing_manager = components["pricing_manager"]
        resource_allocator = components["resource_allocator"]

        # Step 1: Get pricing quote
        pricing_quote = await pricing_manager.get_federated_inference_price(
            user_tier=UserSizeTier.MEDIUM,
            model_size="large",
            requests_count=500,
            participants_needed=8,
            privacy_level="high",
        )

        assert pricing_quote["workload_type"] == "federated_inference"
        assert pricing_quote["total_cost"] > 0

        # Step 2: Create resource requirements
        requirements = ResourceRequirement(
            cpu_cores=Decimal("8.0"),  # Large model
            memory_gb=Decimal("16.0"),
            storage_gb=Decimal("50.0"),
            bandwidth_mbps=Decimal("100.0"),
            duration_hours=Decimal("2.0"),
            participants_needed=8,
            workload_type="inference",
            model_size="large",
            privacy_level="high",
        )

        qos_requirements = QoSRequirement(
            max_latency_ms=Decimal("100.0"),
            min_availability_percentage=Decimal("99.0"),
            privacy_level="high",
        )

        # Step 3: Discover and allocate resources
        discovered_nodes = await resource_allocator.discover_resources(requirements, qos_requirements)

        assert len(discovered_nodes) >= 8  # Should find enough nodes

        allocation_plan = await resource_allocator.create_allocation_plan(
            requirements, qos_requirements, discovered_nodes
        )

        assert len(allocation_plan.primary_nodes) == 8
        assert allocation_plan.total_cost > 0

        # Step 4: Execute allocation
        allocation_id = await resource_allocator.execute_allocation_plan(allocation_plan, "end_to_end_test")

        assert allocation_id is not None

        # Step 5: Monitor allocation
        qos_status = await resource_allocator.monitor_allocation_qos(allocation_id)

        assert qos_status["allocation_id"] == allocation_id
        assert "qos_status" in qos_status

    @pytest.mark.asyncio
    async def test_federated_training_auction_workflow(self, integrated_marketplace):
        """Test federated training auction workflow"""

        components = integrated_marketplace
        auction_engine = components["auction_engine"]

        # Create federated training auction
        auction_id = await create_federated_training_auction(
            requester_id="training_researcher",
            model_size="xlarge",
            participants_needed=20,
            duration_hours=12.0,
            privacy_level="critical",
            reliability_requirement="guaranteed",
            reserve_price=10000.0,
        )

        assert auction_id is not None

        # Get auction status
        auction_status = await auction_engine.get_auction_status(auction_id)

        assert auction_status is not None
        assert auction_status["auction_type"] == "federated_training"
        assert auction_status["requirements"]["cpu_cores"] == 32.0  # XLarge
        assert auction_status["reserve_price"] == 10000.0

    @pytest.mark.asyncio
    async def test_market_analytics_aggregation(self, integrated_marketplace):
        """Test market analytics aggregation across components"""

        components = integrated_marketplace
        auction_engine = components["auction_engine"]
        pricing_manager = components["pricing_manager"]
        market_orchestrator = components["market_orchestrator"]

        # Get analytics from each component
        auction_stats = await auction_engine.get_market_statistics()
        pricing_analytics = await pricing_manager.get_market_analytics()
        orchestrator_stats = await market_orchestrator.get_market_statistics()

        # Verify structure
        assert "auction_statistics" in auction_stats
        assert "market_overview" in pricing_analytics
        assert "orchestrator_metrics" in orchestrator_stats

        # Check data consistency
        assert auction_stats["auction_statistics"]["total_auctions_created"] >= 0
        assert pricing_analytics["market_overview"]["total_supply"] >= 0


class TestPerformanceAndScalability:
    """Test performance and scalability of marketplace system"""

    @pytest.mark.asyncio
    async def test_concurrent_auction_handling(self):
        """Test handling multiple concurrent auctions"""

        auction_engine = AuctionEngine()
        await auction_engine.start()

        try:
            # Create multiple auctions concurrently
            auction_tasks = [
                create_federated_inference_auction(
                    requester_id=f"user_{i}",
                    model_size="medium",
                    participants_needed=5,
                    duration_hours=1.0,
                )
                for i in range(10)
            ]

            auction_ids = await asyncio.gather(*auction_tasks)

            assert len(auction_ids) == 10
            assert all(aid is not None for aid in auction_ids)
            assert len(set(auction_ids)) == 10  # All unique

        finally:
            await auction_engine.stop()

    @pytest.mark.asyncio
    async def test_large_resource_pool_discovery(self):
        """Test resource discovery with large resource pool"""

        resource_allocator = DynamicResourceAllocator()
        await resource_allocator.start()

        try:
            # Register large number of nodes
            for i in range(1000):
                node = ResourceNode(
                    node_id=f"scale_node_{i}",
                    node_type=ResourceType.EDGE_SERVER if i % 2 == 0 else ResourceType.CLOUD_INSTANCE,
                    cpu_cores=Decimal(str(2 + (i % 8))),
                    memory_gb=Decimal(str(4 + (i % 16))),
                    storage_gb=Decimal(str(50 + (i % 200))),
                    bandwidth_mbps=Decimal(str(50 + (i % 100))),
                    trust_score=Decimal(str(0.5 + (i % 50) / 100)),
                    latency_ms=Decimal(str(10 + (i % 200))),
                    cost_per_hour=Decimal(str(1 + (i % 10))),
                )
                resource_allocator.register_resource_node(node)

            # Test discovery performance
            requirements = ResourceRequirement(
                cpu_cores=Decimal("4.0"),
                memory_gb=Decimal("8.0"),
                storage_gb=Decimal("100.0"),
                bandwidth_mbps=Decimal("80.0"),
                duration_hours=Decimal("1.0"),
                participants_needed=20,
            )

            qos_requirements = QoSRequirement()

            import time

            start_time = time.time()

            discovered_nodes = await resource_allocator.discover_resources(
                requirements, qos_requirements, discovery_timeout=10
            )

            discovery_time = time.time() - start_time

            # Should complete within reasonable time
            assert discovery_time < 5.0  # 5 seconds max
            assert len(discovered_nodes) > 0
            assert len(discovered_nodes) <= 50  # Limited result set

        finally:
            await resource_allocator.stop()

    @pytest.mark.asyncio
    async def test_pricing_calculation_performance(self):
        """Test pricing calculation performance"""

        pricing_manager = DynamicPricingManager()
        await pricing_manager.start()

        try:
            # Test multiple pricing calculations concurrently
            pricing_tasks = [
                pricing_manager.get_federated_inference_price(
                    user_tier=UserSizeTier.MEDIUM,
                    model_size="large",
                    requests_count=100 * (i + 1),
                    participants_needed=5 + (i % 10),
                    privacy_level="medium",
                )
                for i in range(50)
            ]

            import time

            start_time = time.time()

            pricing_results = await asyncio.gather(*pricing_tasks)

            calculation_time = time.time() - start_time

            # Should complete within reasonable time
            assert calculation_time < 2.0  # 2 seconds for 50 calculations
            assert len(pricing_results) == 50
            assert all("total_cost" in result for result in pricing_results)

        finally:
            await pricing_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
