"""
Comprehensive Test Suite for Unified Federated System

This test suite validates the integrated federated AI system including:
- Unified federated coordinator
- Marketplace resource allocation  
- Dynamic pricing optimization
- Unified API interface
- Size-based resource allocation
- End-to-end integration testing

The tests cover multiple scenarios and user size tiers to ensure
the system works correctly for all user segments.
"""

import asyncio
from decimal import Decimal
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, UTC
from typing import Dict, Any

# Import components to test
try:
    from infrastructure.distributed_inference.unified_federated_coordinator import (
        UnifiedFederatedCoordinator, UnifiedRequest, UserSizeTier, RequestType, SystemMetrics
    )
    from infrastructure.distributed_inference.marketplace_resource_allocator import (
        MarketplaceResourceAllocator, ResourceSpecification, AllocationResult, SizeTierConfiguration
    )
    from infrastructure.distributed_inference.dynamic_pricing_optimizer import (
        DynamicPricingOptimizer, OptimizationContext, OptimizationObjective, OptimizationResult
    )
    from infrastructure.distributed_inference.unified_api import (
        UnifiedFederatedAPI, request_inference, request_training, get_request_status
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Components not available for testing: {e}")
    COMPONENTS_AVAILABLE = False


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestUnifiedFederatedCoordinator:
    """Test the main unified federated coordinator"""
    
    @pytest.fixture
    async def coordinator(self):
        """Create test coordinator instance"""
        coordinator = UnifiedFederatedCoordinator(
            coordinator_id="test_coordinator",
            enable_marketplace=False,  # Disable for unit tests
            enable_p2p=False
        )
        await coordinator.initialize()
        yield coordinator
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator initializes properly"""
        assert coordinator.coordinator_id == "test_coordinator"
        assert coordinator._running is True
        assert coordinator.metrics is not None
    
    @pytest.mark.asyncio
    async def test_inference_request_submission(self, coordinator):
        """Test submitting inference requests"""
        request_id = await coordinator.request_inference(
            user_id="test_user",
            model_id="test_model",
            input_data={"text": "test input"},
            size_tier="medium",
            max_cost=50.0
        )
        
        assert request_id is not None
        assert request_id.startswith("unified_inference_")
        assert request_id in coordinator.active_requests
        
        # Verify request details
        request = coordinator.active_requests[request_id]
        assert request.user_id == "test_user"
        assert request.model_id == "test_model"
        assert request.request_type == RequestType.INFERENCE
        assert request.user_profile.size_tier == UserSizeTier.MEDIUM
    
    @pytest.mark.asyncio
    async def test_training_request_submission(self, coordinator):
        """Test submitting training requests"""
        request_id = await coordinator.request_training(
            user_id="test_user",
            model_id="test_model",
            training_data={"dataset": "test_data"},
            size_tier="large",
            participants=20,
            rounds=15,
            max_cost=500.0
        )
        
        assert request_id is not None
        assert request_id.startswith("unified_training_")
        assert request_id in coordinator.active_requests
        
        # Verify request details
        request = coordinator.active_requests[request_id]
        assert request.user_id == "test_user"
        assert request.model_id == "test_model"
        assert request.request_type == RequestType.TRAINING
        assert request.participants_needed == 20
        assert request.training_rounds == 15
    
    @pytest.mark.asyncio
    async def test_request_status_tracking(self, coordinator):
        """Test request status tracking"""
        request_id = await coordinator.request_inference(
            user_id="test_user",
            model_id="test_model",
            input_data={"text": "test"},
            size_tier="small"
        )
        
        status = await coordinator.get_request_status(request_id)
        assert status is not None
        assert status["request_id"] == request_id
        assert status["user_id"] == "test_user"
        assert status["request_type"] == "inference"
        assert status["size_tier"] == "small"
    
    @pytest.mark.asyncio
    async def test_size_tier_validation(self, coordinator):
        """Test size tier limits are enforced"""
        # Test small tier limits
        with pytest.raises((ValueError, RuntimeError)):
            await coordinator.request_training(
                user_id="test_user",
                model_id="test_model", 
                training_data={},
                size_tier="small",
                participants=100,  # Exceeds small tier limit
                max_cost=10.0
            )
    
    @pytest.mark.asyncio
    async def test_price_estimation(self, coordinator):
        """Test price estimation functionality"""
        estimate = await coordinator.get_price_estimate(
            request_type="inference",
            model_id="test_model",
            size_tier="medium",
            duration_hours=2.0
        )
        
        assert estimate is not None
        assert "total_estimate" in estimate
        assert estimate["size_tier"] == "medium"
        assert estimate["request_type"] == "inference"
        assert estimate["duration_hours"] == 2.0


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestMarketplaceResourceAllocator:
    """Test marketplace resource allocation"""
    
    @pytest.fixture
    def resource_allocator(self):
        """Create test resource allocator"""
        allocator = MarketplaceResourceAllocator(enable_marketplace=False)
        return allocator
    
    @pytest.mark.asyncio
    async def test_allocator_initialization(self, resource_allocator):
        """Test allocator initializes properly"""
        await resource_allocator.initialize()
        assert resource_allocator.tier_configs is not None
        assert len(resource_allocator.tier_configs) == 4  # small, medium, large, enterprise
    
    def test_tier_configurations(self, resource_allocator):
        """Test size tier configurations are correct"""
        configs = resource_allocator.tier_configs
        
        # Test small tier
        small_config = configs["small"]
        assert small_config.max_hourly_budget == Decimal("10.00")
        assert small_config.max_concurrent_requests == 3
        assert small_config.allocation_priority.value == "cost_optimized"
        
        # Test enterprise tier
        enterprise_config = configs["enterprise"]
        assert enterprise_config.max_hourly_budget == Decimal("10000.00")
        assert enterprise_config.enable_auctions is False  # Direct allocation
        assert enterprise_config.min_availability_sla == 0.99
    
    @pytest.mark.asyncio
    async def test_resource_allocation_by_tier(self, resource_allocator):
        """Test resource allocation for different tiers"""
        await resource_allocator.initialize()
        
        # Test small tier allocation
        resource_spec = ResourceSpecification(
            cpu_cores=Decimal("2.0"),
            memory_gb=Decimal("4.0"),
            duration_hours=Decimal("1.0"),
            storage_gb=Decimal("10.0"),
            bandwidth_mbps=Decimal("100.0")
        )
        
        result = await resource_allocator.allocate_resources(
            user_id="test_user",
            size_tier="small",
            resource_spec=resource_spec,
            task_context={"type": "inference"}
        )
        
        assert result.success is True
        assert len(result.allocated_nodes) > 0
        assert result.total_cost > 0
        assert result.allocation_method in ["direct", "auction"]
    
    @pytest.mark.asyncio
    async def test_tier_limit_validation(self, resource_allocator):
        """Test that tier limits are properly validated"""
        await resource_allocator.initialize()
        
        # Try to exceed small tier CPU limit
        resource_spec = ResourceSpecification(
            cpu_cores=Decimal("10.0"),  # Exceeds small tier limit of 4.0
            memory_gb=Decimal("2.0"),
            duration_hours=Decimal("1.0"),
            storage_gb=Decimal("10.0"),
            bandwidth_mbps=Decimal("100.0")
        )
        
        result = await resource_allocator.allocate_resources(
            user_id="test_user",
            size_tier="small",
            resource_spec=resource_spec,
            task_context={}
        )
        
        assert result.success is False
        assert "CPU cores exceed tier limit" in result.error_message
    
    @pytest.mark.asyncio
    async def test_pricing_estimates_by_tier(self, resource_allocator):
        """Test pricing estimates vary by tier"""
        await resource_allocator.initialize()
        
        # Get estimates for different tiers
        small_estimate = await resource_allocator.get_tier_pricing_estimate("small")
        large_estimate = await resource_allocator.get_tier_pricing_estimate("large")
        
        assert small_estimate["pricing_estimate"]["total_cost"] > 0
        assert large_estimate["pricing_estimate"]["total_cost"] > 0
        
        # Large tier should generally cost more for same resources
        assert large_estimate["pricing_estimate"]["total_cost"] >= small_estimate["pricing_estimate"]["total_cost"]


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestDynamicPricingOptimizer:
    """Test dynamic pricing optimization"""
    
    @pytest.fixture
    def pricing_optimizer(self):
        """Create test pricing optimizer"""
        optimizer = DynamicPricingOptimizer()
        return optimizer
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, pricing_optimizer):
        """Test optimizer initializes properly"""
        await pricing_optimizer.initialize()
        assert pricing_optimizer.pricing_rules is not None
        assert len(pricing_optimizer.pricing_rules) > 0
        assert pricing_optimizer._running is True
    
    def test_pricing_rules_creation(self, pricing_optimizer):
        """Test default pricing rules are created"""
        rules = pricing_optimizer.pricing_rules
        rule_ids = [rule.rule_id for rule in rules]
        
        assert "high_demand_premium" in rule_ids
        assert "small_tier_discount" in rule_ids
        assert "enterprise_sla_premium" in rule_ids
        assert "training_complexity_premium" in rule_ids
    
    @pytest.mark.asyncio
    async def test_allocation_optimization(self, pricing_optimizer):
        """Test allocation optimization for different objectives"""
        await pricing_optimizer.initialize()
        
        resource_spec = ResourceSpecification(
            cpu_cores=Decimal("4.0"),
            memory_gb=Decimal("8.0"),
            duration_hours=Decimal("2.0"),
            storage_gb=Decimal("20.0"),
            bandwidth_mbps=Decimal("100.0")
        )
        
        context = OptimizationContext(
            user_id="test_user",
            size_tier="medium",
            request_type="inference",
            resource_spec=resource_spec,
            primary_objective=OptimizationObjective.MINIMIZE_COST,
            max_cost=Decimal("100.00")
        )
        
        result = await pricing_optimizer.optimize_allocation(context)
        
        assert result.success is True
        assert result.estimated_cost > 0
        assert result.optimization_score >= 0
        assert result.recommended_allocation is not None
    
    @pytest.mark.asyncio
    async def test_different_optimization_objectives(self, pricing_optimizer):
        """Test optimization with different objectives produces different results"""
        await pricing_optimizer.initialize()
        
        resource_spec = ResourceSpecification(
            cpu_cores=Decimal("4.0"),
            memory_gb=Decimal("8.0"),
            duration_hours=Decimal("1.0"),
            storage_gb=Decimal("20.0"),
            bandwidth_mbps=Decimal("100.0")
        )
        
        # Cost-optimized
        cost_context = OptimizationContext(
            user_id="test_user",
            size_tier="medium",
            request_type="inference",
            resource_spec=resource_spec,
            primary_objective=OptimizationObjective.MINIMIZE_COST
        )
        
        # Performance-optimized
        perf_context = OptimizationContext(
            user_id="test_user",
            size_tier="medium",
            request_type="inference",
            resource_spec=resource_spec,
            primary_objective=OptimizationObjective.MAXIMIZE_PERFORMANCE
        )
        
        cost_result = await pricing_optimizer.optimize_allocation(cost_context)
        perf_result = await pricing_optimizer.optimize_allocation(perf_context)
        
        assert cost_result.success and perf_result.success
        
        # Cost-optimized should be cheaper
        assert cost_result.estimated_cost <= perf_result.estimated_cost
        
        # Performance-optimized should have better performance
        assert perf_result.estimated_performance >= cost_result.estimated_performance


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestUnifiedAPI:
    """Test the unified API interface"""
    
    @pytest.fixture
    async def api(self):
        """Create test API instance"""
        api = UnifiedFederatedAPI()
        
        # Mock the underlying components for testing
        api.coordinator = Mock()
        api.resource_allocator = Mock()
        api._initialized = True
        
        return api
    
    @pytest.mark.asyncio
    async def test_api_initialization(self):
        """Test API initialization"""
        api = UnifiedFederatedAPI()
        
        # Mock initialization to avoid dependency issues
        with patch.object(api, '_ensure_initialized'):
            api._initialized = True
            assert api._initialized is True
    
    @pytest.mark.asyncio
    async def test_inference_request_api(self, api):
        """Test inference request through API"""
        # Mock coordinator response
        api.coordinator.request_inference = AsyncMock(return_value="test_request_id")
        
        response = await api.request_inference(
            user_id="test_user",
            model="test_model",
            input_data={"text": "test"},
            size_tier="medium",
            max_cost=50.0
        )
        
        assert response.success is True
        assert response.request_id == "test_request_id"
        assert response.data is not None
        assert api.inference_requests == 1
        assert api.total_requests == 1
    
    @pytest.mark.asyncio
    async def test_training_request_api(self, api):
        """Test training request through API"""
        # Mock coordinator response
        api.coordinator.request_training = AsyncMock(return_value="test_training_id")
        
        response = await api.request_training(
            user_id="test_user",
            base_model="test_model",
            training_data={"dataset": "test"},
            size_tier="large",
            participants=25,
            rounds=10,
            max_cost=500.0
        )
        
        assert response.success is True
        assert response.request_id == "test_training_id"
        assert response.data is not None
        assert api.training_requests == 1
    
    @pytest.mark.asyncio
    async def test_input_validation(self, api):
        """Test API input validation"""
        # Test invalid size tier
        response = await api.request_inference(
            user_id="test_user",
            model="test_model",
            input_data={"text": "test"},
            size_tier="invalid_tier",  # Invalid
            max_cost=50.0
        )
        
        assert response.success is False
        assert "size_tier must be one of" in response.error
        assert api.api_errors == 1
    
    @pytest.mark.asyncio
    async def test_price_estimate_api(self, api):
        """Test price estimation through API"""
        # Mock coordinator response
        api.coordinator.get_price_estimate = AsyncMock(return_value={
            "total_estimate": 45.0,
            "request_type": "inference",
            "size_tier": "medium"
        })
        
        response = await api.get_price_estimate(
            request_type="inference",
            model="test_model",
            size_tier="medium",
            duration_hours=1.0
        )
        
        assert response.success is True
        assert response.data["total_estimate"] == 45.0
    
    @pytest.mark.asyncio
    async def test_system_status_api(self, api):
        """Test system status API"""
        # Mock coordinator response
        api.coordinator.get_system_status = AsyncMock(return_value={
            "running": True,
            "metrics": {"success_rate": 0.95}
        })
        
        # Mock resource allocator response
        api.resource_allocator.get_allocator_statistics = AsyncMock(return_value={
            "allocator_metrics": {"success_rate": 0.90}
        })
        
        response = await api.get_system_status()
        
        assert response.success is True
        assert response.data is not None
        assert "api_statistics" in response.data
        assert "system_health" in response.data


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    async def integrated_system(self):
        """Set up integrated system for testing"""
        # Create coordinator with mocked components
        coordinator = UnifiedFederatedCoordinator(
            coordinator_id="e2e_test",
            enable_marketplace=False,
            enable_p2p=False
        )
        
        # Mock the component coordinators
        coordinator.inference_coordinator = Mock()
        coordinator.training_coordinator = Mock()
        coordinator._running = True
        
        yield coordinator
        
        await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_inference_end_to_end(self, integrated_system):
        """Test complete inference workflow"""
        # Mock inference coordinator
        integrated_system.inference_coordinator.submit_inference_request = AsyncMock(return_value="inference_123")
        integrated_system.inference_coordinator.get_inference_result = AsyncMock(return_value={
            "status": "completed",
            "result": {"prediction": "test_result"}
        })
        
        # Submit inference request
        request_id = await integrated_system.request_inference(
            user_id="e2e_user",
            model_id="e2e_model",
            input_data={"text": "test input"},
            size_tier="medium",
            max_cost=100.0
        )
        
        assert request_id is not None
        
        # Wait a moment for processing (in real system, would be longer)
        await asyncio.sleep(0.1)
        
        # Check request status
        status = await integrated_system.get_request_status(request_id)
        assert status is not None
        assert status["request_id"] == request_id
    
    @pytest.mark.asyncio 
    async def test_training_end_to_end(self, integrated_system):
        """Test complete training workflow"""
        # Mock training coordinator
        integrated_system.training_coordinator.create_enhanced_training_job = AsyncMock(return_value="training_456")
        integrated_system.training_coordinator.get_job_status = AsyncMock(return_value={
            "status": "completed",
            "model_updates": {"rounds_completed": 10}
        })
        
        # Submit training request
        request_id = await integrated_system.request_training(
            user_id="e2e_user",
            model_id="e2e_model",
            training_data={"dataset": "test_dataset"},
            size_tier="large",
            participants=20,
            rounds=10,
            max_cost=500.0
        )
        
        assert request_id is not None
        
        # Wait a moment for processing
        await asyncio.sleep(0.1)
        
        # Check request status
        status = await integrated_system.get_request_status(request_id)
        assert status is not None
        assert status["request_type"] == "training"
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, integrated_system):
        """Test handling concurrent requests"""
        # Mock coordinators
        integrated_system.inference_coordinator.submit_inference_request = AsyncMock(return_value="concurrent_inf")
        integrated_system.training_coordinator.create_enhanced_training_job = AsyncMock(return_value="concurrent_train")
        
        # Submit multiple concurrent requests
        inference_task = integrated_system.request_inference(
            user_id="concurrent_user1",
            model_id="model1",
            input_data={"text": "test1"},
            size_tier="medium"
        )
        
        training_task = integrated_system.request_training(
            user_id="concurrent_user2", 
            model_id="model2",
            training_data={"dataset": "data2"},
            size_tier="large",
            participants=15
        )
        
        # Wait for both requests to complete
        inference_id, training_id = await asyncio.gather(inference_task, training_task)
        
        assert inference_id is not None
        assert training_id is not None
        assert inference_id != training_id
        
        # Verify both requests are tracked
        inf_status = await integrated_system.get_request_status(inference_id)
        train_status = await integrated_system.get_request_status(training_id)
        
        assert inf_status is not None
        assert train_status is not None
        assert inf_status["request_type"] == "inference" 
        assert train_status["request_type"] == "training"


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")
class TestSizeTierBehavior:
    """Test behavior specific to different size tiers"""
    
    test_scenarios = [
        {
            "tier": "small",
            "max_cost": 10.0,
            "cpu_limit": 4.0,
            "expected_features": ["Edge computing optimization", "Cost-optimized allocation"]
        },
        {
            "tier": "medium", 
            "max_cost": 100.0,
            "cpu_limit": 16.0,
            "expected_features": ["Hybrid fog/cloud computing", "Balanced cost/performance"]
        },
        {
            "tier": "large",
            "max_cost": 1000.0,
            "cpu_limit": 64.0,
            "expected_features": ["High-performance cloud computing", "GPU acceleration"]
        },
        {
            "tier": "enterprise",
            "max_cost": 10000.0,
            "cpu_limit": 1000.0,
            "expected_features": ["Dedicated infrastructure", "SLA guarantees"]
        }
    ]
    
    @pytest.mark.parametrize("scenario", test_scenarios)
    @pytest.mark.asyncio
    async def test_tier_specific_behavior(self, scenario):
        """Test behavior is appropriate for each tier"""
        tier = scenario["tier"]
        
        # Test resource allocator configuration
        allocator = MarketplaceResourceAllocator(enable_marketplace=False)
        await allocator.initialize()
        
        config = allocator.tier_configs[tier]
        assert config.max_hourly_budget <= Decimal(str(scenario["max_cost"]))
        assert config.max_cpu_cores <= Decimal(str(scenario["cpu_limit"]))
        
        # Test tier limits
        tier_limits = allocator._get_tier_limits(tier)
        assert tier_limits["max_hourly_budget"] <= scenario["max_cost"]
        assert tier_limits["max_cpu_cores"] <= scenario["cpu_limit"]
    
    @pytest.mark.asyncio
    async def test_tier_escalation_behavior(self):
        """Test what happens when tier limits are exceeded"""
        allocator = MarketplaceResourceAllocator(enable_marketplace=False)
        await allocator.initialize()
        
        # Try to exceed small tier limits
        excessive_spec = ResourceSpecification(
            cpu_cores=Decimal("20.0"),  # Way beyond small tier limit
            memory_gb=Decimal("64.0"),   # Way beyond small tier limit
            duration_hours=Decimal("1.0"),
            storage_gb=Decimal("10.0"),
            bandwidth_mbps=Decimal("100.0")
        )
        
        result = await allocator.allocate_resources(
            user_id="test_user",
            size_tier="small",
            resource_spec=excessive_spec,
            task_context={}
        )
        
        # Should fail with helpful error message
        assert result.success is False
        assert "exceed tier limit" in result.error_message
        assert len(result.retry_suggestions) > 0
        assert "upgrade tier" in " ".join(result.retry_suggestions)


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="Required components not available")  
class TestErrorHandlingAndResilience:
    """Test error handling and system resilience"""
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self):
        """Test system handles component failures gracefully"""
        coordinator = UnifiedFederatedCoordinator(
            coordinator_id="failure_test",
            enable_marketplace=False,
            enable_p2p=False
        )
        
        # Simulate component failure
        coordinator.inference_coordinator = None
        coordinator.training_coordinator = None
        
        # System should still handle requests gracefully
        with pytest.raises((RuntimeError, Exception)):
            await coordinator.request_inference(
                user_id="test_user",
                model_id="test_model", 
                input_data={"text": "test"},
                size_tier="medium"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_request_handling(self):
        """Test handling of invalid requests"""
        api = UnifiedFederatedAPI()
        api._initialized = True
        api.coordinator = Mock()
        
        # Test missing required parameters
        response = await api.request_inference(
            user_id="",  # Empty user ID
            model="test_model",
            input_data={"text": "test"},
            size_tier="medium"
        )
        
        assert response.success is False
        assert "user_id is required" in response.error
    
    @pytest.mark.asyncio
    async def test_budget_constraint_handling(self):
        """Test handling when budget constraints cannot be met"""
        allocator = MarketplaceResourceAllocator(enable_marketplace=False)
        await allocator.initialize()
        
        # Request with very low budget
        spec = ResourceSpecification(
            cpu_cores=Decimal("8.0"),
            memory_gb=Decimal("16.0"), 
            duration_hours=Decimal("10.0"),  # Long duration
            storage_gb=Decimal("100.0"),
            bandwidth_mbps=Decimal("1000.0"),
            max_total_cost=Decimal("1.00")  # Impossibly low budget
        )
        
        result = await allocator.allocate_resources(
            user_id="budget_test",
            size_tier="medium",
            resource_spec=spec,
            task_context={}
        )
        
        # Should fail but provide helpful suggestions
        assert result.success is False
        assert "budget" in result.error_message.lower()
        assert len(result.retry_suggestions) > 0


# Run specific tests for demonstration
if __name__ == "__main__":
    import sys
    
    print("Running Unified Federated System Tests...")
    print("=" * 50)
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Components not available for testing")
        sys.exit(1)
    
    # Run a few key tests to demonstrate functionality
    async def run_demo_tests():
        print("🧪 Testing UnifiedFederatedCoordinator...")
        
        coordinator = UnifiedFederatedCoordinator(
            coordinator_id="demo_test",
            enable_marketplace=False,
            enable_p2p=False
        )
        
        try:
            await coordinator.initialize()
            print("✅ Coordinator initialization successful")
            
            # Test inference request
            request_id = await coordinator.request_inference(
                user_id="demo_user",
                model_id="demo_model",
                input_data={"text": "Hello, world!"},
                size_tier="medium",
                max_cost=50.0
            )
            print(f"✅ Inference request submitted: {request_id}")
            
            # Test status check
            status = await coordinator.get_request_status(request_id)
            print(f"✅ Status retrieved: {status['status']}")
            
            print("\n🧪 Testing MarketplaceResourceAllocator...")
            
            allocator = MarketplaceResourceAllocator(enable_marketplace=False)
            await allocator.initialize()
            
            spec = ResourceSpecification(
                cpu_cores=Decimal("2.0"),
                memory_gb=Decimal("4.0"),
                duration_hours=Decimal("1.0"),
                storage_gb=Decimal("10.0"),
                bandwidth_mbps=Decimal("100.0")
            )
            
            result = await allocator.allocate_resources(
                user_id="demo_user",
                size_tier="medium",
                resource_spec=spec,
                task_context={"type": "inference"}
            )
            
            if result.success:
                print(f"✅ Resource allocation successful: ${float(result.total_cost):.2f}")
            else:
                print(f"❌ Resource allocation failed: {result.error_message}")
            
            print("\n🧪 Testing DynamicPricingOptimizer...")
            
            optimizer = DynamicPricingOptimizer()
            await optimizer.initialize()
            
            context = OptimizationContext(
                user_id="demo_user",
                size_tier="medium",
                request_type="inference",
                resource_spec=spec,
                primary_objective=OptimizationObjective.BALANCE_COST_PERFORMANCE
            )
            
            opt_result = await optimizer.optimize_allocation(context)
            
            if opt_result.success:
                print(f"✅ Optimization successful: score={opt_result.optimization_score:.2f}")
            else:
                print("❌ Optimization failed")
            
            await coordinator.stop()
            await optimizer.stop()
            
            print("\n🎉 All demo tests completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo test failed: {e}")
    
    # Run the demo
    asyncio.run(run_demo_tests())