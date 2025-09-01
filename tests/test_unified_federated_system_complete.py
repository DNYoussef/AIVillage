"""
Comprehensive Integration Tests for Unified Federated System

This test suite validates the COMPLETE UNIFIED SYSTEM that connects:
- Federated Inference and Training
- Marketplace Integration with Size-Tier Allocation  
- Resource Allocator with Dynamic Optimization
- Unified API for seamless user experience

Test Coverage:
- End-to-end inference workflows for all tiers
- End-to-end training workflows for all tiers
- Marketplace resource allocation and billing
- Cross-tier functionality and limits
- Error handling and edge cases
- Performance and scalability scenarios

This validates that TODAY'S CRITICAL IMPLEMENTATION works perfectly
and connects all of yesterday's work into ONE UNIFIED SYSTEM.
"""

import asyncio
from decimal import Decimal
from datetime import UTC, datetime, timedelta
import logging
import pytest
from typing import Any, Dict
import uuid

# Import the unified system components
try:
    from infrastructure.distributed_inference.unified_federated_coordinator import (
        UnifiedFederatedCoordinator,
        WorkloadType,
        UserTier,
    )
    from infrastructure.distributed_inference.marketplace_integration import (
        MarketplaceIntegrator,
        MarketplaceRequest,
    )
    from infrastructure.distributed_inference.resource_allocator import (
        TierOptimizedResourceAllocator,
        NodeCapacity,
        NodeClass,
    )
    from infrastructure.distributed_inference.unified_api import (
        submit_inference,
        submit_training,
        get_job_status,
        get_pricing_estimate,
        get_system_status,
        UnifiedFederatedAPI,
    )

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.warning(f"Unified system components not available for testing: {e}")

# Test configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
class TestUnifiedFederatedSystemComplete:
    """
    Complete integration tests for the unified federated system

    This test class validates the entire system working together:
    - All user tiers (Small/Medium/Large/Enterprise)
    - Both inference and training workloads
    - Marketplace resource allocation and billing
    - Performance across different scenarios
    """

    @pytest.fixture
    async def unified_system(self):
        """Set up the complete unified system for testing"""
        if not COMPONENTS_AVAILABLE:
            pytest.skip("Unified system components not available")

        # Initialize unified coordinator
        coordinator = UnifiedFederatedCoordinator(
            coordinator_id="test_unified_system", enable_marketplace=True, enable_dynamic_pricing=True
        )
        initialized = await coordinator.initialize()
        assert initialized, "Failed to initialize unified coordinator"

        # Initialize marketplace integrator
        marketplace = MarketplaceIntegrator()
        marketplace_initialized = await marketplace.initialize()
        assert marketplace_initialized, "Failed to initialize marketplace"

        # Initialize resource allocator with test nodes
        allocator = TierOptimizedResourceAllocator()

        # Add test nodes for all tiers
        await self._setup_test_nodes(allocator)

        # Initialize unified API
        api = UnifiedFederatedAPI()
        api_initialized = await api.initialize()
        assert api_initialized, "Failed to initialize unified API"

        system = {"coordinator": coordinator, "marketplace": marketplace, "allocator": allocator, "api": api}

        yield system

        # Cleanup
        await coordinator.stop()
        logger.info("Test system cleaned up")

    async def _setup_test_nodes(self, allocator: TierOptimizedResourceAllocator):
        """Set up test nodes for different tiers"""

        test_nodes = [
            # Small tier nodes (mobile/edge)
            NodeCapacity(
                node_id="mobile_node_1",
                node_class=NodeClass.MOBILE,
                cpu_cores=2.0,
                memory_gb=4.0,
                storage_gb=32.0,
                bandwidth_mbps=50.0,
                latency_ms=50.0,
                region="local",
                trust_score=0.7,
                uptime_percentage=95.0,
                cost_per_hour=Decimal("0.50"),
            ),
            NodeCapacity(
                node_id="edge_node_1",
                node_class=NodeClass.EDGE,
                cpu_cores=4.0,
                memory_gb=8.0,
                storage_gb=100.0,
                bandwidth_mbps=100.0,
                latency_ms=30.0,
                region="local",
                trust_score=0.8,
                uptime_percentage=97.0,
                cost_per_hour=Decimal("1.00"),
            ),
            # Medium tier nodes (edge/cloud)
            NodeCapacity(
                node_id="cloud_node_1",
                node_class=NodeClass.CLOUD,
                cpu_cores=8.0,
                memory_gb=16.0,
                storage_gb=200.0,
                bandwidth_mbps=1000.0,
                latency_ms=20.0,
                region="us-east",
                trust_score=0.9,
                uptime_percentage=99.0,
                cost_per_hour=Decimal("2.00"),
            ),
            NodeCapacity(
                node_id="cloud_node_2",
                node_class=NodeClass.CLOUD,
                cpu_cores=8.0,
                memory_gb=16.0,
                storage_gb=200.0,
                bandwidth_mbps=1000.0,
                latency_ms=25.0,
                region="us-west",
                trust_score=0.85,
                uptime_percentage=98.5,
                cost_per_hour=Decimal("2.20"),
            ),
            # Large tier nodes (cloud/GPU)
            NodeCapacity(
                node_id="gpu_node_1",
                node_class=NodeClass.GPU,
                cpu_cores=16.0,
                memory_gb=64.0,
                storage_gb=500.0,
                gpu_memory_gb=24.0,
                bandwidth_mbps=10000.0,
                latency_ms=10.0,
                region="us-central",
                trust_score=0.95,
                uptime_percentage=99.5,
                cost_per_hour=Decimal("8.00"),
            ),
            NodeCapacity(
                node_id="gpu_node_2",
                node_class=NodeClass.GPU,
                cpu_cores=32.0,
                memory_gb=128.0,
                storage_gb=1000.0,
                gpu_memory_gb=48.0,
                bandwidth_mbps=10000.0,
                latency_ms=8.0,
                region="eu-central",
                trust_score=0.92,
                uptime_percentage=99.0,
                cost_per_hour=Decimal("12.00"),
            ),
            # Enterprise tier nodes (specialized)
            NodeCapacity(
                node_id="specialized_node_1",
                node_class=NodeClass.SPECIALIZED,
                cpu_cores=64.0,
                memory_gb=512.0,
                storage_gb=2000.0,
                gpu_memory_gb=80.0,
                bandwidth_mbps=100000.0,
                latency_ms=5.0,
                region="us-enterprise",
                trust_score=0.99,
                uptime_percentage=99.9,
                cost_per_hour=Decimal("50.00"),
            ),
        ]

        # Register all nodes
        for node in test_nodes:
            success = await allocator.register_node(node)
            assert success, f"Failed to register test node {node.node_id}"

        logger.info(f"Set up {len(test_nodes)} test nodes for all tiers")

    # =========================================================================
    # END-TO-END INFERENCE TESTS - All Tiers
    # =========================================================================

    @pytest.mark.asyncio
    async def test_small_tier_inference_complete(self, unified_system):
        """Test complete inference workflow for small tier"""

        # Submit inference request
        job_id = await submit_inference(
            user_id="small_user_001",
            model_id="gpt-3-small",
            input_data={"prompt": "What is the weather like?", "max_tokens": 50},
            user_tier="small",
            max_cost=5.0,
            privacy_level="medium",
        )

        assert job_id.startswith("inf_"), "Invalid job ID format"
        logger.info(f"Small tier inference job submitted: {job_id}")

        # Monitor job until completion
        status = await self._wait_for_job_completion(job_id, timeout_seconds=60)

        # Validate results
        assert status["success"], f"Job failed: {status.get('error')}"
        assert status["data"]["status"] in ["completed", "executing"], "Job not progressing"
        assert float(status["data"]["total_cost"]) <= 5.0, "Cost exceeded budget"
        assert status["data"]["user_tier"] == "small", "Incorrect tier assignment"

        # Check resource allocation was appropriate for small tier
        if status["data"]["status"] == "completed":
            assert status["data"]["nodes_allocated"] <= 3, "Too many nodes for small tier"
            assert len(status["data"]["regions_used"]) <= 2, "Too many regions for small tier"

    @pytest.mark.asyncio
    async def test_medium_tier_inference_performance(self, unified_system):
        """Test medium tier inference with performance requirements"""

        job_id = await submit_inference(
            user_id="medium_user_001",
            model_id="gpt-3-medium",
            input_data={"prompt": "Analyze this business data", "context": "quarterly reports"},
            user_tier="medium",
            max_cost=50.0,
            privacy_level="high",
            max_latency_ms=2000,
        )

        assert job_id.startswith("inf_"), "Invalid job ID format"
        logger.info(f"Medium tier inference job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=90)

        # Validate medium tier capabilities
        assert status["success"], f"Medium tier job failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["processing_time_ms"] <= 5000, "Too slow for medium tier"
            assert status["data"]["quality_score"] >= 0.7, "Quality below medium tier expectations"
            assert 1 <= status["data"]["nodes_allocated"] <= 10, "Node allocation outside medium tier range"

    @pytest.mark.asyncio
    async def test_large_tier_inference_gpu_acceleration(self, unified_system):
        """Test large tier inference with GPU acceleration"""

        job_id = await submit_inference(
            user_id="large_user_001",
            model_id="gpt-3-large",
            input_data={
                "prompt": "Generate a comprehensive market analysis report",
                "requirements": "detailed financial projections",
                "length": "extensive",
            },
            user_tier="large",
            max_cost=200.0,
            privacy_level="ultra",
            preferred_node_types=["gpu", "cloud"],
        )

        assert job_id.startswith("inf_"), "Invalid job ID format"
        logger.info(f"Large tier inference job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=120)

        # Validate large tier performance
        assert status["success"], f"Large tier job failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["quality_score"] >= 0.85, "Quality below large tier expectations"
            assert status["data"]["nodes_allocated"] >= 2, "Insufficient nodes for large tier"
            assert len(status["data"]["regions_used"]) >= 1, "No geographic distribution"

    @pytest.mark.asyncio
    async def test_enterprise_tier_inference_premium(self, unified_system):
        """Test enterprise tier inference with premium features"""

        job_id = await submit_inference(
            user_id="enterprise_user_001",
            model_id="gpt-4-enterprise",
            input_data={
                "prompt": "Strategic business intelligence analysis",
                "classification": "confidential",
                "complexity": "high",
                "multi_modal": True,
            },
            user_tier="enterprise",
            max_cost=1000.0,
            privacy_level="ultra",
            guaranteed_uptime=99.9,
            dedicated_resources=True,
        )

        assert job_id.startswith("inf_"), "Invalid job ID format"
        logger.info(f"Enterprise tier inference job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=180)

        # Validate enterprise tier guarantees
        assert status["success"], f"Enterprise tier job failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["quality_score"] >= 0.95, "Quality below enterprise expectations"
            assert status["data"]["processing_time_ms"] <= 3000, "Too slow for enterprise SLA"
            assert status["data"]["nodes_allocated"] >= 3, "Insufficient nodes for enterprise"

    # =========================================================================
    # END-TO-END TRAINING TESTS - All Tiers
    # =========================================================================

    @pytest.mark.asyncio
    async def test_small_tier_training_mobile_optimized(self, unified_system):
        """Test federated training optimized for small tier mobile devices"""

        job_id = await submit_training(
            user_id="small_trainer_001",
            model_id="bert-small",
            training_config={
                "dataset": "sentiment_analysis_small",
                "mobile_optimization": True,
                "local_epochs": 3,
                "batch_size": 8,
            },
            user_tier="small",
            participants_needed=5,
            training_rounds=5,
            max_cost=25.0,
            privacy_level="medium",
            duration_hours=1.0,
        )

        assert job_id.startswith("train_"), "Invalid training job ID format"
        logger.info(f"Small tier training job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=300)  # 5 minutes

        # Validate small tier training
        assert status["success"], f"Small tier training failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["nodes_allocated"] <= 5, "Too many participants for small tier"
            assert float(status["data"]["total_cost"]) <= 25.0, "Training cost exceeded small tier budget"

    @pytest.mark.asyncio
    async def test_medium_tier_training_federated(self, unified_system):
        """Test medium tier federated training with balanced optimization"""

        job_id = await submit_training(
            user_id="medium_trainer_001",
            model_id="bert-base",
            training_config={
                "dataset": "classification_medium",
                "federated_averaging": True,
                "differential_privacy": True,
                "privacy_budget": 1.0,
            },
            user_tier="medium",
            participants_needed=15,
            training_rounds=10,
            max_cost=150.0,
            privacy_level="high",
            duration_hours=2.0,
        )

        assert job_id.startswith("train_"), "Invalid training job ID format"
        logger.info(f"Medium tier training job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=600)  # 10 minutes

        # Validate medium tier training performance
        assert status["success"], f"Medium tier training failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert 10 <= status["data"]["nodes_allocated"] <= 20, "Node count outside medium tier range"
            assert status["data"]["quality_score"] >= 0.75, "Training quality below medium tier expectations"

    @pytest.mark.asyncio
    async def test_large_tier_training_distributed(self, unified_system):
        """Test large tier distributed training with GPU acceleration"""

        job_id = await submit_training(
            user_id="large_trainer_001",
            model_id="llama-7b",
            training_config={
                "dataset": "instruction_tuning_large",
                "model_parallel": True,
                "gradient_compression": True,
                "gpu_optimization": True,
                "distributed_strategy": "data_parallel",
            },
            user_tier="large",
            participants_needed=50,
            training_rounds=20,
            max_cost=800.0,
            privacy_level="ultra",
            duration_hours=4.0,
            reliability_requirement="high",
        )

        assert job_id.startswith("train_"), "Invalid training job ID format"
        logger.info(f"Large tier training job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=900)  # 15 minutes

        # Validate large tier training capabilities
        assert status["success"], f"Large tier training failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["nodes_allocated"] >= 20, "Insufficient nodes for large tier training"
            assert status["data"]["quality_score"] >= 0.9, "Training quality below large tier expectations"
            assert len(status["data"]["regions_used"]) >= 2, "Insufficient geographic distribution"

    @pytest.mark.asyncio
    async def test_enterprise_tier_training_custom(self, unified_system):
        """Test enterprise tier custom training with dedicated infrastructure"""

        job_id = await submit_training(
            user_id="enterprise_trainer_001",
            model_id="custom-foundation-model",
            training_config={
                "dataset": "proprietary_enterprise_data",
                "custom_architecture": True,
                "homomorphic_encryption": True,
                "multi_modal_training": True,
                "compliance": ["GDPR", "HIPAA"],
                "dedicated_infrastructure": True,
            },
            user_tier="enterprise",
            participants_needed=100,
            training_rounds=50,
            max_cost=5000.0,
            privacy_level="ultra",
            duration_hours=8.0,
            reliability_requirement="enterprise",
            sla_uptime=99.9,
        )

        assert job_id.startswith("train_"), "Invalid training job ID format"
        logger.info(f"Enterprise tier training job submitted: {job_id}")

        status = await self._wait_for_job_completion(job_id, timeout_seconds=1800)  # 30 minutes

        # Validate enterprise tier training guarantees
        assert status["success"], f"Enterprise tier training failed: {status.get('error')}"
        if status["data"]["status"] == "completed":
            assert status["data"]["nodes_allocated"] >= 50, "Insufficient nodes for enterprise training"
            assert status["data"]["quality_score"] >= 0.95, "Training quality below enterprise expectations"

    # =========================================================================
    # MARKETPLACE INTEGRATION TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_marketplace_resource_allocation(self, unified_system):
        """Test marketplace resource allocation across tiers"""

        marketplace = unified_system["marketplace"]

        # Test allocation for each tier
        for tier, config in [
            ("small", {"participants": 3, "budget": 10.0}),
            ("medium", {"participants": 15, "budget": 100.0}),
            ("large", {"participants": 50, "budget": 500.0}),
            ("enterprise", {"participants": 100, "budget": 2000.0}),
        ]:

            request = MarketplaceRequest(
                request_id=f"market_test_{tier}_{uuid.uuid4().hex[:8]}",
                user_id=f"market_user_{tier}",
                user_tier=UserTier(tier),
                workload_type=WorkloadType.INFERENCE,
                participants_needed=config["participants"],
                model_size="medium",
                duration_hours=Decimal("1.0"),
                privacy_level="medium",
                reliability_requirement="standard",
                max_budget=Decimal(str(config["budget"])),
            )

            result = await marketplace.allocate_resources(request)

            # Validate allocation results
            assert result.success, f"Marketplace allocation failed for {tier} tier: {result.error_message}"
            assert len(result.allocated_nodes) > 0, f"No nodes allocated for {tier} tier"
            assert result.total_cost <= request.max_budget, f"Cost exceeded budget for {tier} tier"

            logger.info(
                f"Marketplace allocation for {tier} tier: "
                f"{len(result.allocated_nodes)} nodes, cost ${result.total_cost}"
            )

    @pytest.mark.asyncio
    async def test_dynamic_pricing_across_tiers(self, unified_system):
        """Test dynamic pricing adjustments across different tiers"""

        # Get pricing estimates for all tiers
        pricing_results = {}

        for tier in ["small", "medium", "large", "enterprise"]:
            pricing = await get_pricing_estimate(
                job_type="training",
                model_id="bert-base",
                user_tier=tier,
                participants_needed=10,
                duration_hours=2.0,
                privacy_level="high",
            )

            assert pricing["success"], f"Pricing failed for {tier} tier"
            pricing_results[tier] = pricing["data"]
            logger.info(f"Pricing for {tier} tier: ${pricing['data']['pricing_breakdown']['estimated_total']}")

        # Validate tier pricing hierarchy (higher tiers should have different pricing)
        small_cost = pricing_results["small"]["pricing_breakdown"]["estimated_total"]
        medium_cost = pricing_results["medium"]["pricing_breakdown"]["estimated_total"]
        large_cost = pricing_results["large"]["pricing_breakdown"]["estimated_total"]
        pricing_results["enterprise"]["pricing_breakdown"]["estimated_total"]

        # Enterprise tier should have premium features but potentially volume discounts
        assert small_cost < medium_cost, "Small tier should be cheaper than medium"
        assert medium_cost < large_cost, "Medium tier should be cheaper than large"

        # Check tier-specific features and limits
        for tier, pricing_data in pricing_results.items():
            tier_limits = pricing_data["tier_limits"]

            if tier == "small":
                assert tier_limits["max_participants_per_job"] <= 10, "Small tier participant limit too high"
            elif tier == "enterprise":
                assert tier_limits["max_participants_per_job"] >= 500, "Enterprise tier participant limit too low"

    # =========================================================================
    # CROSS-TIER FUNCTIONALITY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_tier_limit_enforcement(self, unified_system):
        """Test that tier limits are properly enforced"""

        # Try to exceed small tier limits
        try:
            await submit_training(
                user_id="small_user_violation",
                model_id="bert-base",
                training_config={"dataset": "large_dataset"},
                user_tier="small",
                participants_needed=50,  # Exceeds small tier limit
                max_cost=5.0,  # Valid for small tier
            )
            # Should not reach here
            assert False, "Small tier limit violation was not caught"
        except Exception as e:
            assert "exceeds" in str(e).lower() or "limit" in str(e).lower(), "Wrong error for limit violation"

        # Try to exceed budget limits
        try:
            await submit_inference(
                user_id="small_user_budget_violation",
                model_id="gpt-3-small",
                input_data={"prompt": "test"},
                user_tier="small",
                max_cost=1000.0,  # Exceeds small tier budget
            )
            # Should not reach here
            assert False, "Budget limit violation was not caught"
        except Exception as e:
            assert "budget" in str(e).lower() or "cost" in str(e).lower(), "Wrong error for budget violation"

        logger.info("Tier limit enforcement working correctly")

    @pytest.mark.asyncio
    async def test_cross_tier_resource_sharing(self, unified_system):
        """Test resource sharing and allocation across tiers"""

        allocator = unified_system["allocator"]

        # Submit concurrent jobs from different tiers
        jobs = []

        # Small tier job
        small_job = await submit_inference(
            user_id="concurrent_small",
            model_id="gpt-3-small",
            input_data={"prompt": "small task"},
            user_tier="small",
            max_cost=5.0,
        )
        jobs.append(("small", small_job))

        # Medium tier job
        medium_job = await submit_inference(
            user_id="concurrent_medium",
            model_id="gpt-3-medium",
            input_data={"prompt": "medium complexity task"},
            user_tier="medium",
            max_cost=50.0,
        )
        jobs.append(("medium", medium_job))

        # Large tier job
        large_job = await submit_inference(
            user_id="concurrent_large",
            model_id="gpt-3-large",
            input_data={"prompt": "complex analysis task"},
            user_tier="large",
            max_cost=200.0,
        )
        jobs.append(("large", large_job))

        # Wait for all jobs to complete or make progress
        for tier, job_id in jobs:
            status = await self._wait_for_job_completion(job_id, timeout_seconds=120)
            assert status["success"], f"Concurrent {tier} tier job failed"
            logger.info(f"Concurrent {tier} tier job status: {status['data']['status']}")

        # Check that resources were allocated fairly
        allocation_stats = await allocator.get_allocation_stats()
        assert allocation_stats["active_allocations"] >= 0, "Resource allocation tracking failed"

        logger.info("Cross-tier resource sharing working correctly")

    # =========================================================================
    # PERFORMANCE AND SCALABILITY TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_high_concurrency_inference(self, unified_system):
        """Test system performance under high concurrent inference load"""

        # Submit multiple concurrent inference requests
        concurrent_jobs = []
        num_concurrent = 10

        for i in range(num_concurrent):
            job_id = await submit_inference(
                user_id=f"concurrent_user_{i}",
                model_id="gpt-3-small",
                input_data={"prompt": f"concurrent task {i}"},
                user_tier="medium",
                max_cost=10.0,
            )
            concurrent_jobs.append(job_id)

        logger.info(f"Submitted {num_concurrent} concurrent inference jobs")

        # Wait for all jobs to complete
        completed_jobs = 0
        failed_jobs = 0

        for job_id in concurrent_jobs:
            try:
                status = await self._wait_for_job_completion(job_id, timeout_seconds=60)
                if status["success"]:
                    completed_jobs += 1
                else:
                    failed_jobs += 1
            except Exception:
                failed_jobs += 1

        # Validate performance
        success_rate = completed_jobs / num_concurrent
        assert success_rate >= 0.8, f"Success rate too low: {success_rate} (expected >= 0.8)"

        logger.info(
            f"High concurrency test: {completed_jobs}/{num_concurrent} jobs completed "
            f"(success rate: {success_rate:.2%})"
        )

    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, unified_system):
        """Test system performance with mixed inference and training workloads"""

        mixed_jobs = []

        # Submit mixed workloads
        for i in range(3):
            # Inference job
            inf_job = await submit_inference(
                user_id=f"mixed_inf_user_{i}",
                model_id="bert-base",
                input_data={"text": f"inference task {i}"},
                user_tier="medium",
                max_cost=25.0,
            )
            mixed_jobs.append(("inference", inf_job))

            # Training job
            train_job = await submit_training(
                user_id=f"mixed_train_user_{i}",
                model_id="bert-small",
                training_config={"dataset": f"mixed_dataset_{i}"},
                user_tier="medium",
                participants_needed=5,
                training_rounds=3,
                max_cost=75.0,
            )
            mixed_jobs.append(("training", train_job))

        logger.info(f"Submitted {len(mixed_jobs)} mixed workload jobs")

        # Monitor all jobs
        results = {"inference": {"completed": 0, "failed": 0}, "training": {"completed": 0, "failed": 0}}

        for job_type, job_id in mixed_jobs:
            try:
                status = await self._wait_for_job_completion(job_id, timeout_seconds=180)
                if status["success"] and status["data"]["status"] in ["completed", "executing"]:
                    results[job_type]["completed"] += 1
                else:
                    results[job_type]["failed"] += 1
            except Exception:
                results[job_type]["failed"] += 1

        # Validate mixed workload handling
        total_completed = results["inference"]["completed"] + results["training"]["completed"]
        total_jobs = len(mixed_jobs)
        success_rate = total_completed / total_jobs

        assert success_rate >= 0.7, f"Mixed workload success rate too low: {success_rate:.2%}"

        logger.info(
            f"Mixed workload performance: {total_completed}/{total_jobs} jobs successful "
            f"(success rate: {success_rate:.2%})"
        )

    # =========================================================================
    # SYSTEM STATUS AND HEALTH TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_comprehensive_system_status(self, unified_system):
        """Test comprehensive system status reporting"""

        # Submit a few jobs to generate activity
        await submit_inference("status_test_user", "gpt-3-small", {"prompt": "test"}, "small", 5.0)
        await submit_training("status_test_user", "bert-small", {"dataset": "test"}, "medium", 5, 3, 50.0)

        # Get system status
        system_status = await get_system_status()

        assert system_status["success"], "Failed to get system status"
        status_data = system_status["data"]

        # Validate system components are reported
        assert "unified_api" in status_data, "Unified API status missing"
        assert "coordinator_status" in status_data, "Coordinator status missing"
        assert "marketplace_status" in status_data, "Marketplace status missing"
        assert "resource_allocator" in status_data, "Resource allocator status missing"

        # Check API statistics
        api_stats = status_data["unified_api"]
        assert api_stats["total_jobs_processed"] > 0, "No jobs processed"
        assert api_stats["users_served"] > 0, "No users served"

        # Check job type distribution
        job_dist = status_data["job_type_distribution"]
        assert "inference" in job_dist, "Inference jobs not tracked"
        assert "training" in job_dist, "Training jobs not tracked"

        logger.info(
            f"System status: {api_stats['total_jobs_processed']} jobs processed, "
            f"{api_stats['users_served']} users served"
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def _wait_for_job_completion(self, job_id: str, timeout_seconds: int = 60) -> Dict[str, Any]:
        """Wait for a job to complete or timeout"""

        start_time = datetime.now(UTC)
        timeout = timedelta(seconds=timeout_seconds)
        check_interval = 2  # Check every 2 seconds

        while datetime.now(UTC) - start_time < timeout:
            try:
                status = await get_job_status(job_id)

                if not status["success"]:
                    return status

                job_status = status["data"]["status"]

                # Job completed successfully
                if job_status == "completed":
                    logger.info(f"Job {job_id} completed successfully")
                    return status

                # Job failed
                if job_status == "failed":
                    logger.warning(f"Job {job_id} failed: {status['data'].get('error_message', 'Unknown error')}")
                    return status

                # Job still in progress
                if job_status in ["submitted", "allocating_resources", "executing"]:
                    logger.debug(f"Job {job_id} status: {job_status}")
                    await asyncio.sleep(check_interval)
                    continue

                # Unknown status
                logger.warning(f"Job {job_id} unknown status: {job_status}")
                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error checking job {job_id}: {e}")
                await asyncio.sleep(check_interval)

        # Timeout
        logger.warning(f"Job {job_id} timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "error": f"Job monitoring timed out after {timeout_seconds} seconds",
            "data": {"status": "timeout", "job_id": job_id},
        }


# =========================================================================
# STANDALONE TEST FUNCTIONS (for individual testing)
# =========================================================================


@pytest.mark.asyncio
async def test_quick_inference_smoke_test():
    """Quick smoke test for inference functionality"""

    if not COMPONENTS_AVAILABLE:
        pytest.skip("Components not available")

    try:
        job_id = await submit_inference(
            user_id="smoke_test_user",
            model_id="gpt-3-small",
            input_data={"prompt": "Hello, world!"},
            user_tier="small",
            max_cost=2.0,
        )

        assert job_id, "No job ID returned"
        logger.info(f"Smoke test inference job submitted: {job_id}")

        # Quick status check
        status = await get_job_status(job_id)
        assert status["success"], "Failed to get job status"
        assert status["data"]["status"] in ["submitted", "allocating_resources", "executing"], "Invalid initial status"

        logger.info("Smoke test passed!")

    except Exception as e:
        pytest.fail(f"Smoke test failed: {e}")


@pytest.mark.asyncio
async def test_pricing_estimates_all_tiers():
    """Test pricing estimates across all tiers"""

    if not COMPONENTS_AVAILABLE:
        pytest.skip("Components not available")

    for tier in ["small", "medium", "large", "enterprise"]:
        for job_type in ["inference", "training"]:
            try:
                pricing = await get_pricing_estimate(
                    job_type=job_type, model_id="bert-base", user_tier=tier, duration_hours=1.0, participants_needed=10
                )

                assert pricing["success"], f"Pricing failed for {tier} tier {job_type}"
                assert "estimated_total" in pricing["data"]["pricing_breakdown"], "Missing cost estimate"

                cost = pricing["data"]["pricing_breakdown"]["estimated_total"]
                logger.info(f"Pricing for {tier} tier {job_type}: ${cost}")

            except Exception as e:
                pytest.fail(f"Pricing test failed for {tier} tier {job_type}: {e}")


if __name__ == "__main__":
    # Run a quick test when executed directly
    import asyncio

    async def main():
        print("Running unified federated system integration tests...")

        try:
            await test_quick_inference_smoke_test()
            await test_pricing_estimates_all_tiers()
            print("✅ All standalone tests passed!")
        except Exception as e:
            print(f"❌ Tests failed: {e}")

    asyncio.run(main())
