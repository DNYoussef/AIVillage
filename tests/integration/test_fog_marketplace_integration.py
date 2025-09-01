"""
Comprehensive Integration Test for Enhanced Fog Marketplace

Tests the complete end-to-end workflow of the enhanced fog marketplace including:
- Size-tier pricing system with federated workload support
- P2P resource discovery and participant matching
- Dynamic resource allocation with QoS guarantees
- Auction engine with federated-specific auctions
- Market orchestrator coordination
- Performance monitoring and analytics

This test demonstrates the full capabilities of the AI Village fog marketplace
for federated learning and inference workloads.
"""

import asyncio
import pytest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import logging

# Import marketplace components
from infrastructure.fog.market.marketplace_api import get_marketplace_api
from infrastructure.fog.market.auction_engine import (
    ResourceRequirement,
    create_federated_inference_auction,
    create_federated_training_auction,
)
from infrastructure.fog.market.pricing_manager import (
    UserSizeTier,
    get_pricing_manager,
)
from infrastructure.fog.market.resource_allocator import (
    AllocationStrategy,
    QoSRequirement,
    get_resource_allocator,
)
from infrastructure.fog.market.p2p_integration import (
    P2PResourceInfo,
    NetworkTier,
    get_p2p_integration,
)
from infrastructure.fog.market.market_orchestrator import get_market_orchestrator

logger = logging.getLogger(__name__)


class TestFogMarketplaceIntegration:
    """Comprehensive test suite for fog marketplace integration"""

    @pytest.fixture(autouse=True)
    async def setup_marketplace(self):
        """Setup marketplace components for testing"""

        # Initialize core components
        self.pricing_manager = await get_pricing_manager()
        self.auction_engine = await self.pricing_manager.auction_engine
        self.resource_allocator = await get_resource_allocator()
        self.market_orchestrator = await get_market_orchestrator()
        self.marketplace_api = get_marketplace_api()

        # Initialize P2P integration
        self.p2p_integration = get_p2p_integration(
            peer_id="test_marketplace_node",
            auction_engine=self.auction_engine,
            pricing_manager=self.pricing_manager,
            market_orchestrator=self.market_orchestrator,
            resource_allocator=self.resource_allocator,
        )

        await self.p2p_integration.start()
        await self.marketplace_api.initialize_market_components()

        logger.info("Marketplace components initialized for testing")

    async def test_size_tier_pricing_system(self):
        """Test the size-tier pricing system for different user types"""

        logger.info("Testing size-tier pricing system...")

        # Test pricing for each user tier
        test_cases = [
            {
                "tier": UserSizeTier.SMALL,
                "workload": "federated_inference",
                "model_size": "small",
                "requests": 50,
                "participants": 3,
                "expected_price_range": (0.01, 0.10),
            },
            {
                "tier": UserSizeTier.MEDIUM,
                "workload": "federated_inference",
                "model_size": "medium",
                "requests": 200,
                "participants": 10,
                "expected_price_range": (0.10, 1.00),
            },
            {
                "tier": UserSizeTier.LARGE,
                "workload": "federated_training",
                "model_size": "large",
                "duration_hours": 4.0,
                "participants": 25,
                "expected_price_range": (100.0, 1000.0),
            },
            {
                "tier": UserSizeTier.ENTERPRISE,
                "workload": "federated_training",
                "model_size": "xlarge",
                "duration_hours": 8.0,
                "participants": 50,
                "expected_price_range": (1000.0, 10000.0),
            },
        ]

        for case in test_cases:
            if case["workload"] == "federated_inference":
                quote = await self.pricing_manager.get_federated_inference_price(
                    user_tier=case["tier"],
                    model_size=case["model_size"],
                    requests_count=case["requests"],
                    participants_needed=case["participants"],
                )

                price_per_request = quote["price_per_request"]
                total_cost = quote["total_cost"]

                # Verify pricing is within expected range
                assert (
                    case["expected_price_range"][0] <= price_per_request <= case["expected_price_range"][1]
                ), f"Price per request {price_per_request} not in range {case['expected_price_range']} for {case['tier'].value}"

                # Verify tier-specific benefits
                tier_info = quote["tier_info"]
                assert "max_concurrent_jobs" in tier_info
                assert "guaranteed_uptime" in tier_info

                logger.info(
                    f"{case['tier'].value} inference: ${price_per_request:.4f}/request, total=${total_cost:.2f}"
                )

            else:  # federated_training
                quote = await self.pricing_manager.get_federated_training_price(
                    user_tier=case["tier"],
                    model_size=case["model_size"],
                    duration_hours=case["duration_hours"],
                    participants_needed=case["participants"],
                )

                price_per_hour = quote["price_per_hour"]
                total_cost = quote["total_cost"]

                # Verify pricing is within expected range
                assert (
                    case["expected_price_range"][0] <= price_per_hour <= case["expected_price_range"][1]
                ), f"Price per hour {price_per_hour} not in range {case['expected_price_range']} for {case['tier'].value}"

                logger.info(f"{case['tier'].value} training: ${price_per_hour:.2f}/hour, total=${total_cost:.2f}")

        logger.info("✓ Size-tier pricing system test passed")

    async def test_p2p_resource_discovery(self):
        """Test P2P resource discovery and participant matching"""

        logger.info("Testing P2P resource discovery...")

        # Advertise some mock resources
        test_resources = [
            P2PResourceInfo(
                resource_id="p2p_resource_001",
                node_id="edge_node_001",
                provider_id="provider_alpha",
                cpu_cores=8.0,
                memory_gb=32.0,
                storage_gb=500.0,
                bandwidth_mbps=100.0,
                network_tier=NetworkTier.EDGE,
                region="us-east",
                price_per_cpu_hour=0.15,
                trust_score=0.8,
                federated_learning_support=True,
                privacy_techniques=["differential_privacy"],
            ),
            P2PResourceInfo(
                resource_id="p2p_resource_002",
                node_id="fog_node_001",
                provider_id="provider_beta",
                cpu_cores=16.0,
                memory_gb=64.0,
                storage_gb=1000.0,
                bandwidth_mbps=500.0,
                network_tier=NetworkTier.FOG,
                region="us-west",
                price_per_cpu_hour=0.25,
                trust_score=0.9,
                federated_learning_support=True,
                privacy_techniques=["homomorphic_encryption"],
            ),
        ]

        # Advertise resources through P2P network
        for resource in test_resources:
            success = await self.p2p_integration.discovery.advertise_resource(resource)
            assert success, f"Failed to advertise resource {resource.resource_id}"

        # Discover resources with various requirements
        discovery_tests = [
            {
                "name": "Basic federated inference",
                "requirements": {
                    "cpu_cores": 4.0,
                    "memory_gb": 16.0,
                    "participants_needed": 2,
                    "workload_type": "inference",
                },
                "min_expected": 1,
            },
            {
                "name": "High-performance federated training",
                "requirements": {
                    "cpu_cores": 12.0,
                    "memory_gb": 48.0,
                    "participants_needed": 5,
                    "workload_type": "training",
                    "min_trust_score": 0.7,
                },
                "min_expected": 1,
            },
            {
                "name": "Privacy-preserving inference",
                "requirements": {"cpu_cores": 2.0, "memory_gb": 8.0, "privacy_level": "high", "participants_needed": 3},
                "min_expected": 0,  # May not find suitable resources
            },
        ]

        for test in discovery_tests:
            resources = await self.p2p_integration.discovery.discover_resources(
                requirements=test["requirements"], max_results=10, timeout_seconds=15
            )

            assert (
                len(resources) >= test["min_expected"]
            ), f"Discovery test '{test['name']}' found {len(resources)} resources, expected >= {test['min_expected']}"

            logger.info(f"Discovery test '{test['name']}': found {len(resources)} resources")

        # Test federated participant recruitment
        job_requirements = {
            "job_type": "federated_training",
            "model_size": "large",
            "privacy_level": "high",
            "min_trust_score": 0.6,
            "compensation": 50.0,
        }

        participants = await self.p2p_integration.discovery.find_federated_participants(
            job_requirements=job_requirements, min_participants=3, max_participants=10
        )

        assert len(participants) >= 3, f"Expected at least 3 participants, got {len(participants)}"

        for participant in participants:
            assert participant.allocated_resources is not None
            assert participant.allocated_resources.trust_score >= 0.6
            assert participant.status == "available"

        logger.info(f"✓ P2P resource discovery test passed: found {len(participants)} participants")

    async def test_federated_auction_workflow(self):
        """Test end-to-end federated auction workflow"""

        logger.info("Testing federated auction workflow...")

        # Test federated inference auction
        inference_auction_id = await create_federated_inference_auction(
            requester_id="test_user_001",
            model_size="medium",
            participants_needed=5,
            duration_hours=2.0,
            privacy_level="high",
            max_latency_ms=150.0,
            reserve_price=25.0,
        )

        assert inference_auction_id is not None, "Failed to create federated inference auction"

        # Announce auction to P2P network
        auction_info = {
            "auction_id": inference_auction_id,
            "auction_type": "federated_inference",
            "requirements": {"model_size": "medium", "participants_needed": 5, "privacy_level": "high"},
            "reserve_price": 25.0,
            "end_time": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
        }

        announcement_success = await self.p2p_integration.announce_auction(auction_info)
        assert announcement_success, "Failed to announce auction to P2P network"

        # Simulate bid submissions
        test_bids = [
            {
                "bidder_id": "provider_001",
                "bid_price": 20.0,
                "trust_score": 0.8,
                "available_resources": {"cpu_cores": 8.0, "memory_gb": 32.0, "bandwidth_mbps": 100.0},
            },
            {
                "bidder_id": "provider_002",
                "bid_price": 18.0,
                "trust_score": 0.9,
                "available_resources": {"cpu_cores": 12.0, "memory_gb": 48.0, "bandwidth_mbps": 200.0},
            },
        ]

        for bid in test_bids:
            bid_success = await self.p2p_integration.submit_federated_bid(
                auction_id=inference_auction_id, coordinator_peer_id="test_marketplace_node", **bid
            )
            assert bid_success, f"Failed to submit bid from {bid['bidder_id']}"

        # Wait for auction processing
        await asyncio.sleep(2)

        # Check auction status
        auction_status = await self.auction_engine.get_auction_status(inference_auction_id)
        assert auction_status is not None, "Failed to get auction status"
        assert auction_status["total_bids"] >= len(test_bids), "Not all bids were received"

        # Test federated training auction
        training_auction_id = await create_federated_training_auction(
            requester_id="test_researcher_001",
            model_size="large",
            participants_needed=10,
            duration_hours=6.0,
            privacy_level="critical",
            reliability_requirement="guaranteed",
            reserve_price=500.0,
        )

        assert training_auction_id is not None, "Failed to create federated training auction"

        logger.info(
            f"✓ Federated auction workflow test passed: created auctions {inference_auction_id} and {training_auction_id}"
        )

    async def test_dynamic_resource_allocation(self):
        """Test dynamic resource allocation with QoS guarantees"""

        logger.info("Testing dynamic resource allocation...")

        # Create resource requirements for different scenarios
        test_scenarios = [
            {
                "name": "Small mobile inference",
                "requirements": ResourceRequirement(
                    cpu_cores=Decimal("2.0"),
                    memory_gb=Decimal("4.0"),
                    storage_gb=Decimal("10.0"),
                    bandwidth_mbps=Decimal("10.0"),
                    duration_hours=Decimal("1.0"),
                    participants_needed=2,
                    workload_type="inference",
                    privacy_level="medium",
                ),
                "qos": QoSRequirement(
                    max_latency_ms=Decimal("200.0"),
                    min_availability_percentage=Decimal("95.0"),
                    max_cost_per_hour=Decimal("10.0"),
                ),
                "strategy": AllocationStrategy.LOWEST_COST,
            },
            {
                "name": "Enterprise federated training",
                "requirements": ResourceRequirement(
                    cpu_cores=Decimal("16.0"),
                    memory_gb=Decimal("64.0"),
                    storage_gb=Decimal("500.0"),
                    bandwidth_mbps=Decimal("1000.0"),
                    duration_hours=Decimal("8.0"),
                    participants_needed=20,
                    workload_type="training",
                    privacy_level="critical",
                ),
                "qos": QoSRequirement(
                    max_latency_ms=Decimal("50.0"),
                    min_availability_percentage=Decimal("99.9"),
                    max_cost_per_hour=Decimal("1000.0"),
                    privacy_level="critical",
                ),
                "strategy": AllocationStrategy.BEST_QUALITY,
            },
        ]

        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")

            # Discover resources
            discovered_nodes = await self.resource_allocator.discover_resources(
                requirements=scenario["requirements"], qos_requirements=scenario["qos"]
            )

            assert len(discovered_nodes) > 0, f"No resources discovered for scenario {scenario['name']}"

            # Create allocation plan
            allocation_plan = await self.resource_allocator.create_allocation_plan(
                requirements=scenario["requirements"],
                qos_requirements=scenario["qos"],
                discovered_nodes=discovered_nodes,
                allocation_strategy=scenario["strategy"],
            )

            assert allocation_plan is not None, f"Failed to create allocation plan for {scenario['name']}"
            assert len(allocation_plan.primary_nodes) >= 1, "No primary nodes allocated"
            assert (
                allocation_plan.total_cost
                <= scenario["requirements"].duration_hours * scenario["qos"].max_cost_per_hour
            )

            # Execute allocation
            allocation_id = await self.resource_allocator.execute_allocation_plan(
                plan=allocation_plan, requester_id=f"test_user_{scenario['name'].replace(' ', '_')}"
            )

            assert allocation_id is not None, f"Failed to execute allocation plan for {scenario['name']}"

            # Monitor allocation status
            allocation_status = await self.resource_allocator.get_allocation_status(allocation_id)
            assert allocation_status is not None, "Failed to get allocation status"
            assert allocation_status["status"] in ["monitoring", "active"]

            # Test QoS monitoring
            qos_status = await self.resource_allocator.monitor_allocation_qos(allocation_id)
            assert qos_status is not None, "Failed to monitor QoS"
            assert "qos_status" in qos_status
            assert "current_metrics" in qos_status

            logger.info(
                f"Scenario '{scenario['name']}': allocated {len(allocation_plan.primary_nodes)} nodes, cost=${float(allocation_plan.total_cost):.2f}"
            )

        logger.info("✓ Dynamic resource allocation test passed")

    async def test_marketplace_api_endpoints(self):
        """Test marketplace API endpoints for federated workloads"""

        logger.info("Testing marketplace API endpoints...")

        # Test pricing quote endpoint
        quote_request = {
            "user_tier": "medium",
            "workload_type": "federated_inference",
            "model_size": "large",
            "requests_count": 500,
            "participants_needed": 15,
            "privacy_level": "high",
        }

        quote_response = await self.marketplace_api.get_pricing_quote(quote_request)
        assert quote_response is not None, "Failed to get pricing quote"
        assert "pricing_details" in quote_response
        assert "valid_until" in quote_response

        # Test federated inference request
        inference_request = {
            "requester_id": "api_test_user_001",
            "user_tier": "large",
            "model_size": "xlarge",
            "requests_count": 1000,
            "participants_needed": 25,
            "privacy_level": "critical",
            "max_latency_ms": 100.0,
            "max_budget": 2500.0,
            "preferred_regions": ["us-east", "us-west"],
        }

        inference_response = await self.marketplace_api.request_federated_inference(inference_request)
        assert inference_response is not None, "Failed to request federated inference"
        assert "request_id" in inference_response
        assert "allocation_id" in inference_response
        assert inference_response["status"] == "submitted"

        # Test federated training request
        training_request = {
            "requester_id": "api_test_researcher_001",
            "user_tier": "enterprise",
            "model_size": "xlarge",
            "duration_hours": 12.0,
            "participants_needed": 50,
            "privacy_level": "critical",
            "reliability_requirement": "guaranteed",
            "min_trust_score": 0.9,
            "max_budget": 10000.0,
            "preferred_regions": ["us-west", "eu-west"],
        }

        training_response = await self.marketplace_api.request_federated_training(training_request)
        assert training_response is not None, "Failed to request federated training"
        assert "request_id" in training_response
        assert "allocation_id" in training_response
        assert training_response["status"] == "submitted"

        # Test request status monitoring
        await asyncio.sleep(2)  # Allow processing time

        inference_status = await self.marketplace_api.get_request_status(inference_response["request_id"])
        assert inference_status is not None, "Failed to get inference request status"
        assert "request_id" in inference_status
        assert "status" in inference_status

        training_status = await self.marketplace_api.get_request_status(training_response["request_id"])
        assert training_status is not None, "Failed to get training request status"
        assert "request_id" in training_status
        assert "status" in training_status

        # Test pricing tiers endpoint
        pricing_tiers = await self.marketplace_api.get_pricing_tiers()
        assert pricing_tiers is not None, "Failed to get pricing tiers"
        assert "pricing_tiers" in pricing_tiers
        assert "small" in pricing_tiers["pricing_tiers"]
        assert "medium" in pricing_tiers["pricing_tiers"]
        assert "large" in pricing_tiers["pricing_tiers"]
        assert "enterprise" in pricing_tiers["pricing_tiers"]

        # Test market analytics
        market_analytics = await self.marketplace_api.get_market_analytics()
        assert market_analytics is not None, "Failed to get market analytics"
        assert "api_metrics" in market_analytics
        assert "active_requests" in market_analytics["api_metrics"]

        logger.info("✓ Marketplace API endpoints test passed")

    async def test_performance_and_scalability(self):
        """Test marketplace performance and scalability"""

        logger.info("Testing marketplace performance and scalability...")

        # Test concurrent request handling
        concurrent_requests = []

        for i in range(10):  # Create 10 concurrent requests
            request = {
                "requester_id": f"perf_test_user_{i:03d}",
                "user_tier": "medium",
                "model_size": "medium",
                "requests_count": 100,
                "participants_needed": 5,
                "privacy_level": "medium",
                "max_latency_ms": 200.0,
                "max_budget": 100.0,
            }

            task = asyncio.create_task(self.marketplace_api.request_federated_inference(request))
            concurrent_requests.append(task)

        # Wait for all requests to complete
        start_time = datetime.now(UTC)
        results = await asyncio.gather(*concurrent_requests, return_exceptions=True)
        end_time = datetime.now(UTC)

        processing_time = (end_time - start_time).total_seconds()
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))

        assert successful_requests >= 8, f"Only {successful_requests}/10 concurrent requests succeeded"
        assert processing_time < 30.0, f"Concurrent processing took {processing_time:.2f}s, expected < 30s"

        logger.info(f"Concurrent requests: {successful_requests}/10 succeeded in {processing_time:.2f}s")

        # Test market statistics collection performance
        stats_start = datetime.now(UTC)

        market_stats = await self.resource_allocator.get_market_statistics()
        pricing_analytics = await self.pricing_manager.get_market_analytics()

        stats_time = (datetime.now(UTC) - stats_start).total_seconds()

        assert market_stats is not None, "Failed to get market statistics"
        assert pricing_analytics is not None, "Failed to get pricing analytics"
        assert stats_time < 5.0, f"Statistics collection took {stats_time:.2f}s, expected < 5s"

        logger.info(f"Statistics collection completed in {stats_time:.2f}s")

        # Test resource discovery scaling
        discovery_start = datetime.now(UTC)

        large_discovery_req = {"cpu_cores": 2.0, "memory_gb": 4.0, "max_budget": 50.0, "workload_type": "inference"}

        discovered_resources = await self.p2p_integration.discovery.discover_resources(
            requirements=large_discovery_req, max_results=50, timeout_seconds=20
        )

        discovery_time = (datetime.now(UTC) - discovery_start).total_seconds()

        assert discovery_time < 20.0, f"Resource discovery took {discovery_time:.2f}s, expected < 20s"
        assert len(discovered_resources) > 0, "No resources discovered in scalability test"

        logger.info(f"Discovered {len(discovered_resources)} resources in {discovery_time:.2f}s")

        logger.info("✓ Performance and scalability test passed")

    async def test_full_federated_workflow(self):
        """Test complete end-to-end federated AI workflow"""

        logger.info("Testing complete federated AI workflow...")

        # Scenario: University research team wants to run federated training
        # across multiple institutions with high privacy requirements

        workflow_steps = []
        start_time = datetime.now(UTC)

        # Step 1: Get pricing quote for federated training
        quote_start = datetime.now(UTC)
        quote = await self.pricing_manager.get_federated_training_price(
            user_tier=UserSizeTier.LARGE,
            model_size="xlarge",
            duration_hours=10.0,
            participants_needed=30,
            privacy_level="critical",
            reliability_requirement="guaranteed",
        )
        quote_time = (datetime.now(UTC) - quote_start).total_seconds()
        workflow_steps.append(f"Pricing quote: {quote_time:.2f}s")

        assert quote["total_cost"] > 0, "Invalid pricing quote"
        assert quote["tier_info"]["dedicated_support"], "Enterprise features not applied"

        # Step 2: Create federated training auction
        auction_start = datetime.now(UTC)
        auction_id = await create_federated_training_auction(
            requester_id="university_research_team",
            model_size="xlarge",
            participants_needed=30,
            duration_hours=10.0,
            privacy_level="critical",
            reliability_requirement="guaranteed",
            reserve_price=float(quote["total_cost"]) * 0.9,  # 10% below quote
        )
        auction_time = (datetime.now(UTC) - auction_start).total_seconds()
        workflow_steps.append(f"Auction creation: {auction_time:.2f}s")

        assert auction_id is not None, "Failed to create training auction"

        # Step 3: Announce auction and collect bids via P2P
        p2p_start = datetime.now(UTC)

        auction_info = {
            "auction_id": auction_id,
            "auction_type": "federated_training",
            "requirements": {"model_size": "xlarge", "participants_needed": 30, "privacy_level": "critical"},
            "reserve_price": float(quote["total_cost"]) * 0.9,
        }

        p2p_success = await self.p2p_integration.announce_auction(auction_info)
        assert p2p_success, "Failed to announce auction via P2P"

        # Simulate participant recruitment
        participants = await self.p2p_integration.discovery.find_federated_participants(
            job_requirements={
                "job_type": "federated_training",
                "model_size": "xlarge",
                "privacy_level": "critical",
                "min_trust_score": 0.8,
                "compensation": quote["total_cost"] / 30,  # Per participant
            },
            min_participants=25,
            max_participants=35,
        )

        p2p_time = (datetime.now(UTC) - p2p_start).total_seconds()
        workflow_steps.append(f"P2P recruitment: {p2p_time:.2f}s, {len(participants)} participants")

        assert len(participants) >= 25, f"Insufficient participants: {len(participants)}"

        # Step 4: Allocate resources for selected participants
        allocation_start = datetime.now(UTC)

        requirements = ResourceRequirement(
            cpu_cores=Decimal("32.0"),
            memory_gb=Decimal("128.0"),
            storage_gb=Decimal("1000.0"),
            bandwidth_mbps=Decimal("1000.0"),
            duration_hours=Decimal("10.0"),
            participants_needed=len(participants),
            workload_type="training",
            privacy_level="critical",
        )

        qos_requirements = QoSRequirement(
            max_latency_ms=Decimal("50.0"),
            min_availability_percentage=Decimal("99.9"),
            max_cost_per_hour=Decimal("2000.0"),
            privacy_level="critical",
            reliability_requirement="guaranteed",
        )

        # Use the resource allocator to create allocation plan
        discovered_nodes = await self.resource_allocator.discover_resources(requirements, qos_requirements)

        allocation_plan = await self.resource_allocator.create_allocation_plan(
            requirements, qos_requirements, discovered_nodes, AllocationStrategy.BEST_QUALITY
        )

        allocation_id = await self.resource_allocator.execute_allocation_plan(
            allocation_plan, "university_research_team"
        )

        allocation_time = (datetime.now(UTC) - allocation_start).total_seconds()
        workflow_steps.append(f"Resource allocation: {allocation_time:.2f}s")

        assert allocation_id is not None, "Failed to allocate resources"

        # Step 5: Monitor training progress and QoS
        monitoring_start = datetime.now(UTC)

        qos_status = await self.resource_allocator.monitor_allocation_qos(allocation_id)
        allocation_status = await self.resource_allocator.get_allocation_status(allocation_id)

        monitoring_time = (datetime.now(UTC) - monitoring_start).total_seconds()
        workflow_steps.append(f"QoS monitoring: {monitoring_time:.2f}s")

        assert qos_status["sla_compliance"], "SLA compliance issues detected"
        assert allocation_status["compliance"]["sla_compliant"], "Allocation not SLA compliant"

        # Step 6: Generate workflow summary
        total_time = (datetime.now(UTC) - start_time).total_seconds()

        workflow_summary = {
            "workflow_type": "federated_training",
            "user_tier": "large",
            "model_size": "xlarge",
            "participants": len(participants),
            "total_cost": float(quote["total_cost"]),
            "total_time_seconds": total_time,
            "steps": workflow_steps,
            "success": True,
            "qos_compliance": qos_status["sla_compliance"],
            "resource_utilization": {
                "primary_nodes": len(allocation_plan.primary_nodes),
                "backup_nodes": len(allocation_plan.backup_nodes),
                "expected_quality": float(allocation_plan.expected_quality_score),
            },
        }

        logger.info("✓ Full federated workflow completed successfully:")
        logger.info(f"  - Total time: {total_time:.2f}s")
        logger.info(f"  - Participants: {len(participants)}")
        logger.info(f"  - Total cost: ${quote['total_cost']:.2f}")
        logger.info(f"  - QoS compliant: {qos_status['sla_compliance']}")

        for step in workflow_steps:
            logger.info(f"  - {step}")

        # Verify workflow completed within reasonable time
        assert total_time < 120.0, f"Workflow took {total_time:.2f}s, expected < 120s"
        assert workflow_summary["success"], "Workflow did not complete successfully"

        return workflow_summary


# Integration test fixtures and utilities


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_comprehensive_marketplace_integration():
    """Main integration test that runs all test scenarios"""

    logger.info("=" * 80)
    logger.info("STARTING COMPREHENSIVE FOG MARKETPLACE INTEGRATION TEST")
    logger.info("=" * 80)

    test_suite = TestFogMarketplaceIntegration()
    await test_suite.setup_marketplace()

    try:
        # Run all test scenarios
        await test_suite.test_size_tier_pricing_system()
        await test_suite.test_p2p_resource_discovery()
        await test_suite.test_federated_auction_workflow()
        await test_suite.test_dynamic_resource_allocation()
        await test_suite.test_marketplace_api_endpoints()
        await test_suite.test_performance_and_scalability()

        # Run the complete end-to-end workflow test
        workflow_summary = await test_suite.test_full_federated_workflow()

        logger.info("=" * 80)
        logger.info("ALL MARKETPLACE INTEGRATION TESTS PASSED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("Key Results:")
        logger.info("  ✓ Size-tier pricing system working correctly")
        logger.info("  ✓ P2P resource discovery and participant matching operational")
        logger.info("  ✓ Federated auction workflow functional")
        logger.info("  ✓ Dynamic resource allocation with QoS guarantees")
        logger.info("  ✓ Marketplace API endpoints responding correctly")
        logger.info("  ✓ Performance and scalability requirements met")
        logger.info(f"  ✓ Complete federated workflow: {workflow_summary['total_time_seconds']:.2f}s")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise

    finally:
        # Cleanup
        if hasattr(test_suite, "p2p_integration"):
            await test_suite.p2p_integration.stop()


if __name__ == "__main__":
    # Run the comprehensive integration test
    asyncio.run(test_comprehensive_marketplace_integration())
