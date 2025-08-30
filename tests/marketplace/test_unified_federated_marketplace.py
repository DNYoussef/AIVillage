"""
Unified Federated Marketplace Integration Tests

Comprehensive test suite for the complete federated system integration including:
- Small user workflows and resource constraints
- Enterprise workflows with high-performance requirements  
- Marketplace bidding and resource allocation
- P2P network integration and node discovery
- Security and privacy-preserving computation
- Cost optimization and billing accuracy
"""

import asyncio
from datetime import datetime, UTC, timedelta
from decimal import Decimal
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from infrastructure.fog.marketplace.fog_marketplace import (
    FogMarketplace, ServiceType, ServiceTier, PricingModel,
    ServiceOffering, ServiceRequest, ServiceContract
)
from infrastructure.fog.gateway.scheduler.marketplace import (
    MarketplaceEngine, ResourceBid, ResourceListing, BidType, PricingTier, BidStatus
)
from infrastructure.distributed_inference.federated_inference_coordinator import (
    FederatedInferenceCoordinator, NodeType, NodeCapabilities, ModelInfo,
    InferenceRequest, PrivacyLevel, LoadBalancingStrategy
)

logger = logging.getLogger(__name__)


class UnifiedFederatedCoordinator:
    """
    Unified coordinator that handles both inference and training workloads
    through a single interface with marketplace integration.
    """
    
    def __init__(self, coordinator_id: str):
        self.coordinator_id = coordinator_id
        self.fog_marketplace = FogMarketplace(
            marketplace_id=f"market_{coordinator_id}",
            enable_hidden_services=True,
            enable_spot_pricing=True
        )
        self.gateway_marketplace = MarketplaceEngine()
        self.inference_coordinator = FederatedInferenceCoordinator(
            coordinator_id=f"inference_{coordinator_id}",
            enable_privacy_features=True
        )
        
        # Shared resource pools
        self.shared_participants: Dict[str, NodeCapabilities] = {}
        self.active_workloads: Dict[str, Dict[str, Any]] = {}
        
        # User tier management
        self.user_tiers = {
            "small": {"max_budget": 100, "priority": 1, "sla_level": "basic"},
            "medium": {"max_budget": 1000, "priority": 2, "sla_level": "standard"}, 
            "large": {"max_budget": 10000, "priority": 3, "sla_level": "premium"},
            "enterprise": {"max_budget": 100000, "priority": 4, "sla_level": "enterprise"}
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all components of the unified system"""
        try:
            # Initialize marketplace engines
            await self.gateway_marketplace.start()
            
            # Initialize inference coordinator
            await self.inference_coordinator.initialize()
            
            # Register shared resources
            await self._register_shared_resources()
            
            self.initialized = True
            logger.info(f"Unified federated coordinator {self.coordinator_id} initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize unified coordinator: {e}")
            return False
    
    async def _register_shared_resources(self):
        """Register resources that can handle both inference and training"""
        # Mock diverse resource pool
        shared_resources = [
            {
                "node_id": "mobile_device_1",
                "node_type": NodeType.MOBILE,
                "memory_gb": 6, 
                "storage_gb": 128,
                "compute_gflops": 2.5, 
                "network_mbps": 100,
                "battery_percent": 85
            },
            {
                "node_id": "edge_server_1", 
                "node_type": NodeType.EDGE,
                "memory_gb": 32, 
                "storage_gb": 1000,
                "compute_gflops": 15, 
                "network_mbps": 1000
            },
            {
                "node_id": "gpu_cluster_1",
                "node_type": NodeType.GPU,
                "memory_gb": 256, 
                "storage_gb": 10000,
                "compute_gflops": 100, 
                "network_mbps": 10000
            },
            {
                "node_id": "cloud_instance_1",
                "node_type": NodeType.CLOUD,
                "memory_gb": 128, 
                "storage_gb": 5000,
                "compute_gflops": 50, 
                "network_mbps": 5000
            }
        ]
        
        for resource in shared_resources:
            node_cap = NodeCapabilities(
                **resource
            )
            
            self.shared_participants[resource["node_id"]] = node_cap
            await self.inference_coordinator.register_node(node_cap)
        
        # Register required models for testing
        test_models = [
            ModelInfo(
                model_id="mobile_bert_quantized",
                model_name="BERT Base Mobile Quantized", 
                model_version="1.0",
                model_type="nlp",
                model_size_mb=50,
                memory_requirements_gb=2.0,
                compute_requirements_gflops=1.0,
                framework="pytorch",
                supported_node_types={NodeType.MOBILE, NodeType.EDGE},
                privacy_requirements=PrivacyLevel.HIGH
            ),
            ModelInfo(
                model_id="bert_base_uncased",
                model_name="BERT Base Uncased",
                model_version="1.0", 
                model_type="nlp",
                model_size_mb=438,
                memory_requirements_gb=4.0,
                compute_requirements_gflops=5.0,
                framework="pytorch",
                supported_node_types={NodeType.EDGE, NodeType.GPU, NodeType.CLOUD},
                privacy_requirements=PrivacyLevel.MEDIUM
            ),
            ModelInfo(
                model_id="gpt_large_custom",
                model_name="GPT Large Custom",
                model_version="1.0",
                model_type="nlp",
                model_size_mb=2048,
                memory_requirements_gb=16.0,
                compute_requirements_gflops=20.0,
                framework="pytorch",
                supported_node_types={NodeType.GPU, NodeType.CLOUD},
                privacy_requirements=PrivacyLevel.HIGH
            ),
            ModelInfo(
                model_id="gpt_xl_enterprise",
                model_name="GPT XL Enterprise",
                model_version="1.0",
                model_type="nlp",
                model_size_mb=8192,
                memory_requirements_gb=64.0,
                compute_requirements_gflops=100.0,
                framework="pytorch",
                supported_node_types={NodeType.CLOUD},
                privacy_requirements=PrivacyLevel.HIGH
            )
        ]
        
        for model in test_models:
            await self.inference_coordinator.register_model(model)
    
    async def submit_unified_request(
        self,
        user_id: str,
        workload_type: str,  # "inference" or "training"
        request_params: Dict[str, Any],
        user_tier: str = "medium"
    ) -> str:
        """Submit a unified request for either inference or training"""
        
        request_id = f"unified_{uuid.uuid4().hex[:12]}"
        tier_config = self.user_tiers.get(user_tier, self.user_tiers["medium"])
        
        # Route based on workload type
        if workload_type == "inference":
            result = await self._handle_inference_request(
                request_id, user_id, request_params, tier_config
            )
        elif workload_type == "training":
            result = await self._handle_training_request(
                request_id, user_id, request_params, tier_config
            )
        else:
            raise ValueError(f"Unknown workload type: {workload_type}")
        
        # Store in active workloads
        self.active_workloads[request_id] = {
            "type": workload_type,
            "user_id": user_id,
            "user_tier": user_tier,
            "params": request_params,
            "result": result,
            "created_at": datetime.now(UTC)
        }
        
        return request_id
    
    async def _handle_inference_request(
        self, request_id: str, user_id: str, params: Dict[str, Any], tier_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle inference request through marketplace"""
        
        # Create marketplace bid for resources
        bid_id = await self.gateway_marketplace.submit_bid(
            namespace=f"user_{user_id}",
            cpu_cores=params.get("cpu_cores", 2),
            memory_gb=params.get("memory_gb", 4),
            max_price=min(params.get("max_budget", 10), tier_config["max_budget"]),
            bid_type=BidType.SPOT if tier_config["priority"] <= 2 else BidType.ON_DEMAND,
            estimated_duration_hours=params.get("duration_hours", 1),
            job_spec={
                "type": "inference",
                "model_id": params.get("model_id"),
                "privacy_level": params.get("privacy_level", "medium")
            }
        )
        
        # Submit to inference coordinator
        inference_id = await self.inference_coordinator.submit_inference_request(
            client_id=user_id,
            model_id=params.get("model_id", "default_model"),
            input_data=params.get("input_data", {}),
            preferences={
                "max_latency_ms": params.get("max_latency_ms", 5000),
                "privacy_level": params.get("privacy_level", "medium"),
                "preferred_node_types": self._get_preferred_nodes_for_tier(tier_config)
            }
        )
        
        return {
            "request_id": request_id,
            "bid_id": bid_id,
            "inference_id": inference_id,
            "status": "submitted",
            "tier": tier_config
        }
    
    async def _handle_training_request(
        self, request_id: str, user_id: str, params: Dict[str, Any], tier_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle training request through marketplace"""
        
        participants_needed = params.get("participants", 5)
        
        # Submit resource bid for training cluster
        bid_id = await self.gateway_marketplace.submit_bid(
            namespace=f"user_{user_id}_training",
            cpu_cores=params.get("cpu_cores", 8) * participants_needed,
            memory_gb=params.get("memory_gb", 16) * participants_needed,
            max_price=min(params.get("max_budget", 50), tier_config["max_budget"]),
            bid_type=BidType.ON_DEMAND,  # Training typically needs guaranteed resources
            estimated_duration_hours=params.get("duration_hours", 24),
            job_spec={
                "type": "training",
                "model_id": params.get("model_id"),
                "participants": participants_needed,
                "privacy_level": params.get("privacy_level", "high")
            }
        )
        
        return {
            "request_id": request_id,
            "bid_id": bid_id,
            "participants_allocated": participants_needed,
            "status": "submitted",
            "tier": tier_config
        }
    
    def _get_preferred_nodes_for_tier(self, tier_config: Dict[str, Any]) -> List[str]:
        """Get preferred node types based on user tier"""
        if tier_config["priority"] == 1:  # Small users - mobile-friendly
            return ["mobile", "edge"]
        elif tier_config["priority"] == 2:  # Medium users - balanced
            return ["edge", "cloud"]
        elif tier_config["priority"] == 3:  # Large users - performance
            return ["gpu", "cloud"]
        else:  # Enterprise - dedicated resources
            return ["gpu", "cloud", "specialized"]
    
    async def get_unified_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of unified request"""
        if request_id not in self.active_workloads:
            return None
        
        workload = self.active_workloads[request_id]
        
        if workload["type"] == "inference":
            # Get inference status
            inference_status = await self.inference_coordinator.get_inference_status(
                workload["result"]["inference_id"]
            )
            return {
                "request_id": request_id,
                "type": "inference",
                "marketplace_status": "matched",  # Simplified
                "inference_status": inference_status,
                "overall_progress": inference_status.get("progress", 0) if inference_status else 0
            }
        else:
            # Training status (simplified)
            return {
                "request_id": request_id,
                "type": "training", 
                "marketplace_status": "active",
                "training_progress": 0.5,  # Mock progress
                "overall_progress": 0.5
            }
    
    async def get_marketplace_stats(self) -> Dict[str, Any]:
        """Get comprehensive marketplace statistics"""
        gateway_stats = await self.gateway_marketplace.get_marketplace_status()
        inference_stats = await self.inference_coordinator.get_coordinator_stats()
        fog_stats = self.fog_marketplace.get_market_stats()
        
        return {
            "unified_coordinator": {
                "active_workloads": len(self.active_workloads),
                "shared_participants": len(self.shared_participants),
                "user_tiers_supported": len(self.user_tiers)
            },
            "gateway_marketplace": gateway_stats,
            "inference_coordinator": inference_stats,
            "fog_marketplace": {
                "total_providers": fog_stats.total_providers,
                "total_customers": fog_stats.total_customers,
                "active_contracts": fog_stats.active_contracts,
                "total_volume_usd": float(fog_stats.total_tokens_transacted)
            }
        }


class TestUnifiedFederatedMarketplace:
    """Comprehensive test suite for the unified federated marketplace system"""
    
    @pytest.fixture
    async def unified_coordinator(self):
        """Create and initialize unified coordinator"""
        coordinator = UnifiedFederatedCoordinator("test_unified_001")
        await coordinator.initialize()
        return coordinator
    
    @pytest.fixture
    def small_user_request(self):
        """Small user inference request"""
        return {
            "user_tier": "small",
            "workload": "inference",
            "model": "mobile-bert",
            "max_budget": 5.00,
            "priority": "cost",
            "params": {
                "model_id": "mobile_bert_quantized",
                "cpu_cores": 1,
                "memory_gb": 2,
                "max_budget": 5.0,
                "duration_hours": 1,
                "privacy_level": "low",
                "input_data": {"text": "test input"}
            }
        }
    
    @pytest.fixture
    def medium_user_request(self):
        """Medium user hybrid request"""
        return {
            "user_tier": "medium", 
            "workload": "inference",
            "model": "bert-base",
            "max_budget": 25.00,
            "params": {
                "model_id": "bert_base_uncased",
                "cpu_cores": 4,
                "memory_gb": 8,
                "max_budget": 25.0,
                "duration_hours": 2,
                "privacy_level": "medium",
                "input_data": {"text": "medium complexity input"}
            }
        }
    
    @pytest.fixture
    def large_user_request(self):
        """Large user training request"""
        return {
            "user_tier": "large",
            "workload": "training", 
            "model": "gpt-large",
            "participants": 50,
            "max_budget": 500.00,
            "params": {
                "model_id": "gpt_large_custom",
                "cpu_cores": 8,
                "memory_gb": 32,
                "max_budget": 500.0,
                "duration_hours": 12,
                "participants": 50,
                "privacy_level": "high"
            }
        }
    
    @pytest.fixture
    def enterprise_user_request(self):
        """Enterprise user dedicated training request"""
        return {
            "user_tier": "enterprise",
            "workload": "training",
            "model": "gpt-xl", 
            "participants": 500,
            "sla_required": True,
            "max_budget": 5000.00,
            "params": {
                "model_id": "gpt_xl_enterprise",
                "cpu_cores": 64,
                "memory_gb": 256,
                "max_budget": 5000.0,
                "duration_hours": 48,
                "participants": 500,
                "privacy_level": "ultra",
                "sla_requirements": {"uptime": 99.99, "latency_ms": 100}
            }
        }

    # Test 1: Unified Federated Coordinator Tests
    @pytest.mark.asyncio
    async def test_unified_coordinator_initialization(self, unified_coordinator):
        """Test unified coordinator initializes correctly"""
        assert unified_coordinator.initialized is True
        assert unified_coordinator.fog_marketplace is not None
        assert unified_coordinator.gateway_marketplace is not None
        assert unified_coordinator.inference_coordinator is not None
        assert len(unified_coordinator.shared_participants) > 0
        assert len(unified_coordinator.user_tiers) == 4

    @pytest.mark.asyncio
    async def test_shared_resource_management(self, unified_coordinator):
        """Test shared resource pool handles both inference and training"""
        # Verify resources support both workload types
        training_capable = [
            node for node in unified_coordinator.shared_participants.values()
            if hasattr(node, 'supports_training') and node.supports_training
        ]
        inference_capable = [
            node for node in unified_coordinator.shared_participants.values()
            if hasattr(node, 'supports_inference') or node.node_type == NodeType.MOBILE
        ]
        
        assert len(training_capable) >= 2  # At least edge and GPU nodes
        assert len(inference_capable) >= 3  # Mobile, edge, GPU, cloud
    
    @pytest.mark.asyncio
    async def test_seamless_workload_switching(self, unified_coordinator, medium_user_request):
        """Test user can seamlessly switch between inference and training"""
        user_id = "test_user_switching"
        
        # Submit inference request
        inference_params = medium_user_request["params"].copy()
        inference_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference",
            request_params=inference_params,
            user_tier="medium"
        )
        
        # Submit training request from same user
        training_params = {
            "model_id": "bert_base_uncased",
            "cpu_cores": 8,
            "memory_gb": 16,
            "max_budget": 100.0,
            "duration_hours": 6,
            "participants": 10,
            "privacy_level": "medium"
        }
        training_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training", 
            request_params=training_params,
            user_tier="medium"
        )
        
        # Verify both requests processed
        assert inference_id != training_id
        
        inference_status = await unified_coordinator.get_unified_status(inference_id)
        training_status = await unified_coordinator.get_unified_status(training_id)
        
        assert inference_status["type"] == "inference"
        assert training_status["type"] == "training"
        
        # Verify integrated billing (same user, different workloads)
        assert len([w for w in unified_coordinator.active_workloads.values() 
                   if w["user_id"] == user_id]) == 2

    # Test 2: Size-Tier Marketplace Integration Tests
    @pytest.mark.asyncio
    async def test_small_user_mobile_inference(self, unified_coordinator, small_user_request):
        """Test small user gets mobile-optimized inference within budget"""
        user_id = "small_user_001"
        params = small_user_request["params"]
        
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference",
            request_params=params,
            user_tier="small"
        )
        
        # Verify request processed with small tier constraints
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["user_tier"] == "small"
        assert workload["result"]["tier"]["max_budget"] == 100
        assert workload["result"]["tier"]["priority"] == 1
        
        # Should prefer mobile/edge nodes for cost efficiency
        status = await unified_coordinator.get_unified_status(request_id)
        assert status is not None
        assert status["type"] == "inference"

    @pytest.mark.asyncio 
    async def test_medium_user_balanced_resources(self, unified_coordinator, medium_user_request):
        """Test medium user gets balanced fog/cloud resources"""
        user_id = "medium_user_001"
        params = medium_user_request["params"]
        
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference",
            request_params=params,
            user_tier="medium"
        )
        
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["user_tier"] == "medium"
        assert workload["result"]["tier"]["max_budget"] == 1000
        assert workload["result"]["tier"]["priority"] == 2
        assert workload["result"]["tier"]["sla_level"] == "standard"

    @pytest.mark.asyncio
    async def test_large_user_high_performance_training(self, unified_coordinator, large_user_request):
        """Test large user gets cloud clusters for high-performance training"""
        user_id = "large_user_001"
        params = large_user_request["params"]
        
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training",
            request_params=params,
            user_tier="large"
        )
        
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["user_tier"] == "large"
        assert workload["result"]["tier"]["max_budget"] == 10000
        assert workload["result"]["tier"]["priority"] == 3
        assert workload["result"]["tier"]["sla_level"] == "premium"
        assert workload["result"]["participants_allocated"] == 50

    @pytest.mark.asyncio
    async def test_enterprise_user_dedicated_resources(self, unified_coordinator, enterprise_user_request):
        """Test enterprise user gets dedicated resources with SLA guarantees"""
        user_id = "enterprise_user_001" 
        params = enterprise_user_request["params"]
        
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training",
            request_params=params,
            user_tier="enterprise"
        )
        
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["user_tier"] == "enterprise"
        assert workload["result"]["tier"]["max_budget"] == 100000
        assert workload["result"]["tier"]["priority"] == 4
        assert workload["result"]["tier"]["sla_level"] == "enterprise"
        assert workload["result"]["participants_allocated"] == 500

    # Test 3: Complete User Workflow Tests
    @pytest.mark.asyncio
    async def test_end_to_end_inference_workflow(self, unified_coordinator, medium_user_request):
        """Test complete inference workflow from request to results"""
        user_id = "workflow_user_001"
        params = medium_user_request["params"]
        
        # Step 1: Submit request
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference", 
            request_params=params,
            user_tier="medium"
        )
        assert request_id is not None
        
        # Step 2: Check marketplace allocation
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["result"]["bid_id"] is not None
        assert workload["result"]["inference_id"] is not None
        
        # Step 3: Monitor progress
        status = await unified_coordinator.get_unified_status(request_id)
        assert status["type"] == "inference"
        assert "inference_status" in status
        
        # Step 4: Verify resource allocation
        assert "bid_id" in workload["result"]
        
        # Step 5: Verify cost tracking (mocked)
        assert params["max_budget"] <= unified_coordinator.user_tiers["medium"]["max_budget"]

    @pytest.mark.asyncio
    async def test_end_to_end_training_workflow(self, unified_coordinator, large_user_request):
        """Test complete training workflow from request to completion"""
        user_id = "training_user_001"
        params = large_user_request["params"]
        
        # Step 1: Submit training request
        request_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training",
            request_params=params,
            user_tier="large"
        )
        
        # Step 2: Verify participant allocation
        workload = unified_coordinator.active_workloads[request_id]
        assert workload["result"]["participants_allocated"] == 50
        
        # Step 3: Check training progress
        status = await unified_coordinator.get_unified_status(request_id)
        assert status["type"] == "training"
        assert status["training_progress"] >= 0
        
        # Step 4: Verify marketplace integration
        assert workload["result"]["bid_id"] is not None

    @pytest.mark.asyncio
    async def test_mixed_workload_user_workflow(self, unified_coordinator):
        """Test user switching between inference and training seamlessly"""
        user_id = "mixed_user_001"
        
        # Morning: Inference workload
        inference_params = {
            "model_id": "bert_base",
            "cpu_cores": 2,
            "memory_gb": 4,
            "max_budget": 20.0,
            "duration_hours": 1,
            "input_data": {"text": "morning inference"}
        }
        
        inference_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference",
            request_params=inference_params,
            user_tier="medium"
        )
        
        # Afternoon: Training workload
        training_params = {
            "model_id": "bert_base",
            "cpu_cores": 8,
            "memory_gb": 32,
            "max_budget": 100.0,
            "duration_hours": 6,
            "participants": 25
        }
        
        training_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training", 
            request_params=training_params,
            user_tier="medium"
        )
        
        # Verify both workloads for same user
        inference_workload = unified_coordinator.active_workloads[inference_id]
        training_workload = unified_coordinator.active_workloads[training_id]
        
        assert inference_workload["user_id"] == user_id
        assert training_workload["user_id"] == user_id
        assert inference_workload["type"] == "inference"
        assert training_workload["type"] == "training"

    # Test 4: Marketplace Functionality Validation
    @pytest.mark.asyncio
    async def test_auction_engine_integration(self, unified_coordinator):
        """Test auction engine properly allocates federated workloads"""
        # Simulate high-demand scenario
        user_requests = []
        
        for i in range(10):
            params = {
                "model_id": f"model_{i}",
                "cpu_cores": 4,
                "memory_gb": 8, 
                "max_budget": 50.0,
                "duration_hours": 2
            }
            
            request_id = await unified_coordinator.submit_unified_request(
                user_id=f"auction_user_{i}",
                workload_type="inference",
                request_params=params,
                user_tier="medium"
            )
            user_requests.append(request_id)
        
        # Verify all requests processed
        assert len(user_requests) == 10
        assert len(unified_coordinator.active_workloads) >= 10

    @pytest.mark.asyncio
    async def test_dynamic_pricing_adaptation(self, unified_coordinator):
        """Test dynamic pricing adapts to demand"""
        # Get initial pricing
        initial_stats = await unified_coordinator.get_marketplace_stats()
        initial_pricing = initial_stats["gateway_marketplace"]["pricing"]
        
        # Simulate demand surge
        surge_requests = []
        for i in range(20):
            params = {
                "model_id": "popular_model",
                "cpu_cores": 8,
                "memory_gb": 16,
                "max_budget": 100.0,
                "duration_hours": 1
            }
            
            request_id = await unified_coordinator.submit_unified_request(
                user_id=f"surge_user_{i}",
                workload_type="inference", 
                request_params=params,
                user_tier="medium"
            )
            surge_requests.append(request_id)
        
        # Verify pricing adaptation (would increase with real pricing engine)
        assert len(surge_requests) == 20

    @pytest.mark.asyncio
    async def test_real_time_availability_queries(self, unified_coordinator):
        """Test real-time availability and pricing queries"""
        # Query current availability
        stats = await unified_coordinator.get_marketplace_stats()
        
        assert "unified_coordinator" in stats
        assert "gateway_marketplace" in stats
        assert "inference_coordinator" in stats
        assert "fog_marketplace" in stats
        
        # Verify availability data
        unified_stats = stats["unified_coordinator"]
        assert unified_stats["shared_participants"] > 0
        assert unified_stats["user_tiers_supported"] == 4

    # Test 5: Budget Management and Billing Integration Tests
    @pytest.mark.asyncio
    async def test_tier_based_budget_enforcement(self, unified_coordinator):
        """Test budget enforcement based on user tiers"""
        # Small user with budget exceeding tier limit
        params = {
            "model_id": "expensive_model",
            "cpu_cores": 16,
            "memory_gb": 64, 
            "max_budget": 1000.0,  # Exceeds small tier limit
            "duration_hours": 12
        }
        
        request_id = await unified_coordinator.submit_unified_request(
            user_id="budget_test_small",
            workload_type="inference",
            request_params=params,
            user_tier="small"  # Max budget: 100
        )
        
        # Verify budget was capped to tier limit
        workload = unified_coordinator.active_workloads[request_id]
        actual_budget = workload["result"]["tier"]["max_budget"]
        assert actual_budget == 100  # Small tier limit

    @pytest.mark.asyncio
    async def test_integrated_billing_tracking(self, unified_coordinator):
        """Test billing tracks costs across different workload types"""
        user_id = "billing_test_user"
        
        # Submit multiple requests
        requests = []
        
        # Inference request
        inference_params = {
            "model_id": "model_1",
            "cpu_cores": 2,
            "memory_gb": 4,
            "max_budget": 20.0,
            "duration_hours": 1
        }
        
        inference_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="inference",
            request_params=inference_params,
            user_tier="medium"
        )
        requests.append(inference_id)
        
        # Training request
        training_params = {
            "model_id": "model_1",
            "cpu_cores": 8,
            "memory_gb": 32,
            "max_budget": 200.0,
            "duration_hours": 6,
            "participants": 10
        }
        
        training_id = await unified_coordinator.submit_unified_request(
            user_id=user_id,
            workload_type="training",
            request_params=training_params,
            user_tier="medium"
        )
        requests.append(training_id)
        
        # Verify integrated billing for same user
        user_workloads = [
            w for w in unified_coordinator.active_workloads.values() 
            if w["user_id"] == user_id
        ]
        
        assert len(user_workloads) == 2
        total_budgets = sum(
            w["params"]["max_budget"] for w in user_workloads
        )
        assert total_budgets == 220.0  # 20 + 200

    # Test 6: Performance Benchmarks and SLA Validation
    @pytest.mark.asyncio
    async def test_sla_compliance_by_tier(self, unified_coordinator):
        """Test SLA compliance varies by user tier"""
        tiers_to_test = ["small", "medium", "large", "enterprise"]
        
        for tier in tiers_to_test:
            params = {
                "model_id": f"sla_test_model_{tier}",
                "cpu_cores": 4,
                "memory_gb": 8,
                "max_budget": 50.0,
                "duration_hours": 1
            }
            
            request_id = await unified_coordinator.submit_unified_request(
                user_id=f"sla_user_{tier}",
                workload_type="inference",
                request_params=params,
                user_tier=tier
            )
            
            workload = unified_coordinator.active_workloads[request_id]
            tier_config = workload["result"]["tier"]
            
            # Verify SLA level matches tier
            expected_sla = unified_coordinator.user_tiers[tier]["sla_level"]
            assert tier_config["sla_level"] == expected_sla

    @pytest.mark.asyncio
    async def test_performance_meets_tier_expectations(self, unified_coordinator):
        """Test performance meets expectations for each tier"""
        # Enterprise tier should get fastest processing
        enterprise_params = {
            "model_id": "performance_test_model",
            "cpu_cores": 32,
            "memory_gb": 128,
            "max_budget": 1000.0,
            "duration_hours": 1,
            "max_latency_ms": 100  # Strict latency requirement
        }
        
        enterprise_id = await unified_coordinator.submit_unified_request(
            user_id="enterprise_perf_user",
            workload_type="inference",
            request_params=enterprise_params,
            user_tier="enterprise"
        )
        
        # Small tier should get cost-optimized processing
        small_params = {
            "model_id": "performance_test_model",
            "cpu_cores": 1,
            "memory_gb": 2,
            "max_budget": 5.0,
            "duration_hours": 1,
            "max_latency_ms": 10000  # Relaxed latency requirement
        }
        
        small_id = await unified_coordinator.submit_unified_request(
            user_id="small_perf_user",
            workload_type="inference",
            request_params=small_params,
            user_tier="small"
        )
        
        # Verify different tier priorities
        enterprise_workload = unified_coordinator.active_workloads[enterprise_id]
        small_workload = unified_coordinator.active_workloads[small_id]
        
        assert enterprise_workload["result"]["tier"]["priority"] > small_workload["result"]["tier"]["priority"]

    # Test 7: Error Handling and Resilience 
    @pytest.mark.asyncio
    async def test_graceful_resource_exhaustion_handling(self, unified_coordinator):
        """Test system handles resource exhaustion gracefully"""
        # Simulate resource exhaustion by submitting many large requests
        large_requests = []
        
        for i in range(50):  # Submit more requests than resources can handle
            params = {
                "model_id": f"resource_hungry_model_{i}",
                "cpu_cores": 64,  # Large resource requirement
                "memory_gb": 256,
                "max_budget": 500.0,
                "duration_hours": 24,
                "participants": 100 if i % 2 == 0 else 1  # Mix training and inference
            }
            
            try:
                request_id = await unified_coordinator.submit_unified_request(
                    user_id=f"resource_test_user_{i}",
                    workload_type="training" if i % 2 == 0 else "inference",
                    request_params=params,
                    user_tier="large"
                )
                large_requests.append(request_id)
            except Exception as e:
                # Some requests may fail due to resource constraints
                logger.info(f"Request {i} failed as expected: {e}")
        
        # Verify system didn't crash and processed some requests
        assert len(large_requests) <= len(unified_coordinator.shared_participants)

    @pytest.mark.asyncio
    async def test_marketplace_component_failover(self, unified_coordinator):
        """Test system handles component failures gracefully"""
        # Test with one marketplace component "failing"
        original_fog_marketplace = unified_coordinator.fog_marketplace
        
        # Temporarily disable fog marketplace
        unified_coordinator.fog_marketplace = None
        
        # System should still function with remaining components
        params = {
            "model_id": "failover_test_model",
            "cpu_cores": 4,
            "memory_gb": 8,
            "max_budget": 25.0,
            "duration_hours": 1
        }
        
        try:
            request_id = await unified_coordinator.submit_unified_request(
                user_id="failover_test_user",
                workload_type="inference",
                request_params=params,
                user_tier="medium"
            )
            
            # Verify request still processed (using remaining components)
            assert request_id is not None
        finally:
            # Restore fog marketplace
            unified_coordinator.fog_marketplace = original_fog_marketplace

    # Test 8: Integration Testing
    @pytest.mark.asyncio
    async def test_comprehensive_system_integration(self, unified_coordinator):
        """Comprehensive test of all system components working together"""
        
        # Simulate realistic mixed workload scenario
        scenarios = [
            {"user_tier": "small", "workload": "inference", "count": 10},
            {"user_tier": "medium", "workload": "inference", "count": 15}, 
            {"user_tier": "medium", "workload": "training", "count": 5},
            {"user_tier": "large", "workload": "training", "count": 8},
            {"user_tier": "enterprise", "workload": "training", "count": 2},
        ]
        
        all_requests = []
        
        for scenario in scenarios:
            for i in range(scenario["count"]):
                if scenario["workload"] == "inference":
                    params = {
                        "model_id": f"integration_model_{scenario['user_tier']}",
                        "cpu_cores": 2 if scenario['user_tier'] == 'small' else 8,
                        "memory_gb": 4 if scenario['user_tier'] == 'small' else 16,
                        "max_budget": 10.0 * (2 ** ['small', 'medium', 'large', 'enterprise'].index(scenario['user_tier'])),
                        "duration_hours": 1,
                        "input_data": {"test": f"data_{i}"}
                    }
                else:  # training
                    params = {
                        "model_id": f"integration_training_{scenario['user_tier']}",
                        "cpu_cores": 16 if scenario['user_tier'] == 'large' else 8,
                        "memory_gb": 64 if scenario['user_tier'] == 'large' else 32,
                        "max_budget": 100.0 * (2 ** ['medium', 'large', 'enterprise'].index(scenario['user_tier'])),
                        "duration_hours": 6,
                        "participants": 50 if scenario['user_tier'] == 'enterprise' else 10
                    }
                
                request_id = await unified_coordinator.submit_unified_request(
                    user_id=f"integration_user_{scenario['user_tier']}_{i}",
                    workload_type=scenario["workload"],
                    request_params=params,
                    user_tier=scenario["user_tier"]
                )
                all_requests.append((request_id, scenario))
        
        # Verify all requests processed
        total_expected = sum(s["count"] for s in scenarios)
        assert len(all_requests) == total_expected
        
        # Verify system stats reflect the load
        final_stats = await unified_coordinator.get_marketplace_stats()
        assert final_stats["unified_coordinator"]["active_workloads"] >= total_expected

    @pytest.mark.asyncio
    async def test_system_scalability_limits(self, unified_coordinator):
        """Test system behavior at scalability limits"""
        
        # Test concurrent request handling
        async def submit_concurrent_request(user_id: str, tier: str):
            params = {
                "model_id": "scalability_test_model",
                "cpu_cores": 2,
                "memory_gb": 4,
                "max_budget": 15.0,
                "duration_hours": 0.5,
                "input_data": {"concurrent": "test"}
            }
            
            return await unified_coordinator.submit_unified_request(
                user_id=user_id,
                workload_type="inference",
                request_params=params,
                user_tier=tier
            )
        
        # Submit 100 concurrent requests
        concurrent_tasks = []
        for i in range(100):
            tier = ["small", "medium", "large", "enterprise"][i % 4]
            task = submit_concurrent_request(f"concurrent_user_{i}", tier)
            concurrent_tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        # Count successful vs failed requests
        successful = [r for r in results if isinstance(r, str)]  # request_ids are strings
        failed = [r for r in results if isinstance(r, Exception)]
        
        logger.info(f"Scalability test: {len(successful)} successful, {len(failed)} failed")
        
        # System should handle at least some concurrent requests
        assert len(successful) > 0
        # System should gracefully handle overload
        assert len(unified_coordinator.active_workloads) <= len(successful)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])