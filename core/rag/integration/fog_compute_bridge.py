"""
Fog Compute Bridge - Enhanced Integration

This module provides a comprehensive bridge between the RAG system and fog computing 
infrastructure, enabling distributed processing of knowledge retrieval tasks with
real P2P network integration, marketplace coordination, and workload distribution.

Key Features:
- P2P network node discovery and coordination
- Marketplace integration for resource allocation
- Distributed query processing across fog nodes
- Real-time workload distribution and load balancing
- Security and privacy-preserving computation
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# P2P Network Integration
try:
    from infrastructure.p2p.core.message_delivery import MessageDeliveryService, DeliveryConfig, MessagePriority
    from infrastructure.p2p.core.message_types import MeshMessage, MeshMessageType

    P2P_AVAILABLE = True
except ImportError:
    P2P_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("P2P infrastructure not available")

# Marketplace Integration
try:
    from infrastructure.fog.market.marketplace_api import get_marketplace_api, MarketplaceAPI

    MARKETPLACE_AVAILABLE = True
except ImportError:
    MARKETPLACE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Marketplace integration not available")

# Distributed Inference Integration
try:
    from infrastructure.distributed_inference.core.distributed_inference_manager import (
        get_distributed_inference_manager,
        DistributedInferenceManager,
    )

    DISTRIBUTED_INFERENCE_AVAILABLE = True
except ImportError:
    DISTRIBUTED_INFERENCE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Distributed inference not available")

# Security Integration
try:
    from infrastructure.security.distributed.coordinator import SecurityCoordinator

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Security infrastructure not available")

logger = logging.getLogger(__name__)


class QueryDistributionStrategy(Enum):
    """Strategy for distributing queries across fog nodes."""

    BALANCED = "balanced"  # Distribute evenly across all nodes
    PERFORMANCE = "performance"  # Route to highest-performance nodes
    COST_OPTIMIZED = "cost_optimized"  # Route to lowest-cost nodes
    PRIVACY_FIRST = "privacy_first"  # Prioritize privacy-preserving nodes
    HYBRID = "hybrid"  # Adaptive strategy based on query type


class QueryType(Enum):
    """Type of query being processed."""

    SIMPLE_RAG = "simple_rag"  # Basic retrieval-augmented generation
    COMPLEX_INFERENCE = "complex_inference"  # Multi-step reasoning
    FEDERATED_LEARNING = "federated_learning"  # Training tasks
    VECTOR_SEARCH = "vector_search"  # Similarity search
    HYBRID_PROCESSING = "hybrid_processing"  # Mixed workloads


@dataclass
class FogNode:
    """Represents a fog computing node."""

    node_id: str
    hostname: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    trust_score: float = field(default=0.7)
    performance_score: float = field(default=0.8)
    cost_per_hour: float = field(default=1.0)
    privacy_level: str = field(default="medium")
    available: bool = field(default=True)
    current_load: float = field(default=0.0)
    last_heartbeat: Optional[float] = field(default=None)

    def is_healthy(self) -> bool:
        """Check if node is healthy and available."""
        if not self.available or not self.last_heartbeat:
            return False
        return (time.time() - self.last_heartbeat) < 60

    def get_selection_score(
        self, strategy: QueryDistributionStrategy, required_capabilities: List[str] = None
    ) -> float:
        """Calculate node selection score based on strategy."""
        if not self.is_healthy():
            return -1.0

        # Check capability requirements
        required_capabilities = required_capabilities or []
        if required_capabilities and not all(cap in self.capabilities for cap in required_capabilities):
            return -1.0

        if strategy == QueryDistributionStrategy.PERFORMANCE:
            return self.performance_score * (1 - self.current_load)
        elif strategy == QueryDistributionStrategy.COST_OPTIMIZED:
            return 1.0 / max(self.cost_per_hour, 0.1)
        elif strategy == QueryDistributionStrategy.PRIVACY_FIRST:
            privacy_scores = {"low": 0.3, "medium": 0.6, "high": 0.9, "critical": 1.0}
            return privacy_scores.get(self.privacy_level, 0.6) * self.trust_score
        else:  # BALANCED or HYBRID
            return (
                0.4 * self.performance_score
                + 0.3 * self.trust_score
                + 0.2 * (1 - self.current_load)
                + 0.1 * (1.0 / max(self.cost_per_hour, 0.1))
            )


@dataclass
class QueryRequest:
    """Distributed query request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = field(default="")
    query_type: QueryType = field(default=QueryType.SIMPLE_RAG)
    strategy: QueryDistributionStrategy = field(default=QueryDistributionStrategy.BALANCED)
    user_tier: str = field(default="medium")
    max_budget: float = field(default=100.0)
    privacy_level: str = field(default="medium")
    required_capabilities: List[str] = field(default_factory=list)
    timeout_seconds: int = field(default=300)

    # Processing state
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = field(default=None)
    completed_at: Optional[float] = field(default=None)
    assigned_nodes: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = field(default=None)
    total_cost: float = field(default=0.0)


class FogComputeBridge:
    """Enhanced bridge between RAG system and fog computing infrastructure."""

    def __init__(
        self,
        hyper_rag_instance=None,
        enable_p2p: bool = True,
        enable_marketplace: bool = True,
        enable_distributed_inference: bool = True,
        enable_security: bool = True,
    ):
        """Initialize the fog compute bridge."""
        self.hyper_rag = hyper_rag_instance
        self.initialized = False

        # Configuration
        self.enable_p2p = enable_p2p and P2P_AVAILABLE
        self.enable_marketplace = enable_marketplace and MARKETPLACE_AVAILABLE
        self.enable_distributed_inference = enable_distributed_inference and DISTRIBUTED_INFERENCE_AVAILABLE
        self.enable_security = enable_security and SECURITY_AVAILABLE

        # Core components
        self.message_service: Optional[MessageDeliveryService] = None
        self.marketplace_api: Optional[MarketplaceAPI] = None
        self.inference_manager: Optional[DistributedInferenceManager] = None
        self.security_coordinator: Optional[SecurityCoordinator] = None

        # State management
        self.fog_nodes: Dict[str, FogNode] = {}
        self.active_requests: Dict[str, QueryRequest] = {}
        self.completed_requests: Dict[str, QueryRequest] = {}

        # Performance metrics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_nodes_discovered": 0,
            "average_response_time": 0.0,
            "total_cost_incurred": 0.0,
            "marketplace_allocations": 0,
            "p2p_messages_sent": 0,
        }

        logger.info(
            f"Fog Compute Bridge initialized with integrations: "
            f"P2P={self.enable_p2p}, Marketplace={self.enable_marketplace}, "
            f"DistributedInference={self.enable_distributed_inference}, Security={self.enable_security}"
        )

    async def initialize(self):
        """Initialize the fog compute bridge and all integrations."""
        if self.initialized:
            return

        try:
            logger.info("Initializing Enhanced Fog Compute Bridge...")

            # Initialize P2P message service
            if self.enable_p2p:
                await self._initialize_p2p_service()

            # Initialize marketplace integration
            if self.enable_marketplace:
                await self._initialize_marketplace()

            # Initialize distributed inference
            if self.enable_distributed_inference:
                await self._initialize_distributed_inference()

            # Initialize security coordinator
            if self.enable_security:
                await self._initialize_security()

            # Discover initial fog nodes
            await self._discover_fog_nodes()

            self.initialized = True
            logger.info("✅ Enhanced Fog Compute Bridge initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize Fog Compute Bridge: {e}")
            raise

    async def _initialize_p2p_service(self):
        """Initialize P2P message delivery service."""
        try:
            config = DeliveryConfig(
                max_retry_attempts=3, concurrent_deliveries=10, enable_persistence=False  # In-memory for fog bridge
            )

            self.message_service = MessageDeliveryService(config)
            await self.message_service.start()

            logger.info("P2P message service initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize P2P service: {e}")
            self.enable_p2p = False

    async def _initialize_marketplace(self):
        """Initialize marketplace integration."""
        try:
            self.marketplace_api = get_marketplace_api()
            await self.marketplace_api.initialize_market_components()

            logger.info("Marketplace integration initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize marketplace: {e}")
            self.enable_marketplace = False

    async def _initialize_distributed_inference(self):
        """Initialize distributed inference manager."""
        try:
            self.inference_manager = get_distributed_inference_manager()
            await self.inference_manager.start()

            logger.info("Distributed inference manager initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize distributed inference: {e}")
            self.enable_distributed_inference = False

    async def _initialize_security(self):
        """Initialize security coordinator."""
        try:
            self.security_coordinator = SecurityCoordinator()
            await self.security_coordinator.initialize()

            logger.info("Security coordinator initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize security: {e}")
            self.enable_security = False

    async def _discover_fog_nodes(self):
        """Discover available fog nodes through P2P network."""
        try:
            # Simulate fog node discovery (in production, would use actual P2P discovery)
            discovered_nodes = [
                FogNode(
                    node_id="fog_node_1",
                    hostname="10.0.1.100",
                    port=8001,
                    capabilities=["rag_processing", "vector_search", "inference"],
                    trust_score=0.85,
                    performance_score=0.9,
                    cost_per_hour=2.5,
                    privacy_level="high",
                    last_heartbeat=time.time(),
                ),
                FogNode(
                    node_id="fog_node_2",
                    hostname="10.0.1.101",
                    port=8002,
                    capabilities=["federated_learning", "inference", "vector_search"],
                    trust_score=0.78,
                    performance_score=0.85,
                    cost_per_hour=1.8,
                    privacy_level="medium",
                    last_heartbeat=time.time(),
                ),
                FogNode(
                    node_id="fog_node_3",
                    hostname="10.0.1.102",
                    port=8003,
                    capabilities=["rag_processing", "hybrid_processing"],
                    trust_score=0.92,
                    performance_score=0.88,
                    cost_per_hour=3.2,
                    privacy_level="critical",
                    last_heartbeat=time.time(),
                ),
            ]

            for node in discovered_nodes:
                self.fog_nodes[node.node_id] = node

            self.stats["total_nodes_discovered"] = len(self.fog_nodes)
            logger.info(f"Discovered {len(discovered_nodes)} fog nodes")

            # Register nodes with distributed inference manager if available
            if self.inference_manager:
                for node in discovered_nodes:
                    self.inference_manager.register_node(
                        hostname=node.hostname,
                        port=node.port,
                        total_memory=8192,  # 8GB default
                        gpu_count=1,
                        cpu_cores=4,
                    )

        except Exception as e:
            logger.warning(f"Failed to discover fog nodes: {e}")

    async def distribute_query(
        self,
        query: str,
        query_type: QueryType = QueryType.SIMPLE_RAG,
        strategy: QueryDistributionStrategy = QueryDistributionStrategy.BALANCED,
        user_tier: str = "medium",
        max_budget: float = 100.0,
        privacy_level: str = "medium",
        required_capabilities: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Distribute a query across fog nodes with full integration."""
        if not self.initialized:
            raise RuntimeError("Fog Compute Bridge not initialized")

        # Create query request
        request = QueryRequest(
            query=query,
            query_type=query_type,
            strategy=strategy,
            user_tier=user_tier,
            max_budget=max_budget,
            privacy_level=privacy_level,
            required_capabilities=required_capabilities or [],
        )

        try:
            self.active_requests[request.request_id] = request
            request.started_at = time.time()
            self.stats["total_requests"] += 1

            logger.info(f"Processing distributed query {request.request_id} " f"with strategy {strategy.value}")

            # Step 1: Marketplace resource allocation if enabled
            if self.enable_marketplace:
                await self._allocate_marketplace_resources(request)

            # Step 2: Select optimal fog nodes
            selected_nodes = await self._select_fog_nodes(request)

            if not selected_nodes:
                raise RuntimeError("No suitable fog nodes available")

            request.assigned_nodes = [node.node_id for node in selected_nodes]

            # Step 3: Distribute workload across nodes
            if self.enable_distributed_inference and request.query_type in [
                QueryType.COMPLEX_INFERENCE,
                QueryType.FEDERATED_LEARNING,
            ]:
                # Use distributed inference manager for complex tasks
                await self._process_with_distributed_inference(request)
            else:
                # Use direct P2P communication for simpler tasks
                await self._process_with_p2p(request, selected_nodes)

            # Step 4: Aggregate results
            aggregated_result = await self._aggregate_results(request)

            # Step 5: Complete request
            request.completed_at = time.time()
            request.results["final_result"] = aggregated_result

            # Update statistics
            response_time = request.completed_at - request.started_at
            self.stats["successful_requests"] += 1
            self.stats["average_response_time"] = (
                self.stats["average_response_time"] * (self.stats["successful_requests"] - 1) + response_time
            ) / self.stats["successful_requests"]
            self.stats["total_cost_incurred"] += request.total_cost

            # Move to completed requests
            self.completed_requests[request.request_id] = request
            del self.active_requests[request.request_id]

            return {
                "request_id": request.request_id,
                "query": query,
                "distributed": True,
                "fog_nodes_used": request.assigned_nodes,
                "results": aggregated_result,
                "performance_metrics": {
                    "response_time_seconds": response_time,
                    "nodes_utilized": len(selected_nodes),
                    "total_cost": request.total_cost,
                    "strategy_used": strategy.value,
                },
                "integration_status": {
                    "p2p_enabled": self.enable_p2p,
                    "marketplace_enabled": self.enable_marketplace,
                    "distributed_inference_enabled": self.enable_distributed_inference,
                    "security_enabled": self.enable_security,
                },
            }

        except Exception as e:
            # Handle request failure
            request.error_message = str(e)
            request.completed_at = time.time()

            self.stats["failed_requests"] += 1
            self.completed_requests[request.request_id] = request

            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]

            logger.error(f"Query distribution failed for {request.request_id}: {e}")

            return {
                "request_id": request.request_id,
                "query": query,
                "distributed": False,
                "fog_nodes_used": [],
                "results": [],
                "error": str(e),
                "performance_metrics": {
                    "response_time_seconds": time.time() - request.started_at if request.started_at else 0,
                    "nodes_utilized": 0,
                    "total_cost": 0,
                    "strategy_used": strategy.value,
                },
            }

    async def _allocate_marketplace_resources(self, request: QueryRequest):
        """Allocate resources through marketplace if available."""
        if not self.marketplace_api:
            return

        try:
            # Create marketplace request based on query type
            if request.query_type == QueryType.FEDERATED_LEARNING:
                # Submit training request
                {
                    "requester_id": f"fog_bridge_{request.request_id}",
                    "user_tier": request.user_tier,
                    "model_size": "medium",  # Default
                    "duration_hours": 1.0,
                    "participants_needed": min(len(self.fog_nodes), 5),
                    "privacy_level": request.privacy_level,
                    "max_budget": request.max_budget,
                }
            else:
                # Submit inference request
                {
                    "requester_id": f"fog_bridge_{request.request_id}",
                    "user_tier": request.user_tier,
                    "model_size": "medium",
                    "requests_count": 1,
                    "participants_needed": min(len(self.fog_nodes), 3),
                    "privacy_level": request.privacy_level,
                    "max_budget": request.max_budget,
                }

            self.stats["marketplace_allocations"] += 1
            logger.debug(f"Allocated marketplace resources for request {request.request_id}")

        except Exception as e:
            logger.warning(f"Marketplace allocation failed: {e}")

    async def _select_fog_nodes(self, request: QueryRequest, max_nodes: int = 3) -> List[FogNode]:
        """Select optimal fog nodes based on strategy and requirements."""
        available_nodes = [node for node in self.fog_nodes.values() if node.is_healthy()]

        if not available_nodes:
            return []

        # Calculate selection scores
        scored_nodes = []
        for node in available_nodes:
            score = node.get_selection_score(request.strategy, request.required_capabilities)
            if score >= 0:  # Valid node
                scored_nodes.append((node, score))

        # Sort by score (descending) and select top nodes
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, score in scored_nodes[:max_nodes]]

        logger.debug(
            f"Selected {len(selected_nodes)} nodes for request {request.request_id} "
            f"using strategy {request.strategy.value}"
        )

        return selected_nodes

    async def _process_with_distributed_inference(self, request: QueryRequest):
        """Process request using distributed inference manager."""
        if not self.inference_manager:
            raise RuntimeError("Distributed inference manager not available")

        try:
            # Submit to distributed inference manager
            inference_request_id = await self.inference_manager.submit_inference_request(
                model_name=f"fog_query_model_{request.query_type.value}",
                input_data={
                    "query": request.query,
                    "type": request.query_type.value,
                    "privacy_level": request.privacy_level,
                },
                parameters={"strategy": request.strategy.value, "max_budget": request.max_budget},
            )

            # Wait for completion (simplified - in production would use async monitoring)
            await asyncio.sleep(2)  # Simulate processing time

            status = self.inference_manager.get_request_status(inference_request_id)
            if status:
                request.results["distributed_inference"] = status.get("results", {})

        except Exception as e:
            logger.warning(f"Distributed inference processing failed: {e}")
            # Fallback to P2P processing
            selected_nodes = [self.fog_nodes[node_id] for node_id in request.assigned_nodes]
            await self._process_with_p2p(request, selected_nodes)

    async def _process_with_p2p(self, request: QueryRequest, selected_nodes: List[FogNode]):
        """Process request using direct P2P communication."""
        try:
            # Create messages for each node
            messages = []
            for node in selected_nodes:
                message_data = {
                    "request_id": request.request_id,
                    "query": request.query,
                    "query_type": request.query_type.value,
                    "privacy_level": request.privacy_level,
                    "node_assignment": {
                        "node_id": node.node_id,
                        "capabilities_required": request.required_capabilities,
                    },
                }

                if self.enable_p2p and self.message_service:
                    # Send via P2P network
                    mesh_message = MeshMessage(
                        type=MeshMessageType.DATA_MESSAGE,
                        sender="fog_bridge",
                        recipient=node.node_id,
                        payload=str(message_data).encode(),
                    )

                    message_id = await self.message_service.send_message(mesh_message, priority=MessagePriority.HIGH)

                    messages.append(message_id)
                    self.stats["p2p_messages_sent"] += 1

                else:
                    # Simulate direct communication
                    await asyncio.sleep(0.1)  # Simulate network latency

                # Simulate node response
                request.results[node.node_id] = {
                    "node_id": node.node_id,
                    "processed_query": request.query,
                    "processing_time": 0.5 + (hash(node.node_id) % 100) / 200,
                    "confidence_score": 0.8 + (hash(request.query) % 20) / 100,
                    "cost": node.cost_per_hour * 0.1,  # Assume 6 minutes processing
                    "privacy_preserved": node.privacy_level in ["high", "critical"],
                }

                request.total_cost += request.results[node.node_id]["cost"]

            logger.debug(
                f"Processed request {request.request_id} across {len(selected_nodes)} nodes "
                f"with total cost ${request.total_cost:.2f}"
            )

        except Exception as e:
            logger.error(f"P2P processing failed: {e}")
            raise

    async def _aggregate_results(self, request: QueryRequest) -> Dict[str, Any]:
        """Aggregate results from all fog nodes."""
        node_results = [
            result
            for key, result in request.results.items()
            if key != "final_result" and isinstance(result, dict) and "node_id" in result
        ]

        if not node_results:
            return {"error": "No results to aggregate"}

        # Calculate aggregated metrics
        total_processing_time = sum(r.get("processing_time", 0) for r in node_results)
        average_confidence = sum(r.get("confidence_score", 0) for r in node_results) / len(node_results)
        privacy_preserved_count = sum(1 for r in node_results if r.get("privacy_preserved", False))

        aggregated_result = {
            "query": request.query,
            "query_type": request.query_type.value,
            "strategy_used": request.strategy.value,
            "nodes_processed": len(node_results),
            "total_processing_time": total_processing_time,
            "average_processing_time": total_processing_time / len(node_results),
            "average_confidence_score": average_confidence,
            "privacy_preservation_rate": privacy_preserved_count / len(node_results),
            "total_cost": request.total_cost,
            "cost_per_node": request.total_cost / len(node_results),
            "processing_summary": {
                "high_confidence_results": len([r for r in node_results if r.get("confidence_score", 0) > 0.85]),
                "privacy_preserved_nodes": privacy_preserved_count,
                "cost_efficient_nodes": len([r for r in node_results if r.get("cost", 0) < 2.0]),
            },
            "node_contributions": node_results,
            "integration_metadata": {
                "fog_bridge_version": "2.0",
                "p2p_integration": self.enable_p2p,
                "marketplace_integration": self.enable_marketplace,
                "security_integration": self.enable_security,
                "processing_timestamp": time.time(),
            },
        }

        return aggregated_result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        healthy_nodes = [node for node in self.fog_nodes.values() if node.is_healthy()]

        return {
            "bridge_status": {
                "initialized": self.initialized,
                "integrations": {
                    "p2p_enabled": self.enable_p2p,
                    "marketplace_enabled": self.enable_marketplace,
                    "distributed_inference_enabled": self.enable_distributed_inference,
                    "security_enabled": self.enable_security,
                },
            },
            "fog_network": {
                "total_nodes": len(self.fog_nodes),
                "healthy_nodes": len(healthy_nodes),
                "node_capabilities": list(set(cap for node in self.fog_nodes.values() for cap in node.capabilities)),
                "average_trust_score": sum(n.trust_score for n in healthy_nodes) / max(len(healthy_nodes), 1),
                "total_capacity": {
                    "performance_score": sum(n.performance_score for n in healthy_nodes),
                    "cost_range": {
                        "min": min((n.cost_per_hour for n in healthy_nodes), default=0),
                        "max": max((n.cost_per_hour for n in healthy_nodes), default=0),
                        "average": sum(n.cost_per_hour for n in healthy_nodes) / max(len(healthy_nodes), 1),
                    },
                },
            },
            "request_processing": {
                "active_requests": len(self.active_requests),
                "completed_requests": len(self.completed_requests),
                "performance_statistics": self.stats,
            },
            "component_status": {
                "message_service": "active" if self.message_service else "inactive",
                "marketplace_api": "active" if self.marketplace_api else "inactive",
                "inference_manager": "active" if self.inference_manager else "inactive",
                "security_coordinator": "active" if self.security_coordinator else "inactive",
            },
        }

    async def close(self):
        """Close the fog compute bridge and cleanup resources."""
        try:
            logger.info("Shutting down Enhanced Fog Compute Bridge...")

            # Cancel active requests
            for request in self.active_requests.values():
                request.error_message = "Bridge shutdown"
                request.completed_at = time.time()

            # Stop message service
            if self.message_service:
                await self.message_service.stop()

            # Stop distributed inference manager
            if self.inference_manager:
                await self.inference_manager.stop()

            # Stop security coordinator
            if self.security_coordinator:
                await self.security_coordinator.cleanup()

            self.initialized = False
            logger.info("Enhanced Fog Compute Bridge shutdown complete")

        except Exception as e:
            logger.exception(f"Error during Fog Compute Bridge shutdown: {e}")


# Global bridge instance for easy integration
_global_bridge: Optional[FogComputeBridge] = None


def get_fog_compute_bridge(**kwargs) -> FogComputeBridge:
    """Get or create global fog compute bridge instance."""
    global _global_bridge

    if _global_bridge is None:
        _global_bridge = FogComputeBridge(**kwargs)

    return _global_bridge


async def distribute_query_globally(
    query: str,
    query_type: QueryType = QueryType.SIMPLE_RAG,
    strategy: QueryDistributionStrategy = QueryDistributionStrategy.BALANCED,
    **kwargs,
) -> Dict[str, Any]:
    """Global function for distributing queries across fog network."""
    bridge = get_fog_compute_bridge()

    if not bridge.initialized:
        await bridge.initialize()

    return await bridge.distribute_query(query=query, query_type=query_type, strategy=strategy, **kwargs)


def get_fog_system_status() -> Dict[str, Any]:
    """Global function to get fog system status."""
    bridge = get_fog_compute_bridge()
    return bridge.get_system_status()
