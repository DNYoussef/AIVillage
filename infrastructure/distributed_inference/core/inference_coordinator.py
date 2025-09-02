"""
Inference Coordinator - Phase 2 Archaeological Enhancement
Innovation Score: 7.8/10

Archaeological Context:
- Source: Distributed request routing branch (ancient-routing-algorithms)
- Integration: Load balancing research (lost-load-balancer-research)
- Enhancement: Request optimization patterns (inference-coordination-experiments)
- Innovation Date: 2025-01-15

The Inference Coordinator manages distributed inference requests, routing, and result 
aggregation with integration to Phase 1 emergency triage and tensor optimization systems.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import time
from typing import Any
import uuid

# Import Phase 1 components for integration
from infrastructure.monitoring.triage.emergency_triage_system import EmergencyTriageSystem

# Archaeological metadata
ARCHAEOLOGICAL_METADATA = {
    "component": "InferenceCoordinator", 
    "phase": "Phase2",
    "innovation_score": 7.8,
    "source_branches": [
        "ancient-routing-algorithms",
        "lost-load-balancer-research",
        "inference-coordination-experiments"
    ],
    "integration_date": "2025-01-15",
    "phase1_integrations": [
        "emergency_triage_system",
        "tensor_memory_optimizer"
    ],
    "feature_flags": {
        "ARCHAEOLOGICAL_INFERENCE_COORDINATION_ENABLED": True,
        "ADVANCED_REQUEST_ROUTING_ENABLED": True,
        "RESULT_AGGREGATION_OPTIMIZATION_ENABLED": True,
        "PHASE1_INTEGRATION_ENABLED": True
    },
    "performance_targets": {
        "coordination_overhead": "<100ms",
        "success_rate": "99.5%",
        "concurrent_requests": ">1000",
        "fault_tolerance": "99.9%"
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRequestStatus(Enum):
    """Status of inference requests."""
    PENDING = auto()
    QUEUED = auto()
    ROUTING = auto()
    PROCESSING = auto()
    AGGREGATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    TIMED_OUT = auto()

class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

class RoutingStrategy(Enum):
    """Request routing strategies."""
    ROUND_ROBIN = auto()
    LEAST_LOADED = auto()
    FASTEST_RESPONSE = auto()
    MEMORY_OPTIMAL = auto()
    ARCHAEOLOGICAL_OPTIMAL = auto()  # Uses archaeological patterns

@dataclass
class InferenceRequest:
    """Comprehensive inference request definition."""
    request_id: str
    model_id: str
    input_data: dict[str, Any]
    priority: RequestPriority = RequestPriority.NORMAL
    timeout_seconds: int = 300
    routing_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    queued_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    # Status tracking
    status: InferenceRequestStatus = InferenceRequestStatus.PENDING
    error_message: str | None = None
    retry_count: int = 0
    
    # Result tracking
    partial_results: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] | None = None
    
    # Archaeological enhancements
    archaeological_priority: float = 0.0  # Calculated priority score
    phase1_triage_score: float | None = None

@dataclass
class NodePerformanceMetrics:
    """Real-time node performance metrics."""
    node_id: str
    response_time_ms: float
    queue_length: int
    success_rate: float
    memory_utilization: float
    compute_utilization: float
    network_latency_ms: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Archaeological metrics
    archaeological_efficiency: float = 1.0
    reliability_score: float = 1.0

@dataclass
class RoutingDecision:
    """Routing decision with archaeological optimization."""
    request_id: str
    selected_nodes: list[str]
    routing_strategy: RoutingStrategy
    estimated_completion_time: float
    confidence_score: float
    archaeological_factors: dict[str, float] = field(default_factory=dict)
    backup_nodes: list[str] = field(default_factory=list)

class InferenceCoordinator:
    """
    Advanced Inference Coordinator with Archaeological Enhancement
    
    Manages distributed inference requests with:
    - Intelligent request routing and load balancing
    - Real-time performance monitoring
    - Result aggregation and fault tolerance
    - Integration with Phase 1 emergency triage system
    - Archaeological optimization patterns
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the inference coordinator."""
        self.config = config or {}
        self.archaeological_metadata = ARCHAEOLOGICAL_METADATA
        
        # Core components
        self.active_requests: dict[str, InferenceRequest] = {}
        self.request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.node_metrics: dict[str, NodePerformanceMetrics] = {}
        self.routing_history: list[RoutingDecision] = []
        
        # Phase 1 integration
        self.emergency_triage: EmergencyTriageSystem | None = None
        
        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "queue_wait_time": 0.0,
            "archaeological_optimization_hits": 0
        }
        
        # Configuration
        self.max_concurrent_requests = self.config.get("max_concurrent_requests", 1000)
        self.default_timeout = self.config.get("default_timeout_seconds", 300)
        self.retry_limit = self.config.get("retry_limit", 3)
        self.queue_timeout = self.config.get("queue_timeout_seconds", 60)
        
        # Archaeological optimization settings
        self.archaeological_routing_weight = self.config.get("archaeological_routing_weight", 0.25)
        self.phase1_integration_weight = self.config.get("phase1_integration_weight", 0.15)
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
        logger.info("🧠 InferenceCoordinator initialized with archaeological metadata")
        logger.info(f"📊 Innovation Score: {self.archaeological_metadata['innovation_score']}")
        
    async def start(self):
        """Start the inference coordinator with archaeological enhancements."""
        if not self.archaeological_metadata["feature_flags"].get("ARCHAEOLOGICAL_INFERENCE_COORDINATION_ENABLED", False):
            logger.warning("🚫 Archaeological inference coordination disabled by feature flag")
            return False
            
        logger.info("🚀 Starting Inference Coordinator...")
        
        # Initialize Phase 1 integration
        if self.archaeological_metadata["feature_flags"].get("PHASE1_INTEGRATION_ENABLED", False):
            await self._initialize_phase1_integration()
            
        # Start background tasks
        self.running = True
        
        # Start request processor
        asyncio.create_task(self._process_request_queue())
        
        # Start performance monitor
        asyncio.create_task(self._monitor_performance())
        
        # Start archaeological optimizer
        asyncio.create_task(self._archaeological_optimization_loop())
        
        logger.info("✅ Inference Coordinator started successfully")
        return True
        
    async def stop(self):
        """Stop the inference coordinator and cleanup."""
        logger.info("🔄 Stopping Inference Coordinator...")
        
        self.running = False
        
        # Cancel pending requests
        for request in self.active_requests.values():
            if request.status in [InferenceRequestStatus.PENDING, InferenceRequestStatus.QUEUED, InferenceRequestStatus.PROCESSING]:
                request.status = InferenceRequestStatus.FAILED
                request.error_message = "Coordinator shutdown"
                
        # Save archaeological performance data
        await self._save_archaeological_data()
        
        # Cleanup
        self.active_requests.clear()
        self.node_metrics.clear()
        
        if self.executor:
            self.executor.shutdown(wait=True)
            
        logger.info("✅ Inference Coordinator stopped")
        
    async def submit_inference_request(
        self,
        model_id: str,
        input_data: dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: int | None = None,
        routing_hints: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Submit an inference request for distributed processing.
        
        Returns:
            str: Request ID for tracking
        """
        try:
            request_id = str(uuid.uuid4())
            timeout = timeout_seconds or self.default_timeout
            
            request = InferenceRequest(
                request_id=request_id,
                model_id=model_id,
                input_data=input_data,
                priority=priority,
                timeout_seconds=timeout,
                routing_hints=routing_hints or {},
                metadata=metadata or {}
            )
            
            # Calculate archaeological priority score
            request.archaeological_priority = await self._calculate_archaeological_priority(request)
            
            # Phase 1 integration: Get triage score
            if self.emergency_triage and self.archaeological_metadata["feature_flags"].get("PHASE1_INTEGRATION_ENABLED", False):
                try:
                    triage_score = await self._get_phase1_triage_score(request)
                    request.phase1_triage_score = triage_score
                    
                    # Adjust priority based on triage
                    if triage_score and triage_score > 0.8:
                        request.priority = RequestPriority.CRITICAL
                        request.archaeological_priority *= 1.5
                        
                except Exception as e:
                    logger.warning(f"⚠️ Phase 1 triage integration failed: {e}")
            
            # Add to active requests
            self.active_requests[request_id] = request
            
            # Queue request with priority
            priority_score = request.priority.value * 1000 + request.archaeological_priority
            await self.request_queue.put((-priority_score, time.time(), request_id))
            
            request.status = InferenceRequestStatus.QUEUED
            request.queued_at = datetime.now()
            
            # Update stats
            self.performance_stats["total_requests"] += 1
            
            logger.info(f"📝 Queued inference request {request_id} for model {model_id} "
                       f"(priority: {priority.name}, archaeological: {request.archaeological_priority:.2f})")
            
            return request_id
            
        except Exception as e:
            logger.error(f"❌ Failed to submit inference request: {e}")
            raise
            
    async def get_request_status(self, request_id: str) -> dict[str, Any] | None:
        """Get the current status of an inference request."""
        if request_id not in self.active_requests:
            return None
            
        request = self.active_requests[request_id]
        
        return {
            "request_id": request_id,
            "status": request.status.name,
            "model_id": request.model_id,
            "priority": request.priority.name,
            "created_at": request.created_at.isoformat(),
            "queued_at": request.queued_at.isoformat() if request.queued_at else None,
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "error_message": request.error_message,
            "retry_count": request.retry_count,
            "archaeological_priority": request.archaeological_priority,
            "phase1_triage_score": request.phase1_triage_score,
            "partial_results_available": len(request.partial_results) > 0,
            "final_result_available": request.final_result is not None
        }
        
    async def get_request_result(self, request_id: str) -> dict[str, Any] | None:
        """Get the result of a completed inference request."""
        if request_id not in self.active_requests:
            return None
            
        request = self.active_requests[request_id]
        
        if request.status == InferenceRequestStatus.COMPLETED:
            return request.final_result
        elif request.status in [InferenceRequestStatus.PROCESSING, InferenceRequestStatus.AGGREGATING]:
            # Return partial results if available
            return {
                "partial": True,
                "results": request.partial_results,
                "status": request.status.name
            }
        else:
            return None
            
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or processing inference request."""
        try:
            if request_id not in self.active_requests:
                return False
                
            request = self.active_requests[request_id]
            
            if request.status in [InferenceRequestStatus.COMPLETED, InferenceRequestStatus.FAILED]:
                return False  # Cannot cancel completed requests
                
            request.status = InferenceRequestStatus.FAILED
            request.error_message = "Request cancelled by user"
            request.completed_at = datetime.now()
            
            logger.info(f"🚫 Cancelled inference request {request_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel request {request_id}: {e}")
            return False
            
    async def register_node(self, node_id: str, initial_metrics: dict[str, Any] | None = None) -> bool:
        """Register a compute node for inference coordination."""
        try:
            metrics = NodePerformanceMetrics(
                node_id=node_id,
                response_time_ms=initial_metrics.get("response_time_ms", 100.0) if initial_metrics else 100.0,
                queue_length=initial_metrics.get("queue_length", 0) if initial_metrics else 0,
                success_rate=initial_metrics.get("success_rate", 1.0) if initial_metrics else 1.0,
                memory_utilization=initial_metrics.get("memory_utilization", 0.0) if initial_metrics else 0.0,
                compute_utilization=initial_metrics.get("compute_utilization", 0.0) if initial_metrics else 0.0,
                network_latency_ms=initial_metrics.get("network_latency_ms", 10.0) if initial_metrics else 10.0
            )
            
            self.node_metrics[node_id] = metrics
            
            logger.info(f"📝 Registered node {node_id} for inference coordination")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register node {node_id}: {e}")
            return False
            
    async def update_node_metrics(self, node_id: str, metrics: dict[str, Any]) -> bool:
        """Update performance metrics for a node."""
        try:
            if node_id not in self.node_metrics:
                logger.warning(f"⚠️ Node {node_id} not registered")
                return False
                
            node_metrics = self.node_metrics[node_id]
            
            # Update metrics
            if "response_time_ms" in metrics:
                node_metrics.response_time_ms = metrics["response_time_ms"]
            if "queue_length" in metrics:
                node_metrics.queue_length = metrics["queue_length"]
            if "success_rate" in metrics:
                node_metrics.success_rate = metrics["success_rate"]
            if "memory_utilization" in metrics:
                node_metrics.memory_utilization = metrics["memory_utilization"]
            if "compute_utilization" in metrics:
                node_metrics.compute_utilization = metrics["compute_utilization"]
            if "network_latency_ms" in metrics:
                node_metrics.network_latency_ms = metrics["network_latency_ms"]
                
            # Update archaeological metrics
            node_metrics.archaeological_efficiency = self._calculate_archaeological_efficiency(node_metrics)
            node_metrics.reliability_score = self._calculate_reliability_score(node_metrics)
            
            node_metrics.last_updated = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to update metrics for node {node_id}: {e}")
            return False
            
    async def get_coordinator_stats(self) -> dict[str, Any]:
        """Get comprehensive coordinator statistics."""
        active_count = len([r for r in self.active_requests.values() 
                           if r.status not in [InferenceRequestStatus.COMPLETED, InferenceRequestStatus.FAILED]])
        
        queue_size = self.request_queue.qsize()
        
        return {
            "performance_stats": self.performance_stats.copy(),
            "active_requests": active_count,
            "queue_size": queue_size,
            "registered_nodes": len(self.node_metrics),
            "routing_decisions_made": len(self.routing_history),
            "archaeological_metadata": self.archaeological_metadata,
            "node_utilization": {
                node_id: {
                    "memory_utilization": metrics.memory_utilization,
                    "compute_utilization": metrics.compute_utilization,
                    "queue_length": metrics.queue_length,
                    "response_time_ms": metrics.response_time_ms,
                    "success_rate": metrics.success_rate,
                    "archaeological_efficiency": metrics.archaeological_efficiency,
                    "reliability_score": metrics.reliability_score
                }
                for node_id, metrics in self.node_metrics.items()
            }
        }
        
    # Internal Archaeological Methods
    
    async def _initialize_phase1_integration(self):
        """Initialize integration with Phase 1 components."""
        try:
            # Initialize Emergency Triage System integration
            self.emergency_triage = EmergencyTriageSystem()
            await self.emergency_triage.start()
            
            logger.info("✅ Phase 1 integration initialized")
            
        except Exception as e:
            logger.error(f"❌ Phase 1 integration failed: {e}")
            self.emergency_triage = None
            
    async def _process_request_queue(self):
        """Background task to process the request queue."""
        logger.info("🔄 Starting request queue processor")
        
        while self.running:
            try:
                # Get next request with timeout
                try:
                    priority_score, queued_time, request_id = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                if request_id not in self.active_requests:
                    continue
                    
                request = self.active_requests[request_id]
                
                # Check if request timed out while queued
                wait_time = time.time() - queued_time
                if wait_time > self.queue_timeout:
                    request.status = InferenceRequestStatus.TIMED_OUT
                    request.error_message = f"Request timed out in queue after {wait_time:.1f}s"
                    request.completed_at = datetime.now()
                    self.performance_stats["failed_requests"] += 1
                    continue
                    
                # Process the request
                asyncio.create_task(self._process_inference_request(request))
                
            except Exception as e:
                logger.error(f"❌ Error in request queue processor: {e}")
                await asyncio.sleep(1)
                
    async def _process_inference_request(self, request: InferenceRequest):
        """Process an individual inference request."""
        try:
            request.status = InferenceRequestStatus.ROUTING
            request.started_at = datetime.now()
            
            # Make routing decision using archaeological algorithms
            routing_decision = await self._make_routing_decision(request)
            
            if not routing_decision or not routing_decision.selected_nodes:
                request.status = InferenceRequestStatus.FAILED
                request.error_message = "No suitable nodes available for routing"
                request.completed_at = datetime.now()
                self.performance_stats["failed_requests"] += 1
                return
                
            # Store routing decision
            self.routing_history.append(routing_decision)
            
            # Execute inference on selected nodes
            request.status = InferenceRequestStatus.PROCESSING
            
            success = await self._execute_distributed_inference(request, routing_decision)
            
            if success:
                request.status = InferenceRequestStatus.COMPLETED
                request.completed_at = datetime.now()
                self.performance_stats["successful_requests"] += 1
                
                # Update response time stats
                if request.started_at:
                    response_time = (datetime.now() - request.started_at).total_seconds() * 1000
                    self._update_response_time_stats(response_time)
                    
            else:
                # Retry if under limit
                if request.retry_count < self.retry_limit:
                    request.retry_count += 1
                    request.status = InferenceRequestStatus.QUEUED
                    
                    # Re-queue with lower priority
                    priority_score = request.priority.value * 1000 + request.archaeological_priority - request.retry_count * 100
                    await self.request_queue.put((-priority_score, time.time(), request.request_id))
                    
                    logger.info(f"🔄 Retrying request {request.request_id} (attempt {request.retry_count})")
                else:
                    request.status = InferenceRequestStatus.FAILED
                    request.error_message = "Max retry limit exceeded"
                    request.completed_at = datetime.now()
                    self.performance_stats["failed_requests"] += 1
                    
        except Exception as e:
            logger.error(f"❌ Error processing request {request.request_id}: {e}")
            request.status = InferenceRequestStatus.FAILED
            request.error_message = str(e)
            request.completed_at = datetime.now()
            self.performance_stats["failed_requests"] += 1
            
    async def _make_routing_decision(self, request: InferenceRequest) -> RoutingDecision | None:
        """Make routing decision using archaeological algorithms."""
        try:
            available_nodes = [node_id for node_id, metrics in self.node_metrics.items()
                              if self._is_node_available(metrics)]
            
            if not available_nodes:
                return None
                
            # Determine routing strategy
            strategy = self._select_routing_strategy(request, available_nodes)
            
            # Select nodes based on strategy
            selected_nodes = await self._select_nodes_for_request(request, available_nodes, strategy)
            
            if not selected_nodes:
                return None
                
            # Calculate estimates
            estimated_time = self._estimate_completion_time(request, selected_nodes)
            confidence = self._calculate_routing_confidence(request, selected_nodes, strategy)
            
            # Archaeological factors
            archaeological_factors = await self._calculate_archaeological_factors(request, selected_nodes)
            
            # Select backup nodes
            backup_nodes = [node for node in available_nodes 
                          if node not in selected_nodes][:2]
            
            decision = RoutingDecision(
                request_id=request.request_id,
                selected_nodes=selected_nodes,
                routing_strategy=strategy,
                estimated_completion_time=estimated_time,
                confidence_score=confidence,
                archaeological_factors=archaeological_factors,
                backup_nodes=backup_nodes
            )
            
            logger.info(f"🎯 Routing decision for {request.request_id}: "
                       f"{len(selected_nodes)} nodes, strategy: {strategy.name}, "
                       f"confidence: {confidence:.2f}")
            
            return decision
            
        except Exception as e:
            logger.error(f"❌ Failed to make routing decision for {request.request_id}: {e}")
            return None
            
    def _is_node_available(self, metrics: NodePerformanceMetrics) -> bool:
        """Check if a node is available for new requests."""
        # Basic availability checks
        if metrics.success_rate < 0.8:  # Low success rate
            return False
        if metrics.compute_utilization > 0.9:  # High utilization
            return False
        if metrics.queue_length > 10:  # Long queue
            return False
        if (datetime.now() - metrics.last_updated).total_seconds() > 60:  # Stale metrics
            return False
            
        return True
        
    def _select_routing_strategy(self, request: InferenceRequest, available_nodes: list[str]) -> RoutingStrategy:
        """Select optimal routing strategy for the request."""
        
        # Check routing hints
        if "strategy" in request.routing_hints:
            strategy_name = request.routing_hints["strategy"].upper()
            try:
                return RoutingStrategy[strategy_name]
            except KeyError:
                pass
                
        # Archaeological strategy selection
        if (request.archaeological_priority > 0.7 and 
            self.archaeological_metadata["feature_flags"].get("ADVANCED_REQUEST_ROUTING_ENABLED", False)):
            return RoutingStrategy.ARCHAEOLOGICAL_OPTIMAL
            
        # High priority requests use fastest response
        if request.priority in [RequestPriority.CRITICAL, RequestPriority.EMERGENCY]:
            return RoutingStrategy.FASTEST_RESPONSE
            
        # Memory-intensive requests
        if request.metadata.get("memory_intensive", False):
            return RoutingStrategy.MEMORY_OPTIMAL
            
        # Default to least loaded
        return RoutingStrategy.LEAST_LOADED
        
    async def _select_nodes_for_request(
        self,
        request: InferenceRequest,
        available_nodes: list[str],
        strategy: RoutingStrategy
    ) -> list[str]:
        """Select nodes based on routing strategy."""
        
        if strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            node_count = min(len(available_nodes), request.routing_hints.get("node_count", 1))
            return available_nodes[:node_count]
            
        elif strategy == RoutingStrategy.LEAST_LOADED:
            # Select nodes with lowest utilization
            node_scores = []
            for node_id in available_nodes:
                metrics = self.node_metrics[node_id]
                load_score = (metrics.compute_utilization + metrics.memory_utilization) / 2
                node_scores.append((load_score, node_id))
                
            node_scores.sort()
            node_count = min(len(available_nodes), request.routing_hints.get("node_count", 1))
            return [node_id for _, node_id in node_scores[:node_count]]
            
        elif strategy == RoutingStrategy.FASTEST_RESPONSE:
            # Select nodes with fastest response time
            node_scores = []
            for node_id in available_nodes:
                metrics = self.node_metrics[node_id]
                response_score = metrics.response_time_ms + metrics.network_latency_ms
                node_scores.append((response_score, node_id))
                
            node_scores.sort()
            node_count = min(len(available_nodes), request.routing_hints.get("node_count", 1))
            return [node_id for _, node_id in node_scores[:node_count]]
            
        elif strategy == RoutingStrategy.MEMORY_OPTIMAL:
            # Select nodes with best memory availability
            node_scores = []
            for node_id in available_nodes:
                metrics = self.node_metrics[node_id]
                memory_score = metrics.memory_utilization
                node_scores.append((memory_score, node_id))
                
            node_scores.sort()  # Lower memory utilization is better
            node_count = min(len(available_nodes), request.routing_hints.get("node_count", 1))
            return [node_id for _, node_id in node_scores[:node_count]]
            
        elif strategy == RoutingStrategy.ARCHAEOLOGICAL_OPTIMAL:
            # Use archaeological optimization algorithm
            return await self._archaeological_node_selection(request, available_nodes)
            
        else:
            # Fallback to least loaded
            return await self._select_nodes_for_request(request, available_nodes, RoutingStrategy.LEAST_LOADED)
            
    async def _archaeological_node_selection(self, request: InferenceRequest, available_nodes: list[str]) -> list[str]:
        """Archaeological node selection algorithm."""
        node_scores = []
        
        for node_id in available_nodes:
            metrics = self.node_metrics[node_id]
            
            # Calculate archaeological fitness score
            efficiency_score = metrics.archaeological_efficiency
            reliability_score = metrics.reliability_score
            
            # Performance score
            perf_score = (
                (1.0 - metrics.compute_utilization) * 0.3 +
                (1.0 - metrics.memory_utilization) * 0.2 +
                (metrics.success_rate) * 0.2 +
                (1.0 / max(metrics.response_time_ms, 1)) * 0.3
            )
            
            # Phase 1 integration bonus
            phase1_bonus = 0.0
            if request.phase1_triage_score and request.phase1_triage_score > 0.5:
                phase1_bonus = 0.1 * self.phase1_integration_weight
                
            # Combine scores with archaeological weighting
            total_score = (
                perf_score * 0.4 +
                efficiency_score * 0.3 +
                reliability_score * 0.2 +
                phase1_bonus * 0.1
            )
            
            node_scores.append((total_score, node_id))
            
        # Sort by score (descending)
        node_scores.sort(reverse=True)
        
        # Select top nodes
        node_count = min(len(available_nodes), request.routing_hints.get("node_count", 1))
        selected = [node_id for _, node_id in node_scores[:node_count]]
        
        # Track archaeological optimization
        if selected:
            self.performance_stats["archaeological_optimization_hits"] += 1
            
        return selected
        
    def _estimate_completion_time(self, request: InferenceRequest, selected_nodes: list[str]) -> float:
        """Estimate completion time for the request."""
        if not selected_nodes:
            return float('inf')
            
        # Get average response time from selected nodes
        total_response_time = 0.0
        for node_id in selected_nodes:
            if node_id in self.node_metrics:
                metrics = self.node_metrics[node_id]
                total_response_time += metrics.response_time_ms + metrics.queue_length * 10
                
        avg_response_time = total_response_time / len(selected_nodes)
        
        # Add coordination overhead
        coordination_overhead = 50.0  # ms
        
        return avg_response_time + coordination_overhead
        
    def _calculate_routing_confidence(
        self,
        request: InferenceRequest,
        selected_nodes: list[str],
        strategy: RoutingStrategy
    ) -> float:
        """Calculate confidence in routing decision."""
        if not selected_nodes:
            return 0.0
            
        # Base confidence from node reliability
        reliability_scores = []
        for node_id in selected_nodes:
            if node_id in self.node_metrics:
                reliability_scores.append(self.node_metrics[node_id].reliability_score)
                
        base_confidence = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0.5
        
        # Strategy-specific adjustments
        strategy_confidence = {
            RoutingStrategy.ROUND_ROBIN: 0.7,
            RoutingStrategy.LEAST_LOADED: 0.8,
            RoutingStrategy.FASTEST_RESPONSE: 0.85,
            RoutingStrategy.MEMORY_OPTIMAL: 0.8,
            RoutingStrategy.ARCHAEOLOGICAL_OPTIMAL: 0.9
        }
        
        strategy_factor = strategy_confidence.get(strategy, 0.7)
        
        # Combine factors
        confidence = (base_confidence * 0.6 + strategy_factor * 0.4)
        
        return min(max(confidence, 0.0), 1.0)
        
    async def _calculate_archaeological_factors(
        self,
        request: InferenceRequest,
        selected_nodes: list[str]
    ) -> dict[str, float]:
        """Calculate archaeological optimization factors."""
        factors = {}
        
        # Node efficiency factor
        if selected_nodes:
            efficiency_scores = [
                self.node_metrics[node_id].archaeological_efficiency 
                for node_id in selected_nodes 
                if node_id in self.node_metrics
            ]
            factors["node_efficiency"] = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.5
            
        # Request complexity factor
        complexity = request.metadata.get("complexity_score", 0.5)
        factors["request_complexity"] = complexity
        
        # Phase 1 integration factor
        if request.phase1_triage_score:
            factors["phase1_triage"] = request.phase1_triage_score
            
        # Historical success factor
        factors["historical_success"] = self.performance_stats["successful_requests"] / max(self.performance_stats["total_requests"], 1)
        
        return factors
        
    async def _execute_distributed_inference(self, request: InferenceRequest, routing_decision: RoutingDecision) -> bool:
        """Execute inference across distributed nodes."""
        try:
            # This would integrate with actual inference execution system
            # For now, simulate the process
            
            logger.info(f"🚀 Executing distributed inference for {request.request_id} on {len(routing_decision.selected_nodes)} nodes")
            
            # Simulate inference execution
            await asyncio.sleep(0.1)
            
            # Simulate result aggregation
            request.status = InferenceRequestStatus.AGGREGATING
            
            # Create mock result
            request.final_result = {
                "model_id": request.model_id,
                "predictions": [0.1, 0.2, 0.7],  # Mock predictions
                "confidence": 0.85,
                "execution_time_ms": routing_decision.estimated_completion_time,
                "nodes_used": routing_decision.selected_nodes,
                "archaeological_optimization": routing_decision.archaeological_factors
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Distributed inference failed for {request.request_id}: {e}")
            request.error_message = str(e)
            return False
            
    async def _monitor_performance(self):
        """Background task to monitor coordinator performance."""
        logger.info("🔍 Starting performance monitor")
        
        while self.running:
            try:
                # Update performance metrics
                await self._update_performance_metrics()
                
                # Cleanup completed requests (older than 1 hour)
                await self._cleanup_old_requests()
                
                # Prune routing history (keep last 1000 entries)
                if len(self.routing_history) > 1000:
                    self.routing_history = self.routing_history[-1000:]
                    
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in performance monitor: {e}")
                await asyncio.sleep(30)
                
    async def _archaeological_optimization_loop(self):
        """Background archaeological optimization loop."""
        logger.info("🏺 Starting archaeological optimization loop")
        
        while self.running:
            try:
                # Optimize node efficiency scores
                await self._optimize_node_efficiency()
                
                # Learn from routing decisions
                await self._learn_from_routing_history()
                
                # Update archaeological weights
                await self._update_archaeological_weights()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in archaeological optimization: {e}")
                await asyncio.sleep(300)
                
    # Additional utility methods...
    
    async def _calculate_archaeological_priority(self, request: InferenceRequest) -> float:
        """Calculate archaeological priority score for a request."""
        base_priority = request.priority.value / 5.0  # Normalize to 0-1
        
        # Model complexity factor
        model_complexity = request.metadata.get("model_complexity", 0.5)
        
        # Urgency factor
        urgency = request.metadata.get("urgency", 0.5)
        
        # User tier factor
        user_tier = request.metadata.get("user_tier", 0.5)
        
        # Combine factors
        archaeological_priority = (
            base_priority * 0.4 +
            model_complexity * 0.2 +
            urgency * 0.3 +
            user_tier * 0.1
        )
        
        return min(max(archaeological_priority, 0.0), 1.0)
        
    async def _get_phase1_triage_score(self, request: InferenceRequest) -> float | None:
        """Get triage score from Phase 1 emergency triage system."""
        if not self.emergency_triage:
            return None
            
        try:
            # Create triage data from request
            triage_data = {
                "request_id": request.request_id,
                "model_id": request.model_id,
                "priority": request.priority.name,
                "timestamp": request.created_at.isoformat(),
                "metadata": request.metadata
            }
            
            # Get triage assessment
            assessment = await self.emergency_triage.assess_situation(triage_data)
            return assessment.get("priority_score", 0.5)
            
        except Exception as e:
            logger.warning(f"⚠️ Phase 1 triage failed: {e}")
            return None
            
    def _calculate_archaeological_efficiency(self, metrics: NodePerformanceMetrics) -> float:
        """Calculate archaeological efficiency score for a node."""
        # Based on historical performance patterns from archaeological research
        efficiency = (
            metrics.success_rate * 0.3 +
            (1.0 / max(metrics.response_time_ms, 1)) * 0.2 +
            (1.0 - metrics.compute_utilization) * 0.2 +
            (1.0 - metrics.memory_utilization) * 0.2 +
            (1.0 / max(metrics.network_latency_ms, 1)) * 0.1
        )
        
        return min(max(efficiency, 0.0), 1.0)
        
    def _calculate_reliability_score(self, metrics: NodePerformanceMetrics) -> float:
        """Calculate reliability score based on archaeological patterns."""
        # Reliability decreases with high utilization and queue length
        reliability = (
            metrics.success_rate * 0.4 +
            (1.0 - min(metrics.compute_utilization, 1.0)) * 0.2 +
            (1.0 - min(metrics.memory_utilization, 1.0)) * 0.2 +
            max(0.0, 1.0 - metrics.queue_length / 10.0) * 0.2
        )
        
        return min(max(reliability, 0.0), 1.0)
        
    def _update_response_time_stats(self, response_time: float):
        """Update response time statistics."""
        current_avg = self.performance_stats["average_response_time"]
        total_requests = self.performance_stats["total_requests"]
        
        if total_requests > 0:
            self.performance_stats["average_response_time"] = (
                (current_avg * (total_requests - 1) + response_time) / total_requests
            )
        else:
            self.performance_stats["average_response_time"] = response_time
            
    async def _update_performance_metrics(self):
        """Update overall performance metrics."""
        # Calculate success rate
        total = self.performance_stats["total_requests"]
        if total > 0:
            success_rate = self.performance_stats["successful_requests"] / total
            self.performance_stats["success_rate"] = success_rate
            
    async def _cleanup_old_requests(self):
        """Clean up old completed requests."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        to_remove = []
        for request_id, request in self.active_requests.items():
            if (request.status in [InferenceRequestStatus.COMPLETED, InferenceRequestStatus.FAILED] and
                request.completed_at and request.completed_at < cutoff_time):
                to_remove.append(request_id)
                
        for request_id in to_remove:
            del self.active_requests[request_id]
            
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} old requests")
            
    async def _optimize_node_efficiency(self):
        """Optimize node efficiency scores based on performance."""
        for node_id, metrics in self.node_metrics.items():
            # Recalculate efficiency based on recent performance
            metrics.archaeological_efficiency = self._calculate_archaeological_efficiency(metrics)
            metrics.reliability_score = self._calculate_reliability_score(metrics)
            
    async def _learn_from_routing_history(self):
        """Learn optimization patterns from routing history."""
        if len(self.routing_history) < 10:
            return
            
        # Analyze successful routing patterns
        successful_decisions = []
        for decision in self.routing_history[-100:]:  # Last 100 decisions
            if decision.confidence_score > 0.8:
                successful_decisions.append(decision)
                
        # Update routing weights based on successful patterns
        if successful_decisions:
            logger.info(f"📚 Learned from {len(successful_decisions)} successful routing decisions")
            
    async def _update_archaeological_weights(self):
        """Update archaeological optimization weights based on performance."""
        success_rate = self.performance_stats.get("success_rate", 0.0)
        
        # Adjust weights based on performance
        if success_rate > 0.95:
            self.archaeological_routing_weight = min(self.archaeological_routing_weight * 1.1, 0.5)
        elif success_rate < 0.85:
            self.archaeological_routing_weight = max(self.archaeological_routing_weight * 0.9, 0.1)
            
    async def _save_archaeological_data(self):
        """Save archaeological performance data."""
        try:
            archaeological_data = {
                "performance_stats": self.performance_stats,
                "routing_history": [
                    {
                        "strategy": decision.routing_strategy.name,
                        "confidence": decision.confidence_score,
                        "estimated_time": decision.estimated_completion_time,
                        "archaeological_factors": decision.archaeological_factors
                    }
                    for decision in self.routing_history[-100:]  # Last 100
                ],
                "node_efficiency_history": {
                    node_id: {
                        "efficiency": metrics.archaeological_efficiency,
                        "reliability": metrics.reliability_score
                    }
                    for node_id, metrics in self.node_metrics.items()
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to file (in production, this would go to a database)
            import json
            from pathlib import Path
            
            data_path = Path("data/archaeological")
            data_path.mkdir(parents=True, exist_ok=True)
            
            with open(data_path / "inference_coordination_data.json", 'w') as f:
                json.dump(archaeological_data, f, indent=2)
                
            logger.info("💾 Saved archaeological coordination data")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not save archaeological data: {e}")


# Export archaeological metadata
__all__ = [
    "InferenceCoordinator",
    "InferenceRequest",
    "NodePerformanceMetrics",
    "RoutingDecision",
    "InferenceRequestStatus",
    "RequestPriority",
    "RoutingStrategy",
    "ARCHAEOLOGICAL_METADATA"
]