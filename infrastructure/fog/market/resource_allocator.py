"""
Dynamic Resource Allocation System with QoS Guarantees

Provides intelligent resource discovery, allocation, and quality-of-service management:
- Multi-criteria resource matching and optimization
- Real-time QoS monitoring and enforcement
- Dynamic scaling and load balancing
- SLA compliance and penalty management
- Federated-aware resource coordination

Key Features:
- Heterogeneous resource pool management
- Quality-weighted allocation algorithms
- Automatic failover and redundancy
- Performance prediction and optimization
- Integration with P2P network discovery
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
import logging
from typing import Any, Dict, List, Optional
import uuid

from .auction_engine import AuctionEngine, ResourceRequirement
from .pricing_manager import DynamicPricingManager
from .market_orchestrator import AllocationStrategy

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of resources available in the system"""

    MOBILE_DEVICE = "mobile_device"
    EDGE_SERVER = "edge_server"
    CLOUD_INSTANCE = "cloud_instance"
    GPU_CLUSTER = "gpu_cluster"
    SPECIALIZED_HARDWARE = "specialized_hardware"


class QoSMetric(str, Enum):
    """Quality of Service metrics"""

    LATENCY = "latency"
    THROUGHPUT = "throughput"
    RELIABILITY = "reliability"
    AVAILABILITY = "availability"
    PRIVACY = "privacy"
    COST = "cost"


class AllocationStatus(str, Enum):
    """Resource allocation status"""

    DISCOVERING = "discovering"
    EVALUATING = "evaluating"
    ALLOCATING = "allocating"
    MONITORING = "monitoring"
    SCALING = "scaling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ResourceNode:
    """Represents a resource node in the system"""

    node_id: str
    node_type: ResourceType

    # Capabilities
    cpu_cores: Decimal
    memory_gb: Decimal
    storage_gb: Decimal
    bandwidth_mbps: Decimal
    gpu_count: int = 0

    # Quality metrics
    trust_score: Decimal = Decimal("0.5")
    reputation_score: Decimal = Decimal("0.5")
    availability_score: Decimal = Decimal("0.95")
    latency_ms: Decimal = Decimal("100")

    # Location and network
    region: str = "unknown"
    zone: str = "unknown"
    network_tier: str = "standard"

    # Status and capacity
    current_load: Decimal = Decimal("0.0")  # 0.0 to 1.0
    max_concurrent_jobs: int = 10
    active_jobs: int = 0

    # Pricing and costs
    cost_per_hour: Decimal = Decimal("1.0")
    cost_per_request: Decimal = Decimal("0.1")

    # Metadata
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    uptime_percentage: Decimal = Decimal("99.0")
    specializations: List[str] = field(default_factory=list)

    def calculate_quality_score(self, weights: Dict[str, Decimal] = None) -> Decimal:
        """Calculate composite quality score"""
        if weights is None:
            weights = {
                "trust": Decimal("0.3"),
                "reputation": Decimal("0.2"),
                "availability": Decimal("0.2"),
                "latency": Decimal("0.2"),
                "load": Decimal("0.1"),
            }

        # Normalize latency (lower is better)
        latency_score = max(Decimal("0"), Decimal("1") - (self.latency_ms / Decimal("1000")))

        # Normalize load (lower is better)
        load_score = Decimal("1") - self.current_load

        quality_score = (
            self.trust_score * weights["trust"]
            + self.reputation_score * weights["reputation"]
            + self.availability_score * weights["availability"]
            + latency_score * weights["latency"]
            + load_score * weights["load"]
        )

        return min(Decimal("1"), max(Decimal("0"), quality_score))

    def can_handle_workload(self, requirements: ResourceRequirement) -> bool:
        """Check if node can handle the workload requirements"""
        return (
            self.cpu_cores >= requirements.cpu_cores
            and self.memory_gb >= requirements.memory_gb
            and self.storage_gb >= requirements.storage_gb
            and self.bandwidth_mbps >= requirements.bandwidth_mbps
            and self.trust_score >= requirements.min_trust_score
            and self.latency_ms <= requirements.max_latency_ms
            and self.active_jobs < self.max_concurrent_jobs
        )

    def is_healthy(self) -> bool:
        """Check if node is healthy and available"""
        return (
            (datetime.now(UTC) - self.last_heartbeat).total_seconds() < 300  # 5 min heartbeat
            and self.current_load < Decimal("0.9")  # Not overloaded
            and self.availability_score > Decimal("0.8")  # Reasonable availability
        )


@dataclass
class QoSRequirement:
    """Quality of Service requirements for resource allocation"""

    max_latency_ms: Decimal = Decimal("500")
    min_throughput_ops_per_sec: Decimal = Decimal("100")
    min_availability_percentage: Decimal = Decimal("95")
    min_reliability_percentage: Decimal = Decimal("95")
    max_cost_per_hour: Decimal = Decimal("100")
    privacy_level: str = "medium"  # low, medium, high, critical

    # SLA requirements
    uptime_guarantee_percentage: Decimal = Decimal("99")
    response_time_p95_ms: Decimal = Decimal("200")
    recovery_time_minutes: Decimal = Decimal("5")

    # Penalties for SLA violations
    latency_penalty_per_ms: Decimal = Decimal("0.01")
    downtime_penalty_per_minute: Decimal = Decimal("1.0")
    reliability_penalty_per_failure: Decimal = Decimal("10.0")


@dataclass
class AllocationPlan:
    """Resource allocation plan with QoS guarantees"""

    plan_id: str
    requirements: ResourceRequirement
    qos_requirements: QoSRequirement

    # Selected resources
    primary_nodes: List[ResourceNode]
    backup_nodes: List[ResourceNode] = field(default_factory=list)

    # Allocation details
    total_cost: Decimal = Decimal("0")
    expected_quality_score: Decimal = Decimal("0")
    estimated_completion_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    # QoS predictions
    predicted_latency_ms: Decimal = Decimal("0")
    predicted_throughput: Decimal = Decimal("0")
    predicted_availability: Decimal = Decimal("0")

    # Risk assessment
    risk_score: Decimal = Decimal("0")  # 0.0 to 1.0
    failure_probability: Decimal = Decimal("0.05")
    sla_compliance_probability: Decimal = Decimal("0.95")

    def calculate_redundancy_level(self) -> int:
        """Calculate redundancy level based on backup nodes"""
        return len(self.backup_nodes)

    def estimate_total_cost(self, duration_hours: Decimal) -> Decimal:
        """Estimate total cost including primary and backup nodes"""
        primary_cost = sum(node.cost_per_hour for node in self.primary_nodes) * duration_hours
        backup_cost = sum(node.cost_per_hour for node in self.backup_nodes) * duration_hours * Decimal("0.5")
        return primary_cost + backup_cost


class DynamicResourceAllocator:
    """
    Dynamic resource allocation system with QoS guarantees

    Features:
    - Multi-criteria resource discovery and matching
    - Real-time quality monitoring and enforcement
    - Automatic scaling and failover
    - SLA compliance tracking and penalty management
    """

    def __init__(self, pricing_manager: DynamicPricingManager = None, auction_engine: AuctionEngine = None):
        self.pricing_manager = pricing_manager
        self.auction_engine = auction_engine

        # Resource pools
        self.available_nodes: Dict[str, ResourceNode] = {}
        self.allocated_nodes: Dict[str, ResourceNode] = {}
        self.node_performance_history: Dict[str, List[Dict]] = {}

        # Active allocations
        self.active_allocations: Dict[str, AllocationPlan] = {}
        self.allocation_monitoring: Dict[str, Dict[str, Any]] = {}

        # QoS monitoring
        self.qos_violations: List[Dict[str, Any]] = []
        self.sla_penalties: Dict[str, Decimal] = {}

        # Configuration
        self.config = {
            "discovery_timeout_seconds": 30,
            "allocation_timeout_seconds": 300,
            "health_check_interval_seconds": 60,
            "redundancy_factor": 1.2,  # Over-provision by 20%
            "quality_weight": Decimal("0.6"),
            "cost_weight": Decimal("0.4"),
        }

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        logger.info("Dynamic resource allocator initialized")

    async def start(self):
        """Start the resource allocator"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Dynamic resource allocator started")

    async def stop(self):
        """Stop the resource allocator"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        logger.info("Dynamic resource allocator stopped")

    async def discover_resources(
        self, requirements: ResourceRequirement, qos_requirements: QoSRequirement, discovery_timeout: int = None
    ) -> List[ResourceNode]:
        """Discover suitable resources based on requirements"""

        discovery_timeout or self.config["discovery_timeout_seconds"]
        start_time = datetime.now(UTC)

        # Phase 1: Filter by basic requirements
        candidate_nodes = []

        for node in self.available_nodes.values():
            if (
                node.can_handle_workload(requirements)
                and node.is_healthy()
                and self._meets_qos_requirements(node, qos_requirements)
            ):
                candidate_nodes.append(node)

        # Phase 2: Score and rank candidates
        scored_candidates = []

        for node in candidate_nodes:
            quality_score = node.calculate_quality_score()
            cost_score = self._calculate_cost_score(node, requirements)

            # Composite score
            composite_score = quality_score * self.config["quality_weight"] + cost_score * self.config["cost_weight"]

            scored_candidates.append((node, composite_score))

        # Sort by composite score (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top candidates
        discovered_nodes = [node for node, _ in scored_candidates[:50]]  # Top 50

        discovery_time = (datetime.now(UTC) - start_time).total_seconds()

        logger.info(
            f"Resource discovery completed: {len(discovered_nodes)} nodes found "
            f"from {len(candidate_nodes)} candidates in {discovery_time:.2f}s"
        )

        return discovered_nodes

    async def create_allocation_plan(
        self,
        requirements: ResourceRequirement,
        qos_requirements: QoSRequirement,
        discovered_nodes: List[ResourceNode],
        allocation_strategy: AllocationStrategy = AllocationStrategy.BALANCED,
    ) -> AllocationPlan:
        """Create optimized allocation plan with QoS guarantees"""

        plan_id = f"plan_{uuid.uuid4().hex[:8]}"

        # Select primary nodes based on strategy
        primary_nodes = self._select_primary_nodes(requirements, discovered_nodes, allocation_strategy)

        # Select backup nodes for redundancy
        backup_nodes = self._select_backup_nodes(requirements, qos_requirements, discovered_nodes, primary_nodes)

        # Create allocation plan
        plan = AllocationPlan(
            plan_id=plan_id,
            requirements=requirements,
            qos_requirements=qos_requirements,
            primary_nodes=primary_nodes,
            backup_nodes=backup_nodes,
        )

        # Calculate metrics
        plan.total_cost = plan.estimate_total_cost(requirements.duration_hours)
        plan.expected_quality_score = self._calculate_plan_quality_score(plan)
        plan.predicted_latency_ms = self._predict_latency(plan)
        plan.predicted_throughput = self._predict_throughput(plan)
        plan.predicted_availability = self._predict_availability(plan)
        plan.risk_score = self._calculate_risk_score(plan)

        return plan

    async def execute_allocation_plan(self, plan: AllocationPlan, requester_id: str) -> str:
        """Execute allocation plan and start resource monitoring"""

        allocation_id = f"alloc_{uuid.uuid4().hex[:8]}"

        try:
            # Reserve primary nodes
            for node in plan.primary_nodes:
                await self._reserve_node(node, allocation_id, "primary")

            # Reserve backup nodes
            for node in plan.backup_nodes:
                await self._reserve_node(node, allocation_id, "backup")

            # Start monitoring
            self.active_allocations[allocation_id] = plan
            self.allocation_monitoring[allocation_id] = {
                "requester_id": requester_id,
                "status": AllocationStatus.MONITORING,
                "start_time": datetime.now(UTC),
                "qos_violations": [],
                "performance_metrics": {},
                "sla_compliance": True,
            }

            logger.info(
                f"Allocation plan {plan.plan_id} executed as {allocation_id}: "
                f"{len(plan.primary_nodes)} primary + {len(plan.backup_nodes)} backup nodes"
            )

            return allocation_id

        except Exception as e:
            logger.error(f"Failed to execute allocation plan {plan.plan_id}: {e}")
            # Cleanup any partially allocated resources
            await self._cleanup_failed_allocation(plan, allocation_id)
            raise

    async def monitor_allocation_qos(self, allocation_id: str) -> Dict[str, Any]:
        """Monitor QoS metrics for active allocation"""

        if allocation_id not in self.active_allocations:
            raise ValueError(f"Allocation {allocation_id} not found")

        plan = self.active_allocations[allocation_id]
        monitoring_info = self.allocation_monitoring[allocation_id]

        # Collect current metrics from nodes
        current_metrics = {}
        violations = []

        for node in plan.primary_nodes:
            node_metrics = await self._collect_node_metrics(node)
            current_metrics[node.node_id] = node_metrics

            # Check for QoS violations
            node_violations = self._check_qos_violations(node, node_metrics, plan.qos_requirements)
            violations.extend(node_violations)

        # Update monitoring data
        monitoring_info["performance_metrics"] = current_metrics
        monitoring_info["qos_violations"].extend(violations)

        # Calculate overall QoS compliance
        qos_status = self._calculate_qos_compliance(plan, current_metrics)

        # Check if scaling/failover is needed
        if violations:
            await self._handle_qos_violations(allocation_id, violations)

        return {
            "allocation_id": allocation_id,
            "qos_status": qos_status,
            "current_metrics": current_metrics,
            "violations": violations,
            "sla_compliance": monitoring_info["sla_compliance"],
            "uptime": self._calculate_uptime(allocation_id),
        }

    async def scale_allocation(self, allocation_id: str, scale_factor: Decimal) -> bool:
        """Dynamically scale allocation based on demand"""

        if allocation_id not in self.active_allocations:
            return False

        plan = self.active_allocations[allocation_id]

        try:
            if scale_factor > Decimal("1"):
                # Scale up - add more nodes
                additional_nodes_needed = int((scale_factor - Decimal("1")) * len(plan.primary_nodes))

                # Discover additional resources
                additional_nodes = await self.discover_resources(plan.requirements, plan.qos_requirements)

                # Select best additional nodes
                if len(additional_nodes) >= additional_nodes_needed:
                    new_nodes = additional_nodes[:additional_nodes_needed]

                    # Add to allocation
                    for node in new_nodes:
                        await self._reserve_node(node, allocation_id, "scaled")
                        plan.primary_nodes.append(node)

                    logger.info(f"Scaled up allocation {allocation_id}: added {len(new_nodes)} nodes")
                    return True

            elif scale_factor < Decimal("1"):
                # Scale down - remove nodes
                nodes_to_remove = int((Decimal("1") - scale_factor) * len(plan.primary_nodes))

                # Select nodes to remove (lowest performing first)
                nodes_by_performance = sorted(plan.primary_nodes, key=lambda n: n.calculate_quality_score())

                for i in range(min(nodes_to_remove, len(nodes_by_performance))):
                    node = nodes_by_performance[i]
                    await self._release_node(node, allocation_id)
                    plan.primary_nodes.remove(node)

                logger.info(f"Scaled down allocation {allocation_id}: removed {nodes_to_remove} nodes")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to scale allocation {allocation_id}: {e}")
            return False

    async def get_allocation_status(self, allocation_id: str) -> Dict[str, Any]:
        """Get comprehensive allocation status"""

        if allocation_id not in self.active_allocations:
            return {"error": "Allocation not found"}

        plan = self.active_allocations[allocation_id]
        monitoring_info = self.allocation_monitoring[allocation_id]

        # Calculate current status
        status = {
            "allocation_id": allocation_id,
            "plan_id": plan.plan_id,
            "status": monitoring_info["status"].value,
            "start_time": monitoring_info["start_time"].isoformat(),
            "duration": (datetime.now(UTC) - monitoring_info["start_time"]).total_seconds(),
            "resources": {
                "primary_nodes": len(plan.primary_nodes),
                "backup_nodes": len(plan.backup_nodes),
                "total_cost": float(plan.total_cost),
                "expected_quality": float(plan.expected_quality_score),
            },
            "qos_metrics": {
                "predicted_latency_ms": float(plan.predicted_latency_ms),
                "predicted_throughput": float(plan.predicted_throughput),
                "predicted_availability": float(plan.predicted_availability),
                "risk_score": float(plan.risk_score),
            },
            "compliance": {
                "sla_compliant": monitoring_info["sla_compliance"],
                "violation_count": len(monitoring_info["qos_violations"]),
                "uptime_percentage": float(self._calculate_uptime(allocation_id)),
            },
            "node_details": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type.value,
                    "current_load": float(node.current_load),
                    "quality_score": float(node.calculate_quality_score()),
                    "role": "primary" if node in plan.primary_nodes else "backup",
                }
                for node in plan.primary_nodes + plan.backup_nodes
            ],
        }

        return status

    def register_resource_node(self, node: ResourceNode):
        """Register a new resource node"""
        self.available_nodes[node.node_id] = node
        logger.info(f"Registered resource node {node.node_id} ({node.node_type.value})")

    def unregister_resource_node(self, node_id: str):
        """Unregister a resource node"""
        if node_id in self.available_nodes:
            del self.available_nodes[node_id]
            logger.info(f"Unregistered resource node {node_id}")

    async def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]):
        """Update performance metrics for a node"""
        if node_id in self.available_nodes:
            node = self.available_nodes[node_id]

            # Update basic metrics
            node.current_load = Decimal(str(metrics.get("load", 0.0)))
            node.active_jobs = metrics.get("active_jobs", 0)
            node.last_heartbeat = datetime.now(UTC)

            # Update performance history
            if node_id not in self.node_performance_history:
                self.node_performance_history[node_id] = []

            self.node_performance_history[node_id].append(
                {
                    "timestamp": datetime.now(UTC),
                    "metrics": metrics.copy(),
                }
            )

            # Keep only last 100 entries
            if len(self.node_performance_history[node_id]) > 100:
                self.node_performance_history[node_id] = self.node_performance_history[node_id][-100:]

    # Private methods

    def _meets_qos_requirements(self, node: ResourceNode, qos_req: QoSRequirement) -> bool:
        """Check if node meets QoS requirements"""
        return (
            node.latency_ms <= qos_req.max_latency_ms
            and node.availability_score >= (qos_req.min_availability_percentage / Decimal("100"))
            and node.cost_per_hour <= qos_req.max_cost_per_hour
            and node.uptime_percentage >= (qos_req.uptime_guarantee_percentage / Decimal("100"))
        )

    def _calculate_cost_score(self, node: ResourceNode, requirements: ResourceRequirement) -> Decimal:
        """Calculate normalized cost score (lower cost = higher score)"""
        total_cost = node.cost_per_hour * requirements.duration_hours

        # Normalize to 0-1 range (assuming max cost of $1000/hour)
        max_cost = Decimal("1000") * requirements.duration_hours
        cost_score = Decimal("1") - (total_cost / max_cost)

        return max(Decimal("0"), min(Decimal("1"), cost_score))

    def _select_primary_nodes(
        self, requirements: ResourceRequirement, candidates: List[ResourceNode], strategy: AllocationStrategy
    ) -> List[ResourceNode]:
        """Select primary nodes based on allocation strategy"""

        participants_needed = getattr(requirements, "participants_needed", 1)

        if strategy == AllocationStrategy.LOWEST_COST:
            # Sort by cost
            candidates.sort(key=lambda n: n.cost_per_hour)
        elif strategy == AllocationStrategy.BEST_QUALITY:
            # Sort by quality score
            candidates.sort(key=lambda n: n.calculate_quality_score(), reverse=True)
        elif strategy == AllocationStrategy.FASTEST_DELIVERY:
            # Sort by latency
            candidates.sort(key=lambda n: n.latency_ms)
        else:  # BALANCED
            # Sort by composite score
            candidates.sort(
                key=lambda n: (
                    n.calculate_quality_score() * Decimal("0.6")
                    + self._calculate_cost_score(n, requirements) * Decimal("0.4")
                ),
                reverse=True,
            )

        return candidates[:participants_needed]

    def _select_backup_nodes(
        self,
        requirements: ResourceRequirement,
        qos_requirements: QoSRequirement,
        candidates: List[ResourceNode],
        primary_nodes: List[ResourceNode],
    ) -> List[ResourceNode]:
        """Select backup nodes for redundancy"""

        # Calculate redundancy needs based on reliability requirements
        reliability_req = qos_requirements.min_reliability_percentage
        backup_count = 0

        if reliability_req >= Decimal("99.9"):
            backup_count = len(primary_nodes)  # 1:1 backup ratio
        elif reliability_req >= Decimal("99.0"):
            backup_count = max(1, len(primary_nodes) // 2)  # 2:1 ratio
        elif reliability_req >= Decimal("95.0"):
            backup_count = max(1, len(primary_nodes) // 4)  # 4:1 ratio

        # Filter out primary nodes
        backup_candidates = [n for n in candidates if n not in primary_nodes]

        # Select best backup nodes
        backup_candidates.sort(key=lambda n: n.calculate_quality_score(), reverse=True)

        return backup_candidates[:backup_count]

    def _calculate_plan_quality_score(self, plan: AllocationPlan) -> Decimal:
        """Calculate overall quality score for allocation plan"""
        if not plan.primary_nodes:
            return Decimal("0")

        scores = [node.calculate_quality_score() for node in plan.primary_nodes]
        return sum(scores) / len(scores)

    def _predict_latency(self, plan: AllocationPlan) -> Decimal:
        """Predict average latency for allocation plan"""
        if not plan.primary_nodes:
            return Decimal("0")

        latencies = [node.latency_ms for node in plan.primary_nodes]
        return sum(latencies) / len(latencies)

    def _predict_throughput(self, plan: AllocationPlan) -> Decimal:
        """Predict throughput for allocation plan"""
        # Simplified throughput calculation
        total_cpu = sum(node.cpu_cores for node in plan.primary_nodes)
        return total_cpu * Decimal("100")  # Assume 100 ops/sec per CPU core

    def _predict_availability(self, plan: AllocationPlan) -> Decimal:
        """Predict availability for allocation plan"""
        if not plan.primary_nodes:
            return Decimal("0")

        # Calculate combined availability with redundancy
        primary_availability = Decimal("1")
        for node in plan.primary_nodes:
            primary_availability *= node.availability_score

        # Factor in backup nodes
        if plan.backup_nodes:
            backup_availability = Decimal("1")
            for node in plan.backup_nodes:
                backup_availability *= Decimal("1") - node.availability_score
            backup_availability = Decimal("1") - backup_availability

            # Combined availability with redundancy
            combined_availability = Decimal("1") - (
                (Decimal("1") - primary_availability) * (Decimal("1") - backup_availability)
            )
            return combined_availability

        return primary_availability

    def _calculate_risk_score(self, plan: AllocationPlan) -> Decimal:
        """Calculate risk score for allocation plan"""
        # Factors: node reliability, geographic distribution, resource diversity

        risk_factors = []

        # Node reliability risk
        avg_trust = sum(node.trust_score for node in plan.primary_nodes) / len(plan.primary_nodes)
        reliability_risk = Decimal("1") - avg_trust
        risk_factors.append(reliability_risk)

        # Geographic concentration risk
        regions = set(node.region for node in plan.primary_nodes)
        if len(regions) < len(plan.primary_nodes) / 2:
            risk_factors.append(Decimal("0.3"))  # Geographic concentration penalty

        # Resource type diversity risk
        node_types = set(node.node_type for node in plan.primary_nodes)
        if len(node_types) == 1:
            risk_factors.append(Decimal("0.2"))  # Single point of failure

        return min(Decimal("1"), sum(risk_factors) / len(risk_factors))

    async def _reserve_node(self, node: ResourceNode, allocation_id: str, role: str):
        """Reserve a node for allocation"""
        node.active_jobs += 1
        self.allocated_nodes[f"{allocation_id}:{node.node_id}"] = node
        logger.debug(f"Reserved node {node.node_id} for allocation {allocation_id} ({role})")

    async def _release_node(self, node: ResourceNode, allocation_id: str):
        """Release a node from allocation"""
        node.active_jobs = max(0, node.active_jobs - 1)
        allocation_key = f"{allocation_id}:{node.node_id}"
        if allocation_key in self.allocated_nodes:
            del self.allocated_nodes[allocation_key]
        logger.debug(f"Released node {node.node_id} from allocation {allocation_id}")

    async def _cleanup_failed_allocation(self, plan: AllocationPlan, allocation_id: str):
        """Cleanup resources from failed allocation"""
        all_nodes = plan.primary_nodes + plan.backup_nodes
        for node in all_nodes:
            await self._release_node(node, allocation_id)

    async def _collect_node_metrics(self, node: ResourceNode) -> Dict[str, Any]:
        """Collect current performance metrics from node"""
        # This would integrate with actual node monitoring systems
        return {
            "cpu_utilization": float(node.current_load),
            "memory_utilization": 0.6,  # Mock data
            "latency_ms": float(node.latency_ms),
            "throughput_ops_per_sec": 150.0,
            "availability": float(node.availability_score),
            "error_rate": 0.01,
        }

    def _check_qos_violations(
        self, node: ResourceNode, metrics: Dict[str, Any], qos_req: QoSRequirement
    ) -> List[Dict[str, Any]]:
        """Check for QoS violations on a node"""
        violations = []

        # Check latency violation
        if metrics.get("latency_ms", 0) > float(qos_req.max_latency_ms):
            violations.append(
                {
                    "type": "latency",
                    "node_id": node.node_id,
                    "expected": float(qos_req.max_latency_ms),
                    "actual": metrics["latency_ms"],
                    "severity": "high" if metrics["latency_ms"] > float(qos_req.max_latency_ms) * 2 else "medium",
                }
            )

        # Check availability violation
        if metrics.get("availability", 1.0) < float(qos_req.min_availability_percentage) / 100:
            violations.append(
                {
                    "type": "availability",
                    "node_id": node.node_id,
                    "expected": float(qos_req.min_availability_percentage) / 100,
                    "actual": metrics["availability"],
                    "severity": "critical",
                }
            )

        return violations

    def _calculate_qos_compliance(self, plan: AllocationPlan, metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate overall QoS compliance for allocation"""
        total_nodes = len(plan.primary_nodes)
        compliant_nodes = 0

        for node in plan.primary_nodes:
            node_metrics = metrics.get(node.node_id, {})
            violations = self._check_qos_violations(node, node_metrics, plan.qos_requirements)
            if not violations:
                compliant_nodes += 1

        compliance_percentage = (compliant_nodes / total_nodes) * 100 if total_nodes > 0 else 0

        return {
            "compliance_percentage": compliance_percentage,
            "compliant_nodes": compliant_nodes,
            "total_nodes": total_nodes,
            "status": "compliant" if compliance_percentage >= 95 else "non_compliant",
        }

    async def _handle_qos_violations(self, allocation_id: str, violations: List[Dict[str, Any]]):
        """Handle QoS violations through scaling or failover"""

        critical_violations = [v for v in violations if v.get("severity") == "critical"]

        if critical_violations:
            # Trigger failover for critical violations
            logger.warning(f"Critical QoS violations in allocation {allocation_id}, triggering failover")
            await self._trigger_failover(allocation_id, critical_violations)

        elif len(violations) > len(self.active_allocations[allocation_id].primary_nodes) * 0.3:
            # More than 30% of nodes have violations, scale up
            logger.info(f"Multiple QoS violations in allocation {allocation_id}, scaling up")
            await self.scale_allocation(allocation_id, Decimal("1.5"))

    async def _trigger_failover(self, allocation_id: str, violations: List[Dict[str, Any]]):
        """Trigger failover to backup nodes"""
        if allocation_id not in self.active_allocations:
            return

        plan = self.active_allocations[allocation_id]

        # Identify failed nodes
        failed_node_ids = set(v["node_id"] for v in violations)

        # Replace failed nodes with backup nodes
        replacements = 0
        for node in plan.primary_nodes.copy():
            if node.node_id in failed_node_ids and plan.backup_nodes:
                # Move backup node to primary
                backup_node = plan.backup_nodes.pop(0)
                plan.primary_nodes.remove(node)
                plan.primary_nodes.append(backup_node)

                # Release failed node
                await self._release_node(node, allocation_id)

                replacements += 1

        if replacements > 0:
            logger.info(f"Failover completed for allocation {allocation_id}: {replacements} nodes replaced")

    def _calculate_uptime(self, allocation_id: str) -> Decimal:
        """Calculate uptime percentage for allocation"""
        if allocation_id not in self.allocation_monitoring:
            return Decimal("0")

        monitoring_info = self.allocation_monitoring[allocation_id]
        start_time = monitoring_info["start_time"]
        current_time = datetime.now(UTC)

        total_seconds = (current_time - start_time).total_seconds()

        # Calculate downtime from violations
        violations = monitoring_info.get("qos_violations", [])
        downtime_seconds = (
            sum(1 for v in violations if v.get("type") == "availability") * 60
        )  # Assume 1 min per violation

        uptime_seconds = max(0, total_seconds - downtime_seconds)
        uptime_percentage = (uptime_seconds / max(1, total_seconds)) * 100

        return Decimal(str(min(100, uptime_percentage)))

    async def _monitoring_loop(self):
        """Background monitoring loop for active allocations"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                for allocation_id in list(self.active_allocations.keys()):
                    try:
                        await self.monitor_allocation_qos(allocation_id)
                    except Exception as e:
                        logger.error(f"Error monitoring allocation {allocation_id}: {e}")

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _health_check_loop(self):
        """Background health check loop for resource nodes"""
        while True:
            try:
                await asyncio.sleep(self.config["health_check_interval_seconds"])

                # Check health of all nodes
                unhealthy_nodes = []

                for node_id, node in self.available_nodes.items():
                    if not node.is_healthy():
                        unhealthy_nodes.append(node_id)

                # Remove unhealthy nodes
                for node_id in unhealthy_nodes:
                    self.unregister_resource_node(node_id)
                    logger.warning(f"Removed unhealthy node: {node_id}")

            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(120)


# Global allocator instance
_resource_allocator: Optional[DynamicResourceAllocator] = None


async def get_resource_allocator() -> DynamicResourceAllocator:
    """Get global resource allocator instance"""
    global _resource_allocator

    if _resource_allocator is None:
        _resource_allocator = DynamicResourceAllocator()
        await _resource_allocator.start()

    return _resource_allocator
