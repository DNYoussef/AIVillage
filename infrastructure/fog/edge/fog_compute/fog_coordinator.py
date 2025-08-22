"""
Fog Computing Coordinator

Orchestrates distributed computing across edge devices using idle resources:
- Manages fog node clusters with battery/charging awareness
- Distributes workloads based on device capabilities
- Coordinates idle charging compute utilization
- Provides mobile-optimized fog computing policies

This creates a decentralized cloud alternative using edge device resources.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .fog_node import FogNode

from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of fog computing tasks"""

    INFERENCE = "inference"
    TRAINING = "training"
    EMBEDDING = "embedding"
    PREPROCESSING = "preprocessing"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class TaskPriority(Enum):
    """Task priority levels"""

    LOW = 1
    NORMAL = 3
    HIGH = 7
    CRITICAL = 10


class FogNodeState(Enum):
    """Fog node operational states"""

    OFFLINE = "offline"
    IDLE = "idle"
    ACTIVE = "active"
    CHARGING = "charging"
    OVERLOADED = "overloaded"
    ERROR = "error"


@dataclass
class ComputeCapacity:
    """Represents compute capacity of a fog node"""

    cpu_cores: int
    cpu_utilization: float  # 0.0 to 1.0
    memory_mb: int
    memory_used_mb: int
    gpu_available: bool
    gpu_memory_mb: int

    # Power/thermal constraints
    battery_powered: bool
    battery_percent: int | None = None
    is_charging: bool = False
    thermal_state: str = "normal"
    power_budget_watts: float = 100.0

    # Network characteristics
    network_bandwidth_mbps: float = 10.0
    network_latency_ms: float = 50.0

    @property
    def available_cpu_cores(self) -> float:
        """Available CPU cores considering utilization"""
        return self.cpu_cores * (1.0 - self.cpu_utilization)

    @property
    def available_memory_mb(self) -> int:
        """Available memory in MB"""
        return max(0, self.memory_mb - self.memory_used_mb)

    @property
    def compute_score(self) -> float:
        """Overall compute capacity score (0.0 to 1.0)"""
        cpu_score = self.available_cpu_cores / max(1, self.cpu_cores)
        memory_score = self.available_memory_mb / max(1, self.memory_mb)

        # Adjust for power constraints
        power_factor = 1.0
        if self.battery_powered and self.battery_percent:
            if self.battery_percent < 20:
                power_factor = 0.2
            elif self.battery_percent < 50:
                power_factor = 0.6
            elif self.is_charging:
                power_factor = 1.2  # Boost when charging

        # Adjust for thermal state
        thermal_factor = {
            "normal": 1.0,
            "warm": 0.8,
            "hot": 0.4,
            "critical": 0.1,
        }.get(self.thermal_state, 0.5)

        base_score = (cpu_score + memory_score) / 2
        return min(1.0, base_score * power_factor * thermal_factor)


@dataclass
class FogTask:
    """Represents a fog computing task"""

    task_id: str
    task_type: TaskType
    priority: TaskPriority

    # Resource requirements
    cpu_cores_required: float = 1.0
    memory_mb_required: int = 512
    estimated_duration_seconds: float = 60.0
    requires_gpu: bool = False

    # Task data
    input_data: bytes = b""
    input_size_mb: float = 0.0
    expected_output_size_mb: float = 0.0

    # Scheduling info
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deadline: datetime | None = None
    assigned_node: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Status tracking
    status: str = "pending"  # pending, assigned, running, completed, failed
    progress: float = 0.0  # 0.0 to 1.0
    result_data: bytes = b""
    error_message: str = ""

    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded its deadline"""
        if self.deadline is None:
            return False
        return datetime.now(UTC) > self.deadline

    @property
    def execution_time(self) -> float | None:
        """Get task execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class FogCluster:
    """Represents a fog computing cluster"""

    cluster_id: str
    coordinator_node: str
    member_nodes: set[str] = field(default_factory=set)
    active_tasks: dict[str, str] = field(default_factory=dict)  # task_id -> node_id
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def size(self) -> int:
        """Number of nodes in cluster"""
        return len(self.member_nodes)


class FogCoordinator:
    """
    Fog Computing Coordinator

    Orchestrates distributed computing across edge devices using idle resources.
    Implements mobile-aware policies for battery/charging optimization.
    """

    def __init__(self, coordinator_id: str | None = None):
        self.coordinator_id = coordinator_id or f"fog_coord_{uuid4().hex[:8]}"

        # Node and cluster management
        self.nodes: dict[str, "FogNode"] = {}
        self.node_capacities: dict[str, ComputeCapacity] = {}
        self.clusters: dict[str, FogCluster] = {}

        # Task management
        self.pending_tasks: list[FogTask] = []
        self.active_tasks: dict[str, FogTask] = {}
        self.completed_tasks: list[FogTask] = []

        # Scheduling policies
        self.policies = {
            "prefer_charging_nodes": True,
            "battery_minimum_percent": 30,
            "thermal_throttle_threshold": 60.0,  # Celsius
            "max_concurrent_tasks_per_node": 2,
            "task_timeout_multiplier": 2.0,
            "cluster_size_target": 5,
        }

        # Statistics
        self.stats = {
            "nodes_registered": 0,
            "clusters_formed": 0,
            "tasks_scheduled": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_compute_hours": 0.0,
            "charging_compute_hours": 0.0,
            "battery_saves": 0,
        }

        # Background services
        self.scheduler_active = True

        logger.info(f"Fog Coordinator {self.coordinator_id} initialized")

    async def register_node(
        self, node_id: str, capacity: ComputeCapacity, node_metadata: dict[str, Any] | None = None
    ) -> bool:
        """Register a fog node with its compute capacity"""

        from .fog_node import FogNode

        if node_id in self.nodes:
            logger.warning(f"Node {node_id} already registered")
            return False

        # Create fog node instance
        node = FogNode(
            node_id=node_id, capacity=capacity, coordinator_id=self.coordinator_id, metadata=node_metadata or {}
        )

        self.nodes[node_id] = node
        self.node_capacities[node_id] = capacity
        self.stats["nodes_registered"] += 1

        logger.info(f"Registered fog node {node_id} with {capacity.cpu_cores} cores, {capacity.memory_mb}MB RAM")

        # Start node monitoring
        asyncio.create_task(self._monitor_node(node))

        # Consider cluster formation
        await self._evaluate_cluster_formation()

        # Start scheduler if this is the first node
        if len(self.nodes) == 1:
            asyncio.create_task(self._task_scheduler_loop())

        return True

    async def submit_task(
        self,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.NORMAL,
        cpu_cores: float = 1.0,
        memory_mb: int = 512,
        estimated_duration: float = 60.0,
        input_data: bytes = b"",
        deadline: datetime | None = None,
        requires_gpu: bool = False,
    ) -> str:
        """Submit a task to the fog computing system"""

        task_id = f"task_{uuid4().hex[:12]}"

        task = FogTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            cpu_cores_required=cpu_cores,
            memory_mb_required=memory_mb,
            estimated_duration_seconds=estimated_duration,
            input_data=input_data,
            input_size_mb=len(input_data) / (1024 * 1024),
            deadline=deadline,
            requires_gpu=requires_gpu,
        )

        self.pending_tasks.append(task)
        self.stats["tasks_scheduled"] += 1

        logger.info(f"Submitted task {task_id} ({task_type.value}) requiring {cpu_cores} cores, {memory_mb}MB")

        return task_id

    async def _task_scheduler_loop(self) -> None:
        """Main task scheduling loop"""

        while self.scheduler_active:
            try:
                await asyncio.sleep(5)  # Schedule every 5 seconds

                # Remove expired tasks
                self._cleanup_expired_tasks()

                # Sort pending tasks by priority and age
                self.pending_tasks.sort(
                    key=lambda t: (-t.priority.value, t.submitted_at)  # Higher priority first  # Older tasks first
                )

                # Schedule tasks
                scheduled_count = 0
                for task in self.pending_tasks[:]:  # Copy to avoid modification during iteration
                    node_id = await self._find_suitable_node(task)
                    if node_id:
                        await self._assign_task_to_node(task, node_id)
                        self.pending_tasks.remove(task)
                        scheduled_count += 1

                if scheduled_count > 0:
                    logger.debug(f"Scheduled {scheduled_count} tasks")

                # Update cluster formations
                await self._maintain_clusters()

                # Update statistics
                await self._update_statistics()

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(10)

    async def _find_suitable_node(self, task: FogTask) -> str | None:
        """Find the most suitable node for a task"""

        suitable_nodes = []

        for node_id, node in self.nodes.items():
            capacity = self.node_capacities.get(node_id)
            if not capacity:
                continue

            # Check basic resource requirements
            if (
                capacity.available_cpu_cores < task.cpu_cores_required
                or capacity.available_memory_mb < task.memory_mb_required
            ):
                continue

            # Check GPU requirement
            if task.requires_gpu and not capacity.gpu_available:
                continue

            # Check concurrent task limit
            node_active_tasks = len([t for t in self.active_tasks.values() if t.assigned_node == node_id])
            if node_active_tasks >= self.policies["max_concurrent_tasks_per_node"]:
                continue

            # Apply mobile-specific policies
            if not self._check_mobile_policies(capacity, task):
                continue

            # Calculate node suitability score
            score = self._calculate_node_score(capacity, task)
            suitable_nodes.append((node_id, score))

        if not suitable_nodes:
            return None

        # Sort by score (highest first) and return best node
        suitable_nodes.sort(key=lambda x: x[1], reverse=True)
        return suitable_nodes[0][0]

    def _check_mobile_policies(self, capacity: ComputeCapacity, task: FogTask) -> bool:
        """Check mobile-specific scheduling policies"""

        # Battery level check for battery-powered devices
        if capacity.battery_powered and capacity.battery_percent:
            if capacity.battery_percent < self.policies["battery_minimum_percent"]:
                # Only allow if device is charging
                if not capacity.is_charging:
                    return False

        # Thermal throttling check
        if capacity.thermal_state in ["hot", "critical"]:
            return False

        # Prefer charging devices for heavy tasks
        if (
            self.policies["prefer_charging_nodes"]
            and task.estimated_duration_seconds > 300  # > 5 minutes
            and capacity.battery_powered
            and not capacity.is_charging
            and capacity.battery_percent
            and capacity.battery_percent < 70
        ):
            return False

        return True

    def _calculate_node_score(self, capacity: ComputeCapacity, task: FogTask) -> float:
        """Calculate suitability score for a node"""

        # Base score from compute capacity
        score = capacity.compute_score

        # Bonus for charging devices (mobile optimization)
        if capacity.battery_powered and capacity.is_charging:
            score *= 1.5

        # Bonus for non-battery devices (desktop/laptop plugged in)
        if not capacity.battery_powered:
            score *= 1.2

        # Penalty for high thermal state
        thermal_penalties = {
            "normal": 1.0,
            "warm": 0.9,
            "hot": 0.5,
            "critical": 0.1,
        }
        score *= thermal_penalties.get(capacity.thermal_state, 0.5)

        # Bonus for GPU availability if required
        if task.requires_gpu and capacity.gpu_available:
            score *= 1.3

        # Network consideration for large data transfers
        if task.input_size_mb > 10:  # > 10MB
            network_factor = min(1.0, capacity.network_bandwidth_mbps / 100.0)
            score *= 0.5 + 0.5 * network_factor

        return score

    async def _assign_task_to_node(self, task: FogTask, node_id: str) -> None:
        """Assign a task to a specific node"""

        task.assigned_node = node_id
        task.status = "assigned"
        task.started_at = datetime.now(UTC)

        self.active_tasks[task.task_id] = task

        # Update node capacity (reserve resources)
        if node_id in self.node_capacities:
            capacity = self.node_capacities[node_id]
            capacity.cpu_utilization = min(
                1.0, capacity.cpu_utilization + (task.cpu_cores_required / capacity.cpu_cores)
            )
            capacity.memory_used_mb += task.memory_mb_required

        # Notify node to start task
        node = self.nodes.get(node_id)
        if node:
            asyncio.create_task(node.execute_task(task))

        logger.info(f"Assigned task {task.task_id} to node {node_id}")

    async def _monitor_node(self, node: "FogNode") -> None:
        """Monitor a fog node for capacity updates"""

        while node.node_id in self.nodes:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update node capacity
                updated_capacity = await node.get_current_capacity()
                if updated_capacity:
                    self.node_capacities[node.node_id] = updated_capacity

                # Check for completed tasks
                completed_tasks = await node.get_completed_tasks()
                for task in completed_tasks:
                    await self._handle_task_completion(task)

            except Exception as e:
                logger.warning(f"Error monitoring node {node.node_id}: {e}")
                await asyncio.sleep(60)

    async def _handle_task_completion(self, task: FogTask) -> None:
        """Handle completion of a task"""

        if task.task_id not in self.active_tasks:
            return

        # Remove from active tasks
        del self.active_tasks[task.task_id]

        # Add to completed tasks
        task.completed_at = datetime.now(UTC)
        self.completed_tasks.append(task)

        # Update statistics
        if task.status == "completed":
            self.stats["tasks_completed"] += 1

            # Track compute hours
            execution_time_hours = (task.execution_time or 0) / 3600.0
            self.stats["total_compute_hours"] += execution_time_hours

            # Track charging compute hours
            if (
                task.assigned_node
                and task.assigned_node in self.node_capacities
                and self.node_capacities[task.assigned_node].is_charging
            ):
                self.stats["charging_compute_hours"] += execution_time_hours
        else:
            self.stats["tasks_failed"] += 1

        # Release node resources
        if task.assigned_node and task.assigned_node in self.node_capacities:
            capacity = self.node_capacities[task.assigned_node]
            capacity.cpu_utilization = max(
                0.0, capacity.cpu_utilization - (task.cpu_cores_required / capacity.cpu_cores)
            )
            capacity.memory_used_mb = max(0, capacity.memory_used_mb - task.memory_mb_required)

        logger.info(f"Task {task.task_id} completed with status: {task.status}")

    def _cleanup_expired_tasks(self) -> None:
        """Remove expired tasks from pending queue"""

        initial_count = len(self.pending_tasks)
        self.pending_tasks = [t for t in self.pending_tasks if not t.is_expired]

        expired_count = initial_count - len(self.pending_tasks)
        if expired_count > 0:
            logger.warning(f"Removed {expired_count} expired tasks from queue")

    async def _evaluate_cluster_formation(self) -> None:
        """Evaluate whether to form new fog clusters"""

        # Only form clusters if we have enough nodes
        if len(self.nodes) < self.policies["cluster_size_target"]:
            return

        # Find nodes not in clusters
        unclustered_nodes = set(self.nodes.keys())
        for cluster in self.clusters.values():
            unclustered_nodes -= cluster.member_nodes
            unclustered_nodes.discard(cluster.coordinator_node)

        if len(unclustered_nodes) >= self.policies["cluster_size_target"]:
            await self._form_cluster(list(unclustered_nodes))

    async def _form_cluster(self, node_ids: list[str]) -> str:
        """Form a new fog cluster"""

        cluster_id = f"cluster_{uuid4().hex[:8]}"

        # Select coordinator (node with highest compute score)
        coordinator_node = max(
            node_ids,
            key=lambda nid: self.node_capacities.get(nid, ComputeCapacity(1, 0.5, 1024, 512, False, 0)).compute_score,
        )

        cluster = FogCluster(cluster_id=cluster_id, coordinator_node=coordinator_node, member_nodes=set(node_ids))

        self.clusters[cluster_id] = cluster
        self.stats["clusters_formed"] += 1

        logger.info(f"Formed cluster {cluster_id} with {len(node_ids)} nodes, coordinator: {coordinator_node}")

        return cluster_id

    async def _maintain_clusters(self) -> None:
        """Maintain existing clusters and handle node changes"""

        for cluster_id, cluster in list(self.clusters.items()):
            # Check if coordinator is still available
            if cluster.coordinator_node not in self.nodes:
                # Need to elect new coordinator or dissolve cluster
                available_members = [nid for nid in cluster.member_nodes if nid in self.nodes]

                if len(available_members) >= 2:
                    # Elect new coordinator
                    new_coordinator = max(
                        available_members,
                        key=lambda nid: self.node_capacities.get(
                            nid, ComputeCapacity(1, 0.5, 1024, 512, False, 0)
                        ).compute_score,
                    )
                    cluster.coordinator_node = new_coordinator
                    logger.info(f"Elected new coordinator {new_coordinator} for cluster {cluster_id}")
                else:
                    # Dissolve cluster
                    del self.clusters[cluster_id]
                    logger.info(f"Dissolved cluster {cluster_id} due to insufficient nodes")

            # Remove unavailable members
            cluster.member_nodes &= set(self.nodes.keys())

    async def _update_statistics(self) -> None:
        """Update fog computing statistics"""

        # Count battery saves (tasks that would have drained battery but ran on charging nodes)
        battery_saves = 0
        for task in self.active_tasks.values():
            if task.assigned_node and task.assigned_node in self.node_capacities:
                capacity = self.node_capacities[task.assigned_node]
                if (
                    capacity.battery_powered and capacity.is_charging and task.estimated_duration_seconds > 180
                ):  # > 3 minutes
                    battery_saves += 1

        self.stats["battery_saves"] = battery_saves

    def get_task_status(self, task_id: str) -> dict[str, Any] | None:
        """Get status of a specific task"""

        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task.task_id,
                "status": task.status,
                "progress": task.progress,
                "assigned_node": task.assigned_node,
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "estimated_completion": None,  # Could be calculated
            }

        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task.task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "assigned_node": task.assigned_node,
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "execution_time": task.execution_time,
                    "error_message": task.error_message,
                }

        # Check pending tasks
        for task in self.pending_tasks:
            if task.task_id == task_id:
                return {
                    "task_id": task.task_id,
                    "status": "pending",
                    "progress": 0.0,
                    "submitted_at": task.submitted_at.isoformat(),
                    "estimated_start_time": None,  # Could be calculated
                }

        return None

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive fog computing system status"""

        return {
            "coordinator_id": self.coordinator_id,
            "nodes": {
                "total": len(self.nodes),
                "by_state": {
                    "active": len([n for n in self.nodes.values() if n.state == FogNodeState.ACTIVE]),
                    "idle": len([n for n in self.nodes.values() if n.state == FogNodeState.IDLE]),
                    "charging": len(
                        [nid for nid, cap in self.node_capacities.items() if cap.battery_powered and cap.is_charging]
                    ),
                },
                "total_compute_capacity": {
                    "cpu_cores": sum(cap.cpu_cores for cap in self.node_capacities.values()),
                    "memory_gb": sum(cap.memory_mb for cap in self.node_capacities.values()) / 1024,
                    "available_cpu_cores": sum(cap.available_cpu_cores for cap in self.node_capacities.values()),
                    "available_memory_gb": sum(cap.available_memory_mb for cap in self.node_capacities.values()) / 1024,
                },
            },
            "clusters": {
                "total": len(self.clusters),
                "average_size": sum(c.size for c in self.clusters.values()) / max(1, len(self.clusters)),
            },
            "tasks": {
                "pending": len(self.pending_tasks),
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "by_type": {
                    task_type.value: len([t for t in self.active_tasks.values() if t.task_type == task_type])
                    for task_type in TaskType
                },
            },
            "statistics": self.stats.copy(),
            "policies": self.policies.copy(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown fog coordinator"""

        self.scheduler_active = False

        # Cancel all active tasks
        for task in self.active_tasks.values():
            task.status = "cancelled"

        # Notify all nodes
        for node in self.nodes.values():
            await node.shutdown()

        logger.info(f"Fog Coordinator {self.coordinator_id} shutdown complete")
