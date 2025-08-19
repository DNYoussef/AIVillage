# Edge Computing & Fog Coordination - Fog Computing Guide

## Overview

The AIVillage Fog Computing system enables distributed computing across heterogeneous edge devices, creating a decentralized cloud alternative. This guide covers fog cluster formation, task scheduling, resource management, and production deployment strategies for distributed AI workloads.

## Fog Computing Architecture

### Core Components

The fog computing system consists of three primary components:

1. **Fog Coordinator** - Central orchestration of distributed computing tasks
2. **Fog Nodes** - Individual edge devices participating in fog computing
3. **Fog Clusters** - Groups of nodes working together on distributed workloads

### System Overview

```python
class FogCoordinator:
    """
    Fog Computing Coordinator

    Orchestrates distributed computing across edge devices using idle resources.
    Implements mobile-aware policies for battery/charging optimization.
    """

    def __init__(self, coordinator_id: str | None = None):
        self.coordinator_id = coordinator_id or f"fog_coord_{uuid4().hex[:8]}"

        # Node and cluster management
        self.nodes: dict[str, FogNode] = {}
        self.node_capacities: dict[str, ComputeCapacity] = {}
        self.clusters: dict[str, FogCluster] = {}

        # Task management
        self.pending_tasks: list[FogTask] = []
        self.active_tasks: dict[str, FogTask] = {}
        self.completed_tasks: list[FogTask] = []
```

## Node Management

### Node Registration

Fog nodes register with the coordinator providing their compute capacity:

```python
async def register_node(
    self, node_id: str, capacity: ComputeCapacity, node_metadata: dict[str, Any] | None = None
) -> bool:
    """Register a fog node with its compute capacity"""

    if node_id in self.nodes:
        logger.warning(f"Node {node_id} already registered")
        return False

    # Create fog node instance
    node = FogNode(
        node_id=node_id,
        capacity=capacity,
        coordinator_id=self.coordinator_id,
        metadata=node_metadata or {}
    )

    self.nodes[node_id] = node
    self.node_capacities[node_id] = capacity
    self.stats["nodes_registered"] += 1

    # Start node monitoring
    asyncio.create_task(self._monitor_node(node))

    # Consider cluster formation
    await self._evaluate_cluster_formation()

    return True
```

### Compute Capacity Model

Each node reports its compute capacity including mobile-specific constraints:

```python
@dataclass
class ComputeCapacity:
    """Represents compute capacity of a fog node"""

    # Core compute resources
    cpu_cores: int
    cpu_utilization: float          # 0.0 to 1.0
    memory_mb: int
    memory_used_mb: int
    gpu_available: bool
    gpu_memory_mb: int

    # Power/thermal constraints (mobile-specific)
    battery_powered: bool
    battery_percent: int | None = None
    is_charging: bool = False
    thermal_state: str = "normal"   # normal, warm, hot, critical
    power_budget_watts: float = 100.0

    # Network characteristics
    network_bandwidth_mbps: float = 10.0
    network_latency_ms: float = 50.0

    @property
    def compute_score(self) -> float:
        """Overall compute capacity score (0.0 to 1.0)"""
        cpu_score = self.available_cpu_cores / max(1, self.cpu_cores)
        memory_score = self.available_memory_mb / max(1, self.memory_mb)

        # Apply power constraints for mobile devices
        power_factor = 1.0
        if self.battery_powered and self.battery_percent:
            if self.battery_percent < 20:
                power_factor = 0.2      # Severe penalty for low battery
            elif self.battery_percent < 50:
                power_factor = 0.6      # Moderate penalty
            elif self.is_charging:
                power_factor = 1.2      # Bonus when charging

        # Apply thermal constraints
        thermal_factor = {
            "normal": 1.0,
            "warm": 0.8,    # -20% for warm
            "hot": 0.4,     # -60% for hot
            "critical": 0.1 # -90% for critical
        }.get(self.thermal_state, 0.5)

        base_score = (cpu_score + memory_score) / 2
        return min(1.0, base_score * power_factor * thermal_factor)
```

### Node Monitoring

The coordinator continuously monitors node health and capacity:

```python
async def _monitor_node(self, node: FogNode) -> None:
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
```

## Task Scheduling

### Task Model

Fog tasks define compute requirements and constraints:

```python
@dataclass
class FogTask:
    """Represents a fog computing task"""

    task_id: str
    task_type: TaskType             # INFERENCE, TRAINING, EMBEDDING, etc.
    priority: TaskPriority          # LOW, NORMAL, HIGH, CRITICAL

    # Resource requirements
    cpu_cores_required: float = 1.0
    memory_mb_required: int = 512
    estimated_duration_seconds: float = 60.0
    requires_gpu: bool = False

    # Task data
    input_data: bytes = b""
    input_size_mb: float = 0.0
    expected_output_size_mb: float = 0.0

    # Scheduling metadata
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deadline: datetime | None = None
    assigned_node: str | None = None

    # Status tracking
    status: str = "pending"         # pending, assigned, running, completed, failed
    progress: float = 0.0           # 0.0 to 1.0
```

### Task Submission

Submit tasks to the fog computing system:

```python
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

    return task_id
```

### Scheduling Algorithm

The coordinator implements priority-based scheduling with mobile awareness:

```python
async def _task_scheduler_loop(self) -> None:
    """Main task scheduling loop"""

    while self.scheduler_active:
        try:
            await asyncio.sleep(5)  # Schedule every 5 seconds

            # Remove expired tasks
            self._cleanup_expired_tasks()

            # Sort pending tasks by priority and age
            self.pending_tasks.sort(
                key=lambda t: (-t.priority.value, t.submitted_at)
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

        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            await asyncio.sleep(10)
```

### Node Selection Algorithm

Mobile-aware node selection with charging device preference:

```python
async def _find_suitable_node(self, task: FogTask) -> str | None:
    """Find the most suitable node for a task"""

    suitable_nodes = []

    for node_id, node in self.nodes.items():
        capacity = self.node_capacities.get(node_id)
        if not capacity:
            continue

        # Check basic resource requirements
        if (capacity.available_cpu_cores < task.cpu_cores_required or
            capacity.available_memory_mb < task.memory_mb_required):
            continue

        # Check GPU requirement
        if task.requires_gpu and not capacity.gpu_available:
            continue

        # Check concurrent task limit
        node_active_tasks = len([t for t in self.active_tasks.values()
                               if t.assigned_node == node_id])
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
```

### Mobile-Specific Scheduling Policies

```python
def _check_mobile_policies(self, capacity: ComputeCapacity, task: FogTask) -> bool:
    """Check mobile-specific scheduling policies"""

    # Battery level check for battery-powered devices
    if capacity.battery_powered and capacity.battery_percent:
        if capacity.battery_percent < self.policies["battery_minimum_percent"]:  # Default: 30%
            # Only allow if device is charging
            if not capacity.is_charging:
                return False

    # Thermal throttling check
    if capacity.thermal_state in ["hot", "critical"]:
        return False

    # Prefer charging devices for heavy tasks
    if (self.policies["prefer_charging_nodes"] and
        task.estimated_duration_seconds > 300 and  # > 5 minutes
        capacity.battery_powered and
        not capacity.is_charging and
        capacity.battery_percent and capacity.battery_percent < 70):
        return False

    return True
```

### Node Scoring System

```python
def _calculate_node_score(self, capacity: ComputeCapacity, task: FogTask) -> float:
    """Calculate suitability score for a node"""

    # Base score from compute capacity
    score = capacity.compute_score

    # Bonus for charging devices (mobile optimization)
    if capacity.battery_powered and capacity.is_charging:
        score *= 1.5  # 50% bonus for charging devices

    # Bonus for non-battery devices (desktop/laptop plugged in)
    if not capacity.battery_powered:
        score *= 1.2  # 20% bonus for unlimited power

    # Penalty for high thermal state
    thermal_penalties = {
        "normal": 1.0,
        "warm": 0.9,     # -10% penalty
        "hot": 0.5,      # -50% penalty
        "critical": 0.1  # -90% penalty
    }
    score *= thermal_penalties.get(capacity.thermal_state, 0.5)

    # Bonus for GPU availability if required
    if task.requires_gpu and capacity.gpu_available:
        score *= 1.3  # 30% bonus for GPU tasks

    # Network consideration for large data transfers
    if task.input_size_mb > 10:  # > 10MB
        network_factor = min(1.0, capacity.network_bandwidth_mbps / 100.0)
        score *= 0.5 + 0.5 * network_factor

    return score
```

## Cluster Management

### Automatic Cluster Formation

The system automatically forms clusters when sufficient nodes are available:

```python
async def _evaluate_cluster_formation(self) -> None:
    """Evaluate whether to form new fog clusters"""

    # Only form clusters if we have enough nodes
    if len(self.nodes) < self.policies["cluster_size_target"]:  # Default: 5
        return

    # Find nodes not in clusters
    unclustered_nodes = set(self.nodes.keys())
    for cluster in self.clusters.values():
        unclustered_nodes -= cluster.member_nodes
        unclustered_nodes.discard(cluster.coordinator_node)

    if len(unclustered_nodes) >= self.policies["cluster_size_target"]:
        await self._form_cluster(list(unclustered_nodes))
```

### Cluster Creation

```python
async def _form_cluster(self, node_ids: list[str]) -> str:
    """Form a new fog cluster"""

    cluster_id = f"cluster_{uuid4().hex[:8]}"

    # Select coordinator (node with highest compute score)
    coordinator_node = max(
        node_ids,
        key=lambda nid: self.node_capacities.get(nid,
            ComputeCapacity(1, 0.5, 1024, 512, False, 0)).compute_score
    )

    cluster = FogCluster(
        cluster_id=cluster_id,
        coordinator_node=coordinator_node,
        member_nodes=set(node_ids)
    )

    self.clusters[cluster_id] = cluster
    self.stats["clusters_formed"] += 1

    logger.info(f"Formed cluster {cluster_id} with {len(node_ids)} nodes, coordinator: {coordinator_node}")

    return cluster_id
```

### Cluster Maintenance

```python
async def _maintain_clusters(self) -> None:
    """Maintain existing clusters and handle node changes"""

    for cluster_id, cluster in list(self.clusters.items()):
        # Check if coordinator is still available
        if cluster.coordinator_node not in self.nodes:
            # Need to elect new coordinator or dissolve cluster
            available_members = [nid for nid in cluster.member_nodes if nid in self.nodes]

            if len(available_members) >= 2:
                # Elect new coordinator (highest compute score)
                new_coordinator = max(
                    available_members,
                    key=lambda nid: self.node_capacities.get(nid,
                        ComputeCapacity(1, 0.5, 1024, 512, False, 0)).compute_score
                )
                cluster.coordinator_node = new_coordinator
                logger.info(f"Elected new coordinator {new_coordinator} for cluster {cluster_id}")
            else:
                # Dissolve cluster - insufficient nodes
                del self.clusters[cluster_id]
                logger.info(f"Dissolved cluster {cluster_id} due to insufficient nodes")

        # Remove unavailable members
        cluster.member_nodes &= set(self.nodes.keys())
```

## Task Types and Workloads

### Supported Task Types

```python
class TaskType(Enum):
    """Types of fog computing tasks"""
    INFERENCE = "inference"           # Model inference
    TRAINING = "training"             # Distributed training
    EMBEDDING = "embedding"           # Vector embedding generation
    PREPROCESSING = "preprocessing"   # Data preprocessing
    OPTIMIZATION = "optimization"     # Hyperparameter optimization
    VALIDATION = "validation"         # Model validation
```

### Task Priority Levels

```python
class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1        # Background tasks
    NORMAL = 3     # Standard priority
    HIGH = 7       # Important tasks
    CRITICAL = 10  # Emergency/real-time tasks
```

### Distributed Training Example

```python
async def submit_distributed_training_task():
    """Example: Submit distributed training task"""

    fog_coordinator = FogCoordinator()

    # Submit training task with high priority
    task_id = await fog_coordinator.submit_task(
        task_type=TaskType.TRAINING,
        priority=TaskPriority.HIGH,
        cpu_cores=2.0,           # Require 2 CPU cores
        memory_mb=2048,          # Require 2GB memory
        estimated_duration=1800, # 30 minutes
        requires_gpu=False,
        input_data=training_data_bytes
    )

    # Monitor task progress
    while True:
        status = fog_coordinator.get_task_status(task_id)
        if status["status"] in ["completed", "failed"]:
            break
        await asyncio.sleep(10)

    return status
```

### Inference Workload Example

```python
async def submit_inference_tasks():
    """Example: Submit batch inference tasks"""

    fog_coordinator = FogCoordinator()

    # Submit multiple inference tasks
    task_ids = []
    for i in range(10):
        task_id = await fog_coordinator.submit_task(
            task_type=TaskType.INFERENCE,
            priority=TaskPriority.NORMAL,
            cpu_cores=0.5,        # Light compute requirement
            memory_mb=512,        # 512MB memory
            estimated_duration=30, # 30 seconds
            input_data=inference_data[i]
        )
        task_ids.append(task_id)

    # Wait for all tasks to complete
    completed_tasks = []
    while len(completed_tasks) < len(task_ids):
        for task_id in task_ids:
            if task_id not in completed_tasks:
                status = fog_coordinator.get_task_status(task_id)
                if status["status"] == "completed":
                    completed_tasks.append(task_id)
        await asyncio.sleep(5)

    return completed_tasks
```

## Resource Management

### Resource Allocation

The system tracks and manages resources across all nodes:

```python
async def _assign_task_to_node(self, task: FogTask, node_id: str) -> None:
    """Assign a task to a specific node"""

    task.assigned_node = node_id
    task.status = "assigned"
    task.started_at = datetime.now(UTC)

    self.active_tasks[task.task_id] = task

    # Update node capacity (reserve resources)
    if node_id in self.node_capacities:
        capacity = self.node_capacities[node_id]
        capacity.cpu_utilization = min(1.0,
            capacity.cpu_utilization + (task.cpu_cores_required / capacity.cpu_cores))
        capacity.memory_used_mb += task.memory_mb_required

    # Notify node to start task
    node = self.nodes.get(node_id)
    if node:
        asyncio.create_task(node.execute_task(task))

    logger.info(f"Assigned task {task.task_id} to node {node_id}")
```

### Resource Release

```python
async def _handle_task_completion(self, task: FogTask) -> None:
    """Handle completion of a task and release resources"""

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
        execution_time_hours = (task.execution_time or 0) / 3600.0
        self.stats["total_compute_hours"] += execution_time_hours

        # Track charging compute hours for mobile optimization
        if (task.assigned_node and
            task.assigned_node in self.node_capacities and
            self.node_capacities[task.assigned_node].is_charging):
            self.stats["charging_compute_hours"] += execution_time_hours
    else:
        self.stats["tasks_failed"] += 1

    # Release node resources
    if task.assigned_node and task.assigned_node in self.node_capacities:
        capacity = self.node_capacities[task.assigned_node]
        capacity.cpu_utilization = max(0.0,
            capacity.cpu_utilization - (task.cpu_cores_required / capacity.cpu_cores))
        capacity.memory_used_mb = max(0, capacity.memory_used_mb - task.memory_mb_required)
```

### Battery-Aware Statistics

The system tracks mobile-specific metrics:

```python
async def _update_statistics(self) -> None:
    """Update fog computing statistics with mobile awareness"""

    # Count battery saves (tasks running on charging nodes)
    battery_saves = 0
    for task in self.active_tasks.values():
        if task.assigned_node and task.assigned_node in self.node_capacities:
            capacity = self.node_capacities[task.assigned_node]
            if (capacity.battery_powered and capacity.is_charging and
                task.estimated_duration_seconds > 180):  # > 3 minutes
                battery_saves += 1

    self.stats["battery_saves"] = battery_saves
```

## Performance Optimization

### Scheduling Policies

Configure scheduling behavior through policies:

```python
self.policies = {
    "prefer_charging_nodes": True,           # Prefer charging devices
    "battery_minimum_percent": 30,           # Minimum battery for non-charging devices
    "thermal_throttle_threshold": 60.0,      # CPU temperature threshold (Celsius)
    "max_concurrent_tasks_per_node": 2,      # Maximum tasks per node
    "task_timeout_multiplier": 2.0,          # Task timeout as multiple of estimated duration
    "cluster_size_target": 5,                # Target cluster size
}
```

### Performance Monitoring

```python
def get_system_status(self) -> dict[str, Any]:
    """Get comprehensive fog computing system status"""

    return {
        "coordinator_id": self.coordinator_id,
        "nodes": {
            "total": len(self.nodes),
            "by_state": {
                "active": len([n for n in self.nodes.values() if n.state == FogNodeState.ACTIVE]),
                "idle": len([n for n in self.nodes.values() if n.state == FogNodeState.IDLE]),
                "charging": len([nid for nid, cap in self.node_capacities.items()
                               if cap.battery_powered and cap.is_charging]),
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
```

## Production Deployment

### Basic Setup

```python
from packages.edge.fog_compute.fog_coordinator import FogCoordinator, ComputeCapacity

async def setup_fog_computing():
    """Set up fog computing coordinator and register nodes"""

    # Initialize coordinator
    coordinator = FogCoordinator("production_coordinator")

    # Register desktop node
    desktop_capacity = ComputeCapacity(
        cpu_cores=8,
        cpu_utilization=0.3,        # 30% current usage
        memory_mb=16384,            # 16GB RAM
        memory_used_mb=4096,        # 4GB used
        gpu_available=True,
        gpu_memory_mb=8192,         # 8GB GPU
        battery_powered=False,      # Desktop - unlimited power
        thermal_state="normal",
        network_bandwidth_mbps=100.0
    )

    await coordinator.register_node("desktop_001", desktop_capacity)

    # Register mobile node (only when charging)
    mobile_capacity = ComputeCapacity(
        cpu_cores=4,
        cpu_utilization=0.5,        # 50% current usage
        memory_mb=6144,             # 6GB RAM
        memory_used_mb=2048,        # 2GB used
        gpu_available=False,
        gpu_memory_mb=0,
        battery_powered=True,       # Mobile device
        battery_percent=85,         # 85% battery
        is_charging=True,           # Currently charging
        thermal_state="normal",
        network_bandwidth_mbps=50.0
    )

    await coordinator.register_node("mobile_001", mobile_capacity)

    return coordinator
```

### Production Configuration

```python
# Production fog coordinator setup
production_policies = {
    "prefer_charging_nodes": True,
    "battery_minimum_percent": 40,          # Conservative battery threshold
    "thermal_throttle_threshold": 55.0,     # Lower thermal threshold
    "max_concurrent_tasks_per_node": 1,     # Conservative task limit
    "task_timeout_multiplier": 3.0,         # Longer timeout for reliability
    "cluster_size_target": 3,               # Smaller clusters for stability
}

coordinator = FogCoordinator()
coordinator.policies.update(production_policies)
```

### Integration with Edge Manager

```python
async def integrated_edge_fog_deployment():
    """Integrate edge manager with fog computing"""

    edge_manager = EdgeManager()
    fog_coordinator = FogCoordinator()

    # Register device with edge manager
    device = await edge_manager.register_device(
        device_id="fog_node_001",
        device_name="Production Node",
        auto_detect=True
    )

    # Register same device as fog node
    capacity = ComputeCapacity(
        cpu_cores=device.capabilities.cpu_cores,
        cpu_utilization=0.2,
        memory_mb=device.capabilities.ram_total_mb,
        memory_used_mb=device.capabilities.ram_total_mb - device.capabilities.ram_available_mb,
        gpu_available=device.capabilities.gpu_available,
        gpu_memory_mb=device.capabilities.gpu_memory_mb,
        battery_powered=device.capabilities.battery_powered,
        battery_percent=device.capabilities.battery_percent,
        is_charging=device.capabilities.battery_charging,
        thermal_state=device.capabilities.thermal_state
    )

    await fog_coordinator.register_node(device.device_id, capacity)

    return edge_manager, fog_coordinator
```

This fog computing framework enables efficient distributed computing across heterogeneous edge devices with mobile-first optimization and intelligent resource management.
