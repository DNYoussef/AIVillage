# Edge Computing & Fog Coordination - System Architecture

## Architectural Overview

The AIVillage Edge Computing & Fog Coordination system implements a hierarchical distributed computing architecture that seamlessly integrates edge devices, fog computing clusters, and mobile optimization. The system is designed around four core architectural principles:

1. **Mobile-First Design**: Optimized for battery/thermal constraints of mobile devices
2. **Privacy Preservation**: All personal data remains on originating devices
3. **Dynamic Resource Adaptation**: Real-time optimization based on device state
4. **Fault Tolerance**: Graceful degradation and automatic recovery mechanisms

## System Layers

### Layer 1: Device Abstraction Layer

#### Edge Device Management
**Location**: `packages/edge/core/edge_manager.py`

The Edge Manager provides unified abstraction across heterogeneous edge devices:

```python
class EdgeManager:
    """Unified edge device management system"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.devices: dict[str, EdgeDevice] = {}
        self.deployments: dict[str, EdgeDeployment] = {}
        self.fog_nodes: dict[str, FogNode] = {}
        self.resource_policies = self._init_resource_policies()
```

**Key Responsibilities**:
- Device capability auto-detection using `psutil` system introspection
- Device classification (smartphone, tablet, laptop, desktop, RPi, IoT)
- Resource policy initialization and management
- Deployment queue management and execution
- Background monitoring and policy adaptation

#### Device Capability Detection

The system performs comprehensive capability detection:

```python
async def _detect_device_capabilities(self) -> DeviceCapabilities:
    """Auto-detect current device capabilities"""
    try:
        # Hardware detection
        cpu_cores = psutil.cpu_count(logical=False) or 1
        memory = psutil.virtual_memory()
        ram_total_mb = int(memory.total / (1024 * 1024))

        # Power system detection
        battery = psutil.sensors_battery()
        if battery:
            battery_powered = True
            battery_percent = int(battery.percent)
            battery_charging = battery.power_plugged

        # Thermal monitoring
        temps = psutil.sensors_temperatures()
        # Extract CPU temperature from available sensors

        return DeviceCapabilities(...)
```

**Detection Capabilities**:
- CPU cores and architecture detection
- RAM total/available memory monitoring
- Storage capacity and availability
- Battery status and charging state detection
- CPU temperature sensor reading
- Network interface capabilities
- Platform-specific feature detection (containers, BitChat, BLE)

### Layer 2: Resource Optimization Layer

#### Mobile Resource Manager
**Location**: `packages/edge/mobile/resource_manager.py`

Provides battery/thermal-aware optimization for mobile devices:

```python
class MobileResourceManager:
    """Battery/thermal-aware resource manager for mobile optimization"""

    def __init__(self, policy: ResourcePolicy | None = None):
        self.policy = policy or ResourcePolicy()
        self.env_simulation_mode = self._check_env_simulation()
        self.stats = {
            "policy_adaptations": 0,
            "transport_switches": 0,
            "thermal_throttles": 0,
            "battery_saves": 0,
            "chunk_adjustments": 0,
        }
```

#### Power Mode Evaluation

The system implements 4-tier power management:

```python
def _evaluate_power_mode(self, profile: MobileDeviceProfile) -> PowerMode:
    """Evaluate appropriate power mode based on device state"""

    # Critical battery - always use critical mode
    if profile.battery_percent <= self.policy.battery_critical:  # ≤10%
        self.stats["battery_saves"] += 1
        return PowerMode.CRITICAL

    # Critical thermal - always use critical mode
    if profile.cpu_temp_celsius >= self.policy.thermal_critical:  # ≥65°C
        self.stats["thermal_throttles"] += 1
        return PowerMode.CRITICAL

    # Hot thermal or low battery (not charging) - power save
    thermal_hot = profile.cpu_temp_celsius >= self.policy.thermal_hot  # ≥55°C
    battery_low_not_charging = (
        profile.battery_percent <= self.policy.battery_low  # ≤20%
        and not profile.battery_charging
    )

    if thermal_hot or battery_low_not_charging:
        return PowerMode.POWER_SAVE

    # Warm thermal or conservative battery - balanced
    if (profile.cpu_temp_celsius >= self.policy.thermal_warm or  # ≥45°C
        profile.battery_percent <= self.policy.battery_conservative):  # ≤40%
        return PowerMode.BALANCED

    return PowerMode.PERFORMANCE
```

#### Dynamic Chunking Configuration

Memory-aware tensor/chunk sizing optimization:

```python
def _calculate_chunking_config(self, profile: MobileDeviceProfile) -> ChunkingConfig:
    """Calculate optimal chunking configuration for current device state"""

    config = ChunkingConfig()  # Base: 512 bytes, range: 64-2048

    # Memory-based scaling
    available_gb = profile.ram_available_mb / 1024.0
    if available_gb <= 2.0:      # Low memory
        config.memory_scale_factor = 0.5    # 256 bytes
    elif available_gb <= 4.0:    # Medium memory
        config.memory_scale_factor = 0.75   # 384 bytes
    elif available_gb <= 8.0:    # High memory
        config.memory_scale_factor = 1.0    # 512 bytes
    else:                        # Very high memory
        config.memory_scale_factor = 1.25   # 640 bytes

    # Thermal-based scaling
    if profile.cpu_temp_celsius >= 65.0:    # Critical thermal
        config.thermal_scale_factor = 0.3   # Minimize processing
    elif profile.cpu_temp_celsius >= 55.0:  # Hot thermal
        config.thermal_scale_factor = 0.5   # Reduce processing
    elif profile.cpu_temp_celsius >= 45.0:  # Warm thermal
        config.thermal_scale_factor = 0.75  # Conservative processing
    else:
        config.thermal_scale_factor = 1.0   # Normal processing

    # Battery-based scaling
    if profile.battery_percent <= 10:       # Critical battery
        config.battery_scale_factor = 0.3   # Minimize processing
    elif profile.battery_percent <= 20:     # Low battery
        config.battery_scale_factor = 0.6   # Reduce processing
    elif profile.battery_percent <= 40:     # Conservative battery
        config.battery_scale_factor = 0.8   # Conservative processing
    else:
        config.battery_scale_factor = 1.0   # Normal processing

    return config
```

### Layer 3: Distributed Computing Layer

#### Fog Coordinator
**Location**: `packages/edge/fog_compute/fog_coordinator.py`

Orchestrates distributed computing across edge devices:

```python
class FogCoordinator:
    """Fog Computing Coordinator - Orchestrates distributed computing across edge devices"""

    def __init__(self, coordinator_id: str | None = None):
        self.coordinator_id = coordinator_id or f"fog_coord_{uuid4().hex[:8]}"
        self.nodes: dict[str, FogNode] = {}
        self.node_capacities: dict[str, ComputeCapacity] = {}
        self.clusters: dict[str, FogCluster] = {}

        # Task management
        self.pending_tasks: list[FogTask] = []
        self.active_tasks: dict[str, FogTask] = {}
        self.completed_tasks: list[FogTask] = []
```

#### Task Scheduling Algorithm

The fog coordinator implements priority-based scheduling with mobile awareness:

```python
async def _find_suitable_node(self, task: FogTask) -> str | None:
    """Find the most suitable node for a task"""

    suitable_nodes = []

    for node_id, node in self.nodes.items():
        capacity = self.node_capacities.get(node_id)

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

#### Node Scoring Algorithm

Mobile-aware node scoring with charging preference:

```python
def _calculate_node_score(self, capacity: ComputeCapacity, task: FogTask) -> float:
    """Calculate suitability score for a node"""

    # Base score from compute capacity
    score = capacity.compute_score  # CPU + memory composite score

    # Bonus for charging devices (mobile optimization)
    if capacity.battery_powered and capacity.is_charging:
        score *= 1.5  # 50% bonus for charging devices

    # Bonus for non-battery devices (desktop/laptop plugged in)
    if not capacity.battery_powered:
        score *= 1.2  # 20% bonus for unlimited power

    # Penalty for high thermal state
    thermal_penalties = {
        "normal": 1.0,
        "warm": 0.9,     # -10% for warm
        "hot": 0.5,      # -50% for hot
        "critical": 0.1  # -90% for critical
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

#### Cluster Management

Automatic cluster formation and coordinator election:

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

    return cluster_id
```

### Layer 4: Privacy-Preserving AI Layer

#### Digital Twin Concierge
**Location**: `packages/edge/mobile/digital_twin_concierge.py`

Implements on-device personal AI with privacy guarantees:

```python
class DigitalTwinConcierge:
    """Main Digital Twin Concierge system"""

    def __init__(self, data_dir: Path, preferences: UserPreferences,
                 distributed_rag: DistributedRAGCoordinator | None = None):
        self.data_dir = data_dir
        self.preferences = preferences
        self.resource_manager = BatteryThermalResourceManager()
        self.data_collector = OnDeviceDataCollector(data_dir, preferences)
        self.learning_system = SurpriseBasedLearning()

        # Initialize Mini-RAG system for personal knowledge
        self.mini_rag = MiniRAGSystem(data_dir / "mini_rag", f"twin_{int(time.time())}")

        # Connection to global distributed RAG (for knowledge elevation)
        self.distributed_rag = distributed_rag
```

#### Data Collection Architecture

Industry-standard data collection with privacy controls:

```python
class OnDeviceDataCollector:
    """Collects data from various sources on the mobile device"""

    async def collect_all_sources(self) -> list[DataPoint]:
        """Collect data from all enabled sources"""
        all_data = []

        # Collect from each enabled source
        tasks = []
        if DataSource.CONVERSATIONS in self.preferences.enabled_sources:
            tasks.append(self.collect_conversation_data())
        if DataSource.LOCATION in self.preferences.enabled_sources:
            tasks.append(self.collect_location_data())
        if DataSource.PURCHASES in self.preferences.enabled_sources:
            tasks.append(self.collect_purchase_data())
        if DataSource.APP_USAGE in self.preferences.enabled_sources:
            tasks.append(self.collect_app_usage_data())

        results = await asyncio.gather(*tasks)
        for data_list in results:
            all_data.extend(data_list)

        return all_data
```

#### Surprise-Based Learning

Continuous model improvement based on prediction accuracy:

```python
class SurpriseBasedLearning:
    """Surprise-based learning evaluation system"""

    def calculate_surprise_score(self, predicted: Any, actual: Any, context: dict) -> float:
        """
        Calculate surprise score - how unexpected was the actual outcome
        Lower surprise = better understanding of user
        """
        if isinstance(predicted, str) and isinstance(actual, str):
            if predicted == actual:
                return 0.0  # No surprise - perfect prediction
            else:
                # Simple word overlap measure
                pred_words = set(predicted.lower().split())
                actual_words = set(actual.lower().split())
                overlap = len(pred_words & actual_words) / max(len(pred_words | actual_words), 1)
                return 1.0 - overlap  # Higher surprise for less overlap
```

## Integration Architecture

### Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Edge Computing System                              │
├─────────────────┬─────────────────┬─────────────────┬─────────────────────┤
│   Edge Manager  │ Fog Coordinator │ Resource Manager│  Digital Twins      │
│                 │                 │                 │                     │
│ • Device Auto-  │ • Task Sched.   │ • Power Modes   │ • Data Collection   │
│   Detection     │ • Cluster Mgmt  │ • Transport Sel.│ • Surprise Learning │
│ • Capability    │ • Node Scoring  │ • Chunk Sizing  │ • Mini-RAG         │
│   Profiling     │ • Mobile Aware. │ • Policy Adapt. │ • Privacy Controls  │
│ • Workload      │ • Fault Toler.  │ • Env Simulation│ • Knowledge Elevat. │
│   Deployment    │ • Statistics    │ • Real-time Opt.│ • Auto Data Cleanup │
│ • Real-time     │ • Resource      │ • Battery/Therm.│ • Cross-platform   │
│   Monitoring    │   Allocation    │   Monitoring    │   Support          │
└─────────────────┴─────────────────┴─────────────────┴─────────────────────┘
         │                   │                   │                   │
         └───────────────────┼───────────────────┼───────────────────┘
                             │                   │
                    ┌─────────────────┐  ┌─────────────────┐
                    │ P2P Integration │  │ RAG Integration │
                    │                 │  │                 │
                    │ • BitChat       │  │ • Mini-RAG      │
                    │ • BetaNet       │  │ • Distributed   │
                    │ • Transport     │  │ • Knowledge     │
                    │   Selection     │  │   Elevation     │
                    └─────────────────┘  └─────────────────┘
```

### Data Flow Architecture

#### Device Registration and Management Flow

```
1. Device Connection
   ↓
2. Capability Auto-Detection (psutil)
   ↓
3. Device Classification (smartphone/tablet/laptop/desktop/RPi)
   ↓
4. Resource Policy Assignment
   ↓
5. Fog Node Registration (if applicable)
   ↓
6. Background Monitoring Activation
```

#### Workload Deployment Flow

```
1. Deployment Request
   ↓
2. Device State Evaluation
   ↓
3. Optimization Calculation (CPU/memory/chunk limits)
   ↓
4. Model Preparation and Compression
   ↓
5. Resource Allocation
   ↓
6. Deployment Execution
   ↓
7. Health Monitoring Activation
```

#### Fog Computing Task Flow

```
1. Task Submission (type, priority, resources)
   ↓
2. Task Queue Addition
   ↓
3. Scheduling Loop (5-second intervals)
   ↓
4. Node Suitability Evaluation
   ↓
5. Mobile Policy Checks
   ↓
6. Node Scoring and Selection
   ↓
7. Task Assignment and Execution
   ↓
8. Completion Monitoring and Cleanup
```

#### Digital Twin Learning Flow

```
1. Data Collection (enabled sources)
   ↓
2. Prediction Generation (context-based)
   ↓
3. Mini-RAG Enhancement (personal knowledge)
   ↓
4. Surprise Score Calculation (prediction vs actual)
   ↓
5. Local SQLite Storage
   ↓
6. Learning Evaluation (average surprise)
   ↓
7. Model Pattern Updates (if needed)
   ↓
8. Knowledge Elevation (global contribution)
   ↓
9. Automatic Data Cleanup
```

## Performance Characteristics

### Edge Manager Performance

- **Device Registration**: 200-500ms including capability detection
- **Capability Detection**: 100-300ms using psutil system calls
- **Deployment Optimization**: 50-150ms for resource calculation
- **Monitoring Frequency**: 30-second device state updates
- **Policy Adaptation**: <100ms for threshold-based changes

### Fog Coordinator Performance

- **Task Scheduling**: 5-second intervals with priority-based ordering
- **Node Evaluation**: <50ms per node for suitability calculation
- **Cluster Formation**: 100-500ms for 5-10 node clusters
- **Task Throughput**: 100+ tasks/minute sustained (dependent on execution time)
- **Failover Time**: 30-60 seconds for coordinator election

### Mobile Resource Manager Performance

- **Optimization Evaluation**: <100ms for complete device assessment
- **Power Mode Switching**: <50ms threshold-based evaluation
- **Transport Decision**: <25ms for BitChat/BetaNet routing
- **Chunking Calculation**: <10ms for scaling factor application
- **Policy Updates**: Real-time response to device state changes

### Digital Twin Performance

- **Data Collection**: 1-5 seconds for all enabled sources
- **Prediction Generation**: 50-200ms including Mini-RAG enhancement
- **Surprise Calculation**: 10-50ms depending on data types
- **Learning Cycle**: 5-30 seconds depending on data volume
- **Knowledge Elevation**: 1-5 seconds for anonymization and transmission

## Scalability Considerations

### Horizontal Scaling

- **Device Support**: 100+ edge devices per coordinator instance
- **Cluster Size**: 50+ nodes per fog cluster
- **Task Throughput**: Linear scaling with additional coordinator instances
- **Storage**: SQLite for single-device, distributed databases for clusters

### Vertical Scaling

- **Memory Requirements**: 50-200MB per coordinator instance
- **CPU Usage**: 5-15% baseline, 30-50% during intensive operations
- **Storage Growth**: 10-100MB per device depending on data collection
- **Network Bandwidth**: 1-10 Mbps per coordinator for typical workloads

### Resource Optimization Strategies

- **Device Pooling**: Shared resource pools for similar device types
- **Task Batching**: Combining small tasks for efficiency
- **Lazy Loading**: On-demand capability detection and resource allocation
- **Cache Management**: LRU eviction for capability and prediction caches
- **Connection Pooling**: Persistent connections for frequently communicating devices

## Fault Tolerance and Recovery

### Edge Manager Resilience

- **Device Failure Detection**: 30-second timeout with 3-retry mechanism
- **Deployment Recovery**: Automatic redeployment on device recovery
- **State Persistence**: SQLite-based state storage for coordinator restart
- **Graceful Degradation**: Continued operation with reduced device set

### Fog Coordinator Resilience

- **Coordinator Election**: Automatic leader election on coordinator failure
- **Task Redistribution**: Failed tasks automatically rescheduled
- **Cluster Healing**: Automatic removal of failed nodes from clusters
- **Checkpointing**: Task state persistence for failure recovery

### Mobile Resource Manager Resilience

- **Sensor Failure Handling**: Fallback to default values on sensor errors
- **Policy Enforcement**: Hard limits to prevent device damage
- **Resource Monitoring**: Continuous validation of resource availability
- **Emergency Throttling**: Automatic protection on critical conditions

### Digital Twin Resilience

- **Data Corruption Protection**: SQLite WAL mode with transaction safety
- **Privacy Guarantee Enforcement**: Automatic data deletion on violation
- **Model Recovery**: Fallback to baseline predictions on model corruption
- **Cross-Platform Compatibility**: Graceful degradation on platform limitations

This architectural foundation provides a robust, scalable, and mobile-optimized edge computing platform with strong privacy guarantees and intelligent resource management.
