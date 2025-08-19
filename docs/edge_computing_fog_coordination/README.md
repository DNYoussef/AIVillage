# Edge Computing & Fog Coordination - System Overview

## Introduction

The AIVillage Edge Computing & Fog Coordination system provides a comprehensive framework for distributed edge computing with mobile-first optimization. This system orchestrates computing resources across heterogeneous edge devices, from smartphones and tablets to desktop machines and IoT devices, creating a unified fog computing infrastructure.

## Architecture Overview

### System Components

The edge computing system consists of four primary components working in harmony:

1. **Edge Manager** - Central orchestration of edge devices and workload deployment
2. **Fog Coordinator** - Distributed computing task scheduling and cluster management
3. **Mobile Resource Manager** - Battery/thermal-aware optimization for mobile devices
4. **Digital Twin Concierge** - On-device personal AI with privacy-preserving data collection

### Core Capabilities

#### 🔄 **Unified Edge Orchestration**
- **Device Registration**: Auto-detection of device capabilities and classification
- **Resource Management**: Dynamic resource allocation with mobile-aware policies
- **Workload Deployment**: Optimized AI model deployment across edge devices
- **Real-time Monitoring**: Continuous device state tracking and policy adaptation

#### 🌐 **Distributed Fog Computing**
- **Cluster Formation**: Automatic fog cluster creation with 5+ node targets
- **Task Scheduling**: Priority-based task distribution with mobile optimization
- **Battery Awareness**: Charging device preference for compute-intensive tasks
- **Thermal Management**: Progressive throttling based on device temperature

#### 📱 **Mobile-First Optimization**
- **Power Modes**: 4-tier power management (Performance → Balanced → Power Save → Critical)
- **Transport Selection**: BitChat-preferred routing under battery constraints
- **Chunking Optimization**: Dynamic tensor sizing for 2-4GB mobile devices
- **Network Cost Awareness**: Cellular vs WiFi transport preference

#### 🤖 **Privacy-Preserving Digital Twins**
- **On-Device Learning**: Local AI training with industry-standard data collection
- **Surprise-Based Learning**: Model improvement based on prediction accuracy
- **Automatic Data Deletion**: Configurable retention periods (24-hour default)
- **Cross-Platform Support**: iOS/Android native integration

## Key Features

### Device Capability Auto-Detection

The system automatically detects and classifies edge devices:

```python
# Device Types Supported
SMARTPHONE = "smartphone"      # ≤ 3GB RAM, battery-powered
TABLET = "tablet"             # 3-6GB RAM, battery-powered
LAPTOP = "laptop"             # > 6GB RAM, battery-powered
DESKTOP = "desktop"           # > 6GB RAM, not battery-powered
RASPBERRY_PI = "raspberry_pi" # ≤ 2 cores, ≤ 2GB RAM
```

### Battery/Thermal Optimization

Mobile devices receive specialized optimization:

- **Critical Battery (≤10%)**: BitChat-only, 20% CPU limit, 256MB memory
- **Low Battery (≤20%)**: BitChat-preferred, 35% CPU limit, reduced chunking
- **Hot Thermal (≥55°C)**: 30% CPU throttling, smaller chunks
- **Critical Thermal (≥65°C)**: 15% CPU emergency throttling

### Fog Computing Coordination

Distributed task scheduling with mobile awareness:

- **Node Scoring**: Battery charging = 1.5x bonus, thermal penalties
- **Task Prioritization**: Critical (10) → High (7) → Normal (3) → Low (1)
- **Resource Allocation**: Dynamic CPU/memory limits based on device state
- **Cluster Management**: Automatic coordinator election and failover

### Digital Twin Data Collection

Following industry patterns (Meta/Google/Apple) with privacy guarantees:

**Data Sources**:
- Conversations and messaging patterns
- Location and movement data
- Purchase history and shopping preferences
- App usage and digital behavior
- Calendar and scheduling patterns

**Privacy Protection**:
- All data remains on-device
- Automatic deletion after training cycles
- Differential privacy noise injection
- User consent required for all sources

## Performance Characteristics

### Edge Manager Metrics
- **Device Registration**: Sub-second capability detection
- **Deployment Optimization**: Battery/thermal-aware resource allocation
- **Monitoring Frequency**: 30-second device state updates
- **Policy Adaptation**: Real-time response to capability changes

### Fog Coordinator Throughput
- **Task Scheduling**: 5-second scheduling intervals
- **Node Capacity**: 50+ concurrent devices supported
- **Cluster Formation**: Automatic 5-node target clusters
- **Resource Efficiency**: 90%+ capacity utilization under optimal conditions

### Mobile Resource Manager
- **Power Mode Switching**: <100ms evaluation time
- **Transport Decisions**: Real-time BitChat/BetaNet routing
- **Chunking Adaptation**: Dynamic 64-2048 byte range
- **Battery Impact**: 2-15% reduction in power consumption

### Digital Twin Learning
- **Data Collection**: Configurable 1-24 hour retention
- **Learning Cycles**: Battery-aware training frequency
- **Surprise Evaluation**: Continuous prediction accuracy tracking
- **Model Size**: 1-10MB on-device models

## Integration Architecture

### System Interconnections

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Edge Manager  │────│ Fog Coordinator │────│ Digital Twins   │
│                 │    │                 │    │                 │
│ • Device Reg.   │    │ • Task Sched.   │    │ • Local Learning│
│ • Workload      │    │ • Cluster Mgmt  │    │ • Data Privacy  │
│ • Monitoring    │    │ • Resource Opt. │    │ • Mini-RAG      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Mobile Resource │
                    │    Manager      │
                    │                 │
                    │ • Battery Opt.  │
                    │ • Thermal Mgmt  │
                    │ • Transport Sel.│
                    │ • Chunking      │
                    └─────────────────┘
```

### P2P Network Integration
- **BitChat Transport**: Bluetooth mesh for offline-first mobile scenarios
- **BetaNet Transport**: HTX v1.1 for high-throughput internet communication
- **Transport Selection**: Intelligent routing based on device constraints

### RAG System Integration
- **Mini-RAG**: Personal knowledge base for each digital twin
- **Global Elevation**: Privacy-preserving knowledge contribution to distributed RAG
- **Context Enhancement**: Personal patterns boost prediction confidence

## Security Model

### Privacy Guarantees
- **Data Locality**: All personal data remains on originating device
- **Encryption**: AES-GCM for all data at rest and in transit
- **Access Control**: User consent required for each data source
- **Anonymization**: Differential privacy for any data sharing

### Resource Protection
- **Resource Isolation**: Containerized workload execution where supported
- **Thermal Protection**: Hardware-level temperature monitoring
- **Battery Protection**: Critical level enforcement (10% minimum)
- **Memory Protection**: Process memory limits with OOM protection

### Network Security
- **Transport Encryption**: All communications use Noise XK protocol
- **Device Authentication**: Ed25519 digital signatures for device identity
- **Access Tickets**: Time-limited access tokens for fog computing
- **Replay Protection**: Cryptographic nonces prevent replay attacks

## Deployment Scenarios

### Mobile Edge Computing
**Use Case**: Smartphone/tablet edge inference
- BitChat-preferred networking for battery conservation
- 64-512 byte chunking for memory constraints
- Thermal throttling at 45°C+ temperatures
- Background processing during charging periods

### Fog Computing Clusters
**Use Case**: Distributed training across multiple devices
- Automatic cluster formation with 5+ devices
- Charging device preference for compute tasks
- Load balancing based on thermal/battery state
- Fault tolerance with coordinator election

### Hybrid Edge-Cloud
**Use Case**: Edge-cloud workload distribution
- Local inference on mobile devices
- Heavy computation offloaded to fog clusters
- Intelligent routing based on latency requirements
- Cost-aware transport selection (cellular vs WiFi)

### Privacy-Preserving Personal AI
**Use Case**: On-device digital assistant learning
- Local data collection following industry standards
- Surprise-based learning for continuous improvement
- Privacy-preserving knowledge elevation to global systems
- Cross-platform mobile deployment (iOS/Android)

## Getting Started

### Basic Usage

```python
from packages.edge.core.edge_manager import EdgeManager
from packages.edge.fog_compute.fog_coordinator import FogCoordinator
from packages.edge.mobile.resource_manager import MobileResourceManager

# Initialize edge management system
edge_manager = EdgeManager()

# Register local device with auto-detection
device = await edge_manager.register_device(
    device_id="my_device",
    device_name="Mobile Device",
    auto_detect=True  # Automatically detect capabilities
)

# Deploy AI workload with optimization
deployment_id = await edge_manager.deploy_workload(
    device_id=device.device_id,
    model_id="tutor_agent_v1",
    deployment_type="inference"
)

# Monitor deployment status
status = edge_manager.get_device_status(device.device_id)
print(f"Device status: {status}")
```

### Mobile Optimization

```python
from packages.edge.mobile.resource_manager import MobileResourceManager

# Initialize mobile-aware resource management
resource_manager = MobileResourceManager()

# Get optimization for current device state
optimization = await resource_manager.optimize_for_device()

print(f"Power mode: {optimization.power_mode.value}")
print(f"Transport preference: {optimization.transport_preference.value}")
print(f"Chunk size: {optimization.chunking_config.effective_chunk_size()}")
print(f"CPU limit: {optimization.cpu_limit_percent}%")
```

### Digital Twin Setup

```python
from packages.edge.mobile.digital_twin_concierge import DigitalTwinConcierge, UserPreferences, DataSource

# Configure privacy preferences
preferences = UserPreferences(
    enabled_sources={DataSource.CONVERSATIONS, DataSource.APP_USAGE},
    max_data_retention_hours=24,
    privacy_mode="balanced"
)

# Initialize digital twin
twin = DigitalTwinConcierge(
    data_dir=Path("./twin_data"),
    preferences=preferences
)

# Run learning cycle
device_profile = await resource_manager.create_device_profile_from_env()
learning_cycle = await twin.run_learning_cycle(device_profile)

print(f"Learning complete: {learning_cycle.improvement_score:.3f} accuracy")
```

## Advanced Features

### Custom Resource Policies

```python
from packages.edge.mobile.resource_manager import ResourcePolicy

# Define custom mobile resource policy
custom_policy = ResourcePolicy(
    battery_critical=5,      # More aggressive battery saving
    thermal_hot=50.0,        # Earlier thermal throttling
    memory_low_gb=1.5        # Tighter memory constraints
)

resource_manager = MobileResourceManager(policy=custom_policy)
```

### Fog Computing Task Submission

```python
from packages.edge.fog_compute.fog_coordinator import FogCoordinator, TaskType, TaskPriority

# Initialize fog coordinator
fog_coordinator = FogCoordinator()

# Submit distributed training task
task_id = await fog_coordinator.submit_task(
    task_type=TaskType.TRAINING,
    priority=TaskPriority.HIGH,
    cpu_cores=2.0,
    memory_mb=2048,
    estimated_duration=300,  # 5 minutes
    requires_gpu=False
)

# Monitor task progress
task_status = fog_coordinator.get_task_status(task_id)
print(f"Task status: {task_status}")
```

## Next Steps

1. **[System Architecture](system_architecture.md)** - Detailed technical architecture and component interactions
2. **[Mobile Optimization Guide](mobile_optimization_guide.md)** - Comprehensive mobile device optimization strategies
3. **[Fog Computing Guide](fog_computing_guide.md)** - Distributed computing setup and management
4. **[Digital Twin Implementation](digital_twin_implementation.md)** - Privacy-preserving personal AI development
5. **[Deployment Guide](deployment_guide.md)** - Production deployment and scaling strategies
6. **[Performance Benchmarks](performance_benchmarks.md)** - System performance metrics and optimization results

The AIVillage Edge Computing & Fog Coordination system provides a production-ready foundation for distributed edge AI deployment with mobile-first optimization and privacy-preserving personal assistance.
