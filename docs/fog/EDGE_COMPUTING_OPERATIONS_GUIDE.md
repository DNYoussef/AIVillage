# Edge Computing Operations Guide

## STREAM D: Mobile & Edge Computing Implementation

This comprehensive guide covers the complete edge device deployment procedures and fog computing capabilities implemented in the AIVillage distributed computing architecture.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Edge Device Deployment](#edge-device-deployment)
3. [Fog Computing Orchestration](#fog-computing-orchestration)
4. [Battery/Thermal-Aware Resource Management](#batterythermal-aware-resource-management)
5. [Cross-Device Coordination Protocols](#cross-device-coordination-protocols)
6. [Mobile Optimization & Adaptive QoS](#mobile-optimization--adaptive-qos)
7. [Monitoring & Diagnostics](#monitoring--diagnostics)
8. [Operational Procedures](#operational-procedures)
9. [Troubleshooting](#troubleshooting)

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Edge Computing Architecture               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────── │
│  │  Edge Deployer  │  │ Fog Coordinator │  │ Harvest Manager│
│  │                 │  │                 │  │                │
│  │ • Device Reg    │  │ • Task Sched    │  │ • Idle Compute │
│  │ • Deployment    │  │ • Load Balance  │  │ • Token Rewards│
│  │ • Monitoring    │  │ • Clustering    │  │ • P2P Coord    │
│  └─────────────────┘  └─────────────────┘  └─────────────── │
├─────────────────────────────────────────────────────────────┤
│                     Mobile Devices Layer                    │
├─────────────────────────────────────────────────────────────┤
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────  │
│ │ Smartphones  │ │   Tablets    │ │   Laptops    │ │  IoT   │
│ │              │ │              │ │              │ │ Device │
│ │ • ARM64      │ │ • ARM64/x86  │ │ • x86_64     │ │ • Mixed│
│ │ • 2-8 cores  │ │ • 4-8 cores  │ │ • 4-16 cores │ │ • Var  │
│ │ • 2-8GB RAM  │ │ • 4-16GB RAM │ │ • 8-32GB RAM │ │ • Var  │
│ │ • Battery    │ │ • Battery    │ │ • AC/Battery │ │ • Var  │
│ └──────────────┘ └──────────────┘ └──────────────┘ └────── │
└─────────────────────────────────────────────────────────────┘
```

### Key Features Implemented

1. **Edge Device Deployment Automation**
   - Automated device registration and capability discovery
   - Battery/thermal-aware deployment validation
   - Blue-green, canary, and rolling deployment strategies
   - Real-time deployment health monitoring

2. **Fog Computing Orchestration**
   - Idle resource harvesting from charging devices
   - Distributed task scheduling and load balancing
   - Mobile-optimized policies for battery preservation
   - Cross-device coordination and clustering

3. **Mobile Optimization**
   - Adaptive QoS based on network conditions
   - Battery-aware transport selection (BitChat-first)
   - Thermal throttling with progressive limits
   - Dynamic tensor/chunk size tuning

4. **Cross-Device Coordination**
   - Device discovery and P2P connection management
   - Automatic cluster formation and maintenance
   - Task redistribution on device failures
   - Consensus mechanisms for distributed decisions

## Edge Device Deployment

### Quick Start

```python
from infrastructure.fog.edge.deployment.edge_deployer import (
    EdgeDeployer, DeviceCapabilities, DeploymentConfig, DeviceType, NetworkQuality
)

# Initialize edge deployer
deployer = EdgeDeployer(
    coordinator_id="main_deployer",
    enable_fog_computing=True,
    enable_cross_device_coordination=True
)

# Register a mobile device
device_caps = DeviceCapabilities(
    device_id="phone_001",
    device_type=DeviceType.SMARTPHONE,
    device_name="User's iPhone",
    cpu_cores=6,
    cpu_freq_ghz=2.4,
    ram_total_mb=4096,
    ram_available_mb=2048,
    battery_percent=80,
    is_charging=True,
    network_quality=NetworkQuality.GOOD
)

await deployer.register_device("phone_001", device_caps)

# Create deployment configuration
config = DeploymentConfig(
    deployment_id="ml_inference_v1",
    model_id="lightweight_llm_7b",
    deployment_type="inference",
    target_devices=["phone_001"],
    battery_aware=True,
    thermal_aware=True,
    rollout_strategy="rolling"
)

# Deploy to devices
deployment_ids = await deployer.deploy(config)
```

### Device Registration Process

1. **Capability Discovery**
   ```python
   # Automatic capability detection
   capabilities = await auto_detect_device_capabilities(device_id)

   # Manual capability specification
   capabilities = DeviceCapabilities(
       device_id="tablet_002",
       device_type=DeviceType.TABLET,
       cpu_cores=8,
       ram_total_mb=8192,
       has_gpu=True,
       gpu_model="Adreno 660",
       battery_powered=True,
       network_type="wifi",
       supports_ml_frameworks=["tflite", "onnx", "pytorch_mobile"]
   )
   ```

2. **Resource Profiling**
   - Hardware specifications (CPU, RAM, GPU, Storage)
   - Power characteristics (battery, charging, thermal)
   - Network capabilities (bandwidth, latency, quality)
   - Software capabilities (ML frameworks, containers)

3. **Security Assessment**
   - Trust level evaluation
   - Attestation support verification
   - Secure enclave availability

### Deployment Strategies

#### Blue-Green Deployment
```python
config = DeploymentConfig(
    deployment_id="safe_rollout",
    rollout_strategy="blue_green",
    rollback_on_failure=True
)
# Deploys to 50% of devices first, then remaining 50%
```

#### Canary Deployment
```python
config = DeploymentConfig(
    deployment_id="canary_test",
    rollout_strategy="canary",
    max_concurrent_deployments=2
)
# Deploys to 10% of devices first for testing
```

#### Rolling Deployment
```python
config = DeploymentConfig(
    deployment_id="gradual_rollout",
    rollout_strategy="rolling",
    max_concurrent_deployments=5
)
# Deploys to small batches with delays between batches
```

### Deployment Validation

The system validates devices against multiple criteria:

1. **Hardware Requirements**
   - Minimum CPU cores and frequency
   - Required RAM and storage
   - GPU availability if needed

2. **Power Constraints**
   - Battery level thresholds
   - Charging status requirements
   - Thermal state limits

3. **Network Requirements**
   - Connection quality minimums
   - Bandwidth availability
   - Latency constraints

4. **Security Requirements**
   - Attestation capabilities
   - Trust level thresholds
   - Encryption support

## Fog Computing Orchestration

### Fog Coordinator Overview

```python
from infrastructure.fog.edge.fog_compute.fog_coordinator import (
    FogCoordinator, TaskType, TaskPriority
)

# Initialize fog coordinator
coordinator = FogCoordinator("fog_main")

# Register fog nodes
await coordinator.register_node(
    node_id="phone_001",
    capacity=ComputeCapacity(
        cpu_cores=6,
        cpu_utilization=0.2,
        memory_mb=4096,
        memory_used_mb=1024,
        battery_powered=True,
        battery_percent=75,
        is_charging=True
    )
)

# Submit compute tasks
task_id = await coordinator.submit_task(
    task_type=TaskType.INFERENCE,
    priority=TaskPriority.HIGH,
    cpu_cores=1.0,
    memory_mb=512,
    estimated_duration=120.0  # 2 minutes
)
```

### Task Scheduling Policies

1. **Mobile-Aware Scheduling**
   - Prefer charging devices for heavy tasks
   - Battery level minimums (default: 30%)
   - Thermal throttling thresholds
   - Network cost awareness

2. **Resource Optimization**
   - CPU and memory utilization tracking
   - Dynamic load balancing
   - Overload detection and redistribution

3. **Fault Tolerance**
   - Node failure detection
   - Automatic task redistribution
   - Graceful degradation

### Idle Resource Harvesting

```python
from infrastructure.fog.compute.harvest_manager import FogHarvestManager

# Initialize harvest manager
harvest_manager = FogHarvestManager(
    node_id="harvest_coordinator",
    token_rate_per_hour=100
)

# Register device for harvesting
device_caps = DeviceCapabilities(
    device_id="laptop_003",
    device_type=DeviceType.LAPTOP,
    cpu_cores=8,
    ram_total_mb=16384
)

await harvest_manager.register_device("laptop_003", device_caps)

# Start harvesting session (when device is charging and idle)
session_id = await harvest_manager.start_harvesting(
    "laptop_003",
    {
        "battery_percent": 60,
        "is_charging": True,
        "screen_on": False,
        "network_type": "wifi"
    }
)
```

### Harvesting Policies

1. **Eligibility Criteria**
   - Battery > 20% and device charging
   - CPU temperature < 45°C
   - WiFi connection (no metered networks)
   - Screen off and user inactive

2. **Resource Limits**
   - Max 50% CPU utilization
   - Max 30% memory usage
   - Max 10 Mbps bandwidth
   - Thermal throttling at 55°C

3. **Scheduling Windows**
   - Default: 22:00-07:00 (overnight)
   - Max 8-hour continuous sessions
   - 30-minute cooldown between sessions

## Battery/Thermal-Aware Resource Management

### Mobile Resource Manager

```python
from infrastructure.fog.edge.mobile.resource_manager import MobileResourceManager

# Initialize with harvest capability
manager = MobileResourceManager(
    harvest_enabled=True,
    token_rewards_enabled=True
)

# Create device profile
profile = manager.create_device_profile_from_env("device_001")

# Get optimization recommendations
optimization = await manager.optimize_for_device(profile)

print(f"Power mode: {optimization.power_mode}")
print(f"Transport preference: {optimization.transport_preference}")
print(f"Chunk size: {optimization.chunking_config.effective_chunk_size()}")
```

### Environment-Driven Testing

Set environment variables for testing different scenarios:

```bash
# Low battery scenario
export BATTERY=15
export AIV_MOBILE_PROFILE=battery_save

# Thermal throttling scenario
export THERMAL=65
export AIV_MOBILE_PROFILE=thermal_throttle

# Low memory scenario
export MEMORY_GB=2
export AIV_MOBILE_PROFILE=low_ram

# Performance scenario
export BATTERY=90
export THERMAL=normal
export MEMORY_GB=8
export AIV_MOBILE_PROFILE=performance
```

### Optimization Policies

1. **Power Management**
   - Critical battery (<10%): Minimal processing
   - Low battery (<20%): BitChat-only transport
   - Conservative battery (<40%): Reduced chunk sizes
   - Charging: Enhanced performance mode

2. **Thermal Management**
   - Normal (<35°C): Full performance
   - Warm (35-45°C): 75% performance
   - Hot (45-55°C): 50% performance
   - Critical (>65°C): Emergency throttling

3. **Memory Management**
   - Dynamic chunk sizing based on available RAM
   - Aggressive cleanup on low memory
   - Tensor size optimization for 2-4GB devices

## Cross-Device Coordination Protocols

### Device Clustering

Devices are automatically clustered based on:

1. **Similarity Metrics**
   - Device type compatibility
   - Performance characteristics
   - Network proximity
   - Usage patterns

2. **Cluster Formation**
   ```python
   # Automatic cluster formation
   await deployer._evaluate_cluster_formation("device_001")

   # Manual cluster creation
   cluster_id = "high_performance_cluster"
   deployer.device_clusters[cluster_id] = {
       "laptop_001", "laptop_002", "desktop_001"
   }
   ```

3. **Cluster Maintenance**
   - Health monitoring of cluster members
   - Automatic removal of failed devices
   - Leader election for coordination roles
   - Load distribution within clusters

### P2P Communication

```python
# P2P connection management
deployer.p2p_connections["device_001"] = {
    "device_002", "device_003", "device_004"
}

# Cross-device task coordination
await deployer._update_p2p_connections()
```

### Consensus Mechanisms

1. **Leader Election**
   - Highest compute score becomes leader
   - Automatic re-election on leader failure
   - Cluster-wide decision coordination

2. **Task Distribution**
   - Distributed task queues
   - Load-aware task assignment
   - Failure detection and redistribution

3. **Resource Sharing**
   - Cross-device memory pooling
   - Distributed caching
   - Bandwidth aggregation

## Mobile Optimization & Adaptive QoS

### Network-Adaptive Transport Selection

```python
# Get transport routing decision
routing = await manager.get_transport_routing_decision(
    message_size_bytes=1024,
    priority=7,
    profile=device_profile
)

print(f"Primary transport: {routing['primary_transport']}")
print(f"Fallback transport: {routing['fallback_transport']}")
print(f"Rationale: {routing['rationale']}")
```

### Transport Preferences

1. **BitChat-First Strategy**
   - Preferred for low battery conditions
   - Better for offline/intermittent connectivity
   - Lower power consumption
   - Store-and-forward capability

2. **Balanced Approach**
   - BitChat for small messages (<10KB)
   - Betanet for large messages or high priority
   - Dynamic switching based on conditions

3. **Network Quality Adaptation**
   - Excellent: Full capabilities enabled
   - Good: Standard operation
   - Fair: Reduced quality, increased buffering
   - Poor: Offline-first mode
   - Offline: Local processing only

### QoS Policies

```python
# Set device-specific QoS policy
deployer.qos_policies["device_001"] = {
    "priority": "battery_save",  # battery_save, balanced, performance
    "max_cpu_percent": 30.0,
    "max_memory_mb": 1024,
    "thermal_throttle_temp": 50.0,
    "battery_save_threshold": 25.0
}
```

## Monitoring & Diagnostics

### Comprehensive System Monitoring

```python
# Get system status
status = await deployer.get_system_status()
print(f"Total devices: {status['total_devices']}")
print(f"Active deployments: {status['active_deployments']}")
print(f"Device clusters: {status['device_clusters']}")

# Get device-specific status
device_status = await deployer.get_device_status("device_001")
print(f"Deployment score: {device_status['deployment_score']}")
print(f"Battery state: {device_status['battery_state']}")
print(f"Thermal state: {device_status['thermal_state']}")
```

### Health Monitoring

1. **Device Health Indicators**
   - CPU and memory utilization
   - Battery level and charging status
   - Thermal state and temperature
   - Network quality and latency

2. **Deployment Health**
   - Service availability and uptime
   - Error rates and restart counts
   - Performance metrics (latency, throughput)
   - Resource consumption tracking

3. **System Health**
   - Cluster stability and membership
   - Task completion rates
   - Resource utilization efficiency
   - Power consumption optimization

### Diagnostic Data Collection

```python
# Get harvest statistics
harvest_stats = harvest_manager.get_harvest_stats()
print(f"Active sessions: {harvest_stats['active_sessions']}")
print(f"Total tokens earned: {harvest_stats['total_tokens_earned']}")

# Get network statistics
network_stats = await harvest_manager.get_network_stats()
print(f"Total compute score: {network_stats['total_compute_score']}")
print(f"State distribution: {network_stats['state_distribution']}")
```

## Operational Procedures

### Daily Operations

1. **Morning Startup**
   ```bash
   # Check system health
   python -c "
   import asyncio
   from infrastructure.fog.edge.deployment.edge_deployer import EdgeDeployer
   async def check_health():
       deployer = EdgeDeployer()
       status = await deployer.get_system_status()
       print('System Status:', status)
   asyncio.run(check_health())
   "
   ```

2. **Device Registration**
   ```python
   # Register new devices
   await deployer.register_device(device_id, capabilities, initial_state)

   # Verify registration
   device_status = await deployer.get_device_status(device_id)
   assert device_status is not None
   ```

3. **Deployment Management**
   ```python
   # Deploy new models
   deployment_ids = await deployer.deploy(config)

   # Monitor deployment progress
   for dep_id in deployment_ids:
       status = await deployer.get_deployment_status(dep_id)
       print(f"Deployment {dep_id}: {status.status}")
   ```

### Performance Optimization

1. **Resource Allocation Tuning**
   ```python
   # Optimize resource allocation
   await deployer._optimize_resource_allocation()

   # Check allocation efficiency
   for device_id, allocations in deployer.resource_allocations.items():
       utilization = calculate_utilization(device_id, allocations)
       if utilization < 0.6:  # Less than 60% utilization
           print(f"Device {device_id} underutilized: {utilization}")
   ```

2. **Load Balancing**
   ```python
   # Manual load balancing trigger
   await deployer._balance_workload()

   # Check task distribution
   max_tasks = max(len(tasks) for tasks in deployer.task_queues.values())
   min_tasks = min(len(tasks) for tasks in deployer.task_queues.values())
   balance_ratio = min_tasks / max(max_tasks, 1)
   print(f"Load balance ratio: {balance_ratio:.2f}")
   ```

3. **Network Optimization**
   ```python
   # Update network monitoring
   await deployer._update_network_monitoring()

   # Adjust QoS based on network conditions
   for device_id, network_data in deployer.network_monitors.items():
       if network_data["latency_ms"] > 200:  # High latency
           deployer.qos_policies[device_id]["priority"] = "low_latency"
   ```

### Maintenance Tasks

1. **Weekly Maintenance**
   - Clear old diagnostic data
   - Update device capability profiles
   - Optimize cluster formations
   - Review and adjust policies

2. **Monthly Maintenance**
   - Analyze performance trends
   - Update deployment strategies
   - Review security policies
   - Plan capacity expansions

### Scaling Operations

1. **Horizontal Scaling**
   ```python
   # Add more edge deployers
   secondary_deployer = EdgeDeployer("secondary_deployer")

   # Distribute device management
   await secondary_deployer.register_device(device_id, capabilities)
   ```

2. **Cluster Expansion**
   ```python
   # Expand existing clusters
   cluster_devices = deployer.device_clusters["cluster_001"]
   cluster_devices.add("new_device_001")

   # Create specialized clusters
   gpu_cluster = {d_id for d_id, caps in deployer.registered_devices.items()
                  if caps.has_gpu}
   deployer.device_clusters["gpu_cluster"] = gpu_cluster
   ```

## Troubleshooting

### Common Issues

1. **Device Registration Failures**
   ```python
   # Debug capability detection
   try:
       await deployer.register_device(device_id, capabilities)
   except Exception as e:
       logger.error(f"Registration failed: {e}")
       # Check device connectivity
       # Validate capability data
       # Verify security requirements
   ```

2. **Deployment Failures**
   ```python
   # Check deployment status
   status = await deployer.get_deployment_status(deployment_id)
   if status.status == DeploymentStatus.FAILED:
       print(f"Failure reason: {status.last_error}")
       # Common causes:
       # - Insufficient resources
       # - Network connectivity issues
       # - Security requirement failures
       # - Thermal/battery constraints
   ```

3. **Performance Issues**
   ```python
   # Monitor resource utilization
   for device_id in deployer.registered_devices:
       device_status = await deployer.get_device_status(device_id)
       thermal_state = device_status["thermal_state"]["state"]
       if thermal_state in ["hot", "critical"]:
           print(f"Device {device_id} thermal throttling")
   ```

### Diagnostic Commands

```bash
# Check device states
python -c "
import asyncio
from infrastructure.fog.edge.deployment.edge_deployer import EdgeDeployer
async def diagnose():
    deployer = EdgeDeployer()
    for device_id in deployer.registered_devices:
        status = await deployer.get_device_status(device_id)
        print(f'{device_id}: {status[\"current_state\"][\"status\"]}')
asyncio.run(diagnose())
"

# Monitor harvest performance
python -c "
from infrastructure.fog.compute.harvest_manager import FogHarvestManager
manager = FogHarvestManager('test')
stats = manager.get_harvest_stats()
print('Harvest Stats:', stats)
"

# Test mobile optimization
export BATTERY=20 THERMAL=45 MEMORY_GB=3
python -c "
import asyncio
from infrastructure.fog.edge.mobile.resource_manager import MobileResourceManager
async def test_mobile():
    manager = MobileResourceManager()
    optimization = await manager.optimize_for_device()
    print('Mobile Optimization:', optimization.to_dict())
asyncio.run(test_mobile())
"
```

### Performance Metrics

Monitor these key metrics for system health:

1. **Deployment Success Rate**: Target >95%
2. **Device Availability**: Target >90%
3. **Task Completion Rate**: Target >98%
4. **Average Latency**: Target <500ms
5. **Battery Efficiency**: Minimal impact on user experience
6. **Thermal Performance**: <5% time in throttled state

### Log Analysis

Key log patterns to monitor:

```bash
# Deployment failures
grep "Deployment failed" /var/log/edge-deployer.log

# Device failures
grep "Device.*failed" /var/log/edge-deployer.log

# Thermal throttling
grep "thermal.*throttle" /var/log/resource-manager.log

# Resource exhaustion
grep "Memory allocation exceeded" /var/log/edge-deployer.log
```

## Success Metrics

The implementation successfully achieves:

✅ **Complete edge device deployment automation**
- Automated device registration and capability discovery
- Battery/thermal-aware deployment validation
- Multiple deployment strategies (blue-green, canary, rolling)
- Real-time deployment health monitoring

✅ **Fog computing system operational**
- Idle resource harvesting from mobile devices
- Distributed task scheduling and load balancing
- Cross-device coordination and clustering
- Token-based incentive system

✅ **Battery/thermal-aware scheduling working**
- Dynamic power mode selection
- Thermal throttling with progressive limits
- Battery-aware transport selection (BitChat-first)
- Adaptive chunk sizing for resource constraints

✅ **Cross-device coordination protocols functional**
- Automatic device clustering
- P2P connection management
- Distributed task queues and load balancing
- Fault tolerance with automatic recovery

✅ **Comprehensive operational documentation**
- Complete deployment procedures
- Monitoring and diagnostics guide
- Troubleshooting procedures
- Performance optimization guidelines

This implementation provides a production-ready edge computing platform optimized for mobile devices with comprehensive battery/thermal awareness, cross-device coordination, and fog computing capabilities.
