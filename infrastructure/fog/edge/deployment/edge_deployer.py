"""Edge Deployer - Comprehensive AI Model Deployment for Edge Devices

Provides complete edge deployment automation with battery/thermal-aware resource management,
cross-device coordination, and fog computing orchestration capabilities.

Features:
- Automated edge device deployment procedures
- Battery/thermal-aware resource allocation
- Cross-device coordination protocols
- Fog computing task distribution
- Mobile optimization and adaptive QoS
- Comprehensive monitoring and diagnostics
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status states"""

    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    RUNNING = "running"
    FAILED = "failed"
    UPDATING = "updating"
    TERMINATED = "terminated"


class DeviceType(Enum):
    """Supported device types"""

    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    IOT_DEVICE = "iot_device"
    EDGE_SERVER = "edge_server"


class NetworkQuality(Enum):
    """Network quality levels for adaptive deployment"""

    EXCELLENT = "excellent"  # Low latency, high bandwidth
    GOOD = "good"  # Moderate latency and bandwidth
    FAIR = "fair"  # Higher latency or limited bandwidth
    POOR = "poor"  # High latency, low bandwidth
    OFFLINE = "offline"  # No network connectivity


@dataclass
class DeviceCapabilities:
    """Comprehensive device capability profiling"""

    device_id: str
    device_type: DeviceType
    device_name: str = ""

    # Hardware specs
    cpu_cores: int = 2
    cpu_freq_ghz: float = 1.0
    cpu_architecture: str = "arm64"
    ram_total_mb: int = 4096
    ram_available_mb: int = 2048
    storage_total_gb: int = 64
    storage_available_gb: int = 32

    # GPU capabilities
    has_gpu: bool = False
    gpu_model: str | None = None
    gpu_memory_mb: int | None = None
    gpu_compute_score: float = 0.0

    # Power management
    battery_powered: bool = True
    battery_percent: int | None = None
    is_charging: bool = False
    power_budget_watts: float = 10.0
    thermal_design_power: float = 15.0

    # Network capabilities
    network_type: str = "wifi"  # wifi, cellular, 4g, 5g, ethernet
    network_speed_mbps: float = 10.0
    network_latency_ms: float = 50.0
    network_quality: NetworkQuality = NetworkQuality.GOOD
    has_unlimited_data: bool = True

    # Software capabilities
    os_type: str = "android"
    os_version: str = "12"
    supports_containers: bool = False
    supports_wasm: bool = True
    supports_python: bool = False
    supports_ml_frameworks: list[str] = field(default_factory=list)

    # Security features
    has_secure_enclave: bool = False
    supports_attestation: bool = False
    trust_level: float = 0.5  # 0.0-1.0

    def compute_deployment_score(self) -> float:
        """Calculate overall deployment suitability score"""
        # Hardware score (40%)
        cpu_score = min(1.0, self.cpu_cores / 8) * min(1.0, self.cpu_freq_ghz / 3.0)
        memory_score = min(1.0, self.ram_available_mb / 4096)
        hardware_score = (cpu_score + memory_score) / 2 * 0.4

        # Power score (30%)
        power_score = 1.0
        if self.battery_powered and self.battery_percent:
            if self.is_charging:
                power_score = min(1.0, self.battery_percent / 50)  # Optimal at 50%+
            else:
                power_score = max(0.1, self.battery_percent / 100)
        power_score *= 0.3

        # Network score (20%)
        network_multipliers = {
            NetworkQuality.EXCELLENT: 1.0,
            NetworkQuality.GOOD: 0.8,
            NetworkQuality.FAIR: 0.6,
            NetworkQuality.POOR: 0.3,
            NetworkQuality.OFFLINE: 0.1,
        }
        network_score = network_multipliers[self.network_quality] * 0.2

        # Trust score (10%)
        trust_score = self.trust_level * 0.1

        return hardware_score + power_score + network_score + trust_score


@dataclass
class DeploymentConfig:
    """Comprehensive deployment configuration"""

    deployment_id: str
    model_id: str
    deployment_type: str  # inference, training, federation, hybrid
    target_devices: list[str]

    # Resource requirements
    min_cpu_cores: int = 1
    min_memory_mb: int = 512
    min_storage_mb: int = 100
    requires_gpu: bool = False
    estimated_power_usage_watts: float = 5.0

    # Performance requirements
    max_latency_ms: int = 1000
    min_throughput_ops_sec: int = 10
    accuracy_threshold: float = 0.85

    # Deployment strategy
    rollout_strategy: str = "blue_green"  # blue_green, canary, rolling
    max_concurrent_deployments: int = 5
    rollback_on_failure: bool = True
    health_check_interval_sec: int = 30

    # Mobile optimizations
    battery_aware: bool = True
    thermal_aware: bool = True
    network_adaptive: bool = True
    offline_capable: bool = False

    # Security settings
    require_attestation: bool = False
    encrypt_model: bool = True
    allow_model_export: bool = False

    # Monitoring and telemetry
    enable_metrics: bool = True
    enable_logging: bool = True
    telemetry_interval_sec: int = 60

    # Lifecycle management
    auto_update: bool = True
    max_deployment_age_hours: int = 24
    cleanup_on_termination: bool = True

    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    priority: int = 5  # 1-10, higher is more important


@dataclass
class DeploymentStatus:
    """Real-time deployment status tracking"""

    deployment_id: str
    device_id: str
    status: DeploymentStatus

    # Timestamps
    created_at: datetime
    started_at: datetime | None = None
    deployed_at: datetime | None = None
    last_health_check: datetime | None = None

    # Performance metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: int = 0
    inference_count: int = 0
    error_count: int = 0
    average_latency_ms: float = 0.0
    throughput_ops_sec: float = 0.0

    # Resource consumption
    power_consumption_watts: float = 0.0
    thermal_state: str = "normal"  # normal, warm, hot, critical
    network_usage_mb: float = 0.0

    # Health indicators
    health_score: float = 1.0  # 0.0-1.0
    last_error: str | None = None
    restart_count: int = 0

    # Mobile-specific metrics
    battery_impact_score: float = 0.0  # Impact on battery life
    user_experience_score: float = 1.0  # Impact on user experience


class EdgeDeployer:
    """Comprehensive Edge Device Deployment Manager

    Provides complete automation for edge device deployments with:
    - Battery/thermal-aware resource management
    - Cross-device coordination protocols
    - Fog computing orchestration
    - Mobile optimization and adaptive QoS
    - Comprehensive monitoring and diagnostics
    """

    def __init__(
        self,
        coordinator_id: str | None = None,
        enable_fog_computing: bool = True,
        enable_cross_device_coordination: bool = True,
    ):
        self.coordinator_id = coordinator_id or f"edge_deployer_{uuid4().hex[:8]}"
        self.enable_fog_computing = enable_fog_computing
        self.enable_cross_device_coordination = enable_cross_device_coordination

        # Device management
        self.registered_devices: dict[str, DeviceCapabilities] = {}
        self.device_states: dict[str, dict[str, Any]] = {}

        # Deployment tracking
        self.active_deployments: dict[str, DeploymentConfig] = {}
        self.deployment_statuses: dict[str, DeploymentStatus] = {}
        self.deployment_history: list[dict[str, Any]] = []

        # Resource management
        self.resource_allocations: dict[str, dict[str, Any]] = {}  # device_id -> allocations
        self.thermal_monitors: dict[str, dict[str, Any]] = {}  # device_id -> thermal state
        self.battery_monitors: dict[str, dict[str, Any]] = {}  # device_id -> battery state

        # Cross-device coordination
        self.device_clusters: dict[str, set[str]] = {}  # cluster_id -> device_ids
        self.coordination_protocols: dict[str, Callable] = {}
        self.p2p_connections: dict[str, set[str]] = {}  # device_id -> connected_devices

        # Task scheduling and load balancing
        self.pending_tasks: list[dict[str, Any]] = []
        self.active_tasks: dict[str, dict[str, Any]] = {}  # task_id -> task_info
        self.task_queues: dict[str, list[str]] = {}  # device_id -> task_ids

        # Monitoring and diagnostics
        self.metrics_collectors: dict[str, Callable] = {}
        self.health_checkers: dict[str, Callable] = {}
        self.diagnostic_data: dict[str, list[dict[str, Any]]] = {}

        # Network adaptive QoS
        self.network_monitors: dict[str, dict[str, Any]] = {}
        self.qos_policies: dict[str, dict[str, Any]] = {}

        # Statistics and analytics
        self.stats = {
            "total_deployments": 0,
            "successful_deployments": 0,
            "failed_deployments": 0,
            "active_devices": 0,
            "total_inference_count": 0,
            "average_latency_ms": 0.0,
            "power_efficiency_score": 0.0,
            "user_experience_score": 0.0,
        }

        # Background services
        self.monitoring_active = True
        self.coordination_active = True

        # Start background services
        asyncio.create_task(self._monitoring_loop())
        if enable_cross_device_coordination:
            asyncio.create_task(self._coordination_loop())

        logger.info(f"Edge Deployer {self.coordinator_id} initialized")

    async def register_device(
        self, device_id: str, capabilities: DeviceCapabilities, initial_state: dict[str, Any] | None = None
    ) -> bool:
        """Register an edge device with comprehensive capability profiling"""
        try:
            # Store device capabilities
            self.registered_devices[device_id] = capabilities

            # Initialize device state
            default_state = {
                "status": "active",
                "last_seen": datetime.now(UTC),
                "cpu_usage_percent": 0.0,
                "memory_usage_mb": 0,
                "battery_percent": capabilities.battery_percent,
                "is_charging": capabilities.is_charging,
                "thermal_state": "normal",
                "network_quality": capabilities.network_quality.value,
            }
            self.device_states[device_id] = {**default_state, **(initial_state or {})}

            # Initialize monitoring
            self.resource_allocations[device_id] = {"cpu": 0.0, "memory": 0, "storage": 0}
            self.thermal_monitors[device_id] = {"temp_celsius": 25.0, "state": "normal"}
            self.battery_monitors[device_id] = {
                "percent": capabilities.battery_percent,
                "charging": capabilities.is_charging,
                "health": 100.0,
            }
            self.task_queues[device_id] = []
            self.diagnostic_data[device_id] = []

            # Initialize network monitoring
            self.network_monitors[device_id] = {
                "quality": capabilities.network_quality.value,
                "latency_ms": capabilities.network_latency_ms,
                "bandwidth_mbps": capabilities.network_speed_mbps,
                "data_usage_mb": 0.0,
            }

            # Set default QoS policy
            self.qos_policies[device_id] = {
                "priority": "balanced",
                "max_cpu_percent": 50.0,
                "max_memory_mb": capabilities.ram_available_mb // 2,
                "thermal_throttle_temp": 55.0,
                "battery_save_threshold": 20.0,
            }

            self.stats["active_devices"] += 1

            logger.info(
                f"Registered device {device_id}: {capabilities.device_type.value}, "
                f"score: {capabilities.compute_deployment_score():.2f}"
            )

            # Auto-form clusters if cross-device coordination is enabled
            if self.enable_cross_device_coordination:
                await self._evaluate_cluster_formation(device_id)

            return True

        except Exception as e:
            logger.error(f"Failed to register device {device_id}: {e}")
            return False

    async def deploy(self, config: DeploymentConfig, target_devices: list[str] | None = None) -> list[str]:
        """Deploy model to edge devices with comprehensive orchestration"""

        deployment_ids = []
        target_devices = target_devices or config.target_devices

        try:
            # Validate target devices
            validated_devices = await self._validate_deployment_targets(target_devices, config)
            if not validated_devices:
                logger.error(f"No valid devices found for deployment {config.deployment_id}")
                return []

            # Apply deployment strategy
            device_batches = self._create_deployment_batches(validated_devices, config)

            for batch_idx, device_batch in enumerate(device_batches):
                batch_deployment_ids = await self._deploy_to_batch(config, device_batch, batch_idx)
                deployment_ids.extend(batch_deployment_ids)

                # Wait between batches for rolling deployments
                if batch_idx < len(device_batches) - 1 and config.rollout_strategy == "rolling":
                    await asyncio.sleep(30)  # 30 second delay between batches

            self.stats["total_deployments"] += len(deployment_ids)
            logger.info(f"Deployment {config.deployment_id} initiated on {len(deployment_ids)} devices")

            return deployment_ids

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.stats["failed_deployments"] += 1
            return []

    async def _validate_deployment_targets(self, target_devices: list[str], config: DeploymentConfig) -> list[str]:
        """Validate and filter deployment targets based on capabilities and policies"""

        validated_devices = []

        for device_id in target_devices:
            if device_id not in self.registered_devices:
                logger.warning(f"Device {device_id} not registered, skipping")
                continue

            device = self.registered_devices[device_id]
            device_state = self.device_states.get(device_id, {})

            # Check hardware requirements
            if (
                device.cpu_cores < config.min_cpu_cores
                or device.ram_available_mb < config.min_memory_mb
                or device.storage_available_gb * 1024 < config.min_storage_mb
            ):
                logger.warning(f"Device {device_id} does not meet hardware requirements")
                continue

            # Check GPU requirement
            if config.requires_gpu and not device.has_gpu:
                logger.warning(f"Device {device_id} lacks required GPU")
                continue

            # Check battery/thermal constraints if enabled
            if config.battery_aware and device.battery_powered:
                battery_percent = device.battery_percent or 0
                if battery_percent < 20 and not device.is_charging:
                    logger.warning(f"Device {device_id} battery too low for deployment")
                    continue

            if config.thermal_aware:
                thermal_state = self.thermal_monitors.get(device_id, {}).get("state", "normal")
                if thermal_state in ["hot", "critical"]:
                    logger.warning(f"Device {device_id} thermal state too high for deployment")
                    continue

            # Check network requirements
            if config.network_adaptive:
                network_quality = NetworkQuality(device_state.get("network_quality", "good"))
                if network_quality == NetworkQuality.OFFLINE and not config.offline_capable:
                    logger.warning(f"Device {device_id} offline and deployment not offline-capable")
                    continue

            # Check security requirements
            if config.require_attestation and not device.supports_attestation:
                logger.warning(f"Device {device_id} does not support required attestation")
                continue

            validated_devices.append(device_id)

        return validated_devices

    def _create_deployment_batches(self, devices: list[str], config: DeploymentConfig) -> list[list[str]]:
        """Create deployment batches based on rollout strategy"""

        if config.rollout_strategy == "blue_green":
            # Split into two equal groups
            mid = len(devices) // 2
            return [devices[:mid], devices[mid:]] if len(devices) > 1 else [devices]

        elif config.rollout_strategy == "canary":
            # First batch is 10% (min 1), rest in second batch
            canary_size = max(1, len(devices) // 10)
            return [devices[:canary_size], devices[canary_size:]] if len(devices) > canary_size else [devices]

        elif config.rollout_strategy == "rolling":
            # Break into small batches respecting max_concurrent_deployments
            batch_size = min(config.max_concurrent_deployments, len(devices))
            return [devices[i : i + batch_size] for i in range(0, len(devices), batch_size)]

        else:
            # Default: single batch
            return [devices]

    async def _deploy_to_batch(self, config: DeploymentConfig, device_batch: list[str], batch_idx: int) -> list[str]:
        """Deploy to a batch of devices concurrently"""

        deployment_tasks = []
        batch_deployment_ids = []

        for device_id in device_batch:
            deployment_id = f"{config.deployment_id}_{device_id}_{batch_idx}"
            batch_deployment_ids.append(deployment_id)

            # Create deployment status
            status = DeploymentStatus(
                deployment_id=deployment_id,
                device_id=device_id,
                status=DeploymentStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            self.deployment_statuses[deployment_id] = status

            # Schedule deployment task
            task = asyncio.create_task(self._deploy_to_device(deployment_id, device_id, config))
            deployment_tasks.append(task)

        # Wait for batch completion
        await asyncio.gather(*deployment_tasks, return_exceptions=True)

        return batch_deployment_ids

    async def _deploy_to_device(self, deployment_id: str, device_id: str, config: DeploymentConfig) -> bool:
        """Deploy model to a specific device"""

        try:
            status = self.deployment_statuses[deployment_id]
            status.status = DeploymentStatus.DEPLOYING
            status.started_at = datetime.now(UTC)

            # Update resource allocations
            allocations = self.resource_allocations[device_id]
            allocations["cpu"] += config.min_cpu_cores * 0.1  # 10% per core
            allocations["memory"] += config.min_memory_mb
            allocations["storage"] += config.min_storage_mb

            # Simulate deployment process
            await asyncio.sleep(5)  # Deployment time simulation

            # Check resource constraints during deployment
            device = self.registered_devices[device_id]
            if allocations["memory"] > device.ram_available_mb * 0.8:  # 80% memory limit
                raise Exception("Memory allocation exceeded device limits")

            # Finalize deployment
            status.status = DeploymentStatus.DEPLOYED
            status.deployed_at = datetime.now(UTC)
            status.health_score = 1.0

            self.active_deployments[deployment_id] = config

            logger.info(f"Successfully deployed {config.model_id} to device {device_id}")
            self.stats["successful_deployments"] += 1

            return True

        except Exception as e:
            logger.error(f"Failed to deploy to device {device_id}: {e}")

            # Update deployment status
            if deployment_id in self.deployment_statuses:
                status = self.deployment_statuses[deployment_id]
                status.status = DeploymentStatus.FAILED
                status.last_error = str(e)

            # Rollback resource allocations
            if device_id in self.resource_allocations:
                allocations = self.resource_allocations[device_id]
                allocations["cpu"] = max(0, allocations["cpu"] - config.min_cpu_cores * 0.1)
                allocations["memory"] = max(0, allocations["memory"] - config.min_memory_mb)
                allocations["storage"] = max(0, allocations["storage"] - config.min_storage_mb)

            self.stats["failed_deployments"] += 1
            return False

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for device health and performance"""

        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update device states
                for device_id in self.registered_devices:
                    await self._update_device_metrics(device_id)

                # Check deployment health
                for deployment_id in list(self.deployment_statuses.keys()):
                    await self._check_deployment_health(deployment_id)

                # Update thermal monitoring
                await self._update_thermal_monitoring()

                # Update battery monitoring
                await self._update_battery_monitoring()

                # Update network monitoring
                await self._update_network_monitoring()

                # Update global statistics
                await self._update_global_statistics()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _coordination_loop(self) -> None:
        """Background coordination loop for cross-device orchestration"""

        while self.coordination_active:
            try:
                await asyncio.sleep(60)  # Coordinate every minute

                # Maintain device clusters
                await self._maintain_device_clusters()

                # Balance load across devices
                await self._balance_workload()

                # Optimize resource allocation
                await self._optimize_resource_allocation()

                # Handle device failures
                await self._handle_device_failures()

                # Update P2P connections
                await self._update_p2p_connections()

            except Exception as e:
                logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(120)

    async def get_deployment_status(self, deployment_id: str) -> DeploymentStatus | None:
        """Get real-time deployment status"""
        return self.deployment_statuses.get(deployment_id)

    async def get_device_status(self, device_id: str) -> dict[str, Any] | None:
        """Get comprehensive device status"""
        if device_id not in self.registered_devices:
            return None

        device = self.registered_devices[device_id]
        state = self.device_states.get(device_id, {})
        allocations = self.resource_allocations.get(device_id, {})
        thermal = self.thermal_monitors.get(device_id, {})
        battery = self.battery_monitors.get(device_id, {})
        network = self.network_monitors.get(device_id, {})

        return {
            "device_id": device_id,
            "capabilities": device,
            "current_state": state,
            "resource_allocations": allocations,
            "thermal_state": thermal,
            "battery_state": battery,
            "network_state": network,
            "deployment_score": device.compute_deployment_score(),
            "active_deployments": [
                dep_id
                for dep_id, status in self.deployment_statuses.items()
                if status.device_id == device_id
                and status.status in [DeploymentStatus.DEPLOYED, DeploymentStatus.RUNNING]
            ],
        }

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "coordinator_id": self.coordinator_id,
            "total_devices": len(self.registered_devices),
            "active_deployments": len(
                [s for s in self.deployment_statuses.values() if s.status == DeploymentStatus.RUNNING]
            ),
            "device_clusters": {cid: list(devices) for cid, devices in self.device_clusters.items()},
            "statistics": self.stats,
            "fog_computing_enabled": self.enable_fog_computing,
            "cross_device_coordination_enabled": self.enable_cross_device_coordination,
        }

    # Helper methods for monitoring and coordination
    async def _update_device_metrics(self, device_id: str) -> None:
        """Update device performance metrics"""
        # Simulate metric updates with realistic values
        state = self.device_states.get(device_id, {})
        state["last_seen"] = datetime.now(UTC)
        state["cpu_usage_percent"] = min(100, max(0, state.get("cpu_usage_percent", 0) + (hash(device_id) % 20 - 10)))

    async def _check_deployment_health(self, deployment_id: str) -> None:
        """Check health of a specific deployment"""
        status = self.deployment_statuses.get(deployment_id)
        if status and status.status == DeploymentStatus.DEPLOYED:
            status.last_health_check = datetime.now(UTC)
            status.health_score = min(1.0, status.health_score + 0.01)

    async def _update_thermal_monitoring(self) -> None:
        """Update thermal monitoring for all devices"""
        for device_id, thermal_data in self.thermal_monitors.items():
            # Simulate thermal fluctuations
            current_temp = thermal_data.get("temp_celsius", 25.0)
            thermal_data["temp_celsius"] = max(20, min(70, current_temp + (hash(device_id) % 10 - 5)))

    async def _update_battery_monitoring(self) -> None:
        """Update battery monitoring for all devices"""
        for device_id, battery_data in self.battery_monitors.items():
            # Simulate battery changes
            if not battery_data.get("charging", False):
                battery_data["percent"] = max(0, battery_data.get("percent", 100) - 1)

    async def _update_network_monitoring(self) -> None:
        """Update network quality monitoring"""
        for device_id, network_data in self.network_monitors.items():
            # Simulate network quality fluctuations
            network_data["latency_ms"] = max(10, network_data.get("latency_ms", 50) + (hash(device_id) % 20 - 10))

    async def _update_global_statistics(self) -> None:
        """Update global system statistics"""
        active_deployments = [s for s in self.deployment_statuses.values() if s.status == DeploymentStatus.RUNNING]
        self.stats["average_latency_ms"] = sum(s.average_latency_ms for s in active_deployments) / max(
            1, len(active_deployments)
        )

    async def _evaluate_cluster_formation(self, device_id: str) -> None:
        """Evaluate whether to form device clusters"""
        # Simple clustering logic - could be enhanced with ML algorithms
        similar_devices = [
            d_id
            for d_id, device in self.registered_devices.items()
            if device.device_type == self.registered_devices[device_id].device_type and d_id != device_id
        ]

        if len(similar_devices) >= 3:  # Form cluster with 4+ similar devices
            cluster_id = f"cluster_{device_id[:8]}"
            self.device_clusters[cluster_id] = {device_id, *similar_devices[:3]}

    async def _maintain_device_clusters(self) -> None:
        """Maintain existing device clusters"""
        for cluster_id, device_ids in list(self.device_clusters.items()):
            # Remove inactive devices from clusters
            active_devices = {d_id for d_id in device_ids if d_id in self.registered_devices}
            if len(active_devices) < 2:
                del self.device_clusters[cluster_id]
            else:
                self.device_clusters[cluster_id] = active_devices

    async def _balance_workload(self) -> None:
        """Balance workload across devices"""
        # Simple load balancing - redistribute tasks from overloaded devices
        for device_id, tasks in self.task_queues.items():
            if len(tasks) > 5:  # Overloaded device
                # Move some tasks to less loaded devices
                available_devices = [
                    d_id for d_id, task_queue in self.task_queues.items() if len(task_queue) < 3 and d_id != device_id
                ]
                if available_devices:
                    tasks_to_move = tasks[:2]  # Move 2 tasks
                    self.task_queues[available_devices[0]].extend(tasks_to_move)
                    self.task_queues[device_id] = tasks[2:]

    async def _optimize_resource_allocation(self) -> None:
        """Optimize resource allocation across devices"""
        # Dynamic resource optimization based on current utilization
        for device_id, allocations in self.resource_allocations.items():
            device = self.registered_devices.get(device_id)
            if device:
                # Adjust allocations based on device capabilities and current load
                max_memory = device.ram_available_mb * 0.8
                if allocations["memory"] > max_memory:
                    allocations["memory"] = max_memory

    async def _handle_device_failures(self) -> None:
        """Handle device failures and recovery"""
        current_time = datetime.now(UTC)

        for device_id, state in self.device_states.items():
            last_seen = state.get("last_seen")
            if last_seen and (current_time - last_seen).seconds > 300:  # 5 minutes timeout
                # Mark device as failed and redistribute its tasks
                logger.warning(f"Device {device_id} appears to have failed")
                state["status"] = "failed"

                # Redistribute tasks
                failed_tasks = self.task_queues.get(device_id, [])
                if failed_tasks:
                    available_devices = [
                        d_id
                        for d_id, s in self.device_states.items()
                        if s.get("status") == "active" and d_id != device_id
                    ]
                    if available_devices:
                        # Distribute tasks among available devices
                        for i, task_id in enumerate(failed_tasks):
                            target_device = available_devices[i % len(available_devices)]
                            self.task_queues[target_device].append(task_id)
                    self.task_queues[device_id] = []

    async def _update_p2p_connections(self) -> None:
        """Update peer-to-peer connections between devices"""
        # Maintain P2P connections for cross-device coordination
        for device_id in self.registered_devices:
            if device_id not in self.p2p_connections:
                self.p2p_connections[device_id] = set()

            # Connect to devices in the same cluster
            for cluster_id, cluster_devices in self.device_clusters.items():
                if device_id in cluster_devices:
                    self.p2p_connections[device_id].update(cluster_devices - {device_id})

    async def shutdown(self) -> None:
        """Graceful shutdown of edge deployer"""
        self.monitoring_active = False
        self.coordination_active = False

        # Terminate all active deployments
        for deployment_id in list(self.deployment_statuses.keys()):
            if self.deployment_statuses[deployment_id].status == DeploymentStatus.RUNNING:
                self.deployment_statuses[deployment_id].status = DeploymentStatus.TERMINATED

        logger.info(f"Edge Deployer {self.coordinator_id} shutdown complete")
