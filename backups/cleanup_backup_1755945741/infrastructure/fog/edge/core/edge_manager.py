"""
Unified Edge Manager - Orchestrates all edge device operations

Consolidates functionality from:
- src/digital_twin/deployment/edge_manager.py (primary implementation)
- src/production/monitoring/mobile/resource_management.py
- Multiple device profiling and deployment systems

This provides a single entry point for:
- Device registration and lifecycle management
- Mobile-optimized deployment with battery/thermal awareness
- Fog computing coordination across edge devices
- Real-time resource policy adaptation
- Cross-platform deployment (Android/iOS/Desktop)
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import platform
import time
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported edge device types"""

    SMARTPHONE = "smartphone"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    CHROMEBOOK = "chromebook"
    RASPBERRY_PI = "raspberry_pi"
    SMART_TV = "smart_tv"
    IOT_DEVICE = "iot_device"


class EdgeState(Enum):
    """Edge device operational states"""

    OFFLINE = "offline"
    CONNECTING = "connecting"
    ONLINE = "online"
    DEPLOYING = "deploying"
    RUNNING = "running"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class PowerMode(Enum):
    """Device power management modes"""

    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"
    CRITICAL = "critical"


@dataclass
class DeviceCapabilities:
    """Comprehensive device capability profile"""

    # Hardware specs
    cpu_cores: int
    ram_total_mb: int
    ram_available_mb: int
    storage_available_gb: float
    gpu_available: bool
    gpu_memory_mb: int

    # Power and thermal
    battery_powered: bool
    battery_percent: int | None = None
    battery_charging: bool = False
    cpu_temp_celsius: float | None = None
    thermal_state: str = "normal"

    # Network capabilities
    network_type: str = "wifi"  # wifi, cellular, ethernet
    network_latency_ms: float | None = None
    has_internet: bool = True
    is_metered_connection: bool = False

    # Software support
    supports_python: bool = True
    supports_containers: bool = False
    supports_background_tasks: bool = True
    max_concurrent_tasks: int = 4

    # Mobile-specific
    supports_bitchat: bool = False
    supports_nearby: bool = False
    supports_ble: bool = False
    screen_always_on: bool = False


@dataclass
class EdgeDeployment:
    """Represents a deployed AI workload on an edge device"""

    deployment_id: str
    device_id: str
    model_id: str
    deployment_type: str  # tutor, agent, inference, training

    # Resource allocation
    cpu_limit_percent: float = 50.0
    memory_limit_mb: int = 1024
    priority: int = 5  # 1=low, 10=critical

    # State tracking
    state: str = "pending"
    deployed_at: datetime | None = None
    last_used: datetime | None = None
    usage_count: int = 0

    # Performance metrics
    avg_response_time_ms: float = 0.0
    success_rate: float = 0.0
    resource_efficiency: float = 0.0

    # Mobile optimization
    offline_capable: bool = False
    compressed_size_mb: float = 0.0
    chunk_size_bytes: int = 1024


class EdgeManager:
    """
    Unified edge device management system

    Provides comprehensive orchestration of:
    - Device discovery and registration
    - Resource-aware deployment
    - Mobile optimization policies
    - Fog computing coordination
    - Performance monitoring
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

        # Core components
        self.devices: dict[str, "EdgeDevice"] = {}
        self.deployments: dict[str, EdgeDeployment] = {}
        self.fog_nodes: dict[str, "FogNode"] = {}

        # Resource management
        self.resource_policies = self._init_resource_policies()
        self.deployment_queue: list[EdgeDeployment] = []

        # Background services
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitor_active = True

        # Statistics
        self.stats = {
            "devices_registered": 0,
            "deployments_active": 0,
            "fog_compute_tasks": 0,
            "battery_optimizations": 0,
            "thermal_throttles": 0,
            "policy_adaptations": 0,
        }

        logger.info("Edge Manager initialized")

    def _init_resource_policies(self) -> dict[str, Any]:
        """Initialize mobile/edge resource management policies"""
        return {
            "battery_critical": 10,  # %
            "battery_low": 20,  # %
            "battery_conservative": 40,  # %
            "thermal_normal": 35.0,  # Celsius
            "thermal_warm": 45.0,  # Celsius
            "thermal_hot": 55.0,  # Celsius
            "thermal_critical": 65.0,  # Celsius
            "memory_low_gb": 2.0,  # GB
            "memory_medium_gb": 4.0,  # GB
            "memory_high_gb": 8.0,  # GB
            "chunk_size_base": 512,  # bytes
            "chunk_size_min": 64,  # bytes
            "chunk_size_max": 2048,  # bytes
        }

    async def register_device(
        self, device_id: str, device_name: str, auto_detect: bool = True, capabilities: DeviceCapabilities | None = None
    ) -> "EdgeDevice":
        """Register a new edge device with capability detection"""

        if device_id in self.devices:
            logger.warning(f"Device {device_id} already registered")
            return self.devices[device_id]

        # Auto-detect device capabilities if not provided
        if capabilities is None and auto_detect:
            capabilities = await self._detect_device_capabilities()
        elif capabilities is None:
            capabilities = self._get_default_capabilities()

        # Classify device type
        device_type = self._classify_device_type(capabilities)

        # Create edge device instance
        device = EdgeDevice(
            device_id=device_id,
            device_name=device_name,
            device_type=device_type,
            capabilities=capabilities,
            state=EdgeState.ONLINE,
            registered_at=datetime.now(UTC),
            last_seen=datetime.now(UTC),
        )

        self.devices[device_id] = device
        self.stats["devices_registered"] += 1

        logger.info(f"Registered device {device_name} ({device_type.value}) with {capabilities.ram_total_mb}MB RAM")

        # Start monitoring if this is the first device
        if len(self.devices) == 1:
            asyncio.create_task(self._start_monitoring())

        return device

    async def _detect_device_capabilities(self) -> DeviceCapabilities:
        """Auto-detect current device capabilities"""
        try:
            # Basic system info
            cpu_cores = psutil.cpu_count(logical=False) or 1
            memory = psutil.virtual_memory()
            ram_total_mb = int(memory.total / (1024 * 1024))
            ram_available_mb = int(memory.available / (1024 * 1024))

            # Storage info
            disk = psutil.disk_usage("/")
            storage_available_gb = disk.free / (1024 * 1024 * 1024)

            # Network detection
            network_type = "wifi"  # Default assumption
            has_internet = True  # Default assumption

            # Battery detection (if available)
            battery_powered = False
            battery_percent = None
            battery_charging = False
            try:
                battery = psutil.sensors_battery()
                if battery:
                    battery_powered = True
                    battery_percent = int(battery.percent)
                    battery_charging = battery.power_plugged
            except (AttributeError, OSError):
                pass

            # Thermal detection (if available)
            cpu_temp = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for sensor_name, sensor_list in temps.items():
                        if sensor_list:
                            cpu_temp = sensor_list[0].current
                            break
            except (AttributeError, OSError):
                pass

            # Platform-specific detection
            os_name = platform.system().lower()
            supports_containers = os_name in ["linux", "darwin"]

            # Mobile platform detection
            supports_bitchat = battery_powered  # Assume mobile devices support BitChat
            supports_nearby = os_name == "android"
            supports_ble = battery_powered

            return DeviceCapabilities(
                cpu_cores=cpu_cores,
                ram_total_mb=ram_total_mb,
                ram_available_mb=ram_available_mb,
                storage_available_gb=storage_available_gb,
                gpu_available=False,  # Would need more sophisticated detection
                gpu_memory_mb=0,
                battery_powered=battery_powered,
                battery_percent=battery_percent,
                battery_charging=battery_charging,
                cpu_temp_celsius=cpu_temp,
                thermal_state="normal",
                network_type=network_type,
                has_internet=has_internet,
                supports_python=True,
                supports_containers=supports_containers,
                supports_bitchat=supports_bitchat,
                supports_nearby=supports_nearby,
                supports_ble=supports_ble,
            )

        except Exception as e:
            logger.warning(f"Error detecting device capabilities: {e}")
            return self._get_default_capabilities()

    def _get_default_capabilities(self) -> DeviceCapabilities:
        """Get safe default device capabilities"""
        return DeviceCapabilities(
            cpu_cores=2,
            ram_total_mb=4096,
            ram_available_mb=2048,
            storage_available_gb=50.0,
            gpu_available=False,
            gpu_memory_mb=0,
            battery_powered=False,
            thermal_state="normal",
            network_type="wifi",
            has_internet=True,
            supports_python=True,
            supports_containers=False,
        )

    def _classify_device_type(self, capabilities: DeviceCapabilities) -> DeviceType:
        """Classify device type based on capabilities"""
        # Mobile device detection
        if capabilities.battery_powered and capabilities.ram_total_mb <= 6000:
            if capabilities.ram_total_mb <= 3000:
                return DeviceType.SMARTPHONE
            return DeviceType.TABLET

        # Laptop detection
        if capabilities.battery_powered and capabilities.ram_total_mb > 6000:
            return DeviceType.LAPTOP

        # Low-resource device detection
        if capabilities.cpu_cores <= 2 and capabilities.ram_total_mb <= 2000:
            return DeviceType.RASPBERRY_PI

        # Default to desktop
        return DeviceType.DESKTOP

    async def deploy_workload(
        self, device_id: str, model_id: str, deployment_type: str = "inference", config: dict[str, Any] | None = None
    ) -> str:
        """Deploy AI workload to edge device with optimization"""

        if device_id not in self.devices:
            raise ValueError(f"Device {device_id} not registered")

        device = self.devices[device_id]
        config = config or {}

        # Generate deployment ID
        deployment_id = f"deploy_{device_id[:8]}_{model_id[:8]}_{int(time.time())}"

        # Evaluate deployment optimization strategy
        optimization = await self._optimize_deployment(device, model_id, deployment_type)

        # Create deployment record
        deployment = EdgeDeployment(
            deployment_id=deployment_id,
            device_id=device_id,
            model_id=model_id,
            deployment_type=deployment_type,
            cpu_limit_percent=optimization["cpu_limit"],
            memory_limit_mb=optimization["memory_limit"],
            priority=config.get("priority", 5),
            state="pending",
            offline_capable=optimization["offline_capable"],
            compressed_size_mb=optimization["compressed_size_mb"],
            chunk_size_bytes=optimization["chunk_size"],
        )

        self.deployments[deployment_id] = deployment
        self.deployment_queue.append(deployment)

        # Execute deployment asynchronously
        asyncio.create_task(self._execute_deployment(deployment))

        logger.info(f"Queued deployment {deployment_id} for device {device_id}")
        return deployment_id

    async def _optimize_deployment(self, device: "EdgeDevice", model_id: str, deployment_type: str) -> dict[str, Any]:
        """Optimize deployment for device capabilities and state"""

        caps = device.capabilities
        optimization = {
            "cpu_limit": 50.0,
            "memory_limit": min(1024, caps.ram_available_mb // 2),
            "chunk_size": self.resource_policies["chunk_size_base"],
            "offline_capable": False,
            "compressed_size_mb": 50.0,
        }

        # Battery-aware optimization
        if caps.battery_powered and caps.battery_percent:
            if caps.battery_percent <= self.resource_policies["battery_critical"]:
                optimization["cpu_limit"] = 20.0
                optimization["memory_limit"] = min(512, optimization["memory_limit"])
                optimization["chunk_size"] = self.resource_policies["chunk_size_min"]
                optimization["offline_capable"] = True
                self.stats["battery_optimizations"] += 1

            elif caps.battery_percent <= self.resource_policies["battery_low"]:
                optimization["cpu_limit"] = 35.0
                optimization["memory_limit"] = int(optimization["memory_limit"] * 0.7)
                optimization["chunk_size"] = int(optimization["chunk_size"] * 0.7)

        # Thermal-aware optimization
        if caps.cpu_temp_celsius:
            if caps.cpu_temp_celsius >= self.resource_policies["thermal_critical"]:
                optimization["cpu_limit"] = 15.0
                optimization["memory_limit"] = min(256, optimization["memory_limit"])
                optimization["chunk_size"] = self.resource_policies["chunk_size_min"]
                self.stats["thermal_throttles"] += 1

            elif caps.cpu_temp_celsius >= self.resource_policies["thermal_hot"]:
                optimization["cpu_limit"] = 30.0
                optimization["memory_limit"] = int(optimization["memory_limit"] * 0.6)

        # Memory-constrained optimization
        available_gb = caps.ram_available_mb / 1024.0
        if available_gb <= self.resource_policies["memory_low_gb"]:
            optimization["memory_limit"] = min(512, optimization["memory_limit"])
            optimization["chunk_size"] = self.resource_policies["chunk_size_min"]
            optimization["compressed_size_mb"] = 25.0

        # Mobile device optimization
        if device.device_type in [DeviceType.SMARTPHONE, DeviceType.TABLET]:
            optimization["offline_capable"] = True
            optimization["compressed_size_mb"] = min(optimization["compressed_size_mb"], 100.0)

        # Ensure chunk size is within bounds
        optimization["chunk_size"] = max(
            self.resource_policies["chunk_size_min"],
            min(self.resource_policies["chunk_size_max"], optimization["chunk_size"]),
        )

        return optimization

    async def _execute_deployment(self, deployment: EdgeDeployment) -> None:
        """Execute the deployment process"""
        try:
            deployment.state = "deploying"
            deployment.deployed_at = datetime.now(UTC)

            # Simulate deployment steps
            logger.info(f"Starting deployment {deployment.deployment_id}")

            # Model preparation and compression
            await self._prepare_model(deployment)

            # Resource allocation
            await self._allocate_resources(deployment)

            # Deployment execution
            await self._deploy_to_device(deployment)

            deployment.state = "running"
            self.stats["deployments_active"] += 1

            logger.info(f"Deployment {deployment.deployment_id} completed successfully")

        except Exception as e:
            deployment.state = "error"
            logger.error(f"Deployment {deployment.deployment_id} failed: {e}")

    async def _prepare_model(self, deployment: EdgeDeployment) -> None:
        """Prepare and compress model for edge deployment"""
        # Simulate model preparation
        await asyncio.sleep(0.5)
        logger.debug(f"Model prepared for deployment {deployment.deployment_id}")

    async def _allocate_resources(self, deployment: EdgeDeployment) -> None:
        """Allocate device resources for deployment"""
        # Simulate resource allocation
        await asyncio.sleep(0.3)
        logger.debug(f"Resources allocated for deployment {deployment.deployment_id}")

    async def _deploy_to_device(self, deployment: EdgeDeployment) -> None:
        """Deploy to the actual edge device"""
        # Simulate deployment to device
        await asyncio.sleep(0.7)
        logger.debug(f"Deployed to device for deployment {deployment.deployment_id}")

    async def _start_monitoring(self) -> None:
        """Start background monitoring of edge devices"""
        while self.monitor_active:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update device states
                for device in self.devices.values():
                    await self._update_device_state(device)

                # Check deployment health
                await self._check_deployment_health()

                # Process fog computing tasks
                await self._process_fog_tasks()

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def _update_device_state(self, device: "EdgeDevice") -> None:
        """Update device state and capabilities"""
        try:
            # Update capabilities if possible
            if device.device_id == "local":  # Local device
                updated_caps = await self._detect_device_capabilities()
                old_caps = device.capabilities

                # Check for significant changes requiring policy adaptation
                if self._capabilities_changed_significantly(old_caps, updated_caps):
                    device.capabilities = updated_caps
                    await self._adapt_policies(device)
                    self.stats["policy_adaptations"] += 1

            device.last_seen = datetime.now(UTC)

        except Exception as e:
            logger.warning(f"Error updating device {device.device_id}: {e}")

    def _capabilities_changed_significantly(self, old: DeviceCapabilities, new: DeviceCapabilities) -> bool:
        """Check if capabilities changed enough to warrant policy adaptation"""

        # Battery level changes
        if old.battery_percent and new.battery_percent:
            if abs(old.battery_percent - new.battery_percent) > 10:
                return True

        # Thermal changes
        if old.cpu_temp_celsius and new.cpu_temp_celsius:
            if abs(old.cpu_temp_celsius - new.cpu_temp_celsius) > 10:
                return True

        # Memory changes
        if abs(old.ram_available_mb - new.ram_available_mb) > 512:
            return True

        return False

    async def _adapt_policies(self, device: "EdgeDevice") -> None:
        """Adapt deployment policies based on device state changes"""

        # Find deployments on this device
        device_deployments = [
            d for d in self.deployments.values() if d.device_id == device.device_id and d.state == "running"
        ]

        for deployment in device_deployments:
            # Recalculate optimization
            optimization = await self._optimize_deployment(device, deployment.model_id, deployment.deployment_type)

            # Update deployment resources if needed
            if (
                optimization["cpu_limit"] != deployment.cpu_limit_percent
                or optimization["memory_limit"] != deployment.memory_limit_mb
            ):
                deployment.cpu_limit_percent = optimization["cpu_limit"]
                deployment.memory_limit_mb = optimization["memory_limit"]
                deployment.chunk_size_bytes = optimization["chunk_size"]

                logger.info(f"Adapted resources for deployment {deployment.deployment_id}")

    async def _check_deployment_health(self) -> None:
        """Check health of all deployments"""
        for deployment in self.deployments.values():
            if deployment.state == "running":
                # Simulate health check
                deployment.last_used = datetime.now(UTC)
                deployment.usage_count += 1
                deployment.success_rate = min(1.0, deployment.success_rate + 0.01)

    async def _process_fog_tasks(self) -> None:
        """Process distributed fog computing tasks"""
        # Placeholder for fog computing coordination
        self.stats["fog_compute_tasks"] += len(self.devices)

    def get_device_status(self, device_id: str) -> dict[str, Any]:
        """Get comprehensive device status"""
        if device_id not in self.devices:
            return {"error": "Device not found"}

        device = self.devices[device_id]
        device_deployments = [d for d in self.deployments.values() if d.device_id == device_id]

        return {
            "device_info": {
                "device_id": device.device_id,
                "device_name": device.device_name,
                "device_type": device.device_type.value,
                "state": device.state.value,
                "registered_at": device.registered_at.isoformat(),
                "last_seen": device.last_seen.isoformat(),
            },
            "capabilities": {
                "cpu_cores": device.capabilities.cpu_cores,
                "ram_total_mb": device.capabilities.ram_total_mb,
                "ram_available_mb": device.capabilities.ram_available_mb,
                "battery_percent": device.capabilities.battery_percent,
                "cpu_temp_celsius": device.capabilities.cpu_temp_celsius,
                "thermal_state": device.capabilities.thermal_state,
                "network_type": device.capabilities.network_type,
            },
            "deployments": [
                {
                    "deployment_id": d.deployment_id,
                    "model_id": d.model_id,
                    "deployment_type": d.deployment_type,
                    "state": d.state,
                    "cpu_limit_percent": d.cpu_limit_percent,
                    "memory_limit_mb": d.memory_limit_mb,
                    "usage_count": d.usage_count,
                    "success_rate": d.success_rate,
                }
                for d in device_deployments
            ],
            "statistics": self.stats.copy(),
        }

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "devices": {
                "total": len(self.devices),
                "by_type": {
                    device_type.value: len([d for d in self.devices.values() if d.device_type == device_type])
                    for device_type in DeviceType
                },
                "online": len([d for d in self.devices.values() if d.state == EdgeState.ONLINE]),
            },
            "deployments": {
                "total": len(self.deployments),
                "active": len([d for d in self.deployments.values() if d.state == "running"]),
                "by_state": {
                    state: len([d for d in self.deployments.values() if d.state == state])
                    for state in ["pending", "deploying", "running", "error"]
                },
            },
            "statistics": self.stats.copy(),
            "resource_policies": self.resource_policies.copy(),
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown edge manager"""
        self.monitor_active = False
        self.executor.shutdown(wait=True)
        logger.info("Edge Manager shutdown complete")


@dataclass
class EdgeDevice:
    """Represents a registered edge device"""

    device_id: str
    device_name: str
    device_type: DeviceType
    capabilities: DeviceCapabilities
    state: EdgeState
    registered_at: datetime
    last_seen: datetime
    deployments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FogNode:
    """Represents a fog computing node for distributed processing"""

    node_id: str
    device_ids: list[str]
    coordinator_device: str
    compute_capacity: float
    active_tasks: int = 0
    max_tasks: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
