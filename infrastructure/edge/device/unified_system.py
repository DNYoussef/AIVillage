"""
Unified Edge Device System - Complete Edge Device Lifecycle Management

This system provides comprehensive management for edge devices including resource
optimization, task scheduling, performance monitoring, and P2P integration.

Key Features:
- Complete edge device lifecycle management
- Resource-aware task scheduling and distribution
- Battery, thermal, and memory optimization
- Multi-platform support (mobile, IoT, desktop)
- P2P networking integration
- Real-time health monitoring and analytics
- Adaptive performance tuning

Architecture Components:
- EdgeDeviceRegistry: Device discovery and management
- EdgeTaskScheduler: Priority-based task distribution
- EdgeDevice: Individual device abstraction
- Performance monitoring and health tracking
- Resource management and optimization
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from typing import Any
from uuid import uuid4

try:
    import numpy as np
    import psutil
except ImportError:
    # Graceful degradation
    psutil = None
    np = None

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Types of edge devices"""

    MOBILE = "mobile"
    DESKTOP = "desktop"
    IOT = "iot"
    SERVER = "server"
    RASPBERRY_PI = "raspberry_pi"
    EMBEDDED = "embedded"


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class DeviceStatus(Enum):
    """Device operational status"""

    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""

    timestamp: datetime
    cpu_percent: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    storage_used_gb: float = 0.0
    storage_total_gb: float = 0.0
    battery_percent: float | None = None
    battery_charging: bool = False
    temperature_celsius: float | None = None
    network_rx_mb: float = 0.0
    network_tx_mb: float = 0.0

    @property
    def ram_percent(self) -> float:
        """Calculate RAM usage percentage"""
        if self.ram_total_mb > 0:
            return (self.ram_used_mb / self.ram_total_mb) * 100
        return 0.0

    @property
    def storage_percent(self) -> float:
        """Calculate storage usage percentage"""
        if self.storage_total_gb > 0:
            return (self.storage_used_gb / self.storage_total_gb) * 100
        return 0.0


@dataclass
class EdgeTask:
    """Edge computing task definition"""

    task_id: str
    name: str
    priority: TaskPriority
    resource_requirements: dict[str, Any]
    estimated_duration: timedelta
    created_at: datetime
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution state
    assigned_device: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None

    @property
    def is_completed(self) -> bool:
        """Check if task is completed"""
        return self.completed_at is not None

    @property
    def execution_time(self) -> timedelta | None:
        """Get actual execution time"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None


class EdgeDevice:
    """Individual edge device representation"""

    def __init__(
        self, device_id: str, device_type: DeviceType, capabilities: dict[str, Any], location: str | None = None
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.capabilities = capabilities
        self.location = location
        self.status = DeviceStatus.IDLE

        # Resource tracking
        self.current_metrics: ResourceMetrics | None = None
        self.metrics_history: list[ResourceMetrics] = []
        self.max_history_size = 1000

        # Task management
        self.assigned_tasks: dict[str, EdgeTask] = {}
        self.completed_tasks: list[str] = []
        self.task_lock = threading.RLock()

        # Performance tracking
        self.performance_score = 1.0
        self.reliability_score = 1.0
        self.last_heartbeat = datetime.now()

        # Optimization settings
        self.power_save_mode = False
        self.thermal_throttling = False
        self.max_concurrent_tasks = capabilities.get("max_concurrent_tasks", 4)

    def update_metrics(self, metrics: ResourceMetrics):
        """Update device resource metrics"""
        self.current_metrics = metrics
        self.metrics_history.append(metrics)

        # Trim history to prevent memory bloat
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size :]

        # Update device status based on metrics
        self._update_status_from_metrics(metrics)

        # Check for thermal issues
        self._check_thermal_status(metrics)

        self.last_heartbeat = datetime.now()

    def _update_status_from_metrics(self, metrics: ResourceMetrics):
        """Update device status based on current metrics"""
        cpu_threshold = 90.0
        ram_threshold = 85.0

        if metrics.cpu_percent > cpu_threshold or metrics.ram_percent > ram_threshold:
            self.status = DeviceStatus.OVERLOADED
        elif len(self.assigned_tasks) >= self.max_concurrent_tasks:
            self.status = DeviceStatus.BUSY
        elif len(self.assigned_tasks) > 0:
            self.status = DeviceStatus.ACTIVE
        else:
            self.status = DeviceStatus.IDLE

    def _check_thermal_status(self, metrics: ResourceMetrics):
        """Check and handle thermal conditions"""
        if metrics.temperature_celsius:
            # Mobile devices typically throttle at 40-45°C
            thermal_threshold = 42.0 if self.device_type == DeviceType.MOBILE else 70.0

            if metrics.temperature_celsius > thermal_threshold:
                if not self.thermal_throttling:
                    logger.warning(
                        f"Device {self.device_id} entering thermal throttling at {metrics.temperature_celsius}°C"
                    )
                    self.thermal_throttling = True
                    # Reduce max concurrent tasks
                    self.max_concurrent_tasks = max(1, self.max_concurrent_tasks // 2)
            else:
                if self.thermal_throttling and metrics.temperature_celsius < thermal_threshold - 5:
                    logger.info(f"Device {self.device_id} exiting thermal throttling")
                    self.thermal_throttling = False
                    # Restore normal capacity
                    self.max_concurrent_tasks = self.capabilities.get("max_concurrent_tasks", 4)

    def can_accept_task(self, task: EdgeTask) -> bool:
        """Check if device can accept a new task"""
        if self.status in [DeviceStatus.OFFLINE, DeviceStatus.MAINTENANCE]:
            return False

        # Check concurrent task limit
        if len(self.assigned_tasks) >= self.max_concurrent_tasks:
            return False

        # Check resource requirements
        if not self._check_resource_requirements(task):
            return False

        # Check battery level for mobile devices
        if self.device_type == DeviceType.MOBILE and self.current_metrics:
            if (
                self.current_metrics.battery_percent
                and self.current_metrics.battery_percent < 20
                and not self.current_metrics.battery_charging
            ):
                return False

        return True

    def _check_resource_requirements(self, task: EdgeTask) -> bool:
        """Check if device meets task resource requirements"""
        if not self.current_metrics:
            return True  # No metrics available, assume capable

        requirements = task.resource_requirements

        # Check CPU requirement
        if "cpu_percent" in requirements:
            available_cpu = 100 - self.current_metrics.cpu_percent
            if available_cpu < requirements["cpu_percent"]:
                return False

        # Check RAM requirement
        if "ram_mb" in requirements:
            if self.current_metrics.ram_available_mb < requirements["ram_mb"]:
                return False

        # Check storage requirement
        if "storage_gb" in requirements:
            available_storage = self.current_metrics.storage_total_gb - self.current_metrics.storage_used_gb
            if available_storage < requirements["storage_gb"]:
                return False

        return True

    def assign_task(self, task: EdgeTask) -> bool:
        """Assign a task to this device"""
        with self.task_lock:
            if not self.can_accept_task(task):
                return False

            task.assigned_device = self.device_id
            task.started_at = datetime.now()
            self.assigned_tasks[task.task_id] = task

            logger.info(f"Task {task.task_id} assigned to device {self.device_id}")
            return True

    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark a task as completed"""
        with self.task_lock:
            if task_id in self.assigned_tasks:
                task = self.assigned_tasks[task_id]
                task.completed_at = datetime.now()
                task.result = result
                task.error = error

                # Move to completed tasks
                del self.assigned_tasks[task_id]
                self.completed_tasks.append(task_id)

                # Update performance score
                if error is None:
                    self.performance_score = min(1.0, self.performance_score + 0.01)
                else:
                    self.performance_score = max(0.0, self.performance_score - 0.05)

                logger.info(f"Task {task_id} completed on device {self.device_id}")

    def get_capability_score(self, task: EdgeTask) -> float:
        """Calculate how well this device can handle the task"""
        if not self.can_accept_task(task):
            return 0.0

        score = self.performance_score * self.reliability_score

        # Adjust for current load
        load_factor = len(self.assigned_tasks) / self.max_concurrent_tasks
        score *= 1.0 - load_factor * 0.5

        # Adjust for resource availability
        if self.current_metrics:
            resource_score = 1.0
            resource_score *= (100 - self.current_metrics.cpu_percent) / 100
            resource_score *= self.current_metrics.ram_available_mb / self.current_metrics.ram_total_mb
            score *= resource_score

        # Bonus for device type matching
        preferred_devices = task.metadata.get("preferred_devices", [])
        if self.device_type.value in preferred_devices:
            score *= 1.2

        return min(1.0, score)


class EdgeTaskScheduler:
    """Advanced task scheduler for edge devices"""

    def __init__(self):
        self.pending_tasks: list[EdgeTask] = []
        self.active_tasks: dict[str, EdgeTask] = {}
        self.completed_tasks: dict[str, EdgeTask] = {}
        self.failed_tasks: dict[str, EdgeTask] = {}
        self.task_lock = threading.RLock()

        # Scheduling policies
        self.enable_load_balancing = True
        self.enable_locality_optimization = True
        self.max_retries = 3

    def submit_task(self, task: EdgeTask):
        """Submit a task for scheduling"""
        with self.task_lock:
            self.pending_tasks.append(task)
            logger.info(f"Task {task.task_id} submitted for scheduling")

    def schedule_tasks(self, devices: list[EdgeDevice]) -> list[tuple[EdgeTask, EdgeDevice]]:
        """Schedule pending tasks to available devices"""
        assignments = []

        with self.task_lock:
            # Sort tasks by priority and creation time
            sorted_tasks = sorted(self.pending_tasks, key=lambda t: (t.priority.value, t.created_at))

            for task in sorted_tasks[:]:  # Copy to avoid modification during iteration
                best_device = self._select_best_device(task, devices)

                if best_device and best_device.assign_task(task):
                    assignments.append((task, best_device))
                    self.pending_tasks.remove(task)
                    self.active_tasks[task.task_id] = task

        return assignments

    def _select_best_device(self, task: EdgeTask, devices: list[EdgeDevice]) -> EdgeDevice | None:
        """Select the best device for a task"""
        available_devices = [d for d in devices if d.can_accept_task(task)]

        if not available_devices:
            return None

        # Calculate capability scores for all devices
        scored_devices = [(device, device.get_capability_score(task)) for device in available_devices]

        # Filter devices with non-zero scores
        viable_devices = [(d, s) for d, s in scored_devices if s > 0]

        if not viable_devices:
            return None

        # Sort by score (highest first)
        viable_devices.sort(key=lambda x: x[1], reverse=True)

        # Apply load balancing if enabled
        if self.enable_load_balancing and len(viable_devices) > 1:
            # Consider top 3 devices and balance load
            top_devices = viable_devices[:3]
            return min(top_devices, key=lambda x: len(x[0].assigned_tasks))[0]

        return viable_devices[0][0]

    def handle_task_completion(self, task_id: str, result: Any = None, error: str = None):
        """Handle task completion notification"""
        with self.task_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                del self.active_tasks[task_id]

                if error is None:
                    self.completed_tasks[task_id] = task
                    logger.info(f"Task {task_id} completed successfully")
                else:
                    self.failed_tasks[task_id] = task
                    logger.error(f"Task {task_id} failed: {error}")

                    # Consider retry if within limits
                    retry_count = task.metadata.get("retry_count", 0)
                    if retry_count < self.max_retries:
                        task.metadata["retry_count"] = retry_count + 1
                        task.assigned_device = None
                        task.started_at = None
                        task.error = None
                        self.pending_tasks.append(task)
                        logger.info(f"Task {task_id} queued for retry ({retry_count + 1}/{self.max_retries})")

    def get_scheduling_stats(self) -> dict[str, int]:
        """Get scheduling statistics"""
        with self.task_lock:
            return {
                "pending": len(self.pending_tasks),
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
            }


class EdgeDeviceRegistry:
    """Registry for managing edge devices"""

    def __init__(self):
        self.devices: dict[str, EdgeDevice] = {}
        self.device_lock = threading.RLock()
        self.heartbeat_timeout = timedelta(minutes=5)

    def register_device(self, device: EdgeDevice):
        """Register a new edge device"""
        with self.device_lock:
            self.devices[device.device_id] = device
            logger.info(f"Device {device.device_id} registered ({device.device_type.value})")

    def unregister_device(self, device_id: str):
        """Unregister an edge device"""
        with self.device_lock:
            if device_id in self.devices:
                device = self.devices[device_id]
                # Mark offline and reassign tasks
                device.status = DeviceStatus.OFFLINE
                del self.devices[device_id]
                logger.info(f"Device {device_id} unregistered")

    def get_device(self, device_id: str) -> EdgeDevice | None:
        """Get device by ID"""
        return self.devices.get(device_id)

    def get_devices_by_type(self, device_type: DeviceType) -> list[EdgeDevice]:
        """Get all devices of a specific type"""
        return [d for d in self.devices.values() if d.device_type == device_type]

    def get_available_devices(self) -> list[EdgeDevice]:
        """Get all available devices"""
        return [d for d in self.devices.values() if d.status not in [DeviceStatus.OFFLINE, DeviceStatus.MAINTENANCE]]

    def cleanup_stale_devices(self):
        """Remove devices that haven't sent heartbeat"""
        with self.device_lock:
            current_time = datetime.now()
            stale_devices = []

            for device_id, device in self.devices.items():
                if current_time - device.last_heartbeat > self.heartbeat_timeout:
                    stale_devices.append(device_id)

            for device_id in stale_devices:
                logger.warning(f"Removing stale device {device_id}")
                self.unregister_device(device_id)

    def get_registry_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        with self.device_lock:
            stats = {"total_devices": len(self.devices), "by_type": {}, "by_status": {}}

            for device in self.devices.values():
                # Count by type
                device_type = device.device_type.value
                stats["by_type"][device_type] = stats["by_type"].get(device_type, 0) + 1

                # Count by status
                status = device.status.value
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            return stats


class UnifiedEdgeDeviceSystem:
    """Main unified edge device system"""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.device_registry = EdgeDeviceRegistry()
        self.task_scheduler = EdgeTaskScheduler()
        self.running = False

        # Background tasks
        self.scheduler_task: asyncio.Task | None = None
        self.metrics_collector_task: asyncio.Task | None = None
        self.cleanup_task: asyncio.Task | None = None

        # Performance tracking
        self.system_metrics: dict[str, Any] = {}
        self.start_time = datetime.now()

    async def initialize(self) -> bool:
        """Initialize the edge device system"""
        try:
            logger.info("Initializing Unified Edge Device System...")

            # Start background tasks
            self.running = True
            self.scheduler_task = asyncio.create_task(self._scheduler_loop())
            self.metrics_collector_task = asyncio.create_task(self._metrics_collection_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("Unified Edge Device System initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize edge device system: {e}")
            return False

    async def shutdown(self):
        """Shutdown the edge device system"""
        logger.info("Shutting down Unified Edge Device System...")

        self.running = False

        # Cancel background tasks
        for task in [self.scheduler_task, self.metrics_collector_task, self.cleanup_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Unified Edge Device System shutdown complete")

    def register_device(
        self, device_id: str, device_type: DeviceType, capabilities: dict[str, Any], location: str | None = None
    ) -> EdgeDevice:
        """Register a new edge device"""
        device = EdgeDevice(device_id, device_type, capabilities, location)
        self.device_registry.register_device(device)
        return device

    def submit_task(
        self,
        name: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        resource_requirements: dict[str, Any] | None = None,
        estimated_duration: timedelta | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EdgeTask:
        """Submit a task to the system"""
        task = EdgeTask(
            task_id=str(uuid4()),
            name=name,
            priority=priority,
            resource_requirements=resource_requirements or {},
            estimated_duration=estimated_duration or timedelta(minutes=5),
            created_at=datetime.now(),
            dependencies=dependencies or [],
            metadata=metadata or {},
        )

        self.task_scheduler.submit_task(task)
        return task

    def update_device_metrics(self, device_id: str, metrics: ResourceMetrics):
        """Update metrics for a specific device"""
        device = self.device_registry.get_device(device_id)
        if device:
            device.update_metrics(metrics)

    def get_system_status(self) -> dict[str, Any]:
        """Get overall system status"""
        registry_stats = self.device_registry.get_registry_stats()
        scheduler_stats = self.task_scheduler.get_scheduling_stats()

        return {
            "system_uptime": str(datetime.now() - self.start_time),
            "devices": registry_stats,
            "tasks": scheduler_stats,
            "system_healthy": registry_stats["total_devices"] > 0,
        }

    async def _scheduler_loop(self):
        """Background task scheduling loop"""
        while self.running:
            try:
                available_devices = self.device_registry.get_available_devices()
                if available_devices:
                    assignments = self.task_scheduler.schedule_tasks(available_devices)

                    if assignments:
                        logger.debug(f"Scheduled {len(assignments)} tasks")

                await asyncio.sleep(1)  # Schedule every second

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)

    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                # Collect system-wide metrics
                if psutil:
                    self.system_metrics = {
                        "timestamp": datetime.now().isoformat(),
                        "system_cpu_percent": psutil.cpu_percent(interval=1),
                        "system_memory": dict(psutil.virtual_memory()._asdict()),
                        "system_disk": dict(psutil.disk_usage("/")._asdict()) if hasattr(psutil, "disk_usage") else {},
                    }

                await asyncio.sleep(30)  # Collect metrics every 30 seconds

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self.running:
            try:
                # Clean up stale devices
                self.device_registry.cleanup_stale_devices()

                await asyncio.sleep(300)  # Cleanup every 5 minutes

            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)


# Example usage and testing
async def demo_unified_edge_system():
    """Demonstrate the unified edge device system"""
    print("🖥️ Unified Edge Device System Demo")
    print("⚡ Complete Device Lifecycle Management")
    print("=" * 50)

    # Initialize system
    system = UnifiedEdgeDeviceSystem()

    if not await system.initialize():
        print("Failed to initialize system")
        return

    try:
        # Register some devices
        system.register_device(
            "mobile-001", DeviceType.MOBILE, {"max_concurrent_tasks": 2, "cpu_cores": 8, "ram_gb": 6}
        )

        system.register_device(
            "desktop-001", DeviceType.DESKTOP, {"max_concurrent_tasks": 8, "cpu_cores": 16, "ram_gb": 32}
        )

        # Simulate device metrics
        mobile_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=25.0,
            ram_used_mb=2000,
            ram_total_mb=6000,
            ram_available_mb=4000,
            battery_percent=75.0,
            temperature_celsius=35.0,
        )

        desktop_metrics = ResourceMetrics(
            timestamp=datetime.now(), cpu_percent=15.0, ram_used_mb=8000, ram_total_mb=32000, ram_available_mb=24000
        )

        system.update_device_metrics("mobile-001", mobile_metrics)
        system.update_device_metrics("desktop-001", desktop_metrics)

        # Submit some tasks
        tasks = []
        for i in range(5):
            task = system.submit_task(
                f"Task-{i+1}",
                priority=TaskPriority.NORMAL,
                resource_requirements={"cpu_percent": 20, "ram_mb": 500},
                estimated_duration=timedelta(seconds=30),
            )
            tasks.append(task)
            print(f"Submitted {task.name} (ID: {task.task_id})")

        # Let the system run for a bit
        print("\n📊 System running - tasks being scheduled...")
        await asyncio.sleep(5)

        # Show system status
        status = system.get_system_status()
        print("\n🖥️ System Status:")
        print(f"  Uptime: {status['system_uptime']}")
        print(f"  Devices: {status['devices']['total_devices']}")
        print(f"  Device Types: {status['devices']['by_type']}")
        print(f"  Tasks Pending: {status['tasks']['pending']}")
        print(f"  Tasks Active: {status['tasks']['active']}")
        print(f"  Tasks Completed: {status['tasks']['completed']}")

        # Simulate task completions
        for task in tasks[:2]:
            if task.assigned_device:
                device = system.device_registry.get_device(task.assigned_device)
                if device:
                    device.complete_task(task.task_id, result=f"Result for {task.name}")
                    system.task_scheduler.handle_task_completion(task.task_id, result=f"Result for {task.name}")

        await asyncio.sleep(2)

        # Final status
        final_status = system.get_system_status()
        print("\n✅ Final Status:")
        print(f"  Tasks Completed: {final_status['tasks']['completed']}")
        print(f"  System Healthy: {final_status['system_healthy']}")

    finally:
        await system.shutdown()
        print("\n🔚 Demo completed")


if __name__ == "__main__":
    asyncio.run(demo_unified_edge_system())
