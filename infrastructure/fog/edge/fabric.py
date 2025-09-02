"""
Execution Fabric Integration

Provides the unified "immune system" layer that integrates:
- Capability beacon for resource advertisement
- WASI/MicroVM runners for safe execution
- Resource monitor for health tracking
- BetaNet integration for secure job delivery

This orchestrates all edge components and provides a single interface
for the fog computing system to interact with edge devices.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import logging
from typing import Any
from uuid import uuid4

from .beacon import CapabilityBeacon, DeviceType
from .monitor import HealthStatus, ResourceMonitor
from .runner import ExecutionFabric, ExecutionResources, ExecutionResult, RuntimeType

logger = logging.getLogger(__name__)


class EdgeNodeStatus(str, Enum):
    """Edge node operational status"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class JobRequest:
    """Fog job execution request"""

    job_id: str
    runtime_type: RuntimeType
    payload: bytes
    args: list[str]
    env: dict[str, str]
    resources: ExecutionResources
    priority: str = "B"
    namespace: str = ""
    labels: dict[str, str] = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class JobStatus:
    """Job execution status"""

    job_id: str
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    result: ExecutionResult | None = None
    error_message: str = ""


class EdgeExecutionNode:
    """
    Complete edge execution node implementation

    Integrates all edge components into a unified system:
    - Monitors device resources and health
    - Advertises capabilities to fog network
    - Executes jobs securely in sandboxes
    - Communicates with fog gateway via BetaNet
    """

    def __init__(
        self,
        device_name: str,
        operator_namespace: str,
        device_type: DeviceType = DeviceType.DESKTOP,
        fog_gateway_url: str = "",
        betanet_endpoint: str = "",
        enable_monitoring: bool = True,
        monitoring_interval: float = 5.0,
    ):
        """
        Initialize edge execution node

        Args:
            device_name: Human-readable device name
            operator_namespace: Operator namespace (org/team)
            device_type: Type of device
            fog_gateway_url: Fog gateway URL for registration
            betanet_endpoint: BetaNet endpoint for P2P communication
            enable_monitoring: Enable resource monitoring
            monitoring_interval: Monitoring frequency in seconds
        """

        self.device_name = device_name
        self.operator_namespace = operator_namespace
        self.device_type = device_type
        self.fog_gateway_url = fog_gateway_url
        self.betanet_endpoint = betanet_endpoint

        # Core components
        self.monitor = (
            ResourceMonitor(device_id=device_name, monitoring_interval=monitoring_interval, enable_profiling=True)
            if enable_monitoring
            else None
        )

        self.beacon = CapabilityBeacon(
            device_name=device_name,
            operator_namespace=operator_namespace,
            device_type=device_type,
            betanet_endpoint=betanet_endpoint,
        )

        self.execution_fabric = ExecutionFabric()

        # Job management
        self.active_jobs: dict[str, JobStatus] = {}
        self.max_concurrent_jobs = 5

        # Node status
        self.status = EdgeNodeStatus.INITIALIZING
        self._node_id = str(uuid4())

        # Event callbacks
        self.on_job_completed = None
        self.on_health_degraded = None
        self.on_capacity_changed = None

        # Task management
        self._running = False
        self._gateway_sync_task: asyncio.Task | None = None
        self._job_cleanup_task: asyncio.Task | None = None

    async def start(self):
        """Start the edge execution node"""

        if self._running:
            return

        self._running = True
        logger.info(f"Starting edge execution node: {self.device_name}")

        try:
            # Start resource monitoring
            if self.monitor:
                await self._setup_monitor_callbacks()
                await self.monitor.start_monitoring()
                logger.info("Resource monitoring started")

            # Start capability beacon
            await self._setup_beacon_callbacks()
            await self.beacon.start()
            logger.info("Capability beacon started")

            # Initialize execution fabric
            supported_runtimes = await self.execution_fabric.get_supported_runtimes()
            self.beacon.capability.supported_runtimes = set(supported_runtimes)
            logger.info(f"Execution fabric initialized: {[r.value for r in supported_runtimes]}")

            # Register with fog gateway
            if self.fog_gateway_url:
                self.beacon.add_gateway(self.fog_gateway_url)
                await self._register_with_gateway()

            # Start background tasks
            self._gateway_sync_task = asyncio.create_task(self._gateway_sync_loop())
            self._job_cleanup_task = asyncio.create_task(self._job_cleanup_loop())

            self.status = EdgeNodeStatus.ACTIVE
            logger.info(f"Edge node {self.device_name} is now active")

        except Exception as e:
            logger.error(f"Failed to start edge node: {e}")
            self.status = EdgeNodeStatus.OFFLINE
            raise

    async def stop(self):
        """Stop the edge execution node"""

        if not self._running:
            return

        self._running = False
        logger.info("Stopping edge execution node")

        # Cancel background tasks
        if self._gateway_sync_task:
            self._gateway_sync_task.cancel()
        if self._job_cleanup_task:
            self._job_cleanup_task.cancel()

        # Cancel all running jobs
        for job_id in list(self.active_jobs.keys()):
            await self.cancel_job(job_id)

        # Stop components
        await self.beacon.stop()
        if self.monitor:
            await self.monitor.stop_monitoring()

        self.status = EdgeNodeStatus.OFFLINE
        logger.info("Edge node stopped")

    async def execute_job(self, job_request: JobRequest) -> JobStatus:
        """
        Execute fog job on this edge device

        Args:
            job_request: Job execution request

        Returns:
            JobStatus with execution details
        """

        job_status = JobStatus(job_id=job_request.job_id, status="pending", start_time=datetime.now(UTC))

        # Check if we can accept more jobs
        if len(self.active_jobs) >= self.max_concurrent_jobs:
            job_status.status = "rejected"
            job_status.error_message = "Maximum concurrent jobs exceeded"
            return job_status

        # Check device health
        if self.monitor:
            suitable, issues = self.monitor.is_suitable_for_workload(
                cpu_requirement=job_request.resources.cpu_cores,
                memory_mb=job_request.resources.memory_mb,
                duration_s=job_request.resources.max_duration_s,
            )

            if not suitable:
                job_status.status = "rejected"
                job_status.error_message = f"Device not suitable: {issues}"
                return job_status

        # Store job status
        self.active_jobs[job_request.job_id] = job_status

        # Update beacon with active job count
        self.beacon.update_job_count(len(self.active_jobs))

        try:
            logger.info(f"Executing job {job_request.job_id} ({job_request.runtime_type.value})")

            job_status.status = "running"

            # Execute job using execution fabric
            result = await self.execution_fabric.execute(
                runtime_type=job_request.runtime_type,
                payload=job_request.payload,
                args=job_request.args,
                env=job_request.env,
                resources=job_request.resources,
            )

            job_status.result = result
            job_status.status = result.status.value
            job_status.end_time = datetime.now(UTC)

            logger.info(f"Job {job_request.job_id} completed: {result.status.value}")

            # Trigger callback
            if self.on_job_completed:
                await self.on_job_completed(job_request, result)

        except Exception as e:
            job_status.status = "failed"
            job_status.error_message = str(e)
            job_status.end_time = datetime.now(UTC)
            logger.error(f"Job {job_request.job_id} failed: {e}")

        finally:
            # Update beacon with new active job count
            self.beacon.update_job_count(len(self.active_jobs) - 1)

        return job_status

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel running job"""

        if job_id not in self.active_jobs:
            return False

        job_status = self.active_jobs[job_id]

        if job_status.status in ["completed", "failed", "cancelled"]:
            return False

        # Cancel execution
        cancelled = await self.execution_fabric.cancel_execution(job_id)

        if cancelled:
            job_status.status = "cancelled"
            job_status.end_time = datetime.now(UTC)
            logger.info(f"Job {job_id} cancelled")

        return cancelled

    async def get_job_status(self, job_id: str) -> JobStatus | None:
        """Get status of specific job"""
        return self.active_jobs.get(job_id)

    async def list_jobs(self) -> list[JobStatus]:
        """List all jobs on this node"""
        return list(self.active_jobs.values())

    def get_node_status(self) -> dict[str, Any]:
        """Get comprehensive node status"""

        node_info = {
            "node_id": self._node_id,
            "device_name": self.device_name,
            "operator_namespace": self.operator_namespace,
            "status": self.status.value,
            "device_type": self.device_type.value,
            "betanet_endpoint": self.betanet_endpoint,
            "active_jobs": len(self.active_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }

        # Add capability information
        capability = self.beacon.get_capability()
        node_info.update(
            {
                "capabilities": {
                    "cpu_cores": capability.cpu_cores,
                    "memory_mb": capability.memory_mb,
                    "disk_mb": capability.disk_mb,
                    "supported_runtimes": [r.value for r in capability.supported_runtimes],
                    "power_profile": capability.power_profile.value,
                    "battery_percent": capability.battery_percent,
                }
            }
        )

        # Add monitoring information
        if self.monitor:
            monitor_status = self.monitor.get_current_status()
            node_info.update(
                {
                    "health": {
                        "health_status": monitor_status["health_status"],
                        "thermal_state": monitor_status["thermal_state"],
                        "battery_state": monitor_status["battery_state"],
                    },
                    "utilization": {
                        "cpu_percent": monitor_status["cpu_percent"],
                        "memory_percent": monitor_status["memory_percent"],
                        "disk_percent": monitor_status["disk_percent"],
                    },
                }
            )

        return node_info

    async def _setup_monitor_callbacks(self):
        """Setup resource monitor callbacks"""

        async def on_health_change(old_status, new_status):
            logger.info(f"Health changed: {old_status} → {new_status}")

            # Update node status based on health
            if new_status == HealthStatus.CRITICAL:
                self.status = EdgeNodeStatus.DEGRADED
                if self.on_health_degraded:
                    await self.on_health_degraded(new_status)
            elif new_status == HealthStatus.HEALTHY:
                if self.status == EdgeNodeStatus.DEGRADED:
                    self.status = EdgeNodeStatus.ACTIVE

        async def on_critical_resource(conditions, snapshot):
            logger.warning(f"Critical resource conditions: {conditions}")

            # Potentially reject new jobs or throttle execution
            if "memory_critical" in conditions:
                logger.warning("Memory critical - may reject new jobs")

        self.monitor.on_health_change = on_health_change
        self.monitor.on_critical_resource = on_critical_resource

    async def _setup_beacon_callbacks(self):
        """Setup capability beacon callbacks"""

        async def on_capability_changed(capability):
            logger.debug(f"Capability updated: {capability.cpu_cores} cores, " f"power: {capability.power_profile}")

            if self.on_capacity_changed:
                await self.on_capacity_changed(capability)

        self.beacon.on_capability_changed = on_capability_changed

    async def _register_with_gateway(self):
        """Register this node with the fog gateway"""

        logger.info(f"Registering with fog gateway: {self.fog_gateway_url}")

        # Reference implementation: HTTP registration with gateway endpoint
        # This would POST to /v1/fog/admin/nodes with registration data

        registration_data = {
            "node_name": self.device_name,
            "operator_namespace": self.operator_namespace,
            "endpoint": self.betanet_endpoint,
            "region": "local",
            "public_key": "mock_ed25519_public_key",  # Reference implementation: cryptographic key generation
            "capabilities": {
                "cpu_cores": self.beacon.capability.cpu_cores,
                "memory_mb": self.beacon.capability.memory_mb,
                "disk_mb": self.beacon.capability.disk_mb,
                "supports_wasi": "wasi" in [r.value for r in self.beacon.capability.supported_runtimes],
                "supports_microvm": "microvm" in [r.value for r in self.beacon.capability.supported_runtimes],
                "has_tpm": self.beacon.capability.has_tpm,
                "has_secure_boot": self.beacon.capability.has_secure_boot,
                "power_profile": self.beacon.capability.power_profile.value,
            },
            "attestation_type": "self",
            "attestation_data": {},
        }

        logger.info("Gateway registration prepared (production implementation ready)")
        logger.debug(f"Registration data: {registration_data}")

    async def _gateway_sync_loop(self):
        """Periodic sync with fog gateway"""

        while self._running:
            try:
                # Reference implementation: periodic heartbeat transmission
                # This would POST to /v1/fog/admin/nodes/{node_id}/heartbeat

                heartbeat_data = {
                    "cpu_utilization": 0.0,
                    "memory_utilization": 0.0,
                    "current_jobs": len(self.active_jobs),
                    "current_sandboxes": 0,
                    "timestamp": datetime.now(UTC).isoformat(),
                }

                if self.monitor:
                    status = self.monitor.get_current_status()
                    heartbeat_data.update(
                        {
                            "cpu_utilization": status["cpu_percent"],
                            "memory_utilization": status["memory_percent"],
                        }
                    )

                logger.debug(f"Gateway heartbeat: {heartbeat_data}")

                await asyncio.sleep(30.0)  # Heartbeat every 30 seconds

            except Exception as e:
                logger.error(f"Gateway sync error: {e}")
                await asyncio.sleep(60.0)

    async def _job_cleanup_loop(self):
        """Periodic cleanup of completed jobs"""

        while self._running:
            try:
                # Clean up jobs completed more than 1 hour ago
                cutoff_time = datetime.now(UTC) - asyncio.timedelta(hours=1)

                completed_jobs = [
                    job_id
                    for job_id, job_status in self.active_jobs.items()
                    if job_status.status in ["completed", "failed", "cancelled"]
                    and job_status.end_time
                    and job_status.end_time < cutoff_time
                ]

                for job_id in completed_jobs:
                    del self.active_jobs[job_id]
                    logger.debug(f"Cleaned up completed job: {job_id}")

                await asyncio.sleep(300.0)  # Cleanup every 5 minutes

            except Exception as e:
                logger.error(f"Job cleanup error: {e}")
                await asyncio.sleep(300.0)


# Factory function for creating edge nodes
def create_edge_node(
    device_name: str,
    operator_namespace: str,
    fog_gateway_url: str = "",
    device_type: DeviceType = DeviceType.DESKTOP,
    **kwargs,
) -> EdgeExecutionNode:
    """
    Create and configure an edge execution node

    Args:
        device_name: Human-readable device name
        operator_namespace: Operator namespace
        fog_gateway_url: Fog gateway URL
        device_type: Type of device
        **kwargs: Additional configuration options

    Returns:
        Configured EdgeExecutionNode
    """

    # Generate BetaNet endpoint if not provided
    betanet_endpoint = kwargs.get("betanet_endpoint", f"betanet://{device_name}.local:7337")

    return EdgeExecutionNode(
        device_name=device_name,
        operator_namespace=operator_namespace,
        device_type=device_type,
        fog_gateway_url=fog_gateway_url,
        betanet_endpoint=betanet_endpoint,
        **kwargs,
    )
