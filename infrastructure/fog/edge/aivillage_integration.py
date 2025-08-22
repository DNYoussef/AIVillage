"""
AIVillage Edge Integration Bridge

Integrates the fog computing edge components with the existing AIVillage edge infrastructure:
- Connects with existing EdgeManager and FogCoordinator
- Integrates with P2P transport system (BitChat/BetaNet)
- Reuses existing device profiling and resource management
- Maintains compatibility with digital twin and federated learning systems

This ensures the fog computing layer builds on top of existing AIVillage infrastructure
rather than duplicating functionality.
"""

import asyncio
import logging
from typing import Any

# Import existing AIVillage edge infrastructure
try:
    from packages.edge.bridges.p2p_integration import EdgeP2PBridge
    from packages.edge.core.edge_manager import EdgeManager, EdgeState
    from packages.edge.fog_compute.fog_coordinator import FogCoordinator
    from packages.edge.mobile.resource_manager import BatteryThermalResourceManager

    AIVILLAGE_EDGE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AIVillage edge infrastructure not available: {e}")
    AIVILLAGE_EDGE_AVAILABLE = False

# Import our fog edge components
from .beacon import DeviceType as FogDeviceType
from .beacon import EdgeCapability
from .fabric import EdgeExecutionNode, JobRequest, JobStatus

logger = logging.getLogger(__name__)


class AIVillageEdgeIntegration:
    """
    Integration bridge between fog computing components and AIVillage edge infrastructure

    This class acts as an adapter that:
    1. Connects fog edge nodes with existing EdgeManager
    2. Integrates capability beacon with existing device registry
    3. Coordinates with existing fog computing infrastructure
    4. Maintains compatibility with P2P transport system
    """

    def __init__(
        self,
        device_name: str,
        operator_namespace: str,
        fog_gateway_url: str = "",
        use_existing_infrastructure: bool = True,
    ):
        """
        Initialize integration bridge

        Args:
            device_name: Device identifier
            operator_namespace: Operator namespace
            fog_gateway_url: Fog gateway URL
            use_existing_infrastructure: Whether to use existing AIVillage infrastructure
        """

        self.device_name = device_name
        self.operator_namespace = operator_namespace
        self.fog_gateway_url = fog_gateway_url
        self.use_existing_infrastructure = use_existing_infrastructure and AIVILLAGE_EDGE_AVAILABLE

        # Fog computing components
        self.fog_edge_node: EdgeExecutionNode | None = None

        # AIVillage infrastructure components (if available)
        self.edge_manager: EdgeManager | None = None
        self.fog_coordinator: FogCoordinator | None = None
        self.resource_manager: BatteryThermalResourceManager | None = None
        self.p2p_bridge: EdgeP2PBridge | None = None

        # Integration state
        self.is_integrated = False
        self._integration_tasks: list[asyncio.Task] = []

    async def initialize(self):
        """Initialize the integrated edge system"""

        logger.info(f"Initializing AIVillage edge integration for {self.device_name}")

        if self.use_existing_infrastructure:
            await self._initialize_aivillage_infrastructure()

        await self._initialize_fog_components()
        await self._setup_integration_bridges()

        self.is_integrated = True
        logger.info("Edge integration completed successfully")

    async def _initialize_aivillage_infrastructure(self):
        """Initialize existing AIVillage edge infrastructure"""

        if not AIVILLAGE_EDGE_AVAILABLE:
            logger.warning("AIVillage edge infrastructure not available")
            return

        try:
            # Initialize EdgeManager
            self.edge_manager = EdgeManager()
            await self.edge_manager.initialize()

            # Initialize FogCoordinator
            self.fog_coordinator = FogCoordinator()
            await self.fog_coordinator.start()

            # Initialize resource manager
            self.resource_manager = BatteryThermalResourceManager()

            # Initialize P2P bridge
            if self.edge_manager:
                self.p2p_bridge = EdgeP2PBridge(self.edge_manager)
                await self.p2p_bridge.initialize()

            logger.info("AIVillage edge infrastructure initialized")

        except Exception as e:
            logger.error(f"Failed to initialize AIVillage infrastructure: {e}")
            self.use_existing_infrastructure = False

    async def _initialize_fog_components(self):
        """Initialize fog computing components"""

        # Create fog edge node
        self.fog_edge_node = EdgeExecutionNode(
            device_name=self.device_name,
            operator_namespace=self.operator_namespace,
            fog_gateway_url=self.fog_gateway_url,
            device_type=self._map_device_type(),
            betanet_endpoint=f"betanet://{self.device_name}.local:7337",
        )

        # Set up callbacks for integration
        if self.fog_edge_node:
            self.fog_edge_node.on_job_completed = self._on_fog_job_completed
            self.fog_edge_node.on_health_degraded = self._on_device_health_degraded
            self.fog_edge_node.on_capacity_changed = self._on_capacity_changed

        # Start fog edge node
        await self.fog_edge_node.start()

        logger.info("Fog computing components initialized")

    async def _setup_integration_bridges(self):
        """Set up integration bridges between systems"""

        if not self.use_existing_infrastructure or not self.fog_edge_node:
            return

        try:
            # Register device with existing EdgeManager
            if self.edge_manager:
                await self._register_with_edge_manager()

            # Integrate with existing fog coordinator
            if self.fog_coordinator:
                await self._integrate_with_fog_coordinator()

            # Bridge resource monitoring
            if self.resource_manager:
                await self._bridge_resource_monitoring()

            # Set up P2P integration
            if self.p2p_bridge:
                await self._setup_p2p_integration()

            logger.info("Integration bridges established")

        except Exception as e:
            logger.error(f"Failed to setup integration bridges: {e}")

    async def _register_with_edge_manager(self):
        """Register fog device with existing EdgeManager"""

        if not self.edge_manager or not self.fog_edge_node:
            return

        # Create EdgeDevice from fog capability
        capability = self.fog_edge_node.beacon.get_capability()

        # Map fog device to AIVillage EdgeDevice format
        edge_device_data = {
            "device_id": capability.device_id,
            "device_name": capability.device_name,
            "device_type": capability.device_type.value,
            "capabilities": {
                "cpu_cores": capability.cpu_cores,
                "memory_mb": capability.memory_mb,
                "disk_mb": capability.disk_mb,
                "battery_level": capability.battery_percent,
                "is_charging": capability.power_profile.value == "charging",
            },
            "network_info": {
                "endpoint": capability.endpoint,
                "private_ip": capability.private_ip,
                "region": capability.region,
            },
        }

        # Register with EdgeManager
        try:
            await self.edge_manager.register_device(edge_device_data)
            logger.info(f"Device {self.device_name} registered with EdgeManager")
        except Exception as e:
            logger.error(f"Failed to register with EdgeManager: {e}")

    async def _integrate_with_fog_coordinator(self):
        """Integrate with existing FogCoordinator"""

        if not self.fog_coordinator or not self.fog_edge_node:
            return

        # Get node status for fog coordinator
        node_status = self.fog_edge_node.get_node_status()

        # Register node with fog coordinator
        try:
            await self.fog_coordinator.register_node(
                {
                    "node_id": node_status["node_id"],
                    "capabilities": node_status["capabilities"],
                    "health": node_status.get("health", {}),
                    "endpoint": node_status.get("betanet_endpoint", ""),
                }
            )

            logger.info("Node integrated with FogCoordinator")
        except Exception as e:
            logger.error(f"Failed to integrate with FogCoordinator: {e}")

    async def _bridge_resource_monitoring(self):
        """Bridge resource monitoring between systems"""

        if not self.resource_manager or not self.fog_edge_node:
            return

        # Set up monitoring bridge task
        bridge_task = asyncio.create_task(self._resource_monitoring_bridge())
        self._integration_tasks.append(bridge_task)

    async def _resource_monitoring_bridge(self):
        """Bridge resource monitoring data between systems"""

        while self.is_integrated:
            try:
                if self.resource_manager and self.fog_edge_node and self.fog_edge_node.monitor:
                    # Get status from fog monitor
                    fog_status = self.fog_edge_node.monitor.get_current_status()

                    # Create mock device profile for resource manager
                    device_profile = type(
                        "DeviceProfile",
                        (),
                        {
                            "battery_percent": fog_status.get("battery_percent", 100),
                            "cpu_temp_celsius": fog_status.get("cpu_temp_celsius", 50),
                            "network_type": "wifi",
                            "is_charging": fog_status.get("battery_state") == "charging",
                            "memory_pressure": fog_status.get("memory_percent", 0) > 80,
                        },
                    )()

                    # Update resource manager with current state
                    state = await self.resource_manager.evaluate_and_adapt(device_profile)

                    # Apply resource policies to fog node
                    if state.transport_preference.value == "bitchat_only":
                        # Prefer BitChat transport for battery savings
                        self.fog_edge_node.beacon.capability.power_profile = "battery_saver"
                    elif state.power_mode.value == "performance":
                        self.fog_edge_node.beacon.capability.power_profile = "performance"

                await asyncio.sleep(10.0)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Resource monitoring bridge error: {e}")
                await asyncio.sleep(30.0)

    async def _setup_p2p_integration(self):
        """Set up P2P transport integration"""

        if not self.p2p_bridge or not self.fog_edge_node:
            return

        # Map fog edge node to P2P bridge
        try:
            # Register device with P2P bridge
            device_info = {
                "device_id": self.fog_edge_node._node_id,
                "device_name": self.device_name,
                "transport_preferences": ["betanet", "bitchat"],
                "capabilities": self.fog_edge_node.get_node_status()["capabilities"],
            }

            await self.p2p_bridge.register_device(device_info)
            logger.info("P2P integration established")

        except Exception as e:
            logger.error(f"Failed to setup P2P integration: {e}")

    def _map_device_type(self) -> FogDeviceType:
        """Map AIVillage device type to fog device type"""

        # Try to detect device type
        import platform

        system = platform.system().lower()

        if system == "android":
            return FogDeviceType.MOBILE_PHONE
        elif system == "ios":
            return FogDeviceType.MOBILE_PHONE
        elif system == "darwin":
            return FogDeviceType.LAPTOP
        elif system == "windows":
            return FogDeviceType.DESKTOP
        elif system == "linux":
            # Could be desktop, laptop, or embedded
            return FogDeviceType.DESKTOP
        else:
            return FogDeviceType.DESKTOP

    # Event handlers for fog edge node

    async def _on_fog_job_completed(self, job_request: JobRequest, result):
        """Handle fog job completion"""

        logger.info(f"Fog job {job_request.job_id} completed: {result.status}")

        # Forward to fog coordinator if available
        if self.fog_coordinator:
            try:
                await self.fog_coordinator.report_task_completion(
                    {
                        "task_id": job_request.job_id,
                        "status": result.status.value,
                        "execution_time": result.duration_ms / 1000.0,
                        "resource_usage": {
                            "cpu_time": result.cpu_time_s,
                            "memory_peak": result.memory_peak_mb,
                            "disk_used": result.disk_used_mb,
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Failed to report to fog coordinator: {e}")

    async def _on_device_health_degraded(self, health_status):
        """Handle device health degradation"""

        logger.warning(f"Device health degraded: {health_status}")

        # Notify edge manager if available
        if self.edge_manager:
            try:
                await self.edge_manager.update_device_status(self.device_name, EdgeState.DEGRADED)
            except Exception as e:
                logger.error(f"Failed to update edge manager: {e}")

    async def _on_capacity_changed(self, capability: EdgeCapability):
        """Handle capacity changes"""

        logger.info(f"Device capacity changed: {capability.cpu_cores} cores")

        # Update fog coordinator if available
        if self.fog_coordinator:
            try:
                await self.fog_coordinator.update_node_capacity(
                    self.fog_edge_node._node_id,
                    {
                        "cpu_cores": capability.cpu_cores,
                        "memory_mb": capability.memory_mb,
                        "disk_mb": capability.disk_mb,
                        "battery_percent": capability.battery_percent,
                    },
                )
            except Exception as e:
                logger.error(f"Failed to update fog coordinator: {e}")

    # Public interface methods

    async def execute_job(self, job_request: JobRequest) -> JobStatus:
        """Execute fog job using integrated system"""

        if not self.fog_edge_node:
            raise RuntimeError("Fog edge node not initialized")

        return await self.fog_edge_node.execute_job(job_request)

    async def get_device_status(self) -> dict[str, Any]:
        """Get comprehensive device status"""

        status = {}

        if self.fog_edge_node:
            status.update(self.fog_edge_node.get_node_status())

        # Add AIVillage infrastructure status if available
        if self.use_existing_infrastructure:
            status["aivillage_integration"] = {
                "edge_manager_connected": self.edge_manager is not None,
                "fog_coordinator_connected": self.fog_coordinator is not None,
                "p2p_bridge_connected": self.p2p_bridge is not None,
                "resource_manager_active": self.resource_manager is not None,
            }

        return status

    async def shutdown(self):
        """Shutdown integrated edge system"""

        logger.info("Shutting down integrated edge system")

        self.is_integrated = False

        # Cancel integration tasks
        for task in self._integration_tasks:
            task.cancel()

        # Shutdown fog components
        if self.fog_edge_node:
            await self.fog_edge_node.stop()

        # Shutdown AIVillage components
        if self.p2p_bridge:
            await self.p2p_bridge.cleanup()

        if self.fog_coordinator:
            await self.fog_coordinator.stop()

        if self.edge_manager:
            await self.edge_manager.cleanup()

        logger.info("Edge integration shutdown complete")


# Factory function for easy creation
async def create_integrated_edge_node(
    device_name: str, operator_namespace: str, fog_gateway_url: str = "", **kwargs
) -> AIVillageEdgeIntegration:
    """
    Create an integrated edge node that works with existing AIVillage infrastructure

    Args:
        device_name: Device identifier
        operator_namespace: Operator namespace
        fog_gateway_url: Fog gateway URL
        **kwargs: Additional configuration

    Returns:
        Initialized AIVillageEdgeIntegration
    """

    integration = AIVillageEdgeIntegration(
        device_name=device_name, operator_namespace=operator_namespace, fog_gateway_url=fog_gateway_url, **kwargs
    )

    await integration.initialize()
    return integration
