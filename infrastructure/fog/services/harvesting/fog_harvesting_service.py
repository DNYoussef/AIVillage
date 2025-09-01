"""
Fog Harvesting Service

Manages mobile compute harvesting coordination including:
- Device registration and capability management
- Resource allocation and task assignment
- Harvesting policy enforcement
- Performance monitoring and optimization
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from ..interfaces.base_service import BaseFogService, ServiceStatus, ServiceHealthCheck
from ...compute.harvest_manager import FogHarvestManager, HarvestPolicy, DeviceCapabilities
from ...edge.mobile.resource_manager import MobileResourceManager


class FogHarvestingService(BaseFogService):
    """Service for managing fog computing resource harvesting"""

    def __init__(self, service_name: str, config: Dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)

        # Core components
        self.harvest_manager: Optional[FogHarvestManager] = None
        self.resource_manager: Optional[MobileResourceManager] = None

        # Harvesting configuration
        self.harvest_config = config.get("harvest", {})
        self.node_id = config.get("node_id", "default")

        # Service metrics
        self.metrics = {
            "active_devices": 0,
            "total_tasks_assigned": 0,
            "successful_harvests": 0,
            "failed_harvests": 0,
            "total_compute_hours": 0.0,
            "average_device_utilization": 0.0,
        }

    async def initialize(self) -> bool:
        """Initialize the harvesting service"""
        try:
            # Create harvest policy from configuration
            harvest_policy = HarvestPolicy(
                min_battery_percent=self.harvest_config.get("min_battery_percent", 20),
                max_cpu_temp=self.harvest_config.get("max_thermal_temp", 45.0),
                require_charging=self.harvest_config.get("require_charging", True),
                require_wifi=self.harvest_config.get("require_wifi", True),
            )

            # Initialize harvest manager
            self.harvest_manager = FogHarvestManager(
                node_id=self.node_id,
                policy=harvest_policy,
                token_rate_per_hour=self.harvest_config.get("token_rate_per_hour", 10),
            )

            # Initialize resource manager
            self.resource_manager = MobileResourceManager(
                harvest_enabled=True, token_rewards_enabled=self.config.get("enable_tokens", True)
            )

            # Subscribe to relevant events
            self.subscribe_to_events("device_registered", self._handle_device_registered)
            self.subscribe_to_events("task_completed", self._handle_task_completed)
            self.subscribe_to_events("harvest_session_ended", self._handle_harvest_ended)

            # Start background monitoring tasks
            self.add_background_task(self._monitor_harvesting_performance(), "performance_monitor")
            self.add_background_task(self._optimize_resource_allocation(), "resource_optimizer")

            self.logger.info("Fog harvesting service initialized")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize harvesting service: {e}")
            return False

    async def cleanup(self) -> bool:
        """Cleanup harvesting service resources"""
        try:
            # Stop all active harvesting sessions
            if self.harvest_manager:
                for device_id in list(self.harvest_manager.active_sessions.keys()):
                    await self.harvest_manager.stop_harvesting(device_id, "service_shutdown")

            self.logger.info("Fog harvesting service cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up harvesting service: {e}")
            return False

    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on harvesting service"""
        try:
            error_messages = []

            # Check harvest manager
            if not self.harvest_manager:
                error_messages.append("Harvest manager not initialized")

            # Check resource manager
            if not self.resource_manager:
                error_messages.append("Resource manager not initialized")

            # Check for excessive errors
            error_rate = self.metrics["failed_harvests"] / max(
                1, self.metrics["successful_harvests"] + self.metrics["failed_harvests"]
            )
            if error_rate > 0.1:  # More than 10% error rate
                error_messages.append(f"High error rate: {error_rate:.2%}")

            status = ServiceStatus.RUNNING if not error_messages else ServiceStatus.ERROR

            return ServiceHealthCheck(
                service_name=self.service_name,
                status=status,
                last_check=datetime.now(UTC),
                error_message="; ".join(error_messages) if error_messages else None,
                metrics=self.metrics.copy(),
            )

        except Exception as e:
            return ServiceHealthCheck(
                service_name=self.service_name,
                status=ServiceStatus.ERROR,
                last_check=datetime.now(UTC),
                error_message=f"Health check failed: {e}",
                metrics=self.metrics.copy(),
            )

    async def register_mobile_device(
        self, device_id: str, capabilities: Dict[str, Any], initial_state: Dict[str, Any]
    ) -> bool:
        """Register a mobile device for harvesting"""
        try:
            if not self.harvest_manager:
                self.logger.error("Harvest manager not initialized")
                return False

            # Convert capabilities to DeviceCapabilities object
            device_caps = DeviceCapabilities(
                device_id=device_id,
                device_type=capabilities.get("device_type", "smartphone"),
                cpu_cores=capabilities.get("cpu_cores", 4),
                cpu_freq_mhz=capabilities.get("cpu_freq_mhz", 2000),
                cpu_architecture=capabilities.get("cpu_architecture", "arm64"),
                ram_total_mb=capabilities.get("ram_total_mb", 4096),
                ram_available_mb=capabilities.get("ram_available_mb", 2048),
                storage_total_gb=capabilities.get("storage_total_gb", 64),
                storage_available_gb=capabilities.get("storage_available_gb", 32),
                has_gpu=capabilities.get("has_gpu", False),
                network_speed_mbps=capabilities.get("network_speed_mbps", 50.0),
            )

            # Register with harvest manager
            success = await self.harvest_manager.register_device(device_id, device_caps, initial_state)

            if success:
                # Publish device registration event
                await self.publish_event(
                    "device_registered",
                    {"device_id": device_id, "capabilities": capabilities, "timestamp": datetime.now(UTC).isoformat()},
                )

                self.logger.info(f"Successfully registered device: {device_id}")

            return success

        except Exception as e:
            self.logger.error(f"Failed to register device {device_id}: {e}")
            return False

    async def assign_compute_task(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Assign a compute task to an available device"""
        try:
            if not self.harvest_manager:
                return None

            assigned_device = await self.harvest_manager.assign_task(request_data)

            if assigned_device:
                self.metrics["total_tasks_assigned"] += 1

                # Publish task assignment event
                await self.publish_event(
                    "task_assigned",
                    {
                        "task_id": request_data.get("task_id"),
                        "device_id": assigned_device,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

            return assigned_device

        except Exception as e:
            self.logger.error(f"Failed to assign compute task: {e}")
            return None

    async def get_harvesting_stats(self) -> Dict[str, Any]:
        """Get comprehensive harvesting statistics"""
        try:
            stats = self.metrics.copy()

            if self.harvest_manager:
                harvest_stats = await self.harvest_manager.get_network_stats()
                stats.update(
                    {
                        "harvest_manager_stats": harvest_stats,
                        "active_sessions": len(self.harvest_manager.active_sessions),
                        "registered_devices": len(getattr(self.harvest_manager, "devices", {})),
                    }
                )

            if self.resource_manager:
                # Get resource manager stats if available
                stats["resource_manager_active"] = True

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get harvesting stats: {e}")
            return self.metrics.copy()

    async def _handle_device_registered(self, event):
        """Handle device registration events"""
        device_id = event.data.get("device_id")
        if device_id:
            self.metrics["active_devices"] += 1
            self.logger.debug(f"Device registered: {device_id}")

    async def _handle_task_completed(self, event):
        """Handle task completion events"""
        success = event.data.get("success", False)
        if success:
            self.metrics["successful_harvests"] += 1
        else:
            self.metrics["failed_harvests"] += 1

        compute_time = event.data.get("compute_time_hours", 0.0)
        self.metrics["total_compute_hours"] += compute_time

    async def _handle_harvest_ended(self, event):
        """Handle harvest session end events"""
        device_id = event.data.get("device_id")
        reason = event.data.get("reason")

        self.logger.debug(f"Harvest session ended for {device_id}: {reason}")

        # Update device utilization metrics
        await self._update_device_utilization_metrics()

    async def _monitor_harvesting_performance(self):
        """Background task to monitor harvesting performance"""
        while not self._shutdown_event.is_set():
            try:
                # Collect performance metrics
                if self.harvest_manager:
                    stats = await self.harvest_manager.get_network_stats()
                    self.metrics["active_devices"] = stats.get("active_devices", 0)

                # Check for performance issues
                if self.metrics["active_devices"] == 0:
                    self.logger.warning("No active devices available for harvesting")

                error_rate = self.metrics["failed_harvests"] / max(
                    1, self.metrics["successful_harvests"] + self.metrics["failed_harvests"]
                )
                if error_rate > 0.05:  # More than 5% error rate
                    self.logger.warning(f"High harvest error rate: {error_rate:.2%}")

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(30)

    async def _optimize_resource_allocation(self):
        """Background task to optimize resource allocation"""
        while not self._shutdown_event.is_set():
            try:
                # Analyze device performance and adjust allocation
                if self.harvest_manager and hasattr(self.harvest_manager, "devices"):
                    for device_id, device_info in self.harvest_manager.devices.items():
                        # Check device performance metrics
                        # Implement allocation optimization logic here
                        pass

                await asyncio.sleep(300)  # Optimize every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Resource optimization error: {e}")
                await asyncio.sleep(60)

    async def _update_device_utilization_metrics(self):
        """Update device utilization metrics"""
        try:
            if self.harvest_manager and hasattr(self.harvest_manager, "devices"):
                device_count = len(self.harvest_manager.devices)

                if device_count > 0:
                    # Calculate average utilization based on active sessions
                    active_count = len(self.harvest_manager.active_sessions)
                    self.metrics["average_device_utilization"] = active_count / device_count

        except Exception as e:
            self.logger.error(f"Failed to update utilization metrics: {e}")
