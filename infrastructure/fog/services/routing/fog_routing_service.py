"""
Fog Routing Service

Manages onion routing and privacy layer including:
- Circuit creation and management
- Hidden service hosting
- Privacy-aware task routing
- Mixnet integration
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, UTC

from ..interfaces.base_service import BaseFogService, ServiceStatus, ServiceHealthCheck
from ...privacy.onion_routing import OnionRouter, NodeType
from ...integration.fog_onion_coordinator import FogOnionCoordinator, PrivacyLevel, PrivacyAwareTask


class FogRoutingService(BaseFogService):
    """Service for managing fog computing routing and privacy"""

    def __init__(self, service_name: str, config: Dict[str, Any], event_bus):
        super().__init__(service_name, config, event_bus)

        # Core components
        self.onion_router: Optional[OnionRouter] = None
        self.onion_coordinator: Optional[FogOnionCoordinator] = None

        # Routing configuration
        self.onion_config = config.get("onion", {})
        self.node_id = config.get("node_id", "default")

        # Service metrics
        self.metrics = {
            "active_circuits": 0,
            "hidden_services": 0,
            "circuits_created": 0,
            "circuits_failed": 0,
            "privacy_tasks_processed": 0,
            "mixnet_messages_sent": 0,
            "circuit_rotation_count": 0,
        }

    async def initialize(self) -> bool:
        """Initialize the routing service"""
        try:
            # Determine node types based on configuration
            node_types = {NodeType.MIDDLE}  # All nodes are middle relays by default

            if self.onion_config.get("enable_exit", False):
                node_types.add(NodeType.EXIT)

            if self.onion_config.get("enable_guard", True):
                node_types.add(NodeType.GUARD)

            # Initialize onion router
            self.onion_router = OnionRouter(
                node_id=self.node_id,
                node_types=node_types,
                enable_hidden_services=self.onion_config.get("enable_hidden_services", True),
                num_guards=self.onion_config.get("num_guards", 3),
                circuit_lifetime_hours=self.onion_config.get("circuit_lifetime_hours", 1),
            )

            # Fetch network consensus
            await self.onion_router.fetch_consensus()

            # Initialize onion coordinator
            self.onion_coordinator = FogOnionCoordinator(
                node_id=f"onion-{self.node_id}",
                fog_coordinator=None,  # Will be injected as dependency
                enable_mixnet=self.onion_config.get("enable_mixnet", True),
                default_privacy_level=PrivacyLevel.PRIVATE,
                max_circuits=self.onion_config.get("max_circuits", 50),
            )

            # Subscribe to relevant events
            self.subscribe_to_events("create_hidden_service", self._handle_create_hidden_service)
            self.subscribe_to_events("submit_privacy_task", self._handle_privacy_task)
            self.subscribe_to_events("circuit_failure", self._handle_circuit_failure)

            # Start background tasks
            self.add_background_task(self._circuit_rotation_task(), "circuit_rotation")
            self.add_background_task(self._monitor_routing_health(), "routing_monitor")

            self.logger.info(f"Fog routing service initialized with node types: {node_types}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize routing service: {e}")
            return False

    async def cleanup(self) -> bool:
        """Cleanup routing service resources"""
        try:
            # Close all circuits
            if self.onion_router:
                for circuit_id in list(self.onion_router.circuits.keys()):
                    await self.onion_router.close_circuit(circuit_id)

            # Stop onion coordinator
            if self.onion_coordinator:
                await self.onion_coordinator.stop()

            self.logger.info("Fog routing service cleaned up")
            return True

        except Exception as e:
            self.logger.error(f"Error cleaning up routing service: {e}")
            return False

    async def health_check(self) -> ServiceHealthCheck:
        """Perform health check on routing service"""
        try:
            error_messages = []

            # Check onion router
            if not self.onion_router:
                error_messages.append("Onion router not initialized")
            else:
                # Check circuit health
                stats = self.onion_router.get_stats()
                active_circuits = stats.get("active_circuits", 0)
                if active_circuits == 0 and self.metrics["circuits_created"] > 0:
                    error_messages.append("No active circuits available")

            # Check onion coordinator
            if not self.onion_coordinator:
                error_messages.append("Onion coordinator not initialized")

            # Check circuit failure rate
            total_circuits = self.metrics["circuits_created"] + self.metrics["circuits_failed"]
            if total_circuits > 0:
                failure_rate = self.metrics["circuits_failed"] / total_circuits
                if failure_rate > 0.2:  # More than 20% failure rate
                    error_messages.append(f"High circuit failure rate: {failure_rate:.2%}")

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

    async def create_hidden_service(self, ports: Dict[int, int], service_type: str = "web") -> Optional[str]:
        """Create a .fog hidden service"""
        try:
            if not self.onion_router:
                self.logger.error("Onion router not initialized")
                return None

            hidden_service = await self.onion_router.create_hidden_service(ports=ports, descriptor_cookie=None)

            if hidden_service:
                self.metrics["hidden_services"] += 1

                # Publish hidden service creation event
                await self.publish_event(
                    "hidden_service_created",
                    {
                        "service_id": hidden_service.service_id,
                        "onion_address": hidden_service.onion_address,
                        "service_type": service_type,
                        "ports": ports,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

                self.logger.info(f"Created hidden service: {hidden_service.onion_address}")
                return hidden_service.onion_address

            return None

        except Exception as e:
            self.logger.error(f"Failed to create hidden service: {e}")
            return None

    async def submit_privacy_aware_task(self, task_data: Dict[str, Any]) -> bool:
        """Submit a privacy-aware task for processing"""
        try:
            if not self.onion_coordinator:
                self.logger.error("Onion coordinator not initialized")
                return False

            privacy_task = PrivacyAwareTask(
                task_id=task_data["task_id"],
                privacy_level=PrivacyLevel[task_data.get("privacy_level", "PRIVATE")],
                task_data=(
                    task_data["task_data"].encode()
                    if isinstance(task_data["task_data"], str)
                    else task_data["task_data"]
                ),
                compute_requirements=task_data.get("compute_requirements", {}),
                client_id=task_data["client_id"],
                require_onion_circuit=task_data.get("require_onion_circuit", True),
                require_mixnet=task_data.get("require_mixnet", False),
                min_circuit_hops=task_data.get("min_circuit_hops", 3),
            )

            success = await self.onion_coordinator.submit_privacy_aware_task(privacy_task)

            if success:
                self.metrics["privacy_tasks_processed"] += 1

                # Publish privacy task event
                await self.publish_event(
                    "privacy_task_submitted",
                    {
                        "task_id": task_data["task_id"],
                        "privacy_level": privacy_task.privacy_level.value,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

            return success

        except Exception as e:
            self.logger.error(f"Failed to submit privacy task: {e}")
            return False

    async def send_private_message(self, recipient_id: str, message: bytes, privacy_level: str = "PRIVATE") -> bool:
        """Send a private message through mixnet"""
        try:
            if not self.onion_coordinator:
                return False

            success = await self.onion_coordinator.send_private_gossip(
                recipient_id=recipient_id,
                message=message,
                privacy_level=PrivacyLevel[privacy_level],
            )

            if success:
                self.metrics["mixnet_messages_sent"] += 1

            return success

        except Exception as e:
            self.logger.error(f"Failed to send private message: {e}")
            return False

    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics"""
        try:
            stats = self.metrics.copy()

            if self.onion_router:
                onion_stats = self.onion_router.get_stats()
                stats.update(
                    {
                        "onion_router_stats": onion_stats,
                        "active_circuits": onion_stats.get("active_circuits", 0),
                        "hidden_services_active": onion_stats.get("hidden_services", 0),
                    }
                )

                # Update metrics from router stats
                self.metrics["active_circuits"] = onion_stats.get("active_circuits", 0)
                self.metrics["hidden_services"] = onion_stats.get("hidden_services", 0)

            if self.onion_coordinator:
                coordinator_stats = await self.onion_coordinator.get_coordinator_stats()
                stats["onion_coordinator_stats"] = coordinator_stats

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get routing stats: {e}")
            return self.metrics.copy()

    async def _handle_create_hidden_service(self, event):
        """Handle hidden service creation requests"""
        ports = event.data.get("ports", {})
        service_type = event.data.get("service_type", "web")

        onion_address = await self.create_hidden_service(ports, service_type)

        if onion_address:
            # Respond with the created service address
            await self.publish_event(
                "hidden_service_response",
                {"request_id": event.data.get("request_id"), "onion_address": onion_address, "success": True},
            )
        else:
            await self.publish_event(
                "hidden_service_response",
                {
                    "request_id": event.data.get("request_id"),
                    "success": False,
                    "error": "Failed to create hidden service",
                },
            )

    async def _handle_privacy_task(self, event):
        """Handle privacy-aware task submissions"""
        success = await self.submit_privacy_aware_task(event.data)

        await self.publish_event("privacy_task_response", {"task_id": event.data.get("task_id"), "success": success})

    async def _handle_circuit_failure(self, event):
        """Handle circuit failure events"""
        circuit_id = event.data.get("circuit_id")
        reason = event.data.get("reason")

        self.metrics["circuits_failed"] += 1
        self.logger.warning(f"Circuit {circuit_id} failed: {reason}")

        # Attempt to create a new circuit to replace the failed one
        if self.onion_router:
            try:
                # This would be implemented based on the specific circuit creation logic
                pass
            except Exception as e:
                self.logger.error(f"Failed to replace failed circuit: {e}")

    async def _circuit_rotation_task(self):
        """Background task to rotate onion circuits"""
        while not self._shutdown_event.is_set():
            try:
                if self.onion_router:
                    rotated = await self.onion_router.rotate_circuits()
                    if rotated > 0:
                        self.metrics["circuit_rotation_count"] += rotated
                        self.logger.debug(f"Rotated {rotated} onion circuits")

                await asyncio.sleep(300)  # Check every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Circuit rotation error: {e}")
                await asyncio.sleep(60)

    async def _monitor_routing_health(self):
        """Background task to monitor routing health"""
        while not self._shutdown_event.is_set():
            try:
                # Monitor circuit health
                if self.onion_router:
                    stats = self.onion_router.get_stats()
                    active_circuits = stats.get("active_circuits", 0)

                    if active_circuits == 0:
                        self.logger.warning("No active circuits - attempting to create new circuits")
                        # Attempt to create circuits if needed

                    # Check for stale circuits
                    # Implementation would check circuit age and performance

                await asyncio.sleep(120)  # Check every 2 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Routing health monitoring error: {e}")
                await asyncio.sleep(60)
