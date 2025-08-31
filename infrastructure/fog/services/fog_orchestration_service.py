"""
Fog Orchestration Service

The foundation service that manages the lifecycle of the entire fog computing system.
Handles system initialization, component coordination, health monitoring, and graceful shutdown.

This service eliminates circular dependencies by using dependency injection and provides
a clean foundation that other fog services depend on.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


class ComponentProtocol(Protocol):
    """Protocol for fog computing components that can be orchestrated."""

    async def start(self) -> bool:
        """Start the component."""
        ...

    async def stop(self) -> None:
        """Stop the component gracefully."""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Get component health status."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        ...


class ServiceDependency:
    """Represents a service dependency relationship."""

    def __init__(self, service_name: str, required: bool = True):
        self.service_name = service_name
        self.required = required


class ComponentRegistry:
    """Registry for fog computing components with dependency management."""

    def __init__(self):
        self._components: Dict[str, ComponentProtocol] = {}
        self._dependencies: Dict[str, List[ServiceDependency]] = {}
        self._initialization_order: List[str] = []
        self._started_components: Set[str] = set()

    def register_component(
        self,
        name: str,
        component: ComponentProtocol,
        dependencies: Optional[List[ServiceDependency]] = None,
    ) -> None:
        """Register a component with its dependencies."""
        self._components[name] = component
        self._dependencies[name] = dependencies or []

    def get_component(self, name: str) -> Optional[ComponentProtocol]:
        """Get a component by name."""
        return self._components.get(name)

    def get_initialization_order(self) -> List[str]:
        """Get the order in which components should be initialized."""
        if not self._initialization_order:
            self._calculate_initialization_order()
        return self._initialization_order

    def _calculate_initialization_order(self) -> None:
        """Calculate component initialization order based on dependencies."""
        visited = set()
        temp_visited = set()
        order = []

        def visit(component_name: str) -> None:
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {component_name}")

            if component_name not in visited:
                temp_visited.add(component_name)

                # Visit dependencies first
                for dep in self._dependencies.get(component_name, []):
                    if dep.service_name in self._components:
                        visit(dep.service_name)

                temp_visited.remove(component_name)
                visited.add(component_name)
                order.append(component_name)

        for component_name in self._components:
            if component_name not in visited:
                visit(component_name)

        self._initialization_order = order

    def mark_started(self, component_name: str) -> None:
        """Mark a component as started."""
        self._started_components.add(component_name)

    def mark_stopped(self, component_name: str) -> None:
        """Mark a component as stopped."""
        self._started_components.discard(component_name)

    def is_started(self, component_name: str) -> bool:
        """Check if a component is started."""
        return component_name in self._started_components

    def get_started_components(self) -> Set[str]:
        """Get set of started components."""
        return self._started_components.copy()


class FogOrchestrationService:
    """
    Foundation orchestration service for the fog computing system.

    Manages system lifecycle, component coordination, health monitoring,
    and configuration distribution across all fog services.
    """

    def __init__(
        self,
        node_id: str,
        config_path: Optional[Path] = None,
        enable_health_monitoring: bool = True,
    ):
        self.node_id = node_id
        self.config_path = config_path
        self.enable_health_monitoring = enable_health_monitoring

        # Core orchestration state
        self.is_running = False
        self.startup_time: Optional[datetime] = None
        self.shutdown_time: Optional[datetime] = None

        # Component management
        self.registry = ComponentRegistry()
        self.background_tasks: Set[asyncio.Task] = set()

        # Configuration and health monitoring
        self.config: Dict[str, Any] = self._load_config()
        self.health_status: Dict[str, Any] = {}
        self.system_stats: Dict[str, Any] = {}

        # Event coordination
        self._event_handlers: Dict[str, List[callable]] = {}
        self._coordination_lock = asyncio.Lock()

        logger.info(f"FogOrchestrationService initialized for node {node_id}")

    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from file or defaults."""
        default_config = {
            "orchestration": {
                "startup_timeout_seconds": 300,
                "shutdown_timeout_seconds": 60,
                "health_check_interval_seconds": 30,
                "stats_collection_interval_seconds": 60,
                "component_retry_attempts": 3,
                "component_retry_delay_seconds": 5,
            },
            "logging": {
                "level": "INFO",
                "enable_component_logs": True,
                "log_retention_days": 7,
            },
            "monitoring": {
                "enable_metrics_collection": True,
                "enable_health_checks": True,
                "alert_on_component_failure": True,
                "performance_tracking": True,
            },
        }

        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    file_config = json.load(f)
                    # Merge configurations, file config takes precedence
                    default_config.update(file_config)
                logger.info(f"Loaded orchestration config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}, using defaults")

        return default_config

    async def start(self) -> bool:
        """Start the fog orchestration service and all registered components."""
        async with self._coordination_lock:
            if self.is_running:
                logger.warning("Orchestration service is already running")
                return True

            try:
                logger.info("Starting fog orchestration service...")
                self.startup_time = datetime.now(UTC)

                # Start components in dependency order
                initialization_order = self.registry.get_initialization_order()
                config = self.config["orchestration"]

                for component_name in initialization_order:
                    success = await self._start_component(
                        component_name,
                        retry_attempts=config["component_retry_attempts"],
                        retry_delay=config["component_retry_delay_seconds"],
                    )

                    if not success:
                        # Check if component is required
                        logger.error(f"Failed to start component: {component_name}")
                        await self._cleanup_started_components()
                        return False

                    self.registry.mark_started(component_name)

                # Start background monitoring tasks
                await self._start_background_tasks()

                self.is_running = True
                await self._emit_event("orchestration_started", {"startup_time": self.startup_time})

                logger.info(f"Fog orchestration service started successfully with {len(initialization_order)} components")
                return True

            except Exception as e:
                logger.error(f"Failed to start orchestration service: {e}")
                await self._cleanup_started_components()
                return False

    async def _start_component(
        self, component_name: str, retry_attempts: int = 3, retry_delay: float = 5.0
    ) -> bool:
        """Start a single component with retry logic."""
        component = self.registry.get_component(component_name)
        if not component:
            logger.error(f"Component not found: {component_name}")
            return False

        for attempt in range(retry_attempts):
            try:
                logger.info(f"Starting component {component_name} (attempt {attempt + 1}/{retry_attempts})")
                success = await component.start()

                if success:
                    logger.info(f"Component {component_name} started successfully")
                    return True
                else:
                    logger.warning(f"Component {component_name} start returned False")

            except Exception as e:
                logger.error(f"Component {component_name} start failed: {e}")

            if attempt < retry_attempts - 1:
                await asyncio.sleep(retry_delay)

        logger.error(f"Failed to start component {component_name} after {retry_attempts} attempts")
        return False

    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        if self.enable_health_monitoring:
            health_task = asyncio.create_task(self._health_monitoring_task())
            self.background_tasks.add(health_task)

        stats_task = asyncio.create_task(self._stats_collection_task())
        self.background_tasks.add(stats_task)

        coordination_task = asyncio.create_task(self._coordination_task())
        self.background_tasks.add(coordination_task)

        logger.info(f"Started {len(self.background_tasks)} background tasks")

    async def stop(self) -> bool:
        """Stop the orchestration service and all components gracefully."""
        async with self._coordination_lock:
            if not self.is_running:
                logger.warning("Orchestration service is not running")
                return True

            try:
                logger.info("Stopping fog orchestration service...")
                self.shutdown_time = datetime.now(UTC)
                self.is_running = False

                # Cancel background tasks
                await self._stop_background_tasks()

                # Stop components in reverse order
                started_components = list(self.registry.get_started_components())
                initialization_order = self.registry.get_initialization_order()

                # Reverse the order for shutdown
                shutdown_order = [comp for comp in reversed(initialization_order) if comp in started_components]

                for component_name in shutdown_order:
                    await self._stop_component(component_name)
                    self.registry.mark_stopped(component_name)

                await self._emit_event("orchestration_stopped", {"shutdown_time": self.shutdown_time})

                logger.info("Fog orchestration service stopped successfully")
                return True

            except Exception as e:
                logger.error(f"Error during orchestration shutdown: {e}")
                return False

    async def _stop_component(self, component_name: str) -> None:
        """Stop a single component."""
        component = self.registry.get_component(component_name)
        if not component:
            return

        try:
            logger.info(f"Stopping component: {component_name}")
            await component.stop()
            logger.info(f"Component {component_name} stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping component {component_name}: {e}")

    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        for task in self.background_tasks:
            task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()

    async def _cleanup_started_components(self) -> None:
        """Clean up any components that were started during failed initialization."""
        started_components = list(self.registry.get_started_components())
        for component_name in reversed(started_components):
            await self._stop_component(component_name)
            self.registry.mark_stopped(component_name)

    async def health_check(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health_result = {
            "orchestration_service": {
                "status": "healthy" if self.is_running else "stopped",
                "uptime_seconds": (
                    (datetime.now(UTC) - self.startup_time).total_seconds()
                    if self.startup_time and self.is_running
                    else 0
                ),
                "node_id": self.node_id,
                "components_running": len(self.registry.get_started_components()),
            },
            "components": {},
            "overall_status": "healthy",
        }

        unhealthy_components = 0

        # Check health of all started components
        for component_name in self.registry.get_started_components():
            component = self.registry.get_component(component_name)
            if component:
                try:
                    component_health = await component.health_check()
                    health_result["components"][component_name] = component_health

                    if component_health.get("status") != "healthy":
                        unhealthy_components += 1

                except Exception as e:
                    health_result["components"][component_name] = {
                        "status": "error",
                        "error": str(e),
                    }
                    unhealthy_components += 1

        # Determine overall status
        if unhealthy_components == 0:
            health_result["overall_status"] = "healthy"
        elif unhealthy_components < len(self.registry.get_started_components()):
            health_result["overall_status"] = "degraded"
        else:
            health_result["overall_status"] = "unhealthy"

        self.health_status = health_result
        return health_result

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            "orchestration": {
                "node_id": self.node_id,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None,
                "uptime_seconds": (
                    (datetime.now(UTC) - self.startup_time).total_seconds()
                    if self.startup_time and self.is_running
                    else 0
                ),
                "is_running": self.is_running,
                "registered_components": len(self.registry._components),
                "running_components": len(self.registry.get_started_components()),
                "background_tasks": len(self.background_tasks),
            },
            "components": {},
            "system": {
                "memory_usage": self._get_memory_usage(),
                "config_loaded": bool(self.config),
            },
        }

        # Collect stats from all components
        for component_name in self.registry.get_started_components():
            component = self.registry.get_component(component_name)
            if component:
                try:
                    component_stats = component.get_stats()
                    stats["components"][component_name] = component_stats
                except Exception as e:
                    stats["components"][component_name] = {"error": str(e)}

        self.system_stats = stats
        return stats

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service component."""
        async with self._coordination_lock:
            if not self.registry.is_started(service_name):
                logger.warning(f"Service {service_name} is not currently running")
                return False

            try:
                logger.info(f"Restarting service: {service_name}")

                # Stop the service
                await self._stop_component(service_name)
                self.registry.mark_stopped(service_name)

                # Start the service
                config = self.config["orchestration"]
                success = await self._start_component(
                    service_name,
                    retry_attempts=config["component_retry_attempts"],
                    retry_delay=config["component_retry_delay_seconds"],
                )

                if success:
                    self.registry.mark_started(service_name)
                    await self._emit_event("service_restarted", {"service_name": service_name})
                    logger.info(f"Service {service_name} restarted successfully")
                    return True
                else:
                    logger.error(f"Failed to restart service: {service_name}")
                    return False

            except Exception as e:
                logger.error(f"Error restarting service {service_name}: {e}")
                return False

    def register_component(
        self,
        name: str,
        component: ComponentProtocol,
        dependencies: Optional[List[ServiceDependency]] = None,
    ) -> None:
        """Register a component with the orchestration service."""
        self.registry.register_component(name, component, dependencies)
        logger.info(f"Registered component: {name}")

    def register_event_handler(self, event_type: str, handler: callable) -> None:
        """Register an event handler for system coordination."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a system event to registered handlers."""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Event handler error for {event_type}: {e}")

    async def _health_monitoring_task(self) -> None:
        """Background task for health monitoring."""
        interval = self.config["orchestration"]["health_check_interval_seconds"]

        while self.is_running:
            try:
                await self.health_check()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)

    async def _stats_collection_task(self) -> None:
        """Background task for statistics collection."""
        interval = self.config["orchestration"]["stats_collection_interval_seconds"]

        while self.is_running:
            try:
                await self.get_system_stats()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Stats collection error: {e}")
                await asyncio.sleep(interval)

    async def _coordination_task(self) -> None:
        """Background task for system coordination and maintenance."""
        while self.is_running:
            try:
                # Perform coordination activities
                await self._perform_system_maintenance()
                await asyncio.sleep(300)  # Every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordination task error: {e}")
                await asyncio.sleep(60)

    async def _perform_system_maintenance(self) -> None:
        """Perform routine system maintenance tasks."""
        # Check for failed components and attempt restart if configured
        if self.config["monitoring"]["alert_on_component_failure"]:
            health_status = await self.health_check()
            for component_name, health in health_status["components"].items():
                if health.get("status") == "error" and self.registry.is_started(component_name):
                    logger.warning(f"Detected failed component: {component_name}, attempting restart")
                    await self.restart_service(component_name)

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get basic memory usage information."""
        try:
            import psutil
            process = psutil.Process()
            return {
                "rss_mb": process.memory_info().rss / 1024 / 1024,
                "vms_mb": process.memory_info().vms / 1024 / 1024,
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            return {"error": str(e)}