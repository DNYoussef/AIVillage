"""
Service Registry for Fog Computing Services

Manages service discovery, dependency injection, and service lifecycle coordination.
Implements the Service Locator pattern with dependency injection capabilities.
"""

import logging
from typing import Any, TypeVar

from .base_service import BaseFogService, EventBus, ServiceStatus

T = TypeVar("T", bound=BaseFogService)


class ServiceDependency:
    """Service dependency specification"""

    def __init__(self, service_type: type[BaseFogService], required: bool = True, lazy: bool = False):
        self.service_type = service_type
        self.required = required
        self.lazy = lazy  # Lazy loading - resolve when first accessed


class ServiceRegistry:
    """Central registry for fog computing services"""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.services: dict[str, BaseFogService] = {}
        self.service_types: dict[type[BaseFogService], str] = {}
        self.dependencies: dict[str, list[ServiceDependency]] = {}
        self.startup_order: list[str] = []
        self.logger = logging.getLogger(f"{__name__}.ServiceRegistry")

    def register_service(self, service: BaseFogService, dependencies: list[ServiceDependency] | None = None):
        """Register a service with optional dependencies"""
        service_name = service.service_name

        if service_name in self.services:
            raise ValueError(f"Service {service_name} already registered")

        self.services[service_name] = service
        self.service_types[type(service)] = service_name

        if dependencies:
            self.dependencies[service_name] = dependencies
        else:
            self.dependencies[service_name] = []

        self.logger.info(f"Registered service: {service_name}")

    def get_service(self, service_name: str) -> BaseFogService | None:
        """Get a service by name"""
        return self.services.get(service_name)

    def get_service_by_type(self, service_type: type[T]) -> T | None:
        """Get a service by type"""
        service_name = self.service_types.get(service_type)
        if service_name:
            return self.services.get(service_name)
        return None

    def get_all_services(self) -> dict[str, BaseFogService]:
        """Get all registered services"""
        return self.services.copy()

    def resolve_dependencies(self) -> list[str]:
        """Resolve service dependencies and return startup order"""
        # Topological sort to determine startup order
        visited = set()
        temp_visited = set()
        startup_order = []

        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")

            if service_name in visited:
                return

            temp_visited.add(service_name)

            # Visit dependencies first
            for dep in self.dependencies.get(service_name, []):
                if dep.required and not dep.lazy:
                    dep_service_name = self.service_types.get(dep.service_type)
                    if dep_service_name and dep_service_name in self.services:
                        visit(dep_service_name)

            temp_visited.remove(service_name)
            visited.add(service_name)
            startup_order.append(service_name)

        # Visit all services
        for service_name in self.services.keys():
            if service_name not in visited:
                visit(service_name)

        self.startup_order = startup_order
        self.logger.info(f"Service startup order: {startup_order}")
        return startup_order

    async def start_all_services(self) -> bool:
        """Start all services in dependency order"""
        startup_order = self.resolve_dependencies()
        started_services = []

        try:
            for service_name in startup_order:
                service = self.services[service_name]
                self.logger.info(f"Starting service: {service_name}")

                success = await service.start()
                if not success:
                    self.logger.error(f"Failed to start service: {service_name}")
                    await self._stop_started_services(started_services)
                    return False

                started_services.append(service_name)
                self.logger.info(f"Successfully started service: {service_name}")

            self.logger.info("All services started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error starting services: {e}")
            await self._stop_started_services(started_services)
            return False

    async def stop_all_services(self) -> bool:
        """Stop all services in reverse dependency order"""
        # Stop in reverse order
        stop_order = (
            list(reversed(self.startup_order)) if self.startup_order else list(reversed(list(self.services.keys())))
        )

        success = True
        for service_name in stop_order:
            service = self.services.get(service_name)
            if service and service.status == ServiceStatus.RUNNING:
                self.logger.info(f"Stopping service: {service_name}")

                service_success = await service.stop()
                if not service_success:
                    self.logger.error(f"Failed to stop service: {service_name}")
                    success = False
                else:
                    self.logger.info(f"Successfully stopped service: {service_name}")

        return success

    async def _stop_started_services(self, started_services: list[str]):
        """Stop services that were successfully started (for cleanup)"""
        for service_name in reversed(started_services):
            service = self.services[service_name]
            try:
                await service.stop()
            except Exception as e:
                self.logger.error(f"Error stopping service {service_name} during cleanup: {e}")

    def get_service_status(self) -> dict[str, Any]:
        """Get status of all services"""
        status = {"total_services": len(self.services), "running_services": 0, "error_services": 0, "services": {}}

        for name, service in self.services.items():
            service_status = service.get_status()
            status["services"][name] = service_status

            if service_status["status"] == ServiceStatus.RUNNING.value:
                status["running_services"] += 1
            elif service_status["status"] == ServiceStatus.ERROR.value:
                status["error_services"] += 1

        return status

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        service = self.services.get(service_name)
        if not service:
            self.logger.error(f"Service not found: {service_name}")
            return False

        self.logger.info(f"Restarting service: {service_name}")
        return await service.restart()

    def inject_dependencies(self, service: BaseFogService):
        """Inject dependencies into a service"""
        service_name = service.service_name
        dependencies = self.dependencies.get(service_name, [])

        for dep in dependencies:
            dep_service = self.get_service_by_type(dep.service_type)
            if dep_service:
                # Set the dependency as an attribute on the service
                attr_name = f"_{dep.service_type.__name__.lower()}"
                setattr(service, attr_name, dep_service)
                self.logger.debug(f"Injected {dep.service_type.__name__} into {service_name}")
            elif dep.required and not dep.lazy:
                raise ValueError(f"Required dependency {dep.service_type.__name__} not found for {service_name}")


class ServiceFactory:
    """Factory for creating configured services"""

    def __init__(self, registry: ServiceRegistry, base_config: dict[str, Any]):
        self.registry = registry
        self.base_config = base_config
        self.logger = logging.getLogger(f"{__name__}.ServiceFactory")

    def create_service(
        self,
        service_class: type[T],
        service_name: str,
        service_config: dict[str, Any] | None = None,
        dependencies: list[ServiceDependency] | None = None,
    ) -> T:
        """Create and register a service"""
        # Merge configuration
        config = self.base_config.copy()
        if service_config:
            config.update(service_config)

        # Create service instance
        service = service_class(service_name, config, self.registry.event_bus)

        # Register with registry
        self.registry.register_service(service, dependencies)

        # Inject dependencies
        self.registry.inject_dependencies(service)

        self.logger.info(f"Created service: {service_name} ({service_class.__name__})")
        return service
