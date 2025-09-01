"""
Dependency Injection Containers for Phase 3 Services
Provides clean dependency injection for fog and graph services.
"""

import asyncio
from typing import Any, Dict, List, Optional, Type, TypeVar, Callable
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceContainer:
    """Dependency injection container for services."""

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._configurations: Dict[str, Dict[str, Any]] = {}

    def register_singleton(self, service_type: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._singletons[service_type.__name__] = implementation

    def register_transient(self, service_type: Type[T], implementation: Type[T]) -> None:
        """Register a transient service."""
        self._services[service_type.__name__] = implementation

    def register_factory(self, service_type: Type[T], factory: Callable[..., T]) -> None:
        """Register a factory function."""
        self._factories[service_type.__name__] = factory

    def register_configuration(self, service_name: str, config: Dict[str, Any]) -> None:
        """Register configuration for a service."""
        self._configurations[service_name] = config

    def resolve(self, service_type: Type[T]) -> T:
        """Resolve a service instance."""
        service_name = service_type.__name__

        # Check if it's a singleton that's already created
        if service_name in self._singletons:
            if service_name not in self._services:
                self._services[service_name] = self._create_instance(self._singletons[service_name])
            return self._services[service_name]

        # Check if it's a factory
        if service_name in self._factories:
            return self._factories[service_name]()

        # Create transient instance
        if service_name in self._services:
            return self._create_instance(self._services[service_name])

        raise ValueError(f"Service {service_name} not registered")

    def _create_instance(self, service_class: Type[T]) -> T:
        """Create service instance with dependency injection."""
        # Get constructor parameters
        signature = inspect.signature(service_class.__init__)
        parameters = signature.parameters

        # Prepare constructor arguments
        kwargs = {}
        for param_name, param in parameters.items():
            if param_name == "self":
                continue

            # Try to resolve dependency
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # Use configuration if available
                    config = self._configurations.get(service_class.__name__, {})
                    if param_name in config:
                        kwargs[param_name] = config[param_name]
                    elif param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default

        return service_class(**kwargs)


class FogServiceContainer(ServiceContainer):
    """Specialized container for fog computing services."""

    def __init__(self):
        super().__init__()
        self._setup_fog_services()

    def _setup_fog_services(self):
        """Setup default fog service registrations."""
        # Register common fog configurations
        self.register_configuration(
            "FogOrchestrationService", {"port": 8080, "enable_health_checks": True, "startup_timeout": 30}
        )

        self.register_configuration(
            "FogHarvestingService", {"min_battery_percent": 20, "max_thermal_temp": 45.0, "token_rate_per_hour": 10}
        )

        self.register_configuration(
            "FogPrivacyService",
            {"default_privacy_level": "private", "circuit_rotation_interval": 300, "max_circuits": 10},
        )


class GraphServiceContainer(ServiceContainer):
    """Specialized container for graph processing services."""

    def __init__(self):
        super().__init__()
        self._setup_graph_services()

    def _setup_graph_services(self):
        """Setup default graph service registrations."""
        self.register_configuration(
            "GapDetectionService", {"similarity_threshold": 0.7, "batch_size": 32, "timeout_ms": 30000}
        )

        self.register_configuration(
            "KnowledgeProposalService",
            {"max_proposals_per_gap": 5, "confidence_threshold": 0.8, "ml_model_path": "models/knowledge_proposal"},
        )

        self.register_configuration(
            "GraphAnalysisService", {"max_nodes": 100000, "analysis_depth": 3, "enable_gpu": True}
        )


class ServiceRegistry:
    """Global service registry for service discovery."""

    def __init__(self):
        self._registry: Dict[str, Dict[str, Any]] = {}

    async def register(self, service_name: str, service_info: Dict[str, Any]) -> None:
        """Register a service."""
        self._registry[service_name] = {**service_info, "registered_at": asyncio.get_event_loop().time()}
        logger.info(f"Service registered: {service_name}")

    async def discover(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover a service."""
        return self._registry.get(service_name)

    async def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services."""
        return list(self._registry.values())

    async def unregister(self, service_name: str) -> None:
        """Unregister a service."""
        if service_name in self._registry:
            del self._registry[service_name]
            logger.info(f"Service unregistered: {service_name}")


class ConfigurationManager:
    """Centralized configuration management."""

    def __init__(self):
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._watchers: Dict[str, List[Callable]] = {}

    def set_configuration(self, service_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for a service."""
        old_config = self._configurations.get(service_name, {})
        self._configurations[service_name] = config

        # Notify watchers if configuration changed
        if old_config != config:
            self._notify_watchers(service_name, config)

    def get_configuration(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a service."""
        return self._configurations.get(service_name, {})

    def watch_configuration(self, service_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Watch for configuration changes."""
        if service_name not in self._watchers:
            self._watchers[service_name] = []
        self._watchers[service_name].append(callback)

    def _notify_watchers(self, service_name: str, config: Dict[str, Any]) -> None:
        """Notify configuration watchers."""
        if service_name in self._watchers:
            for callback in self._watchers[service_name]:
                try:
                    callback(config)
                except Exception as e:
                    logger.error(f"Error notifying configuration watcher: {e}")


# Global instances
fog_container = FogServiceContainer()
graph_container = GraphServiceContainer()
service_registry = ServiceRegistry()
config_manager = ConfigurationManager()

__all__ = [
    "ServiceContainer",
    "FogServiceContainer",
    "GraphServiceContainer",
    "ServiceRegistry",
    "ConfigurationManager",
    "fog_container",
    "graph_container",
    "service_registry",
    "config_manager",
]
