"""
Service Locator for SageAgent dependency management.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, Optional, Type, TypeVar
import weakref

from .config import SageAgentConfig, ServiceConfig
from .interfaces import AbstractServiceBase

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceNotFoundError(Exception):
    """Raised when a requested service is not found."""

    pass


class ServiceInitializationError(Exception):
    """Raised when a service fails to initialize."""

    pass


class SageAgentServiceLocator:
    """
    Service Locator for managing SageAgent dependencies.

    Provides centralized service management with lazy loading,
    caching, and lifecycle management.
    """

    def __init__(self, config: SageAgentConfig):
        self.config = config
        self._services: Dict[str, Any] = {}
        self._service_factories: Dict[str, callable] = {}
        self._service_configs: Dict[str, ServiceConfig] = {}
        self._initialization_locks: Dict[str, asyncio.Lock] = {}
        self._weak_references: Dict[str, weakref.ref] = {}
        self._usage_stats: Dict[str, Dict[str, Any]] = {}

        # Performance tracking
        self._service_timings: Dict[str, list] = {}
        self._memory_usage: Dict[str, int] = {}

        # Initialize locks for all potential services
        self._initialize_locks()

    def _initialize_locks(self) -> None:
        """Initialize async locks for all services."""
        service_names = [
            "cognitive_composite",
            "processing_chain",
            "rag_system",
            "vector_store",
            "knowledge_tracker",
            "error_controller",
            "confidence_estimator",
            "collaboration_manager",
            "research_capabilities",
            "user_intent_interpreter",
        ]

        for service_name in service_names:
            self._initialization_locks[service_name] = asyncio.Lock()

    def register_service_factory(
        self, service_name: str, factory_func: callable, config: Optional[ServiceConfig] = None
    ) -> None:
        """Register a service factory function."""
        self._service_factories[service_name] = factory_func
        if config:
            self._service_configs[service_name] = config

        # Initialize usage stats
        self._usage_stats[service_name] = {
            "access_count": 0,
            "last_accessed": None,
            "initialization_time": None,
            "total_usage_time": 0.0,
        }

        logger.debug(f"Registered service factory: {service_name}")

    def register_instance(self, service_name: str, instance: Any) -> None:
        """Register a pre-created service instance."""
        self._services[service_name] = instance

        # Create weak reference for memory management
        self._weak_references[service_name] = weakref.ref(instance)

        # Initialize usage stats
        self._usage_stats[service_name] = {
            "access_count": 0,
            "last_accessed": None,
            "initialization_time": 0.0,
            "total_usage_time": 0.0,
        }

        logger.debug(f"Registered service instance: {service_name}")

    async def get_service(self, service_name: str, service_type: Optional[Type[T]] = None) -> T:
        """
        Get a service by name with lazy loading and caching.

        Args:
            service_name: Name of the service to retrieve
            service_type: Expected type of the service (for type checking)

        Returns:
            The requested service instance

        Raises:
            ServiceNotFoundError: If service is not registered
            ServiceInitializationError: If service initialization fails
        """
        start_time = datetime.now()

        try:
            # Check if service is already instantiated
            if service_name in self._services:
                service = self._services[service_name]
                self._update_usage_stats(service_name, start_time)

                if service_type and not isinstance(service, service_type):
                    raise ServiceNotFoundError(f"Service {service_name} exists but is not of type {service_type}")

                return service

            # Check weak references (for memory management)
            if service_name in self._weak_references:
                weak_ref = self._weak_references[service_name]
                if weak_ref() is not None:
                    service = weak_ref()
                    self._services[service_name] = service  # Re-cache
                    self._update_usage_stats(service_name, start_time)
                    return service
                else:
                    # Weak reference is dead, remove it
                    del self._weak_references[service_name]

            # Lazy loading with async lock to prevent race conditions
            async with self._initialization_locks.get(service_name, asyncio.Lock()):
                # Double-check pattern
                if service_name in self._services:
                    service = self._services[service_name]
                    self._update_usage_stats(service_name, start_time)
                    return service

                # Create service using factory
                if service_name not in self._service_factories:
                    raise ServiceNotFoundError(
                        f"Service '{service_name}' not registered. Available services: {list(self._service_factories.keys())}"
                    )

                factory_func = self._service_factories[service_name]
                service_config = self._service_configs.get(service_name)

                try:
                    # Create service instance
                    init_start = datetime.now()

                    if service_config:
                        service = await factory_func(service_config)
                    else:
                        service = await factory_func()

                    init_time = (datetime.now() - init_start).total_seconds()

                    # Initialize service if it supports it
                    if isinstance(service, AbstractServiceBase):
                        if not service.is_initialized():
                            await service.initialize()

                    # Cache the service
                    self._services[service_name] = service
                    self._weak_references[service_name] = weakref.ref(service)

                    # Update initialization stats
                    self._usage_stats[service_name]["initialization_time"] = init_time

                    logger.info(f"Successfully initialized service: {service_name} in {init_time:.3f}s")

                except Exception as e:
                    logger.error(f"Failed to initialize service {service_name}: {e}")
                    raise ServiceInitializationError(f"Failed to initialize service '{service_name}': {e}") from e

            self._update_usage_stats(service_name, start_time)
            return service

        except Exception as e:
            logger.error(f"Error retrieving service {service_name}: {e}")
            raise

    def _update_usage_stats(self, service_name: str, start_time: datetime) -> None:
        """Update usage statistics for a service."""
        if service_name in self._usage_stats:
            stats = self._usage_stats[service_name]
            stats["access_count"] += 1
            stats["last_accessed"] = datetime.now()

            usage_time = (datetime.now() - start_time).total_seconds()
            stats["total_usage_time"] += usage_time

            # Track service timings for performance analysis
            if service_name not in self._service_timings:
                self._service_timings[service_name] = []
            self._service_timings[service_name].append(usage_time)

            # Keep only recent timings (last 100)
            if len(self._service_timings[service_name]) > 100:
                self._service_timings[service_name] = self._service_timings[service_name][-100:]

    def is_service_registered(self, service_name: str) -> bool:
        """Check if a service is registered."""
        return service_name in self._service_factories or service_name in self._services

    def is_service_instantiated(self, service_name: str) -> bool:
        """Check if a service is already instantiated."""
        return service_name in self._services

    def get_registered_services(self) -> list[str]:
        """Get list of all registered service names."""
        return list(set(self._service_factories.keys()) | set(self._services.keys()))

    def get_service_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get usage statistics for services."""
        if service_name:
            return self._usage_stats.get(service_name, {})
        return self._usage_stats.copy()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all services."""
        metrics = {
            "total_services": len(self._usage_stats),
            "instantiated_services": len(self._services),
            "service_performance": {},
        }

        for service_name, timings in self._service_timings.items():
            if timings:
                metrics["service_performance"][service_name] = {
                    "avg_response_time": sum(timings) / len(timings),
                    "min_response_time": min(timings),
                    "max_response_time": max(timings),
                    "total_calls": len(timings),
                }

        return metrics

    async def cleanup_unused_services(self, max_idle_hours: int = 24) -> int:
        """Cleanup services that haven't been used recently."""
        cleaned_count = 0
        current_time = datetime.now()

        services_to_remove = []

        for service_name, stats in self._usage_stats.items():
            if stats["last_accessed"]:
                idle_time = current_time - stats["last_accessed"]
                if idle_time > timedelta(hours=max_idle_hours):
                    services_to_remove.append(service_name)

        for service_name in services_to_remove:
            if service_name in self._services:
                service = self._services[service_name]

                # Graceful shutdown if supported
                if isinstance(service, AbstractServiceBase):
                    try:
                        await service.shutdown()
                    except Exception as e:
                        logger.warning(f"Error shutting down service {service_name}: {e}")

                # Remove from cache
                del self._services[service_name]
                if service_name in self._weak_references:
                    del self._weak_references[service_name]

                cleaned_count += 1
                logger.info(f"Cleaned up unused service: {service_name}")

        return cleaned_count

    async def shutdown_all_services(self) -> None:
        """Shutdown all services gracefully."""
        for service_name, service in self._services.items():
            if isinstance(service, AbstractServiceBase):
                try:
                    await service.shutdown()
                    logger.info(f"Shutdown service: {service_name}")
                except Exception as e:
                    logger.error(f"Error shutting down service {service_name}: {e}")

        # Clear all caches
        self._services.clear()
        self._weak_references.clear()
        self._service_timings.clear()

        logger.info("All services shutdown completed")

    def get_memory_usage_estimate(self) -> Dict[str, Any]:
        """Get estimated memory usage for services."""
        import sys

        memory_info = {
            "total_services": len(self._services),
            "service_memory": {},
            "cache_overhead": sys.getsizeof(self._services) + sys.getsizeof(self._usage_stats),
        }

        for service_name, service in self._services.items():
            try:
                memory_info["service_memory"][service_name] = sys.getsizeof(service)
            except Exception:
                memory_info["service_memory"][service_name] = "unknown"

        return memory_info
