#!/usr/bin/env python3
"""Dependency Injection Configuration for AIVillage Architecture

This module implements a comprehensive dependency injection container to eliminate
connascence violations and establish single sources of truth for all services.

Key Principles:
1. Interface Segregation - Small, focused protocols
2. Dependency Inversion - Depend on abstractions, not concretions
3. Single Responsibility - Each service has one reason to change
4. Configuration over Convention - Explicit wiring
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)

# Type variables for generic dependency injection
T = TypeVar("T")
ServiceType = TypeVar("ServiceType")


class ServiceLifetime(Enum):
    """Service lifetime scopes for dependency injection"""

    SINGLETON = "singleton"  # One instance for entire application
    TRANSIENT = "transient"  # New instance every time
    SCOPED = "scoped"  # One instance per scope (e.g., per request)


@dataclass
class ServiceRegistration:
    """Service registration metadata"""

    interface: type
    implementation: type
    lifetime: ServiceLifetime
    factory: callable | None = None
    constructor_args: dict[str, Any] = None

    def __post_init__(self):
        if self.constructor_args is None:
            self.constructor_args = {}


# Core Service Protocols (Interfaces)


class EncryptionService(Protocol):
    """Unified encryption service interface - eliminates CoA violations"""

    def encrypt_data(self, data: Any, field_name: str = "") -> bytes:
        """Encrypt data with metadata for compliance"""
        ...

    def decrypt_data(self, encrypted_data: bytes, field_name: str = "") -> Any:
        """Decrypt data with audit logging"""
        ...

    def hash_data(self, data: str, salt: str | None = None) -> str:
        """Hash data with configurable algorithms"""
        ...

    def sign_data(self, data: Any, key_identifier: str) -> bytes:
        """Sign data with specified key"""
        ...


class RAGService(Protocol):
    """Unified RAG service interface"""

    async def query(self, query: str, mode: str = "balanced", max_results: int = 10) -> dict[str, Any]:
        """Query knowledge base"""
        ...

    async def index_document(self, content: str, metadata: dict[str, Any]) -> str:
        """Index new document"""
        ...

    async def search_knowledge_graph(self, query: str) -> dict[str, Any]:
        """Search Bayesian trust graph"""
        ...


class CommunicationService(Protocol):
    """Unified communication service interface"""

    async def send_message(self, recipient: str, message: str, channel: str = "direct") -> dict[str, Any]:
        """Send message through appropriate channel"""
        ...

    async def broadcast(self, message: str, exclude: list[str] = None) -> dict[str, Any]:
        """Broadcast to all agents"""
        ...

    async def join_channel(self, channel_name: str) -> bool:
        """Join communication channel"""
        ...


class MemoryService(Protocol):
    """Unified memory service interface"""

    async def store_memory(self, content: str, importance: float, tags: list[str]) -> str:
        """Store memory with importance weighting"""
        ...

    async def retrieve_memories(self, query: str, threshold: float = 0.3) -> list[dict[str, Any]]:
        """Retrieve relevant memories"""
        ...

    async def consolidate_memories(self) -> dict[str, Any]:
        """Consolidate and decay memories"""
        ...


class ConfigurationService(Protocol):
    """Unified configuration service interface"""

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...

    def set_value(self, key: str, value: Any) -> None:
        """Set configuration value"""
        ...

    def get_section(self, section: str) -> dict[str, Any]:
        """Get entire configuration section"""
        ...


class LoggingService(Protocol):
    """Unified logging service interface"""

    def log_info(self, message: str, **kwargs) -> None:
        """Log information message"""
        ...

    def log_error(self, message: str, exception: Exception = None, **kwargs) -> None:
        """Log error message"""
        ...

    def log_audit(self, action: str, details: dict[str, Any]) -> None:
        """Log audit event"""
        ...


class MetricsService(Protocol):
    """Unified metrics service interface"""

    def increment_counter(self, name: str, value: float = 1.0, tags: dict[str, str] = None) -> None:
        """Increment counter metric"""
        ...

    def record_gauge(self, name: str, value: float, tags: dict[str, str] = None) -> None:
        """Record gauge metric"""
        ...

    def record_timing(self, name: str, duration_ms: float, tags: dict[str, str] = None) -> None:
        """Record timing metric"""
        ...


class DIContainer:
    """Dependency Injection Container with lifetime management"""

    def __init__(self):
        self._registrations: dict[type, ServiceRegistration] = {}
        self._singletons: dict[type, Any] = {}
        self._scoped_instances: dict[str, dict[type, Any]] = {}
        self._current_scope: str | None = None

    def register(
        self,
        interface: type[T],
        implementation: type[T],
        lifetime: ServiceLifetime = ServiceLifetime.TRANSIENT,
        factory: callable | None = None,
        **constructor_args,
    ) -> "DIContainer":
        """Register service with specified lifetime"""

        registration = ServiceRegistration(
            interface=interface,
            implementation=implementation,
            lifetime=lifetime,
            factory=factory,
            constructor_args=constructor_args,
        )

        self._registrations[interface] = registration
        logger.info(f"Registered {interface.__name__} -> {implementation.__name__} ({lifetime.value})")
        return self

    def register_singleton(self, interface: type[T], implementation: type[T], **kwargs) -> "DIContainer":
        """Register singleton service"""
        return self.register(interface, implementation, ServiceLifetime.SINGLETON, **kwargs)

    def register_transient(self, interface: type[T], implementation: type[T], **kwargs) -> "DIContainer":
        """Register transient service"""
        return self.register(interface, implementation, ServiceLifetime.TRANSIENT, **kwargs)

    def register_scoped(self, interface: type[T], implementation: type[T], **kwargs) -> "DIContainer":
        """Register scoped service"""
        return self.register(interface, implementation, ServiceLifetime.SCOPED, **kwargs)

    def register_instance(self, interface: type[T], instance: T) -> "DIContainer":
        """Register existing instance as singleton"""
        self._singletons[interface] = instance
        registration = ServiceRegistration(
            interface=interface, implementation=type(instance), lifetime=ServiceLifetime.SINGLETON
        )
        self._registrations[interface] = registration
        return self

    def get(self, interface: type[T]) -> T:
        """Get service instance"""

        if interface not in self._registrations:
            raise ValueError(f"Service {interface.__name__} not registered")

        registration = self._registrations[interface]

        # Handle singleton lifetime
        if registration.lifetime == ServiceLifetime.SINGLETON:
            if interface in self._singletons:
                return self._singletons[interface]

            instance = self._create_instance(registration)
            self._singletons[interface] = instance
            return instance

        # Handle scoped lifetime
        elif registration.lifetime == ServiceLifetime.SCOPED:
            if not self._current_scope:
                raise ValueError("No active scope for scoped service")

            if self._current_scope not in self._scoped_instances:
                self._scoped_instances[self._current_scope] = {}

            scope_cache = self._scoped_instances[self._current_scope]

            if interface in scope_cache:
                return scope_cache[interface]

            instance = self._create_instance(registration)
            scope_cache[interface] = instance
            return instance

        # Handle transient lifetime
        else:
            return self._create_instance(registration)

    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """Create service instance with dependency injection"""

        # Use factory if provided
        if registration.factory:
            return registration.factory(**registration.constructor_args)

        # Resolve constructor dependencies
        implementation = registration.implementation

        # Get constructor parameters
        import inspect

        signature = inspect.signature(implementation.__init__)
        resolved_args = {}

        for param_name, param in signature.parameters.items():
            if param_name == "self":
                continue

            # Check if argument provided in registration
            if param_name in registration.constructor_args:
                resolved_args[param_name] = registration.constructor_args[param_name]

            # Try to resolve from container
            elif param.annotation != inspect.Parameter.empty:
                try:
                    resolved_args[param_name] = self.get(param.annotation)
                except ValueError:
                    # Use default value if available
                    if param.default != inspect.Parameter.empty:
                        resolved_args[param_name] = param.default
                    else:
                        raise ValueError(f"Cannot resolve dependency {param_name} for {implementation.__name__}")

        return implementation(**resolved_args)

    def begin_scope(self, scope_id: str = None) -> str:
        """Begin new dependency scope"""
        import uuid

        scope_id = scope_id or str(uuid.uuid4())
        self._current_scope = scope_id
        return scope_id

    def end_scope(self, scope_id: str = None) -> None:
        """End dependency scope and cleanup instances"""
        target_scope = scope_id or self._current_scope

        if target_scope in self._scoped_instances:
            # Cleanup scoped instances
            scoped_cache = self._scoped_instances[target_scope]
            for instance in scoped_cache.values():
                if hasattr(instance, "dispose"):
                    try:
                        instance.dispose()
                    except Exception as e:
                        logger.warning(f"Error disposing {type(instance).__name__}: {e}")

            del self._scoped_instances[target_scope]

        if self._current_scope == target_scope:
            self._current_scope = None

    def get_registration_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all registered services"""
        info = {}

        for interface, registration in self._registrations.items():
            info[interface.__name__] = {
                "implementation": registration.implementation.__name__,
                "lifetime": registration.lifetime.value,
                "has_factory": registration.factory is not None,
                "constructor_args": list(registration.constructor_args.keys()),
                "is_singleton_created": interface in self._singletons,
            }

        return info


class ConfigurableDIContainer(DIContainer):
    """DI Container that can be configured from files"""

    def load_from_config(self, config_path: str | Path) -> "ConfigurableDIContainer":
        """Load service registrations from configuration file"""

        config_path = Path(config_path)

        if config_path.suffix == ".json":
            import json

            with open(config_path) as f:
                config = json.load(f)
        elif config_path.suffix in [".yml", ".yaml"]:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        # Process service registrations
        for service_config in config.get("services", []):
            self._register_from_config(service_config)

        return self

    def _register_from_config(self, service_config: dict[str, Any]) -> None:
        """Register service from configuration"""

        # Import interface and implementation classes
        interface_class = self._import_class(service_config["interface"])
        implementation_class = self._import_class(service_config["implementation"])

        # Parse lifetime
        lifetime = ServiceLifetime(service_config.get("lifetime", "transient"))

        # Get constructor arguments
        constructor_args = service_config.get("constructor_args", {})

        self.register(
            interface=interface_class, implementation=implementation_class, lifetime=lifetime, **constructor_args
        )

    def _import_class(self, class_path: str) -> type:
        """Import class from module path"""
        module_path, class_name = class_path.rsplit(".", 1)

        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, class_name)


# Global container instance
_container = ConfigurableDIContainer()


def get_container() -> ConfigurableDIContainer:
    """Get global DI container instance"""
    return _container


def configure_aivillage_services() -> None:
    """Configure standard AIVillage services in DI container"""

    # Import implementations - these will be created during implementation phase
    try:
        from packages.core.common.configuration_manager import ConfigurationManager
        from packages.core.common.logging_service import LoggingService as LoggingServiceImpl
        from packages.core.common.metrics_service import MetricsService as MetricsServiceImpl
        from packages.core.security.digital_twin_encryption import DigitalTwinEncryption

        from packages.agents.core.communication.channel_manager import ChannelManager
        from packages.agents.core.memory.langroid_memory import LangroidMemoryService
        from packages.rag.unified_rag_client import UnifiedRAGClient

        # Register core services as singletons
        _container.register_singleton(EncryptionService, DigitalTwinEncryption)
        _container.register_singleton(RAGService, UnifiedRAGClient)
        _container.register_singleton(CommunicationService, ChannelManager)
        _container.register_singleton(ConfigurationService, ConfigurationManager)
        _container.register_singleton(LoggingService, LoggingServiceImpl)
        _container.register_singleton(MetricsService, MetricsServiceImpl)

        # Register memory service as scoped (per agent)
        _container.register_scoped(MemoryService, LangroidMemoryService)

        logger.info("AIVillage services configured successfully")

    except ImportError as e:
        logger.warning(f"Some services not available during configuration: {e}")
        logger.info("Services will be registered during implementation phase")


# Dependency injection decorators


def inject(interface: type[T]) -> T:
    """Decorator to inject service into function parameter"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Inject service if not provided
            if interface.__name__.lower() not in kwargs:
                kwargs[interface.__name__.lower()] = _container.get(interface)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def service_method(interface: type[T]):
    """Decorator to mark method as requiring service injection"""

    def decorator(func):
        func._injected_service = interface
        return func

    return decorator


# Usage examples and configuration templates

EXAMPLE_CONFIG = {
    "services": [
        {
            "interface": "config.architecture.dependency_injection_config.EncryptionService",
            "implementation": "packages.core.security.digital_twin_encryption.DigitalTwinEncryption",
            "lifetime": "singleton",
            "constructor_args": {},
        },
        {
            "interface": "config.architecture.dependency_injection_config.RAGService",
            "implementation": "packages.rag.unified_rag_client.UnifiedRAGClient",
            "lifetime": "singleton",
            "constructor_args": {"connection_string": "sqlite:///data/rag.db"},
        },
    ]
}


def create_example_config_file(output_path: str | Path) -> None:
    """Create example configuration file"""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)

    logger.info(f"Example DI config created at {output_path}")


if __name__ == "__main__":
    # Example usage
    create_example_config_file("config/architecture/services.json")
    configure_aivillage_services()

    # Print registration info
    container = get_container()
    print("\n=== REGISTERED SERVICES ===")
    for service, info in container.get_registration_info().items():
        print(f"{service}: {info['implementation']} ({info['lifetime']})")
