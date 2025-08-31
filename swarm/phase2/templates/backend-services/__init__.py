"""
Service Aggregator Template for AI Village Gateway Services

This module provides centralized service management using the aggregator pattern
that proved successful in the security server refactoring. It handles:
- Service registration and discovery
- Dependency injection and resolution
- Lifecycle management coordination
- Health monitoring aggregation
- Configuration distribution
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
import importlib
import inspect

from .base_service import BaseService, ServiceError, ServiceStatus, ServiceMetrics


@dataclass
class ServiceRegistration:
    """Service registration information"""
    name: str
    service_class: Type[BaseService]
    config: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority services start first
    auto_start: bool = True
    singleton: bool = True


@dataclass
class ServiceGraph:
    """Service dependency graph for proper initialization order"""
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=dict)
    
    def add_dependency(self, service: str, depends_on: str) -> None:
        """Add a dependency edge"""
        self.nodes.add(service)
        self.nodes.add(depends_on)
        
        if service not in self.edges:
            self.edges[service] = set()
        self.edges[service].add(depends_on)
    
    def topological_sort(self) -> List[str]:
        """Get services in dependency order (dependencies first)"""
        in_degree = {node: 0 for node in self.nodes}
        
        # Calculate in-degrees
        for node in self.nodes:
            for neighbor in self.edges.get(node, []):
                in_degree[neighbor] += 1
        
        # Find nodes with no incoming edges
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort by priority for deterministic order
            queue.sort()
            node = queue.pop(0)
            result.append(node)
            
            # Remove edges from this node
            for neighbor in self.edges.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            raise ServiceError("Circular dependency detected in service graph")
        
        return result


class ServiceAggregator:
    """
    Centralized service management and coordination.
    
    This class implements the aggregator pattern for managing multiple services:
    - Service registration and discovery
    - Dependency injection and lifecycle coordination
    - Health monitoring and metrics aggregation
    - Configuration management and distribution
    """
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or self._setup_logger()
        
        # Service management
        self._registrations: Dict[str, ServiceRegistration] = {}
        self._instances: Dict[str, BaseService] = {}
        self._dependency_graph = ServiceGraph()
        
        # State tracking
        self._initialized = False
        self._running = False
        self._startup_order: List[str] = []
        
        # Global configuration
        self._global_config: Dict[str, Any] = {}
        
        self.logger.info("Service aggregator initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup aggregator logging"""
        logger = logging.getLogger("gateway.service_aggregator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Service Registration
    
    def register_service(
        self,
        name: str,
        service_class: Type[BaseService],
        config: Dict[str, Any] = None,
        dependencies: List[str] = None,
        priority: int = 0,
        auto_start: bool = True,
        singleton: bool = True
    ) -> None:
        """
        Register a service with the aggregator.
        
        Args:
            name: Unique service name
            service_class: Service class (must inherit from BaseService)
            config: Service-specific configuration
            dependencies: List of service names this service depends on
            priority: Startup priority (higher numbers start first)
            auto_start: Whether to auto-start with aggregator
            singleton: Whether to create only one instance
        """
        if not issubclass(service_class, BaseService):
            raise ServiceError(f"Service class {service_class} must inherit from BaseService")
        
        if name in self._registrations:
            raise ServiceError(f"Service '{name}' already registered")
        
        registration = ServiceRegistration(
            name=name,
            service_class=service_class,
            config=config or {},
            dependencies=dependencies or [],
            priority=priority,
            auto_start=auto_start,
            singleton=singleton
        )
        
        self._registrations[name] = registration
        
        # Add to dependency graph
        self._dependency_graph.nodes.add(name)
        for dep in dependencies or []:
            self._dependency_graph.add_dependency(name, dep)
        
        self.logger.info(f"Registered service '{name}' with dependencies: {dependencies or []}")
    
    def register_service_from_config(self, config: Dict[str, Any]) -> None:
        """Register service from configuration dictionary"""
        name = config.get("name")
        if not name:
            raise ServiceError("Service configuration must include 'name'")
        
        # Import service class
        class_path = config.get("class")
        if not class_path:
            raise ServiceError(f"Service '{name}' configuration must include 'class'")
        
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        service_class = getattr(module, class_name)
        
        self.register_service(
            name=name,
            service_class=service_class,
            config=config.get("config", {}),
            dependencies=config.get("dependencies", []),
            priority=config.get("priority", 0),
            auto_start=config.get("auto_start", True),
            singleton=config.get("singleton", True)
        )
    
    def register_services_from_config(self, services_config: List[Dict[str, Any]]) -> None:
        """Register multiple services from configuration list"""
        for service_config in services_config:
            self.register_service_from_config(service_config)
    
    # Service Lifecycle Management
    
    async def initialize(self, global_config: Dict[str, Any] = None) -> None:
        """Initialize all registered services"""
        if self._initialized:
            self.logger.warning("Service aggregator already initialized")
            return
        
        try:
            self.logger.info("Initializing service aggregator...")
            
            # Store global configuration
            self._global_config = global_config or {}
            
            # Calculate startup order
            self._startup_order = self._calculate_startup_order()
            self.logger.info(f"Service startup order: {self._startup_order}")
            
            # Create and initialize services in dependency order
            for service_name in self._startup_order:
                await self._create_and_initialize_service(service_name)
            
            self._initialized = True
            self.logger.info("Service aggregator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service aggregator: {str(e)}", exc_info=True)
            await self.cleanup()  # Cleanup any partially initialized services
            raise ServiceError(f"Service aggregator initialization failed: {str(e)}")
    
    async def start(self) -> None:
        """Start all auto-start services"""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            self.logger.warning("Service aggregator already running")
            return
        
        try:
            self.logger.info("Starting service aggregator...")
            
            # Start services in dependency order
            for service_name in self._startup_order:
                registration = self._registrations[service_name]
                if registration.auto_start:
                    service = self._instances.get(service_name)
                    if service and not service.is_running:
                        await service.start()
            
            self._running = True
            self.logger.info("Service aggregator started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start service aggregator: {str(e)}", exc_info=True)
            raise ServiceError(f"Service aggregator startup failed: {str(e)}")
    
    async def stop(self) -> None:
        """Stop all running services"""
        if not self._running:
            self.logger.warning("Service aggregator not running")
            return
        
        try:
            self.logger.info("Stopping service aggregator...")
            
            # Stop services in reverse dependency order
            for service_name in reversed(self._startup_order):
                service = self._instances.get(service_name)
                if service and service.is_running:
                    try:
                        await service.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping service '{service_name}': {str(e)}")
            
            self._running = False
            self.logger.info("Service aggregator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping service aggregator: {str(e)}", exc_info=True)
            raise ServiceError(f"Service aggregator stop failed: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup all services"""
        try:
            self.logger.info("Cleaning up service aggregator...")
            
            # Ensure services are stopped
            if self._running:
                await self.stop()
            
            # Cleanup services in reverse dependency order
            for service_name in reversed(self._startup_order):
                service = self._instances.get(service_name)
                if service:
                    try:
                        await service.cleanup()
                    except Exception as e:
                        self.logger.error(f"Error cleaning up service '{service_name}': {str(e)}")
            
            # Clear instances
            self._instances.clear()
            self._initialized = False
            
            self.logger.info("Service aggregator cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up service aggregator: {str(e)}", exc_info=True)
    
    # Service Access and Discovery
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service instance by name"""
        return self._instances.get(name)
    
    def require_service(self, name: str) -> BaseService:
        """Get a service instance by name, raise error if not found"""
        service = self.get_service(name)
        if service is None:
            raise ServiceError(f"Required service '{name}' not found")
        return service
    
    def list_services(self) -> List[str]:
        """List all registered service names"""
        return list(self._registrations.keys())
    
    def list_running_services(self) -> List[str]:
        """List all currently running service names"""
        return [name for name, service in self._instances.items() if service.is_running]
    
    # Health and Monitoring
    
    async def health_check_all(self) -> Dict[str, ServiceStatus]:
        """Perform health check on all services"""
        results = {}
        
        for name, service in self._instances.items():
            try:
                status = await service.health_check()
                results[name] = status
            except Exception as e:
                self.logger.error(f"Health check failed for service '{name}': {str(e)}")
                results[name] = ServiceStatus(
                    is_healthy=False,
                    status="error",
                    message=f"Health check error: {str(e)}",
                    last_check=datetime.utcnow()
                )
        
        return results
    
    def get_metrics_all(self) -> Dict[str, ServiceMetrics]:
        """Get metrics for all services"""
        return {name: service.get_metrics() for name, service in self._instances.items()}
    
    async def get_aggregated_status(self) -> Dict[str, Any]:
        """Get aggregated status of all services"""
        health_results = await self.health_check_all()
        metrics = self.get_metrics_all()
        
        # Calculate overall health
        all_healthy = all(status.is_healthy for status in health_results.values())
        
        # Aggregate metrics
        total_requests = sum(m.requests_total for m in metrics.values())
        total_success = sum(m.requests_success for m in metrics.values())
        total_failed = sum(m.requests_failed for m in metrics.values())
        
        avg_response_time = (
            sum(m.average_response_time * m.requests_total for m in metrics.values()) / 
            max(total_requests, 1)
        )
        
        return {
            "overall_healthy": all_healthy,
            "services_count": len(self._instances),
            "running_services": len(self.list_running_services()),
            "health_status": health_results,
            "aggregated_metrics": {
                "total_requests": total_requests,
                "total_success": total_success,
                "total_failed": total_failed,
                "success_rate": (total_success / max(total_requests, 1)) * 100.0,
                "average_response_time": avg_response_time
            },
            "individual_metrics": metrics
        }
    
    # Configuration Management
    
    def set_global_config(self, config: Dict[str, Any]) -> None:
        """Set global configuration for all services"""
        self._global_config.update(config)
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        return self._global_config.copy()
    
    # Context Manager Support
    
    async def __aenter__(self) -> 'ServiceAggregator':
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.cleanup()
    
    @asynccontextmanager
    async def service_context(self, service_names: List[str] = None):
        """Context manager for specific services"""
        services_to_manage = service_names or list(self._instances.keys())
        started_services = []
        
        try:
            # Start requested services
            for name in services_to_manage:
                service = self.get_service(name)
                if service and not service.is_running:
                    await service.start()
                    started_services.append(name)
            
            yield {name: self.get_service(name) for name in services_to_manage}
            
        finally:
            # Stop services we started
            for name in reversed(started_services):
                service = self.get_service(name)
                if service:
                    try:
                        await service.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping service '{name}': {str(e)}")
    
    # Private Methods
    
    def _calculate_startup_order(self) -> List[str]:
        """Calculate service startup order based on dependencies and priority"""
        # Get topological order (dependencies first)
        topo_order = self._dependency_graph.topological_sort()
        
        # Sort by priority within dependency constraints
        # Higher priority services should start first within their dependency level
        priority_map = {
            name: reg.priority for name, reg in self._registrations.items()
        }
        
        # Stable sort by priority (descending)
        return sorted(topo_order, key=lambda x: priority_map.get(x, 0), reverse=True)
    
    async def _create_and_initialize_service(self, service_name: str) -> None:
        """Create and initialize a single service"""
        registration = self._registrations[service_name]
        
        try:
            self.logger.info(f"Creating service '{service_name}'...")
            
            # Check if already created (for singletons)
            if registration.singleton and service_name in self._instances:
                return
            
            # Resolve dependencies
            dependencies = {}
            for dep_name in registration.dependencies:
                dep_service = self._instances.get(dep_name)
                if dep_service is None:
                    raise ServiceError(f"Dependency '{dep_name}' not available for service '{service_name}'")
                dependencies[dep_name] = dep_service
            
            # Merge configurations
            service_config = {}
            service_config.update(self._global_config)
            service_config.update(registration.config)
            
            # Create service instance
            service = registration.service_class(
                name=service_name,
                config=service_config,
                logger=self.logger.getChild(service_name),
                dependencies=dependencies
            )
            
            # Initialize service
            await service.initialize()
            
            # Store instance
            self._instances[service_name] = service
            
            self.logger.info(f"Service '{service_name}' created and initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to create service '{service_name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg)


# Convenience functions for common service patterns

async def create_service_aggregator(
    services_config: List[Dict[str, Any]] = None,
    global_config: Dict[str, Any] = None,
    auto_start: bool = True
) -> ServiceAggregator:
    """
    Create and initialize a service aggregator with configuration.
    
    Args:
        services_config: List of service configurations
        global_config: Global configuration for all services
        auto_start: Whether to auto-start the aggregator
    
    Returns:
        Initialized (and optionally started) ServiceAggregator
    """
    aggregator = ServiceAggregator()
    
    # Register services from configuration
    if services_config:
        aggregator.register_services_from_config(services_config)
    
    # Initialize
    await aggregator.initialize(global_config)
    
    # Start if requested
    if auto_start:
        await aggregator.start()
    
    return aggregator


def register_common_services(aggregator: ServiceAggregator) -> None:
    """Register common services that most gateways need"""
    # Example registrations - customize for your specific services
    
    # Database service (highest priority, no dependencies)
    aggregator.register_service(
        name="database",
        service_class=DatabaseService,  # Replace with your database service class
        priority=100,
        dependencies=[]
    )
    
    # Cache service (high priority, no dependencies)
    aggregator.register_service(
        name="cache",
        service_class=CacheService,  # Replace with your cache service class
        priority=90,
        dependencies=[]
    )
    
    # Authentication service (depends on database)
    aggregator.register_service(
        name="auth",
        service_class=AuthService,  # Replace with your auth service class
        priority=80,
        dependencies=["database"]
    )
    
    # API Gateway service (depends on auth and cache)
    aggregator.register_service(
        name="api_gateway",
        service_class=APIGatewayService,  # Replace with your API gateway service class
        priority=70,
        dependencies=["auth", "cache"]
    )


# Export public interface
__all__ = [
    'ServiceAggregator',
    'ServiceRegistration',
    'ServiceGraph',
    'create_service_aggregator',
    'register_common_services'
]