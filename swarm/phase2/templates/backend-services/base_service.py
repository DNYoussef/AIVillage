"""
Abstract Base Service Template for AI Village Gateway Services

This template provides a foundation for all gateway services with:
- Async/await patterns
- Dependency injection support
- Comprehensive error handling
- Structured logging
- Resource management
- Health checking capabilities
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type, TypeVar, Generic
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import traceback

# Type definitions
T = TypeVar('T')
ServiceConfig = Dict[str, Any]


@dataclass
class ServiceMetrics:
    """Service performance and health metrics"""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    average_response_time: float = 0.0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.requests_total == 0:
            return 100.0
        return (self.requests_success / self.requests_total) * 100.0
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds"""
        return (datetime.utcnow() - self.uptime_start).total_seconds()


@dataclass
class ServiceStatus:
    """Service health status information"""
    is_healthy: bool = True
    status: str = "running"
    message: str = "Service is operational"
    last_check: datetime = field(default_factory=datetime.utcnow)
    dependencies: Dict[str, bool] = field(default_factory=dict)


class ServiceError(Exception):
    """Base exception for all service-related errors"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SERVICE_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ServiceInitializationError(ServiceError):
    """Raised when service initialization fails"""
    
    def __init__(self, service_name: str, reason: str, details: Dict[str, Any] = None):
        super().__init__(
            f"Failed to initialize service '{service_name}': {reason}",
            "SERVICE_INIT_ERROR",
            details
        )
        self.service_name = service_name
        self.reason = reason


class ServiceDependencyError(ServiceError):
    """Raised when service dependency is unavailable"""
    
    def __init__(self, dependency_name: str, service_name: str, details: Dict[str, Any] = None):
        super().__init__(
            f"Dependency '{dependency_name}' required by '{service_name}' is unavailable",
            "SERVICE_DEPENDENCY_ERROR",
            details
        )
        self.dependency_name = dependency_name
        self.service_name = service_name


class BaseService(ABC, Generic[T]):
    """
    Abstract base class for all AI Village gateway services.
    
    Provides:
    - Lifecycle management (initialize, start, stop, cleanup)
    - Health checking and metrics collection
    - Dependency injection and management
    - Error handling and logging
    - Configuration management
    - Resource management with async context managers
    """
    
    def __init__(
        self,
        name: str,
        config: ServiceConfig = None,
        logger: logging.Logger = None,
        dependencies: Dict[str, 'BaseService'] = None
    ):
        self.name = name
        self.config = config or {}
        self.logger = logger or self._setup_logger()
        self.dependencies = dependencies or {}
        
        # Service state
        self._initialized = False
        self._running = False
        self._stopping = False
        
        # Metrics and monitoring
        self.metrics = ServiceMetrics()
        self.status = ServiceStatus()
        
        # Resource management
        self._cleanup_tasks: List[asyncio.Task] = []
        self._background_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"Service '{self.name}' created with config: {list(self.config.keys())}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging for the service"""
        logger = logging.getLogger(f"gateway.service.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Lifecycle Management
    
    async def initialize(self) -> None:
        """
        Initialize the service.
        Override this method to add service-specific initialization logic.
        """
        if self._initialized:
            self.logger.warning(f"Service '{self.name}' already initialized")
            return
        
        try:
            self.logger.info(f"Initializing service '{self.name}'...")
            
            # Check dependencies
            await self._check_dependencies()
            
            # Service-specific initialization
            await self._initialize_service()
            
            # Mark as initialized
            self._initialized = True
            self.status.status = "initialized"
            self.status.message = "Service initialized successfully"
            
            self.logger.info(f"Service '{self.name}' initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize service '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.status.is_healthy = False
            self.status.status = "failed"
            self.status.message = error_msg
            raise ServiceInitializationError(self.name, str(e), {"traceback": traceback.format_exc()})
    
    async def start(self) -> None:
        """Start the service"""
        if not self._initialized:
            await self.initialize()
        
        if self._running:
            self.logger.warning(f"Service '{self.name}' already running")
            return
        
        try:
            self.logger.info(f"Starting service '{self.name}'...")
            
            # Service-specific startup logic
            await self._start_service()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Mark as running
            self._running = True
            self.status.status = "running"
            self.status.message = "Service is operational"
            
            self.logger.info(f"Service '{self.name}' started successfully")
            
        except Exception as e:
            error_msg = f"Failed to start service '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.status.is_healthy = False
            self.status.status = "failed"
            self.status.message = error_msg
            raise ServiceError(error_msg, "SERVICE_START_ERROR")
    
    async def stop(self) -> None:
        """Stop the service gracefully"""
        if not self._running:
            self.logger.warning(f"Service '{self.name}' not running")
            return
        
        if self._stopping:
            self.logger.warning(f"Service '{self.name}' already stopping")
            return
        
        try:
            self._stopping = True
            self.logger.info(f"Stopping service '{self.name}'...")
            
            # Cancel background tasks
            await self._stop_background_tasks()
            
            # Service-specific shutdown logic
            await self._stop_service()
            
            # Mark as stopped
            self._running = False
            self._stopping = False
            self.status.status = "stopped"
            self.status.message = "Service stopped"
            
            self.logger.info(f"Service '{self.name}' stopped successfully")
            
        except Exception as e:
            error_msg = f"Error stopping service '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.status.message = error_msg
            raise ServiceError(error_msg, "SERVICE_STOP_ERROR")
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        try:
            self.logger.info(f"Cleaning up service '{self.name}'...")
            
            # Ensure service is stopped
            if self._running:
                await self.stop()
            
            # Run cleanup tasks
            await self._run_cleanup_tasks()
            
            # Service-specific cleanup
            await self._cleanup_service()
            
            self.logger.info(f"Service '{self.name}' cleaned up successfully")
            
        except Exception as e:
            error_msg = f"Error cleaning up service '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ServiceError(error_msg, "SERVICE_CLEANUP_ERROR")
    
    # Health and Monitoring
    
    async def health_check(self) -> ServiceStatus:
        """
        Perform health check on the service.
        Override this method to add service-specific health checks.
        """
        try:
            # Update last check time
            self.status.last_check = datetime.utcnow()
            
            # Check if service is running
            if not self._running:
                self.status.is_healthy = False
                self.status.message = "Service not running"
                return self.status
            
            # Check dependencies
            dependency_status = await self._check_dependency_health()
            self.status.dependencies = dependency_status
            
            # Service-specific health checks
            is_healthy = await self._perform_health_check()
            
            if is_healthy and all(dependency_status.values()):
                self.status.is_healthy = True
                self.status.message = "Service is healthy"
            else:
                self.status.is_healthy = False
                self.status.message = "Service health check failed"
            
            return self.status
            
        except Exception as e:
            self.logger.error(f"Health check failed for service '{self.name}': {str(e)}")
            self.status.is_healthy = False
            self.status.message = f"Health check error: {str(e)}"
            return self.status
    
    def get_metrics(self) -> ServiceMetrics:
        """Get service metrics"""
        return self.metrics
    
    # Dependency Management
    
    def add_dependency(self, name: str, service: 'BaseService') -> None:
        """Add a service dependency"""
        self.dependencies[name] = service
        self.logger.info(f"Added dependency '{name}' to service '{self.name}'")
    
    def get_dependency(self, name: str) -> Optional['BaseService']:
        """Get a service dependency"""
        return self.dependencies.get(name)
    
    async def _check_dependencies(self) -> None:
        """Check that all dependencies are available and healthy"""
        for dep_name, dep_service in self.dependencies.items():
            try:
                if not dep_service._initialized:
                    await dep_service.initialize()
                
                status = await dep_service.health_check()
                if not status.is_healthy:
                    raise ServiceDependencyError(dep_name, self.name)
                    
            except Exception as e:
                self.logger.error(f"Dependency check failed for '{dep_name}': {str(e)}")
                raise ServiceDependencyError(dep_name, self.name, {"error": str(e)})
    
    async def _check_dependency_health(self) -> Dict[str, bool]:
        """Check health of all dependencies"""
        dependency_status = {}
        
        for dep_name, dep_service in self.dependencies.items():
            try:
                status = await dep_service.health_check()
                dependency_status[dep_name] = status.is_healthy
            except Exception as e:
                self.logger.error(f"Failed to check dependency '{dep_name}': {str(e)}")
                dependency_status[dep_name] = False
        
        return dependency_status
    
    # Context Manager Support
    
    async def __aenter__(self) -> 'BaseService':
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        await self.cleanup()
    
    @asynccontextmanager
    async def request_context(self, request_id: str = None):
        """Context manager for request processing with metrics"""
        request_start = datetime.utcnow()
        self.metrics.requests_total += 1
        
        try:
            yield
            
            # Success metrics
            self.metrics.requests_success += 1
            response_time = (datetime.utcnow() - request_start).total_seconds()
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.requests_success - 1) + response_time)
                / self.metrics.requests_success
            )
            
        except Exception as e:
            # Error metrics
            self.metrics.requests_failed += 1
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.utcnow()
            raise
    
    # Background Task Management
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks"""
        # Health monitoring task
        task = asyncio.create_task(self._health_monitor_loop())
        self._background_tasks.append(task)
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._background_tasks.clear()
    
    async def _health_monitor_loop(self) -> None:
        """Background health monitoring loop"""
        while self._running and not self._stopping:
            try:
                await asyncio.sleep(30)  # Health check every 30 seconds
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {str(e)}")
    
    async def _run_cleanup_tasks(self) -> None:
        """Run all cleanup tasks"""
        for task in self._cleanup_tasks:
            try:
                await task
            except Exception as e:
                self.logger.error(f"Cleanup task failed: {str(e)}")
        
        self._cleanup_tasks.clear()
    
    def add_cleanup_task(self, task: asyncio.Task) -> None:
        """Add a cleanup task"""
        self._cleanup_tasks.append(task)
    
    # Abstract Methods (Override in subclasses)
    
    @abstractmethod
    async def _initialize_service(self) -> None:
        """Service-specific initialization logic"""
        pass
    
    @abstractmethod
    async def _start_service(self) -> None:
        """Service-specific startup logic"""
        pass
    
    @abstractmethod
    async def _stop_service(self) -> None:
        """Service-specific shutdown logic"""
        pass
    
    async def _cleanup_service(self) -> None:
        """Service-specific cleanup logic (optional override)"""
        pass
    
    async def _perform_health_check(self) -> bool:
        """Service-specific health check (optional override)"""
        return True
    
    # Properties
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized"""
        return self._initialized
    
    @property
    def is_running(self) -> bool:
        """Check if service is running"""
        return self._running
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.status.is_healthy
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status='{self.status.status}')"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"initialized={self._initialized}, "
            f"running={self._running}, "
            f"healthy={self.status.is_healthy}"
            f")"
        )