"""
Service Interface Definitions for AI Village Gateway

This module defines the interfaces and protocols that services must implement
to work within the AI Village gateway ecosystem. It provides:
- Service interface contracts
- Plugin system interfaces
- Event system interfaces
- Configuration interfaces
- Monitoring and metrics interfaces
"""

import asyncio
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Callable, Awaitable, 
    TypeVar, Generic, Protocol, runtime_checkable
)
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


# Type variables for generic interfaces
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
ConfigType = TypeVar('ConfigType')
EventType = TypeVar('EventType')
MetricType = TypeVar('MetricType')


# Event System Interfaces

@dataclass
class Event:
    """Base event structure"""
    event_type: str
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    event_id: str
    correlation_id: Optional[str] = None


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@runtime_checkable
class EventHandler(Protocol):
    """Interface for event handlers"""
    
    async def handle_event(self, event: Event) -> None:
        """Handle an event"""
        ...
    
    def can_handle(self, event_type: str) -> bool:
        """Check if this handler can handle the event type"""
        ...
    
    @property
    def supported_events(self) -> List[str]:
        """List of supported event types"""
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Interface for event emitters"""
    
    async def emit(self, event: Event) -> None:
        """Emit an event"""
        ...
    
    def add_handler(self, handler: EventHandler) -> None:
        """Add an event handler"""
        ...
    
    def remove_handler(self, handler: EventHandler) -> None:
        """Remove an event handler"""
        ...


@runtime_checkable
class EventBus(Protocol):
    """Interface for event bus systems"""
    
    async def publish(self, event: Event, priority: EventPriority = EventPriority.MEDIUM) -> None:
        """Publish an event to the bus"""
        ...
    
    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to events of a specific type"""
        ...
    
    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from events"""
        ...
    
    async def start(self) -> None:
        """Start the event bus"""
        ...
    
    async def stop(self) -> None:
        """Stop the event bus"""
        ...


# Configuration Interfaces

@runtime_checkable
class ConfigProvider(Protocol[ConfigType]):
    """Interface for configuration providers"""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        ...
    
    def has(self, key: str) -> bool:
        """Check if configuration key exists"""
        ...
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        ...
    
    def reload(self) -> None:
        """Reload configuration"""
        ...
    
    def validate(self) -> List[str]:
        """Validate configuration and return errors"""
        ...


@runtime_checkable
class ConfigWatcher(Protocol):
    """Interface for configuration watchers"""
    
    def watch(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """Watch for configuration changes"""
        ...
    
    def unwatch(self, key: str) -> None:
        """Stop watching configuration key"""
        ...
    
    def start_watching(self) -> None:
        """Start the configuration watcher"""
        ...
    
    def stop_watching(self) -> None:
        """Stop the configuration watcher"""
        ...


# Cache Interface

@runtime_checkable
class CacheProvider(Protocol[K, V]):
    """Interface for cache providers"""
    
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache"""
        ...
    
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        ...
    
    async def delete(self, key: K) -> bool:
        """Delete value from cache"""
        ...
    
    async def exists(self, key: K) -> bool:
        """Check if key exists in cache"""
        ...
    
    async def clear(self) -> None:
        """Clear all cache entries"""
        ...
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        ...


# Database Interfaces

@runtime_checkable
class DatabaseConnection(Protocol):
    """Interface for database connections"""
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a database query"""
        ...
    
    async def fetch_one(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Fetch one row from database"""
        ...
    
    async def fetch_all(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch all rows from database"""
        ...
    
    async def begin_transaction(self) -> 'Transaction':
        """Begin a database transaction"""
        ...
    
    async def close(self) -> None:
        """Close the database connection"""
        ...
    
    @property
    def is_closed(self) -> bool:
        """Check if connection is closed"""
        ...


@runtime_checkable
class Transaction(Protocol):
    """Interface for database transactions"""
    
    async def commit(self) -> None:
        """Commit the transaction"""
        ...
    
    async def rollback(self) -> None:
        """Rollback the transaction"""
        ...
    
    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute query within transaction"""
        ...


@runtime_checkable
class DatabasePool(Protocol):
    """Interface for database connection pools"""
    
    async def acquire(self) -> DatabaseConnection:
        """Acquire a connection from the pool"""
        ...
    
    async def release(self, connection: DatabaseConnection) -> None:
        """Release a connection back to the pool"""
        ...
    
    async def close(self) -> None:
        """Close all connections in the pool"""
        ...
    
    @property
    def size(self) -> int:
        """Current pool size"""
        ...
    
    @property
    def available(self) -> int:
        """Number of available connections"""
        ...


# Authentication and Authorization Interfaces

@dataclass
class User:
    """User information"""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[str] = None
    permissions: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AuthToken:
    """Authentication token"""
    token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None
    scopes: List[str] = None
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


@runtime_checkable
class AuthProvider(Protocol):
    """Interface for authentication providers"""
    
    async def authenticate(self, credentials: Dict[str, Any]) -> Optional[AuthToken]:
        """Authenticate user with credentials"""
        ...
    
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token"""
        ...
    
    async def refresh_token(self, refresh_token: str) -> Optional[AuthToken]:
        """Refresh authentication token"""
        ...
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke authentication token"""
        ...
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        ...


@runtime_checkable
class AuthorizationProvider(Protocol):
    """Interface for authorization providers"""
    
    async def check_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for action on resource"""
        ...
    
    async def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user"""
        ...
    
    async def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant permission to user"""
        ...
    
    async def revoke_permission(self, user_id: str, permission: str) -> bool:
        """Revoke permission from user"""
        ...


# Metrics and Monitoring Interfaces

@dataclass
class Metric:
    """Base metric structure"""
    name: str
    value: Union[int, float, str]
    timestamp: datetime
    labels: Dict[str, str] = None
    metric_type: str = "gauge"
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@runtime_checkable
class MetricsCollector(Protocol):
    """Interface for metrics collectors"""
    
    def increment(self, name: str, value: int = 1, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric"""
        ...
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric"""
        ...
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a histogram value"""
        ...
    
    def timing(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a timing value"""
        ...
    
    def get_metrics(self) -> List[Metric]:
        """Get all collected metrics"""
        ...


@runtime_checkable
class MetricsExporter(Protocol):
    """Interface for metrics exporters"""
    
    async def export(self, metrics: List[Metric]) -> None:
        """Export metrics to external system"""
        ...
    
    def start(self) -> None:
        """Start the metrics exporter"""
        ...
    
    def stop(self) -> None:
        """Stop the metrics exporter"""
        ...


# Health Check Interfaces

@dataclass
class HealthStatus:
    """Health check status"""
    service_name: str
    is_healthy: bool
    status: str
    message: str
    details: Dict[str, Any] = None
    last_check: datetime = None
    response_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
        if self.last_check is None:
            self.last_check = datetime.utcnow()


@runtime_checkable
class HealthChecker(Protocol):
    """Interface for health checkers"""
    
    async def check_health(self) -> HealthStatus:
        """Perform health check"""
        ...
    
    async def check_dependency_health(self, dependency: str) -> HealthStatus:
        """Check health of a specific dependency"""
        ...
    
    async def get_overall_health(self) -> HealthStatus:
        """Get overall system health"""
        ...


# Plugin System Interfaces

@runtime_checkable
class Plugin(Protocol):
    """Interface for plugins"""
    
    @property
    def name(self) -> str:
        """Plugin name"""
        ...
    
    @property
    def version(self) -> str:
        """Plugin version"""
        ...
    
    @property
    def dependencies(self) -> List[str]:
        """Plugin dependencies"""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin"""
        ...
    
    async def start(self) -> None:
        """Start the plugin"""
        ...
    
    async def stop(self) -> None:
        """Stop the plugin"""
        ...
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        ...


@runtime_checkable
class PluginRegistry(Protocol):
    """Interface for plugin registries"""
    
    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin"""
        ...
    
    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin"""
        ...
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Get plugin by name"""
        ...
    
    def list_plugins(self) -> List[str]:
        """List all registered plugin names"""
        ...
    
    async def start_all_plugins(self) -> None:
        """Start all registered plugins"""
        ...
    
    async def stop_all_plugins(self) -> None:
        """Stop all registered plugins"""
        ...


# Rate Limiting Interface

@dataclass
class RateLimit:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: int
    burst_limit: Optional[int] = None
    key_generator: Optional[Callable[[Any], str]] = None


@runtime_checkable
class RateLimiter(Protocol):
    """Interface for rate limiters"""
    
    async def is_allowed(self, key: str, limit: RateLimit) -> bool:
        """Check if request is allowed"""
        ...
    
    async def get_remaining(self, key: str, limit: RateLimit) -> int:
        """Get remaining requests for key"""
        ...
    
    async def reset_key(self, key: str) -> None:
        """Reset rate limit for key"""
        ...
    
    async def get_stats(self, key: str) -> Dict[str, Any]:
        """Get rate limiting stats for key"""
        ...


# Circuit Breaker Interface

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@runtime_checkable
class CircuitBreaker(Protocol):
    """Interface for circuit breakers"""
    
    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection"""
        ...
    
    @property
    def state(self) -> CircuitBreakerState:
        """Current circuit breaker state"""
        ...
    
    @property
    def failure_count(self) -> int:
        """Current failure count"""
        ...
    
    def reset(self) -> None:
        """Reset circuit breaker"""
        ...


# Message Queue Interface

@runtime_checkable
class MessageQueue(Protocol[T]):
    """Interface for message queues"""
    
    async def put(self, item: T, priority: int = 0) -> None:
        """Put item in queue"""
        ...
    
    async def get(self, timeout: Optional[float] = None) -> T:
        """Get item from queue"""
        ...
    
    async def get_nowait(self) -> T:
        """Get item from queue without waiting"""
        ...
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        ...
    
    def qsize(self) -> int:
        """Get queue size"""
        ...
    
    async def join(self) -> None:
        """Wait for all tasks to complete"""
        ...
    
    def task_done(self) -> None:
        """Mark task as done"""
        ...


# File Storage Interface

@runtime_checkable
class FileStorage(Protocol):
    """Interface for file storage systems"""
    
    async def store(self, path: str, content: bytes, metadata: Dict[str, Any] = None) -> str:
        """Store file content and return file ID"""
        ...
    
    async def retrieve(self, file_id: str) -> Optional[bytes]:
        """Retrieve file content by ID"""
        ...
    
    async def delete(self, file_id: str) -> bool:
        """Delete file by ID"""
        ...
    
    async def exists(self, file_id: str) -> bool:
        """Check if file exists"""
        ...
    
    async def get_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        ...
    
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files with optional prefix filter"""
        ...


# AI Village Specific Interfaces

@runtime_checkable
class FogComputeProvider(Protocol):
    """Interface for fog computing providers"""
    
    async def submit_job(self, job_spec: Dict[str, Any]) -> str:
        """Submit compute job and return job ID"""
        ...
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status"""
        ...
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel job"""
        ...
    
    async def get_available_resources(self) -> Dict[str, Any]:
        """Get available compute resources"""
        ...
    
    async def register_node(self, node_info: Dict[str, Any]) -> str:
        """Register compute node"""
        ...


@runtime_checkable
class P2PNetworkProvider(Protocol):
    """Interface for P2P network providers"""
    
    async def connect_peer(self, peer_id: str) -> bool:
        """Connect to peer"""
        ...
    
    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from peer"""
        ...
    
    async def send_message(self, peer_id: str, message: Dict[str, Any]) -> bool:
        """Send message to peer"""
        ...
    
    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all peers"""
        ...
    
    async def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs"""
        ...
    
    async def get_peer_info(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get peer information"""
        ...


@runtime_checkable
class FederatedLearningProvider(Protocol):
    """Interface for federated learning providers"""
    
    async def start_training_round(self, round_config: Dict[str, Any]) -> str:
        """Start federated training round"""
        ...
    
    async def submit_model_update(self, round_id: str, client_id: str, update: Dict[str, Any]) -> bool:
        """Submit model update from client"""
        ...
    
    async def get_global_model(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get global model for round"""
        ...
    
    async def get_round_status(self, round_id: str) -> Dict[str, Any]:
        """Get training round status"""
        ...
    
    async def register_client(self, client_info: Dict[str, Any]) -> str:
        """Register federated learning client"""
        ...


@runtime_checkable
class ConstitutionalAIProvider(Protocol):
    """Interface for constitutional AI providers"""
    
    async def validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request against constitutional principles"""
        ...
    
    async def audit_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Audit AI decision for compliance"""
        ...
    
    async def get_principles(self) -> List[Dict[str, Any]]:
        """Get constitutional principles"""
        ...
    
    async def update_principles(self, principles: List[Dict[str, Any]]) -> bool:
        """Update constitutional principles"""
        ...


@runtime_checkable
class TransparencyProvider(Protocol):
    """Interface for transparency providers"""
    
    async def log_action(self, action: Dict[str, Any]) -> str:
        """Log action for transparency"""
        ...
    
    async def get_audit_trail(self, resource_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for resource"""
        ...
    
    async def verify_integrity(self, log_id: str) -> bool:
        """Verify log integrity"""
        ...
    
    async def generate_transparency_report(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transparency report"""
        ...


# Service Registry Interface

@runtime_checkable
class ServiceRegistry(Protocol):
    """Interface for service registries"""
    
    async def register_service(self, service_name: str, service_info: Dict[str, Any]) -> bool:
        """Register a service"""
        ...
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service"""
        ...
    
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover service by name"""
        ...
    
    async def list_services(self) -> List[str]:
        """List all registered services"""
        ...
    
    async def health_check_service(self, service_name: str) -> HealthStatus:
        """Check health of registered service"""
        ...


# Factory Interfaces

@runtime_checkable
class ServiceFactory(Protocol[T]):
    """Interface for service factories"""
    
    def create(self, config: Dict[str, Any]) -> T:
        """Create service instance"""
        ...
    
    def can_create(self, service_type: str) -> bool:
        """Check if factory can create service type"""
        ...
    
    @property
    def supported_types(self) -> List[str]:
        """List of supported service types"""
        ...


# Generic Service Interface

class IService(ABC, Generic[T]):
    """Generic service interface"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the service"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the service"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the service"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        pass
    
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """Perform health check"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Service name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Service version"""
        pass


# Export all interfaces
__all__ = [
    # Base types
    'Event', 'EventPriority', 'User', 'AuthToken', 'Metric', 'HealthStatus',
    'RateLimit', 'CircuitBreakerState',
    
    # Event system interfaces
    'EventHandler', 'EventEmitter', 'EventBus',
    
    # Configuration interfaces
    'ConfigProvider', 'ConfigWatcher',
    
    # Cache interface
    'CacheProvider',
    
    # Database interfaces
    'DatabaseConnection', 'Transaction', 'DatabasePool',
    
    # Authentication interfaces
    'AuthProvider', 'AuthorizationProvider',
    
    # Metrics interfaces
    'MetricsCollector', 'MetricsExporter',
    
    # Health check interface
    'HealthChecker',
    
    # Plugin system interfaces
    'Plugin', 'PluginRegistry',
    
    # Rate limiting interface
    'RateLimiter',
    
    # Circuit breaker interface
    'CircuitBreaker',
    
    # Message queue interface
    'MessageQueue',
    
    # File storage interface
    'FileStorage',
    
    # AI Village specific interfaces
    'FogComputeProvider', 'P2PNetworkProvider', 'FederatedLearningProvider',
    'ConstitutionalAIProvider', 'TransparencyProvider',
    
    # Service registry interface
    'ServiceRegistry',
    
    # Factory interface
    'ServiceFactory',
    
    # Generic service interface
    'IService',
]