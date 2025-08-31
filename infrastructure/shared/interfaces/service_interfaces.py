"""
Shared Service Interfaces for Phase 3 Architecture
Common interfaces and protocols for fog and graph services.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from enum import Enum
import asyncio

class ServiceStatus(Enum):
    """Common service status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class HealthStatus(Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

# Base Service Protocol
class BaseService(Protocol):
    """Base protocol that all services must implement."""
    
    @property
    def service_name(self) -> str: ...
    
    @property 
    def service_version(self) -> str: ...
    
    @property
    def status(self) -> ServiceStatus: ...
    
    async def start(self) -> bool: ...
    
    async def stop(self) -> bool: ...
    
    async def health_check(self) -> Dict[str, Any]: ...
    
    async def get_metrics(self) -> Dict[str, Any]: ...

# Fog Computing Service Interfaces
class FogOrchestrationServiceInterface(ABC):
    """Interface for fog orchestration service."""
    
    @abstractmethod
    async def start(self) -> bool:
        """Start the fog system."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the fog system gracefully."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Get system health status."""
        pass
    
    @abstractmethod
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        pass
    
    @abstractmethod
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        pass

class FogHarvestingServiceInterface(ABC):
    """Interface for fog harvesting service."""
    
    @abstractmethod
    async def start_harvesting(self) -> bool:
        """Start mobile compute harvesting."""
        pass
    
    @abstractmethod
    async def stop_harvesting(self) -> bool:
        """Stop harvesting operations."""
        pass
    
    @abstractmethod
    async def register_device(self, device_info: Dict[str, Any]) -> str:
        """Register a harvesting device."""
        pass
    
    @abstractmethod
    async def get_harvesting_stats(self) -> Dict[str, Any]:
        """Get harvesting statistics."""
        pass

class FogPrivacyServiceInterface(ABC):
    """Interface for fog privacy service."""
    
    @abstractmethod
    async def create_circuit(self, privacy_level: str, client_id: str) -> str:
        """Create an onion circuit for privacy."""
        pass
    
    @abstractmethod
    async def route_task(self, task_data: bytes, privacy_level: str) -> bool:
        """Route a task through privacy layer."""
        pass
    
    @abstractmethod
    async def create_hidden_service(self, service_config: Dict[str, Any]) -> str:
        """Create a hidden service."""
        pass

# Graph Processing Service Interfaces
class GapDetectionServiceInterface(ABC):
    """Interface for gap detection service."""
    
    @abstractmethod
    async def detect_gaps(self, graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect knowledge gaps in the graph."""
        pass
    
    @abstractmethod
    async def analyze_completeness(self, graph_data: Dict[str, Any]) -> float:
        """Analyze graph completeness score."""
        pass

class KnowledgeProposalServiceInterface(ABC):
    """Interface for knowledge proposal service."""
    
    @abstractmethod
    async def generate_proposals(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate knowledge proposals for gaps."""
        pass
    
    @abstractmethod
    async def rank_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank proposals by quality."""
        pass

class GraphAnalysisServiceInterface(ABC):
    """Interface for graph analysis service."""
    
    @abstractmethod
    async def analyze_structure(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze graph structure."""
        pass
    
    @abstractmethod
    async def calculate_metrics(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate graph metrics."""
        pass

# Event System
class ServiceEvent:
    """Base class for service events."""
    
    def __init__(self, event_type: str, source_service: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.source_service = source_service
        self.data = data
        self.timestamp = asyncio.get_event_loop().time()

class EventHandler(Protocol):
    """Protocol for event handlers."""
    
    async def handle_event(self, event: ServiceEvent) -> None: ...

class EventBus(ABC):
    """Abstract event bus for service communication."""
    
    @abstractmethod
    async def publish(self, event: ServiceEvent) -> None:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Subscribe to event type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Unsubscribe from event type."""
        pass

# Service Discovery
class ServiceDiscovery(ABC):
    """Abstract service discovery interface."""
    
    @abstractmethod
    async def register_service(self, service_name: str, service_info: Dict[str, Any]) -> None:
        """Register a service."""
        pass
    
    @abstractmethod
    async def discover_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Discover a service by name."""
        pass
    
    @abstractmethod
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all registered services."""
        pass

# Configuration Management
class ConfigurationProvider(ABC):
    """Abstract configuration provider."""
    
    @abstractmethod
    async def get_config(self, service_name: str) -> Dict[str, Any]:
        """Get configuration for a service."""
        pass
    
    @abstractmethod
    async def update_config(self, service_name: str, config: Dict[str, Any]) -> None:
        """Update service configuration."""
        pass

# Metrics Collection
class MetricsCollector(ABC):
    """Abstract metrics collector."""
    
    @abstractmethod
    async def record_metric(self, metric_name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record a metric value."""
        pass
    
    @abstractmethod
    async def get_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get metrics for a service."""
        pass

__all__ = [
    'ServiceStatus', 'HealthStatus', 'BaseService', 'ServiceEvent', 'EventHandler',
    'FogOrchestrationServiceInterface', 'FogHarvestingServiceInterface', 
    'FogPrivacyServiceInterface', 'GapDetectionServiceInterface',
    'KnowledgeProposalServiceInterface', 'GraphAnalysisServiceInterface',
    'EventBus', 'ServiceDiscovery', 'ConfigurationProvider', 'MetricsCollector'
]