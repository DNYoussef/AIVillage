"""
Service Interface Contracts for Agent Forge Microservices

This module defines the interface contracts and data models used for 
communication between the five microservices in the Agent Forge system.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid


# =============================================================================
# SHARED DATA MODELS
# =============================================================================

class ServiceStatus(str, Enum):
    """Standard status values across all services."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelPhase(str, Enum):
    """Model training phases."""
    COGNATE = "cognate"
    EVOMERGE = "evomerge"
    QUIETSTAR = "quietstar"
    BITNET = "bitnet"
    FORGE_TRAINING = "forge-training"
    TOOL_PERSONA = "tool-persona"
    ADAS = "adas"
    FINAL_COMPRESSION = "final-compression"


# =============================================================================
# TRAINING SERVICE INTERFACES
# =============================================================================

class TrainingConfig(BaseModel):
    """Configuration for training jobs."""
    max_steps: int = Field(..., description="Maximum training steps")
    batch_size: int = Field(..., description="Training batch size")
    learning_rate: float = Field(..., description="Learning rate")
    output_dir: str = Field(..., description="Output directory for models")
    max_train_samples: Optional[int] = Field(None, description="Limit training samples")
    max_eval_samples: Optional[int] = Field(None, description="Limit evaluation samples")
    use_grokfast: bool = Field(True, description="Enable GrokFast optimization")


class TrainingJob(BaseModel):
    """Training job definition."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    phase: ModelPhase
    config: TrainingConfig
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0)
    current_step: int = Field(default=0)
    message: str = Field(default="")
    error_message: Optional[str] = None


class TrainingProgress(BaseModel):
    """Training progress update."""
    job_id: str
    progress: float = Field(..., ge=0.0, le=1.0)
    current_step: int
    total_steps: int
    message: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ITrainingService(ABC):
    """Interface for the Training Service."""
    
    @abstractmethod
    async def start_training_job(self, job: TrainingJob) -> str:
        """Start a new training job and return job ID."""
        pass
    
    @abstractmethod
    async def get_job_status(self, job_id: str) -> TrainingJob:
        """Get current status of a training job."""
        pass
    
    @abstractmethod
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        pass
    
    @abstractmethod
    async def list_jobs(self, phase: Optional[ModelPhase] = None) -> List[TrainingJob]:
        """List training jobs, optionally filtered by phase."""
        pass


# =============================================================================
# MODEL SERVICE INTERFACES
# =============================================================================

class ModelMetadata(BaseModel):
    """Model metadata and information."""
    model_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    phase: ModelPhase
    version: str = Field(default="1.0.0")
    file_path: str
    file_size: int
    checksum: str
    created_at: datetime = Field(default_factory=datetime.now)
    training_job_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    is_winner: bool = Field(default=False)
    tags: List[str] = Field(default_factory=list)


class ModelHandoff(BaseModel):
    """Model handoff between phases."""
    handoff_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_phase: ModelPhase
    to_phase: ModelPhase
    model_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    status: TaskStatus = Field(default=TaskStatus.PENDING)


class ModelExportRequest(BaseModel):
    """Request to export models."""
    model_ids: List[str] = Field(..., description="List of model IDs to export")
    export_format: str = Field(default="pytorch", description="Export format")
    include_metadata: bool = Field(default=True)
    compression: bool = Field(default=True)


class IModelService(ABC):
    """Interface for the Model Service."""
    
    @abstractmethod
    async def save_model(self, metadata: ModelMetadata, model_data: bytes) -> str:
        """Save model and return model ID."""
        pass
    
    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        pass
    
    @abstractmethod
    async def load_model_data(self, model_id: str) -> Optional[bytes]:
        """Load model binary data."""
        pass
    
    @abstractmethod
    async def list_models(self, phase: Optional[ModelPhase] = None) -> List[ModelMetadata]:
        """List models, optionally filtered by phase."""
        pass
    
    @abstractmethod
    async def create_handoff(self, handoff: ModelHandoff) -> str:
        """Create model handoff between phases."""
        pass
    
    @abstractmethod
    async def export_models(self, request: ModelExportRequest) -> str:
        """Export models and return export path."""
        pass


# =============================================================================
# WEBSOCKET SERVICE INTERFACES
# =============================================================================

class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    type: str = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    source_service: str = Field(..., description="Service that generated the message")


class ConnectionInfo(BaseModel):
    """WebSocket connection information."""
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    connected_at: datetime = Field(default_factory=datetime.now)
    client_ip: str
    user_agent: Optional[str] = None
    subscriptions: List[str] = Field(default_factory=list)


class IWebSocketService(ABC):
    """Interface for the WebSocket Service."""
    
    @abstractmethod
    async def broadcast_message(self, message: WebSocketMessage) -> bool:
        """Broadcast message to all connected clients."""
        pass
    
    @abstractmethod
    async def send_to_connection(self, connection_id: str, message: WebSocketMessage) -> bool:
        """Send message to specific connection."""
        pass
    
    @abstractmethod
    async def subscribe_to_topic(self, connection_id: str, topic: str) -> bool:
        """Subscribe connection to a topic."""
        pass
    
    @abstractmethod
    async def get_active_connections(self) -> List[ConnectionInfo]:
        """Get list of active connections."""
        pass


# =============================================================================
# API SERVICE INTERFACES
# =============================================================================

class APIResponse(BaseModel):
    """Standard API response structure."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class PhaseStartRequest(BaseModel):
    """Request to start a training phase."""
    phase_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    real_training: bool = Field(default=True)


class ChatRequest(BaseModel):
    """Chat request with a model."""
    model_id: str
    message: str
    context: Dict[str, Any] = Field(default_factory=dict)


class IAPIService(ABC):
    """Interface for the API Service."""
    
    @abstractmethod
    async def start_phase(self, request: PhaseStartRequest) -> APIResponse:
        """Start a training phase."""
        pass
    
    @abstractmethod
    async def get_phase_status(self, phase_id: str) -> APIResponse:
        """Get status of a training phase."""
        pass
    
    @abstractmethod
    async def chat_with_model(self, request: ChatRequest) -> APIResponse:
        """Chat with a trained model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> APIResponse:
        """Health check endpoint."""
        pass


# =============================================================================
# MONITORING SERVICE INTERFACES
# =============================================================================

class SystemMetrics(BaseModel):
    """System performance metrics."""
    cpu_usage: float = Field(..., ge=0.0, le=1.0)
    memory_usage: float = Field(..., ge=0.0, le=1.0)
    disk_usage: float = Field(..., ge=0.0, le=1.0)
    network_io: Dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class ServiceHealth(BaseModel):
    """Service health status."""
    service_name: str
    status: ServiceStatus
    response_time: float = Field(..., description="Response time in milliseconds")
    last_check: datetime = Field(default_factory=datetime.now)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    uptime_percentage: float = Field(default=1.0, ge=0.0, le=1.0)


class Alert(BaseModel):
    """System alert."""
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    severity: str = Field(..., description="Alert severity (info, warning, error, critical)")
    service: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    resolved: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IMonitoringService(ABC):
    """Interface for the Monitoring Service."""
    
    @abstractmethod
    async def record_metrics(self, service_name: str, metrics: SystemMetrics) -> bool:
        """Record system metrics for a service."""
        pass
    
    @abstractmethod
    async def get_service_health(self, service_name: str) -> ServiceHealth:
        """Get health status of a service."""
        pass
    
    @abstractmethod
    async def create_alert(self, alert: Alert) -> str:
        """Create a new alert."""
        pass
    
    @abstractmethod
    async def get_system_overview(self) -> Dict[str, Any]:
        """Get overall system health overview."""
        pass


# =============================================================================
# EVENT DEFINITIONS
# =============================================================================

class Event(BaseModel):
    """Base event structure."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    source_service: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)


# Training Events
class TrainingStartedEvent(Event):
    event_type: str = "training_started"
    

class TrainingProgressEvent(Event):
    event_type: str = "training_progress"


class TrainingCompletedEvent(Event):
    event_type: str = "training_completed"


class TrainingFailedEvent(Event):
    event_type: str = "training_failed"


# Model Events
class ModelSavedEvent(Event):
    event_type: str = "model_saved"


class ModelHandoffCreatedEvent(Event):
    event_type: str = "model_handoff_created"


# System Events
class ServiceHealthChangedEvent(Event):
    event_type: str = "service_health_changed"


class AlertCreatedEvent(Event):
    event_type: str = "alert_created"