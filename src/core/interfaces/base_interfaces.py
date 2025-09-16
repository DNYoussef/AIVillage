"""
Base Interface Abstractions

Defines core interfaces to reduce coupling between components
and improve testability and maintainability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime
import torch
import torch.nn as nn

from ..exceptions import AIVillageError


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the component with the given configuration."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        ...


@runtime_checkable
class Validatable(Protocol):
    """Protocol for components that can validate themselves."""
    
    def validate(self) -> bool:
        """Validate the component state. Returns True if valid."""
        ...
    
    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable components."""
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        ...
    
    def deserialize(self, data: Dict[str, Any]) -> None:
        """Deserialize component from dictionary."""
        ...


class BaseModel(ABC):
    """Abstract base class for all AI models."""
    
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
        self._model: Optional[nn.Module] = None
        self._is_loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass
    
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    @property
    def model(self) -> nn.Module:
        """Get the underlying PyTorch model."""
        if not self._is_loaded:
            raise AIVillageError(f"Model {self.model_id} is not loaded")
        return self._model


class BasePhase(ABC):
    """Abstract base class for pipeline phases."""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        self.phase_name = phase_name
        self.config = config
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
    
    @abstractmethod
    async def run(self, model: BaseModel) -> 'PhaseResult':
        """Execute the phase."""
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """Validate phase configuration."""
        pass
    
    def get_duration(self) -> Optional[float]:
        """Get phase execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class PhaseResult:
    """Result of a phase execution."""
    
    def __init__(
        self,
        success: bool,
        model: Optional[BaseModel] = None,
        phase_name: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.model = model
        self.phase_name = phase_name
        self.metrics = metrics or {}
        self.error = error
        self.artifacts = artifacts or {}
        self.timestamp = datetime.now()


class BaseDataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data."""
        pass
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Validate input data format."""
        pass


class BaseTrainer(ABC):
    """Abstract base class for model trainers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Any]:
        """Execute a single training step."""
        pass
    
    @abstractmethod
    def evaluate_step(self, batch: Any) -> Dict[str, Any]:
        """Execute a single evaluation step."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        pass


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    @abstractmethod
    def evaluate(self, model: BaseModel, dataset: Any) -> Dict[str, Any]:
        """Evaluate model on dataset."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get evaluation metrics."""
        pass


class BaseOptimizer(ABC):
    """Abstract base class for optimizers."""
    
    @abstractmethod
    def optimize(self, model: BaseModel, objective: Any) -> BaseModel:
        """Optimize model according to objective."""
        pass


class ResourceManager(ABC):
    """Abstract base class for resource management."""
    
    @abstractmethod
    def allocate_gpu(self, memory_gb: float) -> str:
        """Allocate GPU with specified memory."""
        pass
    
    @abstractmethod
    def allocate_cpu(self, cores: int) -> str:
        """Allocate CPU cores."""
        pass
    
    @abstractmethod
    def deallocate_resource(self, resource_id: str) -> None:
        """Deallocate a resource."""
        pass
    
    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        pass


class EventEmitter(ABC):
    """Abstract base class for event emission."""
    
    @abstractmethod
    def emit(self, event: str, data: Any = None) -> None:
        """Emit an event with optional data."""
        pass
    
    @abstractmethod
    def subscribe(self, event: str, callback: callable) -> None:
        """Subscribe to an event."""
        pass
    
    @abstractmethod
    def unsubscribe(self, event: str, callback: callable) -> None:
        """Unsubscribe from an event."""
        pass


class BaseMonitor(ABC):
    """Abstract base class for monitoring components."""
    
    @abstractmethod
    def start_monitoring(self) -> None:
        """Start monitoring."""
        pass
    
    @abstractmethod
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics."""
        pass


class BaseFactory(ABC):
    """Abstract base class for factory patterns."""
    
    @abstractmethod
    def create(self, type_name: str, config: Dict[str, Any]) -> Any:
        """Create an instance of the specified type."""
        pass
    
    @abstractmethod
    def register(self, type_name: str, constructor: callable) -> None:
        """Register a constructor for a type."""
        pass
    
    @abstractmethod
    def get_available_types(self) -> List[str]:
        """Get list of available types."""
        pass


class BaseRepository(ABC):
    """Abstract base class for data repositories."""
    
    @abstractmethod
    def save(self, entity: Any) -> str:
        """Save an entity and return its ID."""
        pass
    
    @abstractmethod
    def load(self, entity_id: str) -> Any:
        """Load an entity by ID."""
        pass
    
    @abstractmethod
    def delete(self, entity_id: str) -> None:
        """Delete an entity by ID."""
        pass
    
    @abstractmethod
    def find(self, criteria: Dict[str, Any]) -> List[Any]:
        """Find entities matching criteria."""
        pass


# Type aliases for commonly used interfaces
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, Any]
ArtifactsDict = Dict[str, Any]