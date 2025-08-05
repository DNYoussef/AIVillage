"""Standardized Training Interface

This module defines the standard interface for training operations,
model management, and training pipeline coordination.
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from core import ErrorContext


class TrainingStatus(Enum):
    """Standard training status values."""

    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EVALUATING = "evaluating"


class ModelStatus(Enum):
    """Standard model status values."""

    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    FINE_TUNING = "fine_tuning"
    COMPRESSED = "compressed"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class TrainingPhase(Enum):
    """Standard training phases."""

    PHASE_1_FOUNDATION = "phase_1_foundation"
    PHASE_2_CORE_TRAINING = "phase_2_core_training"
    PHASE_3_SELF_MODELING = "phase_3_self_modeling"
    PHASE_4_INTEGRATION = "phase_4_integration"
    PHASE_5_DEPLOYMENT = "phase_5_deployment"
    CUSTOM = "custom"


@dataclass
class TrainingMetrics:
    """Training metrics and statistics."""

    epoch: int = 0
    total_epochs: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    accuracy: float = 0.0
    validation_loss: float = 0.0
    validation_accuracy: float = 0.0
    training_time_seconds: float = 0.0
    samples_processed: int = 0
    tokens_processed: int = 0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0

    # Geometric metrics (for Agent Forge)
    intrinsic_dimensionality: float | None = None
    compression_ratio: float | None = None
    entropy: float | None = None
    grokking_detected: bool = False

    # Custom metrics
    custom_metrics: dict[str, float] = field(default_factory=dict)

    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ModelMetadata:
    """Metadata for ML models."""

    model_id: str
    name: str
    description: str
    model_type: str
    architecture: str
    version: str
    framework: str = "pytorch"
    parameters_count: int = 0
    model_size_mb: float = 0.0
    training_phase: TrainingPhase | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training operations."""

    training_id: str
    model_id: str
    dataset_path: str
    output_path: str

    # Training parameters
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    optimizer: str = "adam"
    scheduler: str | None = None

    # Advanced parameters
    mixed_precision: bool = False
    gradient_clipping: float | None = None
    weight_decay: float = 0.0
    dropout_rate: float = 0.1

    # Agent Forge specific
    enable_quiet_star: bool = False
    enable_geometric_monitoring: bool = True
    grokking_threshold: float = 0.5
    phase: TrainingPhase | None = None

    # Validation
    validation_split: float = 0.1
    validation_frequency: int = 1
    early_stopping_patience: int = 5

    # Checkpointing
    save_frequency: int = 1000
    max_checkpoints: int = 5

    # Monitoring
    log_frequency: int = 100
    enable_tensorboard: bool = True
    enable_wandb: bool = False

    # Resources
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True

    # Custom configuration
    custom_config: dict[str, Any] = field(default_factory=dict)


class ModelInterface(ABC):
    """Standard interface for ML models."""

    def __init__(self, metadata: ModelMetadata):
        self.metadata = metadata
        self.status = ModelStatus.UNTRAINED
        self._model = None
        self._device = None

    @abstractmethod
    async def load_model(self, model_path: str) -> bool:
        """Load model from file.

        Args:
            model_path: Path to model file

        Returns:
            bool: True if model loaded successfully
        """

    @abstractmethod
    async def save_model(self, model_path: str) -> bool:
        """Save model to file.

        Args:
            model_path: Path to save model

        Returns:
            bool: True if model saved successfully
        """

    @abstractmethod
    async def forward(self, inputs: Any, **kwargs) -> Any:
        """Forward pass through model.

        Args:
            inputs: Model inputs
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""

    @abstractmethod
    def get_parameter_count(self) -> int:
        """Get total number of model parameters."""

    # Optional methods

    async def compress_model(self, compression_config: dict[str, Any]) -> bool:
        """Compress model using specified configuration."""
        # Default implementation - can be overridden
        return False

    async def fine_tune(self, dataset: Any, config: TrainingConfig) -> bool:
        """Fine-tune model on new dataset."""
        # Default implementation - can be overridden
        return False

    def to_device(self, device: str) -> None:
        """Move model to specified device."""
        self._device = device

    def get_device(self) -> str | None:
        """Get current device."""
        return self._device

    def set_status(self, status: ModelStatus) -> None:
        """Set model status."""
        self.status = status
        self.metadata.updated_at = datetime.now()

    def get_status(self) -> ModelStatus:
        """Get current model status."""
        return self.status


class TrainingInterface(ABC):
    """Standard interface for training operations.

    This interface defines the standard API for training ML models
    within the Agent Forge pipeline and other training systems.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.status = TrainingStatus.INITIALIZING
        self.metrics = TrainingMetrics()
        self.model: ModelInterface | None = None
        self.callbacks: list[Callable] = []
        self._training_task: asyncio.Task | None = None

    @abstractmethod
    async def initialize(self, model: ModelInterface) -> bool:
        """Initialize training with model.

        Args:
            model: Model to train

        Returns:
            bool: True if initialization successful
        """

    @abstractmethod
    async def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train single epoch.

        Args:
            epoch: Epoch number

        Returns:
            Training metrics for the epoch
        """

    @abstractmethod
    async def validate(self) -> TrainingMetrics:
        """Run validation and return metrics.

        Returns:
            Validation metrics
        """

    @abstractmethod
    async def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save training checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint

        Returns:
            bool: True if checkpoint saved successfully
        """

    @abstractmethod
    async def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            bool: True if checkpoint loaded successfully
        """

    # High-level training control

    async def start_training(self) -> bool:
        """Start training process.

        Returns:
            bool: True if training started successfully
        """
        if self.status == TrainingStatus.TRAINING:
            return False  # Already training

        self.set_status(TrainingStatus.TRAINING)

        try:
            self._training_task = asyncio.create_task(self._training_loop())
            return True
        except Exception:
            self.set_status(TrainingStatus.FAILED)
            return False

    async def pause_training(self) -> bool:
        """Pause training process."""
        if self.status == TrainingStatus.TRAINING:
            self.set_status(TrainingStatus.PAUSED)
            return True
        return False

    async def resume_training(self) -> bool:
        """Resume paused training."""
        if self.status == TrainingStatus.PAUSED:
            self.set_status(TrainingStatus.TRAINING)
            return True
        return False

    async def stop_training(self) -> bool:
        """Stop training process."""
        if self._training_task and not self._training_task.done():
            self._training_task.cancel()
            try:
                await self._training_task
            except asyncio.CancelledError:
                pass

        self.set_status(TrainingStatus.CANCELLED)
        return True

    async def _training_loop(self) -> None:
        """Main training loop."""
        try:
            for epoch in range(self.metrics.epoch, self.config.epochs):
                if self.status != TrainingStatus.TRAINING:
                    break  # Training paused or stopped

                # Train epoch
                epoch_metrics = await self.train_epoch(epoch)
                self.metrics = epoch_metrics

                # Validation
                if epoch % self.config.validation_frequency == 0:
                    val_metrics = await self.validate()
                    self.metrics.validation_loss = val_metrics.validation_loss
                    self.metrics.validation_accuracy = val_metrics.validation_accuracy

                # Checkpointing
                if epoch % self.config.save_frequency == 0:
                    checkpoint_path = f"{self.config.output_path}/checkpoint_epoch_{epoch}.pt"
                    await self.save_checkpoint(checkpoint_path)

                # Early stopping
                if self._should_early_stop():
                    break

                # Callbacks
                await self._run_callbacks("epoch_end", epoch, self.metrics)

            self.set_status(TrainingStatus.COMPLETED)

        except Exception:
            self.set_status(TrainingStatus.FAILED)
            raise

    def _should_early_stop(self) -> bool:
        """Check if training should stop early."""
        # Simple implementation - can be overridden
        return False

    # Callback management

    def add_callback(self, callback: Callable) -> None:
        """Add training callback."""
        self.callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> None:
        """Remove training callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    async def _run_callbacks(self, event: str, *args, **kwargs) -> None:
        """Run callbacks for specific event."""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, *args, **kwargs)
                else:
                    callback(event, *args, **kwargs)
            except Exception as e:
                # Log error but continue
                print(f"Error in callback: {e}")

    # Status and metrics

    def set_status(self, status: TrainingStatus) -> None:
        """Set training status."""
        self.status = status

    def get_status(self) -> TrainingStatus:
        """Get current training status."""
        return self.status

    def get_metrics(self) -> TrainingMetrics:
        """Get current training metrics."""
        return self.metrics

    def update_metrics(self, **metrics) -> None:
        """Update training metrics."""
        for key, value in metrics.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

        self.metrics.last_updated = datetime.now()

    # Geometric monitoring (Agent Forge specific)

    async def monitor_geometry(self, hidden_states: Any) -> dict[str, float]:
        """Monitor geometric properties during training.

        Args:
            hidden_states: Hidden states from model

        Returns:
            Dictionary of geometric metrics
        """
        if not self.config.enable_geometric_monitoring:
            return {}

        try:
            # Import geometry monitoring
            from agent_forge.geometry.snapshot import snapshot

            geom_state = snapshot(hidden_states)

            # Update metrics
            self.metrics.intrinsic_dimensionality = geom_state["ID_nl"]
            self.metrics.compression_ratio = geom_state["ratio"]
            self.metrics.entropy = geom_state["entropy"]

            # Check for grokking
            if geom_state["ID_nl"] < self.config.grokking_threshold:
                self.metrics.grokking_detected = True

            return {
                "intrinsic_dimensionality": geom_state["ID_nl"],
                "compression_ratio": geom_state["ratio"],
                "entropy": geom_state["entropy"],
            }

        except ImportError:
            # Geometry monitoring not available
            return {}
        except Exception as e:
            print(f"Error in geometry monitoring: {e}")
            return {}

    def create_error_context(self, operation: str, **metadata) -> ErrorContext:
        """Create error context for training operations."""
        from core import ErrorContext

        return ErrorContext(
            component=f"Training.{self.config.training_id}",
            operation=operation,
            metadata={
                "model_id": self.config.model_id,
                "phase": self.config.phase.value if self.config.phase else None,
                "epoch": self.metrics.epoch,
                **metadata,
            },
        )


# Utility functions


def create_training_config(model_id: str, dataset_path: str, output_path: str, **config_kwargs) -> TrainingConfig:
    """Create training configuration with auto-generated ID.

    Args:
        model_id: ID of model to train
        dataset_path: Path to training dataset
        output_path: Path for training outputs
        **config_kwargs: Additional configuration

    Returns:
        TrainingConfig instance
    """
    import uuid

    return TrainingConfig(
        training_id=str(uuid.uuid4()),
        model_id=model_id,
        dataset_path=dataset_path,
        output_path=output_path,
        **config_kwargs,
    )


def create_model_metadata(name: str, model_type: str, architecture: str, **metadata_kwargs) -> ModelMetadata:
    """Create model metadata with auto-generated ID.

    Args:
        name: Model name
        model_type: Type of model
        architecture: Model architecture
        **metadata_kwargs: Additional metadata

    Returns:
        ModelMetadata instance
    """
    import uuid

    return ModelMetadata(
        model_id=str(uuid.uuid4()),
        name=name,
        model_type=model_type,
        architecture=architecture,
        **metadata_kwargs,
    )


def validate_training_interface(trainer: Any) -> bool:
    """Validate that an object implements TrainingInterface.

    Args:
        trainer: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "initialize",
        "train_epoch",
        "validate",
        "save_checkpoint",
        "load_checkpoint",
    ]

    for method in required_methods:
        if not hasattr(trainer, method) or not callable(getattr(trainer, method)):
            return False

    required_attributes = ["config", "status", "metrics"]
    for attr in required_attributes:
        if not hasattr(trainer, attr):
            return False

    return True


def validate_model_interface(model: Any) -> bool:
    """Validate that an object implements ModelInterface.

    Args:
        model: Object to validate

    Returns:
        bool: True if object implements interface correctly
    """
    required_methods = [
        "load_model",
        "save_model",
        "forward",
        "get_model_info",
        "get_parameter_count",
    ]

    for method in required_methods:
        if not hasattr(model, method) or not callable(getattr(model, method)):
            return False

    required_attributes = ["metadata", "status"]
    for attr in required_attributes:
        if not hasattr(model, attr):
            return False

    return True
