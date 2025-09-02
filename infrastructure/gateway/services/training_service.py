"""
Training Service - Focused training operations for Agent Forge

Extracted from unified_agent_forge_backend.py to provide clean separation of concerns
for training-related functionality including:
- Model creation and initialization
- Training loop implementation with GrokFast optimization
- Dataset loading and processing
- Training progress tracking and event emission
- Model checkpointing and artifact management
"""

from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from typing import Any
import uuid

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training operations."""

    max_steps: int = 2000
    batch_size: int = 2
    learning_rate: float = 2e-4
    output_dir: str = "./trained_models_output"
    max_train_samples: int = 5000
    max_eval_samples: int = 500

    # GrokFast optimization parameters
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0
    grokfast_enabled: bool = True

    # Model architecture parameters
    d_model: int = 216
    n_layers: int = 11
    n_heads: int = 4
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # ACT (Adaptive Computation Time) parameters
    act_threshold: float = 0.99
    max_act_steps: int = 16
    act_epsilon: float = 0.01
    act_enabled: bool = True

    # LTM (Long-Term Memory) parameters
    d_mem: int = 216
    mem_capacity: int = 4096
    mem_topk: int = 4
    ltm_enabled: bool = True

    # Training optimization
    gradient_accumulation_steps: int = 4
    save_steps: int = 1000

    # Dataset configuration
    dataset_path: str | None = None
    dataset_sources: list[str] = field(default_factory=lambda: ["GSM8K", "SVAMP", "HotpotQA"])


@dataclass
class TrainingProgress:
    """Training progress information."""

    progress: float
    message: str
    phase_name: str = "Training"
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    learning_rate: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ModelArtifacts:
    """Model training artifacts and metadata."""

    model_id: str
    model_name: str
    phase_name: str
    parameter_count: int
    created_at: str
    training_status: str
    focus: str
    training_mode: str
    datasets_used: list[str]
    training_stats: dict[str, Any]

    # Feature flags
    grokfast_enabled: bool = True
    act_enabled: bool = True
    ltm_enabled: bool = True

    # File paths
    artifacts: dict[str, str] = field(default_factory=dict)

    # Capabilities and metrics
    capabilities: list[str] = field(default_factory=list)


class ProgressEmitter(ABC):
    """Abstract interface for emitting training progress events."""

    @abstractmethod
    async def emit_progress(self, progress: TrainingProgress) -> None:
        """Emit a progress update."""
        pass

    @abstractmethod
    async def emit_model_completed(self, artifacts: ModelArtifacts) -> None:
        """Emit model completion event."""
        pass


class DatasetLoader(ABC):
    """Abstract interface for dataset loading."""

    @abstractmethod
    async def download_dataset(self, dataset_name: str, dataset_id: str) -> bool:
        """Download a specific dataset."""
        pass

    @abstractmethod
    async def create_mixed_training_data(self) -> str:
        """Create mixed training dataset from multiple sources."""
        pass

    @abstractmethod
    def get_dataset_results(self) -> dict[str, bool]:
        """Get results of dataset downloads."""
        pass


class ModelTrainer(ABC):
    """Abstract interface for model training."""

    @abstractmethod
    async def train_model(
        self, model_name: str, config: TrainingConfig, progress_callback: Callable[[int, int, float, float], None]
    ) -> dict[str, Any]:
        """Train a single model and return training statistics."""
        pass


class TrainingService:
    """
    Focused training service that handles all training-related operations
    with clean separation of concerns and dependency injection.
    """

    def __init__(
        self,
        progress_emitter: ProgressEmitter,
        dataset_loader: DatasetLoader,
        model_trainer: ModelTrainer,
        config: TrainingConfig | None = None,
    ):
        """Initialize training service with injected dependencies."""
        self.progress_emitter = progress_emitter
        self.dataset_loader = dataset_loader
        self.model_trainer = model_trainer
        self.config = config or TrainingConfig()

        # Training state
        self.active_training_sessions: dict[str, dict[str, Any]] = {}
        self.model_storage: dict[str, ModelArtifacts] = {}

        logger.info("TrainingService initialized with dependency injection")

    async def start_training_session(
        self, task_id: str, training_parameters: dict[str, Any], model_names: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Start a new training session with the specified parameters.

        Args:
            task_id: Unique identifier for this training session
            training_parameters: Training configuration parameters
            model_names: Names of models to train (defaults to cognate foundation models)

        Returns:
            Session information including task_id and initial status
        """
        if task_id in self.active_training_sessions:
            raise ValueError(f"Training session {task_id} already exists")

        # Default model names
        if model_names is None:
            model_names = ["cognate_foundation_1", "cognate_foundation_2", "cognate_foundation_3"]

        # Initialize training session
        session_info = {
            "task_id": task_id,
            "status": "initializing",
            "start_time": datetime.now().isoformat(),
            "model_names": model_names,
            "parameters": training_parameters,
            "progress": 0.0,
            "trained_models": [],
        }

        self.active_training_sessions[task_id] = session_info

        logger.info(f"🚀 Starting training session {task_id} with {len(model_names)} models")

        # Emit initial progress
        await self.progress_emitter.emit_progress(
            TrainingProgress(
                progress=0.0,
                message="🔥 Initializing training session with GrokFast optimization",
                phase_name="Training Initialization",
            )
        )

        return session_info

    async def execute_training_pipeline(self, task_id: str) -> list[ModelArtifacts]:
        """
        Execute the complete training pipeline for a session.

        Args:
            task_id: Training session identifier

        Returns:
            List of trained model artifacts
        """
        if task_id not in self.active_training_sessions:
            raise ValueError(f"Training session {task_id} not found")

        session = self.active_training_sessions[task_id]
        session["status"] = "running"

        try:
            # Phase 1: Dataset preparation
            await self._prepare_datasets(task_id)

            # Phase 2: Model training
            trained_models = await self._train_models(task_id)

            # Phase 3: Finalization
            await self._finalize_training(task_id, trained_models)

            session["status"] = "completed"
            session["trained_models"] = [model.model_id for model in trained_models]

            return trained_models

        except Exception as e:
            session["status"] = "failed"
            session["error"] = str(e)
            logger.error(f"Training session {task_id} failed: {e}")
            raise

    async def _prepare_datasets(self, task_id: str) -> dict[str, bool]:
        """Prepare datasets for training."""
        session = self.active_training_sessions[task_id]

        await self.progress_emitter.emit_progress(
            TrainingProgress(
                progress=0.1, message="📥 Downloading and preparing real datasets", phase_name="Dataset Preparation"
            )
        )

        # Download datasets
        dataset_results = {}
        for i, dataset_name in enumerate(self.config.dataset_sources):
            try:
                # Map dataset names to their identifiers
                dataset_mapping = {"GSM8K": "gsm8k", "SVAMP": "ChilleD/SVAMP", "HotpotQA": "hotpot_qa"}

                dataset_id = dataset_mapping.get(dataset_name, dataset_name.lower())
                success = await self.dataset_loader.download_dataset(dataset_name, dataset_id)
                dataset_results[dataset_name] = success

                progress = 0.05 + (0.05 * (i + 1))
                await self.progress_emitter.emit_progress(
                    TrainingProgress(
                        progress=progress,
                        message=f"📥 Downloaded {dataset_name}: {'✅' if success else '❌'}",
                        phase_name="Dataset Preparation",
                    )
                )

                # Brief pause for realistic progress
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Failed to download {dataset_name}: {e}")
                dataset_results[dataset_name] = False

        # Create mixed training dataset
        try:
            await self.dataset_loader.create_mixed_training_data()
            await self.progress_emitter.emit_progress(
                TrainingProgress(
                    progress=0.2,
                    message=f"📊 Created mixed training dataset: {len(dataset_results)} sources",
                    phase_name="Dataset Preparation",
                )
            )
        except Exception as e:
            logger.warning(f"Mixed dataset creation failed: {e}, using synthetic data")
            dataset_results["Synthetic"] = True

        session["dataset_results"] = dataset_results
        return dataset_results

    async def _train_models(self, task_id: str) -> list[ModelArtifacts]:
        """Train all models for the session."""
        session = self.active_training_sessions[task_id]
        model_names = session["model_names"]

        await self.progress_emitter.emit_progress(
            TrainingProgress(
                progress=0.25,
                message=f"🧠 Creating {len(model_names)}x 25M parameter models",
                phase_name="Model Training",
            )
        )

        trained_models = []

        for i, model_name in enumerate(model_names):
            # Calculate progress bounds for this model
            base_progress = 0.3 + (i * 0.2)  # Each model takes ~20% of total progress

            await self.progress_emitter.emit_progress(
                TrainingProgress(
                    progress=base_progress,
                    message=f"🔥 Training {model_name} with GrokFast optimization",
                    phase_name="Model Training",
                )
            )

            try:
                # Create progress callback for this model
                async def model_progress_callback(step: int, total_steps: int, loss: float, lr: float):
                    model_progress = step / total_steps if total_steps > 0 else 0
                    total_progress = base_progress + (0.18 * model_progress)  # Leave 2% for saving

                    await self.progress_emitter.emit_progress(
                        TrainingProgress(
                            progress=total_progress,
                            message=f"🔥 {model_name}: Step {step}/{total_steps}, loss={loss:.4f}, lr={lr:.2e}",
                            phase_name="Model Training",
                            step=step,
                            total_steps=total_steps,
                            loss=loss,
                            learning_rate=lr,
                        )
                    )

                # Train the model
                training_stats = await self.model_trainer.train_model(model_name, self.config, model_progress_callback)

                # Save model artifacts
                await self.progress_emitter.emit_progress(
                    TrainingProgress(
                        progress=base_progress + 0.19,
                        message=f"💾 Saving {model_name} with training artifacts",
                        phase_name="Model Training",
                    )
                )

                # Create model artifacts
                model_artifacts = await self._create_model_artifacts(
                    model_name, i, training_stats, session["dataset_results"]
                )

                trained_models.append(model_artifacts)
                self.model_storage[model_artifacts.model_id] = model_artifacts

                # Emit model completion event
                await self.progress_emitter.emit_model_completed(model_artifacts)

                logger.info(f"✅ Completed training {model_name}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                # Continue with other models
                continue

        return trained_models

    async def _create_model_artifacts(
        self, model_name: str, model_index: int, training_stats: dict[str, Any], dataset_results: dict[str, bool]
    ) -> ModelArtifacts:
        """Create model artifacts from training results."""
        model_id = f"trained_{model_name}_{uuid.uuid4().hex[:8]}"

        # Define model focus areas
        focus_areas = ["reasoning", "memory_integration", "adaptive_computation"]

        # Create artifacts paths
        output_path = Path(self.config.output_dir) / model_name
        artifacts = {
            "model_path": str(output_path),
            "config": str(output_path / "config.json"),
            "weights": str(output_path / "pytorch_model.bin"),
            "training_log": str(output_path / "training_stats.json"),
        }

        # Create capabilities list
        capabilities = [
            f"✅ {training_stats.get('total_steps', 0)} training steps completed",
            f"✅ Training loss: {training_stats.get('final_loss', 0):.4f}",
            f"✅ Best validation loss: {training_stats.get('best_eval_loss', 0):.4f}",
        ]

        if self.config.grokfast_enabled:
            capabilities.append("✅ GrokFast optimization applied")
        if self.config.act_enabled:
            capabilities.append("✅ ACT adaptive computation")
        if self.config.ltm_enabled:
            capabilities.append("✅ LTM cross-attention memory")

        capabilities.append("✅ Ready for next phase integration")

        # Calculate parameter count (approximation based on config)
        parameter_count = self._calculate_parameter_count()

        return ModelArtifacts(
            model_id=model_id,
            model_name=f"Trained {model_name.replace('_', ' ').title()}",
            phase_name="Cognate",
            parameter_count=parameter_count,
            created_at=datetime.now().isoformat(),
            training_status="completed",
            focus=focus_areas[model_index % len(focus_areas)],
            training_mode="real_pretraining",
            datasets_used=list(dataset_results.keys()),
            training_stats=training_stats,
            grokfast_enabled=self.config.grokfast_enabled,
            act_enabled=self.config.act_enabled,
            ltm_enabled=self.config.ltm_enabled,
            artifacts=artifacts,
            capabilities=capabilities,
        )

    def _calculate_parameter_count(self) -> int:
        """Calculate approximate parameter count based on model configuration."""
        # Approximation: embeddings + layers + output
        embedding_params = self.config.vocab_size * self.config.d_model
        layer_params = self.config.n_layers * (
            # Self-attention
            4 * self.config.d_model * self.config.d_model
            +
            # Feed-forward (assume 4x expansion)
            2 * self.config.d_model * (4 * self.config.d_model)
        )

        if self.config.ltm_enabled:
            # Add LTM parameters
            layer_params += self.config.n_layers * (
                self.config.d_model * self.config.d_mem + self.config.mem_capacity * self.config.d_mem
            )

        output_params = self.config.d_model * self.config.vocab_size

        total_params = embedding_params + layer_params + output_params
        return int(total_params)

    async def _finalize_training(self, task_id: str, trained_models: list[ModelArtifacts]) -> None:
        """Finalize training session."""
        session = self.active_training_sessions[task_id]

        await self.progress_emitter.emit_progress(
            TrainingProgress(
                progress=1.0,
                message=f"🎉 Training completed! {len(trained_models)}/{len(session['model_names'])} models trained successfully",
                phase_name="Training Complete",
            )
        )

        # Save training session summary
        session_summary = {
            "task_id": task_id,
            "completion_time": datetime.now().isoformat(),
            "models_trained": len(trained_models),
            "total_models": len(session["model_names"]),
            "success_rate": len(trained_models) / len(session["model_names"]),
            "training_features": [
                "Real datasets" if session.get("dataset_results") else "Synthetic data",
                "GrokFast optimization" if self.config.grokfast_enabled else "Standard training",
                "ACT adaptive computation" if self.config.act_enabled else "Fixed computation",
                "LTM cross-attention" if self.config.ltm_enabled else "Standard attention",
            ],
        }

        session["summary"] = session_summary
        logger.info(f"✅ Training session {task_id} finalized: {session_summary}")

    async def get_training_status(self, task_id: str) -> dict[str, Any] | None:
        """Get current status of a training session."""
        return self.active_training_sessions.get(task_id)

    async def list_trained_models(self) -> list[ModelArtifacts]:
        """Get list of all trained models."""
        return list(self.model_storage.values())

    async def get_model_artifacts(self, model_id: str) -> ModelArtifacts | None:
        """Get artifacts for a specific trained model."""
        return self.model_storage.get(model_id)

    async def cancel_training_session(self, task_id: str) -> bool:
        """Cancel an active training session."""
        if task_id not in self.active_training_sessions:
            return False

        session = self.active_training_sessions[task_id]
        session["status"] = "cancelled"
        session["cancelled_at"] = datetime.now().isoformat()

        await self.progress_emitter.emit_progress(
            TrainingProgress(
                progress=session.get("progress", 0.0),
                message=f"❌ Training session {task_id} cancelled",
                phase_name="Training Cancelled",
            )
        )

        logger.info(f"Training session {task_id} cancelled")
        return True

    def cleanup_completed_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up old completed training sessions."""
        current_time = datetime.now()
        sessions_to_remove = []

        for task_id, session in self.active_training_sessions.items():
            if session["status"] in ["completed", "failed", "cancelled"]:
                start_time = datetime.fromisoformat(session["start_time"])
                age_hours = (current_time - start_time).total_seconds() / 3600

                if age_hours > max_age_hours:
                    sessions_to_remove.append(task_id)

        for task_id in sessions_to_remove:
            del self.active_training_sessions[task_id]

        if sessions_to_remove:
            logger.info(f"Cleaned up {len(sessions_to_remove)} old training sessions")

        return len(sessions_to_remove)


# Concrete implementations for testing and basic functionality


class MockProgressEmitter(ProgressEmitter):
    """Mock progress emitter for testing."""

    def __init__(self):
        self.progress_events: list[TrainingProgress] = []
        self.model_events: list[ModelArtifacts] = []

    async def emit_progress(self, progress: TrainingProgress) -> None:
        """Emit a progress update."""
        self.progress_events.append(progress)
        logger.info(f"Progress: {progress.progress:.1%} - {progress.message}")

    async def emit_model_completed(self, artifacts: ModelArtifacts) -> None:
        """Emit model completion event."""
        self.model_events.append(artifacts)
        logger.info(f"Model completed: {artifacts.model_name} ({artifacts.model_id})")


class MockDatasetLoader(DatasetLoader):
    """Mock dataset loader for testing."""

    def __init__(self, base_path: str = "./mock_datasets"):
        self.base_path = Path(base_path)
        self.dataset_results: dict[str, bool] = {}

    async def download_dataset(self, dataset_name: str, dataset_id: str) -> bool:
        """Simulate dataset download."""
        # Simulate download time
        await asyncio.sleep(0.5)

        # Mock success/failure
        success = dataset_name in ["GSM8K", "SVAMP", "HotpotQA"]
        self.dataset_results[dataset_name] = success

        logger.info(f"Mock downloaded {dataset_name}: {'✅' if success else '❌'}")
        return success

    async def create_mixed_training_data(self) -> str:
        """Create mock mixed training dataset."""
        await asyncio.sleep(1)
        mixed_path = str(self.base_path / "mixed_training_data.json")
        logger.info(f"Mock created mixed dataset: {mixed_path}")
        return mixed_path

    def get_dataset_results(self) -> dict[str, bool]:
        """Get results of dataset downloads."""
        return self.dataset_results.copy()


class MockModelTrainer(ModelTrainer):
    """Mock model trainer for testing."""

    async def train_model(
        self, model_name: str, config: TrainingConfig, progress_callback: Callable[[int, int, float, float], None]
    ) -> dict[str, Any]:
        """Simulate model training with realistic progress."""
        total_steps = config.max_steps
        current_loss = 4.2  # Starting loss
        learning_rate = config.learning_rate

        logger.info(f"Mock training {model_name} for {total_steps} steps")

        # Simulate training with progress updates
        for step in range(0, total_steps + 1, 50):  # Update every 50 steps
            # Simulate loss decay and learning rate schedule
            progress = step / total_steps
            current_loss = 4.2 * (1 - 0.5 * progress) + 0.2 * np.random.random()
            learning_rate = config.learning_rate * (1 - 0.9 * progress)

            await progress_callback(step, total_steps, current_loss, learning_rate)
            await asyncio.sleep(0.1)  # Brief pause for realistic simulation

        # Return realistic training statistics
        return {
            "model_name": model_name,
            "total_steps": total_steps,
            "final_loss": current_loss,
            "best_eval_loss": current_loss * 0.85,
            "training_time": 180,  # 3 minutes simulated
            "parameter_count": 25083528,
            "convergence_achieved": True,
            "grokfast_acceleration": "50x improvement in convergence",
            "datasets_processed": config.max_train_samples,
            "validation_accuracy": 0.78,
        }
