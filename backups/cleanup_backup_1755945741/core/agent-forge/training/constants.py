"""Training-related constants for AIVillage.

This module centralizes all magic literals used in training workflows,
model configurations, and dataset processing to eliminate connascence
of meaning and improve maintainability.
"""

from enum import Enum
from typing import Final

# Model size constants
PARAMETERS_PER_MILLION: Final[int] = 1_000_000
HRRM_MODEL_SIZE_50M: Final[int] = 50_000_000
HRRM_MODEL_COUNT: Final[int] = 3

# Training configuration
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_LEARNING_RATE: Final[float] = 1e-4
DEFAULT_EPOCHS: Final[int] = 100
MAX_SEQUENCE_LENGTH: Final[int] = 512
GRADIENT_CLIP_VALUE: Final[float] = 1.0

# Directory paths
DEFAULT_CONFIG_DIR: Final[str] = "configs/hrrm"
DEFAULT_DATASET_DIR: Final[str] = "packages/core/training/datasets"
DEFAULT_OUTPUT_DIR: Final[str] = "packages/agent_forge/models/hrrm_models"
ARTIFACTS_DIR: Final[str] = "artifacts"

# File naming patterns
TRAINING_SUMMARY_FILENAME: Final[str] = "50m_hrrm_training_summary.json"
MODEL_CHECKPOINT_PATTERN: Final[str] = "checkpoint_{epoch:04d}.pt"
BEST_MODEL_FILENAME: Final[str] = "best_model.pt"


# Logging and display constants
class LogMessages:
    """Standardized log messages to eliminate magic string literals."""

    TRAINING_SUMMARY_SAVED: Final[str] = "\n✅ Training summary saved to {path}"
    TRAINING_COMPLETE: Final[str] = "🚀 All 3 HRRM models (50M parameters each) trained successfully!"
    MODELS_SAVED: Final[str] = "📁 Models saved to: {path}"
    DEVICE_SELECTED: Final[str] = "Using device: {device}"
    EPOCH_PROGRESS: Final[str] = "Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"
    PARAMETER_COUNT: Final[str] = "{model_name:>10}: {params:>12,} params, loss: {loss:.4f}"
    TOTAL_PARAMETERS: Final[str] = "{'TOTAL':>10}: {total:>12,} parameters (~{millions:.1f}M)"


# Training hyperparameters
class TrainingDefaults:
    """Default hyperparameters for training configurations."""

    OPTIMIZER_BETAS: Final[tuple[float, float]] = (0.9, 0.999)
    WEIGHT_DECAY: Final[float] = 1e-5
    SCHEDULER_GAMMA: Final[float] = 0.95
    WARMUP_STEPS: Final[int] = 1000
    SAVE_EVERY_N_EPOCHS: Final[int] = 10
    EVAL_EVERY_N_STEPS: Final[int] = 100

    # Early stopping
    PATIENCE: Final[int] = 10
    MIN_DELTA: Final[float] = 1e-4

    # Validation split
    VALIDATION_SPLIT: Final[float] = 0.2
    TEST_SPLIT: Final[float] = 0.1


# Model architecture constants
class ModelArchitecture:
    """Model architecture constants."""

    HIDDEN_SIZE: Final[int] = 768
    INTERMEDIATE_SIZE: Final[int] = 3072
    NUM_ATTENTION_HEADS: Final[int] = 12
    NUM_HIDDEN_LAYERS: Final[int] = 12
    DROPOUT_RATE: Final[float] = 0.1
    LAYER_NORM_EPS: Final[float] = 1e-12

    # HRRM specific
    MEMORY_SIZE: Final[int] = 1024
    PLANNER_HEADS: Final[int] = 8
    REASONER_DEPTH: Final[int] = 6


# Dataset configuration
class DatasetConfig:
    """Dataset processing configuration."""

    MAX_SAMPLES_PER_DATASET: Final[int] = 100_000
    MIN_TEXT_LENGTH: Final[int] = 10
    MAX_TEXT_LENGTH: Final[int] = 2048
    TOKENIZER_CACHE_DIR: Final[str] = ".cache/tokenizers"

    # Data preprocessing
    NORMALIZE_TEXT: Final[bool] = True
    REMOVE_DUPLICATES: Final[bool] = True
    SHUFFLE_DATA: Final[bool] = True
    RANDOM_SEED: Final[int] = 42


# File extensions and formats
class FileFormats:
    """Standard file formats used in training."""

    MODEL_EXTENSION: Final[str] = ".pt"
    CONFIG_EXTENSION: Final[str] = ".yaml"
    DATA_EXTENSION: Final[str] = ".json"
    LOG_EXTENSION: Final[str] = ".log"
    CHECKPOINT_EXTENSION: Final[str] = ".ckpt"


# Resource limits
class ResourceLimits:
    """Resource usage limits for training."""

    MAX_GPU_MEMORY_GB: Final[int] = 16
    MAX_CPU_CORES: Final[int] = 8
    MAX_DATASET_SIZE_GB: Final[int] = 10
    CHECKPOINT_RETENTION_DAYS: Final[int] = 30

    # Memory management
    BATCH_SIZE_AUTO_REDUCE_FACTOR: Final[float] = 0.75
    GRADIENT_ACCUMULATION_STEPS: Final[int] = 4


class TrainingPhases(Enum):
    """Training phase enumeration."""

    PRETRAINING = "pretraining"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    INFERENCE = "inference"


class OptimizationStrategy(Enum):
    """Optimization strategies for training."""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"


class SchedulerType(Enum):
    """Learning rate scheduler types."""

    COSINE = "cosine"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
