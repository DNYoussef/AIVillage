"""
Training Configuration Management

Centralized configuration for HRRM model training.
Eliminates magic numbers and provides type-safe configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TokenizerConfig:
    """Configuration for tokenizer setup."""

    DEFAULT_VOCAB_SIZE: int = 32000
    TOKENIZER_PATH: str = "artifacts/tokenizer/hrrm_bpe_32k.json"


@dataclass(frozen=True)
class DataConfig:
    """Configuration for training data generation."""

    BATCH_SIZE: int = 2
    SEQUENCE_LENGTH: int = 64
    NUM_BATCHES: int = 20
    MAX_TOKEN_VALUE: int = 1000


@dataclass(frozen=True)
class ModelConfig:
    """Base configuration for model parameters."""

    D_MODEL: int = 256
    N_LAYERS: int = 8
    N_HEAD: int = 8
    MAX_H: int = 2
    INNER_T: int = 2


@dataclass(frozen=True)
class PlannerModelConfig(ModelConfig):
    """Configuration specific to HRMPlanner model."""

    LAMBDA_CTRL: float = 0.2
    CONTROL_TOKENS: list[str] = None

    def __post_init__(self):
        if self.CONTROL_TOKENS is None:
            # Use object.__setattr__ to modify frozen dataclass
            object.__setattr__(self, "CONTROL_TOKENS", ["<PLAN>", "<SUBGOAL>", "<ACTION>", "<CHECK>", "<ENDPLAN>"])


@dataclass(frozen=True)
class ReasonerModelConfig(ModelConfig):
    """Configuration specific to HRMReasoner model."""

    SELF_CONSISTENCY_K: int = 3


@dataclass(frozen=True)
class MemoryModelConfig(ModelConfig):
    """Configuration specific to MemoryAsContextTiny model."""

    MEM_DIM: int = 128
    MEM_TOKENS: int = 32
    MEM_SLOTS: int = 64
    ALPHA: float = 1.0
    BETA: float = 0.9
    ETA: float = 0.01
    ETA_DECAY: float = 0.001


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training process."""

    EPOCHS: int = 1
    LEARNING_RATE: float = 3e-4
    DEVICE_PREFERENCE: str = "cuda"  # "cuda" or "cpu"


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output paths and formats."""

    CHECKPOINT_BASE_DIR: str = "artifacts/checkpoints"
    SUMMARY_PATH: str = "artifacts/hrrm_training_summary.json"

    @property
    def planner_checkpoint_dir(self) -> Path:
        """Path to planner model checkpoint directory."""
        return Path(self.CHECKPOINT_BASE_DIR) / "planner"

    @property
    def reasoner_checkpoint_dir(self) -> Path:
        """Path to reasoner model checkpoint directory."""
        return Path(self.CHECKPOINT_BASE_DIR) / "reasoner"

    @property
    def memory_checkpoint_dir(self) -> Path:
        """Path to memory model checkpoint directory."""
        return Path(self.CHECKPOINT_BASE_DIR) / "memory"


@dataclass(frozen=True)
class HRRMTrainingConfig:
    """Complete configuration for HRRM training pipeline."""

    tokenizer: TokenizerConfig = TokenizerConfig()
    data: DataConfig = DataConfig()
    planner: PlannerModelConfig = PlannerModelConfig()
    reasoner: ReasonerModelConfig = ReasonerModelConfig()
    memory: MemoryModelConfig = MemoryModelConfig()
    training: TrainingConfig = TrainingConfig()
    output: OutputConfig = OutputConfig()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "tokenizer": self.tokenizer.__dict__,
            "data": self.data.__dict__,
            "planner": self.planner.__dict__,
            "reasoner": self.reasoner.__dict__,
            "memory": self.memory.__dict__,
            "training": self.training.__dict__,
            "output": self.output.__dict__,
        }
