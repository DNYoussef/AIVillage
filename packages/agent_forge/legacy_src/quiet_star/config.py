"""
Quiet-STaR Configuration System
Manages special tokens, training parameters, and thought control settings.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QuietSTaRConfig:
    """Configuration for Quiet-STaR thought-token system."""

    # Core feature flags
    enable_quiet_star: bool = True
    enable_training: bool = True
    enable_inference_stripping: bool = True

    # Special tokens
    start_of_thought_token: str = "<SoT>"
    end_of_thought_token: str = "</SoT>"
    no_thought_token: str = "<NoT>"

    # Training parameters
    thought_ratio: float = 0.5  # Probability of generating thoughts during training
    max_thought_tokens: int = 128  # Maximum tokens in thought sequence

    # Loss weights
    w_task: float = 1.0  # Weight for primary task loss
    w_reflect: float = 0.3  # Weight for reflection quality loss
    w_leak: float = 10.0  # Weight for thought leak penalty (high to prevent leakage)

    # Reflection quality settings
    reflection_teacher_model: str | None = None  # Model to generate supervision
    reflection_heuristic_threshold: float = 0.7  # Quality threshold for heuristics

    # Leak prevention
    leak_penalty_type: str = "exponential"  # "linear" or "exponential"
    leak_detection_threshold: float = 0.01  # Minimum probability to consider a leak

    # Sampling and generation
    thought_temperature: float = 0.8  # Temperature for thought generation
    strip_thoughts_in_inference: bool = True  # Always strip thoughts from user output

    # Token vocabulary integration
    special_token_ids: dict[str, int] = field(default_factory=dict)
    vocab_size_expansion: int = 3  # Number of special tokens added

    # Training curriculum
    thought_curriculum: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"stage": "bootstrap", "thought_ratio": 0.8, "max_tokens": 64},
            {"stage": "intermediate", "thought_ratio": 0.6, "max_tokens": 96},
            {"stage": "advanced", "thought_ratio": 0.5, "max_tokens": 128},
        ]
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.thought_ratio <= 1:
            raise ValueError("thought_ratio must be between 0 and 1")

        if self.max_thought_tokens <= 0:
            raise ValueError("max_thought_tokens must be positive")

        if any(w < 0 for w in [self.w_task, self.w_reflect, self.w_leak]):
            raise ValueError("All loss weights must be non-negative")

        if self.w_leak == 0:
            raise ValueError("w_leak must be positive to prevent thought leakage")

    def get_special_tokens(self) -> list[str]:
        """Get list of all special tokens."""
        return [
            self.start_of_thought_token,
            self.end_of_thought_token,
            self.no_thought_token,
        ]

    def update_token_ids(self, tokenizer) -> None:
        """Update special token IDs from tokenizer."""
        for token in self.get_special_tokens():
            if hasattr(tokenizer, "convert_tokens_to_ids"):
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    self.special_token_ids[token] = token_id

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_quiet_star": self.enable_quiet_star,
            "enable_training": self.enable_training,
            "enable_inference_stripping": self.enable_inference_stripping,
            "special_tokens": {
                "start_of_thought": self.start_of_thought_token,
                "end_of_thought": self.end_of_thought_token,
                "no_thought": self.no_thought_token,
            },
            "training_params": {
                "thought_ratio": self.thought_ratio,
                "max_thought_tokens": self.max_thought_tokens,
                "loss_weights": {
                    "task": self.w_task,
                    "reflect": self.w_reflect,
                    "leak": self.w_leak,
                },
            },
            "reflection_settings": {
                "teacher_model": self.reflection_teacher_model,
                "heuristic_threshold": self.reflection_heuristic_threshold,
            },
            "leak_prevention": {
                "penalty_type": self.leak_penalty_type,
                "detection_threshold": self.leak_detection_threshold,
            },
            "generation_params": {
                "thought_temperature": self.thought_temperature,
                "strip_thoughts_in_inference": self.strip_thoughts_in_inference,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "QuietSTaRConfig":
        """Create config from dictionary."""
        # Flatten nested structure for dataclass initialization
        kwargs = {
            "enable_quiet_star": config_dict.get("enable_quiet_star", True),
            "enable_training": config_dict.get("enable_training", True),
            "enable_inference_stripping": config_dict.get("enable_inference_stripping", True),
        }

        # Extract special tokens
        if "special_tokens" in config_dict:
            tokens = config_dict["special_tokens"]
            kwargs.update(
                {
                    "start_of_thought_token": tokens.get("start_of_thought", "<SoT>"),
                    "end_of_thought_token": tokens.get("end_of_thought", "</SoT>"),
                    "no_thought_token": tokens.get("no_thought", "<NoT>"),
                }
            )

        # Extract training parameters
        if "training_params" in config_dict:
            params = config_dict["training_params"]
            kwargs.update(
                {
                    "thought_ratio": params.get("thought_ratio", 0.5),
                    "max_thought_tokens": params.get("max_thought_tokens", 128),
                }
            )

            if "loss_weights" in params:
                weights = params["loss_weights"]
                kwargs.update(
                    {
                        "w_task": weights.get("task", 1.0),
                        "w_reflect": weights.get("reflect", 0.3),
                        "w_leak": weights.get("leak", 10.0),
                    }
                )

        # Extract other nested parameters
        if "reflection_settings" in config_dict:
            settings = config_dict["reflection_settings"]
            kwargs.update(
                {
                    "reflection_teacher_model": settings.get("teacher_model"),
                    "reflection_heuristic_threshold": settings.get("heuristic_threshold", 0.7),
                }
            )

        if "leak_prevention" in config_dict:
            leak_settings = config_dict["leak_prevention"]
            kwargs.update(
                {
                    "leak_penalty_type": leak_settings.get("penalty_type", "exponential"),
                    "leak_detection_threshold": leak_settings.get("detection_threshold", 0.01),
                }
            )

        if "generation_params" in config_dict:
            gen_params = config_dict["generation_params"]
            kwargs.update(
                {
                    "thought_temperature": gen_params.get("thought_temperature", 0.8),
                    "strip_thoughts_in_inference": gen_params.get("strip_thoughts_in_inference", True),
                }
            )

        return cls(**kwargs)

    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "QuietSTaRConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> QuietSTaRConfig:
    """Get default Quiet-STaR configuration."""
    return QuietSTaRConfig()


def get_training_config(stage: str = "intermediate") -> QuietSTaRConfig:
    """Get training configuration for specific curriculum stage."""
    config = QuietSTaRConfig()

    # Find curriculum stage
    stage_config = None
    for curriculum_stage in config.thought_curriculum:
        if curriculum_stage["stage"] == stage:
            stage_config = curriculum_stage
            break

    if stage_config:
        config.thought_ratio = stage_config.get("thought_ratio", config.thought_ratio)
        config.max_thought_tokens = stage_config.get("max_tokens", config.max_thought_tokens)

    return config


def get_inference_config() -> QuietSTaRConfig:
    """Get configuration optimized for inference (thoughts stripped)."""
    config = QuietSTaRConfig()
    config.enable_training = False
    config.enable_inference_stripping = True
    config.strip_thoughts_in_inference = True
    config.thought_ratio = 0.0  # No thoughts generated during inference

    return config
