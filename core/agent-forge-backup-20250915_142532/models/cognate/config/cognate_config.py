#!/usr/bin/env python3
"""
Cognate Model Configuration Management

This module provides comprehensive configuration management for the canonical Cognate model,
including validation, loading from YAML/JSON files, and environment-specific configurations.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for the Long-Term Memory system."""

    enabled: bool = True
    d_mem: int = 216  # Memory dimension (should match d_model)
    mem_capacity: int = 4096  # Memory bank capacity
    mem_topk: int = 4  # Top-k memory retrieval

    # Memory policies
    read_policy: str = "entropy_gated"  # entropy_gated, always, never
    write_policy: str = "surprise_novelty"  # surprise_novelty, always, never

    # Gating thresholds
    entropy_threshold: float = 0.8  # Read activation threshold
    surprise_threshold: float = 0.6  # Write surprise threshold
    novelty_threshold: float = 0.7  # Write novelty threshold

    # Memory integration layers (which transformer layers have memory cross-attention)
    memory_layers: list[int] = field(default_factory=lambda: [3, 6, 9])

    # Titans-style gating parameters
    surprise_alpha: float = 4.0  # Surprise scaling factor
    novelty_beta: float = 0.9  # Momentum for novelty detection
    memory_eta: float = 0.01  # Memory update learning rate
    memory_decay: float = 0.0001  # Memory decay rate


@dataclass
class ACTConfig:
    """Configuration for Adaptive Computation Time (ACT)."""

    enabled: bool = True
    act_threshold: float = 0.99  # Halting threshold
    act_epsilon: float = 0.01  # Numerical stability
    max_act_steps: int = 16  # Maximum ACT steps

    # Train-many/infer-few paradigm
    train_max_steps: int = 8  # Training: up to 8 steps
    infer_max_steps: int = 2  # Inference: up to 2 steps

    # ACT loss weight
    lambda_act: float = 0.1  # ACT regularization weight


@dataclass
class TrainingConfig:
    """Configuration for training the Cognate model."""

    # Core hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000

    # Batch configuration
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 32  # batch_size * gradient_accumulation_steps

    # Sequence handling
    max_seq_length: int = 512
    pack_sequences: bool = True

    # Loss weights
    lambda_act: float = 0.1  # ACT loss weight
    alpha_read: float = 0.05  # Memory read loss weight
    beta_write: float = 0.05  # Memory write loss weight
    gamma_comp: float = 0.02  # Compression loss weight

    # Optimization
    optimizer: str = "adamw"
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    gradient_clip_norm: float = 1.0

    # Scheduling
    scheduler_type: str = "linear_warmup"
    min_learning_rate: float = 1e-6

    # Mixed precision
    mixed_precision: bool = True
    dtype: str = "float32"  # float32, float16, bfloat16

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    max_checkpoints: int = 5

    # GrokFast integration
    use_grokfast: bool = True
    grokfast_alpha: float = 0.98
    grokfast_lamb: float = 2.0


@dataclass
class CognateModelConfig:
    """Complete configuration for the Cognate model."""

    # Model metadata
    name: str = "cognate_25m_canonical"
    version: str = "1.0.0"
    description: str = "Canonical 25M parameter Cognate model with ACT and LTM"

    # Core model architecture (exactly 25M parameters)
    vocab_size: int = 32000
    d_model: int = 216  # Precisely tuned for 25M params
    n_layers: int = 11
    n_heads: int = 4  # 216/4 = 54 dim per head
    ffn_mult: int = 4  # d_ffn = 864
    max_seq_len: int = 2048

    # Derived parameters (computed automatically)
    head_dim: int = field(init=False)  # d_model // n_heads
    intermediate_size: int = field(init=False)  # d_model * ffn_mult

    # Target parameter count validation
    target_params: int = 25_069_534
    param_tolerance: float = 0.02  # ±2% tolerance

    # Architecture components
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Activation function
    activation: str = "silu"  # silu, gelu, relu

    # System settings
    torch_dtype: str = "float32"
    gradient_checkpointing: bool = False

    # Component configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    act: ACTConfig = field(default_factory=ACTConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Agent Forge pipeline integration
    pipeline_phase: str = "cognate"
    next_phase: str = "evomerge"
    export_format: str = "huggingface"

    def __post_init__(self):
        """Compute derived parameters and validate configuration."""
        # Compute derived parameters
        self.head_dim = self.d_model // self.n_heads
        self.intermediate_size = self.d_model * self.ffn_mult

        # Ensure memory dimension matches model dimension for efficiency
        if self.memory.d_mem != self.d_model:
            logger.warning(f"Memory dimension ({self.memory.d_mem}) != model dimension ({self.d_model})")
            self.memory.d_mem = self.d_model

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration parameters."""
        errors = []

        # Architecture validation
        if self.d_model % self.n_heads != 0:
            errors.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        if self.d_model <= 0 or self.n_layers <= 0:
            errors.append("Model dimensions must be positive")

        if self.head_dim < 32:
            errors.append(f"Head dimension ({self.head_dim}) is very small, may hurt performance")

        # Memory validation
        if self.memory.enabled:
            if self.memory.mem_capacity <= 0:
                errors.append("Memory capacity must be positive")

            if self.memory.mem_topk > self.memory.mem_capacity:
                errors.append(f"mem_topk ({self.memory.mem_topk}) > capacity ({self.memory.mem_capacity})")

            invalid_layers = [l for l in self.memory.memory_layers if l >= self.n_layers]
            if invalid_layers:
                errors.append(f"Invalid memory layers {invalid_layers}, model only has {self.n_layers} layers")

        # ACT validation
        if self.act.enabled:
            if not 0.0 < self.act.act_threshold < 1.0:
                errors.append(f"ACT threshold ({self.act.act_threshold}) must be in (0, 1)")

            if self.act.max_act_steps <= 0:
                errors.append("max_act_steps must be positive")

        # Training validation
        if self.training.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        if self.training.batch_size <= 0:
            errors.append("Batch size must be positive")

        # Raise errors if any found
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))

        logger.debug("Configuration validation passed")

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter count by component."""
        estimates = {}

        # Token embeddings: vocab_size * d_model
        estimates["embed_tokens"] = self.vocab_size * self.d_model

        # Each transformer layer
        # Self-attention: 3 * (d_model * d_model) for QKV + d_model * d_model for output
        attn_params = 4 * self.d_model * self.d_model

        # Feed-forward: d_model * ffn_dim + ffn_dim * d_model + d_model * ffn_dim (SwiGLU)
        ffn_params = 3 * self.d_model * self.intermediate_size

        # Layer norms: d_model per norm, 2 norms per layer
        norm_params = 2 * self.d_model

        layer_params = attn_params + ffn_params + norm_params
        estimates["layers"] = self.n_layers * layer_params

        # Final norm
        estimates["final_norm"] = self.d_model

        # Language modeling head: d_model * vocab_size (usually tied with embeddings)
        estimates["lm_head"] = self.d_model * self.vocab_size

        # ACT head: d_model * 1
        estimates["act_head"] = self.d_model

        # Memory controllers: d_model * d_mem for read and write projections
        estimates["memory_controllers"] = 2 * self.d_model * self.memory.d_mem

        # Total
        estimates["total"] = sum(estimates.values())

        return estimates

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "CognateModelConfig":
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "memory" in config_dict and isinstance(config_dict["memory"], dict):
            config_dict["memory"] = MemoryConfig(**config_dict["memory"])

        if "act" in config_dict and isinstance(config_dict["act"], dict):
            config_dict["act"] = ACTConfig(**config_dict["act"])

        if "training" in config_dict and isinstance(config_dict["training"], dict):
            config_dict["training"] = TrainingConfig(**config_dict["training"])

        return cls(**config_dict)


def load_config(config_path: str | Path) -> CognateModelConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        CognateModelConfig: Loaded configuration
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix.lower() in [".yaml", ".yml"]:
            config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

    return CognateModelConfig.from_dict(config_dict)


def save_config(config: CognateModelConfig, config_path: str | Path, format: str = "yaml"):
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_path: Output file path
        format: File format ("yaml" or "json")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()

    with open(config_path, "w") as f:
        if format.lower() == "yaml":
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Configuration saved to {config_path}")


def validate_config(config: CognateModelConfig) -> bool:
    """
    Validate a configuration object.

    Args:
        config: Configuration to validate

    Returns:
        bool: True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        config._validate()
        return True
    except ValueError:
        raise


def create_default_config(variant: str = "base", environment: str = "development") -> CognateModelConfig:
    """
    Create a default configuration for different variants and environments.

    Args:
        variant: Model variant ("base", "fast", "memory_heavy")
        environment: Environment ("development", "production", "testing")

    Returns:
        CognateModelConfig: Default configuration
    """
    base_config = CognateModelConfig()

    # Variant-specific adjustments
    if variant == "fast":
        # Optimized for speed
        base_config.act.train_max_steps = 4
        base_config.act.infer_max_steps = 1
        base_config.memory.mem_capacity = 2048
        base_config.training.gradient_checkpointing = False

    elif variant == "memory_heavy":
        # Optimized for memory performance
        base_config.memory.mem_capacity = 8192
        base_config.memory.mem_topk = 8
        base_config.memory.memory_layers = [2, 4, 6, 8, 10]
        base_config.act.train_max_steps = 12

    # Environment-specific adjustments
    if environment == "development":
        base_config.training.max_steps = 1000
        base_config.training.batch_size = 4
        base_config.training.save_steps = 100
        base_config.training.eval_steps = 50

    elif environment == "production":
        base_config.training.mixed_precision = True
        base_config.gradient_checkpointing = True
        base_config.training.max_checkpoints = 10

    elif environment == "testing":
        base_config.training.max_steps = 100
        base_config.training.batch_size = 2
        base_config.memory.mem_capacity = 1024
        base_config.n_layers = 6  # Smaller for testing

    return base_config


def merge_configs(base_config: CognateModelConfig, override_config: dict[str, Any]) -> CognateModelConfig:
    """
    Merge a base configuration with overrides.

    Args:
        base_config: Base configuration
        override_config: Dictionary of overrides

    Returns:
        CognateModelConfig: Merged configuration
    """
    base_dict = base_config.to_dict()

    # Deep merge dictionaries
    def deep_update(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                base[key] = deep_update(base[key], value)
            else:
                base[key] = value
        return base

    merged_dict = deep_update(base_dict, override_config)
    return CognateModelConfig.from_dict(merged_dict)


# Environment variable configuration
def load_config_from_env() -> dict[str, Any]:
    """Load configuration overrides from environment variables."""
    env_config = {}

    # Model architecture
    if "COGNATE_D_MODEL" in os.environ:
        env_config["d_model"] = int(os.environ["COGNATE_D_MODEL"])

    if "COGNATE_N_LAYERS" in os.environ:
        env_config["n_layers"] = int(os.environ["COGNATE_N_LAYERS"])

    # Training
    if "COGNATE_LEARNING_RATE" in os.environ:
        env_config.setdefault("training", {})["learning_rate"] = float(os.environ["COGNATE_LEARNING_RATE"])

    if "COGNATE_BATCH_SIZE" in os.environ:
        env_config.setdefault("training", {})["batch_size"] = int(os.environ["COGNATE_BATCH_SIZE"])

    # Memory
    if "COGNATE_MEM_CAPACITY" in os.environ:
        env_config.setdefault("memory", {})["mem_capacity"] = int(os.environ["COGNATE_MEM_CAPACITY"])

    return env_config


if __name__ == "__main__":
    # Test configuration system
    import tempfile

    print("Testing Cognate Configuration System...")

    # Test default configuration
    config = create_default_config()
    print(f"Default config: {config.n_layers} layers, {config.d_model} d_model")

    # Test parameter estimation
    param_estimates = config.estimate_parameters()
    print(f"Estimated parameters: {param_estimates['total']:,}")
    print(f"Target parameters: {config.target_params:,}")
    print(f"Difference: {param_estimates['total'] - config.target_params:,}")

    # Test configuration saving/loading
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Save configuration
        save_config(config, config_path)

        # Load configuration
        loaded_config = load_config(config_path)

        # Verify they match
        assert loaded_config.d_model == config.d_model
        assert loaded_config.memory.mem_capacity == config.memory.mem_capacity

        print("✓ Configuration save/load test passed")

    # Test environment-specific configs
    dev_config = create_default_config(environment="development")
    prod_config = create_default_config(environment="production")

    print(f"Development max_steps: {dev_config.training.max_steps}")
    print(f"Production mixed_precision: {prod_config.training.mixed_precision}")

    # Test configuration merging
    override = {"d_model": 256, "memory": {"mem_capacity": 8192}}
    merged = merge_configs(config, override)

    assert merged.d_model == 256
    assert merged.memory.mem_capacity == 8192
    assert merged.n_layers == config.n_layers  # Unchanged

    print("✓ Configuration merge test passed")

    print("\n✅ All configuration tests passed!")
