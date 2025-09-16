"""
BitNet Configuration for Agent Forge Phase 4
============================================

Configuration management for BitNet quantization and compression.
Defines parameters for layer conversion, optimization, and integration.

Author: BitNet Core Implementation Specialist - Agent Forge Phase 4
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import torch
from pathlib import Path


class QuantizationMode(Enum):
    """Quantization modes for BitNet."""
    TERNARY = "ternary"      # {-1, 0, 1}
    BINARY = "binary"        # {-1, 1}
    ABSMEAN = "absmean"      # Absolute mean scaling


class CompressionLevel(Enum):
    """Compression levels with different quality/speed tradeoffs."""
    CONSERVATIVE = "conservative"  # Minimal accuracy loss, slower compression
    BALANCED = "balanced"         # Balanced accuracy/speed
    AGGRESSIVE = "aggressive"     # Maximum compression, potential accuracy loss


@dataclass
class BitNetLayerConfig:
    """Configuration for individual BitNet layers."""

    # Quantization settings
    quantization_mode: QuantizationMode = QuantizationMode.TERNARY
    weight_scaling: bool = True
    activation_scaling: bool = True
    bias_quantization: bool = False

    # Layer-specific settings
    preserve_first_layer: bool = True
    preserve_last_layer: bool = True
    preserve_embedding: bool = True
    preserve_norm_layers: bool = True

    # Attention preservation
    preserve_attention_weights: bool = True
    attention_scaling_factor: float = 1.0


@dataclass
class BitNetOptimizationConfig:
    """Performance optimization configuration."""

    # CUDA optimization
    enable_cuda_kernels: bool = True
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # Memory optimization
    enable_memory_efficient_attention: bool = True
    max_batch_size: int = 32
    chunk_size: int = 1024

    # Parallel processing
    num_workers: int = 4
    enable_parallel_quantization: bool = True

    # Cache settings
    enable_weight_cache: bool = True
    cache_size_mb: int = 256


@dataclass
class BitNetIntegrationConfig:
    """Integration configuration with other phases."""

    # Phase 2 (EvoMerge) integration
    evomerge_model_path: str = "../phase2_evomerge/outputs/"
    evomerge_checkpoint_format: str = "evolved_model_gen_{generation}.pt"

    # Phase 3 (Quiet-STaR) integration
    quietstar_model_path: str = "../phase3_quietstar/outputs/"
    quietstar_attention_integration: bool = True
    preserve_thought_generation: bool = True

    # Phase 5 preparation
    output_path: str = "./outputs/compressed_models/"
    export_format: str = "pytorch"  # pytorch, onnx, torchscript
    export_metadata: bool = True

    # Compatibility checks
    verify_model_compatibility: bool = True
    validate_output_quality: bool = True


@dataclass
class BitNetValidationConfig:
    """Validation and quality assurance configuration."""

    # Quality thresholds
    max_accuracy_loss: float = 0.10  # 10% maximum accuracy loss
    min_compression_ratio: float = 4.0  # Minimum 4x compression
    max_memory_overhead: float = 0.05  # 5% memory overhead during conversion

    # Validation datasets
    validation_tasks: List[str] = field(default_factory=lambda: [
        "language_modeling", "text_classification", "reasoning"
    ])
    validation_metrics: List[str] = field(default_factory=lambda: [
        "perplexity", "accuracy", "f1_score", "inference_time", "memory_usage"
    ])

    # Performance benchmarks
    benchmark_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 16, 32])
    benchmark_sequence_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])

    # Theater detection
    enable_theater_detection: bool = True
    quality_correlation_threshold: float = 0.95
    evidence_validation: bool = True


@dataclass
class BitNetConfig:
    """Complete BitNet configuration for Agent Forge Phase 4."""

    # Core configuration sections
    layer_config: BitNetLayerConfig = field(default_factory=BitNetLayerConfig)
    optimization_config: BitNetOptimizationConfig = field(default_factory=BitNetOptimizationConfig)
    integration_config: BitNetIntegrationConfig = field(default_factory=BitNetIntegrationConfig)
    validation_config: BitNetValidationConfig = field(default_factory=BitNetValidationConfig)

    # Global settings
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    seed: int = 42

    # Logging and monitoring
    log_level: str = "INFO"
    log_interval: int = 10
    enable_wandb: bool = False
    wandb_project: str = "agent-forge-bitnet"

    # Checkpointing
    checkpoint_interval: int = 100
    checkpoint_dir: str = "./checkpoints/bitnet"
    keep_checkpoints: int = 3

    # Output configuration
    output_dir: str = "./outputs/"
    save_intermediate_models: bool = True
    save_quantization_stats: bool = True

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Ensure output directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.integration_config.output_path).mkdir(parents=True, exist_ok=True)

        # Validate device availability
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.optimization_config.enable_cuda_kernels = False
            self.optimization_config.use_mixed_precision = False

        # Adjust optimization settings based on device
        if self.device == "cpu":
            self.optimization_config.enable_cuda_kernels = False
            self.optimization_config.use_mixed_precision = False
            self.dtype = torch.float32

        # Validate compression level settings
        if self.compression_level == CompressionLevel.CONSERVATIVE:
            self.layer_config.preserve_first_layer = True
            self.layer_config.preserve_last_layer = True
            self.layer_config.preserve_attention_weights = True
            self.validation_config.max_accuracy_loss = 0.05
        elif self.compression_level == CompressionLevel.AGGRESSIVE:
            self.layer_config.preserve_first_layer = False
            self.layer_config.preserve_last_layer = False
            self.validation_config.max_accuracy_loss = 0.15
            self.validation_config.min_compression_ratio = 6.0

        # Ensure reasonable batch sizes based on available memory
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            if gpu_memory < 8:  # Less than 8GB GPU memory
                self.optimization_config.max_batch_size = min(16, self.optimization_config.max_batch_size)
                self.optimization_config.chunk_size = min(512, self.optimization_config.chunk_size)


# Predefined configurations for common use cases
DEFAULT_CONFIG = BitNetConfig()

CONSERVATIVE_CONFIG = BitNetConfig(
    compression_level=CompressionLevel.CONSERVATIVE,
    layer_config=BitNetLayerConfig(
        quantization_mode=QuantizationMode.TERNARY,
        preserve_first_layer=True,
        preserve_last_layer=True,
        preserve_attention_weights=True
    ),
    validation_config=BitNetValidationConfig(
        max_accuracy_loss=0.05,
        min_compression_ratio=3.0
    )
)

AGGRESSIVE_CONFIG = BitNetConfig(
    compression_level=CompressionLevel.AGGRESSIVE,
    layer_config=BitNetLayerConfig(
        quantization_mode=QuantizationMode.BINARY,
        preserve_first_layer=False,
        preserve_last_layer=False,
        preserve_attention_weights=False
    ),
    validation_config=BitNetValidationConfig(
        max_accuracy_loss=0.15,
        min_compression_ratio=8.0
    )
)

DEFENSE_INDUSTRY_CONFIG = BitNetConfig(
    compression_level=CompressionLevel.CONSERVATIVE,
    layer_config=BitNetLayerConfig(
        quantization_mode=QuantizationMode.TERNARY,
        preserve_first_layer=True,
        preserve_last_layer=True,
        preserve_attention_weights=True,
        bias_quantization=False
    ),
    validation_config=BitNetValidationConfig(
        max_accuracy_loss=0.03,  # Very strict for defense applications
        min_compression_ratio=4.0,
        enable_theater_detection=True,
        quality_correlation_threshold=0.98,
        evidence_validation=True
    ),
    optimization_config=BitNetOptimizationConfig(
        enable_cuda_kernels=True,
        use_mixed_precision=False,  # Prefer precision over speed
        gradient_checkpointing=True
    )
)


def get_config(config_name: str = "default") -> BitNetConfig:
    """Get predefined configuration by name."""
    configs = {
        "default": DEFAULT_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG,
        "defense": DEFENSE_INDUSTRY_CONFIG
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")

    return configs[config_name]


def save_config(config: BitNetConfig, path: Union[str, Path]):
    """Save configuration to file."""
    import json

    # Convert dataclass to dict for JSON serialization
    config_dict = {
        "layer_config": config.layer_config.__dict__,
        "optimization_config": config.optimization_config.__dict__,
        "integration_config": config.integration_config.__dict__,
        "validation_config": config.validation_config.__dict__,
        "compression_level": config.compression_level.value,
        "device": config.device,
        "dtype": str(config.dtype),
        "seed": config.seed,
        "log_level": config.log_level,
        "log_interval": config.log_interval,
        "enable_wandb": config.enable_wandb,
        "wandb_project": config.wandb_project,
        "checkpoint_interval": config.checkpoint_interval,
        "checkpoint_dir": config.checkpoint_dir,
        "keep_checkpoints": config.keep_checkpoints,
        "output_dir": config.output_dir,
        "save_intermediate_models": config.save_intermediate_models,
        "save_quantization_stats": config.save_quantization_stats
    }

    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: Union[str, Path]) -> BitNetConfig:
    """Load configuration from file."""
    import json

    with open(path, 'r') as f:
        config_dict = json.load(f)

    # Convert back to dataclass
    config = BitNetConfig(
        layer_config=BitNetLayerConfig(**config_dict["layer_config"]),
        optimization_config=BitNetOptimizationConfig(**config_dict["optimization_config"]),
        integration_config=BitNetIntegrationConfig(**config_dict["integration_config"]),
        validation_config=BitNetValidationConfig(**config_dict["validation_config"]),
        compression_level=CompressionLevel(config_dict["compression_level"]),
        device=config_dict["device"],
        dtype=getattr(torch, config_dict["dtype"].split('.')[-1]),
        seed=config_dict["seed"],
        log_level=config_dict["log_level"],
        log_interval=config_dict["log_interval"],
        enable_wandb=config_dict["enable_wandb"],
        wandb_project=config_dict["wandb_project"],
        checkpoint_interval=config_dict["checkpoint_interval"],
        checkpoint_dir=config_dict["checkpoint_dir"],
        keep_checkpoints=config_dict["keep_checkpoints"],
        output_dir=config_dict["output_dir"],
        save_intermediate_models=config_dict["save_intermediate_models"],
        save_quantization_stats=config_dict["save_quantization_stats"]
    )

    return config