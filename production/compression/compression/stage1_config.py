from dataclasses import dataclass

import torch


@dataclass
class Stage1Config:
    """Configuration for Stage-1 compression pipeline (BitNet + SeedLM)"""

    # BitNet Fine-tuning Parameters
    bitnet_enabled: bool = True
    bitnet_learning_rate: float = 1e-4  # Conservative for stability
    bitnet_epochs: int = 2  # Minimal epochs to avoid overfitting
    bitnet_batch_size: int = 2  # Small batch for 16GB VRAM constraint
    bitnet_zero_threshold: float = 0.02  # Threshold for weight binarization
    bitnet_warmup_ratio: float = 0.4  # Gradual ramping as in existing code

    # SeedLM Encoding Parameters
    seedlm_enabled: bool = True
    seedlm_block_size: int = 8  # Balance between compression and accuracy
    seedlm_latent_dim: int = 4  # Reduced dimensionality
    seedlm_num_seeds: int = 512  # Increased for better representation
    seedlm_lfsr_seed: int = 0x1D2C3  # Default LFSR seed
    seedlm_lfsr_taps: list[int] = None  # Will use [16, 14, 13, 11] default

    # Training Configuration
    max_sequence_length: int = 1024  # Memory constraint
    gradient_checkpointing: bool = True
    fp16: bool = True
    gradient_accumulation_steps: int = 4  # Effective batch size = 2 * 4 = 8

    # Compression Targets
    target_compression_ratio: float = 10.0  # Minimum 10x compression
    max_accuracy_drop: float = 0.05  # 5% accuracy drop constraint

    # Model I/O
    input_model_path: str = "models/raw/"
    output_model_path: str = "models/compressed/"
    checkpoint_dir: str = "checkpoints/stage1/"

    # Evaluation
    eval_dataset_path: str = "eval/hellaswag_sample.jsonl"
    eval_batch_size: int = 1
    eval_max_samples: int = 100

    # Hardware Constraints
    max_memory_gb: float = 16.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging and Monitoring
    log_interval: int = 10
    save_interval: int = 500
    prometheus_enabled: bool = True

    def __post_init__(self):
        if self.seedlm_lfsr_taps is None:
            self.seedlm_lfsr_taps = [16, 14, 13, 11]  # Standard LFSR configuration

        # Validate constraints
        if self.target_compression_ratio < 10.0:
            raise ValueError("Target compression ratio must be >= 10x")
        if self.max_accuracy_drop > 0.05:
            raise ValueError("Maximum accuracy drop must be <= 5%")
        if self.bitnet_batch_size * self.gradient_accumulation_steps > 32:
            raise ValueError("Effective batch size too large for 16GB VRAM")


# Import torch for device check

# Default configuration for Stage-1 pipeline
DEFAULT_STAGE1_CONFIG = Stage1Config()
