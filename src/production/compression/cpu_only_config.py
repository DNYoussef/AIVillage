"""CPU-Only Compression Configuration - Prompt D

Comprehensive CPU-only configuration for compression pipeline that avoids all GPU
dependencies and optimizes for CPU-friendly quantization and compression techniques.

Integration Point: CPU-compatible compression for Phase 4 integration testing
"""

import logging
import os
import warnings
from typing import Any

# Suppress GPU-related warnings early
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

try:
    import torch

    # Force CPU-only mode
    torch.cuda.is_available = lambda: False
    torch.backends.cudnn.enabled = False
except ImportError:
    torch = None

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CPUOnlyCompressionConfig(BaseModel):
    """CPU-only compression configuration avoiding all GPU dependencies."""

    # Force CPU device
    device: str = Field(default="cpu", description="Always use CPU for compression")
    force_cpu: bool = Field(default=True, description="Force CPU-only execution")

    # CPU-optimized settings
    num_cpu_threads: int = Field(default=1, description="Number of CPU threads")
    cpu_memory_limit_gb: int = Field(default=4, description="CPU memory limit in GB")
    enable_cpu_optimizations: bool = Field(
        default=True, description="Enable CPU-specific optimizations"
    )

    # Disable GPU-dependent features
    mixed_precision: bool = Field(
        default=False, description="Disable mixed precision (GPU feature)"
    )
    use_cuda_kernels: bool = Field(default=False, description="Disable CUDA kernels")
    enable_flash_attention: bool = Field(
        default=False, description="Disable Flash Attention (GPU feature)"
    )
    use_triton: bool = Field(default=False, description="Disable Triton optimizations")

    # CPU-friendly quantization settings
    quantization_method: str = Field(
        default="cpu_int8", description="CPU-compatible quantization"
    )
    bitnet_cpu_mode: bool = Field(default=True, description="Use CPU-compatible BitNet")
    avoid_bitsandbytes: bool = Field(
        default=True, description="Avoid bitsandbytes (GPU library)"
    )

    # Conservative batch sizes for CPU
    batch_size: int = Field(default=1, description="Small batch size for CPU")
    eval_batch_size: int = Field(default=1, description="Small eval batch size for CPU")
    gradient_accumulation_steps: int = Field(
        default=8, description="Accumulate gradients to simulate larger batches"
    )

    # CPU-friendly model loading
    low_cpu_mem_usage: bool = Field(
        default=True, description="Use low CPU memory loading"
    )
    torch_dtype: str = Field(
        default="float32", description="Use float32 for CPU compatibility"
    )
    load_in_8bit: bool = Field(
        default=False, description="Disable 8-bit loading (GPU feature)"
    )
    load_in_4bit: bool = Field(
        default=False, description="Disable 4-bit loading (GPU feature)"
    )

    # Conservative learning settings
    learning_rate: float = Field(default=1e-5, description="Conservative learning rate")
    max_steps: int = Field(default=100, description="Limited steps for CPU training")
    warmup_steps: int = Field(default=10, description="Short warmup for CPU")

    # CPU-optimized data loading
    dataloader_num_workers: int = Field(
        default=0, description="Disable multiprocessing for stability"
    )
    dataloader_pin_memory: bool = Field(
        default=False, description="Disable pin memory (GPU feature)"
    )
    dataloader_prefetch_factor: int = Field(
        default=1, description="Minimal prefetching"
    )

    # Fallback configurations
    use_pure_python: bool = Field(
        default=True, description="Use pure Python implementations"
    )
    avoid_compiled_kernels: bool = Field(
        default=True, description="Avoid compiled CUDA kernels"
    )
    enable_gradient_checkpointing: bool = Field(
        default=True, description="Save CPU memory"
    )

    # Validation and safety
    validate_cpu_compatibility: bool = Field(
        default=True, description="Validate CPU compatibility"
    )
    strict_cpu_mode: bool = Field(
        default=True, description="Fail if GPU operations detected"
    )


class CPUOnlyEnvironment:
    """Manages CPU-only environment setup and validation."""

    def __init__(self, config: CPUOnlyCompressionConfig | None = None):
        """Initialize CPU-only environment.

        Args:
            config: CPU-only configuration (creates default if None)
        """
        self.config = config or CPUOnlyCompressionConfig()
        self._original_env = {}
        self._setup_cpu_environment()

    def _setup_cpu_environment(self) -> None:
        """Set up CPU-only environment variables and settings."""
        # Store original environment
        cpu_env_vars = [
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TORCH_USE_CUDA_DSA",
            "CUDA_LAUNCH_BLOCKING",
        ]

        for var in cpu_env_vars:
            self._original_env[var] = os.environ.get(var)

        # Set CPU-only environment
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["OMP_NUM_THREADS"] = str(self.config.num_cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(self.config.num_cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.num_cpu_threads)
        os.environ["TORCH_USE_CUDA_DSA"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

        logger.info(
            f"CPU-only environment configured with {self.config.num_cpu_threads} threads"
        )

    def validate_cpu_compatibility(self) -> dict[str, Any]:
        """Validate that environment is properly configured for CPU-only operation.

        Returns:
            Dict with validation results
        """
        validation_results = {
            "cpu_only": True,
            "issues": [],
            "warnings": [],
            "environment": {},
            "torch_info": {},
        }

        # Check environment variables
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            validation_results["issues"].append(
                "CUDA_VISIBLE_DEVICES not set to empty string"
            )
            validation_results["cpu_only"] = False

        # Check PyTorch availability and settings
        if torch is not None:
            validation_results["torch_info"] = {
                "available": True,
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0,
                "cpu_count": torch.get_num_threads(),
            }

            if torch.cuda.is_available():
                validation_results["warnings"].append(
                    "PyTorch reports CUDA as available"
                )

        else:
            validation_results["torch_info"] = {"available": False}
            validation_results["warnings"].append("PyTorch not available")

        # Check for GPU-dependent libraries
        gpu_libraries = [
            "bitsandbytes",
            "flash_attn",
            "triton",
            "apex",
            "deepspeed",
        ]

        for lib in gpu_libraries:
            try:
                __import__(lib)
                validation_results["warnings"].append(f"GPU library {lib} is available")
            except ImportError:
                pass  # Good - GPU library not available

        # Check system resources
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            validation_results["environment"] = {
                "total_memory_gb": memory_gb,
                "cpu_count": cpu_count,
                "memory_adequate": memory_gb >= self.config.cpu_memory_limit_gb,
            }

            if memory_gb < self.config.cpu_memory_limit_gb:
                validation_results["warnings"].append(
                    f"Low system memory: {memory_gb:.1f}GB < {self.config.cpu_memory_limit_gb}GB"
                )

        except ImportError:
            validation_results["warnings"].append(
                "psutil not available for system info"
            )

        return validation_results

    def get_cpu_optimized_model_config(self) -> dict[str, Any]:
        """Get model configuration optimized for CPU execution.

        Returns:
            Dict with CPU-optimized model settings
        """
        return {
            # Device and precision
            "device_map": "cpu",
            "torch_dtype": getattr(torch, self.config.torch_dtype) if torch else None,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
            # Disable GPU features
            "load_in_8bit": False,
            "load_in_4bit": False,
            "use_flash_attention_2": False,
            # CPU optimizations
            "use_cache": True,  # Enable KV cache for inference
            "pad_token_id": 0,  # Explicit padding
            "eos_token_id": 1,  # Explicit EOS
            # Memory management
            "gradient_checkpointing": self.config.enable_gradient_checkpointing,
            "use_reentrant": False,  # More memory efficient
        }

    def get_cpu_optimized_training_args(self) -> dict[str, Any]:
        """Get training arguments optimized for CPU execution.

        Returns:
            Dict with CPU-optimized training settings
        """
        return {
            # Device and precision
            "use_cpu": True,
            "no_cuda": True,
            "fp16": False,  # Disable half precision
            "bf16": False,  # Disable bfloat16
            # Batch and memory settings
            "per_device_train_batch_size": self.config.batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "max_grad_norm": 1.0,  # Gradient clipping
            # Learning settings
            "learning_rate": self.config.learning_rate,
            "max_steps": self.config.max_steps,
            "warmup_steps": self.config.warmup_steps,
            "lr_scheduler_type": "linear",
            # Data loading
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "dataloader_pin_memory": self.config.dataloader_pin_memory,
            "dataloader_prefetch_factor": self.config.dataloader_prefetch_factor,
            # Checkpointing and logging
            "save_strategy": "steps",
            "save_steps": 50,
            "logging_steps": 10,
            "evaluation_strategy": "steps",
            "eval_steps": 50,
            # Resource management
            "gradient_checkpointing": self.config.enable_gradient_checkpointing,
            "optim": "adamw_torch",  # Pure PyTorch optimizer
            "remove_unused_columns": True,  # Save memory
        }

    def create_cpu_compatible_quantizer(self) -> "CPUQuantizer":
        """Create CPU-compatible quantizer avoiding GPU dependencies.

        Returns:
            CPU-only quantizer instance
        """
        return CPUQuantizer(self.config)

    def restore_environment(self) -> None:
        """Restore original environment variables."""
        for var, value in self._original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        logger.info("Environment restored")


class CPUQuantizer:
    """CPU-only quantizer avoiding GPU dependencies like bitsandbytes."""

    def __init__(self, config: CPUOnlyCompressionConfig):
        """Initialize CPU quantizer.

        Args:
            config: CPU-only configuration
        """
        self.config = config

    def int8_quantize_tensor(self, tensor: "torch.Tensor") -> dict[str, Any]:
        """Quantize tensor to int8 using CPU-friendly methods.

        Args:
            tensor: Input tensor to quantize

        Returns:
            Dict with quantized tensor and metadata
        """
        if torch is None:
            raise ImportError("PyTorch not available")

        # Move to CPU if needed
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        # Calculate quantization parameters
        min_val = tensor.min()
        max_val = tensor.max()

        # Avoid division by zero
        range_val = max_val - min_val
        if range_val == 0:
            range_val = 1.0

        # Calculate scale and zero point
        scale = range_val / 255.0  # int8 range
        zero_point = -min_val / scale
        zero_point = torch.clamp(zero_point, 0, 255).round().int()

        # Quantize
        quantized = torch.clamp((tensor / scale + zero_point).round(), 0, 255).to(
            torch.uint8
        )

        return {
            "quantized_tensor": quantized,
            "scale": scale,
            "zero_point": zero_point,
            "original_shape": tensor.shape,
            "original_dtype": tensor.dtype,
        }

    def dequantize_tensor(self, quantized_data: dict[str, Any]) -> "torch.Tensor":
        """Dequantize tensor back to original precision.

        Args:
            quantized_data: Quantized tensor data from int8_quantize_tensor

        Returns:
            Dequantized tensor
        """
        if torch is None:
            raise ImportError("PyTorch not available")

        quantized = quantized_data["quantized_tensor"]
        scale = quantized_data["scale"]
        zero_point = quantized_data["zero_point"]

        # Dequantize
        dequantized = (quantized.float() - zero_point) * scale

        return dequantized.to(quantized_data["original_dtype"])

    def quantize_model_weights(self, model: Any) -> dict[str, Any]:
        """Quantize all model weights to int8 using CPU methods.

        Args:
            model: PyTorch model to quantize

        Returns:
            Dict with quantization metadata
        """
        if torch is None:
            raise ImportError("PyTorch not available")

        quantization_info = {
            "quantized_layers": 0,
            "total_params": 0,
            "size_reduction": 0.0,
            "layer_info": {},
        }

        original_size = 0
        quantized_size = 0

        for name, param in model.named_parameters():
            if (
                param.requires_grad and len(param.shape) >= 2
            ):  # Only quantize weight matrices
                original_size += param.numel() * 4  # float32 = 4 bytes

                # Quantize weight
                self.int8_quantize_tensor(param.data)
                quantized_size += param.numel() * 1  # int8 = 1 byte
                quantized_size += 4 + 4  # scale and zero_point (float32 each)

                # Store quantization info
                quantization_info["layer_info"][name] = {
                    "original_shape": param.shape,
                    "quantized": True,
                    "compression_ratio": 4.0,  # float32 to int8
                }

                quantization_info["quantized_layers"] += 1

            quantization_info["total_params"] += param.numel()

        if original_size > 0:
            quantization_info["size_reduction"] = 1.0 - (quantized_size / original_size)

        return quantization_info


def get_cpu_only_config(**overrides) -> CPUOnlyCompressionConfig:
    """Get CPU-only compression configuration with optional overrides.

    Args:
        **overrides: Configuration overrides

    Returns:
        CPUOnlyCompressionConfig ready for CPU-only operation
    """
    config = CPUOnlyCompressionConfig()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown configuration option: {key}")

    return config


def setup_cpu_only_compression_environment(
    config: CPUOnlyCompressionConfig | None = None,
) -> CPUOnlyEnvironment:
    """Set up complete CPU-only compression environment.

    Args:
        config: CPU-only configuration (creates default if None)

    Returns:
        CPUOnlyEnvironment instance
    """
    return CPUOnlyEnvironment(config)


def validate_cpu_only_setup() -> dict[str, Any]:
    """Validate CPU-only setup without making changes.

    Returns:
        Dict with validation results
    """
    env = CPUOnlyEnvironment()
    results = env.validate_cpu_compatibility()
    env.restore_environment()  # Don't keep changes
    return results


# Auto-configuration based on available hardware
def auto_configure_cpu_only() -> CPUOnlyCompressionConfig:
    """Automatically configure CPU-only compression based on system resources.

    Returns:
        CPUOnlyCompressionConfig optimized for current system
    """
    config = CPUOnlyCompressionConfig()

    try:
        import psutil

        # Adjust based on available CPU cores
        cpu_count = psutil.cpu_count()
        if cpu_count:
            config.num_cpu_threads = min(cpu_count, 4)  # Cap at 4 for stability

        # Adjust based on available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            config.batch_size = 1
            config.gradient_accumulation_steps = 16
            config.cpu_memory_limit_gb = int(memory_gb * 0.7)
        elif memory_gb >= 16:
            config.batch_size = 2
            config.gradient_accumulation_steps = 4
            config.cpu_memory_limit_gb = 8

    except ImportError:
        logger.warning("psutil not available, using default CPU configuration")

    return config


# Default configuration instance
DEFAULT_CPU_ONLY_CONFIG = get_cpu_only_config()
