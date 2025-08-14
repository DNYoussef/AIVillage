"""Tests for CPU-Only Compression Configuration - Prompt D

Comprehensive validation that compression pipeline works completely on CPU without
GPU dependencies, CUDA libraries, or hardware acceleration requirements.

Integration Point: Validates CPU-only compression for Phase 4 testing
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from production.compression.cpu_only_config import (
    CPUOnlyCompressionConfig,
    CPUOnlyEnvironment,
    CPUQuantizer,
    auto_configure_cpu_only,
    get_cpu_only_config,
    setup_cpu_only_compression_environment,
    validate_cpu_only_setup,
)


def _is_torch_available() -> bool:
    """Check if PyTorch is available for testing."""
    try:
        import torch

        return True
    except ImportError:
        return False


class TestCPUOnlyCompressionConfig:
    """Test CPU-only compression configuration."""

    def test_cpu_only_config_creation(self):
        """Test that CPU-only config can be created with proper defaults."""
        config = CPUOnlyCompressionConfig()

        # Verify CPU-only settings
        assert config.device == "cpu"
        assert config.force_cpu is True
        assert config.mixed_precision is False
        assert config.use_cuda_kernels is False
        assert config.enable_flash_attention is False
        assert config.avoid_bitsandbytes is True

    def test_cpu_only_config_conservative_settings(self):
        """Test conservative settings for CPU execution."""
        config = CPUOnlyCompressionConfig()

        # Verify conservative batch sizes
        assert config.batch_size <= 2
        assert config.eval_batch_size <= 2
        assert config.gradient_accumulation_steps >= 4

        # Verify conservative learning settings
        assert config.learning_rate <= 1e-4
        assert config.max_steps <= 1000

    def test_cpu_only_config_memory_settings(self):
        """Test memory-conscious settings."""
        config = CPUOnlyCompressionConfig()

        assert config.low_cpu_mem_usage is True
        assert config.torch_dtype == "float32"  # CPU-compatible
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.enable_gradient_checkpointing is True

    def test_config_customization(self):
        """Test that CPU-only config can be customized."""
        custom_config = CPUOnlyCompressionConfig(
            num_cpu_threads=2,
            batch_size=1,
            cpu_memory_limit_gb=2,
            strict_cpu_mode=False,
        )

        assert custom_config.num_cpu_threads == 2
        assert custom_config.batch_size == 1
        assert custom_config.cpu_memory_limit_gb == 2
        assert custom_config.strict_cpu_mode is False


class TestCPUOnlyEnvironment:
    """Test CPU-only environment management."""

    def test_environment_initialization(self):
        """Test environment initialization."""
        config = CPUOnlyCompressionConfig(num_cpu_threads=2)
        env = CPUOnlyEnvironment(config)

        assert env.config.num_cpu_threads == 2
        assert isinstance(env._original_env, dict)

        # Check that CPU environment variables are set
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        assert os.environ.get("OMP_NUM_THREADS") == "2"

    def test_environment_restoration(self):
        """Test environment restoration."""
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")

        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        # Environment should be modified
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""

        # Restore environment
        env.restore_environment()

        # Should be back to original (or None)
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda

    def test_cpu_compatibility_validation(self):
        """Test CPU compatibility validation."""
        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        validation = env.validate_cpu_compatibility()

        assert isinstance(validation, dict)
        assert "cpu_only" in validation
        assert "issues" in validation
        assert "warnings" in validation
        assert "environment" in validation
        assert "torch_info" in validation

    def test_cpu_optimized_model_config(self):
        """Test CPU-optimized model configuration."""
        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        model_config = env.get_cpu_optimized_model_config()

        assert model_config["device_map"] == "cpu"
        assert model_config["low_cpu_mem_usage"] is True
        assert model_config["load_in_8bit"] is False
        assert model_config["load_in_4bit"] is False
        assert model_config["use_flash_attention_2"] is False

    def test_cpu_optimized_training_args(self):
        """Test CPU-optimized training arguments."""
        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        training_args = env.get_cpu_optimized_training_args()

        assert training_args["use_cpu"] is True
        assert training_args["no_cuda"] is True
        assert training_args["fp16"] is False
        assert training_args["bf16"] is False
        assert training_args["dataloader_pin_memory"] is False
        assert training_args["optim"] == "adamw_torch"


class TestCPUQuantizer:
    """Test CPU-only quantizer functionality."""

    def test_quantizer_creation(self):
        """Test quantizer creation."""
        config = CPUOnlyCompressionConfig()
        quantizer = CPUQuantizer(config)

        assert quantizer.config == config

    @pytest.mark.skipif(not _is_torch_available(), reason="PyTorch not available")
    def test_int8_quantization(self):
        """Test int8 tensor quantization."""
        import torch

        config = CPUOnlyCompressionConfig()
        quantizer = CPUQuantizer(config)

        # Create test tensor
        tensor = torch.randn(10, 10)

        # Quantize
        quantized_data = quantizer.int8_quantize_tensor(tensor)

        assert "quantized_tensor" in quantized_data
        assert "scale" in quantized_data
        assert "zero_point" in quantized_data
        assert quantized_data["quantized_tensor"].dtype == torch.uint8

        # Verify tensor is on CPU
        assert quantized_data["quantized_tensor"].device.type == "cpu"

    @pytest.mark.skipif(not _is_torch_available(), reason="PyTorch not available")
    def test_tensor_dequantization(self):
        """Test tensor dequantization."""
        import torch

        config = CPUOnlyCompressionConfig()
        quantizer = CPUQuantizer(config)

        # Create test tensor
        original_tensor = torch.randn(5, 5)

        # Quantize and dequantize
        quantized_data = quantizer.int8_quantize_tensor(original_tensor)
        dequantized_tensor = quantizer.dequantize_tensor(quantized_data)

        # Check shape preservation
        assert dequantized_tensor.shape == original_tensor.shape

        # Check approximate value preservation (some loss expected)
        mse = torch.mean((original_tensor - dequantized_tensor) ** 2)
        assert mse < 1.0  # Reasonable quantization error

    @pytest.mark.skipif(not _is_torch_available(), reason="PyTorch not available")
    def test_model_weight_quantization(self):
        """Test model weight quantization."""
        import torch.nn as nn

        config = CPUOnlyCompressionConfig()
        quantizer = CPUQuantizer(config)

        # Create simple test model
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))

        # Quantize model weights
        quantization_info = quantizer.quantize_model_weights(model)

        assert "quantized_layers" in quantization_info
        assert "total_params" in quantization_info
        assert "size_reduction" in quantization_info
        assert "layer_info" in quantization_info

        # Should have quantized the linear layers
        assert quantization_info["quantized_layers"] == 2
        assert (
            quantization_info["size_reduction"] > 0.5
        )  # Should achieve good compression


class TestCPUOnlyUtilities:
    """Test CPU-only utility functions."""

    def test_get_cpu_only_config_function(self):
        """Test main config function."""
        config = get_cpu_only_config(num_cpu_threads=4)

        assert isinstance(config, CPUOnlyCompressionConfig)
        assert config.num_cpu_threads == 4
        assert config.device == "cpu"

    def test_setup_cpu_only_compression_environment_function(self):
        """Test environment setup function."""
        config = CPUOnlyCompressionConfig(num_cpu_threads=1)
        env = setup_cpu_only_compression_environment(config)

        assert isinstance(env, CPUOnlyEnvironment)
        assert env.config.num_cpu_threads == 1

    def test_validate_cpu_only_setup_function(self):
        """Test validation function."""
        results = validate_cpu_only_setup()

        assert isinstance(results, dict)
        assert "cpu_only" in results

    @patch("production.compression.cpu_only_config.psutil")
    def test_auto_configure_cpu_only(self, mock_psutil):
        """Test auto-configuration based on system resources."""
        # Mock system with limited resources
        mock_psutil.cpu_count.return_value = 2
        mock_psutil.virtual_memory.return_value = MagicMock(total=4 * 1024**3)  # 4GB

        config = auto_configure_cpu_only()

        assert isinstance(config, CPUOnlyCompressionConfig)
        assert config.num_cpu_threads <= 4
        assert config.batch_size == 1  # Should use small batch for limited memory

    @patch("production.compression.cpu_only_config.psutil")
    def test_auto_configure_high_memory(self, mock_psutil):
        """Test auto-configuration with high memory."""
        # Mock system with more resources
        mock_psutil.cpu_count.return_value = 8
        mock_psutil.virtual_memory.return_value = MagicMock(total=16 * 1024**3)  # 16GB

        config = auto_configure_cpu_only()

        assert config.batch_size == 2  # Should use larger batch
        assert config.gradient_accumulation_steps == 4  # Less accumulation needed


class TestCPUOnlyIntegration:
    """Test integration scenarios for CPU-only compression."""

    def test_complete_cpu_only_workflow(self):
        """Test complete workflow from setup to quantization."""
        # Step 1: Setup environment
        config = get_cpu_only_config(num_cpu_threads=1, batch_size=1)
        env = setup_cpu_only_compression_environment(config)

        # Step 2: Validate setup
        validation = env.validate_cpu_compatibility()
        assert isinstance(validation, dict)

        # Step 3: Get optimized configurations
        model_config = env.get_cpu_optimized_model_config()
        training_args = env.get_cpu_optimized_training_args()

        assert model_config["device_map"] == "cpu"
        assert training_args["use_cpu"] is True

        # Step 4: Create quantizer
        quantizer = env.create_cpu_compatible_quantizer()
        assert isinstance(quantizer, CPUQuantizer)

        # Cleanup
        env.restore_environment()

    def test_environment_variable_management(self):
        """Test that environment variables are properly managed."""
        original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        original_omp = os.environ.get("OMP_NUM_THREADS")

        try:
            config = CPUOnlyCompressionConfig(num_cpu_threads=2)
            env = CPUOnlyEnvironment(config)

            # Should be set to CPU values
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            assert os.environ.get("OMP_NUM_THREADS") == "2"

            env.restore_environment()

            # Should be restored
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == original_cuda
            assert os.environ.get("OMP_NUM_THREADS") == original_omp

        finally:
            # Ensure cleanup even if test fails
            if original_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            if original_omp is not None:
                os.environ["OMP_NUM_THREADS"] = original_omp
            else:
                os.environ.pop("OMP_NUM_THREADS", None)

    def test_gpu_dependency_detection(self):
        """Test detection of GPU dependencies."""
        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        validation = env.validate_cpu_compatibility()

        # Should detect if problematic libraries are available
        gpu_libs = ["bitsandbytes", "flash_attn", "triton"]
        for lib in gpu_libs:
            try:
                __import__(lib)
                # If library is available, should be mentioned in warnings
                lib_warnings = [w for w in validation["warnings"] if lib in w]
                assert len(lib_warnings) > 0, f"Should warn about {lib} availability"
            except ImportError:
                pass  # Library not available, which is good

    @pytest.mark.skipif(not _is_torch_available(), reason="PyTorch not available")
    def test_torch_cpu_mode_enforcement(self):
        """Test that PyTorch is forced into CPU mode."""
        import torch

        config = CPUOnlyCompressionConfig()
        env = CPUOnlyEnvironment(config)

        # Create a tensor and ensure it's on CPU
        tensor = torch.randn(3, 3)
        assert tensor.device.type == "cpu"

        # Model config should force CPU device
        model_config = env.get_cpu_optimized_model_config()
        assert model_config["device_map"] == "cpu"

    def test_fallback_behavior_without_torch(self):
        """Test graceful fallback when PyTorch is not available."""
        with patch.dict("sys.modules", {"torch": None}):
            config = CPUOnlyCompressionConfig()
            env = CPUOnlyEnvironment(config)

            validation = env.validate_cpu_compatibility()

            # Should handle missing PyTorch gracefully
            assert validation["torch_info"]["available"] is False
            assert any("PyTorch not available" in w for w in validation["warnings"])


if __name__ == "__main__":
    # Run CPU-only compression validation
    print("=== Testing CPU-Only Compression ===")

    # Test basic configuration
    print("Testing CPU-only configuration...")
    config = get_cpu_only_config()
    print(
        f"OK CPU config created: device={config.device}, force_cpu={config.force_cpu}"
    )

    # Test environment setup
    print("Testing environment setup...")
    env = setup_cpu_only_compression_environment(config)
    print(f"OK Environment setup: threads={config.num_cpu_threads}")

    # Test validation
    print("Testing CPU compatibility validation...")
    validation = validate_cpu_only_setup()
    print(f"OK Validation completed: cpu_only={validation['cpu_only']}")

    # Test auto-configuration
    print("Testing auto-configuration...")
    auto_config = auto_configure_cpu_only()
    print(f"OK Auto-config: batch_size={auto_config.batch_size}")

    # Test quantizer if torch available
    if _is_torch_available():
        print("Testing CPU quantizer...")
        quantizer = CPUQuantizer(config)
        print("OK CPU quantizer created")

        # Test basic quantization
        import torch

        test_tensor = torch.randn(5, 5)
        quantized = quantizer.int8_quantize_tensor(test_tensor)
        print(f"OK Quantization: dtype={quantized['quantized_tensor'].dtype}")
    else:
        print("Skipping quantizer tests (PyTorch not available)")

    print("=== CPU-only compression validation completed ===")
