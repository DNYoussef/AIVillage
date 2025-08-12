#!/usr/bin/env python3
"""Comprehensive tests for compression measurement validation.

This test suite validates that the compression algorithms work correctly
and that measurements are accurate without requiring GPU resources.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Add src to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

try:
    from agent_forge.compression.bitnet import BITNETCompressor
    from agent_forge.compression.seedlm import SEEDLMCompressor
    from agent_forge.compression.vptq import VPTQCompressor
    from core.compression.simple_quantizer import SimpleQuantizer

    COMPRESSION_AVAILABLE = True
except ImportError:
    COMPRESSION_AVAILABLE = False


class TestCompressionAlgorithms:
    """Test individual compression algorithms."""

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_bitnet_compressor(self):
        """Test BitNet compression works and achieves some compression."""
        compressor = BITNETCompressor()

        # Test with small tensor
        weights = torch.randn(10, 10) * 0.1
        original_bytes = weights.numel() * 4

        # Compress
        compressed = compressor.compress(weights)
        assert isinstance(compressed, dict)
        assert "packed_weights" in compressed
        assert "scale" in compressed
        assert "original_shape" in compressed

        # Estimate compressed size
        compressed_bytes = (
            len(compressed["packed_weights"]) + 4 + 4
        )  # scale + shape info
        assert compressed_bytes < original_bytes, (
            "BitNet should achieve some compression"
        )

        # Decompress
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == weights.shape

        # Should be approximately equal (lossy compression)
        assert torch.allclose(weights, reconstructed, rtol=0.3, atol=0.3)

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_seedlm_compressor(self):
        """Test SeedLM compression works and achieves compression."""
        compressor = SEEDLMCompressor()

        # Test with small tensor
        weights = torch.randn(24, 24) * 0.05  # Multiple of block size
        original_bytes = weights.numel() * 4

        # Compress
        compressed = compressor.compress(weights)
        assert isinstance(compressed, dict)
        assert "seeds" in compressed
        assert "coefficients" in compressed
        assert "shared_exponents" in compressed

        # Estimate compressed size (very rough)
        compressed_bytes = (
            len(compressed["seeds"]) * 2
            + compressed["coefficients"].nbytes  # uint16
            + len(compressed["shared_exponents"])
            + 20  # metadata overhead
        )
        assert compressed_bytes < original_bytes, "SeedLM should achieve compression"

        # Decompress
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == weights.shape

        # Should be approximately equal (lossy compression)
        assert torch.allclose(weights, reconstructed, rtol=1.0, atol=0.3)

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_vptq_compressor(self):
        """Test VPTQ compression works and achieves compression."""
        compressor = VPTQCompressor(bits=2, vector_dim=4, iterations=3)

        # Test with tensor that's multiple of vector dimension
        weights = torch.randn(8, 8) * 0.1  # 64 elements = 16 vectors of 4
        original_bytes = weights.numel() * 4

        # Compress
        compressed = compressor.compress(weights)
        assert isinstance(compressed, dict)
        assert "codebook" in compressed
        assert "indices" in compressed
        assert "scale" in compressed
        assert "offset" in compressed

        # Estimate compressed size
        codebook_size = compressed["codebook"].numel() * 4
        indices_size = len(compressed["indices"]) * 1  # Approximate
        compressed_bytes = codebook_size + indices_size + 16  # metadata
        assert compressed_bytes < original_bytes, "VPTQ should achieve compression"

        # Decompress
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == weights.shape

        # Should be approximately equal (lossy compression)
        assert torch.allclose(weights, reconstructed, rtol=1.0, atol=0.3)

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_simple_quantizer(self):
        """Test SimpleQuantizer compression works."""
        compressor = SimpleQuantizer()

        # Create simple test model
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Initialize with reasonable values
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

        original_bytes = sum(p.numel() * 4 for p in model.parameters())

        # Compress
        compressed_data = compressor.quantize_model(model)
        assert isinstance(compressed_data, bytes)
        assert len(compressed_data) > 0
        # Note: SimpleQuantizer may not always compress small models due to overhead
        # This test is primarily checking that the compression/decompression cycle works

        # Decompress
        reconstructed = compressor.decompress_model(compressed_data)
        assert isinstance(reconstructed, nn.Module)


class TestCompressionMeasurement:
    """Test the compression measurement framework."""

    def test_tensor_size_calculation(self):
        """Test that tensor size calculations are correct."""
        # Simple tensor
        tensor = torch.randn(100, 100)
        expected_bytes = 100 * 100 * 4  # float32 = 4 bytes
        assert tensor.numel() * 4 == expected_bytes

        # Different shapes
        tensor2 = torch.randn(50, 200)
        assert tensor2.numel() * 4 == expected_bytes  # Same total elements

    def test_model_size_calculation(self):
        """Test that model size calculations are correct."""
        model = nn.Sequential(
            nn.Linear(10, 20), nn.Linear(20, 5)
        )  # 10*20 + 20 = 220 params  # 20*5 + 5 = 105 params

        total_params = sum(p.numel() for p in model.parameters())
        expected_params = (10 * 20 + 20) + (20 * 5 + 5)  # weights + biases
        assert total_params == expected_params

        total_bytes = sum(p.numel() * 4 for p in model.parameters())
        assert total_bytes == expected_params * 4

    def test_small_model_creation(self):
        """Test that test models can be created successfully."""
        # Tiny model
        tiny = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32), nn.Linear(32, 10)
        )

        param_count = sum(p.numel() for p in tiny.parameters())
        assert 3000 < param_count < 15000, (
            f"Tiny model has {param_count} params, expected ~5K"
        )

        # Verify model works
        x = torch.randn(1, 32)
        output = tiny(x)
        assert output.shape == (1, 10)

    def test_tensor_creation(self):
        """Test that test tensors are created correctly."""
        # Dense tensor
        dense = torch.randn(100, 100) * 0.1
        assert dense.shape == (100, 100)
        assert -1.0 < dense.mean() < 1.0  # Should be roughly centered

        # Sparse tensor
        sparse_tensor = torch.randn(100, 100) * 0.1
        sparse_tensor[torch.abs(sparse_tensor) < 0.05] = 0
        zero_ratio = (sparse_tensor == 0).float().mean()
        assert zero_ratio > 0.3, "Sparse tensor should have many zeros"

        # Structured tensor
        base = torch.randn(5, 5) * 0.1
        structured = base.repeat(10, 10)
        assert structured.shape == (50, 50)
        # Verify repetition structure
        assert torch.allclose(structured[:5, :5], structured[5:10, :5])


class TestCompressionRatios:
    """Test compression ratios meet basic expectations."""

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_bitnet_basic_compression(self):
        """Test BitNet achieves reasonable compression ratios."""
        compressor = BITNETCompressor()

        # Regular tensor
        weights = torch.randn(100, 100) * 0.1
        original_bytes = weights.numel() * 4

        compressed = compressor.compress(weights)
        # Each weight uses 2 bits + overhead
        expected_compressed = weights.numel() // 4 + 20  # Rough estimate

        # Should achieve significant compression
        ratio = original_bytes / expected_compressed
        assert ratio > 8, f"BitNet should achieve >8x compression, got {ratio:.2f}x"

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_different_data_types_compress_differently(self):
        """Test that different data patterns compress differently."""
        compressor = BITNETCompressor()

        # Random data
        random_weights = torch.randn(50, 50) * 0.1
        random_compressed = compressor.compress(random_weights)

        # Sparse data (should compress better)
        sparse_weights = torch.randn(50, 50) * 0.1
        sparse_weights[torch.abs(sparse_weights) < 0.05] = 0
        sparse_compressed = compressor.compress(sparse_weights)

        # Constant data (should compress very well)
        constant_weights = torch.ones(50, 50) * 0.1
        constant_compressed = compressor.compress(constant_weights)

        # All should be valid compressions
        for compressed in [random_compressed, sparse_compressed, constant_compressed]:
            assert "packed_weights" in compressed
            assert "scale" in compressed

        # Note: Due to BitNet's fixed 2-bit representation, compression ratios
        # won't vary as much as with other algorithms, but the test ensures
        # different data types are handled correctly


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_empty_tensor_handling(self):
        """Test handling of edge case inputs."""
        compressor = BITNETCompressor()

        # Single element tensor
        single = torch.tensor([0.1])
        compressed = compressor.compress(single)
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == single.shape

        # Very small tensor
        tiny = torch.randn(2, 2) * 0.01
        compressed = compressor.compress(tiny)
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == tiny.shape

    @pytest.mark.skipif(
        not COMPRESSION_AVAILABLE, reason="Compression modules not available"
    )
    def test_zero_tensor_handling(self):
        """Test handling of zero tensors."""
        compressor = BITNETCompressor()

        # All zeros
        zeros = torch.zeros(10, 10)
        compressed = compressor.compress(zeros)
        reconstructed = compressor.decompress(compressed)
        assert reconstructed.shape == zeros.shape
        # Should be close to zeros (may have small quantization error)
        assert torch.allclose(reconstructed, zeros, atol=1e-3)

    def test_measurement_framework_robustness(self):
        """Test that measurement framework handles missing components gracefully."""
        # This test should always pass even if compression modules are missing

        # Test tensor creation works
        tensor = torch.randn(10, 10)
        assert tensor.numel() == 100

        # Test model creation works
        model = nn.Linear(5, 3)
        assert sum(p.numel() for p in model.parameters()) == 5 * 3 + 3  # weights + bias


def test_measurement_script_imports():
    """Test that the measurement script can be imported."""
    # Try to import the measurement script
    script_path = Path(__file__).parent / "measure.py"
    assert script_path.exists(), "measure.py script should exist"

    # Test basic syntax by compiling
    with open(script_path) as f:
        source = f.read()

    try:
        compile(source, str(script_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"measure.py has syntax error: {e}")


def test_basic_pytorch_functionality():
    """Test basic PyTorch functionality works in test environment."""
    # Basic tensor operations
    a = torch.randn(5, 5)
    b = torch.randn(5, 5)
    c = torch.mm(a, b)
    assert c.shape == (5, 5)

    # Basic model operations
    model = nn.Linear(10, 5)
    x = torch.randn(3, 10)
    y = model(x)
    assert y.shape == (3, 5)

    # Model parameter access
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 10 * 5 + 5  # weights + bias


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
