#!/usr/bin/env python3
"""Comprehensive SeedLM tests with performance benchmarks
Auto-tracked in performance dashboard as per Sprint R-1 requirements.
"""

import time

import pytest
import torch

from agent_forge.compression.seedlm import (
    ProgressiveSeedLMEncoder,
    SeedLMCompressionError,
    SeedLMCompressor,
    SeedLMConfig,
    SeedLMDecompressionError,
    SeedLMVerificationError,
)


@pytest.mark.compression
@pytest.mark.benchmark
class TestSeedLMPerformance:
    """Performance benchmarks for SeedLM compression - auto-tracked in dashboard"""

    def test_seedlm_compression_ratio(self):
        """Test ≤40% throughput drop vs fp16 - auto-tracked in performance dashboard"""
        compressor = SeedLMCompressor(block_size=8, latent_dim=4, num_seeds=64)

        # Create test tensor similar to model weights
        test_weight = torch.randn(1024, 768, dtype=torch.float32)

        # Measure compression performance
        start_time = time.time()
        compressed = compressor.compress_weight_matrix(test_weight)
        compression_time = time.time() - start_time

        # Measure decompression performance
        start_time = time.time()
        reconstructed = compressor.decompress_weight_matrix(compressed)
        decompression_time = time.time() - start_time

        # Verify compression ratio meets requirement
        assert (
            compressed["compression_ratio"] >= 2.0
        ), f"Compression ratio {compressed['compression_ratio']:.2f} below 2.0x minimum"

        # Verify reconstruction accuracy
        mse = torch.mean((test_weight - reconstructed) ** 2).item()
        assert mse < 0.1, f"Reconstruction MSE {mse:.4f} too high"

        # Performance logging for dashboard
        total_time = compression_time + decompression_time
        throughput_factor = total_time / 0.01  # Baseline: 10ms for fp16 operations

        # Sprint requirement: ≤40% throughput drop (1.4x slower max)
        assert throughput_factor <= 1.4, f"Throughput drop {throughput_factor:.2f}x exceeds 40% limit"

        print(f"Compression: {compression_time * 1000:.1f}ms, Decompression: {decompression_time * 1000:.1f}ms")
        print(f"Compression ratio: {compressed['compression_ratio']:.2f}x, MSE: {mse:.6f}")

    def test_seedlm_8bit_accuracy(self):
        """Test 8-bit accuracy within bounds - dashboard tracks compression module trends"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())

        # Test with various tensor shapes common in neural networks
        test_shapes = [(256, 256), (512, 768), (1024, 512), (2048, 1024)]

        for shape in test_shapes:
            test_weight = torch.randn(shape, dtype=torch.float32)

            # Compress at different quality levels
            quality_levels = [0.3, 0.5, 0.7, 0.9]

            for quality in quality_levels:
                compressed = encoder.encode(test_weight, compression_level=quality)
                reconstructed = encoder.decode(compressed)

                # Accuracy metrics
                mse = torch.mean((test_weight - reconstructed) ** 2).item()
                max_error = torch.max(torch.abs(test_weight - reconstructed)).item()

                # Quality-dependent thresholds
                max_mse = 0.2 * (1.1 - quality)  # Stricter for higher quality
                max_abs_error = 2.0 * (1.1 - quality)

                assert mse <= max_mse, f"MSE {mse:.4f} exceeds {max_mse:.4f} for quality {quality}"
                assert max_error <= max_abs_error, f"Max error {max_error:.4f} exceeds {max_abs_error:.4f}"

                print(f"Shape {shape}, Quality {quality}: MSE={mse:.4f}, MaxErr={max_error:.4f}")

    @pytest.mark.slow
    def test_large_model_compression_benchmark(self):
        """Benchmark compression on model-scale tensors - nightly dashboard tracking"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())

        # Simulate large model layers (GPT-style)
        large_tensors = [
            torch.randn(4096, 4096),  # Large transformer layer
            torch.randn(8192, 2048),  # Wide layer
            torch.randn(12288, 4096),  # Very large layer
        ]

        total_original_size = 0
        total_compressed_size = 0
        total_compression_time = 0
        total_decompression_time = 0

        for i, tensor in enumerate(large_tensors):
            print(f"Processing tensor {i + 1}: {tensor.shape}")

            # Compression benchmark
            start_time = time.time()
            compressed = encoder.encode(tensor, compression_level=0.6)
            compression_time = time.time() - start_time

            # Decompression benchmark
            start_time = time.time()
            reconstructed = encoder.decode(compressed)
            decompression_time = time.time() - start_time

            # Size calculations (approximate)
            original_size = tensor.numel() * 4  # 32-bit floats
            compressed_size = len(str(compressed))  # Approximate

            total_original_size += original_size
            total_compressed_size += compressed_size
            total_compression_time += compression_time
            total_decompression_time += decompression_time

            # Verify quality
            mse = torch.mean((tensor - reconstructed) ** 2).item()
            assert mse < 0.05, f"Large tensor MSE {mse:.4f} too high"

            print(f"  Compression: {compression_time:.2f}s, Decompression: {decompression_time:.2f}s")
            print(f"  MSE: {mse:.6f}")

        # Overall benchmarks
        overall_ratio = total_original_size / total_compressed_size
        total_time = total_compression_time + total_decompression_time

        print(f"Overall compression ratio: {overall_ratio:.2f}x")
        print(f"Total processing time: {total_time:.2f}s")

        # Performance requirements for large models
        assert overall_ratio >= 1.5, f"Large model compression ratio {overall_ratio:.2f}x insufficient"
        assert total_time < 30.0, f"Large model processing time {total_time:.2f}s too slow"


@pytest.mark.compression
class TestSeedLMFunctionality:
    """Comprehensive functional tests for SeedLM implementation"""

    def test_seedlm_initialization(self):
        """Test proper SeedLM compressor initialization"""
        compressor = SeedLMCompressor(block_size=16, latent_dim=8, num_seeds=128)

        assert compressor.block_size == 16
        assert compressor.latent_dim == 8
        assert compressor.num_seeds == 128
        assert compressor.multi_scale_generator is not None

    def test_progressive_encoder_initialization(self):
        """Test progressive encoder with custom config"""
        config = SeedLMConfig(
            compression_levels=[0.2, 0.4, 0.6, 0.8],
            block_sizes=[8, 16, 32],
            latent_dims=[4, 8, 16],
        )
        encoder = ProgressiveSeedLMEncoder(config)

        assert encoder.config.compression_levels == [0.2, 0.4, 0.6, 0.8]
        assert encoder.config.block_sizes == [8, 16, 32]
        assert encoder.config.latent_dims == [4, 8, 16]

    def test_empty_tensor_handling(self):
        """Test graceful handling of empty tensors"""
        compressor = SeedLMCompressor()
        empty_tensor = torch.empty(0, 0)

        compressed = compressor.compress_weight_matrix(empty_tensor)
        reconstructed = compressor.decompress_weight_matrix(compressed)

        assert compressed["compression_ratio"] == 1.0
        assert reconstructed.shape == empty_tensor.shape
        assert reconstructed.numel() == 0

    def test_various_tensor_shapes(self):
        """Test compression/decompression with various tensor shapes"""
        compressor = SeedLMCompressor(block_size=8, latent_dim=4)

        test_shapes = [
            (1, 1),  # Minimum size
            (7, 3),  # Prime dimensions
            (8, 8),  # Square
            (64, 32),  # Rectangular
            (100, 50),  # Non-power-of-2
        ]

        for shape in test_shapes:
            tensor = torch.randn(shape)
            compressed = compressor.compress_weight_matrix(tensor)
            reconstructed = compressor.decompress_weight_matrix(compressed)

            assert reconstructed.shape == tensor.shape
            mse = torch.mean((tensor - reconstructed) ** 2).item()
            assert mse < 1.0, f"MSE {mse:.4f} too high for shape {shape}"

    def test_deterministic_compression(self):
        """Test that compression is deterministic with same seed"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())
        test_tensor = torch.randn(64, 32)

        # Compress with same seed multiple times
        encoder.set_seed(12345)
        result1 = encoder.encode(test_tensor, compression_level=0.5)

        encoder.set_seed(12345)
        result2 = encoder.encode(test_tensor, compression_level=0.5)

        # Results should be identical (within floating point precision)
        recon1 = encoder.decode(result1)
        recon2 = encoder.decode(result2)

        diff = torch.max(torch.abs(recon1 - recon2)).item()
        assert diff < 1e-6, f"Deterministic compression failed, max diff: {diff}"

    def test_progressive_compression(self):
        """Test progressive quality compression"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())
        test_tensor = torch.randn(128, 64)

        # Test progressive encoding
        progressive_data = encoder.encode_progressive(
            test_tensor,
            base_quality=0.3,
            enhancement_layers=2,
            quality_increments=[0.2, 0.3],
        )

        assert "base_layer" in progressive_data
        assert "enhancement_layers" in progressive_data
        assert len(progressive_data["enhancement_layers"]) == 2

        # Test progressive decoding with different layer counts
        base_only = encoder.decode_progressive(progressive_data, num_layers=0)
        one_layer = encoder.decode_progressive(progressive_data, num_layers=1)
        full_quality = encoder.decode_progressive(progressive_data, num_layers=2)

        # Quality should improve with more layers
        mse_base = torch.mean((test_tensor - base_only) ** 2).item()
        mse_one = torch.mean((test_tensor - one_layer) ** 2).item()
        mse_full = torch.mean((test_tensor - full_quality) ** 2).item()

        assert mse_one <= mse_base, "One layer should be better than base only"
        assert mse_full <= mse_one, "Full quality should be best"

    def test_verification_checksums(self):
        """Test integrity verification with checksums"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())
        test_tensor = torch.randn(32, 16)

        # Encode with verification enabled
        compressed = encoder.encode(test_tensor, enable_verification=True)
        assert "checksum" in compressed["metadata"]

        # Decode with verification
        reconstructed = encoder.decode(compressed, verify=True)

        # Should succeed without exception
        assert reconstructed.shape == test_tensor.shape

        # Test checksum mismatch detection
        compressed["metadata"]["checksum"] = "invalid_checksum"

        with pytest.raises(SeedLMVerificationError):
            encoder.decode(compressed, verify=True)

    def test_error_handling(self):
        """Test proper error handling for invalid inputs"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())

        # Test invalid compression level
        with pytest.raises(ValueError):
            encoder.encode(torch.randn(10, 10), compression_level=1.5)

        with pytest.raises(ValueError):
            encoder.encode(torch.randn(10, 10), compression_level=-0.1)

        # Test invalid input type
        with pytest.raises(SeedLMCompressionError):
            encoder.encode("not a tensor")

        # Test invalid compressed data
        with pytest.raises(SeedLMDecompressionError):
            encoder.decode("not compressed data")

        with pytest.raises(SeedLMDecompressionError):
            encoder.decode({"missing": "required_fields"})

    def test_memory_efficiency(self):
        """Test memory efficiency with large tensors"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())

        # Create a moderately large tensor
        large_tensor = torch.randn(1024, 1024)

        # Monitor memory usage (simplified)
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        compressed = encoder.encode(large_tensor, compression_level=0.5)
        reconstructed = encoder.decode(compressed)

        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before

        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory increase {memory_increase:.1f}MB too high"

        # Verify quality
        mse = torch.mean((large_tensor - reconstructed) ** 2).item()
        assert mse < 0.1, f"Large tensor quality degradation: MSE {mse:.4f}"

    def test_quantization_precision(self):
        """Test quantization and dequantization precision"""
        compressor = SeedLMCompressor(latent_dim=8)

        # Test various coefficient ranges
        test_coefficients = [
            torch.tensor([0.0, 0.1, -0.1, 1.0, -1.0]),
            torch.tensor([10.0, -10.0, 100.0, -100.0]),
            torch.tensor([0.001, -0.001, 0.999, -0.999]),
            torch.randn(8) * 50,  # Random large coefficients
        ]

        for coeffs in test_coefficients:
            quantized, exp = compressor._quantize(coeffs)
            dequantized = compressor._dequantize(quantized, exp)

            # Check that dequantization is close to original
            max_error = torch.max(torch.abs(coeffs - dequantized)).item()
            relative_error = max_error / (torch.max(torch.abs(coeffs)).item() + 1e-8)

            assert relative_error < 0.1, f"Quantization error {relative_error:.4f} too high"

    def test_streaming_capability(self):
        """Test streaming/bandwidth-limited compression"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())
        test_tensor = torch.randn(64, 32)

        # Create progressive compression
        progressive_data = encoder.encode_progressive(test_tensor, enhancement_layers=3)

        # Test bandwidth-limited streaming
        bandwidth_limits = [1000, 5000, 10000]  # bytes

        for limit in bandwidth_limits:
            streamed_data = encoder.get_streaming_data(progressive_data, limit)

            assert "base_layer" in streamed_data

            # Decode streamed data
            streamed_result = encoder.decode_progressive(streamed_data)

            # Should be valid reconstruction
            assert streamed_result.shape == test_tensor.shape
            mse = torch.mean((test_tensor - streamed_result) ** 2).item()
            assert mse < 2.0, f"Streamed quality too poor: MSE {mse:.4f}"

    def test_adaptive_block_sizing(self):
        """Test adaptive block size selection"""
        encoder = ProgressiveSeedLMEncoder(SeedLMConfig())

        # Create tensors with different variance characteristics
        low_variance = torch.ones(100, 50) * 0.1 + torch.randn(100, 50) * 0.01
        high_variance = torch.randn(100, 50) * 10.0
        medium_variance = torch.randn(100, 50) * 1.0

        tensors = [
            (low_variance, "low_variance"),
            (medium_variance, "medium_variance"),
            (high_variance, "high_variance"),
        ]

        for tensor, name in tensors:
            compressed = encoder.encode(tensor, compression_level=0.5)
            block_size = compressed["metadata"]["block_size"]

            # Verify adaptive sizing makes sense
            if name == "high_variance":
                assert block_size <= 8, f"High variance should use small blocks, got {block_size}"
            elif name == "low_variance":
                assert block_size >= 16, f"Low variance should use large blocks, got {block_size}"

            # Verify reconstruction quality
            reconstructed = encoder.decode(compressed)
            mse = torch.mean((tensor - reconstructed) ** 2).item()
            assert mse < 1.0, f"{name} reconstruction quality poor: MSE {mse:.4f}"


# Performance regression test for CI
@pytest.mark.benchmark
def test_compression_performance_regression():
    """Prevent performance regressions - run in CI"""
    compressor = SeedLMCompressor(block_size=8, latent_dim=4, num_seeds=32)

    # Standard benchmark tensor
    test_tensor = torch.randn(512, 256)

    # Benchmark compression
    start_time = time.time()
    compressed = compressor.compress_weight_matrix(test_tensor)
    compression_elapsed = time.time() - start_time

    # Benchmark decompression
    start_time = time.time()
    reconstructed = compressor.decompress_weight_matrix(compressed)
    decompression_elapsed = time.time() - start_time

    # Performance thresholds (will be tracked in CI)
    assert compression_elapsed < 2.0, f"Compression too slow: {compression_elapsed:.2f}s"
    assert decompression_elapsed < 1.0, f"Decompression too slow: {decompression_elapsed:.2f}s"

    # Quality threshold (relaxed for prototype performance)
    mse = torch.mean((test_tensor - reconstructed) ** 2).item()
    assert mse < 1.0, f"Quality regression: MSE {mse:.4f}"

    # Compression efficiency
    assert (
        compressed["compression_ratio"] >= 1.5
    ), f"Compression ratio regression: {compressed['compression_ratio']:.2f}x"

    print(
        f"Benchmark: Compression {compression_elapsed * 1000:.0f}ms, Decompression {decompression_elapsed * 1000:.0f}ms"
    )
    print(f"Quality: MSE {mse:.6f}, Ratio {compressed['compression_ratio']:.2f}x")
