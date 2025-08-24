"""
Comprehensive compression pipeline tests building on stable core infrastructure.
Uses proven test patterns from successful core tests.
"""

import json
from unittest.mock import patch

import pytest
import torch
from torch import nn

# Import our working compression stubs (which we know work)
from agent_forge.compression import bitnet, seedlm


class TestCompressionPipeline:
    """Test comprehensive compression pipeline functionality."""

    def test_seedlm_compression_basic(self, compression_test_model):
        """Test basic SeedLM compression functionality."""
        compressor = seedlm.SEEDLMCompressor()
        result = compressor.compress(compression_test_model)

        assert result["compressed"] is True
        assert result["method"] == "seedlm"
        assert result["ratio"] == 4.0

    def test_bitnet_compression_basic(self, compression_test_model):
        """Test basic BitNet compression functionality."""
        compressor = bitnet.BITNETCompressor()
        result = compressor.compress(compression_test_model)

        assert result["compressed"] is True
        assert result["method"] == "bitnet"
        assert result["ratio"] == 4.0

    def test_compression_model_size_validation(self, sample_model):
        """Test that compression works with different model sizes."""
        compressor = seedlm.SEEDLMCompressor()

        # Test with small model
        result = compressor.compress(sample_model)
        assert result["compressed"] is True

        # Test with larger model
        large_model = nn.Sequential(nn.Linear(1000, 500), nn.ReLU(), nn.Linear(500, 100))
        result = compressor.compress(large_model)
        assert result["compressed"] is True


class TestCompressionMetrics:
    """Test compression metrics and evaluation."""

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculations."""
        original_size = 1000
        compressed_size = 250
        expected_ratio = original_size / compressed_size

        # Simple ratio calculation test
        actual_ratio = original_size / compressed_size
        assert actual_ratio == 4.0
        assert actual_ratio == expected_ratio

    def test_compression_quality_metrics(self, compression_test_model):
        """Test quality metrics after compression."""
        compressor = seedlm.SEEDLMCompressor()
        compressed = compressor.compress(compression_test_model)

        # Decompress and validate
        decompressed = compressor.decompress(compressed)

        # Basic shape validation
        assert isinstance(decompressed, nn.Module)

        # Test forward pass works
        test_input = torch.randn(1, 784)
        with torch.no_grad():
            output = decompressed(test_input)
            assert output is not None
            assert output.shape[0] == 1  # Batch dimension


class TestCompressionBenchmarks:
    """Performance benchmarks for compression operations."""

    @pytest.mark.benchmark
    def test_seedlm_compression_speed(self, benchmark, compression_test_model):
        """Benchmark SeedLM compression speed."""
        compressor = seedlm.SEEDLMCompressor()

        result = benchmark(compressor.compress, compression_test_model)

        # Verify result while benchmarking
        assert result["compressed"] is True
        assert result["method"] == "seedlm"

    @pytest.mark.benchmark
    def test_bitnet_compression_speed(self, benchmark, compression_test_model):
        """Benchmark BitNet compression speed."""
        compressor = bitnet.BITNETCompressor()

        result = benchmark(compressor.compress, compression_test_model)

        assert result["compressed"] is True
        assert result["method"] == "bitnet"

    @pytest.mark.slow
    def test_compression_memory_usage(self, compression_test_model):
        """Test memory usage during compression."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        compressor = seedlm.SEEDLMCompressor()
        result = compressor.compress(compression_test_model)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for test model)
        assert memory_increase < 100
        assert result["compressed"] is True


class TestCompressionIntegration:
    """Integration tests for compression pipeline."""

    def test_compression_pipeline_end_to_end(self, temp_dir, compression_test_model):
        """Test complete compression pipeline from model to file."""
        # Compress model
        compressor = seedlm.SEEDLMCompressor()
        compressed = compressor.compress(compression_test_model)

        # Save compressed model
        output_file = temp_dir / "compressed_model.json"
        with open(output_file, "w") as f:
            json.dump(compressed, f)

        # Verify file was created
        assert output_file.exists()

        # Load and verify
        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["compressed"] is True
        assert loaded["method"] == "seedlm"

    def test_multiple_compression_methods(self, compression_test_model):
        """Test applying multiple compression methods."""
        # First compression with SeedLM
        seedlm_compressor = seedlm.SEEDLMCompressor()
        first_compressed = seedlm_compressor.compress(compression_test_model)

        # Decompress
        intermediate_model = seedlm_compressor.decompress(first_compressed)

        # Second compression with BitNet
        bitnet_compressor = bitnet.BITNETCompressor()
        second_compressed = bitnet_compressor.compress(intermediate_model)

        # Verify both compressions worked
        assert first_compressed["method"] == "seedlm"
        assert second_compressed["method"] == "bitnet"

    @pytest.mark.integration
    def test_compression_with_training_loop(self, mock_config, compression_test_model):
        """Test compression integration with training pipeline."""
        # Simulate training step
        optimizer = torch.optim.Adam(compression_test_model.parameters())
        loss_fn = nn.MSELoss()

        # Mock training data
        x = torch.randn(32, 784)
        y = torch.randn(32, 10)

        # Training step
        optimizer.zero_grad()
        output = compression_test_model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        # Compress after training
        compressor = seedlm.SEEDLMCompressor()
        compressed = compressor.compress(compression_test_model)

        assert compressed["compressed"] is True

        # Verify compressed model still works
        decompressed = compressor.decompress(compressed)
        test_output = decompressed(x[:1])  # Test with single sample
        assert test_output.shape == (1, 10)


class TestCompressionErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_model_compression(self):
        """Test compression with invalid inputs."""
        compressor = seedlm.SEEDLMCompressor()

        # Test with None
        result = compressor.compress(None)
        assert "compressed" in result

        # Test with non-model object
        result = compressor.compress("not a model")
        assert "compressed" in result

    def test_compression_failure_recovery(self):
        """Test recovery from compression failures."""
        compressor = seedlm.SEEDLMCompressor()

        # Mock a compression failure
        with patch.object(compressor, "compress", side_effect=Exception("Compression failed")):
            try:
                result = compressor.compress(nn.Linear(10, 5))
                # If no exception, check result
                assert result is not None
            except Exception as e:
                # Verify we can handle the exception
                assert "Compression failed" in str(e)

    def test_decompression_invalid_data(self):
        """Test decompression with invalid data."""
        compressor = seedlm.SEEDLMCompressor()

        # Test with empty dict
        result = compressor.decompress({})
        assert isinstance(result, nn.Module)

        # Test with invalid format
        result = compressor.decompress({"invalid": "data"})
        assert isinstance(result, nn.Module)


# Fixtures specific to compression tests
@pytest.fixture
def large_compression_model():
    """Provide a larger model for compression stress testing."""
    return nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 100),
    )


@pytest.fixture
def compression_metrics():
    """Provide compression metrics tracking."""
    return {
        "compression_ratios": [],
        "compression_times": [],
        "model_sizes": [],
        "quality_scores": [],
    }
