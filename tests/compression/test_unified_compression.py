#!/usr/bin/env python3
"""Comprehensive tests for unified compression system.

This replaces the fragmented test suite with a consolidated set of tests
that cover all compression functionality in a structured way.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.production.compression import (
    CompressionResult,
    CompressionStrategy,
    UnifiedCompressor,
    compress_advanced,
    compress_mobile,
    compress_simple,
)


class TinyModel(nn.Module):
    """Tiny model for testing."""

    def __init__(self, vocab_size=1000, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


class MediumModel(nn.Module):
    """Medium model for testing (simulates 100M+ params)."""

    def __init__(self):
        super().__init__()
        # Create a model with ~100M parameters
        self.layers = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(100)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


@pytest.fixture
def tiny_model():
    """Fixture for tiny model."""
    return TinyModel()


@pytest.fixture
def medium_model():
    """Fixture for medium model."""
    return MediumModel()


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "test output"
    return tokenizer


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


class TestUnifiedCompressor:
    """Test the unified compressor class."""

    def test_initialization(self):
        """Test compressor initialization with different strategies."""
        # Default initialization
        compressor = UnifiedCompressor()
        assert compressor.strategy == CompressionStrategy.AUTO
        assert compressor.mobile_target_mb == 100
        assert compressor.accuracy_threshold == 0.95

        # Custom initialization
        compressor = UnifiedCompressor(
            strategy=CompressionStrategy.SIMPLE,
            mobile_target_mb=50,
            accuracy_threshold=0.90,
        )
        assert compressor.strategy == CompressionStrategy.SIMPLE
        assert compressor.mobile_target_mb == 50
        assert compressor.accuracy_threshold == 0.90

    def test_model_size_estimation(self, tiny_model):
        """Test model size estimation."""
        compressor = UnifiedCompressor()
        size_mb = compressor._estimate_model_size(tiny_model)

        # Should be a reasonable size for tiny model
        assert 0.1 < size_mb < 10.0
        assert isinstance(size_mb, float)

    def test_strategy_selection_tiny_model(self, tiny_model):
        """Test automatic strategy selection for tiny model."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.AUTO)
        strategy = compressor._select_strategy(tiny_model)

        # Tiny model should use simple compression
        assert strategy == CompressionStrategy.SIMPLE

    def test_strategy_selection_medium_model(self, medium_model):
        """Test automatic strategy selection for medium model."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.AUTO)
        strategy = compressor._select_strategy(medium_model)

        # Medium model should use mobile or advanced
        assert strategy in [CompressionStrategy.MOBILE, CompressionStrategy.ADVANCED]

    @pytest.mark.asyncio
    async def test_simple_compression(self, tiny_model):
        """Test simple compression pipeline."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.SIMPLE, enable_benchmarking=False)

        result = await compressor.compress_model(tiny_model)

        assert isinstance(result, CompressionResult)
        assert result.strategy_used == CompressionStrategy.SIMPLE
        assert result.compression_ratio > 1.0
        assert result.compression_time_seconds > 0
        assert result.original_size_mb > result.compressed_size_mb

    @pytest.mark.asyncio
    async def test_mobile_compression(self, tiny_model):
        """Test mobile compression pipeline."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.MOBILE, enable_benchmarking=False)

        result = await compressor.compress_model(tiny_model)

        assert isinstance(result, CompressionResult)
        assert result.strategy_used == CompressionStrategy.MOBILE
        assert result.compression_ratio >= 1.0

    @pytest.mark.asyncio
    async def test_compression_with_output_path(self, tiny_model, temp_dir):
        """Test compression with file output."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.SIMPLE, enable_benchmarking=False)

        output_path = temp_dir / "compressed_model"
        await compressor.compress_model(tiny_model, output_path=output_path)

        # Check that output directory was created
        assert output_path.exists()
        assert output_path.is_dir()

    @pytest.mark.asyncio
    async def test_compression_with_benchmarking(self, tiny_model, mock_tokenizer):
        """Test compression with benchmarking enabled."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.SIMPLE, enable_benchmarking=True)

        with patch.object(tiny_model, "generate", return_value=torch.tensor([[1, 2, 3]])):
            result = await compressor.compress_model(tiny_model, tokenizer=mock_tokenizer)

        assert result.benchmark_metrics is not None
        assert isinstance(result.benchmark_metrics, dict)

    @pytest.mark.asyncio
    async def test_compression_fallback(self, tiny_model):
        """Test compression fallback when advanced fails."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.ADVANCED)

        # Mock the advanced compression to fail
        with patch.object(
            compressor,
            "_apply_advanced_compression",
            side_effect=Exception("Test error"),
        ):
            result = await compressor.compress_model(tiny_model)

        # Should fallback to simple compression
        assert result.strategy_used == CompressionStrategy.SIMPLE

    def test_get_compression_info(self):
        """Test compression information retrieval."""
        compressor = UnifiedCompressor()
        info = compressor.get_compression_info()

        assert "strategies" in info
        assert "simple_ratio" in info
        assert "compressors" in info
        assert isinstance(info["strategies"], list)
        assert CompressionStrategy.AUTO.value in info["strategies"]


class TestConvenienceFunctions:
    """Test convenience functions for compression."""

    @pytest.mark.asyncio
    async def test_compress_simple_function(self, tiny_model):
        """Test compress_simple convenience function."""
        result = await compress_simple(tiny_model, enable_benchmarking=False)

        assert isinstance(result, CompressionResult)
        assert result.strategy_used == CompressionStrategy.SIMPLE

    @pytest.mark.asyncio
    async def test_compress_mobile_function(self, tiny_model):
        """Test compress_mobile convenience function."""
        result = await compress_mobile(tiny_model, enable_benchmarking=False)

        assert isinstance(result, CompressionResult)
        assert result.strategy_used == CompressionStrategy.MOBILE

    @pytest.mark.asyncio
    async def test_compress_advanced_function(self, tiny_model):
        """Test compress_advanced convenience function."""
        # This might fail due to missing dependencies, so we expect fallback
        result = await compress_advanced(tiny_model, enable_benchmarking=False)

        assert isinstance(result, CompressionResult)
        # Could be advanced or simple (if fallback occurred)
        assert result.strategy_used in [
            CompressionStrategy.ADVANCED,
            CompressionStrategy.SIMPLE,
        ]


class TestCompressionResult:
    """Test CompressionResult dataclass."""

    def test_compression_result_creation(self):
        """Test creating compression result."""
        result = CompressionResult(
            original_size_mb=100.0,
            compressed_size_mb=25.0,
            compression_ratio=4.0,
            compression_time_seconds=10.5,
            strategy_used=CompressionStrategy.SIMPLE,
        )

        assert result.original_size_mb == 100.0
        assert result.compressed_size_mb == 25.0
        assert result.compression_ratio == 4.0
        assert result.compression_time_seconds == 10.5
        assert result.strategy_used == CompressionStrategy.SIMPLE
        assert result.model_accuracy_retained is None
        assert result.mobile_compatible is False
        assert result.benchmark_metrics is None

    def test_compression_result_mobile_compatibility(self):
        """Test mobile compatibility detection."""
        # Mobile compatible
        result = CompressionResult(
            original_size_mb=500.0,
            compressed_size_mb=50.0,  # Under 100MB default
            compression_ratio=10.0,
            compression_time_seconds=5.0,
            strategy_used=CompressionStrategy.MOBILE,
            mobile_compatible=True,
        )
        assert result.mobile_compatible is True

        # Not mobile compatible
        result = CompressionResult(
            original_size_mb=500.0,
            compressed_size_mb=150.0,  # Over 100MB default
            compression_ratio=3.33,
            compression_time_seconds=5.0,
            strategy_used=CompressionStrategy.SIMPLE,
            mobile_compatible=False,
        )
        assert result.mobile_compatible is False


class TestCompressionStrategy:
    """Test CompressionStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert CompressionStrategy.AUTO.value == "auto"
        assert CompressionStrategy.SIMPLE.value == "simple"
        assert CompressionStrategy.ADVANCED.value == "advanced"
        assert CompressionStrategy.MOBILE.value == "mobile"

    def test_strategy_from_string(self):
        """Test creating strategy from string."""
        assert CompressionStrategy("auto") == CompressionStrategy.AUTO
        assert CompressionStrategy("simple") == CompressionStrategy.SIMPLE
        assert CompressionStrategy("advanced") == CompressionStrategy.ADVANCED
        assert CompressionStrategy("mobile") == CompressionStrategy.MOBILE


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_model_type(self):
        """Test handling of invalid model types."""
        compressor = UnifiedCompressor()

        with pytest.raises((TypeError, AttributeError)):
            await compressor.compress_model("invalid_model_path")

    @pytest.mark.asyncio
    async def test_compression_with_missing_dependencies(self, tiny_model):
        """Test graceful handling of missing dependencies."""
        compressor = UnifiedCompressor()

        # Mock missing dependencies
        with (
            patch(
                "src.production.compression.unified_compressor.bitnet_compress",
                side_effect=ImportError,
            ),
            pytest.raises(ImportError),
        ):
            await compressor._apply_simple_compression(tiny_model)


class TestIntegration:
    """Integration tests with other components."""

    @pytest.mark.asyncio
    async def test_end_to_end_compression_pipeline(self, tiny_model, temp_dir):
        """Test complete compression pipeline."""
        compressor = UnifiedCompressor(
            strategy=CompressionStrategy.AUTO,
            mobile_target_mb=50,
            enable_benchmarking=False,
        )

        output_path = temp_dir / "integrated_test"
        result = await compressor.compress_model(tiny_model, output_path=output_path)

        # Verify results
        assert result.compression_ratio > 1.0
        assert result.strategy_used == CompressionStrategy.SIMPLE  # Auto-selected for tiny model
        assert output_path.exists()

        # Verify compression info
        info = compressor.get_compression_info()
        assert "strategies" in info
        assert "compressors" in info


# Performance benchmarks (run separately)
class TestPerformance:
    """Performance tests for compression system."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_compression_performance_simple(self, tiny_model):
        """Test performance of simple compression."""
        compressor = UnifiedCompressor(strategy=CompressionStrategy.SIMPLE, enable_benchmarking=False)

        # Measure compression time
        import time

        start_time = time.time()
        result = await compressor.compress_model(tiny_model)
        end_time = time.time()

        # Should complete quickly for tiny model
        assert (end_time - start_time) < 10.0  # Less than 10 seconds
        assert result.compression_time_seconds < 10.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_usage_during_compression(self, tiny_model):
        """Test memory usage during compression."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        compressor = UnifiedCompressor(enable_benchmarking=False)
        await compressor.compress_model(tiny_model)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should not use excessive memory for tiny model
        assert memory_increase < 100  # Less than 100MB increase


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
