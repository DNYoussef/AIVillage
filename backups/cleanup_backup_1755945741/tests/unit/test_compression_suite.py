#!/usr/bin/env python3
"""
Unified Compression Test Suite
Combines best practices from multiple compression test files into single comprehensive suite
"""

from pathlib import Path
import sys
import unittest

import pytest
import torch

# Add project roots to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages"))


class TestCompressionSuite(unittest.TestCase):
    """Comprehensive compression test suite combining all compression test functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_model = self._create_test_model()
        self.config = {"compression_ratio": 0.5, "quantization_bits": 8, "cpu_only": True, "batch_size": 1}

    def _create_test_model(self):
        """Create a test model for compression"""
        return torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    # Core Compression Tests (from test_compression_comprehensive.py)
    def test_basic_compression(self):
        """Test basic model compression"""
        from agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(self.config)
        compressed_model = compressor.compress(self.test_model)

        # Verify compression occurred
        original_size = sum(p.numel() for p in self.test_model.parameters())
        compressed_size = sum(p.numel() for p in compressed_model.parameters())
        self.assertLess(compressed_size, original_size)

    # SeedLM Tests (from test_seedlm_core.py)
    def test_seedlm_compression(self):
        """Test SeedLM-specific compression"""
        from agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(self.config)
        compressed_model = compressor.compress(self.test_model)

        # Test SeedLM features
        self.assertTrue(hasattr(compressed_model, "seed_params"))
        self.assertTrue(compressed_model.seed_params is not None)

    def test_seedlm_decompression(self):
        """Test SeedLM decompression"""
        from agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(self.config)
        compressed_model = compressor.compress(self.test_model)
        decompressed_model = compressor.decompress(compressed_model)

        # Verify shapes match
        for (name1, param1), (name2, param2) in zip(
            self.test_model.named_parameters(), decompressed_model.named_parameters()
        ):
            self.assertEqual(param1.shape, param2.shape)

    # Stage 1 Tests (from test_stage1_compression.py)
    def test_stage1_bitnet_quantization(self):
        """Test Stage 1 BitNet quantization"""
        from agent_forge.compression.stage1_bitnet import BitNetQuantizer

        quantizer = BitNetQuantizer(bits=1.58)
        quantized_model = quantizer.quantize(self.test_model)

        # Verify quantization to {-1, 0, +1}
        for param in quantized_model.parameters():
            unique_vals = torch.unique(param)
            self.assertTrue(len(unique_vals) <= 3)

    # CPU-Only Tests (from test_cpu_only_compression.py)
    def test_cpu_only_compression(self):
        """Test CPU-only compression for mobile deployment"""
        self.config["cpu_only"] = True
        self.config["mobile_optimized"] = True

        from agent_forge.compression.mobile_compressor import MobileCompressor

        compressor = MobileCompressor(self.config)
        compressed_model = compressor.compress(self.test_model)

        # Verify no CUDA operations
        self.assertFalse(next(compressed_model.parameters()).is_cuda)

        # Verify mobile optimization
        self.assertTrue(hasattr(compressed_model, "mobile_optimized"))
        self.assertTrue(compressed_model.mobile_optimized)

    def test_memory_efficient_compression(self):
        """Test memory-efficient compression for resource-constrained devices"""
        self.config["max_memory_mb"] = 100
        self.config["chunk_size"] = 1024

        from agent_forge.compression.memory_efficient import MemoryEfficientCompressor

        compressor = MemoryEfficientCompressor(self.config)
        compressed_model = compressor.compress(self.test_model)

        # Verify memory usage is within limits
        model_size_mb = sum(p.numel() * p.element_size() for p in compressed_model.parameters()) / (1024 * 1024)
        self.assertLess(model_size_mb, self.config["max_memory_mb"])

    # Integration Tests
    def test_compression_pipeline_integration(self):
        """Test full compression pipeline integration"""
        from agent_forge.compression.pipeline import CompressionPipeline

        pipeline = CompressionPipeline(["seedlm", "bitnet", "vptq"])

        compressed_model = pipeline.compress(self.test_model, self.config)

        # Verify all stages applied
        self.assertTrue(hasattr(compressed_model, "compression_stages"))
        self.assertEqual(len(compressed_model.compression_stages), 3)

    # Performance Tests
    @pytest.mark.benchmark
    def test_compression_performance(self):
        """Test compression performance metrics"""
        import time

        from agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(self.config)

        start_time = time.time()
        compressed_model = compressor.compress(self.test_model)
        compression_time = time.time() - start_time

        # Verify reasonable compression time (< 10 seconds for small model)
        self.assertLess(compression_time, 10.0)

        # Verify compression ratio
        original_size = sum(p.numel() for p in self.test_model.parameters())
        compressed_size = sum(p.numel() for p in compressed_model.parameters())
        compression_ratio = compressed_size / original_size

        self.assertLess(compression_ratio, self.config["compression_ratio"])

    # Error Handling Tests
    def test_invalid_model_handling(self):
        """Test handling of invalid model inputs"""
        from agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(self.config)

        # Test with None
        with self.assertRaises(ValueError):
            compressor.compress(None)

        # Test with invalid model type
        with self.assertRaises(TypeError):
            compressor.compress("not a model")

    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        from agent_forge.compression.seedlm import SEEDLMCompressor

        # Test with invalid compression ratio
        invalid_config = {"compression_ratio": 2.0}  # > 1.0
        with self.assertRaises(ValueError):
            SEEDLMCompressor(invalid_config)

        # Test with missing required config
        with self.assertRaises(KeyError):
            SEEDLMCompressor({})


class TestCompressionIntegration(unittest.TestCase):
    """Integration tests for compression with other systems"""

    def test_compression_with_agent_forge(self):
        """Test compression integration with Agent Forge pipeline"""
        from agent_forge.compression.seedlm import SEEDLMCompressor
        from agent_forge.core.unified_pipeline import UnifiedPipeline

        pipeline = UnifiedPipeline()
        compressor = SEEDLMCompressor({"compression_ratio": 0.5})

        # Add compression phase
        pipeline.add_phase("compression", compressor)

        model = torch.nn.Linear(10, 10)
        result = pipeline.run(model)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "model"))

    def test_compression_with_mobile_deployment(self):
        """Test compression for mobile deployment"""
        from agent_forge.compression.mobile_compressor import MobileCompressor

        config = {"target_device": "mobile", "max_model_size_mb": 50, "quantization_bits": 8, "cpu_only": True}

        compressor = MobileCompressor(config)
        model = torch.nn.Linear(1000, 1000)

        compressed = compressor.compress_for_mobile(model)

        # Verify mobile compatibility
        self.assertTrue(compressed.mobile_ready)
        self.assertLess(compressed.size_mb, config["max_model_size_mb"])


if __name__ == "__main__":
    unittest.main()
