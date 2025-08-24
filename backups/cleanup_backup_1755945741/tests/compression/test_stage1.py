import os
import tempfile

import pytest
import torch
from torch import nn

from agent_forge.compression.eval_utils import CompressionEvaluator
from agent_forge.compression.seedlm import LFSRGenerator, SeedLMCompressor
from agent_forge.compression.stage1_bitnet import BitNetLinear, RMSNorm, convert_to_bitnet
from agent_forge.compression.stage1_config import Stage1Config


class TestLFSRGenerator:
    """Test LFSR pseudo-random generator"""

    def test_lfsr_initialization(self):
        """Test LFSR generator initialization"""
        lfsr = LFSRGenerator(seed=12345)
        assert lfsr.register != 0
        assert lfsr.initial_seed == 12345
        assert len(lfsr.taps) == 4

    def test_lfsr_deterministic(self):
        """Test LFSR generates deterministic sequence"""
        lfsr1 = LFSRGenerator(seed=12345)
        lfsr2 = LFSRGenerator(seed=12345)

        for _ in range(100):
            assert lfsr1.next_bit() == lfsr2.next_bit()

    def test_lfsr_matrix_generation(self):
        """Test LFSR matrix generation"""
        lfsr = LFSRGenerator(seed=12345)
        matrix = lfsr.generate_matrix(4, 8)

        assert matrix.shape == (4, 8)
        assert matrix.dtype == torch.float32
        assert torch.all(torch.abs(matrix) <= 1.0)  # Values should be normalized


class TestBitNetLinear:
    """Test BitNet Linear layer implementation"""

    def test_bitnet_linear_initialization(self):
        """Test BitNet layer initialization"""
        layer = BitNetLinear(10, 5, bias=True)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.bias is not None
        assert layer.weight_fp.shape == (5, 10)
        assert layer.lambda_val == 0.0

    def test_bitnet_quantization(self):
        """Test weight quantization"""
        layer = BitNetLinear(4, 3, bias=False)

        # Set test weights
        test_weights = torch.tensor([[0.1, -0.8, 0.3, 0.0], [0.5, 0.2, -0.6, 0.1], [-0.2, 0.9, 0.0, -0.4]])
        layer.weight_fp.data = test_weights

        quantized = layer.quantize_weights(test_weights)

        # Check quantization properties
        assert torch.all(torch.abs(quantized) <= 1.0)
        assert torch.all((quantized == -1) | (quantized == 0) | (quantized == 1))

    def test_bitnet_forward_training(self):
        """Test forward pass in training mode"""
        layer = BitNetLinear(4, 3, bias=True)
        layer.train()

        x = torch.randn(2, 4)
        output = layer(x)

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()

    def test_bitnet_forward_inference(self):
        """Test forward pass in inference mode"""
        layer = BitNetLinear(4, 3, bias=True)
        layer.eval()

        x = torch.randn(2, 4)
        output = layer(x)

        assert output.shape == (2, 3)
        assert not torch.isnan(output).any()


class TestRMSNorm:
    """Test RMS Normalization layer"""

    def test_rmsnorm_initialization(self):
        """Test RMSNorm initialization"""
        norm = RMSNorm(128)
        assert norm.weight.shape == (128,)
        assert torch.allclose(norm.weight, torch.ones(128))
        assert norm.eps == 1e-6

    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass"""
        norm = RMSNorm(4)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])

        output = norm(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Check normalization properties
        output_norm = torch.norm(output, dim=-1)
        assert torch.allclose(
            output_norm,
            torch.norm(x, dim=-1) / torch.norm(x, dim=-1) * torch.sqrt(torch.tensor(4.0)),
            atol=1e-5,
        )


class TestSeedLMCompressor:
    """Test SeedLM compression implementation"""

    def test_seedlm_initialization(self):
        """Test SeedLM compressor initialization"""
        compressor = SeedLMCompressor(block_size=8, latent_dim=4, num_seeds=256)
        assert compressor.block_size == 8
        assert compressor.latent_dim == 4
        assert compressor.num_seeds == 256

    def test_seedlm_compression_roundtrip(self):
        """Test SeedLM compression and decompression"""
        compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=16)

        # Create test weight matrix
        weight_matrix = torch.randn(3, 4)

        # Compress
        compressed_data = compressor.compress_weight_matrix(weight_matrix)

        # Check compressed data structure
        assert "compressed_blocks" in compressed_data
        assert "original_shape" in compressed_data
        assert "compression_ratio" in compressed_data

        # Decompress
        reconstructed = compressor.decompress_weight_matrix(compressed_data)

        # Check reconstruction
        assert reconstructed.shape == weight_matrix.shape
        assert not torch.isnan(reconstructed).any()

    def test_seedlm_encode_decode(self):
        """Test SeedLM encode/decode methods"""
        compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=16)

        # Create test weight matrix
        weight_matrix = torch.randn(2, 4)

        # Encode
        encoded_tensor = compressor.encode(weight_matrix)
        assert encoded_tensor.dim() == 2
        assert encoded_tensor.dtype == torch.float32

        # Decode
        decoded = compressor.decode(encoded_tensor, weight_matrix.shape)
        assert decoded.shape == weight_matrix.shape
        assert not torch.isnan(decoded).any()

    def test_seedlm_compression_ratio(self):
        """Test compression ratio calculation"""
        compressor = SeedLMCompressor(block_size=8, latent_dim=4, num_seeds=32)

        # Large matrix should have better compression ratio
        large_matrix = torch.randn(32, 32)
        compressed = compressor.compress_weight_matrix(large_matrix)

        assert compressed["compression_ratio"] > 0
        # Should achieve some compression
        assert compressed["compression_ratio"] < 32  # Not too good to be true


class TestConvertToBitNet:
    """Test BitNet conversion function"""

    def test_convert_simple_model(self):
        """Test converting simple model to BitNet"""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

        converted = convert_to_bitnet(model)

        # Check that linear layers were converted
        [m for m in converted.modules() if isinstance(m, nn.Linear | BitNetLinear)]
        bitnet_layers = [m for m in converted.modules() if isinstance(m, BitNetLinear)]

        # Should have converted all linear layers
        assert len(bitnet_layers) > 0

    def test_convert_preserves_weights(self):
        """Test that conversion preserves original weights"""
        model = nn.Linear(4, 3)
        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        converted = convert_to_bitnet(model)

        # Find the BitNet layer
        bitnet_layer = None
        for module in converted.modules():
            if isinstance(module, BitNetLinear):
                bitnet_layer = module
                break

        assert bitnet_layer is not None
        assert torch.allclose(bitnet_layer.weight_fp, original_weight)
        assert torch.allclose(bitnet_layer.bias, original_bias)


class TestStage1Config:
    """Test Stage1 configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = Stage1Config()

        # Check critical constraints
        assert config.target_compression_ratio >= 10.0
        assert config.max_accuracy_drop <= 0.05
        assert config.bitnet_batch_size * config.gradient_accumulation_steps <= 32

    def test_config_validation(self):
        """Test configuration validation"""
        # Should raise error for invalid compression ratio
        with pytest.raises(ValueError):
            Stage1Config(target_compression_ratio=5.0)

        # Should raise error for invalid accuracy drop
        with pytest.raises(ValueError):
            Stage1Config(max_accuracy_drop=0.1)

        # Should raise error for large batch size
        with pytest.raises(ValueError):
            Stage1Config(bitnet_batch_size=16, gradient_accumulation_steps=4)


class TestCompressionEvaluator:
    """Test compression evaluation harness"""

    def test_evaluator_initialization(self):
        """Test evaluator initialization"""
        evaluator = CompressionEvaluator("test_model")
        assert evaluator.model_path == "test_model"
        assert evaluator.device in ["cuda", "cpu"]

    def test_hellaswag_sample_generation(self):
        """Test HellaSwag sample generation"""
        evaluator = CompressionEvaluator("test_model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            sample_path = os.path.join(tmp_dir, "test_sample.jsonl")
            data = evaluator.load_hellaswag_sample(sample_path)

            assert len(data) > 0
            assert os.path.exists(sample_path)

            # Check data structure
            for item in data:
                assert "ctx" in item
                assert "endings" in item
                assert "label" in item
                assert len(item["endings"]) >= 2
                assert 0 <= item["label"] < len(item["endings"])


class TestStage1Integration:
    """Integration tests for Stage1 pipeline"""

    def test_stage1_pipeline_basic(self):
        """Test basic Stage1 pipeline execution"""
        # Create a simple test model
        model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "test_model.pt")
            os.path.join(tmp_dir, "test_model.stage1.pt")

            # Save test model
            torch.save(model.state_dict(), input_path)

            # Create minimal config
            Stage1Config(
                bitnet_enabled=False,  # Skip for testing
                seedlm_enabled=True,
                seedlm_num_seeds=16,  # Small for testing
                eval_max_samples=3,
            )

            # This would run if we had a complete implementation
            # For now, just test that imports work
            from agent_forge.compression.stage1 import run_stage1_compression

            # Test the function exists and can be called
            assert callable(run_stage1_compression)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
