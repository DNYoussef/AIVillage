#!/usr/bin/env python3
"""Simple test script for Stage1 compression components"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent_forge"))

import tempfile

import torch


# Test individual components
def test_lfsr_generator():
    """Test LFSR generator"""
    print("Testing LFSR Generator...")

    from compression.seedlm import LFSRGenerator

    # Test initialization
    lfsr = LFSRGenerator(seed=12345)
    assert lfsr.register != 0

    # Test deterministic generation
    lfsr1 = LFSRGenerator(seed=12345)
    lfsr2 = LFSRGenerator(seed=12345)

    for _ in range(10):
        assert lfsr1.next_bit() == lfsr2.next_bit()

    # Test matrix generation
    matrix = lfsr.generate_matrix(4, 8)
    assert matrix.shape == (4, 8)
    assert matrix.dtype == torch.float32

    print("✅ LFSR Generator tests passed")


def test_seedlm_compressor():
    """Test SeedLM compressor"""
    print("Testing SeedLM Compressor...")

    from compression.seedlm import SeedLMCompressor

    compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=16)

    # Test compression roundtrip
    weight_matrix = torch.randn(3, 4)
    compressed_data = compressor.compress_weight_matrix(weight_matrix)

    assert "compressed_blocks" in compressed_data
    assert "original_shape" in compressed_data
    assert "compression_ratio" in compressed_data

    # Test reconstruction
    reconstructed = compressor.decompress_weight_matrix(compressed_data)
    assert reconstructed.shape == weight_matrix.shape

    # Test encode/decode
    encoded = compressor.encode(weight_matrix)
    decoded = compressor.decode(encoded, weight_matrix.shape)
    assert decoded.shape == weight_matrix.shape

    print("✅ SeedLM Compressor tests passed")


def test_bitnet_linear():
    """Test BitNet linear layer"""
    print("Testing BitNet Linear Layer...")

    from compression.stage1_bitnet import BitNetLinear, RMSNorm

    # Test BitNet layer
    layer = BitNetLinear(4, 3, bias=True)
    assert layer.in_features == 4
    assert layer.out_features == 3
    assert layer.bias is not None

    # Test forward pass
    x = torch.randn(2, 4)
    output = layer(x)
    assert output.shape == (2, 3)
    assert not torch.isnan(output).any()

    # Test RMSNorm
    norm = RMSNorm(4)
    x = torch.randn(2, 4)
    output = norm(x)
    assert output.shape == x.shape
    assert not torch.isnan(output).any()

    print("✅ BitNet Linear Layer tests passed")


def test_configuration():
    """Test configuration"""
    print("Testing Configuration...")

    from compression.stage1_config import Stage1Config

    config = Stage1Config()
    assert config.target_compression_ratio >= 10.0
    assert config.max_accuracy_drop <= 0.05

    # Test serialization
    config_dict = config.__dict__
    assert "bitnet_learning_rate" in config_dict
    assert "seedlm_block_size" in config_dict

    print("✅ Configuration tests passed")


def test_evaluation_harness():
    """Test evaluation harness"""
    print("Testing Evaluation Harness...")

    from compression.eval_utils import CompressionEvaluator

    evaluator = CompressionEvaluator("test_model")
    assert evaluator.model_path == "test_model"
    assert evaluator.device in ["cuda", "cpu"]

    # Test HellaSwag sample generation
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

    print("✅ Evaluation Harness tests passed")


def test_cli_interface():
    """Test CLI interface functions"""
    print("Testing CLI Interface...")

    from compression import run_stage1
    from compression.stage1 import main

    # Test that functions exist and are callable
    assert callable(main)
    assert callable(run_stage1)

    print("✅ CLI Interface tests passed")


def main():
    """Run all tests"""
    print("Running Stage1 Compression Tests...")
    print("=" * 50)

    try:
        test_lfsr_generator()
        test_seedlm_compressor()
        test_bitnet_linear()
        test_configuration()
        test_evaluation_harness()
        test_cli_interface()

        print("=" * 50)
        print("🎉 All tests passed!")
        print("Stage1 compression pipeline is ready for use")

        # Print usage example
        print("\nUsage:")
        print(
            "python -m agent_forge.compression.stage1 --input models/raw/model.pt --output models/compressed/model.stage1.pt"
        )

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
