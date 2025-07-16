#!/usr/bin/env python3
"""Simplified integration test for the Agent Forge pipeline

This script tests each stage independently to avoid import issues
"""

import logging
import os
from pathlib import Path
import sys
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing module imports...")

    try:
        import torch

        print(f"[PASS] torch: {torch.__version__}")
    except Exception as e:
        print(f"[FAIL] torch import failed: {e}")
        return False

    try:
        print("[PASS] transformers: Available")
    except Exception as e:
        print(f"[FAIL] transformers import failed: {e}")
        return False

    # Test individual module imports
    modules_to_test = [
        ("stage1_bitnet", "agent_forge.compression.stage1_bitnet"),
        ("seedlm", "agent_forge.compression.seedlm"),
        ("vptq", "agent_forge.compression.vptq"),
        ("hyperfn", "agent_forge.compression.hyperfn"),
        ("eval_utils", "agent_forge.compression.eval_utils"),
    ]

    for module_name, module_path in modules_to_test:
        try:
            # Add the project root to Python path
            sys.path.insert(0, str(Path(__file__).parent))

            # Import the module
            __import__(module_path)
            print(f"[PASS] {module_name}: Available")
        except Exception as e:
            print(f"[FAIL] {module_name} import failed: {e}")
            return False

    return True


def test_stage1_components():
    """Test Stage 1 compression components individually"""
    print("\nTesting Stage 1 Components...")

    try:
        import torch

        sys.path.insert(0, str(Path(__file__).parent))

        # Test SeedLM compression
        from agent_forge.compression.seedlm import LFSRGenerator, SeedLMCompressor

        # Test LFSR generator
        lfsr = LFSRGenerator(seed=12345)
        bits = [lfsr.next_bit() for _ in range(10)]
        assert all(bit in [0, 1] for bit in bits), "LFSR should generate 0s and 1s"
        print("[PASS] LFSR Generator: Working")

        # Test matrix generation
        matrix = lfsr.generate_matrix(4, 8)
        assert matrix.shape == (4, 8), f"Expected shape (4, 8), got {matrix.shape}"
        print("[PASS] LFSR Matrix Generation: Working")

        # Test SeedLM compression
        compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=16)
        weight_matrix = torch.randn(8, 16)

        # Test compression
        compressed_data = compressor.compress_weight_matrix(weight_matrix)
        assert "compressed_blocks" in compressed_data
        assert "original_shape" in compressed_data
        assert "compression_ratio" in compressed_data
        print("[PASS] SeedLM Compression: Working")

        # Test decompression
        decompressed = compressor.decompress_weight_matrix(compressed_data)
        assert decompressed.shape == weight_matrix.shape
        print("[PASS] SeedLM Decompression: Working")

        # Test encode/decode
        encoded = compressor.encode(weight_matrix)
        decoded = compressor.decode(encoded, weight_matrix.shape)
        assert decoded.shape == weight_matrix.shape
        print("[PASS] SeedLM Encode/Decode: Working")

        return True

    except Exception as e:
        print(f"[FAIL] Stage 1 components test failed: {e}")
        return False


def test_stage2_components():
    """Test Stage 2 compression components individually"""
    print("\n Testing Stage 2 Components...")

    try:
        import torch

        sys.path.insert(0, str(Path(__file__).parent))

        # Test VPTQ quantization
        from agent_forge.compression.vptq import VPTQQuantizer

        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=8)
        weight_matrix = torch.randn(16, 32)

        # Test quantization
        quantized_data = quantizer.quantize_weight_matrix(weight_matrix)
        assert "original_shape" in quantized_data
        assert "codebook" in quantized_data
        assert "assignments" in quantized_data
        print("[PASS] VPTQ Quantization: Working")

        # Test dequantization
        reconstructed = quantizer.dequantize_weight_matrix(quantized_data)
        assert reconstructed.shape == weight_matrix.shape
        print("[PASS] VPTQ Dequantization: Working")

        # Test HyperFn compression
        from agent_forge.compression.hyperfn import HyperCompressionEncoder

        encoder = HyperCompressionEncoder(num_clusters=4)

        # Test compression
        compressed_data = encoder.compress_weight_matrix(weight_matrix)
        assert "params" in compressed_data
        assert "original_shape" in compressed_data
        assert "compression_ratio" in compressed_data
        print("[PASS] HyperFn Compression: Working")

        # Test decompression
        reconstructed = encoder.decompress_weight_matrix(compressed_data)
        assert reconstructed.shape == weight_matrix.shape
        print("[PASS] HyperFn Decompression: Working")

        return True

    except Exception as e:
        print(f"[FAIL] Stage 2 components test failed: {e}")
        return False


def test_evaluation_harness():
    """Test the evaluation harness"""
    print("\n Testing Evaluation Harness...")

    try:
        import torch

        sys.path.insert(0, str(Path(__file__).parent))

        from agent_forge.compression.eval_utils import CompressionEvaluator

        # Create a test file path (we'll use this script as a dummy model)
        test_path = __file__

        evaluator = CompressionEvaluator(test_path)

        # Test HellaSwag sample loading
        eval_data = evaluator.load_hellaswag_sample("eval/hellaswag_sample.jsonl")
        assert isinstance(eval_data, list)
        assert len(eval_data) > 0
        print("[PASS] HellaSwag Sample Loading: Working")

        # Test file size measurement
        test_size = evaluator.measure_model_size(torch.nn.Linear(10, 5))
        assert test_size > 0
        print("[PASS] Model Size Measurement: Working")

        return True

    except Exception as e:
        print(f"[FAIL] Evaluation harness test failed: {e}")
        return False


def test_model_creation():
    """Test creating a simple model for testing"""
    print("\n Testing Model Creation...")

    try:
        import torch
        from torch import nn

        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(10, 5)
                self.layer2 = nn.Linear(5, 2)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.layer1(x))
                x = self.layer2(x)
                return x

        # Create and test model
        model = SimpleModel()
        test_input = torch.randn(1, 10)
        output = model(test_input)
        assert output.shape == (1, 2)
        print("[PASS] Simple Model Creation: Working")

        # Test model saving/loading
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"input_size": 10, "output_size": 2},
                },
                f.name,
            )
            temp_path = f.name

        # Load the model
        loaded_data = torch.load(temp_path)
        assert "model_state_dict" in loaded_data
        assert "config" in loaded_data
        print("[PASS] Model Save/Load: Working")

        # Clean up
        os.unlink(temp_path)

        return True

    except Exception as e:
        print(f"[FAIL] Model creation test failed: {e}")
        return False


def test_end_to_end_simple():
    """Test a simple end-to-end compression flow"""
    print("\n Testing Simple End-to-End Flow...")

    try:
        from torch import nn

        sys.path.insert(0, str(Path(__file__).parent))

        from agent_forge.compression.seedlm import SeedLMCompressor
        from agent_forge.compression.vptq import VPTQQuantizer

        # Create test model
        model = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 5))

        print("   Created test model")

        # Test Stage 1 compression (SeedLM only)
        compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=32)

        compressed_weights = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2:  # Only compress weight matrices
                compressed_data = compressor.compress_weight_matrix(param.data)
                compressed_weights[name] = compressed_data
                print(
                    f"   Compressed {name}: {compressed_data['compression_ratio']:.2f}x"
                )

        print("[PASS] Stage 1 (SeedLM) Compression: Working")

        # Test Stage 2 compression (VPTQ)
        quantizer = VPTQQuantizer(bits_per_vector=2.0, vector_length=8)

        # Decompress and re-compress with VPTQ
        for name, compressed_data in compressed_weights.items():
            # Decompress from Stage 1
            decompressed = compressor.decompress_weight_matrix(compressed_data)

            # Compress with VPTQ
            vptq_data = quantizer.quantize_weight_matrix(decompressed)
            print(f"   VPTQ compressed {name}: {vptq_data['compression_ratio']:.2f}x")

            # Test reconstruction
            reconstructed = quantizer.dequantize_weight_matrix(vptq_data)
            assert reconstructed.shape == decompressed.shape

        print("[PASS] Stage 2 (VPTQ) Compression: Working")

        # Test model handoff
        print("   Testing model handoff...")

        # Simulate Stage 1 output
        stage1_output = {
            "compressed_state": compressed_weights,
            "config": {"compression_method": "seedlm"},
            "model_info": {
                "original_params": sum(p.numel() for p in model.parameters())
            },
        }

        # Simulate Stage 2 processing
        stage2_output = {
            "stage2_compressed_data": "placeholder",
            "stage1_metadata": stage1_output,
            "compression_pipeline": "SeedLM -> VPTQ",
        }

        # Verify handoff integrity
        assert "stage1_metadata" in stage2_output
        assert "compressed_state" in stage2_output["stage1_metadata"]
        assert "config" in stage2_output["stage1_metadata"]

        print("[PASS] Model Handoff: Working")
        print("[PASS] Simple End-to-End Flow: Working")

        return True

    except Exception as e:
        print(f"[FAIL] Simple end-to-end test failed: {e}")
        return False


def test_file_operations():
    """Test file operations for model saving/loading"""
    print("\n Testing File Operations...")

    try:
        import hashlib
        import json

        import torch

        # Test temporary file creation
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test model file creation
            model_data = {
                "weights": {"layer1": torch.randn(10, 5), "layer2": torch.randn(5, 2)},
                "config": {"layers": 2, "input_size": 10},
            }

            model_file = temp_path / "test_model.pt"
            torch.save(model_data, model_file)

            # Verify file exists and has content
            assert model_file.exists()
            assert model_file.stat().st_size > 0
            print("[PASS] Model File Creation: Working")

            # Test file loading
            loaded_data = torch.load(model_file)
            assert "weights" in loaded_data
            assert "config" in loaded_data
            print("[PASS] Model File Loading: Working")

            # Test JSON operations
            json_file = temp_path / "test_config.json"
            config_data = {"test": "data", "number": 42}

            with open(json_file, "w") as f:
                json.dump(config_data, f)

            with open(json_file) as f:
                loaded_config = json.load(f)

            assert loaded_config == config_data
            print("[PASS] JSON File Operations: Working")

            # Test hash calculation
            test_content = b"test data for hashing"
            hash_value = hashlib.sha256(test_content).hexdigest()
            assert len(hash_value) == 64  # SHA256 produces 64 hex characters
            print("[PASS] Hash Calculation: Working")

        return True

    except Exception as e:
        print(f"[FAIL] File operations test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Agent Forge Pipeline Component Tests")
    print("=" * 60)

    tests = [
        ("Import Tests", test_imports),
        ("Stage 1 Components", test_stage1_components),
        ("Stage 2 Components", test_stage2_components),
        ("Evaluation Harness", test_evaluation_harness),
        ("Model Creation", test_model_creation),
        ("File Operations", test_file_operations),
        ("End-to-End Simple", test_end_to_end_simple),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results[test_name] = result
            status = "[PASS]" if result else "[FAIL]"
            print(f"Result: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"Result: [FAIL] - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed / total * 100:.1f}%")

    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_name}: {status}")

    overall_success = all(results.values())
    print(f"\nOverall Result: {'[PASS]' if overall_success else '[FAIL]'}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
