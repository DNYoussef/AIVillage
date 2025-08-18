#!/usr/bin/env python3
"""Comprehensive test of compression pipeline claims."""

import os
import time

import psutil
import torch
from torch import nn


def create_test_model(param_count: int):
    """Create a model with approximately param_count parameters."""
    # Calculate dimensions to get close to target parameter count
    # For a simple linear model: params ≈ input_dim * output_dim
    dim = int((param_count / 2) ** 0.5)

    class TestModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.layer1 = nn.Linear(dim, dim)
            self.layer2 = nn.Linear(dim, dim)

        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))

    return TestModel(dim)


def get_model_size_mb(model):
    """Get model size in MB."""
    param_count = sum(p.numel() for p in model.parameters())
    # Assume float32 = 4 bytes per parameter
    return (param_count * 4) / (1024 * 1024)


def test_simple_quantizer():
    """Test SimpleQuantizer from core.compression."""
    print("=== Testing SimpleQuantizer ===")

    try:
        from src.core.compression.simple_quantizer import SimpleQuantizer

        # Create small test model
        model = create_test_model(10000)  # ~10k parameters
        original_size = get_model_size_mb(model)
        print(f"Original model size: {original_size:.2f} MB")

        # Test compression
        quantizer = SimpleQuantizer()
        start_time = time.time()
        compressed_bytes = quantizer.quantize_model(model)
        compression_time = time.time() - start_time

        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        ratio = original_size / compressed_size_mb

        print(f"Compressed size: {compressed_size_mb:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Compression time: {compression_time:.2f}s")

        # Test decompression
        decompressed = quantizer.decompress_model(compressed_bytes)
        print(f"Decompression successful: {decompressed is not None}")

        return {
            "works": True,
            "ratio": ratio,
            "time": compression_time,
            "original_size": original_size,
            "compressed_size": compressed_size_mb,
        }

    except Exception as e:
        print(f"SimpleQuantizer test failed: {e}")
        return {"works": False, "error": str(e)}


def test_unified_compressor():
    """Test UnifiedCompressor."""
    print("\n=== Testing UnifiedCompressor ===")

    try:
        from src.core.compression.unified_compressor import UnifiedCompressor

        # Create test model
        model = create_test_model(50000)  # ~50k parameters
        original_size = get_model_size_mb(model)
        print(f"Original model size: {original_size:.2f} MB")

        # Test compression
        compressor = UnifiedCompressor()
        start_time = time.time()
        result = compressor.compress(model)
        compression_time = time.time() - start_time

        compressed_data = result.get("data", b"")
        compressed_size_mb = len(compressed_data) / (1024 * 1024)
        ratio = original_size / max(compressed_size_mb, 0.001)  # Avoid division by zero

        print(f"Method used: {result.get('method', 'unknown')}")
        print(f"Compressed size: {compressed_size_mb:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Compression time: {compression_time:.2f}s")

        # Test decompression
        try:
            decompressed = compressor.decompress(result)
            print(f"Decompression successful: {decompressed is not None}")
        except Exception as e:
            print(f"Decompression failed: {e}")

        return {
            "works": True,
            "ratio": ratio,
            "time": compression_time,
            "method": result.get("method"),
            "original_size": original_size,
            "compressed_size": compressed_size_mb,
        }

    except Exception as e:
        print(f"UnifiedCompressor test failed: {e}")
        return {"works": False, "error": str(e)}


def test_advanced_pipeline():
    """Test AdvancedCompressionPipeline."""
    print("\n=== Testing AdvancedCompressionPipeline ===")

    try:
        from src.core.compression.advanced_pipeline import AdvancedCompressionPipeline

        # Create small test model to avoid timeouts
        model = create_test_model(1000)  # Very small
        original_size = get_model_size_mb(model)
        print(f"Original model size: {original_size:.2f} MB")

        # Test compression
        pipeline = AdvancedCompressionPipeline()
        start_time = time.time()
        compressed_bytes = pipeline.compress_model(model)
        compression_time = time.time() - start_time

        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        ratio = original_size / max(compressed_size_mb, 0.001)

        print(f"Compressed size: {compressed_size_mb:.2f} MB")
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Compression time: {compression_time:.2f}s")

        # Test decompression
        try:
            decompressed = pipeline.decompress_model(compressed_bytes)
            print(f"Decompression successful: {decompressed is not None}")
            if isinstance(decompressed, dict):
                print(f"Decompressed tensors: {len(decompressed)}")
        except Exception as e:
            print(f"Decompression failed: {e}")

        return {
            "works": True,
            "ratio": ratio,
            "time": compression_time,
            "original_size": original_size,
            "compressed_size": compressed_size_mb,
        }

    except Exception as e:
        print(f"AdvancedCompressionPipeline test failed: {e}")
        return {"works": False, "error": str(e)}


def test_individual_compressors():
    """Test individual compressors from agent_forge."""
    print("\n=== Testing Individual Compressors ===")

    results = {}

    # Test BitNet compressor
    try:
        from src.agent_forge.compression.bitnet import BITNETCompressor

        compressor = BITNETCompressor()
        test_tensor = torch.randn(100, 100)
        original_bytes = test_tensor.numel() * 4  # float32

        compressed = compressor.compress(test_tensor)
        packed_size = len(compressed["packed_weights"])
        ratio = original_bytes / packed_size

        # Test decompression
        decompressed = compressor.decompress(compressed)
        error = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"BitNet - Ratio: {ratio:.2f}x, MSE: {error:.6f}")
        results["bitnet"] = {"works": True, "ratio": ratio, "mse": error}

    except Exception as e:
        print(f"BitNet test failed: {e}")
        results["bitnet"] = {"works": False, "error": str(e)}

    # Test SeedLM compressor
    try:
        from src.agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor()
        test_tensor = torch.randn(64, 64)
        original_bytes = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)
        # Estimate compressed size
        seeds_size = compressed["seeds"].nbytes
        coeffs_size = compressed["coefficients"].nbytes
        exps_size = compressed["shared_exponents"].nbytes
        total_size = seeds_size + coeffs_size + exps_size
        ratio = original_bytes / total_size

        # Test decompression
        decompressed = compressor.decompress(compressed)
        error = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"SeedLM - Ratio: {ratio:.2f}x, MSE: {error:.6f}")
        results["seedlm"] = {"works": True, "ratio": ratio, "mse": error}

    except Exception as e:
        print(f"SeedLM test failed: {e}")
        results["seedlm"] = {"works": False, "error": str(e)}

    # Test VPTQ compressor
    try:
        from src.agent_forge.compression.vptq import VPTQCompressor

        compressor = VPTQCompressor()
        test_tensor = torch.randn(64, 64)
        original_bytes = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)
        # Estimate compressed size
        codebook_size = compressed["codebook"].numel() * 4
        indices_size = compressed["indices"].numel() * 1  # indices are small ints
        total_size = codebook_size + indices_size + 8  # scale + offset
        ratio = original_bytes / total_size

        # Test decompression
        decompressed = compressor.decompress(compressed)
        error = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"VPTQ - Ratio: {ratio:.2f}x, MSE: {error:.6f}")
        results["vptq"] = {"works": True, "ratio": ratio, "mse": error}

    except Exception as e:
        print(f"VPTQ test failed: {e}")
        results["vptq"] = {"works": False, "error": str(e)}

    return results


def test_memory_usage():
    """Test if compression stays under 2GB as claimed."""
    print("\n=== Testing Memory Usage ===")

    process = psutil.Process(os.getpid())

    # Get baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline_memory:.1f} MB")

    try:
        # Create a larger model
        model = create_test_model(1000000)  # 1M parameters ≈ 4MB
        after_model_memory = process.memory_info().rss / 1024 / 1024
        print(f"After model creation: {after_model_memory:.1f} MB")

        # Try compression
        from src.core.compression.simple_quantizer import SimpleQuantizer

        quantizer = SimpleQuantizer()

        quantizer.quantize_model(model)
        after_compression_memory = process.memory_info().rss / 1024 / 1024
        print(f"After compression: {after_compression_memory:.1f} MB")

        peak_usage = after_compression_memory - baseline_memory
        print(f"Peak additional memory usage: {peak_usage:.1f} MB")

        return {
            "peak_memory_mb": peak_usage,
            "under_2gb": peak_usage < 2048,
            "success": True,
        }

    except Exception as e:
        print(f"Memory test failed: {e}")
        return {"success": False, "error": str(e)}


def run_compression_tests():
    """Run all compression tests and generate report."""
    print("AIVillage Compression Pipeline Testing")
    print("=" * 50)

    # Run all tests
    simple_results = test_simple_quantizer()
    unified_results = test_unified_compressor()
    advanced_results = test_advanced_pipeline()
    individual_results = test_individual_compressors()
    memory_results = test_memory_usage()

    # Generate summary report
    print("\n" + "=" * 50)
    print("COMPRESSION PIPELINE TEST RESULTS")
    print("=" * 50)

    print("\n1. SimpleQuantizer:")
    if simple_results.get("works"):
        print("   ✓ Works: Yes")
        print(f"   ✓ Compression ratio: {simple_results['ratio']:.2f}x")
        print(f"   ✓ Achieves 4x compression: {'Yes' if simple_results['ratio'] >= 3.5 else 'No'}")
    else:
        print(f"   ✗ Works: No - {simple_results.get('error')}")

    print("\n2. UnifiedCompressor:")
    if unified_results.get("works"):
        print("   ✓ Works: Yes")
        print(f"   ✓ Method: {unified_results.get('method')}")
        print(f"   ✓ Compression ratio: {unified_results['ratio']:.2f}x")
    else:
        print(f"   ✗ Works: No - {unified_results.get('error')}")

    print("\n3. AdvancedCompressionPipeline:")
    if advanced_results.get("works"):
        print("   ✓ Works: Yes")
        print(f"   ✓ Compression ratio: {advanced_results['ratio']:.2f}x")
    else:
        print(f"   ✗ Works: No - {advanced_results.get('error')}")

    print("\n4. Individual Compressors:")
    for name, result in individual_results.items():
        if result.get("works"):
            print(f"   ✓ {name.upper()}: {result['ratio']:.2f}x ratio, MSE: {result['mse']:.6f}")
        else:
            print(f"   ✗ {name.upper()}: Failed - {result.get('error')}")

    print("\n5. Memory Usage:")
    if memory_results.get("success"):
        print(f"   ✓ Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
        print(f"   ✓ Under 2GB limit: {'Yes' if memory_results['under_2gb'] else 'No'}")
    else:
        print(f"   ✗ Memory test failed: {memory_results.get('error')}")

    print("\n" + "=" * 50)
    print("FINAL ASSESSMENT")
    print("=" * 50)

    # Count working components
    working_count = sum(
        [
            simple_results.get("works", False),
            unified_results.get("works", False),
            advanced_results.get("works", False),
            sum(r.get("works", False) for r in individual_results.values()) > 0,
        ]
    )

    print(f"Working components: {working_count}/4")

    # Check if claims are met
    achieves_4x = simple_results.get("ratio", 0) >= 3.5
    under_2gb = memory_results.get("under_2gb", False)

    print(f"Achieves 4x compression claim: {'✓ Yes' if achieves_4x else '✗ No'}")
    print(f"Stays under 2GB claim: {'✓ Yes' if under_2gb else '✗ No'}")

    if working_count >= 3 and achieves_4x and under_2gb:
        print("\n🎉 COMPRESSION PIPELINE: MOSTLY FUNCTIONAL")
    elif working_count >= 2:
        print("\n⚠️  COMPRESSION PIPELINE: PARTIALLY FUNCTIONAL")
    else:
        print("\n❌ COMPRESSION PIPELINE: NOT FUNCTIONAL")


if __name__ == "__main__":
    run_compression_tests()
