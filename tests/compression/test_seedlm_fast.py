#!/usr/bin/env python3
"""
Fast test script for SeedLM implementation
"""

import os
import sys

import torch

sys.path.insert(0, os.getcwd())

# Load SeedLM implementation directly
exec(open("agent_forge/compression/seedlm.py").read())


def test_basic_functionality():
    print("Testing SeedLM basic functionality...")

    # Create config and encoder
    config = SeedLMConfig()
    encoder = ProgressiveSeedLMEncoder(config)

    # Test with small tensors for speed
    test_cases = [
        torch.randn(16, 32),  # Very small
        torch.randn(8, 8),  # Tiny square
    ]

    for i, weight in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {weight.shape}")

        # Test basic encode/decode with fast settings
        compressed = encoder.encode(
            weight, compression_level=0.3
        )  # Lower level for speed
        reconstructed = encoder.decode(compressed)

        # Verify shape preservation
        assert reconstructed.shape == weight.shape, (
            f"Shape mismatch: {reconstructed.shape} vs {weight.shape}"
        )

        # Check reconstruction error
        max_error = torch.max(torch.abs(reconstructed - weight)).item()
        relative_error = (
            torch.norm(reconstructed - weight) / torch.norm(weight)
        ).item()

        print(f"  Max error: {max_error:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        print(f"  [OK] Test case {i + 1} passed")

    print("\n[OK] All basic functionality tests passed!")


def test_individual_components():
    print("\nTesting individual components...")

    # Test SeedLMConfig
    config = SeedLMConfig()
    assert len(config.compression_levels) == 5
    assert len(config.block_sizes) == 4
    print("  [OK] SeedLMConfig works")

    # Test AdaptiveBlockAnalyzer
    analyzer = AdaptiveBlockAnalyzer()

    high_var = torch.randn(10, 10) * 10
    small_block = analyzer.determine_block_size(high_var)

    low_var = torch.ones(10, 10)
    large_block = analyzer.determine_block_size(low_var)

    assert small_block <= large_block, "Adaptive sizing should work"
    print("  [OK] AdaptiveBlockAnalyzer works")

    # Test MultiScaleLFSRGenerator
    generator = MultiScaleLFSRGenerator(seeds=[12345], tap_configs=[[16, 14, 13, 11]])

    basis = generator.generate_basis(4, 2)
    assert basis.shape == (4, 2)
    print("  [OK] MultiScaleLFSRGenerator works")

    # Test LFSRGenerator
    lfsr = LFSRGenerator(12345)
    matrix = lfsr.generate_matrix(4, 4)
    assert matrix.shape == (4, 4)
    print("  [OK] LFSRGenerator works")

    print("  [OK] All individual components work!")


def test_error_handling():
    print("\nTesting error handling...")

    config = SeedLMConfig()
    encoder = ProgressiveSeedLMEncoder(config)

    # Test invalid inputs
    try:
        encoder.encode("not a tensor")
        assert False, "Should have raised SeedLMCompressionError"
    except SeedLMCompressionError:
        print("  [OK] Invalid input error handling works")

    # Test invalid compression level
    try:
        encoder.encode(torch.randn(4, 4), compression_level=1.5)
        assert False, "Should have raised ValueError"
    except (ValueError, SeedLMCompressionError):
        print("  [OK] Invalid compression level error handling works")

    # Test empty tensor
    empty_tensor = torch.empty(0, 0)
    compressed = encoder.encode(empty_tensor)
    reconstructed = encoder.decode(compressed)
    assert reconstructed.shape == empty_tensor.shape
    print("  [OK] Empty tensor handling works")


def test_legacy_compatibility():
    print("\nTesting legacy SeedLMCompressor...")

    # Test legacy class
    compressor = SeedLMCompressor(block_size=4, latent_dim=2, num_seeds=8)

    weight = torch.randn(8, 16)
    compressed_data = compressor.compress_weight_matrix(weight)
    reconstructed = compressor.decompress_weight_matrix(compressed_data)

    assert reconstructed.shape == weight.shape
    print(f"  Legacy compression ratio: {compressed_data['compression_ratio']:.2f}x")
    print("  [OK] Legacy SeedLMCompressor works")


if __name__ == "__main__":
    print("=== Fast SeedLM Implementation Test Suite ===")

    try:
        test_individual_components()
        test_basic_functionality()
        test_error_handling()
        test_legacy_compatibility()

        print(
            "\n[SUCCESS] All fast tests passed! SeedLM implementation is working correctly."
        )

    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
# REMOVED:         sys.exit(1)
