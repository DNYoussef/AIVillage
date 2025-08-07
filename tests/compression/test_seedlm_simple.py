#!/usr/bin/env python3
"""
Simple test script for SeedLM implementation
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

    # Test with different tensor sizes
    test_cases = [
        torch.randn(128, 256),  # Small
        torch.randn(512, 768),  # Medium
        torch.randn(32, 32),  # Square
    ]

    for i, weight in enumerate(test_cases):
        print(f"\nTest case {i + 1}: {weight.shape}")

        # Test basic encode/decode
        compressed = encoder.encode(weight, compression_level=0.5)
        reconstructed = encoder.decode(compressed)

        # Verify shape preservation
        assert (
            reconstructed.shape == weight.shape
        ), f"Shape mismatch: {reconstructed.shape} vs {weight.shape}"

        # Check compression ratio
        compression_ratio = compressed["data"].get("compression_ratio", 0)
        print(f"  Compression ratio: {compression_ratio:.2f}x")

        # Check reconstruction error
        max_error = torch.max(torch.abs(reconstructed - weight)).item()
        relative_error = (
            torch.norm(reconstructed - weight) / torch.norm(weight)
        ).item()

        print(f"  Max error: {max_error:.6f}")
        print(f"  Relative error: {relative_error:.6f}")

        # Basic error thresholds
        assert max_error < 10.0, f"Max error too high: {max_error}"
        assert relative_error < 1.0, f"Relative error too high: {relative_error}"

        print(f"  ✓ Test case {i + 1} passed")

    print("\n✓ All basic functionality tests passed!")


def test_progressive_encoding():
    print("\nTesting progressive encoding...")

    config = SeedLMConfig()
    encoder = ProgressiveSeedLMEncoder(config)

    weight = torch.randn(256, 512)

    # Test progressive encoding
    progressive_data = encoder.encode_progressive(weight)

    assert "base_layer" in progressive_data
    assert "enhancement_layers" in progressive_data
    assert len(progressive_data["enhancement_layers"]) == 3

    # Test progressive decoding
    for num_layers in [1, 2, 3, 4]:
        reconstructed = encoder.decode_progressive(progressive_data, num_layers)
        assert reconstructed.shape == weight.shape

        relative_error = (
            torch.norm(reconstructed - weight) / torch.norm(weight)
        ).item()
        print(f"  {num_layers} layers - Relative error: {relative_error:.6f}")

    print("  ✓ Progressive encoding tests passed!")


def test_error_handling():
    print("\nTesting error handling...")

    config = SeedLMConfig()
    encoder = ProgressiveSeedLMEncoder(config)

    # Test invalid inputs
    try:
        encoder.encode("not a tensor")
        msg = "Should have raised SeedLMCompressionError"
        raise AssertionError(msg)
    except SeedLMCompressionError:
        print("  ✓ Invalid input error handling works")

    # Test invalid compression level
    try:
        encoder.encode(torch.randn(10, 10), compression_level=1.5)
        msg = "Should have raised ValueError"
        raise AssertionError(msg)
    except ValueError:
        print("  ✓ Invalid compression level error handling works")

    # Test empty tensor
    empty_tensor = torch.empty(0, 0)
    compressed = encoder.encode(empty_tensor)
    reconstructed = encoder.decode(compressed)
    assert reconstructed.shape == empty_tensor.shape
    print("  ✓ Empty tensor handling works")


def test_adaptive_block_sizing():
    print("\nTesting adaptive block sizing...")

    analyzer = AdaptiveBlockAnalyzer()

    # High variance -> small blocks
    high_var = torch.randn(100, 100) * 10
    block_size = analyzer.determine_block_size(high_var)
    print(f"  High variance tensor -> block size: {block_size}")
    assert block_size <= 8, f"Expected small block size, got {block_size}"

    # Low variance -> large blocks
    low_var = torch.ones(100, 100) + torch.randn(100, 100) * 0.01
    block_size = analyzer.determine_block_size(low_var)
    print(f"  Low variance tensor -> block size: {block_size}")
    assert block_size >= 16, f"Expected large block size, got {block_size}"

    print("  ✓ Adaptive block sizing works")


def test_multi_scale_lfsr():
    print("\nTesting multi-scale LFSR generation...")

    generator = MultiScaleLFSRGenerator(
        seeds=[12345, 67890], tap_configs=[[16, 14, 13, 11], [16, 15, 13, 4]]
    )

    # Test basis generation
    for scale in [4, 8, 16]:
        basis = generator.generate_basis(scale, scale)
        assert basis.shape == (scale, scale)

        # Check orthogonality (approximately)
        if scale > 1:
            gram = torch.mm(basis.T, basis)
            identity_like = torch.eye(scale)
            max_off_diag = torch.max(torch.abs(gram - identity_like)).item()
            print(f"  Scale {scale} - Max off-diagonal: {max_off_diag:.3f}")

    print("  ✓ Multi-scale LFSR generation works")


if __name__ == "__main__":
    print("=== SeedLM Implementation Test Suite ===")

    try:
        test_basic_functionality()
        test_progressive_encoding()
        test_error_handling()
        test_adaptive_block_sizing()
        test_multi_scale_lfsr()

        print("\n🎉 All tests passed! SeedLM implementation is working correctly.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
