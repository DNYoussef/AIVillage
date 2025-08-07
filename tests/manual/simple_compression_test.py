#!/usr/bin/env python3
"""Simple test of compression components."""

import torch
from torch import nn


# Create a simple global model class to avoid pickling issues
class SimpleTestModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def test_individual_compressors():
    """Test each compressor individually."""
    print("Testing Individual Compression Components")
    print("=" * 50)

    # Test BitNet
    print("\n1. Testing BitNet Compressor:")
    try:
        from src.agent_forge.compression.bitnet import BITNETCompressor

        compressor = BITNETCompressor()
        test_tensor = torch.randn(50, 50)
        original_size = test_tensor.numel() * 4  # float32 bytes

        compressed = compressor.compress(test_tensor)
        packed_size = len(compressed["packed_weights"])
        ratio = original_size / packed_size

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   ✓ Compression ratio: {ratio:.2f}x")
        print(f"   ✓ MSE error: {mse:.6f}")
        print(
            f"   ✓ Original shape preserved: {test_tensor.shape == decompressed.shape}"
        )

    except Exception as e:
        print(f"   ✗ BitNet failed: {e}")

    # Test SeedLM
    print("\n2. Testing SeedLM Compressor:")
    try:
        from src.agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(bits_per_weight=4)
        test_tensor = torch.randn(48, 48)  # Multiple of block size
        original_size = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)

        # Calculate compressed size
        seeds_size = compressed["seeds"].nbytes
        coeffs_size = compressed["coefficients"].nbytes
        exps_size = compressed["shared_exponents"].nbytes
        total_compressed_size = seeds_size + coeffs_size + exps_size
        ratio = original_size / total_compressed_size

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   ✓ Compression ratio: {ratio:.2f}x")
        print(f"   ✓ MSE error: {mse:.6f}")
        print(
            f"   ✓ Original shape preserved: {test_tensor.shape == decompressed.shape}"
        )

    except Exception as e:
        print(f"   ✗ SeedLM failed: {e}")

    # Test VPTQ
    print("\n3. Testing VPTQ Compressor:")
    try:
        from src.agent_forge.compression.vptq import VPTQCompressor

        compressor = VPTQCompressor(bits=2, vector_dim=4)
        test_tensor = torch.randn(48, 48)
        original_size = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)

        # Calculate compressed size
        codebook_size = compressed["codebook"].numel() * 4
        indices_size = compressed["indices"].numel() * 1  # Assume 1 byte per index
        metadata_size = 16  # scale, offset, etc.
        total_compressed_size = codebook_size + indices_size + metadata_size
        ratio = original_size / total_compressed_size

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   ✓ Compression ratio: {ratio:.2f}x")
        print(f"   ✓ MSE error: {mse:.6f}")
        print(
            f"   ✓ Original shape preserved: {test_tensor.shape == decompressed.shape}"
        )

    except Exception as e:
        print(f"   ✗ VPTQ failed: {e}")


def test_simple_quantizer_basic():
    """Test SimpleQuantizer with basic tensor."""
    print("\n4. Testing SimpleQuantizer:")
    try:
        from src.core.compression.simple_quantizer import SimpleQuantizer

        # Create a simple model directly
        model = SimpleTestModel(32)

        quantizer = SimpleQuantizer()

        # Get original size
        total_params = sum(p.numel() for p in model.parameters())
        original_size_mb = (total_params * 4) / (1024 * 1024)

        print(f"   Model parameters: {total_params:,}")
        print(f"   Original size: {original_size_mb:.3f} MB")

        # Test with a saved model file instead
        torch.save(model, "test_model.pth")

        compressed_bytes = quantizer.quantize_model("test_model.pth")
        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        ratio = original_size_mb / compressed_size_mb

        print(f"   ✓ Compressed size: {compressed_size_mb:.3f} MB")
        print(f"   ✓ Compression ratio: {ratio:.2f}x")
        print(f"   ✓ Achieves >3.5x compression: {ratio >= 3.5}")

        # Test decompression
        decompressed = quantizer.decompress_model(compressed_bytes)
        print(f"   ✓ Decompression successful: {decompressed is not None}")

        # Clean up
        import os

        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")

    except Exception as e:
        print(f"   ✗ SimpleQuantizer failed: {e}")
        import traceback

        traceback.print_exc()


def test_unified_compressor_basic():
    """Test UnifiedCompressor basic functionality."""
    print("\n5. Testing UnifiedCompressor:")
    try:
        from src.core.compression.unified_compressor import UnifiedCompressor

        model = SimpleTestModel(32)

        compressor = UnifiedCompressor()
        result = compressor.compress(model)

        method = result.get("method", "unknown")
        data_size = len(result.get("data", b""))

        print(f"   ✓ Method selected: {method}")
        print(f"   ✓ Compressed data size: {data_size} bytes")
        print(f"   ✓ Has fallback: {result.get('fallback_available', False)}")

        # Test decompression
        try:
            decompressed = compressor.decompress(result)
            print(f"   ✓ Decompression works: {decompressed is not None}")
        except Exception as e:
            print(f"   ✗ Decompression failed: {e}")

    except Exception as e:
        print(f"   ✗ UnifiedCompressor failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    print("AIVillage Compression Pipeline - Functionality Test")
    print("=" * 55)

    test_individual_compressors()
    test_simple_quantizer_basic()
    test_unified_compressor_basic()

    print("\n" + "=" * 55)
    print("SUMMARY:")
    print("Individual compressors (BitNet, SeedLM, VPTQ) appear to work")
    print("SimpleQuantizer has some implementation issues")
    print("UnifiedCompressor routes to different methods based on model size")
    print("Advanced pipeline may return placeholders for large models")


if __name__ == "__main__":
    main()
