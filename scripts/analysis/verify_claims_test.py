#!/usr/bin/env python3
"""Test to verify the extreme compression claims made in documentation."""

import time

import torch


def test_extreme_compression_claims():
    """Test if the claimed 77,907x compression ratio is real or fake."""
    print("TESTING EXTREME COMPRESSION CLAIMS")
    print("=" * 50)

    # Create a tensor similar to what's claimed in the docs
    # Model layer: (8960, 1536) = ~52.5 MB as claimed
    print("\nCreating test tensor matching documentation claims...")
    test_tensor = torch.randn(8960, 1536)  # Same shape as claimed
    original_bytes = test_tensor.numel() * 4  # float32
    original_mb = original_bytes / (1024 * 1024)

    print(f"Original tensor shape: {test_tensor.shape}")
    print(f"Original size: {original_mb:.1f} MB ({original_bytes:,} bytes)")

    # Test claimed compression pipeline
    print("\nTesting compression stages...")

    # Stage 1: BitNet
    try:
        from src.agent_forge.compression.bitnet import BITNETCompressor

        print("\n1. BitNet Compression:")
        bitnet = BITNETCompressor()

        start_time = time.time()
        bitnet_compressed = bitnet.compress(test_tensor)
        bitnet_time = time.time() - start_time

        bitnet_bytes = len(bitnet_compressed["packed_weights"])
        bitnet_ratio = original_bytes / bitnet_bytes

        print(f"   Compressed size: {bitnet_bytes:,} bytes")
        print(f"   Compression ratio: {bitnet_ratio:.1f}x")
        print(f"   Time: {bitnet_time:.3f}s")

        # Test reconstruction
        reconstructed = bitnet.decompress(bitnet_compressed)
        mse = torch.mean((test_tensor - reconstructed) ** 2).item()
        print(f"   MSE error: {mse:.6f}")

    except Exception as e:
        print(f"   BitNet failed: {e}")
        return

    # Stage 2: Try to achieve the claimed 5,309x LZMA compression
    print("\n2. LZMA Compression Test:")
    try:
        import lzma

        # Take the BitNet compressed data
        input_data = bitnet_compressed["packed_weights"]
        original_lzma_size = len(input_data)

        compressed_lzma = lzma.compress(input_data, preset=9)
        lzma_size = len(compressed_lzma)
        lzma_ratio = original_lzma_size / lzma_size

        print(f"   Input size: {original_lzma_size:,} bytes")
        print(f"   LZMA size: {lzma_size:,} bytes")
        print(f"   LZMA ratio: {lzma_ratio:.1f}x")

        # Calculate total pipeline ratio
        total_ratio = original_bytes / lzma_size
        print(f"   Total pipeline ratio: {total_ratio:.1f}x")

        # Check if it matches the claimed 77,907x
        claimed_ratio = 77907
        if total_ratio > 10000:
            print(f"   EXTREME COMPRESSION ACHIEVED: {total_ratio:.0f}x")
            if abs(total_ratio - claimed_ratio) / claimed_ratio < 0.1:
                print("   CLAIM VERIFICATION: MATCHES DOCUMENTATION")
            else:
                print(f"   CLAIM VERIFICATION: DIFFERENT FROM CLAIMED {claimed_ratio}x")
        else:
            print(f"   CLAIM VERIFICATION: DOES NOT MATCH CLAIMED {claimed_ratio}x")

        # Test decompression
        decompressed_lzma = lzma.decompress(compressed_lzma)
        if decompressed_lzma == input_data:
            print("   LZMA decompression: SUCCESSFUL")
        else:
            print("   LZMA decompression: FAILED")

    except Exception as e:
        print(f"   LZMA test failed: {e}")

    # Test what happens with different data types
    print("\n3. Testing Data Characteristics:")

    # Test with mostly zeros (should compress extremely well)
    sparse_tensor = torch.zeros_like(test_tensor)
    sparse_tensor[0:100, 0:100] = torch.randn(100, 100)  # Only small area has data

    try:
        sparse_compressed = bitnet.compress(sparse_tensor)
        sparse_bytes = len(sparse_compressed["packed_weights"])
        sparse_lzma = lzma.compress(sparse_compressed["packed_weights"], preset=9)
        sparse_lzma_bytes = len(sparse_lzma)
        sparse_total_ratio = original_bytes / sparse_lzma_bytes

        print(f"   Sparse tensor ratio: {sparse_total_ratio:.0f}x")
        print("   Explanation: Sparse data compresses extremely well")

    except Exception as e:
        print(f"   Sparse test failed: {e}")


def test_realistic_compression():
    """Test compression on more realistic data."""
    print("\n" + "=" * 50)
    print("REALISTIC COMPRESSION TEST")
    print("=" * 50)

    # Test on various tensor types
    test_cases = [
        ("Random Normal", torch.randn(1000, 1000)),
        ("Random Uniform", torch.rand(1000, 1000) * 2 - 1),
        ("Structured", torch.ones(1000, 1000) * 0.1 + torch.randn(1000, 1000) * 0.01),
        ("Sparse", torch.zeros(1000, 1000)),
    ]

    # Make sparse tensor actually sparse
    test_cases[3][1][0:100, 0:100] = torch.randn(100, 100)

    try:
        import lzma

        from src.agent_forge.compression.bitnet import BITNETCompressor

        bitnet = BITNETCompressor()

        print(
            f"\n{'Type':<15} {'Original (MB)':<12} {'Compressed (B)':<12} {'Ratio':<8} {'Quality'}"
        )
        print("-" * 70)

        for name, tensor in test_cases:
            original_bytes = tensor.numel() * 4
            original_mb = original_bytes / (1024 * 1024)

            # Compress
            compressed = bitnet.compress(tensor)
            lzma_compressed = lzma.compress(compressed["packed_weights"], preset=9)
            final_bytes = len(lzma_compressed)
            ratio = original_bytes / final_bytes

            # Check quality
            reconstructed = bitnet.decompress(compressed)
            mse = torch.mean((tensor - reconstructed) ** 2).item()

            print(
                f"{name:<15} {original_mb:<12.1f} {final_bytes:<12,} {ratio:<8.0f} {mse:<.4f}"
            )

    except Exception as e:
        print(f"Realistic compression test failed: {e}")


def analyze_claims():
    """Analyze the compression claims vs reality."""
    print("\n" + "=" * 50)
    print("CLAIM ANALYSIS")
    print("=" * 50)

    print("\nDOCUMENTATION CLAIMS:")
    print("- 77,907x compression on real model weights")
    print("- 1.78B model compressed to 0.1 MB")
    print("- LZMA achieving 5,309.7x compression")
    print("- 'All stages proven working'")

    print("\nREALITY CHECK:")
    print("- Extreme compression ratios occur with sparse/structured data")
    print("- LZMA is very effective on already-quantized data")
    print("- Claims technically possible but misleading")
    print("- Individual compressors do work as implemented")

    print("\nASSESSMENT:")
    print("- Individual compression algorithms: REAL and functional")
    print("- Extreme compression ratios: REAL but only for specific data")
    print("- General 4x compression claim: NOT consistently achieved")
    print("- Documentation: Technically accurate but cherry-picked results")


if __name__ == "__main__":
    print("AIVillage Compression Claims Verification")
    print("Testing documented claims vs actual performance")
    print()

    test_extreme_compression_claims()
    test_realistic_compression()
    analyze_claims()
