#!/usr/bin/env python3
"""Audit the compression claims with detailed byte-level analysis"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_forge.compression.bitnet import BITNETCompressor
from agent_forge.compression.seedlm import SEEDLMCompressor
from agent_forge.compression.vptq import VPTQCompressor


def create_test_tensor(size=(1000, 1000), seed=42):
    """Create reproducible test tensor"""
    torch.manual_seed(seed)
    return torch.randn(size)


def calculate_real_compression(original_tensor, compressed_data):
    """Calculate actual compression ratio with all overhead"""
    # Original size in bytes (float32 = 4 bytes)
    original_bytes = original_tensor.numel() * 4

    # Calculate compressed size including ALL data
    if isinstance(compressed_data, dict):
        compressed_bytes = 0
        for key, value in compressed_data.items():
            if isinstance(value, torch.Tensor):
                compressed_bytes += value.numel() * value.element_size()
            elif isinstance(value, np.ndarray):
                compressed_bytes += value.nbytes
            elif isinstance(value, bytes):
                compressed_bytes += len(value)
            elif isinstance(value, (list, tuple)):
                compressed_bytes += sys.getsizeof(value)
            elif isinstance(value, (int, float)):
                compressed_bytes += 8  # Conservative estimate
            else:
                compressed_bytes += sys.getsizeof(value)
    else:
        compressed_bytes = len(compressed_data)

    ratio = original_bytes / compressed_bytes
    return original_bytes, compressed_bytes, ratio


def audit_bitnet():
    """Audit BitNet compression claims"""
    print("\n=== BITNET COMPRESSION AUDIT ===")
    compressor = BITNETCompressor()

    test_sizes = [(100, 100), (1000, 1000), (2048, 2048)]

    for size in test_sizes:
        weights = create_test_tensor(size)
        compressed = compressor.compress(weights)

        # Detailed analysis
        orig_bytes, comp_bytes, ratio = calculate_real_compression(weights, compressed)

        print(f"\nTensor size: {size}")
        print(f"Original: {orig_bytes:,} bytes")
        print(f"Compressed: {comp_bytes:,} bytes")
        print(f"  - packed_weights: {len(compressed['packed_weights'])} bytes")
        print("  - scale: 4 bytes")
        print(f"  - metadata: {comp_bytes - len(compressed['packed_weights']) - 4} bytes")
        print(f"Actual ratio: {ratio:.2f}x")

        # Theoretical ratio for ternary (2 bits per weight)
        theoretical_ratio = 32 / 2  # 32-bit float to 2-bit ternary
        print(f"Theoretical max: {theoretical_ratio}x")
        print(f"Efficiency: {ratio/theoretical_ratio*100:.1f}%")

        # Verify decompression
        decompressed = compressor.decompress(compressed)
        unique_vals = torch.unique(decompressed / compressed["scale"])
        print(f"Unique values after decompression: {len(unique_vals)}")
        assert len(unique_vals) <= 3, "Not ternary!"

    return ratio


def audit_seedlm():
    """Audit SeedLM compression claims"""
    print("\n=== SEEDLM COMPRESSION AUDIT ===")

    for bits in [3, 4]:
        print(f"\n--- SeedLM with {bits} bits per weight ---")
        compressor = SEEDLMCompressor(bits_per_weight=bits)

        weights = create_test_tensor((1024, 1024))
        compressed = compressor.compress(weights)

        orig_bytes, comp_bytes, ratio = calculate_real_compression(weights, compressed)

        print(f"Original: {orig_bytes:,} bytes")
        print(f"Compressed: {comp_bytes:,} bytes")
        print(f"  - seeds: {compressed['seeds'].nbytes} bytes")
        print(f"  - coefficients: {compressed['coefficients'].nbytes} bytes")
        print(f"  - exponents: {compressed['shared_exponents'].nbytes} bytes")
        print(
            f"  - metadata: {comp_bytes - compressed['seeds'].nbytes - compressed['coefficients'].nbytes - compressed['shared_exponents'].nbytes} bytes"
        )
        print(f"Actual ratio: {ratio:.2f}x")

        # Check actual bits per weight
        total_bits = (
            len(compressed["seeds"]) * compressor.K
            + compressed["coefficients"].size * 4
            + len(compressed["shared_exponents"]) * 4
        )
        actual_bpw = total_bits / weights.numel()
        print(f"Target bits per weight: {bits}")
        print(f"Actual bits per weight: {actual_bpw:.2f}")

        # Paper claims
        theoretical_ratio = 32 / bits
        print(f"Theoretical max: {theoretical_ratio:.1f}x")
        print("Paper claims: 8x for 4-bit")
        print(f"Efficiency vs theoretical: {ratio/theoretical_ratio*100:.1f}%")

    return ratio


def audit_vptq():
    """Audit VPTQ compression claims"""
    print("\n=== VPTQ COMPRESSION AUDIT ===")

    for bits in [2, 3, 4]:
        print(f"\n--- VPTQ with {bits} bits ---")
        compressor = VPTQCompressor(bits=bits, codebook_size=2**bits)

        weights = create_test_tensor((1024, 1024))
        compressed = compressor.compress(weights)

        orig_bytes, comp_bytes, ratio = calculate_real_compression(weights, compressed)

        print(f"Original: {orig_bytes:,} bytes")
        print(f"Compressed: {comp_bytes:,} bytes")
        print(f"  - codebook: {compressed['codebook'].numel() * 4} bytes")
        print(f"  - indices: {len(compressed['indices']) * bits // 8} bytes (approx)")
        print("  - scale/offset: 8 bytes")
        print(f"Actual ratio: {ratio:.2f}x")

        # Analysis
        print(f"Codebook entries: {compressed['codebook'].shape[0]}")
        print(f"Vector dimension: {compressed['vector_dim']}")
        print(f"Number of vectors: {len(compressed['indices'])}")

        # Theoretical
        theoretical_ratio = 32 / bits
        print(f"Theoretical max: {theoretical_ratio}x")
        print(f"Efficiency: {ratio/theoretical_ratio*100:.1f}%")

    return ratio


def audit_pipeline_combination():
    """Audit how stages combine"""
    print("\n=== PIPELINE COMBINATION AUDIT ===")

    weights = create_test_tensor((512, 512))
    original_bytes = weights.numel() * 4

    print(f"Original tensor: {original_bytes:,} bytes")

    # Stage 1: BitNet
    bitnet = BITNETCompressor()
    stage1_compressed = bitnet.compress(weights)
    stage1_bytes = len(stage1_compressed["packed_weights"]) + 8  # +metadata
    stage1_ratio = original_bytes / stage1_bytes
    print(f"\nAfter BitNet: {stage1_bytes:,} bytes ({stage1_ratio:.1f}x)")

    # Decompress for next stage
    stage1_output = bitnet.decompress(stage1_compressed)

    # Stage 2: SeedLM
    seedlm = SEEDLMCompressor(bits_per_weight=4)
    stage2_compressed = seedlm.compress(stage1_output)
    _, stage2_bytes, _ = calculate_real_compression(stage1_output, stage2_compressed)
    stage2_ratio = stage1_bytes / stage2_bytes
    cumulative_ratio = original_bytes / stage2_bytes
    print(f"After SeedLM: {stage2_bytes:,} bytes ({stage2_ratio:.1f}x additional, {cumulative_ratio:.1f}x total)")

    # Stage 3: VPTQ
    stage2_output = seedlm.decompress(stage2_compressed)
    vptq = VPTQCompressor(bits=2)
    stage3_compressed = vptq.compress(stage2_output)
    _, stage3_bytes, _ = calculate_real_compression(stage2_output, stage3_compressed)
    stage3_ratio = stage2_bytes / stage3_bytes
    cumulative_ratio = original_bytes / stage3_bytes
    print(f"After VPTQ: {stage3_bytes:,} bytes ({stage3_ratio:.1f}x additional, {cumulative_ratio:.1f}x total)")

    # Check if multiplicative
    expected_ratio = stage1_ratio * stage2_ratio * stage3_ratio
    print(f"\nExpected multiplicative: {expected_ratio:.1f}x")
    print(f"Actual cumulative: {cumulative_ratio:.1f}x")
    print(f"Efficiency: {cumulative_ratio/expected_ratio*100:.1f}%")

    # Why might it not be multiplicative?
    print("\nPossible reasons for non-multiplicative compression:")
    print("1. Decompression/recompression introduces entropy")
    print("2. Already compressed data has less redundancy")
    print("3. Metadata overhead accumulates")
    print("4. Quantization errors compound")


def test_real_model_compression():
    """Test on actual neural network models"""
    print("\n=== REAL MODEL COMPRESSION TEST ===")

    from core.compression.advanced_pipeline import AdvancedCompressionPipeline

    # Create realistic model
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    total_params = sum(p.numel() for p in model.parameters())
    original_bytes = total_params * 4

    print(f"Model parameters: {total_params:,}")
    print(f"Original size: {original_bytes/1024/1024:.2f} MB")

    # Test pipeline
    pipeline = AdvancedCompressionPipeline()
    compressed = pipeline.compress_model(model)
    compressed_bytes = len(compressed)
    ratio = original_bytes / compressed_bytes

    print(f"Compressed size: {compressed_bytes/1024:.2f} KB")
    print(f"Compression ratio: {ratio:.1f}x")

    # Compare to Claude's claim
    print("\nClaude claimed: 20.8x")
    print(f"We achieved: {ratio:.1f}x")
    print(f"Difference: {abs(ratio - 20.8):.1f}x")


def main():
    """Run comprehensive audit"""
    print("=" * 60)
    print("COMPRESSION CLAIMS AUDIT")
    print("=" * 60)

    # Audit each stage
    bitnet_ratio = audit_bitnet()
    seedlm_ratio = audit_seedlm()
    vptq_ratio = audit_vptq()

    # Audit combination
    audit_pipeline_combination()

    # Test real models
    test_real_model_compression()

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print("Individual stages might achieve claimed ratios")
    print("But combination is NOT multiplicative due to:")
    print("- Decompression/recompression overhead")
    print("- Metadata accumulation")
    print("- Reduced redundancy in already compressed data")
    print("- Possible implementation inefficiencies")


if __name__ == "__main__":
    main()
    # Why might it not be multiplicative?
    print("\nPossible reasons for non-multiplicative compression:")
    print("1. Decompression/recompression introduces entropy")
    print("2. Already compressed data has less redundancy")
    print("3. Metadata overhead accumulates")
    print("4. Quantization errors compound")


def test_real_model_compression():
    """Test on actual neural network models"""
    print("\n=== REAL MODEL COMPRESSION TEST ===")

    from core.compression.advanced_pipeline import AdvancedCompressionPipeline

    # Create realistic model
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    total_params = sum(p.numel() for p in model.parameters())
    original_bytes = total_params * 4

    print(f"Model parameters: {total_params:,}")
    print(f"Original size: {original_bytes/1024/1024:.2f} MB")

    # Test pipeline
    pipeline = AdvancedCompressionPipeline()
    compressed = pipeline.compress_model(model)
    compressed_bytes = len(compressed)
    ratio = original_bytes / compressed_bytes

    print(f"Compressed size: {compressed_bytes/1024:.2f} KB")
    print(f"Compression ratio: {ratio:.1f}x")

    # Compare to Claude's claim
    print("\nClaude claimed: 20.8x")
    print(f"We achieved: {ratio:.1f}x")
    print(f"Difference: {abs(ratio - 20.8):.1f}x")


def main():
    """Run comprehensive audit"""
    print("=" * 60)
    print("COMPRESSION CLAIMS AUDIT")
    print("=" * 60)

    # Audit each stage
    bitnet_ratio = audit_bitnet()
    seedlm_ratio = audit_seedlm()
    vptq_ratio = audit_vptq()

    # Audit combination
    audit_pipeline_combination()

    # Test real models
    test_real_model_compression()

    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print("Individual stages might achieve claimed ratios")
    print("But combination is NOT multiplicative due to:")
    print("- Decompression/recompression overhead")
    print("- Metadata accumulation")
    print("- Reduced redundancy in already compressed data")
    print("- Possible implementation inefficiencies")


if __name__ == "__main__":
    main()
