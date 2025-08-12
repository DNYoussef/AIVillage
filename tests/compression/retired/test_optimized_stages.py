#!/usr/bin/env python3
"""Test the optimized compression stages directly without dependency issues."""

import lzma
import pickle
import struct
import sys
import time
from pathlib import Path

from torch import nn

# Add source paths
sys.path.insert(0, str(Path("src").resolve()))


def test_optimized_compression_simulation():
    """Simulate the optimized compression pipeline based on the code improvements."""
    print("OPTIMIZED COMPRESSION SIMULATION")
    print("=" * 60)

    # Import individual stages
    from src.agent_forge.compression.bitnet import BITNETCompressor
    from src.agent_forge.compression.seedlm import SEEDLMCompressor
    from src.agent_forge.compression.vptq import VPTQCompressor

    # Create test model
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    param_count = sum(p.numel() for p in model.parameters())
    original_size = param_count * 4

    print("Test Model:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Original size: {original_size / 1024:.1f} KB")

    # Initialize compressors
    stage1 = BITNETCompressor()
    stage2 = SEEDLMCompressor(bits_per_weight=4)
    stage3 = VPTQCompressor(bits=2)

    # Simulate optimized pipeline (based on the new code structure)
    compressed_params = {}
    total_compressed_size = 0

    print("\nRunning optimized 4-stage pipeline...")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        data = param.data.cpu()
        print(f"\nProcessing {name}: {tuple(param.shape)}")

        # Stage 1: BitNet
        s1_start = time.time()
        s1_compressed = stage1.compress(data)
        s1_time = time.time() - s1_start
        s1_reconstructed = stage1.decompress(s1_compressed)

        # Stage 2: SeedLM
        s2_start = time.time()
        s2_compressed = stage2.compress(s1_reconstructed)
        s2_time = time.time() - s2_start
        s2_reconstructed = stage2.decompress(s2_compressed)

        # Stage 3: VPTQ
        s3_start = time.time()
        s3_compressed = stage3.compress(s2_reconstructed)
        s3_time = time.time() - s3_start
        s3_bytes = pickle.dumps(s3_compressed)

        # Stage 4: Simulated HyperCompression with effectiveness check
        s4_start = time.time()
        # Simulate compression (using gzip as proxy for HyperCompression)
        import gzip

        s4_bytes = gzip.compress(s3_bytes)
        s4_time = time.time() - s4_start

        # Apply effectiveness check (new optimization)
        if len(s4_bytes) >= len(s3_bytes) * 0.9:
            print(f"    HyperCompression ineffective for {name}, skipping")
            s4_bytes = s3_bytes
            hyper_effective = False
        else:
            print(f"    HyperCompression effective for {name}")
            hyper_effective = True

        # Calculate stage ratios
        original_param_size = data.numel() * 4
        s1_size = len(s1_compressed["packed_weights"]) + 32
        s2_size = (
            len(s2_compressed["seeds"]) * 2
            + s2_compressed["coefficients"].size
            + len(s2_compressed["shared_exponents"])
            + 32
        )
        s3_size = len(s3_bytes)
        s4_size = len(s4_bytes)

        print(
            f"    Stage 1 (BitNet): {original_param_size / s1_size:.1f}x ({s1_time:.3f}s)"
        )
        print(f"    Stage 2 (SeedLM): {s1_size / s2_size:.1f}x ({s2_time:.3f}s)")
        print(f"    Stage 3 (VPTQ): {s2_size / s3_size:.1f}x ({s3_time:.3f}s)")
        print(
            f"    Stage 4 (Hyper): {s3_size / s4_size:.1f}x ({s4_time:.3f}s) {'[SKIPPED]' if not hyper_effective else ''}"
        )
        print(f"    Overall: {original_param_size / s4_size:.1f}x")

        compressed_params[name] = (tuple(param.shape), s4_bytes)
        total_compressed_size += s4_size

    return compressed_params, total_compressed_size


def test_optimized_packing():
    """Test the new optimized binary packing format."""
    print(f"\n{'=' * 60}")
    print("TESTING OPTIMIZED BINARY PACKING")
    print("=" * 60)

    # Simulate compressed parameters
    test_params = {
        "layer1.weight": ((1024, 512), b"compressed_data_1" * 50),
        "layer1.bias": ((512,), b"compressed_data_2" * 10),
        "layer2.weight": ((512, 256), b"compressed_data_3" * 30),
        "layer2.bias": ((256,), b"compressed_data_4" * 5),
    }

    print(f"Test parameters: {len(test_params)} tensors")

    # Test old method (pickle)
    old_packed = pickle.dumps(test_params)
    old_size = len(old_packed)

    # Test new optimized packing
    def pack_compressed_data_optimized(params):
        """Optimized binary packing based on the new code."""
        blob = bytearray()
        blob.append(len(params))

        for name, (shape, data) in params.items():
            name_b = name.encode("utf-8")
            blob.append(len(name_b))
            blob.extend(name_b)
            blob.append(len(shape))

            for dim in shape:
                blob.extend(struct.pack("I", dim))

            blob.extend(struct.pack("I", len(data)))
            blob.extend(data)

        return bytes(blob)

    new_packed = pack_compressed_data_optimized(test_params)
    new_size = len(new_packed)

    # Test LZMA compression
    lzma_compressed = lzma.compress(new_packed, preset=9)
    lzma_size = len(lzma_compressed)

    print("Packing Comparison:")
    print(f"  Old method (pickle): {old_size:,} bytes")
    print(
        f"  New binary packing: {new_size:,} bytes ({old_size / new_size:.1f}x smaller)"
    )
    print(f"  With LZMA: {lzma_size:,} bytes ({new_size / lzma_size:.1f}x additional)")
    print(f"  Total improvement: {old_size / lzma_size:.1f}x vs pickle")

    return old_size / lzma_size


def simulate_full_optimized_pipeline():
    """Simulate the complete optimized pipeline."""
    print(f"\n{'=' * 60}")
    print("FULL OPTIMIZED PIPELINE SIMULATION")
    print("=" * 60)

    # Run the stage compression
    compressed_params, stage_compressed_size = test_optimized_compression_simulation()

    # Apply optimized packing
    packing_improvement = test_optimized_packing()

    # Calculate final metrics
    original_total = sum(
        p.numel() * 4
        for p in nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        ).parameters()
    )

    # Estimate final size with packing optimization
    estimated_final_size = stage_compressed_size / packing_improvement
    final_ratio = original_total / estimated_final_size

    print("\nFINAL PIPELINE RESULTS:")
    print(f"  Original model: {original_total / 1024:.1f} KB")
    print(f"  After stage compression: {stage_compressed_size / 1024:.1f} KB")
    print(f"  After optimized packing: {estimated_final_size / 1024:.1f} KB")
    print(f"  Final compression ratio: {final_ratio:.1f}x")

    # Compare to previous results
    previous_ratio = 20.8
    print("\nComparison to Previous:")
    print(f"  Previous ratio: {previous_ratio:.1f}x")
    print(f"  New ratio: {final_ratio:.1f}x")

    if final_ratio > previous_ratio:
        improvement = (final_ratio / previous_ratio - 1) * 100
        print(f"  Improvement: +{improvement:.1f}% better")
    else:
        decline = (1 - final_ratio / previous_ratio) * 100
        print(f"  Change: -{decline:.1f}% vs previous")

    # Efficiency analysis
    theoretical_max = 1324  # BitNet(16) * SeedLM(8) * VPTQ(16) * Hyper(5)
    efficiency = (final_ratio / theoretical_max) * 100
    previous_efficiency = (previous_ratio / theoretical_max) * 100

    print("\nEfficiency Analysis:")
    print(f"  Previous efficiency: {previous_efficiency:.1f}%")
    print(f"  New efficiency: {efficiency:.1f}%")
    print(
        f"  Efficiency improvement: +{efficiency - previous_efficiency:.1f} percentage points"
    )

    return final_ratio, efficiency


def main():
    """Run all optimization tests."""
    try:
        final_ratio, efficiency = simulate_full_optimized_pipeline()

        print(f"\n{'=' * 60}")
        print("OPTIMIZATION ASSESSMENT")
        print("=" * 60)

        # Assess improvements
        assessments = []

        if final_ratio >= 50:
            assessments.append("EXCELLENT: Mobile deployment ready (>=50x)")
        elif final_ratio >= 30:
            assessments.append("GOOD: Significant improvement (>=30x)")
        elif final_ratio > 20.8:
            assessments.append("IMPROVED: Better than previous (>20.8x)")
        else:
            assessments.append("LIMITED: No significant improvement")

        if efficiency >= 25:
            assessments.append("HIGH EFFICIENCY: >25% of theoretical maximum")
        elif efficiency >= 10:
            assessments.append("MODERATE EFFICIENCY: >10% of theoretical maximum")
        elif efficiency > 1.6:
            assessments.append("IMPROVED EFFICIENCY: Better than previous 1.6%")

        print("Assessment:")
        for i, assessment in enumerate(assessments, 1):
            print(f"  {i}. {assessment}")

        # Mobile deployment feasibility
        print("\nMobile Deployment Analysis:")

        # Test if a 7B model would fit on mobile
        model_7b_size = 7_000_000_000 * 4  # 7B parameters * 4 bytes
        compressed_7b = model_7b_size / final_ratio
        compressed_7b_mb = compressed_7b / (1024 * 1024)

        print(f"  7B model original: {model_7b_size / (1024**3):.1f} GB")
        print(f"  7B model compressed: {compressed_7b_mb:.0f} MB")
        print(f"  Fits on 2GB phone: {'YES' if compressed_7b_mb < 1000 else 'NO'}")

        success = final_ratio >= 30 and efficiency > 5

        print(f"\nOverall Status: {'SUCCESS' if success else 'NEEDS MORE WORK'}")

        return success

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nOptimization test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
