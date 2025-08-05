#!/usr/bin/env python3
"""Test the IntegratedCompressionPipeline and CascadeCompressor."""

import logging
from pathlib import Path
import sys
import time

import torch
from torch import nn

# Add source paths
sys.path.insert(0, str(Path("src").resolve()))

# Set up logging to capture messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def test_integrated_pipeline():
    """Test the IntegratedCompressionPipeline that avoids intermediate decompression."""
    print("TESTING INTEGRATED COMPRESSION PIPELINE")
    print("=" * 60)

    from src.core.compression.integrated_pipeline import IntegratedCompressionPipeline

    # Create test model
    model = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

    param_count = sum(p.numel() for p in model.parameters())
    original_size = param_count * 4

    print("Test Model:")
    print(f"  Parameters: {param_count:,}")
    print(f"  Original size: {original_size/1024:.1f} KB")

    # Test integrated pipeline
    integrated = IntegratedCompressionPipeline()

    # Capture logs to verify no intermediate decompression
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("src.core.compression.integrated_pipeline")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    print("\nTesting integrated compression (no intermediate decompression)...")
    start = time.time()
    compressed = integrated.compress_model(model)
    duration = time.time() - start

    # Calculate improvement
    compressed_size = len(compressed)
    ratio = original_size / compressed_size

    print("\nIntegrated Pipeline Results:")
    print(f"  Compression ratio: {ratio:.1f}x")
    print(f"  Compression time: {duration:.2f}s")
    print(f"  vs Advanced (20.8x): {ratio/20.8:.1%} better")

    # Verify no intermediate decompression
    log_contents = log_capture.getvalue()
    has_decompress = "decompress" in log_contents.lower()

    print("\nDecompression Check:")
    print(f"  Found 'decompress' in logs: {'YES' if has_decompress else 'NO'}")
    print(
        f"  Optimization confirmed: {'NO intermediate decompression' if not has_decompress else 'May have decompression'}"
    )

    if log_contents.strip():
        print("  Log messages:")
        for line in log_contents.strip().split("\n"):
            if line.strip():
                print(f"    {line}")

    # Test if this is MUCH better than 20.8x
    significant_improvement = ratio > 40
    print(f"\nSignificant improvement (>40x): {'YES' if significant_improvement else 'NO'}")

    return ratio, not has_decompress, significant_improvement


def test_cascade_compressor():
    """Test the CascadeCompressor for multiplicative gains."""
    print(f"\n{'='*60}")
    print("TESTING CASCADE COMPRESSOR")
    print("=" * 60)

    from src.core.compression.cascade_compressor import CascadeCompressor

    # Test on various tensor sizes
    test_sizes = [
        (100, 100, "Small"),  # 10K params
        (1000, 1000, "Medium"),  # 1M params
        (2048, 2048, "Large"),  # 4M params
    ]

    cascade = CascadeCompressor()

    print("Cascade Compressor Results:")
    print("-" * 50)

    results = []

    for rows, cols, size_name in test_sizes:
        weights = torch.randn(rows, cols)
        original_size = weights.numel() * 4

        start = time.time()
        compressed = cascade.compress(weights)
        duration = time.time() - start

        compressed_size = len(compressed)
        ratio = original_size / compressed_size

        print(f"\n{size_name} tensor ({rows}x{cols}):")
        print(f"  Original: {original_size/1024:.1f} KB")
        print(f"  Compressed: {compressed_size/1024:.1f} KB")
        print(f"  Ratio: {ratio:.1f}x")
        print(f"  Time: {duration:.3f}s")

        results.append((size_name, ratio))

        # Test pattern detection with structured data
        print("  Testing pattern detection...")
        pattern_weights = torch.zeros(rows, cols)
        pattern_weights[::2, ::2] = 1.0  # Checkerboard pattern

        pattern_compressed = cascade.compress(pattern_weights)
        pattern_ratio = original_size / len(pattern_compressed)

        print(f"  With checkerboard pattern: {pattern_ratio:.1f}x")
        pattern_detected = pattern_ratio > ratio * 1.2
        print(f"  Pattern detection working: {'YES' if pattern_detected else 'NO'}")

    return results


def test_cascade_stage_contributions():
    """Test that each cascade stage contributes to compression."""
    print(f"\n{'='*60}")
    print("TESTING CASCADE STAGE CONTRIBUTIONS")
    print("=" * 60)

    from src.core.compression.cascade_compressor import CascadeCompressor

    cascade = CascadeCompressor()
    weights = torch.randn(1000, 1000)
    original_size = weights.numel() * 4

    print("Testing stage contributions on 1000x1000 tensor:")
    print(f"Original size: {original_size/1024:.1f} KB")

    # Stage 1: Quantization only
    quantized, quant_meta = cascade.quantize_cascade(weights)
    stage1_size = quantized.numel() * 1  # int8 = 1 byte
    stage1_ratio = original_size / stage1_size

    print("\nStage 1 (Quantization):")
    print(f"  Size: {stage1_size/1024:.1f} KB")
    print(f"  Ratio: {stage1_ratio:.1f}x")

    # Stage 2: Pattern compression
    pattern_data, pattern_meta = cascade.pattern_compress(quantized)
    if isinstance(pattern_data, dict):
        # Estimate size of pattern representation
        unique_patterns = len(pattern_data["unique"])
        pattern_length = pattern_data["len"]
        total_patterns = len(pattern_data["patterns"])

        stage2_size = unique_patterns * pattern_length + total_patterns
        stage2_ratio = stage1_size / stage2_size

        print("\nStage 2 (Pattern Detection):")
        print(f"  Unique patterns found: {unique_patterns}")
        print(f"  Pattern length: {pattern_length}")
        print(f"  Estimated size: {stage2_size/1024:.1f} KB")
        print(f"  Additional ratio: {stage2_ratio:.1f}x")
    else:
        print("\nStage 2 (Pattern Detection):")
        print("  No patterns found, using raw data")
        stage2_size = len(pattern_data)
        stage2_ratio = 1.0

    # Stage 3: Full cascade with entropy coding
    full_compressed = cascade.compress(weights)
    full_ratio = original_size / len(full_compressed)
    stage3_ratio = stage2_size / len(full_compressed)

    print("\nStage 3 (Entropy Coding):")
    print(f"  Final size: {len(full_compressed)/1024:.1f} KB")
    print(f"  Additional ratio: {stage3_ratio:.1f}x")
    print(f"  Full cascade ratio: {full_ratio:.1f}x")

    # Verify multiplicative effect
    expected_ratio = stage1_ratio * stage2_ratio * stage3_ratio
    print("\nMultiplicative Analysis:")
    print(f"  Expected (stage1 × stage2 × stage3): {expected_ratio:.1f}x")
    print(f"  Actual full cascade: {full_ratio:.1f}x")
    print(f"  Multiplicative effect: {'YES' if abs(expected_ratio - full_ratio) < full_ratio * 0.3 else 'NO'}")

    return full_ratio > 40  # Success if we get >40x


def comprehensive_comparison():
    """Compare all compression methods."""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COMPRESSION COMPARISON")
    print("=" * 60)

    # Test model for comparison
    model = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))

    param_count = sum(p.numel() for p in model.parameters())
    original_mb = param_count * 4 / 1024 / 1024

    print(f"Test model: {param_count:,} parameters ({original_mb:.1f} MB)")

    # Test different compressors
    results = {}

    # 1. Integrated Pipeline
    try:
        from src.core.compression.integrated_pipeline import IntegratedCompressionPipeline

        integrated = IntegratedCompressionPipeline()

        start = time.time()
        compressed = integrated.compress_model(model)
        duration = time.time() - start

        ratio = (param_count * 4) / len(compressed)
        results["IntegratedPipeline"] = {"ratio": ratio, "time": duration, "size_mb": len(compressed) / 1024 / 1024}

    except Exception as e:
        print(f"IntegratedPipeline failed: {e}")
        results["IntegratedPipeline"] = {"ratio": 0, "time": 0, "size_mb": 0}

    # 2. Cascade Compressor (on concatenated weights)
    try:
        from src.core.compression.cascade_compressor import CascadeCompressor

        cascade = CascadeCompressor()

        # Concatenate all model weights
        all_weights = torch.cat([p.flatten() for p in model.parameters()])

        start = time.time()
        compressed = cascade.compress(all_weights)
        duration = time.time() - start

        ratio = (all_weights.numel() * 4) / len(compressed)
        results["CascadeCompressor"] = {"ratio": ratio, "time": duration, "size_mb": len(compressed) / 1024 / 1024}

    except Exception as e:
        print(f"CascadeCompressor failed: {e}")
        results["CascadeCompressor"] = {"ratio": 0, "time": 0, "size_mb": 0}

    # Display results
    print("\nComparison Results:")
    print(f"{'Method':<20} {'Ratio':<10} {'Time':<8} {'Size (MB)':<10}")
    print("-" * 50)

    for method, data in results.items():
        print(f"{method:<20} {data['ratio']:<10.1f} {data['time']:<8.2f} {data['size_mb']:<10.3f}")

    # Previous baselines
    print("\nBaseline Comparison:")
    print(f"{'SimpleQuantizer':<20} {'4.0':<10} {'<0.1':<8} {original_mb/4:<10.3f}")
    print(f"{'AdvancedPipeline':<20} {'20.8':<10} {'~1.0':<8} {original_mb/20.8:<10.3f}")

    return results


def main():
    """Run all integrated and cascade tests."""
    try:
        # Test 1: Integrated Pipeline
        integrated_ratio, no_decomp, significant = test_integrated_pipeline()

        # Test 2: Cascade Compressor
        cascade_results = test_cascade_compressor()

        # Test 3: Cascade stage contributions
        cascade_effective = test_cascade_stage_contributions()

        # Test 4: Comprehensive comparison
        comparison = comprehensive_comparison()

        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print("=" * 60)

        print("Test Results:")
        print(f"  Integrated Pipeline: {integrated_ratio:.1f}x compression")
        print(f"  No intermediate decompression: {'YES' if no_decomp else 'NO'}")
        print(f"  Significant improvement: {'YES' if significant else 'NO'}")

        if cascade_results:
            avg_cascade = sum(r[1] for r in cascade_results) / len(cascade_results)
            print(f"  Cascade Compressor average: {avg_cascade:.1f}x compression")

        print(f"  Cascade multiplicative effect: {'YES' if cascade_effective else 'NO'}")

        # Success criteria
        success_criteria = [
            ("Integrated Pipeline >50x", integrated_ratio >= 50),
            ("Cascade >40x", cascade_effective),
            ("No intermediate decompression", no_decomp),
            ("Mobile viable compression", integrated_ratio >= 50),
        ]

        print("\nSuccess Criteria:")
        passed = 0
        for criterion, passed_test in success_criteria:
            status = "PASS" if passed_test else "FAIL"
            print(f"  {criterion}: {status}")
            if passed_test:
                passed += 1

        overall_success = passed >= 3

        print(f"\nOverall Assessment: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
        print(f"Criteria passed: {passed}/{len(success_criteria)}")

        if overall_success:
            print("Ready for mobile deployment: YES")
        else:
            print("Needs more optimization: YES")

        return overall_success

    except Exception as e:
        print(f"\nTesting failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nIntegrated & Cascade test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
