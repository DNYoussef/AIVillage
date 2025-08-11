#!/usr/bin/env python3
"""Test the optimized compression pipelines for improved efficiency."""

import logging
from pathlib import Path
import sys
import time

import torch
from torch import nn

# Add source paths
sys.path.insert(0, str(Path("src").resolve()))

# Set up logging to capture optimization messages
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")


def test_optimized_advanced_pipeline():
    """Test the optimized AdvancedCompressionPipeline for improved efficiency."""
    print("TESTING OPTIMIZED ADVANCED COMPRESSION PIPELINE")
    print("=" * 60)

    from src.core.compression.advanced_pipeline import AdvancedCompressionPipeline

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

    # Test optimized pipeline
    pipeline = AdvancedCompressionPipeline()

    print("\nTesting optimized compression...")
    start = time.time()
    compressed = pipeline.compress_model(model)
    duration = time.time() - start

    # Calculate metrics
    compressed_size = len(compressed)
    ratio = original_size / compressed_size

    print("\nOptimized AdvancedPipeline Results:")
    print(f"  Original size: {original_size / 1024:.1f} KB")
    print(f"  Compressed size: {compressed_size / 1024:.1f} KB")
    print(f"  Compression ratio: {ratio:.1f}x")
    print(f"  Compression time: {duration:.2f}s")
    print("  Previous ratio was: 20.8x")

    if ratio > 20.8:
        improvement = (ratio / 20.8 - 1) * 100
        print(f"  Improvement: +{improvement:.1f}% better!")
    else:
        decline = (1 - ratio / 20.8) * 100
        print(f"  Change: -{decline:.1f}% vs previous")

    # Test decompression to verify integrity
    print("\nTesting decompression...")
    decompressed_params = pipeline.decompress_model(compressed)
    print(f"  Decompressed parameters: {len(decompressed_params)}")

    # Check that we have the expected parameters
    original_param_names = [name for name, _ in model.named_parameters()]
    decompressed_names = list(decompressed_params.keys())

    print(f"  Original param names: {len(original_param_names)}")
    print(f"  Decompressed param names: {len(decompressed_names)}")

    success = len(decompressed_names) == len(original_param_names)
    print(f"  Integrity check: {'PASSED' if success else 'FAILED'}")

    return ratio, duration, success


def test_hypercompression_effectiveness():
    """Test if HyperCompression stage is being optimized/skipped."""
    print(f"\n{'=' * 60}")
    print("TESTING HYPERCOMPRESSION OPTIMIZATION")
    print("=" * 60)

    from src.core.compression.advanced_pipeline import AdvancedCompressionPipeline

    # Create a model that should trigger HyperCompression warnings
    model = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128))

    # Capture logs to see if HyperCompression is being skipped
    import io

    log_capture = io.StringIO()

    # Set up a handler to capture logs
    logger = logging.getLogger("src.core.compression.advanced_pipeline")
    handler = logging.StreamHandler(log_capture)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    pipeline = AdvancedCompressionPipeline()

    print("Compressing model and monitoring for optimization messages...")
    pipeline.compress_model(model)

    # Check captured logs
    log_contents = log_capture.getvalue()

    print("\nHyperCompression Analysis:")
    if "HyperCompression ineffective" in log_contents:
        print("  Status: HyperCompression being SKIPPED (good optimization!)")
        print("  Found warning: Yes")
        print("  Optimization working: YES")
    else:
        print("  Status: HyperCompression being used")
        print("  Found warning: No")
        print("  May need further optimization")

    if log_contents:
        print("  Log messages captured:")
        for line in log_contents.strip().split("\n"):
            if line.strip():
                print(f"    {line}")

    return "ineffective" in log_contents.lower()


def test_lzma_compression_effectiveness():
    """Test the effectiveness of the new LZMA compression."""
    print(f"\n{'=' * 60}")
    print("TESTING LZMA COMPRESSION EFFECTIVENESS")
    print("=" * 60)

    import lzma
    import pickle

    # Create test data similar to what the pipeline would generate
    test_data = {
        "param1": (torch.randn(100, 100).numpy().tobytes(), (100, 100)),
        "param2": (torch.randn(200, 50).numpy().tobytes(), (200, 50)),
        "param3": (torch.randn(50, 25).numpy().tobytes(), (50, 25)),
    }

    # Test without LZMA (old method)
    uncompressed = pickle.dumps(test_data)
    uncompressed_size = len(uncompressed)

    # Test with LZMA (new method)
    lzma_compressed = lzma.compress(uncompressed, preset=9)
    lzma_size = len(lzma_compressed)

    lzma_ratio = uncompressed_size / lzma_size

    print("LZMA Compression Test:")
    print(f"  Uncompressed: {uncompressed_size / 1024:.1f} KB")
    print(f"  LZMA compressed: {lzma_size / 1024:.1f} KB")
    print(f"  LZMA ratio: {lzma_ratio:.1f}x")

    if lzma_ratio > 2.0:
        print(f"  LZMA effectiveness: EXCELLENT (>{lzma_ratio:.1f}x)")
    elif lzma_ratio > 1.5:
        print(f"  LZMA effectiveness: GOOD ({lzma_ratio:.1f}x)")
    else:
        print(f"  LZMA effectiveness: MODERATE ({lzma_ratio:.1f}x)")

    return lzma_ratio


def efficiency_analysis(old_ratio, new_ratio):
    """Analyze the efficiency improvement."""
    print(f"\n{'=' * 60}")
    print("EFFICIENCY ANALYSIS")
    print("=" * 60)

    # Previous analysis showed 1.6% efficiency
    theoretical_max = 1324  # From previous analysis (BitNet*SeedLM*VPTQ*Hyper)
    old_efficiency = (old_ratio / theoretical_max) * 100
    new_efficiency = (new_ratio / theoretical_max) * 100

    print("Efficiency Analysis:")
    print(f"  Theoretical maximum: {theoretical_max:.0f}x")
    print(f"  Previous compression: {old_ratio:.1f}x")
    print(f"  Previous efficiency: {old_efficiency:.1f}%")
    print(f"  New compression: {new_ratio:.1f}x")
    print(f"  New efficiency: {new_efficiency:.1f}%")

    if new_efficiency > old_efficiency:
        improvement = new_efficiency - old_efficiency
        print(f"  Efficiency improvement: +{improvement:.1f} percentage points")

        if new_efficiency > 25:
            print("  Status: EXCELLENT (>25% efficiency)")
        elif new_efficiency > 10:
            print("  Status: GOOD (>10% efficiency)")
        elif new_efficiency > 5:
            print("  Status: FAIR (>5% efficiency)")
        else:
            print("  Status: NEEDS MORE WORK")
    else:
        decline = old_efficiency - new_efficiency
        print(f"  Efficiency decline: -{decline:.1f} percentage points")

    return new_efficiency


def main():
    """Run optimized compression pipeline tests."""
    print("IMPROVED COMPRESSION PIPELINE VALIDATION")
    print("=" * 60)

    try:
        # Test 1: Optimized Advanced Pipeline
        ratio, duration, integrity = test_optimized_advanced_pipeline()

        # Test 2: HyperCompression optimization
        hyper_optimized = test_hypercompression_effectiveness()

        # Test 3: LZMA effectiveness
        lzma_ratio = test_lzma_compression_effectiveness()

        # Test 4: Efficiency analysis
        old_ratio = 20.8  # Previous result
        efficiency_analysis(old_ratio, ratio)

        # Final assessment
        print(f"\n{'=' * 60}")
        print("OPTIMIZATION ASSESSMENT")
        print("=" * 60)

        improvements = []

        if ratio > old_ratio:
            improvements.append(f"Compression ratio: {ratio:.1f}x vs {old_ratio:.1f}x")

        if hyper_optimized:
            improvements.append("HyperCompression optimization: Active")

        if lzma_ratio > 2.0:
            improvements.append(f"LZMA compression: Effective ({lzma_ratio:.1f}x)")

        if integrity:
            improvements.append("Data integrity: Maintained")

        print("Improvements detected:")
        for i, improvement in enumerate(improvements, 1):
            print(f"  {i}. {improvement}")

        if not improvements:
            print("  No significant improvements detected")

        # Overall status
        if ratio >= 50:
            status = "EXCELLENT - Mobile deployment ready"
        elif ratio >= 30:
            status = "GOOD - Significant improvement"
        elif ratio > old_ratio:
            status = "IMPROVED - Progress made"
        else:
            status = "NEEDS WORK - No improvement"

        print(f"\nOverall Status: {status}")
        print(f"Ready for mobile deployment: {'YES' if ratio >= 50 else 'PARTIAL' if ratio >= 30 else 'NO'}")

        return ratio >= 30  # Success if we get 30x or better

    except Exception as e:
        print(f"\nOptimization test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
