#!/usr/bin/env python3
"""Check for hidden issues in compression implementation."""


def check_stage_4_hyper() -> None:
    """Check if HyperCompression stage 4 is actually working."""
    print("\n=== CHECKING HYPERCOMPRESSION (STAGE 4) ===")

    try:
        from production.compression.hyper_compression import HyperCompressionEncoder

        encoder = HyperCompressionEncoder()

        # Test with simple data
        test_data = b"x" * 10000
        compressed = encoder.encode(test_data)

        print("HyperCompression found and tested")
        print(f"Test compression: {len(test_data)} -> {len(compressed)} bytes")
        print(f"Ratio: {len(test_data) / len(compressed):.2f}x")

        # Check if it's real compression or fake
        if compressed == test_data:
            print("WARNING: HyperCompression might be a passthrough!")

    except Exception as e:
        print(f"HyperCompression FAILED: {e}")
        print("This explains why total compression is low!")


def check_overhead_accumulation() -> None:
    """Check how much overhead accumulates through stages."""
    print("\n=== METADATA OVERHEAD ANALYSIS ===")

    import pickle

    # Simulate multi-stage compression metadata
    stage1_output = {
        "data": b"x" * 1000,
        "method": "bitnet",
        "metadata": {"scale": 1.0, "threshold": 0.7},
    }

    stage2_output = {
        "stage1": stage1_output,
        "data": b"y" * 500,
        "method": "seedlm",
        "metadata": {"seeds": [1, 2, 3], "coefficients": [4, 5, 6]},
    }

    stage3_output = {
        "stage2": stage2_output,
        "data": b"z" * 250,
        "method": "vptq",
        "metadata": {"codebook": [[1, 2], [3, 4]], "indices": [0, 1, 0, 1]},
    }

    # Calculate overhead
    data_size = 250  # Final compressed data
    total_size = len(pickle.dumps(stage3_output))
    overhead = total_size - data_size

    print(f"Compressed data: {data_size} bytes")
    print(f"With metadata: {total_size} bytes")
    print(f"Overhead: {overhead} bytes ({overhead / data_size * 100:.1f}%)")
    print("This overhead compounds and reduces effective compression!")


def check_decompression_quality() -> None:
    """Check if quality degrades through stages."""
    print("\n=== QUALITY DEGRADATION CHECK ===")

    import torch

    from agent_forge.compression.bitnet import BITNETCompressor
    from agent_forge.compression.seedlm import SEEDLMCompressor

    # Original data
    original = torch.randn(1000, 1000)

    # Single stage
    bitnet = BITNETCompressor()
    compressed = bitnet.compress(original)
    decompressed = bitnet.decompress(compressed)

    single_stage_error = torch.mean((original - decompressed) ** 2).item()
    print(f"Single stage MSE: {single_stage_error:.6f}")

    # Two stages
    seedlm = SEEDLMCompressor(bits_per_weight=4)
    stage2_compressed = seedlm.compress(decompressed)  # Compress already lossy data
    stage2_decompressed = seedlm.decompress(stage2_compressed)

    two_stage_error = torch.mean((original - stage2_decompressed) ** 2).item()
    print(f"Two stage MSE: {two_stage_error:.6f}")
    print(f"Error amplification: {two_stage_error / single_stage_error:.2f}x")

    print("\nConclusion: Errors compound through stages!")


def verify_claude_math() -> None:
    """Check Claude's reported numbers."""
    print("\n=== VERIFYING CLAUDE'S MATH ===")

    print("Claude claims:")
    print("- BitNet: 15.8x")
    print("- SeedLM: 5.3x")
    print("- VPTQ: 15.8x")
    print("- Combined: 20.8x")

    print("\nIf multiplicative: 15.8 × 5.3 × 15.8 = 1,324x")
    print("Actual: 20.8x")
    print("Efficiency: 20.8 / 1324 = 1.6%")

    print("\nThis suggests:")
    print("1. Stages are NOT multiplicative")
    print("2. Massive overhead or inefficiency")
    print("3. Possible implementation bugs")
    print("4. Or Claude's numbers are wrong")


check_stage_4_hyper()
check_overhead_accumulation()
check_decompression_quality()
verify_claude_math()
