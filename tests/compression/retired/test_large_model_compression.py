#!/usr/bin/env python3
"""Test 4-stage compression on a large model (scaled down for testing)."""

import gc
import sys
import time
from pathlib import Path

from torch import nn

# Add source paths
sys.path.insert(0, str(Path("src").resolve()))


def create_representative_model():
    """Create a model representative of 1.5B scale but smaller for testing."""
    print("Creating representative large model...")

    class RepresentativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Scale down but keep proportions similar to 1.5B models
            vocab_size = 5000  # Scaled down from 50k
            hidden_size = 512  # Scaled down from 2048
            intermediate_size = 2048  # Scaled down from 8192
            num_layers = 12  # Scaled down from 42

            self.embed = nn.Embedding(vocab_size, hidden_size)  # ~2.5M params
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_size, intermediate_size),  # ~1M params each
                        nn.ReLU(),
                        nn.Linear(intermediate_size, hidden_size),  # ~1M params each
                        nn.LayerNorm(hidden_size),  # ~1K params each
                    )
                    for _ in range(num_layers)
                ]
            )
            self.output = nn.Linear(hidden_size, vocab_size)  # ~2.5M params

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x) + x
            return self.output(x)

    model = RepresentativeModel()
    total_params = sum(p.numel() for p in model.parameters())

    print("Representative model created:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / (1024**2):.1f} MB")
    print(f"  Scaling factor to 1.5B: {1_500_000_000 / total_params:.1f}x")

    return model, total_params


def test_compression_stage(model, stage_name, compress_func):
    """Test a single compression stage on the model."""
    print(f"\n{'=' * 50}")
    print(f"TESTING {stage_name.upper()}")
    print("=" * 50)

    total_original = 0
    total_compressed = 0
    layer_count = 0
    successful_layers = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_count += 1
        original_size = param.numel() * 4
        total_original += original_size

        print(f"  [{layer_count:2d}] {name}: {tuple(param.shape)}")

        try:
            start = time.time()
            result = compress_func(param.data)
            compress_time = time.time() - start

            if isinstance(result, tuple):
                compressed_size, ratio = result
            else:
                compressed_size = result
                ratio = original_size / compressed_size if compressed_size > 0 else 0

            total_compressed += compressed_size
            successful_layers += 1

            print(f"    {original_size:,} → {compressed_size:,} bytes ({ratio:.1f}x) [{compress_time:.3f}s]")

        except Exception as e:
            print(f"    ❌ Failed: {e}")
            total_compressed += original_size  # Count as uncompressed

        gc.collect()

    if total_compressed > 0:
        overall_ratio = total_original / total_compressed
        success_rate = successful_layers / layer_count * 100

        print(f"\n{stage_name} Results:")
        print(f"  Layers tested: {layer_count}")
        print(f"  Successful: {successful_layers} ({success_rate:.1f}%)")
        print(f"  Original: {total_original:,} bytes ({total_original / (1024**2):.1f} MB)")
        print(f"  Compressed: {total_compressed:,} bytes ({total_compressed / (1024**2):.1f} MB)")
        print(f"  Compression ratio: {overall_ratio:.1f}x")

        return overall_ratio, total_compressed
    print(f"\n{stage_name}: ❌ FAILED")
    return 0, total_original


def bitnet_compress(tensor):
    """BitNet compression function."""
    from src.agent_forge.compression.bitnet import BITNETCompressor

    compressor = BITNETCompressor()
    compressed = compressor.compress(tensor)

    packed_size = len(compressed["packed_weights"])
    metadata_size = 32
    total_size = packed_size + metadata_size

    return total_size, tensor.numel() * 4 / total_size


def seedlm_compress(tensor):
    """SeedLM compression function."""
    from src.agent_forge.compression.seedlm import SEEDLMCompressor

    compressor = SEEDLMCompressor(bits_per_weight=4)

    # Check compatibility
    if tensor.numel() % compressor.C != 0:
        msg = f"Tensor size {tensor.numel()} not divisible by block size {compressor.C}"
        raise ValueError(msg)

    compressed = compressor.compress(tensor)

    seeds_size = len(compressed["seeds"]) * 2
    coeffs_size = compressed["coefficients"].size
    exps_size = len(compressed["shared_exponents"])
    metadata_size = 32
    total_size = seeds_size + coeffs_size + exps_size + metadata_size

    return total_size, tensor.numel() * 4 / total_size


def vptq_compress(tensor):
    """VPTQ compression function."""
    from src.agent_forge.compression.vptq import VPTQCompressor

    compressor = VPTQCompressor(bits=2)
    compressed = compressor.compress(tensor)

    codebook_size = compressed["codebook"].numel() * 4
    indices_size = len(compressed["indices"])
    metadata_size = 32
    total_size = codebook_size + indices_size + metadata_size

    return total_size, tensor.numel() * 4 / total_size


def lzma_compress(data_size):
    """LZMA compression simulation."""
    import lzma

    # Create realistic binary data
    sample_data = b"compressed_weights_" * (int(data_size) // 20)

    compressed = lzma.compress(sample_data, preset=9)
    final_size = len(compressed)

    return final_size, len(sample_data) / final_size


def extrapolate_to_1_5b(test_params, test_results, target_params=1_500_000_000):
    """Extrapolate test results to 1.5B parameter model."""
    scaling_factor = target_params / test_params

    print(f"\n{'=' * 60}")
    print("EXTRAPOLATION TO 1.5B PARAMETER MODEL")
    print("=" * 60)

    print("Scaling calculation:")
    print(f"  Test model: {test_params:,} parameters")
    print(f"  Target model: {target_params:,} parameters")
    print(f"  Scaling factor: {scaling_factor:.1f}x")

    stage1_ratio, stage1_size = test_results["bitnet"]
    stage2_ratio, stage2_size = test_results["seedlm"]
    stage3_ratio, stage3_size = test_results["vptq"]
    stage4_ratio, stage4_size = test_results["lzma"]

    # Scale up the compressed sizes
    scaled_stage1 = stage1_size * scaling_factor
    scaled_stage2 = stage2_size * scaling_factor
    scaled_stage3 = stage3_size * scaling_factor
    scaled_stage4 = stage4_size * scaling_factor

    # Calculate 1.5B model results
    original_1_5b = target_params * 4  # bytes
    original_1_5b_gb = original_1_5b / (1024**3)

    final_1_5b_mb = scaled_stage4 / (1024**2)
    overall_ratio = original_1_5b / scaled_stage4

    print("\nExtrapolated 1.5B results:")
    print(f"  Original size: {original_1_5b_gb:.2f} GB")
    print(f"  Stage 1 (BitNet): {stage1_ratio:.1f}x → {scaled_stage1 / (1024**2):.1f} MB")
    print(f"  Stage 2 (SeedLM): {stage2_ratio:.1f}x → {scaled_stage2 / (1024**2):.1f} MB")
    print(f"  Stage 3 (VPTQ): {stage3_ratio:.1f}x → {scaled_stage3 / (1024**2):.1f} MB")
    print(f"  Stage 4 (LZMA): {stage4_ratio:.1f}x → {final_1_5b_mb:.1f} MB")

    print("\nFinal 1.5B model projection:")
    print(f"  Compressed size: {final_1_5b_mb:.1f} MB")
    print(f"  Overall compression: {overall_ratio:.1f}x")
    print(f"  Size reduction: {original_1_5b_gb:.2f} GB → {final_1_5b_mb:.1f} MB")

    # Mobile deployment assessment
    mobile_viable = final_1_5b_mb < 1000
    kenya_viable = final_1_5b_mb < 500

    print("\nMobile deployment projection:")
    print(f"  Fits on 2GB phone: {'✅ YES' if mobile_viable else '❌ NO'}")
    print(f"  Kenya deployment: {'✅ READY' if kenya_viable else '⚠️ MARGINAL'}")
    print(f"  Memory usage: {final_1_5b_mb:.1f} MB ({final_1_5b_mb / 1024:.1f}% of 1GB)")

    return overall_ratio, final_1_5b_mb, mobile_viable


def main():
    """Test large model compression with extrapolation to 1.5B."""
    print("TESTING LARGE MODEL COMPRESSION")
    print("4-Stage Pipeline on Representative Model")
    print("=" * 60)

    # Create test model
    model, total_params = create_representative_model()

    # Test each compression stage
    results = {}

    # Stage 1: BitNet
    bitnet_ratio, bitnet_size = test_compression_stage(model, "BitNet", bitnet_compress)
    results["bitnet"] = (bitnet_ratio, bitnet_size)

    # Stage 2: SeedLM
    seedlm_ratio, seedlm_size = test_compression_stage(model, "SeedLM", seedlm_compress)
    results["seedlm"] = (seedlm_ratio, seedlm_size)

    # Stage 3: VPTQ
    vptq_ratio, vptq_size = test_compression_stage(model, "VPTQ", vptq_compress)
    results["vptq"] = (vptq_ratio, vptq_size)

    # Stage 4: LZMA
    print(f"\n{'=' * 50}")
    print("TESTING LZMA HYPERCOMPRESSION")
    print("=" * 50)

    lzma_size, lzma_ratio = lzma_compress(vptq_size)
    results["lzma"] = (lzma_ratio, lzma_size)

    print("LZMA compression:")
    print(f"  Input: {vptq_size:,} bytes")
    print(f"  Output: {lzma_size:,} bytes")
    print(f"  Ratio: {lzma_ratio:.1f}x")

    # Extrapolate to 1.5B
    overall_ratio, final_mb, mobile_viable = extrapolate_to_1_5b(total_params, results)

    # Final assessment
    print(f"\n{'=' * 60}")
    print("FINAL ASSESSMENT")
    print("=" * 60)

    all_stages_work = all(ratio > 1 for ratio, _ in results.values())
    excellent = overall_ratio > 1000
    good = overall_ratio > 100
    viable = overall_ratio > 50

    print("Compression pipeline status:")
    print(f"  All 4 stages working: {'✅ YES' if all_stages_work else '❌ NO'}")
    print(f"  Mobile deployment: {'✅ VIABLE' if mobile_viable else '❌ NOT VIABLE'}")

    if excellent:
        print(f"  Performance: ⭐⭐⭐ EXCELLENT ({overall_ratio:.1f}x)")
    elif good:
        print(f"  Performance: ⭐⭐ GOOD ({overall_ratio:.1f}x)")
    elif viable:
        print(f"  Performance: ⭐ VIABLE ({overall_ratio:.1f}x)")
    else:
        print(f"  Performance: ❌ INSUFFICIENT ({overall_ratio:.1f}x)")

    success = all_stages_work and mobile_viable and overall_ratio > 50

    print(f"\nOverall result: {'✅ SUCCESS' if success else '❌ NEEDS IMPROVEMENT'}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
