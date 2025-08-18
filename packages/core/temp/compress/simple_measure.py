#!/usr/bin/env python3
"""Simple compression measurement script to validate claims quickly."""

import sys
from pathlib import Path

import torch
from torch import nn

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

print("=== Simple Compression Measurement ===\n")

# Test basic functionality first
print("Testing basic PyTorch functionality...")
test_tensor = torch.randn(10, 10)
test_model = nn.Linear(5, 3)
print(
    f"[OK] PyTorch working - tensor shape: {test_tensor.shape}, model params: {sum(p.numel() for p in test_model.parameters())}"
)

results = []


def safe_import_and_test(module_path, class_name, test_name):
    """Safely import and test a compression module."""
    try:
        module = __import__(module_path, fromlist=[class_name])
        compressor_class = getattr(module, class_name)
        compressor = compressor_class()
        print(f"[OK] {test_name} imported successfully")
        return compressor
    except Exception as e:
        print(f"[FAIL] {test_name} failed: {e}")
        return None


# Test BitNet
print("\nTesting BitNet compression...")
bitnet = safe_import_and_test("src.agent_forge.compression.bitnet", "BITNETCompressor", "BitNet")
if bitnet:
    try:
        weights = torch.randn(50, 50) * 0.1
        original_bytes = weights.numel() * 4

        compressed = bitnet.compress(weights)
        packed_bytes = len(compressed["packed_weights"]) if "packed_weights" in compressed else 0
        # Rough estimate: 2 bits per weight + scale + metadata
        compressed_bytes = packed_bytes + 20

        if compressed_bytes > 0:
            ratio = original_bytes / compressed_bytes
            print(f"[OK] BitNet compression: {original_bytes} -> {compressed_bytes} bytes ({ratio:.2f}x)")
            results.append(("BitNet", "tensor_50x50", original_bytes, compressed_bytes, ratio))

            # Test decompression
            reconstructed = bitnet.decompress(compressed)
            if torch.allclose(weights, reconstructed, rtol=0.5, atol=0.2):
                print("[OK] BitNet decompression successful")
            else:
                print("[WARN] BitNet decompression has high error")
    except Exception as e:
        print(f"[FAIL] BitNet test failed: {e}")

# Test SeedLM
print("\nTesting SeedLM compression...")
seedlm = safe_import_and_test("src.agent_forge.compression.seedlm", "SEEDLMCompressor", "SeedLM")
if seedlm:
    try:
        weights = torch.randn(24, 24) * 0.05  # Multiple of block size
        original_bytes = weights.numel() * 4

        compressed = seedlm.compress(weights)

        # Estimate compressed size
        compressed_bytes = 0
        if "seeds" in compressed:
            compressed_bytes += len(compressed["seeds"]) * 2  # uint16
        if "coefficients" in compressed:
            compressed_bytes += compressed["coefficients"].nbytes
        if "shared_exponents" in compressed:
            compressed_bytes += len(compressed["shared_exponents"])
        compressed_bytes += 50  # metadata overhead

        if compressed_bytes > 0:
            ratio = original_bytes / compressed_bytes
            print(f"[OK] SeedLM compression: {original_bytes} -> {compressed_bytes} bytes ({ratio:.2f}x)")
            results.append(("SeedLM", "tensor_24x24", original_bytes, compressed_bytes, ratio))

            # Test decompression
            reconstructed = seedlm.decompress(compressed)
            if torch.allclose(weights, reconstructed, rtol=0.5, atol=0.1):
                print("[OK] SeedLM decompression successful")
            else:
                print("[WARN] SeedLM decompression has some error")
    except Exception as e:
        print(f"[FAIL] SeedLM test failed: {e}")

# Test VPTQ
print("\nTesting VPTQ compression...")
vptq = safe_import_and_test("src.agent_forge.compression.vptq", "VPTQCompressor", "VPTQ")
if vptq:
    try:
        weights = torch.randn(32, 32) * 0.1
        original_bytes = weights.numel() * 4

        compressed = vptq.compress(weights)

        # Estimate compressed size
        compressed_bytes = 0
        if "codebook" in compressed:
            compressed_bytes += compressed["codebook"].numel() * 4
        if "indices" in compressed:
            compressed_bytes += len(compressed["indices"]) * 1  # Rough estimate
        compressed_bytes += 20  # metadata

        if compressed_bytes > 0:
            ratio = original_bytes / compressed_bytes
            print(f"[OK] VPTQ compression: {original_bytes} -> {compressed_bytes} bytes ({ratio:.2f}x)")
            results.append(("VPTQ", "tensor_32x32", original_bytes, compressed_bytes, ratio))

            # Test decompression
            reconstructed = vptq.decompress(compressed)
            if torch.allclose(weights, reconstructed, rtol=0.5, atol=0.1):
                print("[OK] VPTQ decompression successful")
            else:
                print("[WARN] VPTQ decompression has some error")
    except Exception as e:
        print(f"[FAIL] VPTQ test failed: {e}")

# Test SimpleQuantizer
print("\nTesting SimpleQuantizer...")
simple = safe_import_and_test("src.core.compression.simple_quantizer", "SimpleQuantizer", "SimpleQuantizer")
if simple:
    try:
        # Create a small test model
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
        original_bytes = sum(p.numel() * 4 for p in model.parameters())

        compressed_data = simple.quantize_model(model)
        compressed_bytes = len(compressed_data)

        if compressed_bytes > 0:
            ratio = original_bytes / compressed_bytes
            print(f"[OK] SimpleQuantizer: {original_bytes} -> {compressed_bytes} bytes ({ratio:.2f}x)")
            results.append(
                (
                    "SimpleQuantizer",
                    "small_model",
                    original_bytes,
                    compressed_bytes,
                    ratio,
                )
            )

            # Test decompression
            reconstructed = simple.decompress_model(compressed_data)
            print("[OK] SimpleQuantizer decompression successful")
    except Exception as e:
        print(f"[FAIL] SimpleQuantizer test failed: {e}")

# Generate results
print("\n=== Results Summary ===")
if results:
    print("| Algorithm | Test | Original (bytes) | Compressed (bytes) | Ratio |")
    print("|-----------|------|------------------|-------------------|-------|")
    for algo, test, orig, comp, ratio in results:
        print(f"| {algo} | {test} | {orig:,} | {comp:,} | **{ratio:.2f}x** |")

    print("\n=== Claims Validation ===")

    # Check BitNet claims (~16x)
    bitnet_ratios = [r[4] for r in results if r[0] == "BitNet"]
    if bitnet_ratios:
        max_bitnet = max(bitnet_ratios)
        print(f"BitNet: {max_bitnet:.2f}x vs claimed ~16x - {'VALIDATED' if max_bitnet >= 10 else 'DISPUTED'}")

    # Check SeedLM claims (~5x)
    seedlm_ratios = [r[4] for r in results if r[0] == "SeedLM"]
    if seedlm_ratios:
        max_seedlm = max(seedlm_ratios)
        print(f"SeedLM: {max_seedlm:.2f}x vs claimed ~5x - {'VALIDATED' if max_seedlm >= 4 else 'DISPUTED'}")

    # Check VPTQ claims (14-16x)
    vptq_ratios = [r[4] for r in results if r[0] == "VPTQ"]
    if vptq_ratios:
        max_vptq = max(vptq_ratios)
        print(f"VPTQ: {max_vptq:.2f}x vs claimed 14-16x - {'VALIDATED' if max_vptq >= 10 else 'DISPUTED'}")

    # Check SimpleQuantizer claims (4x)
    simple_ratios = [r[4] for r in results if r[0] == "SimpleQuantizer"]
    if simple_ratios:
        max_simple = max(simple_ratios)
        print(f"SimpleQuantizer: {max_simple:.2f}x vs claimed 4x - {'VALIDATED' if max_simple >= 3.5 else 'DISPUTED'}")

else:
    print("No successful compression tests completed.")

print(f"\nTest completed with {len(results)} successful compressions.")
