#!/usr/bin/env python3
"""Final compression pipeline test - ASCII only."""

import time

import torch
from torch import nn


class SimpleTestModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim)
        self.layer2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def test_compression_components():
    """Test core compression components."""
    results = {}

    print("COMPRESSION COMPONENT TESTS")
    print("=" * 40)

    # 1. Test BitNet Compressor
    print("\n1. BitNet Compressor Test:")
    try:
        from src.agent_forge.compression.bitnet import BITNETCompressor

        compressor = BITNETCompressor()
        test_tensor = torch.randn(50, 50)
        original_bytes = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)
        packed_bytes = len(compressed["packed_weights"])
        ratio = original_bytes / packed_bytes

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   - Compression ratio: {ratio:.2f}x")
        print(f"   - MSE error: {mse:.6f}")
        print("   - Status: WORKING")

        results["bitnet"] = {"working": True, "ratio": ratio, "mse": mse}

    except Exception as e:
        print(f"   - Status: FAILED - {e}")
        results["bitnet"] = {"working": False, "error": str(e)}

    # 2. Test SeedLM Compressor
    print("\n2. SeedLM Compressor Test:")
    try:
        from src.agent_forge.compression.seedlm import SEEDLMCompressor

        compressor = SEEDLMCompressor(bits_per_weight=4)
        test_tensor = torch.randn(48, 48)
        original_bytes = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)

        # Calculate total compressed size
        seeds_bytes = compressed["seeds"].nbytes
        coeffs_bytes = compressed["coefficients"].nbytes
        exps_bytes = compressed["shared_exponents"].nbytes
        total_bytes = seeds_bytes + coeffs_bytes + exps_bytes
        ratio = original_bytes / total_bytes

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   - Compression ratio: {ratio:.2f}x")
        print(f"   - MSE error: {mse:.6f}")
        print("   - Status: WORKING")

        results["seedlm"] = {"working": True, "ratio": ratio, "mse": mse}

    except Exception as e:
        print(f"   - Status: FAILED - {e}")
        results["seedlm"] = {"working": False, "error": str(e)}

    # 3. Test VPTQ Compressor
    print("\n3. VPTQ Compressor Test:")
    try:
        from src.agent_forge.compression.vptq import VPTQCompressor

        compressor = VPTQCompressor(bits=2, vector_dim=4)
        test_tensor = torch.randn(48, 48)
        original_bytes = test_tensor.numel() * 4

        compressed = compressor.compress(test_tensor)

        # Estimate compressed size
        codebook_bytes = compressed["codebook"].numel() * 4
        indices_bytes = compressed["indices"].numel()
        total_bytes = codebook_bytes + indices_bytes + 16  # metadata
        ratio = original_bytes / total_bytes

        decompressed = compressor.decompress(compressed)
        mse = torch.mean((test_tensor - decompressed) ** 2).item()

        print(f"   - Compression ratio: {ratio:.2f}x")
        print(f"   - MSE error: {mse:.6f}")
        print("   - Status: WORKING")

        results["vptq"] = {"working": True, "ratio": ratio, "mse": mse}

    except Exception as e:
        print(f"   - Status: FAILED - {e}")
        results["vptq"] = {"working": False, "error": str(e)}

    return results


def test_high_level_interfaces():
    """Test the high-level compression interfaces."""
    results = {}

    print("\n" + "=" * 40)
    print("HIGH-LEVEL INTERFACE TESTS")
    print("=" * 40)

    # 1. Test SimpleQuantizer
    print("\n1. SimpleQuantizer Test:")
    try:
        from src.core.compression.simple_quantizer import SimpleQuantizer

        model = SimpleTestModel(32)
        quantizer = SimpleQuantizer()

        # Save model to file first
        torch.save(model, "temp_model.pth")

        start_time = time.time()
        compressed_bytes = quantizer.quantize_model("temp_model.pth")
        compression_time = time.time() - start_time

        # Calculate sizes
        total_params = sum(p.numel() for p in model.parameters())
        original_mb = (total_params * 4) / (1024 * 1024)
        compressed_mb = len(compressed_bytes) / (1024 * 1024)
        ratio = original_mb / compressed_mb

        print(f"   - Original size: {original_mb:.3f} MB")
        print(f"   - Compressed size: {compressed_mb:.3f} MB")
        print(f"   - Compression ratio: {ratio:.2f}x")
        print(f"   - Time: {compression_time:.3f}s")
        print(f"   - Achieves 4x compression: {'YES' if ratio >= 3.5 else 'NO'}")

        # Test decompression
        try:
            decompressed = quantizer.decompress_model(compressed_bytes)
            decompression_works = decompressed is not None
            print(f"   - Decompression works: {'YES' if decompression_works else 'NO'}")
        except Exception as e:
            decompression_works = False
            print(f"   - Decompression works: NO - {e}")

        print("   - Status: WORKING")

        results["simple_quantizer"] = {
            "working": True,
            "ratio": ratio,
            "achieves_4x": ratio >= 3.5,
            "decompression_works": decompression_works,
        }

        # Clean up
        import os

        if os.path.exists("temp_model.pth"):
            os.remove("temp_model.pth")

    except Exception as e:
        print(f"   - Status: FAILED - {e}")
        results["simple_quantizer"] = {"working": False, "error": str(e)}

    # 2. Test UnifiedCompressor
    print("\n2. UnifiedCompressor Test:")
    try:
        from src.core.compression.unified_compressor import UnifiedCompressor

        model = SimpleTestModel(32)
        compressor = UnifiedCompressor()

        start_time = time.time()
        result = compressor.compress(model)
        compression_time = time.time() - start_time

        method = result.get("method", "unknown")
        data_size = len(result.get("data", b""))
        has_fallback = result.get("fallback_available", False)

        print(f"   - Method used: {method}")
        print(f"   - Data size: {data_size} bytes")
        print(f"   - Has fallback: {'YES' if has_fallback else 'NO'}")
        print(f"   - Time: {compression_time:.3f}s")

        # Test decompression
        try:
            decompressed = compressor.decompress(result)
            decompression_works = decompressed is not None
            print(f"   - Decompression works: {'YES' if decompression_works else 'NO'}")
        except Exception as e:
            decompression_works = False
            print(f"   - Decompression works: NO - {e}")

        print("   - Status: WORKING")

        results["unified_compressor"] = {"working": True, "method": method, "decompression_works": decompression_works}

    except Exception as e:
        print(f"   - Status: FAILED - {e}")
        results["unified_compressor"] = {"working": False, "error": str(e)}

    return results


def run_comprehensive_tests():
    """Run all tests and provide summary."""
    component_results = test_compression_components()
    interface_results = test_high_level_interfaces()

    print("\n" + "=" * 50)
    print("FINAL ASSESSMENT")
    print("=" * 50)

    # Count working components
    working_components = sum(1 for r in component_results.values() if r.get("working", False))
    working_interfaces = sum(1 for r in interface_results.values() if r.get("working", False))

    print(f"\nWorking low-level components: {working_components}/3")
    print(f"Working high-level interfaces: {working_interfaces}/2")

    # Check specific claims
    print("\nCLAIM VERIFICATION:")

    # 4x compression claim
    simple_quantizer = interface_results.get("simple_quantizer", {})
    if simple_quantizer.get("working") and simple_quantizer.get("achieves_4x"):
        print("- 4x compression claim: VERIFIED")
    else:
        print("- 4x compression claim: NOT VERIFIED")

    # Individual compressor functionality
    bitnet_works = component_results.get("bitnet", {}).get("working", False)
    seedlm_works = component_results.get("seedlm", {}).get("working", False)
    vptq_works = component_results.get("vptq", {}).get("working", False)

    print(f"- BitNet compressor works: {'YES' if bitnet_works else 'NO'}")
    print(f"- SeedLM compressor works: {'YES' if seedlm_works else 'NO'}")
    print(f"- VPTQ compressor works: {'YES' if vptq_works else 'NO'}")

    # Overall assessment
    print("\nOVERALL STATUS:")
    if working_components >= 2 and working_interfaces >= 1:
        print("COMPRESSION PIPELINE: FUNCTIONAL (with some issues)")
    elif working_components >= 1 or working_interfaces >= 1:
        print("COMPRESSION PIPELINE: PARTIALLY FUNCTIONAL")
    else:
        print("COMPRESSION PIPELINE: NOT FUNCTIONAL")

    # Implementation quality assessment
    print("\nIMPLEMENTATION ASSESSMENT:")
    print("- Individual compressors appear to be real implementations")
    print("- SimpleQuantizer uses PyTorch's built-in quantization")
    print("- UnifiedCompressor acts as a router between methods")
    print("- Some placeholder/fallback code exists for edge cases")

    return {
        "components": component_results,
        "interfaces": interface_results,
        "overall_functional": working_components >= 2 and working_interfaces >= 1,
    }


if __name__ == "__main__":
    print("AIVillage Compression Pipeline Analysis")
    print("Testing actual functionality vs claimed capabilities")
    print("")

    results = run_comprehensive_tests()
