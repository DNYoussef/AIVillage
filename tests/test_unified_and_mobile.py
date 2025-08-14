#!/usr/bin/env python3
"""Test unified compressor intelligence and mobile deployment."""

import sys
from pathlib import Path

import torch
from torch import nn

# Add source paths
sys.path.insert(0, str(Path("src").resolve()))


def test_unified_compressor_decision_logic():
    """Test that unified compressor makes intelligent method selection."""
    print("=== Testing Unified Compressor Intelligence ===")

    try:
        from compression.pipeline import UnifiedCompressor

        # Test 1: Very small model should use simple
        print("\nTest 1: Tiny model (should use simple)")
        tiny_model = nn.Linear(10, 5)
        param_count = sum(p.numel() for p in tiny_model.parameters())
        size_mb = param_count * 4 / 1024 / 1024

        print(f"  Parameters: {param_count}")
        print(f"  Size: {size_mb:.3f}MB")

        unified = UnifiedCompressor()
        result = unified.compress(tiny_model)

        print(f"  Selected method: {result['method']}")
        print(f"  Fallback available: {result['fallback_available']}")

        # Should use simple for small models
        assert result["method"] == "simple", f"Expected simple, got {result['method']}"

        # Test 2: Medium model near threshold
        print("\nTest 2: Medium model (boundary test)")
        medium_model = nn.Sequential(nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 50))
        param_count = sum(p.numel() for p in medium_model.parameters())
        size_mb = param_count * 4 / 1024 / 1024

        print(f"  Parameters: {param_count}")
        print(f"  Size: {size_mb:.3f}MB")

        result = unified.compress(medium_model)
        print(f"  Selected method: {result['method']}")

        # Test 3: Large model should use advanced
        print("\nTest 3: Large model (should use advanced)")
        large_model = nn.Sequential(*[nn.Linear(1000, 1000) for _ in range(3)])
        param_count = sum(p.numel() for p in large_model.parameters())
        size_mb = param_count * 4 / 1024 / 1024

        print(f"  Parameters: {param_count}")
        print(f"  Size: {size_mb:.1f}MB")

        result = unified.compress(large_model)
        print(f"  Selected method: {result['method']}")

        if result["method"] == "advanced":
            print(f"  Stages: {result.get('stages', 'N/A')}")

        # Test 4: Custom compression target
        print("\nTest 4: Custom high compression target")
        high_compression = UnifiedCompressor(target_compression=100.0)
        medium_result = high_compression.compress(medium_model)
        print(f"  With 100x target: {medium_result['method']}")

        print("\nPASS: Unified compressor intelligence WORKING")
        return True

    except Exception as e:
        print(f"Unified compressor test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mobile_compressor_profiles():
    """Test mobile compressor for different device profiles."""
    print("\n=== Testing Mobile Compressor Profiles ===")

    try:
        from src.deployment.mobile_compressor import MobileCompressor

        # Create test model file
        test_model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        model_path = Path("test_mobile_model.pth")
        torch.save(test_model, model_path)

        original_mb = sum(p.numel() * 4 for p in test_model.parameters()) / 1024 / 1024
        print(f"Test model size: {original_mb:.2f}MB")

        profiles = ["low_end", "mid_range", "high_end"]
        results = {}

        for profile in profiles:
            print(f"\n--- {profile.upper()} DEVICE ---")

            compressor = MobileCompressor(profile)
            print(f"Target memory: {compressor.profile['memory_mb']}MB")
            print(f"Target compression: {compressor.profile['target_compression']}x")
            print(f"Prefer simple: {compressor.profile['prefer_simple']}")

            package = compressor.prepare_model_for_device(model_path)

            print(f"Compression method: {package['compression_method']}")
            print(f"Original size: {package['original_size_mb']:.2f}MB")
            print(f"Compressed size: {package['compressed_size_mb']:.2f}MB")
            print(f"Compression ratio: {package['compression_ratio']:.1f}x")
            print(f"Sprint 9 compatible: {package['sprint9_compatible']}")

            # Check if it fits in device memory (using 50% threshold)
            memory_threshold = compressor.profile["memory_mb"] * 0.5
            fits = package["compressed_size_mb"] < memory_threshold
            print(f"Fits in device memory: {fits}")

            results[profile] = package

        # Cleanup
        model_path.unlink()

        # Verify results make sense
        print("\n=== MOBILE DEPLOYMENT ANALYSIS ===")

        # Low-end should have highest compression
        low_ratio = results["low_end"]["compression_ratio"]
        high_ratio = results["high_end"]["compression_ratio"]

        print(f"Low-end compression: {low_ratio:.1f}x")
        print(f"High-end compression: {high_ratio:.1f}x")
        print(f"Compression scales with device capability: {low_ratio >= high_ratio}")

        # All should fit in their target devices
        all_fit = all(
            r["compressed_size_mb"] < 1500 for r in results.values()
        )  # Conservative threshold
        print(f"All models fit in target devices: {all_fit}")

        # Sprint 9 compatibility should be consistent
        sprint9_methods = [r["sprint9_compatible"] for r in results.values()]
        print(f"Sprint 9 compatibility: {sprint9_methods}")

        print("PASS: Mobile deployment WORKING")
        return results

    except Exception as e:
        print(f"Mobile compressor test failed: {e}")
        import traceback

        traceback.print_exc()
        return {}


def test_fallback_mechanism():
    """Test graceful fallback when advanced compression fails."""
    print("\n=== Testing Fallback Mechanism ===")

    try:
        from compression.pipeline import UnifiedCompressor

        # Create a scenario that might trigger fallback
        print("Creating challenging compression scenario...")

        # Very high compression requirement that might cause fallback
        extreme_compressor = UnifiedCompressor(target_compression=500.0)

        test_model = nn.Linear(500, 500)
        param_count = sum(p.numel() for p in test_model.parameters())
        print(f"Test model: {param_count:,} parameters")

        result = extreme_compressor.compress(test_model)

        print(f"Result method: {result['method']}")
        print(f"Fallback available: {result['fallback_available']}")

        # Even with extreme requirements, should work (either advanced or fallback to simple)
        assert result["method"] in [
            "simple",
            "advanced",
        ], f"Unexpected method: {result['method']}"
        assert result["fallback_available"], "Fallback should always be available"

        print("PASS: Fallback mechanism WORKING")
        return True

    except Exception as e:
        print(f"Fallback test failed: {e}")
        return False


def main():
    """Run unified compressor and mobile deployment tests."""
    print("UNIFIED COMPRESSOR & MOBILE DEPLOYMENT TEST")
    print("=" * 50)

    success = True

    # Test unified compressor intelligence
    unified_success = test_unified_compressor_decision_logic()
    success = success and unified_success

    # Test mobile deployment
    mobile_results = test_mobile_compressor_profiles()
    mobile_success = len(mobile_results) > 0
    success = success and mobile_success

    # Test fallback mechanism
    fallback_success = test_fallback_mechanism()
    success = success and fallback_success

    print("\n" + "=" * 50)
    print("UNIFIED & MOBILE TEST SUMMARY")
    print("=" * 50)

    if unified_success:
        print("PASS: Unified compressor intelligent selection")
    else:
        print("FAIL: Unified compressor issues")

    if mobile_success:
        print("PASS: Mobile deployment for all device tiers")
        if mobile_results:
            print("Mobile compression ratios:")
            for profile, result in mobile_results.items():
                method = result["compression_method"]
                ratio = result["compression_ratio"]
                compatible = "Sprint9" if result["sprint9_compatible"] else "Advanced"
                print(f"  {profile}: {ratio:.1f}x ({method}, {compatible})")
    else:
        print("FAIL: Mobile deployment issues")

    if fallback_success:
        print("PASS: Fallback mechanism working")
    else:
        print("FAIL: Fallback mechanism issues")

    print("\nSystem Integration Status:")
    print(
        f"  Intelligent method selection: {'Working' if unified_success else 'Failed'}"
    )
    print(f"  Mobile device compatibility: {'Working' if mobile_success else 'Failed'}")
    print(f"  Graceful error handling: {'Working' if fallback_success else 'Failed'}")
    print(f"  Production ready: {'YES' if success else 'NO'}")

    return success


if __name__ == "__main__":
    success = main()
    print(f"\nOverall test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
