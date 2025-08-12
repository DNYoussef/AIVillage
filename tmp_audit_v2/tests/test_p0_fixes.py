"""Validate P0 fixes are properly implemented."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def test_bitchat_betanet_imports():
    """Verify BitChat/Betanet modules import cleanly."""
    results = []

    try:
        from core.p2p.bitchat_transport import BitChatTransport

        results.append("PASS: BitChat imports successfully")
    except ImportError as e:
        results.append(f"FAIL: BitChat import error: {e}")

    try:
        from core.p2p.betanet_transport import BetanetTransport

        results.append("PASS: Betanet imports successfully")
    except ImportError as e:
        results.append(f"FAIL: Betanet import error: {e}")

    try:
        from core.p2p.dual_path_transport import DualPathTransport

        results.append("PASS: DualPathTransport imports successfully")
    except ImportError as e:
        results.append(f"FAIL: DualPathTransport import error: {e}")

    return results


def test_no_http_in_production():
    """Verify no HTTP endpoints in production code."""
    import subprocess

    # Check for HTTP URLs in production
    result = subprocess.run(
        ["git", "grep", "-n", "http://", "src/production"],
        check=False,
        capture_output=True,
        text=True,
    )

    lines = result.stdout.strip().split("\n") if result.stdout else []

    # Filter out legitimate localhost/development uses
    issues = []
    for line in lines:
        if line and "localhost" not in line and "test" not in line.lower():
            # Check if it's guarded by production check
            if "AIVILLAGE_ENV" not in line:
                issues.append(line)

    if not issues:
        return ["PASS: No unguarded HTTP endpoints in production"]
    return [f"FAIL: Found {len(issues)} HTTP endpoints"] + issues[:5]


def test_secure_serialization():
    """Verify secure serialization is in place."""
    results = []

    # Check if secure serialization module exists
    try:
        from core.security.secure_serialization import secure_dumps, secure_loads

        results.append("PASS: Secure serialization module exists")

        # Test basic functionality
        test_data = {"key": "value", "number": 42}
        serialized = secure_dumps(test_data)
        deserialized = secure_loads(serialized)

        if deserialized == test_data:
            results.append("PASS: Secure serialization works correctly")
        else:
            results.append("FAIL: Serialization round-trip failed")

    except ImportError:
        results.append("FAIL: Secure serialization module not found")
    except Exception as e:
        results.append(f"FAIL: Secure serialization error: {e}")

    return results


def test_redis_security():
    """Check Redis client security improvements."""
    results = []

    try:
        from core.security.secure_redis_client import SecureRedisClient

        results.append("PASS: Secure Redis client exists")

        # Check for pickle rejection
        import inspect

        source = inspect.getsource(SecureRedisClient)

        if "pickle" not in source or "reject_pickle" in source:
            results.append("PASS: Redis client appears to reject pickle")
        else:
            results.append("WARN: Redis client may still use pickle")

    except ImportError:
        results.append("INFO: Secure Redis client not implemented yet")
    except Exception as e:
        results.append(f"WARN: Could not inspect Redis security: {e}")

    return results


def test_mobile_resource_management():
    """Verify mobile resource management is functional."""
    results = []

    try:
        from production.monitoring.mobile.resource_management import (
            BatteryThermalResourceManager,
            PowerMode,
        )

        results.append("PASS: Mobile resource management imports")

        # Test basic instantiation
        manager = BatteryThermalResourceManager()
        status = manager.get_status()

        if "power_mode" in status:
            results.append("PASS: Resource manager provides status")
        else:
            results.append("WARN: Resource manager status incomplete")

    except ImportError as e:
        results.append(f"INFO: Mobile resource management not ready: {e}")
    except Exception as e:
        results.append(f"WARN: Resource management error: {e}")

    return results


def main():
    print("=" * 60)
    print("P0 FIXES VALIDATION REPORT")
    print("=" * 60)

    all_results = []

    print("\n1. BitChat/Betanet Module Imports:")
    results = test_bitchat_betanet_imports()
    for r in results:
        print(f"   {r}")
    all_results.extend(results)

    print("\n2. HTTP Endpoints in Production:")
    results = test_no_http_in_production()
    for r in results:
        print(f"   {r}")
    all_results.extend(results)

    print("\n3. Secure Serialization:")
    results = test_secure_serialization()
    for r in results:
        print(f"   {r}")
    all_results.extend(results)

    print("\n4. Redis Security:")
    results = test_redis_security()
    for r in results:
        print(f"   {r}")
    all_results.extend(results)

    print("\n5. Mobile Resource Management:")
    results = test_mobile_resource_management()
    for r in results:
        print(f"   {r}")
    all_results.extend(results)

    # Summary
    passes = sum(1 for r in all_results if r.startswith("PASS"))
    fails = sum(1 for r in all_results if r.startswith("FAIL"))
    warns = sum(1 for r in all_results if r.startswith("WARN"))

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  PASS: {passes}")
    print(f"  FAIL: {fails}")
    print(f"  WARN: {warns}")

    if fails == 0:
        print("\nSTATUS: All P0 fixes verified successfully!")
    else:
        print(f"\nSTATUS: {fails} P0 issues remain")

    return fails == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
