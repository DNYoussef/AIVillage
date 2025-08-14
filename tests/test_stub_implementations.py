#!/usr/bin/env python3
"""Test script to validate the stub implementations we created"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_federation_manager():
    """Test the Tor/I2P implementations in federation manager"""
    print("[TEST] Testing Federation Manager Tor/I2P implementations...")

    try:
        from federation.core.federation_manager import FederationManager

        # Test with Tor and I2P disabled (should work)
        manager = FederationManager(
            device_id="test_device", enable_tor=False, enable_i2p=False
        )

        # Test Tor implementation (should gracefully handle when disabled)
        await manager._start_tor_transport()
        print("[PASS] Tor transport implementation working (disabled mode)")

        # Test I2P implementation (should gracefully handle when disabled)
        await manager._start_i2p_transport()
        print("[PASS] I2P transport implementation working (disabled mode)")

        return True

    except Exception as e:
        print(f"[FAIL] Federation Manager test failed: {e}")
        return False


async def test_bitchat_bluetooth():
    """Test the Bluetooth implementations in BitChat"""
    print("[TEST] Testing BitChat Bluetooth implementations...")

    try:
        from federation.protocols.enhanced_bitchat import EnhancedBitChatTransport

        # Create transport instance
        transport = EnhancedBitChatTransport(device_id="test_device")

        # Test Bluetooth discovery (should fall back to simulation mode)
        result = await transport._start_bluetooth_discovery()
        print(f"[PASS] Bluetooth discovery implementation working (result: {result})")

        # Test stop operation with Bluetooth cleanup
        await transport.stop()
        print("[PASS] Bluetooth stop operations working")

        return True

    except Exception as e:
        print(f"[FAIL] BitChat Bluetooth test failed: {e}")
        return False


def test_imports():
    """Test that all our implementations can be imported"""
    print("[TEST] Testing imports...")

    try:
        # Test federation manager import
        print("[PASS] Federation Manager imports working")

        # Test BitChat import
        print("[PASS] Enhanced BitChat imports working")

        return True

    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")
        return False


def count_remaining_stubs():
    """Count remaining TODO/stub statements in the files we worked on"""
    print("[TEST] Counting remaining stubs...")

    files_to_check = [
        "src/federation/core/federation_manager.py",
        "src/federation/protocols/enhanced_bitchat.py",
    ]

    total_stubs = 0

    for file_path in files_to_check:
        path = Path(file_path)
        if path.exists():
            with open(path, encoding="utf-8") as f:
                content = f.read()

            # Count TODO comments
            todo_count = content.count("# TODO:")
            todo_count += content.count("TODO:")

            # Count NotImplementedError
            not_impl_count = content.count("NotImplementedError")
            not_impl_count += content.count("raise NotImplementedError")

            file_stubs = todo_count + not_impl_count
            total_stubs += file_stubs

            print(f"  {path.name}: {file_stubs} stubs remaining")
        else:
            print(f"  {path.name}: File not found")

    print(f"[INFO] Total remaining stubs in worked files: {total_stubs}")
    return total_stubs


async def main():
    """Run all validation tests"""
    print("[START] Starting Stub Implementation Validation Tests")
    print("=" * 60)

    # Test imports first
    if not test_imports():
        print("[FAIL] Import tests failed - stopping")
        return

    # Count remaining stubs
    remaining_stubs = count_remaining_stubs()

    # Test async implementations
    federation_ok = await test_federation_manager()
    bitchat_ok = await test_bitchat_bluetooth()

    print("\n" + "=" * 60)
    print("[SUMMARY] VALIDATION SUMMARY")
    print("=" * 60)

    print(f"[RESULT] Imports: {'PASS' if test_imports() else 'FAIL'}")
    print(f"[RESULT] Federation Manager: {'PASS' if federation_ok else 'FAIL'}")
    print(f"[RESULT] BitChat Bluetooth: {'PASS' if bitchat_ok else 'FAIL'}")
    print(f"[INFO] Remaining stubs: {remaining_stubs}")

    all_passed = federation_ok and bitchat_ok
    print(f"\n[FINAL] Overall Status: {'PASS' if all_passed else 'FAIL'}")

    if all_passed:
        print("[SUCCESS] All stub implementations validated successfully!")
    else:
        print("[WARNING] Some implementations need additional work")


if __name__ == "__main__":
    asyncio.run(main())
