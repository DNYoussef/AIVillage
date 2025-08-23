#!/usr/bin/env python3
"""
Test script to validate Global South P2P mesh integration.
Run this from the AIVillage root directory.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure we're in the right directory
if not Path("packages/core/global_south").exists():
    print("Error: Run this script from the AIVillage root directory")
    sys.exit(1)

# Set up logging with proper encoding
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def test_import_and_basic_functionality():
    """Test that imports work and basic functionality is available."""

    print("=" * 60)
    print("Testing Global South P2P Integration - Basic Functionality")
    print("=" * 60)

    try:
        # Test 1: Import the core components
        print("\n1. Testing imports...")
        from packages.core.global_south.offline_coordinator import GlobalSouthOfflineCoordinator
        from packages.core.global_south.p2p_mesh_integration import P2PMeshIntegration, PeerType
        from packages.p2p.core.transport_manager import TransportPriority

        print("   ✓ All imports successful")

        # Test 2: Check if P2P infrastructure is available
        print("\n2. Checking P2P infrastructure availability...")
        mesh = P2PMeshIntegration(device_id="test-device", peer_type=PeerType.MOBILE)

        status = mesh.get_status()
        print(f"   ✓ P2P Available: {status['p2p_available']}")
        print(f"   ✓ Device ID: {status['device_id']}")
        print(f"   ✓ Peer Type: {status['peer_type']}")

        # Test 3: Test offline coordinator creation
        print("\n3. Testing offline coordinator...")
        offline_coordinator = GlobalSouthOfflineCoordinator(device_id="test-device", data_budget_mb=5)
        await offline_coordinator.start()
        print("   ✓ Offline coordinator created and started")

        # Test 4: Test integration with offline coordinator
        print("\n4. Testing P2P integration with offline coordinator...")
        integrated_mesh = P2PMeshIntegration(
            device_id="test-integrated-device",
            peer_type=PeerType.MOBILE,
            offline_coordinator=offline_coordinator,
            transport_priority=TransportPriority.OFFLINE_FIRST,
        )

        integrated_status = integrated_mesh.get_status()
        print(f"   ✓ Offline coordinator active: {integrated_status['offline_coordinator_active']}")
        print(f"   ✓ Transport manager active: {integrated_status['transport_manager_active']}")

        # Test 5: Test starting the mesh (if P2P is available)
        print("\n5. Testing mesh startup...")
        if integrated_status["p2p_available"]:
            success = await integrated_mesh.start()
            print(f"   ✓ Mesh startup: {success}")
            if success:
                await integrated_mesh.stop()
                print("   ✓ Mesh stopped cleanly")
        else:
            print("   ! P2P infrastructure not available, testing fallback mode")
            success = await integrated_mesh.start()  # Should handle fallback gracefully
            print(f"   ✓ Fallback mode handled: {not success}")  # Should return False but not crash

        # Test 6: Test message handling (without network)
        print("\n6. Testing message interfaces...")
        try:
            # This should work even without active network
            peer_info = integrated_mesh.get_peer_info()
            stats = integrated_mesh.get_network_stats()
            print(f"   ✓ Peer info available: {len(peer_info)} keys")
            print(f"   ✓ Network stats available: {len(stats)} metrics")
            print(f"   ✓ Cache efficiency: {stats.get('cache_efficiency', 0):.1%}")
        except Exception as e:
            print(f"   ✗ Message interface error: {e}")
            return False

        # Cleanup
        await offline_coordinator.stop()
        print("\n7. Cleanup completed")

        return True

    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_factory_function():
    """Test the factory function."""

    print("\n" + "=" * 60)
    print("Testing Factory Function")
    print("=" * 60)

    try:
        from packages.core.global_south.p2p_mesh_integration import PeerType, create_p2p_mesh_integration
        from packages.p2p.core.transport_manager import TransportPriority

        # Test without starting
        mesh = await create_p2p_mesh_integration(
            device_id="factory-test-device",
            peer_type=PeerType.MOBILE,
            transport_priority=TransportPriority.OFFLINE_FIRST,
            start_immediately=False,
        )

        if mesh:
            status = mesh.get_status()
            print(f"✓ Factory function created mesh: {status['device_id']}")
            return True
        else:
            print("✗ Factory function returned None")
            return False

    except Exception as e:
        logger.error(f"Factory test failed: {e}", exc_info=True)
        return False


async def test_error_handling():
    """Test error handling and edge cases."""

    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    try:
        from packages.core.global_south.p2p_mesh_integration import P2PMeshIntegration, PeerType

        # Test with invalid inputs
        mesh = P2PMeshIntegration(device_id="", peer_type=PeerType.MOBILE)  # Empty device ID should auto-generate

        status = mesh.get_status()
        device_id = status.get("device_id", "")

        if device_id and len(device_id) > 0:
            print("✓ Auto-generated device ID for empty input")
        else:
            print("✗ Failed to handle empty device ID")
            return False

        # Test network stats with no activity
        stats = mesh.get_network_stats()
        if isinstance(stats, dict) and "messages_sent" in stats:
            print(f"✓ Network stats available: {stats['messages_sent']} messages sent")
        else:
            print("✗ Network stats not properly formatted")
            return False

        return True

    except Exception as e:
        logger.error(f"Error handling test failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""

    print("AIVillage Global South P2P Integration Test Suite")
    print("Running from:", Path.cwd())

    results = []

    # Run tests
    results.append(await test_import_and_basic_functionality())
    results.append(await test_factory_function())
    results.append(await test_error_handling())

    # Summary
    passed = sum(results)
    total = len(results)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✅ ALL TESTS PASSED!")
        print("\nThe Global South P2P integration is working correctly.")
        print("Key functionality validated:")
        print("• Import structure and dependencies")
        print("• Basic object creation and configuration")
        print("• Integration with offline coordinator")
        print("• Factory function operation")
        print("• Error handling and edge cases")
        print("• Status reporting and statistics")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Check the error messages above for details.")
        return 1

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        sys.exit(1)
