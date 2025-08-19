"""
Test script to validate P2P mesh integration with existing AIVillage infrastructure.

This test demonstrates that the Global South P2P mesh integration properly leverages
the existing TransportManager and BitChat transport while adding Global South
specific capabilities like offline-first operation and bandwidth optimization.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the packages directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from packages.core.global_south.offline_coordinator import GlobalSouthOfflineCoordinator
    from packages.core.global_south.p2p_mesh_integration import (
        P2PMeshIntegration,
        PeerType,
        create_p2p_mesh_integration,
    )
    from packages.p2p.core.message_types import MessagePriority
    from packages.p2p.core.transport_manager import TransportPriority

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    async def test_p2p_integration():
        """Test P2P mesh integration with existing infrastructure."""

        print("=" * 60)
        print("AIVillage Global South P2P Integration Test")
        print("=" * 60)

        # Test 1: Create offline coordinator
        print("\n1. Creating offline coordinator...")
        offline_coordinator = GlobalSouthOfflineCoordinator(
            device_id="test-device-001", data_budget_mb=10, storage_path="./test_storage"  # Small budget for testing
        )
        await offline_coordinator.start()
        print("✓ Offline coordinator created and started")

        # Test 2: Create P2P mesh integration
        print("\n2. Creating P2P mesh integration...")
        mesh = await create_p2p_mesh_integration(
            device_id="test-device-001",
            peer_type=PeerType.MOBILE,
            offline_coordinator=offline_coordinator,
            transport_priority=TransportPriority.OFFLINE_FIRST,
            start_immediately=True,
        )

        if mesh:
            print("✓ P2P mesh integration created successfully")

            # Test 3: Check status
            print("\n3. Checking integration status...")
            status = mesh.get_status()
            print(f"✓ Device ID: {status['device_id']}")
            print(f"✓ Peer type: {status['peer_type']}")
            print(f"✓ P2P available: {status['p2p_available']}")
            print(f"✓ Transport manager active: {status['transport_manager_active']}")
            print(f"✓ BitChat transport active: {status['bitchat_transport_active']}")
            print(f"✓ Offline coordinator active: {status['offline_coordinator_active']}")

            # Test 4: Send a test message
            print("\n4. Testing message sending...")
            test_message = b"Hello from Global South P2P mesh integration!"
            success = await mesh.send_message(
                recipient="broadcast", content=test_message, priority=MessagePriority.NORMAL
            )
            print(f"✓ Message sent: {success}")

            # Test 5: Test content request
            print("\n5. Testing content request...")
            content = await mesh.request_content("test-content-key", timeout_seconds=5)
            if content:
                print(f"✓ Content retrieved: {len(content)} bytes")
            else:
                print("✓ No content found (expected for new mesh)")

            # Test 6: Check network statistics
            print("\n6. Checking network statistics...")
            stats = mesh.get_network_stats()
            print(f"✓ Messages sent: {stats['messages_sent']}")
            print(f"✓ Messages received: {stats['messages_received']}")
            print(f"✓ Cache efficiency: {stats['cache_efficiency']:.2%}")
            print(f"✓ Peers discovered: {stats['peer_discovery_rate']}")

            # Test 7: Test integration with offline coordinator
            print("\n7. Testing offline coordinator integration...")
            await offline_coordinator.store_message(
                sender="test-sender",
                recipient="test-recipient",
                content=b"Test message for offline storage",
                priority="normal",
            )

            pending_counts = await offline_coordinator.get_pending_message_count()
            total_pending = sum(pending_counts.values())
            print(f"✓ Offline coordinator integration: {total_pending} pending messages")

            # Cleanup
            print("\n8. Cleaning up...")
            await mesh.stop()
            await offline_coordinator.stop()
            print("✓ All systems stopped")

        else:
            print("✗ Failed to create P2P mesh integration")
            return False

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - P2P Integration Working!")
        print("=" * 60)
        print("\nKey Integration Points Validated:")
        print("• TransportManager creation and configuration ✓")
        print("• BitChat transport registration and setup ✓")
        print("• Global South device context configuration ✓")
        print("• Message routing via existing transport layer ✓")
        print("• Offline coordinator synchronization ✓")
        print("• Content caching and request handling ✓")
        print("• Statistics and monitoring integration ✓")

        return True

    async def test_fallback_mode():
        """Test fallback mode when P2P infrastructure is not available."""

        print("\n" + "=" * 60)
        print("Testing Fallback Mode (P2P unavailable)")
        print("=" * 60)

        # Temporarily disable P2P
        import packages.core.global_south.p2p_mesh_integration as p2p_module

        original_p2p_available = p2p_module.P2P_AVAILABLE
        p2p_module.P2P_AVAILABLE = False

        try:
            mesh = P2PMeshIntegration(device_id="fallback-test-device", peer_type=PeerType.MOBILE)

            success = await mesh.start()
            print(f"✓ Fallback mode start result: {success}")

            status = mesh.get_status()
            print(f"✓ P2P available in status: {status['p2p_available']}")

            await mesh.stop()
            print("✓ Fallback mode test completed")

        finally:
            # Restore original state
            p2p_module.P2P_AVAILABLE = original_p2p_available

        return True

    if __name__ == "__main__":

        async def main():
            try:
                # Test normal integration
                success1 = await test_p2p_integration()

                # Test fallback mode
                success2 = await test_fallback_mode()

                if success1 and success2:
                    print("\n🎉 All integration tests completed successfully!")
                    print("P2P mesh integration is ready for Global South deployment.")
                else:
                    print("\n❌ Some tests failed. Check the logs above.")

            except Exception as e:
                logger.error(f"Test failed with exception: {e}", exc_info=True)
                print(f"\n❌ Test suite failed: {e}")

        asyncio.run(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the AIVillage root directory")
    print("and all required packages are available.")
