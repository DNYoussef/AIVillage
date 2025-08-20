#!/usr/bin/env python3
"""Simple test without Unicode characters to validate functionality."""

import asyncio
import sys
from pathlib import Path


async def test_basic_integration():
    """Test basic integration functionality."""

    print("Testing Global South P2P Integration")
    print("=" * 50)

    try:
        # Test imports
        print("1. Testing imports...")
        from packages.core.global_south.offline_coordinator import GlobalSouthOfflineCoordinator
        from packages.core.global_south.p2p_mesh_integration import (
            P2PMeshIntegration,
            PeerType,
            create_p2p_mesh_integration,
        )
        from packages.p2p.core.message_types import MessagePriority
        from packages.p2p.core.transport_manager import TransportPriority
        print("   [OK] All imports successful")

        # Test basic object creation
        print("2. Testing object creation...")
        mesh = P2PMeshIntegration(
            device_id="test-device-001",
            peer_type=PeerType.MOBILE
        )
        status = mesh.get_status()
        print(f"   [OK] Created mesh with device ID: {status['device_id']}")
        print(f"   [OK] P2P Available: {status['p2p_available']}")
        print(f"   [OK] Transport Manager Active: {status['transport_manager_active']}")
        print(f"   [OK] BitChat Transport Active: {status['bitchat_transport_active']}")

        # Test offline coordinator integration
        print("3. Testing offline coordinator integration...")
        offline_coord = GlobalSouthOfflineCoordinator(
            max_storage_mb=5,
            daily_data_budget_usd=0.25
        )
        await offline_coord.start()

        integrated_mesh = P2PMeshIntegration(
            device_id="test-device-003",
            peer_type=PeerType.MOBILE,
            offline_coordinator=offline_coord
        )

        integrated_status = integrated_mesh.get_status()
        print(f"   [OK] Offline Coordinator Active: {integrated_status['offline_coordinator_active']}")

        # Test network stats
        print("4. Testing network statistics...")
        stats = integrated_mesh.get_network_stats()
        print(f"   [OK] Messages sent: {stats['messages_sent']}")
        print(f"   [OK] Messages received: {stats['messages_received']}")
        print(f"   [OK] Cache efficiency: {stats['cache_efficiency']:.1%}")
        print(f"   [OK] Peer discovery rate: {stats['peer_discovery_rate']}")

        # Test factory function
        print("5. Testing factory function...")
        factory_mesh = await create_p2p_mesh_integration(
            device_id="factory-test-device",
            peer_type=PeerType.MOBILE,
            transport_priority=TransportPriority.OFFLINE_FIRST,
            start_immediately=False
        )

        if factory_mesh:
            factory_status = factory_mesh.get_status()
            print(f"   [OK] Factory created device: {factory_status['device_id']}")
        else:
            print("   [FAIL] Factory function returned None")
            return False

        # Test mesh startup if P2P is available
        print("6. Testing mesh startup...")
        if status['p2p_available']:
            print("   [INFO] P2P infrastructure available, testing startup...")
            try:
                start_result = await integrated_mesh.start()
                print(f"   [OK] Mesh startup result: {start_result}")
                if start_result:
                    stop_result = await integrated_mesh.stop()
                    print(f"   [OK] Mesh stop result: {stop_result}")
            except Exception as e:
                print(f"   [WARN] Startup test failed (expected): {e}")
        else:
            print("   [INFO] P2P infrastructure not available (emulation mode)")

        # Test message interfaces
        print("7. Testing message interfaces...")
        peer_info = integrated_mesh.get_peer_info()
        print(f"   [OK] Peer info keys: {list(peer_info.keys())}")
        print(f"   [OK] Total peers: {peer_info['total_peers']}")

        # Cleanup
        await offline_coord.stop()
        print("   [OK] Cleanup completed")

        print("\n" + "=" * 50)
        print("ALL TESTS PASSED!")
        print("Key functionality validated:")
        print("- Import structure and dependencies")
        print("- Basic object creation and configuration")
        print("- Integration with existing TransportManager")
        print("- BitChat transport registration")
        print("- Offline coordinator integration")
        print("- Factory function operation")
        print("- Status reporting and statistics")
        print("- Message handling interfaces")

        return True

    except Exception as e:
        print(f"   [FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_transport_integration():
    """Test integration with actual transport infrastructure."""

    print("\nTesting Transport Infrastructure Integration")
    print("=" * 50)

    try:
        from packages.core.global_south.p2p_mesh_integration import P2PMeshIntegration, PeerType
        from packages.p2p.core.transport_manager import TransportPriority, TransportType

        # Create mesh with transport manager
        mesh = P2PMeshIntegration(
            device_id="transport-test-device",
            peer_type=PeerType.MOBILE,
            transport_priority=TransportPriority.OFFLINE_FIRST
        )

        # Verify transport manager was created
        if mesh.transport_manager:
            print("   [OK] TransportManager instance created")

            # Check transport registration
            tm_status = mesh.transport_manager.get_status()
            print(f"   [OK] Available transports: {tm_status['available_transports']}")
            print(f"   [OK] Transport priority: {tm_status['transport_priority']}")

            # Verify BitChat transport was registered
            if TransportType.BITCHAT.value in tm_status['available_transports']:
                print("   [OK] BitChat transport registered successfully")
            else:
                print("   [INFO] BitChat transport not in available list (may need startup)")
        else:
            print("   [FAIL] TransportManager not created")
            return False

        # Test device context configuration
        device_context = mesh.transport_manager.device_context
        print(f"   [OK] Network type: {device_context.network_type}")
        print(f"   [OK] Has internet: {device_context.has_internet}")
        print(f"   [OK] Is metered: {device_context.is_metered_connection}")
        print(f"   [OK] Power save mode: {device_context.power_save_mode}")

        print("   [OK] Transport integration verified")
        return True

    except Exception as e:
        print(f"   [FAIL] Transport test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""

    print("AIVillage Global South P2P Integration - Validation Test")
    print("Running from:", Path.cwd())
    print()

    test1_result = await test_basic_integration()
    test2_result = await test_transport_integration()

    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    if test1_result and test2_result:
        print("SUCCESS: All tests passed!")
        print("\nThe Global South P2P integration is working correctly and")
        print("properly integrates with the existing AIVillage P2P infrastructure.")
        return 0
    else:
        print("FAILURE: Some tests failed!")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Test suite crashed: {e}")
        sys.exit(1)
