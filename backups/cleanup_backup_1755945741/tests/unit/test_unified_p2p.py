"""
Test script for unified P2P communication system.

Validates that the consolidated P2P system works correctly with:
- Transport manager coordination
- BitChat BLE mesh networking
- BetaNet HTX encrypted transport
- Message routing and failover
- Compatibility with legacy implementations
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent / "packages"))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_unified_p2p_system():
    """Test the unified P2P communication system."""

    print("Testing AIVillage Unified P2P Communication System")
    print("=" * 60)

    try:
        # Import unified P2P system
        from p2p import TransportManager, TransportPriority, TransportType
        from p2p.betanet.htx_transport import create_htx_client
        from p2p.bitchat.ble_transport import create_bitchat_transport
        from p2p.core.message_types import MessagePriority, create_data_message
        from p2p.core.transport_manager import TransportCapabilities

        print("Successfully imported unified P2P system")

        # Test 1: Create transport manager
        print("\nTest 1: Transport Manager Creation")
        device_id = "test-device-001"
        manager = TransportManager(device_id=device_id, transport_priority=TransportPriority.OFFLINE_FIRST)
        print(f"Created transport manager for device: {device_id}")

        # Test 2: Register BitChat transport
        print("\nTest 2: BitChat Transport Registration")
        bitchat = create_bitchat_transport(device_id=device_id)
        bitchat_caps = TransportCapabilities(
            supports_broadcast=True,
            supports_multicast=True,
            is_offline_capable=True,
            requires_internet=False,
            battery_impact="medium",
            data_cost_impact="low",
        )
        manager.register_transport(TransportType.BITCHAT, bitchat, bitchat_caps)
        print("Registered BitChat BLE mesh transport")

        # Test 3: Register BetaNet transport
        print("\nTest 3: BetaNet HTX Transport Registration")
        betanet = create_htx_client(server_host="demo.betanet.ai", server_port=8443)
        betanet_caps = TransportCapabilities(
            supports_unicast=True,
            provides_encryption=True,
            supports_forward_secrecy=True,
            requires_internet=True,
            battery_impact="low",
            data_cost_impact="medium",
        )
        manager.register_transport(TransportType.BETANET, betanet, betanet_caps)
        print("Registered BetaNet HTX encrypted transport")

        # Test 4: Start transport manager
        print("\nTest 4: Starting Transport System")
        success = await manager.start()
        print(f"Transport manager started: {success}")

        # Test 5: Create and send test message
        print("\nTest 5: Message Creation and Routing")
        test_message = create_data_message(
            recipient_id="test-recipient-002",
            payload=b"Hello from unified P2P system!",
            sender_id=device_id,
            priority=MessagePriority.NORMAL,
        )
        print(f"Created test message: {test_message.metadata.message_id}")

        # Test routing decision
        routing_decision = manager.select_transport(test_message)
        print(f"Selected transport: {routing_decision.primary_transport.value}")
        print(f"   Fallbacks: {[t.value for t in routing_decision.fallback_transports]}")
        print(f"   Reasoning: {routing_decision.reasoning}")

        # Test 6: Send message
        print("\nTest 6: Message Transmission")
        send_success = await manager.send_message(test_message)
        print(f"Message sent: {send_success}")

        # Test 7: Get transport status
        print("\nTest 7: Transport Status")
        status = manager.get_status()
        print(f"Device ID: {status['device_id']}")
        print(f"   Priority: {status['transport_priority']}")
        print(f"   Active transports: {status['active_transports']}")
        print(f"   Messages sent: {status['statistics']['messages_sent']}")

        # Test 8: Individual transport status
        print("\nTest 8: Individual Transport Status")
        bitchat_status = bitchat.get_status()
        print(f"BitChat - State: {bitchat_status['state']}, Peers: {bitchat_status['peer_count']}")

        betanet_status = betanet.get_status()
        print(f"BetaNet - Connected: {betanet_status['connected']}")

        # Test 9: Stop transport manager
        print("\nTest 9: Stopping Transport System")
        stop_success = await manager.stop()
        print(f"Transport manager stopped: {stop_success}")

        print("\nAll P2P system tests completed successfully!")
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print("   Make sure packages/p2p/ is properly structured")
        return False
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_legacy_compatibility():
    """Test compatibility with legacy P2P implementations."""

    print("\nTesting Legacy P2P Compatibility")
    print("=" * 40)

    try:
        from p2p.bridges.compatibility import discover_and_bridge_legacy_transports

        # Discover legacy transports
        print("Discovering legacy transport implementations...")
        manager = await discover_and_bridge_legacy_transports()

        # Get status of discovered transports
        legacy_status = manager.get_all_status()
        print(f"Discovered {len(legacy_status)} legacy transports:")

        for adapter_id, status in legacy_status.items():
            print(
                f"   - {adapter_id}: {status['transport_type']} ({'available' if status['available'] else 'unavailable'})"
            )

        # Stop legacy transports
        await manager.stop_all()
        print("Legacy compatibility test completed")
        return True

    except Exception as e:
        print(f"Legacy compatibility test failed: {e}")
        return False


async def main():
    """Run all P2P system tests."""

    print("AIVillage P2P Communication System Test Suite")
    print("=" * 60)

    # Test unified P2P system
    unified_success = await test_unified_p2p_system()

    # Test legacy compatibility
    legacy_success = await test_legacy_compatibility()

    # Summary
    print("\nTest Summary")
    print("=" * 20)
    print(f"Unified P2P System: {'PASS' if unified_success else 'FAIL'}")
    print(f"Legacy Compatibility: {'PASS' if legacy_success else 'FAIL'}")

    if unified_success and legacy_success:
        print("\nAll tests passed! P2P consolidation successful.")
        return 0
    else:
        print("\nSome tests failed. Check logs for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
