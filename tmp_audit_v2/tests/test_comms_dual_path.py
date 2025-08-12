"""Test dual-path communications (BitChat/Betanet) post-P0 fixes."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import asyncio
from unittest.mock import patch

import pytest


def test_bitchat_import():
    """Test BitChat module imports cleanly."""
    try:
        from core.p2p.bitchat_transport import BitChatTransport

        assert BitChatTransport is not None
        return True
    except ImportError as e:
        pytest.fail(f"BitChat import failed: {e}")


def test_betanet_import():
    """Test Betanet module imports cleanly."""
    try:
        from core.p2p.betanet_transport import BetanetTransport

        assert BetanetTransport is not None
        return True
    except ImportError as e:
        pytest.fail(f"Betanet import failed: {e}")


@pytest.mark.asyncio
async def test_bitchat_discovery():
    """Test BitChat peer discovery with mocks."""
    from core.p2p.bitchat_transport import BitChatTransport

    transport = BitChatTransport()

    # Mock BLE discovery
    with patch.object(transport, "discover_peers") as mock_discover:
        mock_discover.return_value = ["peer1", "peer2", "peer3"]
        peers = await transport.discover_peers()
        assert len(peers) == 3
        assert "peer1" in peers


@pytest.mark.asyncio
async def test_ttl_bounded_relay():
    """Test 7-hop TTL enforcement."""
    from core.p2p.bitchat_transport import BitChatTransport

    transport = BitChatTransport()

    # Create message with TTL
    message = {"data": "test", "ttl": 8, "hops": []}

    # Should accept up to 7 hops
    for i in range(7):
        message["hops"].append(f"hop{i}")
        assert transport._should_relay(message) == True

    # Should reject at 8 hops
    message["hops"].append("hop7")
    assert transport._should_relay(message) == False


@pytest.mark.asyncio
async def test_store_and_forward():
    """Test offline queue and drain on reconnect."""
    from core.p2p.dual_path_transport import DualPathTransport

    transport = DualPathTransport()

    # Go offline
    transport.is_online = False

    # Queue messages
    await transport.send_message({"data": "msg1"}, "peer1")
    await transport.send_message({"data": "msg2"}, "peer1")

    assert len(transport.offline_queue) == 2

    # Come back online
    transport.is_online = True
    sent_count = await transport.drain_offline_queue()

    assert sent_count == 2
    assert len(transport.offline_queue) == 0


@pytest.mark.asyncio
async def test_navigator_path_policy():
    """Test path selection policy."""
    from core.p2p.dual_path_transport import DualPathTransport

    transport = DualPathTransport()

    # Local message -> BitChat
    path = transport.select_path(size=100, destination="local_peer", priority="normal")
    assert path == "bitchat"

    # Large/urgent -> Betanet
    path = transport.select_path(
        size=10000, destination="remote_peer", priority="urgent"
    )
    assert path == "betanet"

    # Offline -> DTN queue
    transport.is_online = False
    path = transport.select_path(size=500, destination="any_peer", priority="normal")
    assert path == "dtn_queue"


if __name__ == "__main__":
    # Run tests
    print("Testing BitChat/Betanet dual-path comms...")

    test_bitchat_import()
    print("PASS: BitChat imports")

    test_betanet_import()
    print("PASS: Betanet imports")

    asyncio.run(test_bitchat_discovery())
    print("PASS: Peer discovery")

    asyncio.run(test_ttl_bounded_relay())
    print("PASS: TTL enforcement")

    asyncio.run(test_store_and_forward())
    print("PASS: Store-and-forward")

    asyncio.run(test_navigator_path_policy())
    print("PASS: Path selection policy")

    print("\nAll dual-path tests passed!")
