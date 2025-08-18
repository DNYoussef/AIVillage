import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from federation.core.device_registry import DeviceRole
from federation.core.federation_manager import FederationManager, PrivacyLevel


@pytest.mark.asyncio
async def test_build_privacy_circuit_generates_keys():
    manager = FederationManager("test_node")

    # create mock relay devices
    mock_relays = []
    for i in range(2):
        relay = MagicMock()
        relay.identity.device_id = f"relay_{i}"
        relay.role = DeviceRole.RELAY
        mock_relays.append(relay)

    with patch.object(manager.device_registry, "get_devices_by_role", return_value=mock_relays):
        circuit = await manager._build_privacy_circuit("dest", min_hops=3)

    assert circuit is not None
    assert len(circuit) == 3
    for hop in circuit:
        assert set(hop.keys()) == {"node", "key", "protocol"}
        assert hop["key"]


@pytest.mark.asyncio
async def test_send_federated_message_routes_by_protocol():
    manager = FederationManager("test_node", enable_tor=True, enable_i2p=True)
    manager.is_running = True

    manager.dual_path_transport = MagicMock()
    manager.dual_path_transport.send_message = AsyncMock(return_value=True)
    manager.tor_transport = MagicMock()
    manager.tor_transport.send_message = AsyncMock(return_value=True)
    manager.i2p_transport = MagicMock()
    manager.i2p_transport.send_message = AsyncMock(return_value=True)

    tunnel_id = "t1"
    manager.active_tunnels[tunnel_id] = {
        "destination": "dest",
        "circuit_path": [
            {"node": "tor_hop", "key": "aa", "protocol": "tor"},
            {"node": "i2p_hop", "key": "bb", "protocol": "i2p"},
            {"node": "dest", "key": "cc", "protocol": "bitchat"},
        ],
        "privacy_level": PrivacyLevel.ANONYMOUS,
        "created_at": time.time(),
        "last_used": time.time(),
    }

    async def fake_create_privacy_tunnel(*args, **kwargs):
        return tunnel_id

    manager.create_privacy_tunnel = fake_create_privacy_tunnel

    payload = {"hello": "world"}
    success = await manager.send_federated_message("dest", payload, privacy_level=PrivacyLevel.ANONYMOUS)

    assert success
    manager.tor_transport.send_message.assert_called_once()
    manager.i2p_transport.send_message.assert_called_once()
    manager.dual_path_transport.send_message.assert_called_once()
