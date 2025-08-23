"""Instrumentation-style tests for the Android LibP2P bindings.

These tests spin up two :class:`LibP2PMeshNetwork` instances and verify that a
message sent between them is correctly received.  The tests are written using
``pytest`` and can be executed against Android emulators running the Python
runtime, but they also run in a standard Python environment which makes them
useful for CI.
"""

from __future__ import annotations

import asyncio
import os

import pytest

if os.environ.get("ANDROID_EMULATOR", "0") != "1":  # pragma: no cover - CI env
    pytest.skip("Android emulator not available", allow_module_level=True)

from src.android.p2p.libp2p_mesh import LibP2PMeshNetwork, MeshConfiguration, MeshMessage

pytestmark = pytest.mark.asyncio


async def _setup_mesh(node_id: str, port: int, bootstrap: list[str] | None = None):
    config = MeshConfiguration(node_id=node_id, listen_port=port)
    mesh = LibP2PMeshNetwork(config)
    await mesh.connect(bootstrap_peers=bootstrap)
    return mesh


async def test_peer_discovery_and_message_exchange() -> None:
    """Ensure two peers can discover each other and exchange a message."""

    mesh_a = await _setup_mesh("node-a", 4101)
    addr_a = f"/ip4/127.0.0.1/tcp/4101/p2p/{mesh_a.node_id}"

    mesh_b = await _setup_mesh("node-b", 4102, bootstrap=[addr_a])

    # Send a message from B to A
    message = MeshMessage(sender="node-b", recipient="node-a", payload=b"hello")
    await mesh_b.send_message(addr_a, message)

    # Listen for the incoming message on node A
    listener = mesh_a.listen()
    received = await asyncio.wait_for(anext(listener), timeout=5)

    assert received.payload == b"hello"
    assert received.sender == "node-b"

    await mesh_a.close()
    await mesh_b.close()
