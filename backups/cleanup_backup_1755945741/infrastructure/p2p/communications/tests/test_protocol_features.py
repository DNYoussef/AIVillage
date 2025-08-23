import asyncio

import pytest
from packages.p2p.communications.protocol import CommunicationsProtocol
from packages.p2p.communications.service_directory import service_directory


@pytest.mark.asyncio
async def test_reconnect_and_queue_drain(monkeypatch):
    a = CommunicationsProtocol("a", port=43101, heartbeat_interval=0.1)
    b = CommunicationsProtocol("b", port=43102, heartbeat_interval=0.1)
    await a.start_server()
    service_directory.register("b", "ws://localhost:43102/ws")
    # Attempt to connect before b is online
    assert not await a.connect("b")
    await a.send_message("b", {"type": "direct", "content": "hi"})
    assert "b" in a.pending_messages
    await b.start_server()
    await a._reconnect("b")
    await asyncio.sleep(0.2)
    assert "b" not in a.pending_messages or not a.pending_messages["b"]
    history = b.get_message_history("a")
    assert history and history[0]["message"]["content"] == "hi"
    await a.stop_server()
    await b.stop_server()


@pytest.mark.asyncio
async def test_broadcast_fanout():
    a = CommunicationsProtocol("a", port=43103, heartbeat_interval=0.1)
    b = CommunicationsProtocol("b", port=43104, heartbeat_interval=0.1)
    c = CommunicationsProtocol("c", port=43105, heartbeat_interval=0.1)
    await asyncio.gather(a.start_server(), b.start_server(), c.start_server())
    await asyncio.gather(a.connect("b"), a.connect("c"))
    await a.broadcast_message({"type": "direct", "content": "hello"})
    await asyncio.sleep(0.1)
    assert b.get_message_history("a")[0]["message"]["content"] == "hello"
    assert c.get_message_history("a")[0]["message"]["content"] == "hello"
    await asyncio.gather(a.stop_server(), b.stop_server(), c.stop_server())


@pytest.mark.asyncio
async def test_rpc_round_trip():
    a = CommunicationsProtocol("a", port=43106, heartbeat_interval=0.1)
    b = CommunicationsProtocol("b", port=43107, heartbeat_interval=0.1)
    await asyncio.gather(a.start_server(), b.start_server())
    await a.connect("b")

    async def handler(_aid, message):
        await b.send_rpc_response("a", message["correlation_id"], {"ok": True})

    b.register_handler("rpc_request", handler)
    resp = await a.rpc("b", {"ping": 1}, timeout=2.0)
    assert resp and resp["payload"] == {"ok": True}
    await asyncio.gather(a.stop_server(), b.stop_server())


@pytest.mark.asyncio
async def test_heartbeat_timeout():
    a = CommunicationsProtocol("a", port=43108, heartbeat_interval=0.05, heartbeat_miss_limit=1)
    b = CommunicationsProtocol("b", port=43109, heartbeat_interval=0.05, heartbeat_miss_limit=1)
    await asyncio.gather(a.start_server(), b.start_server())
    await a.connect("b")
    await asyncio.sleep(0.1)
    await b.stop_server()
    await asyncio.sleep(0.2)
    assert not a.is_connected("b")
    await a.stop_server()
