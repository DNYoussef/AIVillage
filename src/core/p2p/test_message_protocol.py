import asyncio
import time

import pytest

from .message_protocol import EvolutionMessage, MessageProtocol, MessageType


class DummyNode:
    def __init__(self) -> None:
        self.node_id = "node"
        self.peer_registry = {}
        self.connections = {}


class DummyWriter:
    def __init__(self) -> None:
        self.buffer = b""

    def write(self, data: bytes) -> None:
        self.buffer += data

    async def drain(self) -> None:  # pragma: no cover - no IO
        pass


class DummyReader:
    def __init__(self, data: bytes) -> None:
        self.data = data

    async def readexactly(self, n: int) -> bytes:
        if len(self.data) < n:
            raise asyncio.IncompleteReadError(partial=self.data, expected=n)
        result = self.data[:n]
        self.data = self.data[n:]
        return result


@pytest.mark.asyncio
async def test_send_message_tracks_retries() -> None:
    node = DummyNode()
    protocol = MessageProtocol(node)
    writer = DummyWriter()
    msg = EvolutionMessage(
        message_id="1",
        message_type=MessageType.PING,
        sender_id="node",
        requires_ack=True,
    )
    await protocol.send_message(msg, writer)
    queued = await protocol.retry_queue.get()
    assert queued.message_id == "1"
    assert protocol.stats["messages_sent"] == 1


@pytest.mark.asyncio
async def test_read_message_rejects_large() -> None:
    node = DummyNode()
    protocol = MessageProtocol(node)
    length = 1024 * 1024 + 1
    header = length.to_bytes(4, "big")
    data = b"x" * length
    reader = DummyReader(header + data)
    assert await protocol.read_message(reader) is None


@pytest.mark.asyncio
async def test_message_delivery_rate_and_latency() -> None:
    server_node = DummyNode()
    server_protocol = MessageProtocol(server_node)
    await server_protocol.start_protocol()

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            data = await server_protocol.read_message(reader)
            if data is None:
                break
            await server_protocol.handle_message(data, writer)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()

    client_node = DummyNode()
    client_protocol = MessageProtocol(client_node)
    await client_protocol.start_protocol()

    reader, writer = await asyncio.open_connection(host, port)

    async def client_reader() -> None:
        while True:
            data = await client_protocol.read_message(reader)
            if data is None:
                break
            await client_protocol.handle_message(data, writer)

    reader_task = asyncio.create_task(client_reader())

    messages = 50
    successes = 0
    latencies: list[float] = []

    for i in range(messages):
        msg = EvolutionMessage(
            message_id=str(i),
            message_type=MessageType.PING,
            sender_id=client_node.node_id,
        )
        fut = asyncio.get_event_loop().create_future()
        client_protocol.pending_responses[msg.message_id] = fut
        start = time.perf_counter()
        await client_protocol.send_message(msg, writer)
        try:
            await asyncio.wait_for(fut, timeout=1)
        except asyncio.TimeoutError:
            continue
        successes += 1
        latencies.append(time.perf_counter() - start)

    writer.close()
    await writer.wait_closed()
    reader_task.cancel()
    server.close()
    await server.wait_closed()
    await client_protocol.stop_protocol()
    await server_protocol.stop_protocol()

    success_rate = successes / messages
    avg_latency = sum(latencies) / len(latencies)

    assert success_rate >= 0.99
    assert avg_latency < 0.1
