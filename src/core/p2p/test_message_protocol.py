import asyncio
import json
import pytest

from .message_protocol import MessageProtocol, EvolutionMessage, MessageType


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
    msg = EvolutionMessage(message_id="1", message_type=MessageType.PING, sender_id="node", requires_ack=True)
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
