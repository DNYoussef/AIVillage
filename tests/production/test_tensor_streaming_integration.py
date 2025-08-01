"""Integration tests for tensor streaming over P2P nodes."""

import hashlib
import numpy as np
import pytest

from src.production.communications.p2p.p2p_node import P2PNode, P2PMessage, MessageType
from src.production.communications.p2p.tensor_streaming import TensorStreaming


@pytest.mark.asyncio
async def test_tensor_stream_round_trip(monkeypatch):
    sender = P2PNode(node_id="sender", port=9101)
    receiver = P2PNode(node_id="receiver", port=9102)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)

    # Handle metadata messages as regular tensor chunk messages
    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def fake_send_message(self, peer_id, message_type, payload):
        assert peer_id == receiver.node_id
        msg = P2PMessage(
            message_type=message_type,
            sender_id=self.node_id,
            receiver_id=peer_id,
            payload=payload,
        )
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    # Bind the helper as a method on the sender node
    monkeypatch.setattr(sender, "send_message", fake_send_message.__get__(sender, P2PNode))

    tensor = np.arange(8, dtype=np.float32)
    tensor_id = await send_stream.send_tensor(tensor, "test", receiver.node_id)

    assert tensor_id in recv_stream.pending_chunks
    assert tensor_id in recv_stream.tensor_metadata

    for chunk in recv_stream.pending_chunks[tensor_id].values():
        assert hashlib.md5(chunk.data).hexdigest() == chunk.checksum

    reconstructed = await recv_stream._reconstruct_tensor(tensor_id)
    metadata = recv_stream.tensor_metadata[tensor_id]
    assert np.array_equal(reconstructed, tensor)
    assert metadata.tensor_id == tensor_id


@pytest.mark.asyncio
async def test_missing_chunk_failure(monkeypatch):
    sender = P2PNode(node_id="sender2", port=9111)
    receiver = P2PNode(node_id="receiver2", port=9112)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)
    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def fake_send(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    monkeypatch.setattr(sender, "send_message", fake_send.__get__(sender, P2PNode))

    tensor = np.arange(6, dtype=np.float32)
    tensor_id = await send_stream.send_tensor(tensor, "missing", receiver.node_id)

    recv_stream.pending_chunks[tensor_id].pop(0)
    result = await recv_stream._reconstruct_tensor(tensor_id)
    assert result is None


@pytest.mark.asyncio
async def test_corrupted_chunk_failure(monkeypatch):
    sender = P2PNode(node_id="sender3", port=9121)
    receiver = P2PNode(node_id="receiver3", port=9122)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)
    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def fake_send(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    monkeypatch.setattr(sender, "send_message", fake_send.__get__(sender, P2PNode))

    tensor = np.arange(6, dtype=np.float32)
    tensor_id = await send_stream.send_tensor(tensor, "corrupt", receiver.node_id)

    chunk = recv_stream.pending_chunks[tensor_id][0]
    chunk.data = b"x" + chunk.data[1:]

    with pytest.raises(RuntimeError):
        await recv_stream._reconstruct_tensor(tensor_id)

