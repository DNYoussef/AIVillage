"""Integration tests for tensor streaming over P2P nodes."""

import asyncio
import hashlib

import numpy as np
import pytest
import torch

from src.production.communications.p2p.p2p_node import MessageType, P2PMessage, P2PNode
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
    monkeypatch.setattr(
        sender, "send_message", fake_send_message.__get__(sender, P2PNode)
    )

    tensor = np.arange(8, dtype=np.float32)
    tensor_id = await send_stream.send_tensor(tensor, "test", receiver.node_id)

    assert tensor_id in recv_stream.pending_chunks
    assert tensor_id in recv_stream.tensor_metadata

    for chunk in recv_stream.pending_chunks[tensor_id].values():
        assert hashlib.md5(chunk.data).hexdigest() == chunk.checksum

    reconstructed, metadata = await recv_stream.receive_tensor(tensor_id)
    assert np.array_equal(reconstructed, tensor)
    assert metadata.tensor_id == tensor_id

    assert recv_stream.pending_chunks == {}
    assert recv_stream.active_transfers == {}
    assert recv_stream.tensor_metadata == {}


@pytest.mark.asyncio
async def test_tensor_stream_round_trip_torch(monkeypatch):
    sender = P2PNode(node_id="sender_t", port=9201)
    receiver = P2PNode(node_id="receiver_t", port=9202)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)

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

    monkeypatch.setattr(
        sender, "send_message", fake_send_message.__get__(sender, P2PNode)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.arange(8, dtype=torch.float32, device=device, requires_grad=True)
    tensor_id = await send_stream.send_tensor(tensor, "torch_test", receiver.node_id)

    reconstructed = await recv_stream._reconstruct_tensor(tensor_id)
    assert torch.equal(reconstructed, tensor)
    assert reconstructed.device.type == tensor.device.type
    assert reconstructed.dtype == tensor.dtype
    assert reconstructed.requires_grad == tensor.requires_grad


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


@pytest.mark.asyncio
async def test_key_exchange_after_restart(monkeypatch):
    sender = P2PNode(node_id="sender_dh", port=9301)
    receiver = P2PNode(node_id="receiver_dh", port=9302)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)

    sender.register_handler(MessageType.DATA, send_stream._handle_tensor_chunk)
    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def send_from_sender(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    async def send_from_receiver(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = sender.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    monkeypatch.setattr(
        sender, "send_message", send_from_sender.__get__(sender, P2PNode)
    )
    monkeypatch.setattr(
        receiver, "send_message", send_from_receiver.__get__(receiver, P2PNode)
    )

    await send_stream._ensure_key(receiver.node_id)
    first_key = send_stream._key_cache[receiver.node_id]

    # Restart receiver node and streaming
    receiver = P2PNode(node_id="receiver_dh", port=9302)
    recv_stream = TensorStreaming(node=receiver)
    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def send_from_receiver_new(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = sender.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    def make_send_from_sender(new_receiver):
        async def _send(self, peer_id, message_type, payload):
            msg = P2PMessage(message_type, self.node_id, peer_id, payload)
            handler = new_receiver.message_handlers.get(message_type)
            if handler:
                await handler(msg, None)
                return True
            return False

        return _send

    monkeypatch.setattr(
        sender, "send_message", make_send_from_sender(receiver).__get__(sender, P2PNode)
    )
    monkeypatch.setattr(
        receiver, "send_message", send_from_receiver_new.__get__(receiver, P2PNode)
    )

    await recv_stream._initiate_key_exchange(sender.node_id)

    second_key = send_stream._key_cache[receiver.node_id]
    assert second_key != first_key
    assert second_key == recv_stream._key_cache[sender.node_id]


@pytest.mark.asyncio
async def test_concurrent_transfers(monkeypatch):
    sender = P2PNode(node_id="sender_con", port=9401)
    receiver = P2PNode(node_id="receiver_con", port=9402)

    send_stream = TensorStreaming(node=sender)
    recv_stream = TensorStreaming(node=receiver)

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

    monkeypatch.setattr(
        sender, "send_message", fake_send_message.__get__(sender, P2PNode)
    )

    tensor1 = np.arange(8, dtype=np.float32)
    tensor2 = np.arange(16, dtype=np.float32)

    tensor_id1, tensor_id2 = await asyncio.gather(
        send_stream.send_tensor(tensor1, "test1", receiver.node_id),
        send_stream.send_tensor(tensor2, "test2", receiver.node_id),
    )

    results = await asyncio.gather(
        recv_stream.receive_tensor(tensor_id1),
        recv_stream.receive_tensor(tensor_id2),
    )

    rec1, _ = results[0]
    rec2, _ = results[1]

    assert np.array_equal(rec1, tensor1)
    assert np.array_equal(rec2, tensor2)

    assert recv_stream.pending_chunks == {}
    assert recv_stream.tensor_metadata == {}
    assert recv_stream.active_transfers == {}
