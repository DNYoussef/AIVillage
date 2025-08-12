"""Stress test for global bandwidth throttling in tensor streaming."""

import asyncio
import time

import numpy as np
import pytest

from src.production.communications.p2p.p2p_node import (
    MessageType,
    P2PMessage,
    P2PNode,
)
from src.production.communications.p2p.tensor_streaming import (
    BandwidthController,
    CompressionType,
    StreamingConfig,
    TensorStreaming,
)


@pytest.mark.asyncio
async def test_concurrent_streams_respect_global_limit(monkeypatch):
    """Ensure combined transfers cannot exceed the configured limit."""
    BandwidthController.reset()

    receiver = P2PNode(node_id="receiver_bw", port=9400)
    recv_stream = TensorStreaming(
        node=receiver,
        config=StreamingConfig(
            bandwidth_limit_kbps=64, compression=CompressionType.NONE
        ),
    )

    sender1 = P2PNode(node_id="sender_bw1", port=9401)
    sender2 = P2PNode(node_id="sender_bw2", port=9402)
    stream1 = TensorStreaming(
        node=sender1,
        config=StreamingConfig(
            bandwidth_limit_kbps=64, compression=CompressionType.NONE
        ),
    )
    stream2 = TensorStreaming(
        node=sender2,
        config=StreamingConfig(
            bandwidth_limit_kbps=64, compression=CompressionType.NONE
        ),
    )

    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)
    sender1.register_handler(MessageType.DATA, stream1._handle_tensor_chunk)
    sender2.register_handler(MessageType.DATA, stream2._handle_tensor_chunk)

    async def fake_send(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    monkeypatch.setattr(sender1, "send_message", fake_send.__get__(sender1, P2PNode))
    monkeypatch.setattr(sender2, "send_message", fake_send.__get__(sender2, P2PNode))

    async def recv_send(self, peer_id, message_type, payload):
        target = {sender1.node_id: sender1, sender2.node_id: sender2}.get(peer_id)
        if not target:
            return False
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = target.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    monkeypatch.setattr(receiver, "send_message", recv_send.__get__(receiver, P2PNode))

    # 64KB tensor (float32 -> 4 bytes each)
    tensor = np.random.random(16384).astype(np.float32)

    start = time.time()
    await asyncio.gather(
        stream1.send_tensor(tensor, "t1", receiver.node_id),
        stream2.send_tensor(tensor, "t2", receiver.node_id),
    )
    elapsed = time.time() - start

    # With 64KB/s global limit and two simultaneous 64KB transfers, total
    # time should be at least one second as the second transfer waits for
    # the quota to reset.
    assert elapsed >= 0.9, f"Transfers finished too quickly: {elapsed}s"

    BandwidthController.reset()
