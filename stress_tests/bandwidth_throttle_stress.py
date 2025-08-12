#!/usr/bin/env python3
"""Stress test for global TensorStreaming bandwidth throttling."""

from __future__ import annotations

import asyncio
import time

import numpy as np

from src.production.communications.p2p.p2p_node import MessageType, P2PMessage, P2PNode
from src.production.communications.p2p.tensor_streaming import (
    BandwidthController,
    CompressionType,
    StreamingConfig,
    TensorStreaming,
)


async def _setup_nodes(limit_kbps: int):
    BandwidthController.reset()
    sender = P2PNode(node_id="stress-sender", port=9201)
    receiver = P2PNode(node_id="stress-receiver", port=9202)
    config = StreamingConfig(
        chunk_size=100 * 1024,
        bandwidth_limit_kbps=limit_kbps,
        compression=CompressionType.NONE,
    )
    streams = [TensorStreaming(node=sender, config=config) for _ in range(3)]
    recv_stream = TensorStreaming(node=receiver)

    receiver.register_handler(MessageType.DATA, recv_stream._handle_tensor_chunk)

    async def fake_send(self, peer_id, message_type, payload):
        msg = P2PMessage(message_type, self.node_id, peer_id, payload)
        handler = receiver.message_handlers.get(message_type)
        if handler:
            await handler(msg, None)
            return True
        return False

    sender.send_message = fake_send.__get__(sender, P2PNode)
    return streams, receiver


def _make_tensors(count: int):
    return [np.zeros(100 * 1024, dtype=np.uint8) for _ in range(count)]


async def main() -> None:
    limit = 100  # KB/s shared across all streams
    streams, receiver = await _setup_nodes(limit)
    tensors = _make_tensors(len(streams))

    start = time.perf_counter()
    await asyncio.gather(
        *[
            stream.send_tensor(t, f"tensor{i}", receiver.node_id)
            for i, (stream, t) in enumerate(zip(streams, tensors))
        ]
    )
    elapsed = time.perf_counter() - start
    print(
        f"Transferred {len(tensors)} tensors of 100KB with {limit}KB/s shared limit in {elapsed:.2f}s"
    )


if __name__ == "__main__":
    asyncio.run(main())
