import asyncio
import json
import time

import numpy as np
import pytest

from src.infrastructure.p2p.tensor_streaming import TensorStreamer
from src.production.communications.p2p.p2p_node import MessageType, P2PNode


def test_safe_serialization_round_trip():
    streamer = TensorStreamer()
    array = np.arange(10, dtype=np.float32)
    data = streamer._serialize_tensor(array)
    restored = streamer._deserialize_tensor(data)
    assert np.array_equal(array, restored)


@pytest.mark.asyncio
async def test_receive_message_missing_field_rejected():
    node = P2PNode()
    reader = asyncio.StreamReader()
    message = {
        "message_type": MessageType.DATA.value,
        "sender_id": "sender",
        "receiver_id": "receiver",
        "payload": {},
        "timestamp": time.time(),
        # Missing "message_id"
    }
    encrypted = node.cipher.encrypt(json.dumps(message).encode())
    reader.feed_data(len(encrypted).to_bytes(4, "big"))
    reader.feed_data(encrypted)
    reader.feed_eof()
    assert await node._receive_message_from_reader(reader) is None


@pytest.mark.asyncio
async def test_receive_message_unexpected_field_rejected():
    node = P2PNode()
    reader = asyncio.StreamReader()
    message = {
        "message_type": MessageType.DATA.value,
        "sender_id": "sender",
        "receiver_id": "receiver",
        "payload": {},
        "timestamp": time.time(),
        "message_id": "123",
        "extra": "nope",
    }
    encrypted = node.cipher.encrypt(json.dumps(message).encode())
    reader.feed_data(len(encrypted).to_bytes(4, "big"))
    reader.feed_data(encrypted)
    reader.feed_eof()
    assert await node._receive_message_from_reader(reader) is None
