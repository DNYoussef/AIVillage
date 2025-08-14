"""Test chunk re-request functionality in tensor streaming."""

import asyncio
import types

import numpy as np
import pytest

# Mock the P2PNode and related imports for testing
from src.production.communications.p2p.tensor_streaming import (
    CompressionType,
    MessageType,
    P2PMessage,
    StreamingConfig,
    TensorStreaming,
)


class MockP2PNode:
    """Mock P2P node for testing."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.message_handlers = {}
        self.sent_messages = []
        self.should_fail_send = False

    def register_handler(self, message_type, handler):
        self.message_handlers[message_type] = handler

    async def send_message(self, peer_id, message_type, payload):
        if self.should_fail_send:
            return False

        self.sent_messages.append(
            {
                "peer_id": peer_id,
                "message_type": message_type,
                "payload": payload,
            }
        )
        return True


@pytest.fixture
def mock_nodes():
    """Create mock P2P nodes for testing."""
    sender = MockP2PNode("sender_node")
    receiver = MockP2PNode("receiver_node")
    return sender, receiver


@pytest.fixture
def streaming_instances(mock_nodes):
    """Create tensor streaming instances."""
    sender, receiver = mock_nodes

    config = StreamingConfig(
        chunk_size=1024,  # Small chunks for testing
        compression=CompressionType.NONE,  # No compression for simplicity
        max_retries=2,
        bandwidth_limit_kbps=None,
    )

    sender_stream = TensorStreaming(sender, config)
    receiver_stream = TensorStreaming(receiver, config)

    return sender_stream, receiver_stream, sender, receiver


@pytest.mark.asyncio
async def test_chunk_caching_during_send(streaming_instances):
    """Test that chunks are cached during sending for re-request capability."""
    sender_stream, _, sender_node, receiver_node = streaming_instances

    # Create test tensor
    test_tensor = np.random.random((10, 10)).astype(np.float32)

    # Send tensor
    tensor_id = await sender_stream.send_tensor(
        test_tensor, "test_tensor", receiver_node.node_id
    )

    # Verify that sent chunks are cached
    assert hasattr(sender_stream, "_sent_tensor_cache")
    assert tensor_id in sender_stream._sent_tensor_cache

    cached_chunks = sender_stream._sent_tensor_cache[tensor_id]
    assert len(cached_chunks) > 0

    # Verify all chunks are cached
    metadata = sender_stream.tensor_metadata.get(tensor_id)
    if metadata:
        assert len(cached_chunks) == metadata.total_chunks


@pytest.mark.asyncio
async def test_chunk_re_request_handling(streaming_instances):
    """Test handling of chunk re-request messages."""
    sender_stream, receiver_stream, sender_node, receiver_node = streaming_instances

    # Create and send test tensor to populate cache
    test_tensor = np.random.random((5, 5)).astype(np.float32)
    tensor_id = await sender_stream.send_tensor(
        test_tensor, "test_tensor", receiver_node.node_id
    )

    # Simulate chunk re-request
    request_payload = {
        "action": "request_chunks",
        "tensor_id": tensor_id,
        "chunk_indices": [0, 1],  # Request first two chunks
    }

    request_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=receiver_node.node_id,
        receiver_id=sender_node.node_id,
        payload=request_payload,
    )

    # Clear previous messages
    sender_node.sent_messages.clear()

    # Handle the chunk request
    await sender_stream._handle_tensor_chunk(request_message)

    # Verify that chunks were re-sent
    chunk_messages = [
        msg
        for msg in sender_node.sent_messages
        if msg["message_type"] == MessageType.TENSOR_CHUNK
    ]

    assert len(chunk_messages) == 2  # Should have re-sent 2 chunks

    # Verify the correct chunks were sent
    sent_indices = {msg["payload"]["chunk_index"] for msg in chunk_messages}
    assert sent_indices == {0, 1}


@pytest.mark.asyncio
async def test_missing_tensor_in_cache(streaming_instances):
    """Test handling when requested tensor is not in cache."""
    sender_stream, _, sender_node, receiver_node = streaming_instances

    # Request chunks for non-existent tensor
    request_payload = {
        "action": "request_chunks",
        "tensor_id": "non-existent-tensor-id",
        "chunk_indices": [0, 1],
    }

    request_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=receiver_node.node_id,
        receiver_id=sender_node.node_id,
        payload=request_payload,
    )

    # Clear previous messages
    sender_node.sent_messages.clear()

    # Handle the chunk request
    await sender_stream._handle_tensor_chunk(request_message)

    # Should not send any chunks
    chunk_messages = [
        msg
        for msg in sender_node.sent_messages
        if msg["message_type"] == MessageType.TENSOR_CHUNK
    ]

    assert len(chunk_messages) == 0


@pytest.mark.asyncio
async def test_missing_chunks_in_cache(streaming_instances):
    """Test handling when some requested chunks are missing from cache."""
    sender_stream, _, sender_node, receiver_node = streaming_instances

    # Manually create partial cache
    tensor_id = "test-tensor-partial"
    sender_stream._sent_tensor_cache = {
        tensor_id: {
            0: types.SimpleNamespace(
                tensor_id=tensor_id,
                chunk_index=0,
                total_chunks=3,
                data=b"chunk0",
                checksum="abc123",
            ),
            # Missing chunk 1
            2: types.SimpleNamespace(
                tensor_id=tensor_id,
                chunk_index=2,
                total_chunks=3,
                data=b"chunk2",
                checksum="def456",
            ),
        }
    }

    # Request all chunks including missing one
    request_payload = {
        "action": "request_chunks",
        "tensor_id": tensor_id,
        "chunk_indices": [0, 1, 2],
    }

    request_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=receiver_node.node_id,
        receiver_id=sender_node.node_id,
        payload=request_payload,
    )

    # Clear previous messages
    sender_node.sent_messages.clear()

    # Handle the chunk request
    await sender_stream._handle_tensor_chunk(request_message)

    # Should only send available chunks (0 and 2)
    chunk_messages = [
        msg
        for msg in sender_node.sent_messages
        if msg["message_type"] == MessageType.TENSOR_CHUNK
    ]

    assert len(chunk_messages) == 2
    sent_indices = {msg["payload"]["chunk_index"] for msg in chunk_messages}
    assert sent_indices == {0, 2}


@pytest.mark.asyncio
async def test_chunk_re_request_with_failures(streaming_instances):
    """Test chunk re-request when some sends fail."""
    sender_stream, _, sender_node, receiver_node = streaming_instances

    # Create and send test tensor to populate cache
    test_tensor = np.random.random((3, 3)).astype(np.float32)
    tensor_id = await sender_stream.send_tensor(
        test_tensor, "test_tensor", receiver_node.node_id
    )

    # Make sending fail
    sender_node.should_fail_send = True

    # Request chunks
    request_payload = {
        "action": "request_chunks",
        "tensor_id": tensor_id,
        "chunk_indices": [0],
    }

    request_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=receiver_node.node_id,
        receiver_id=sender_node.node_id,
        payload=request_payload,
    )

    # Clear previous messages
    sender_node.sent_messages.clear()

    # Handle the chunk request (should not crash even if sends fail)
    await sender_stream._handle_tensor_chunk(request_message)

    # No messages should be recorded since sends failed
    assert len(sender_node.sent_messages) == 0


@pytest.mark.asyncio
async def test_end_to_end_chunk_re_request_scenario(streaming_instances):
    """Test complete end-to-end chunk re-request scenario."""
    sender_stream, receiver_stream, sender_node, receiver_node = streaming_instances

    # Create test tensor
    test_tensor = np.random.random((8, 8)).astype(np.float32)

    # 1. Send tensor from sender to receiver
    tensor_id = await sender_stream.send_tensor(
        test_tensor, "test_tensor", receiver_node.node_id
    )

    # 2. Simulate that receiver got metadata but missed some chunks
    # Manually add metadata to receiver
    metadata_payload = {
        "action": "tensor_metadata",
        "metadata": {
            "tensor_id": tensor_id,
            "name": "test_tensor",
            "shape": [8, 8],
            "dtype": "float32",
            "size_bytes": 256,
            "total_chunks": 3,
            "compression": "none",
            "format": "numpy",
            "checksum": "test_checksum",
            "timestamp": 123456789.0,
            "source_node": sender_node.node_id,
            "tags": {},
            "device": None,
            "is_torch": False,
            "requires_grad": False,
        },
    }

    metadata_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=sender_node.node_id,
        receiver_id=receiver_node.node_id,
        payload=metadata_payload,
    )

    await receiver_stream._handle_tensor_chunk(metadata_message)

    # 3. Add some chunks to receiver but not all
    chunk_payload = {
        "tensor_id": tensor_id,
        "chunk_index": 1,
        "total_chunks": 3,
        "data": "deadbeef",  # Dummy hex data
        "checksum": "test_checksum",
        "timestamp": 123456789.0,
        "is_compressed": False,
        "compression_type": None,
    }

    chunk_message = P2PMessage(
        message_type=MessageType.DATA,
        sender_id=sender_node.node_id,
        receiver_id=receiver_node.node_id,
        payload=chunk_payload,
    )

    await receiver_stream._handle_tensor_chunk(chunk_message)

    # 4. Receiver requests missing chunks
    success = await receiver_stream.request_missing_chunks(
        tensor_id, sender_node.node_id
    )

    assert success is True

    # 5. Verify request was sent
    request_messages = [
        msg
        for msg in receiver_node.sent_messages
        if msg["payload"].get("action") == "request_chunks"
    ]

    assert len(request_messages) == 1
    request_payload = request_messages[0]["payload"]
    assert request_payload["tensor_id"] == tensor_id
    # Should request chunks 0 and 2 (missing chunks)
    assert set(request_payload["chunk_indices"]) == {0, 2}


@pytest.mark.asyncio
async def test_cache_initialization():
    """Test that cache is properly initialized."""
    config = StreamingConfig()
    node = MockP2PNode("test_node")
    stream = TensorStreaming(node, config)

    # Cache should not exist initially
    assert not hasattr(stream, "_sent_tensor_cache")

    # After sending a tensor, cache should be initialized
    test_tensor = np.array([[1, 2], [3, 4]], dtype=np.float32)
    await stream.send_tensor(test_tensor, "test", "peer")

    assert hasattr(stream, "_sent_tensor_cache")
    assert isinstance(stream._sent_tensor_cache, dict)


if __name__ == "__main__":
    # Run a simple test
    async def main():
        print("Testing chunk re-request functionality...")

        # Create mock nodes
        sender = MockP2PNode("sender")
        receiver = MockP2PNode("receiver")

        # Create streaming instances
        config = StreamingConfig(chunk_size=512, compression=CompressionType.NONE)
        sender_stream = TensorStreaming(sender, config)
        receiver_stream = TensorStreaming(receiver, config)

        # Test tensor
        test_tensor = np.random.random((10, 10)).astype(np.float32)

        # Send tensor
        tensor_id = await sender_stream.send_tensor(
            test_tensor, "test_tensor", "receiver"
        )

        print(f"Sent tensor with ID: {tensor_id}")
        print(
            f"Cached chunks: {len(sender_stream._sent_tensor_cache.get(tensor_id, {}))}"
        )

        # Test chunk re-request
        request_payload = {
            "action": "request_chunks",
            "tensor_id": tensor_id,
            "chunk_indices": [0, 1],
        }

        request_message = P2PMessage(
            message_type=MessageType.DATA,
            sender_id="receiver",
            receiver_id="sender",
            payload=request_payload,
        )

        sender.sent_messages.clear()
        await sender_stream._handle_tensor_chunk(request_message)

        chunk_messages = [
            msg
            for msg in sender.sent_messages
            if msg["message_type"] == MessageType.TENSOR_CHUNK
        ]

        print(f"Re-sent {len(chunk_messages)} chunks")
        print("Chunk re-request test completed successfully!")

    asyncio.run(main())
