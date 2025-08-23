import time

from src.production.communications.p2p.p2p_node import P2PNode


def test_invalid_handshake_payload_rejected() -> None:
    node = P2PNode()
    message = {
        "message_type": "handshake",
        "sender_id": "a",
        "receiver_id": "b",
        "payload": {"capabilities": {}},
        "timestamp": time.time(),
        "message_id": "1",
    }
    assert node._validate_message_dict(message) is None


def test_valid_handshake_payload() -> None:
    node = P2PNode()
    message = {
        "message_type": "handshake",
        "sender_id": "a",
        "receiver_id": "b",
        "payload": {
            "node_id": "a",
            "capabilities": {},
            "timestamp": time.time(),
        },
        "timestamp": time.time(),
        "message_id": "3",
    }
    assert node._validate_message_dict(message) is not None


def test_invalid_tensor_chunk_payload_rejected() -> None:
    node = P2PNode()
    message = {
        "message_type": "tensor_chunk",
        "sender_id": "a",
        "receiver_id": "b",
        "payload": {"tensor_id": "1", "chunk_index": 0},
        "timestamp": time.time(),
        "message_id": "2",
    }
    assert node._validate_message_dict(message) is None
