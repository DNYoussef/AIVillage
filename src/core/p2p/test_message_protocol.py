import argparse
import asyncio
import pytest
from .message_protocol import EvolutionMessage, MessageProtocol, MessageType

# CLI flag for simulation (unused but available)
parser = argparse.ArgumentParser(description="Message protocol test options")
parser.add_argument("--simulate", action="store_true", help="run protocol tests in simulation mode")
ARGS, _ = parser.parse_known_args()


class DummyWriter:
    """Minimal async writer capturing written data."""

    def __init__(self):
        self.data = b""

    def write(self, b: bytes):
        self.data += b

    async def drain(self):
        pass


def test_message_serialization_roundtrip():
    msg = EvolutionMessage(message_id="1", message_type=MessageType.PING, sender_id="a")
    restored = EvolutionMessage.from_dict(msg.to_dict())
    assert restored.message_id == msg.message_id
    assert restored.message_type == msg.message_type


def test_retry_logic():
    msg = EvolutionMessage(message_id="2", message_type=MessageType.PING, sender_id="a", max_retries=1)
    assert msg.should_retry()
    msg.retry_count = 1
    assert not msg.should_retry()


@pytest.mark.asyncio
async def test_send_tracks_message():
    protocol = MessageProtocol(None)
    writer = DummyWriter()
    msg = EvolutionMessage(message_id="3", message_type=MessageType.PING, sender_id="a")
    result = await protocol.send_message(msg, writer)
    assert result is True
    assert msg.message_id in protocol.sent_messages
