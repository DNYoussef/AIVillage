import argparse
import pytest
from .encryption_layer import EncryptionLayer

parser = argparse.ArgumentParser(description="Encryption layer test options")
parser.add_argument("--simulate", action="store_true", help="enable simulated mode")
ARGS, _ = parser.parse_known_args()


@pytest.mark.asyncio
async def test_encrypt_decrypt_roundtrip():
    layer = EncryptionLayer("node-a")
    await layer.initialize()
    message = "secret"
    encrypted = await layer.encrypt_message(message, recipient_id="peer1")
    assert isinstance(encrypted, bytes)
    assert encrypted != message.encode()
    decrypted = await layer.decrypt_message(encrypted, sender_id="peer1")
    assert decrypted == message
    await layer.shutdown()
