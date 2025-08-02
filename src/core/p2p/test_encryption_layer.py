import pytest

from .encryption_layer import EncryptionLayer


@pytest.mark.asyncio
async def test_encrypt_decrypt_roundtrip() -> None:
    layer = EncryptionLayer("node1")
    await layer.initialize()
    encrypted = await layer.encrypt_message("hello", recipient_id="peer1")
    decrypted = await layer.decrypt_message(encrypted, sender_id="peer1")
    assert decrypted == "hello"
    assert layer.stats["messages_encrypted"] == 1
    assert layer.stats["messages_decrypted"] == 1
    await layer.shutdown()


@pytest.mark.asyncio
async def test_replay_protection() -> None:
    layer = EncryptionLayer("node1")
    await layer.initialize()
    encrypted = await layer.encrypt_message("hi", recipient_id="peer1")
    await layer.decrypt_message(encrypted, sender_id="peer1")
    with pytest.raises(ValueError):
        await layer.decrypt_message(encrypted, sender_id="peer1")
    await layer.shutdown()
