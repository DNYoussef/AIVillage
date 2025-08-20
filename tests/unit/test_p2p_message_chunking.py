import pytest

from packages.p2p.core.message_types import MessageType, UnifiedMessage


def test_create_chunk_valid():
    base = UnifiedMessage(message_type=MessageType.DATA, payload=b"abcdefgh", total_chunks=3, max_chunk_size=4)
    chunk0 = base.create_chunk(b"abcd", 0)
    assert chunk0.chunk_index == 0
    assert chunk0.total_chunks == 3
    assert chunk0.metadata.correlation_id == base.metadata.message_id
    chunk2 = base.create_chunk(b"efgh", 2)
    assert chunk2.is_final_chunk
    assert chunk2.metadata.correlation_id == base.metadata.message_id


def test_create_chunk_invalid_index():
    base = UnifiedMessage(message_type=MessageType.DATA, payload=b"abcd", total_chunks=2, max_chunk_size=2)
    with pytest.raises(ValueError):
        base.create_chunk(b"cd", 2)


@pytest.mark.parametrize(
    "chunk_index,total_chunks,max_chunk_size",
    [
        (-1, 1, 1),
        (1, 1, 1),
        (0, 0, 1),
        (0, 1, 0),
    ],
)
def test_unified_message_validation(chunk_index, total_chunks, max_chunk_size):
    with pytest.raises(ValueError):
        UnifiedMessage(
            message_type=MessageType.DATA,
            payload=b"data",
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            max_chunk_size=max_chunk_size,
        )
