import pytest
from core.evidence import EvidencePack, Chunk


def make_pack():
    return EvidencePack(
        query="test",
        chunks=[Chunk(id="1", text="t", score=0.5, source_uri="https://example.com")],
    )


def test_round_trip_dict():
    pack = make_pack()
    data = pack.dict()
    new = EvidencePack(**data)
    assert new == pack


def test_json_helpers():
    pack = make_pack()
    s = pack.to_json()
    new = EvidencePack.from_json(s)
    assert new == pack


def test_validation():
    with pytest.raises(ValueError):
        EvidencePack(query="x", chunks=[])
    with pytest.raises(ValueError):
        Chunk(id="1", text="t", score=1.5, source_uri="https://example.com")
