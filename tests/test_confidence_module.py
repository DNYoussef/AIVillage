from core.evidence import EvidencePack, Chunk, ConfidenceTier
from rag_system.confidence import score_evidence, assign_tier


def _make(scores):
    return EvidencePack(query="q", chunks=[Chunk(id=str(i), text="t", score=s, source_uri="https://e.com") for i, s in enumerate(scores)])


def test_score_high():
    pack = _make([0.9, 0.8])
    assert score_evidence(pack) == ConfidenceTier.HIGH


def test_assign_tier():
    pack = _make([0.5])
    assign_tier(pack)
    assert pack.confidence_tier == ConfidenceTier.MEDIUM
