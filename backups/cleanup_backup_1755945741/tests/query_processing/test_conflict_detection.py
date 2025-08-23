import sys

sys.path.append("src/production/rag/rag_system/core")

from codex_rag_integration import RetrievalResult
from enhanced_query_processor import EnhancedQueryProcessor, RankedResult


class DummyPipeline:
    pass


def _build_result(text: str, score: float) -> RankedResult:
    return RankedResult(
        result=RetrievalResult(
            chunk_id=f"c{score}",
            document_id=f"d{score}",
            text=text,
            score=score,
            retrieval_method="vector",
            metadata=None,
        ),
        semantic_score=score,
        trust_score=score,
        context_score=score,
        recency_score=score,
        idea_completeness_score=1.0,
        final_score=score,
    )


def test_rank_and_select_detects_conflict() -> None:
    processor = EnhancedQueryProcessor(rag_pipeline=DummyPipeline())
    result1 = _build_result("The sky is blue.", 0.9)
    result2 = _build_result("The sky is not blue.", 0.8)

    primary, supporting, conflicting = processor._rank_and_select([result1, result2])

    assert primary == [result1]
    assert conflicting == [result2]
    assert supporting == []


def test_rank_and_select_no_conflict() -> None:
    processor = EnhancedQueryProcessor(rag_pipeline=DummyPipeline())
    result1 = _build_result("The sky is blue.", 0.9)
    result2 = _build_result("Water is wet.", 0.8)

    primary, supporting, conflicting = processor._rank_and_select([result1, result2])

    assert primary == [result1]
    assert supporting == [result2]
    assert conflicting == []
