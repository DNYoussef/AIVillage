# ruff: noqa: INP001
"""Tests for conflict detection in RAG."""

from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path("src/production/rag/rag_system/core")))

from codex_rag_integration import RetrievalResult
from enhanced_query_processor import EnhancedQueryProcessor, RankedResult


def _make_ranked_result(chunk_id: str, text: str) -> RankedResult:
    retrieval = RetrievalResult(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        text=text,
        score=1.0,
        retrieval_method="keyword",
        metadata={},
    )
    return RankedResult(
        result=retrieval,
        semantic_score=1.0,
        trust_score=1.0,
        context_score=1.0,
        recency_score=1.0,
        idea_completeness_score=1.0,
        final_score=1.0,
    )


def test_detect_conflicting_sources():
    processor = EnhancedQueryProcessor(rag_pipeline=Mock())
    results = [
        _make_ranked_result("a", "Paris is the capital of France."),
        _make_ranked_result("b", "Lyon is the capital of France."),
    ]

    conflicts = processor._detect_conflicting_sources(results)  # noqa: SLF001

    assert len(conflicts) == 2
