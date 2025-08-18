import asyncio
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

# Stub the heavy ``sentence_transformers`` module with a lightweight dummy
# implementation so that importing the pipeline does not attempt to download
# large models.


class DummyEmbedder:
    dim = 32

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), self.dim).astype("float32")

    def get_sentence_embedding_dimension(self):
        return self.dim


sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=DummyEmbedder)

# Import the pipeline module directly from the source tree
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "src" / "production" / "rag" / "rag_system"),
)
from core.pipeline import Document, EnhancedRAGPipeline


def build_docs(n: int = 50):
    """Create ``n`` small synthetic documents for testing."""

    return [Document(id=str(i), text=f"This is document number {i} about AI village") for i in range(n)]


@pytest.mark.slow
def test_rag_pipeline_basic():
    pipeline = EnhancedRAGPipeline()
    docs = build_docs(50)
    pipeline.process_documents(docs)

    query = "document number 42"
    start = time.time()
    results = asyncio.run(pipeline.retrieve(query, k=5))
    latency_ms = (time.time() - start) * 1000
    assert results, "retrieval returned no results"
    # ensure retrieval is reasonably fast – <100ms for this synthetic dataset
    assert latency_ms < 100

    # Cache should yield a hit on subsequent queries
    for _ in range(9):
        asyncio.run(pipeline.retrieve(query, k=5))
    assert pipeline.cache.hit_rate > 0.8

    # Answer generation
    answer = pipeline.generate_answer(query, results)
    assert answer.text
    assert answer.citations
    assert 0.0 <= answer.confidence <= 1.0


@pytest.mark.slow
def test_concurrent_queries():
    pipeline = EnhancedRAGPipeline()
    pipeline.process_documents(build_docs(200))

    async def run_queries():
        await asyncio.gather(*(pipeline.retrieve(f"document number {i}") for i in range(5)))

    asyncio.run(run_queries())
