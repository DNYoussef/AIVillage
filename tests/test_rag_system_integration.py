import asyncio
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.skip("Skipping integration test due to heavy dependencies", allow_module_level=True)

fake_faiss = mock.MagicMock()
fake_faiss.__spec__ = mock.MagicMock()
fake_torch = mock.MagicMock()
fake_torch.__spec__ = mock.MagicMock()
fake_mpl = mock.MagicMock()
fake_mpl.__spec__ = mock.MagicMock()

fake_sklearn = types.ModuleType("sklearn")
fake_metrics = types.ModuleType("sklearn.metrics")
fake_pairwise = types.ModuleType("sklearn.metrics.pairwise")
fake_pairwise.cosine_similarity = lambda *args, **kwargs: None
fake_metrics.pairwise = fake_pairwise
fake_sklearn.metrics = fake_metrics
with mock.patch.dict(
    "sys.modules",
    {
        "faiss": fake_faiss,
        "torch": fake_torch,
        "matplotlib": fake_mpl,
        "matplotlib.pyplot": fake_mpl,
        "seaborn": fake_mpl,
        "pandas": fake_mpl,
        "transformers": fake_mpl,
        "sklearn": fake_sklearn,
        "sklearn.metrics": fake_metrics,
        "sklearn.metrics.pairwise": fake_pairwise,
    },
):
    from rag_system.core.config import UnifiedConfig
    from rag_system.core.structures import RetrievalResult
    from rag_system.main import process_user_query
    from rag_system.retrieval.hybrid_retriever import HybridRetriever
    from rag_system.retrieval.vector_store import VectorStore


class MockVectorStore:
    async def retrieve(self, query_vector, k, timestamp=None):
        return [RetrievalResult(id="1", content="Mock vector result", score=0.9)]


class MockGraphStore:
    async def retrieve(self, query, k, timestamp=None):
        return [RetrievalResult(id="2", content="Mock graph result", score=0.8)]


class TestRAGSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.config = UnifiedConfig()
        self.loop = asyncio.get_event_loop()

    def test_rag_system_integration(self):
        components = self.loop.run_until_complete(self._initialize_mock_components())

        # Verify that all necessary components are initialized
        assert "hybrid_retriever" in components
        assert isinstance(components["hybrid_retriever"], HybridRetriever)

        # Process a sample user query
        sample_query = "What are the key features of the RAG system?"
        result = self.loop.run_until_complete(process_user_query(components, sample_query))

        # Verify that the result is not None and contains expected keys
        assert result is not None
        assert "query" in result
        assert "integrated_result" in result

        # Verify that the response is not empty
        assert len(result["integrated_result"]) > 0

    async def _initialize_mock_components(self):
        config = UnifiedConfig()
        components = {
            "vector_store": MockVectorStore(),
            "graph_store": MockGraphStore(),
            "hybrid_retriever": HybridRetriever(config),
        }
        components["hybrid_retriever"].vector_store = components["vector_store"]
        components["hybrid_retriever"].graph_store = components["graph_store"]
        return components


class DummyEmbeddingModel:
    def __init__(self, size: int = 8) -> None:
        self.hidden_size = size

    def encode(self, text: str):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return [], rng.random(self.hidden_size).astype("float32")


class TestVectorStoreDeterminism(unittest.IsolatedAsyncioTestCase):
    async def test_embeddings_are_deterministic(self):
        model = DummyEmbeddingModel(4)
        store = VectorStore(embedding_model=model, dimension=model.hidden_size)

        await store.add_texts(["test", "test"])

        emb1 = store.documents[0]["embedding"]
        emb2 = store.documents[1]["embedding"]
        assert np.array_equal(emb1, emb2)


if __name__ == "__main__":
    unittest.main()
