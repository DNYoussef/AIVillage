import importlib.util
import unittest

from rag_system.retrieval.graph_store import GraphStore
from rag_system.retrieval.vector_store import VectorStore

if importlib.util.find_spec("numpy") is None:
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)

from datetime import datetime
from pathlib import Path
import sys
import types

import numpy as np

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *args, **kwargs: object()
sys.modules.setdefault("faiss", fake_faiss)

sys.path.append(str(Path(__file__).resolve().parents[1]))


class TestStoreCounts(unittest.IsolatedAsyncioTestCase):
    async def test_vector_store_get_count(self):
        store = VectorStore()

        class DummyIndex:
            def add(self, x):
                pass

            def search(self, x, k):
                return (np.zeros((1, k), dtype="float32"), np.zeros((1, k), dtype=int))

            def remove_ids(self, x):
                pass

        store.index = DummyIndex()
        docs = [
            {
                "id": "1",
                "content": "a",
                "embedding": np.zeros(store.dimension).astype("float32"),
                "timestamp": datetime.now(),
            },
            {
                "id": "2",
                "content": "b",
                "embedding": np.zeros(store.dimension).astype("float32"),
                "timestamp": datetime.now(),
            },
        ]
        store.add_documents(docs)
        count = await store.get_count()
        assert count == 2

    async def test_graph_store_get_count(self):
        store = GraphStore()
        docs = [
            {"id": "1", "content": "a"},
            {"id": "2", "content": "b"},
        ]
        store.add_documents(docs)
        count = await store.get_count()
        assert count == 2


if __name__ == "__main__":
    unittest.main()
