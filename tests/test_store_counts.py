import importlib.util
import unittest

from datetime import datetime
import numpy as np
import sys
from pathlib import Path

import types

fake_faiss = types.ModuleType("faiss")
fake_faiss.IndexFlatL2 = lambda *args, **kwargs: object()
sys.modules.setdefault("faiss", fake_faiss)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore

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
                "embedding": np.zeros(store.dimension).astype('float32'),
                "timestamp": datetime.now(),
            },
            {
                "id": "2",
                "content": "b",
                "embedding": np.zeros(store.dimension).astype('float32'),
                "timestamp": datetime.now(),
            },
        ]
        store.add_documents(docs)
        count = await store.get_count()
        self.assertEqual(count, 2)

    async def test_graph_store_get_count(self):
        store = GraphStore()
        docs = [
            {"id": "1", "content": "a"},
            {"id": "2", "content": "b"},
        ]
        store.add_documents(docs)
        count = await store.get_count()
        self.assertEqual(count, 2)

if __name__ == "__main__":
    unittest.main()
