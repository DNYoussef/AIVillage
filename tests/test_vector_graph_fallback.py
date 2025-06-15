import unittest
import asyncio
from datetime import datetime

from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore


class TestFallbackStores(unittest.TestCase):
    def test_vector_store_fallback(self):
        store = VectorStore()
        doc = {
            "id": "1",
            "content": "hello world",
            "embedding": [0.1] * store.dimension,
            "timestamp": datetime.now(),
        }
        store.add_documents([doc])
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            store.retrieve(doc["embedding"], 1)
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "1")

    def test_graph_store_fallback(self):
        store = GraphStore()
        doc = {
            "id": "1",
            "content": "hello graph",
            "timestamp": datetime.now(),
        }
        store.add_documents([doc])
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(store.retrieve("hello", 1))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "1")


if __name__ == "__main__":
    unittest.main()
