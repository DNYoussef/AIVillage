import unittest
import asyncio
from rag_system.core.config import UnifiedConfig
from rag_system.main import initialize_components, process_user_query
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.core.structures import RetrievalResult

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
        self.assertIn("hybrid_retriever", components)
        self.assertIsInstance(components["hybrid_retriever"], HybridRetriever)
        
        # Process a sample user query
        sample_query = "What are the key features of the RAG system?"
        result = self.loop.run_until_complete(process_user_query(components, sample_query))
        
        # Verify that the result is not None and contains expected keys
        self.assertIsNotNone(result)
        self.assertIn("query", result)
        self.assertIn("integrated_result", result)
        
        # Verify that the response is not empty
        self.assertTrue(len(result["integrated_result"]) > 0)

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

if __name__ == '__main__':
    unittest.main()
