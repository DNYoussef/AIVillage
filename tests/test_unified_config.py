import unittest
from datetime import timedelta
from rag_system.core.config import UnifiedConfig

class TestUnifiedConfig(unittest.TestCase):
    def test_default_values(self):
        config = UnifiedConfig()
        self.assertEqual(config.embedding_model, "bert-base-uncased")
        self.assertEqual(config.vector_store_type, "faiss")
        self.assertEqual(config.graph_store_type, "networkx")
        self.assertEqual(config.retriever_type, "hybrid")
        self.assertEqual(config.reasoning_engine_type, "uncertainty_aware")
        self.assertEqual(config.MAX_RESULTS, 10)
        self.assertEqual(config.FEEDBACK_ITERATIONS, 3)
        self.assertEqual(config.TEMPORAL_GRANULARITY, timedelta(days=1))

    def test_custom_values(self):
        custom_config = UnifiedConfig(
            embedding_model="custom-embedding",
            vector_store_type="custom-vector-store",
            graph_store_type="custom-graph-store",
            retriever_type="custom-retriever",
            reasoning_engine_type="custom-reasoning",
            MAX_RESULTS=20,
            FEEDBACK_ITERATIONS=5,
            TEMPORAL_GRANULARITY=timedelta(hours=12)
        )
        self.assertEqual(custom_config.embedding_model, "custom-embedding")
        self.assertEqual(custom_config.vector_store_type, "custom-vector-store")
        self.assertEqual(custom_config.graph_store_type, "custom-graph-store")
        self.assertEqual(custom_config.retriever_type, "custom-retriever")
        self.assertEqual(custom_config.reasoning_engine_type, "custom-reasoning")
        self.assertEqual(custom_config.MAX_RESULTS, 20)
        self.assertEqual(custom_config.FEEDBACK_ITERATIONS, 5)
        self.assertEqual(custom_config.TEMPORAL_GRANULARITY, timedelta(hours=12))

    def test_update_method(self):
        config = UnifiedConfig()
        config.update(embedding_model="updated-embedding", MAX_RESULTS=15)
        self.assertEqual(config.embedding_model, "updated-embedding")
        self.assertEqual(config.MAX_RESULTS, 15)

    def test_get_method(self):
        config = UnifiedConfig()
        self.assertEqual(config.get("embedding_model"), "bert-base-uncased")
        self.assertEqual(config.get("MAX_RESULTS"), 10)
        self.assertEqual(config.get("non_existent_key", "default"), "default")

    def test_extra_params(self):
        config = UnifiedConfig()
        config.update(custom_param="custom_value")
        self.assertEqual(config.get("custom_param"), "custom_value")

if __name__ == '__main__':
    unittest.main()
