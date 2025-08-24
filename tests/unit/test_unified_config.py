from datetime import timedelta
import sys
import types
import unittest

from rag_system.core.config import UnifiedConfig

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *args, **kwargs: {}
sys.modules.setdefault("yaml", yaml_stub)


class TestUnifiedConfig(unittest.TestCase):
    def test_default_values(self):
        config = UnifiedConfig()
        assert config.embedding_model == "bert-base-uncased"
        assert config.vector_store_type == "faiss"
        assert config.graph_store_type == "networkx"
        assert config.retriever_type == "hybrid"
        assert config.reasoning_engine_type == "uncertainty_aware"
        assert config.MAX_RESULTS == 10
        assert config.FEEDBACK_ITERATIONS == 3
        assert timedelta(days=1) == config.TEMPORAL_GRANULARITY

    def test_custom_values(self):
        custom_config = UnifiedConfig(
            embedding_model="custom-embedding",
            vector_store_type="custom-vector-store",
            graph_store_type="custom-graph-store",
            retriever_type="custom-retriever",
            reasoning_engine_type="custom-reasoning",
            MAX_RESULTS=20,
            FEEDBACK_ITERATIONS=5,
            TEMPORAL_GRANULARITY=timedelta(hours=12),
        )
        assert custom_config.embedding_model == "custom-embedding"
        assert custom_config.vector_store_type == "custom-vector-store"
        assert custom_config.graph_store_type == "custom-graph-store"
        assert custom_config.retriever_type == "custom-retriever"
        assert custom_config.reasoning_engine_type == "custom-reasoning"
        assert custom_config.MAX_RESULTS == 20
        assert custom_config.FEEDBACK_ITERATIONS == 5
        assert timedelta(hours=12) == custom_config.TEMPORAL_GRANULARITY

    def test_update_method(self):
        config = UnifiedConfig()
        config.update(embedding_model="updated-embedding", MAX_RESULTS=15)
        assert config.embedding_model == "updated-embedding"
        assert config.MAX_RESULTS == 15

    def test_get_method(self):
        config = UnifiedConfig()
        assert config.get("embedding_model") == "bert-base-uncased"
        assert config.get("MAX_RESULTS") == 10
        assert config.get("non_existent_key", "default") == "default"

    def test_extra_params(self):
        config = UnifiedConfig()
        config.update(custom_param="custom_value")
        assert config.get("custom_param") == "custom_value"


if __name__ == "__main__":
    unittest.main()
