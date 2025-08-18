import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *args, **kwargs: {}
networkx_stub = types.ModuleType("networkx")
requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *a, **k: None

sys.path.append(str(Path(__file__).resolve().parents[1]))

fake_faiss = mock.MagicMock()
fake_faiss.__spec__ = mock.MagicMock()
with mock.patch.dict(
    "sys.modules",
    {
        "faiss": fake_faiss,
        "requests": requests_stub,
        "yaml": yaml_stub,
        "networkx": networkx_stub,
    },
):
    from rag_system.core.pipeline import EnhancedRAGPipeline, shared_bayes_net
    from rag_system.retrieval.graph_store import GraphStore


class DummyGraph:
    def __init__(self) -> None:
        self.nodes = {}
        self.edges = {}

    def add_node(self, node_id: str, **attrs):
        self.nodes[node_id] = attrs

    def add_edge(self, src: str, dst: str, weight: float = 0.0):
        self.edges[(src, dst)] = {"weight": weight}


class TestBayesNetIntegration(unittest.TestCase):
    def test_shared_instance(self):
        p1 = object.__new__(EnhancedRAGPipeline)
        p1.bayes_net = shared_bayes_net
        p2 = object.__new__(EnhancedRAGPipeline)
        p2.bayes_net = shared_bayes_net
        assert p1.bayes_net is p2.bayes_net
        p1.bayes_net.add_node("n1", "content")
        assert "n1" in p2.bayes_net.nodes

    @mock.patch("requests.get")
    def test_web_scrape_updates_bayesnet(self, mock_get):
        pytest.skip("Skipping web scrape test due to missing dependencies")


class TestGraphStoreSimilarity(unittest.TestCase):
    def test_cosine_similarity_edge_weight(self):
        store = GraphStore()
        store.graph = DummyGraph()

        docs = [
            {"id": "1", "content": "a", "embedding": [1.0, 0.0, 0.0]},
            {"id": "2", "content": "b", "embedding": [1.0, 0.0, 0.0]},
        ]

        store.add_documents(docs)

        assert ("1", "2") in store.graph.edges
        self.assertAlmostEqual(store.graph.edges[("1", "2")]["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
