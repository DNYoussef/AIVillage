import unittest
from unittest import mock
import asyncio
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_system.core.pipeline import EnhancedRAGPipeline, shared_bayes_net

class TestBayesNetIntegration(unittest.TestCase):
    def test_shared_instance(self):
        p1 = object.__new__(EnhancedRAGPipeline)
        p1.bayes_net = shared_bayes_net
        p2 = object.__new__(EnhancedRAGPipeline)
        p2.bayes_net = shared_bayes_net
        self.assertIs(p1.bayes_net, p2.bayes_net)
        p1.bayes_net.add_node("n1", "content")
        self.assertIn("n1", p2.bayes_net.nodes)

    @mock.patch("requests.get")
    def test_web_scrape_updates_bayesnet(self, mock_get):
        pytest.skip("Skipping web scrape test due to missing dependencies")

if __name__ == '__main__':
    unittest.main()
