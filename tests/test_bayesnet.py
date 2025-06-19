import unittest
from unittest import mock
import asyncio
import sys
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

fake_faiss = mock.MagicMock()
fake_faiss.__spec__ = mock.MagicMock()
with mock.patch.dict('sys.modules', {'faiss': fake_faiss}):
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
        mock_resp = mock.Mock()
        mock_resp.text = "<html><body>hi</body></html>"
        mock_resp.raise_for_status = mock.Mock()
        mock_get.return_value = mock_resp

        from rag_system.core.config import UnifiedConfig

        class DummySage:
            def __init__(self):
                self.rag_system = EnhancedRAGPipeline(UnifiedConfig())

            async def get_embedding(self, text: str):
                return []

            async def perform_web_scrape(self, url: str):
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                self.rag_system.hybrid_retriever.vector_store.add_documents([{"id": "1", "content": text, "embedding": [], "timestamp": datetime.now()}])
                await self.rag_system.update_bayes_net("1", text)

        agent = DummySage()
        agent.rag_system.hybrid_retriever.vector_store.add_documents = mock.Mock()
        loop = asyncio.get_event_loop()
        shared_bayes_net.nodes.clear()
        before = len(shared_bayes_net.nodes)
        loop.run_until_complete(agent.perform_web_scrape("http://example.com"))
        after = len(shared_bayes_net.nodes)
        self.assertEqual(after, before + 1)

if __name__ == '__main__':
    unittest.main()
