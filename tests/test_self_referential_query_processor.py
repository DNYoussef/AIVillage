import unittest
from types import SimpleNamespace

from rag_system.processing.self_referential_query_processor import SelfReferentialQueryProcessor


class DummyCountStore:
    async def get_count(self):
        return 0


class DummyRAGPipeline:
    def __init__(self):
        self.hybrid_retriever = SimpleNamespace(
            vector_store=DummyCountStore(),
            graph_store=DummyCountStore(),
        )

    async def process(self, query: str):
        return f"processed {query}"


class TestSelfReferentialQueryProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.pipeline = DummyRAGPipeline()
        self.processor = SelfReferentialQueryProcessor(self.pipeline, history_limit=10)

    async def test_history_accumulation(self):
        await self.processor.process_self_query("What is AI?")
        await self.processor.process_self_query("SELF:STATUS")
        assert self.processor.query_history == ["What is AI?", "SELF:STATUS"]

    async def test_get_query_history(self):
        await self.processor.process_self_query("Q1")
        await self.processor.process_self_query("Q2")
        await self.processor.process_self_query("Q3")
        history = await self.processor._get_query_history(2)
        assert history == "Recent queries: [Q2], [Q3]"


if __name__ == "__main__":
    unittest.main()
