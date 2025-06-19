import unittest
from datetime import datetime

from rag_system.processing.reasoning_engine import UncertaintyAwareReasoningEngine
from rag_system.core.config import UnifiedConfig

class DummyLLM:
    async def score_path(self, query: str, path):
        return float(len(path))

class TestReasoningEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = UncertaintyAwareReasoningEngine(UnifiedConfig())
        self.engine.llm = DummyLLM()
        ts1 = datetime(2020, 1, 1)
        ts2 = datetime(2020, 1, 2)
        self.engine.graph.add_node("A", content="a", timestamp=ts1)
        self.engine.graph.add_node("B", content="b", timestamp=ts2)
        self.engine.graph.add_node("C", content="c", timestamp=ts2)
        self.engine.graph.add_edge("A", "B", weight=1.0, timestamp=ts1)
        self.engine.graph.add_edge("B", "C", weight=1.0, timestamp=ts2)

    async def test_get_snapshot(self):
        ts = datetime(2020, 1, 1, 12)
        snapshot = await self.engine.get_snapshot(ts)
        node_ids = [n for n, _ in snapshot["nodes"]]
        self.assertIn("A", node_ids)
        self.assertNotIn("B", node_ids)
        self.assertEqual(len(snapshot["edges"]), 0)

    async def test_beam_search(self):
        beams = await self.engine.beam_search("a", beam_width=2, max_depth=2)
        paths = [b[0] for b in beams]
        self.assertIn(["A", "B"], paths)

    def test_propagate_uncertainty(self):
        result = self.engine.propagate_uncertainty(["s1", "s2"], [0.1, 0.2])
        self.assertAlmostEqual(result, 1 - (1 - 0.1) * (1 - 0.2))

if __name__ == "__main__":
    unittest.main()
