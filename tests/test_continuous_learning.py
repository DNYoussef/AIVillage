import unittest
import asyncio
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from rag_system.retrieval.vector_store import VectorStore

spec = spec_from_file_location(
    "continuous_learning_layer",
    Path(__file__).resolve().parents[1] / "agents" / "sage" / "continuous_learning_layer.py",
)
cll = module_from_spec(spec)
spec.loader.exec_module(cll)
ContinuousLearningLayer = cll.ContinuousLearningLayer

class TestContinuousLearning(unittest.IsolatedAsyncioTestCase):
    async def test_update_and_retrieve(self):
        store = VectorStore()
        layer = ContinuousLearningLayer(store)
        task = {"content": "example"}
        await layer.update(task, {"performance": 1.0})
        await layer.evolve()
        learnings = await layer.retrieve_relevant_learnings(task)
        self.assertGreaterEqual(len(learnings), 1)

if __name__ == "__main__":
    unittest.main()
