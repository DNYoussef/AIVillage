import unittest
import sys
from pathlib import Path
import types
import importlib.machinery

# Ensure repository root is on path and provide dummy torch module to avoid
# heavy dependency import failures when loading the agent modules.
sys.path.append(str(Path(__file__).resolve().parents[1]))
fake_torch = types.ModuleType('torch')
fake_torch.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None)
fake_torch.Tensor = type('Tensor', (), {})
fake_torch.randn = lambda *args, **kwargs: 0
sys.modules['torch'] = fake_torch

from rag_system.agents.task_planning_agent import TaskPlanningAgent


class DummyPlanningAgent(TaskPlanningAgent):
    def __init__(self):
        pass

    async def generate(self, prompt: str):
        return ""

    async def get_embedding(self, text: str):
        return []

    async def rerank(self, query, results, k):
        return results

    async def introspect(self):
        return {}

    async def communicate(self, message, recipient):
        return ""

    async def activate_latent_space(self, query):
        return "", ""

class TestTaskPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DummyPlanningAgent()

    def test_generate_task_plan_basic(self):
        intent = {"primary_intent": "Summarize research"}
        concepts = {
            "keywords": ["machine", "learning"],
            "entities": [{"text": "AI", "label": "TECH"}]
        }
        plan = self.agent._generate_task_plan(intent, concepts)
        self.assertIn("steps", plan)
        self.assertEqual(plan["analysis"]["primary_intent"], "Summarize research")
        self.assertIn("retrieve_information", [s["action"] for s in plan["steps"]])
        self.assertIn("summarize_information", [s["action"] for s in plan["steps"]])

    def test_generate_task_plan_empty_inputs(self):
        plan = self.agent._generate_task_plan({}, {})
        self.assertIsInstance(plan["steps"], list)
        self.assertEqual(plan["steps"][0]["parameters"], [])

if __name__ == "__main__":
    unittest.main()
