import importlib.util, unittest
if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("Required dependency not installed")

import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
import types
sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
fake_load = types.ModuleType("tiktoken.load")
def load_tiktoken_bpe(*a, **k):
    return {}
fake_load.load_tiktoken_bpe = load_tiktoken_bpe
sys.modules.setdefault("tiktoken.load", fake_load)
sys.modules["tiktoken"].Encoding = object
import agents

import importlib.util

spec = importlib.util.spec_from_file_location(
    "unified_base_agent",
    repo_root / "agents" / "unified_base_agent.py",
)
uba = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uba)
SelfEvolvingSystem = uba.SelfEvolvingSystem

class DummyAgent:
    def __init__(self):
        self.name = "dummy"
        self.capabilities = ["task"]

    def add_capability(self, cap: str):
        self.capabilities.append(cap)

class TestSelfEvolvingSystem(unittest.IsolatedAsyncioTestCase):
    async def test_evolve_agent(self):
        agent = DummyAgent()
        ses = SelfEvolvingSystem([agent])
        ses.sage_framework.assistant_response = AsyncMock(return_value="cap1, cap2")
        await ses.evolve_agent(agent)
        self.assertIn("cap1", agent.capabilities)
        self.assertIn("cap2", agent.capabilities)

    async def test_evolve_decision_maker(self):
        agent = DummyAgent()
        ses = SelfEvolvingSystem([agent])
        await ses.add_decision(np.zeros(5), 1)
        ses.dpo.fit = MagicMock()
        old_weight = ses.mcts.exploration_weight
        old_depth = ses.mcts.simulation_depth
        await ses.evolve_decision_maker()
        ses.dpo.fit.assert_called_once()
        self.assertGreater(ses.mcts.exploration_weight, old_weight)
        self.assertGreater(ses.mcts.simulation_depth, old_depth)

    async def test_system_evolve(self):
        class EvoAgent(DummyAgent):
            async def evolve(self):
                self.add_capability("new")

        agent = EvoAgent()
        ses = SelfEvolvingSystem([agent])
        await ses.evolve()
        self.assertIn("new", agent.capabilities)

if __name__ == "__main__":
    unittest.main()
