import importlib.util
import unittest

if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("torch") is None:
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np

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
        assert "cap1" in agent.capabilities
        assert "cap2" in agent.capabilities

    async def test_evolve_decision_maker(self):
        agent = DummyAgent()
        ses = SelfEvolvingSystem([agent])
        await ses.add_decision(np.zeros(5), 1)
        ses.dpo.fit = MagicMock()
        old_weight = ses.mcts.exploration_weight
        old_depth = ses.mcts.simulation_depth
        await ses.evolve_decision_maker()
        ses.dpo.fit.assert_called_once()
        assert ses.mcts.exploration_weight > old_weight
        assert ses.mcts.simulation_depth > old_depth

    async def test_system_evolve(self):
        class EvoAgent(DummyAgent):
            async def evolve(self):
                self.add_capability("new")

        agent = EvoAgent()
        ses = SelfEvolvingSystem([agent])
        await ses.evolve()
        assert "new" in agent.capabilities

if __name__ == "__main__":
    unittest.main()
