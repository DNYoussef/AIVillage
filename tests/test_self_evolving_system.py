import importlib.util
import unittest
import pytest

pytestmark = pytest.mark.requires_gpu
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
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

if __name__ == "__main__":
    unittest.main()
