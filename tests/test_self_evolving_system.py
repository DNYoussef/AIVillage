import importlib.util
import unittest

import pytest

if (
    importlib.util.find_spec("numpy") is None
    or importlib.util.find_spec("torch") is None
):
    msg = "Required dependency not installed"
    raise unittest.SkipTest(msg)

from pathlib import Path
import sys
from unittest.mock import AsyncMock, MagicMock

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))
import types

# Create complete tiktoken stub with proper __spec__
fake_tiktoken = types.ModuleType("tiktoken")
fake_load = types.ModuleType("tiktoken.load")


# Add fake spec to avoid transformers check
class FakeSpec:
    def __init__(self, name):
        self.name = name
        self.loader = None
        self.origin = None


fake_tiktoken.__spec__ = FakeSpec("tiktoken")


def load_tiktoken_bpe(*a, **k):
    return {}


def encoding_for_model(model_name):
    """Stub for tiktoken.encoding_for_model"""

    class FakeEncoding:
        def encode(self, text):
            return [1, 2, 3]  # Dummy token IDs

        def decode(self, tokens):
            return "dummy text"

    return FakeEncoding()


fake_load.load_tiktoken_bpe = load_tiktoken_bpe
fake_tiktoken.Encoding = object
fake_tiktoken.encoding_for_model = encoding_for_model
sys.modules["tiktoken"] = fake_tiktoken
sys.modules["tiktoken.load"] = fake_load

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
    @pytest.mark.xfail(
        reason="SelfEvolvingSystem is a stub implementation - see unified_base_agent.py:791"
    )
    async def test_evolve_agent(self):
        agent = DummyAgent()
        ses = SelfEvolvingSystem([agent])
        ses.sage_framework.assistant_response = AsyncMock(return_value="cap1, cap2")
        await ses.evolve_agent(agent)
        assert "cap1" in agent.capabilities
        assert "cap2" in agent.capabilities

    @pytest.mark.xfail(
        reason="SelfEvolvingSystem is a stub implementation - see unified_base_agent.py:791"
    )
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

    @pytest.mark.xfail(
        reason="SelfEvolvingSystem is a stub implementation - see unified_base_agent.py:791"
    )
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
