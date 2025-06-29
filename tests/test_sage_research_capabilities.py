import unittest
import importlib.util
import importlib.machinery
from unittest import mock
from pathlib import Path
import types
import sys


def load_module():
    # Minimal stubs for heavy dependencies
    fake_torch = types.ModuleType('torch')
    fake_torch.__spec__ = importlib.machinery.ModuleSpec('torch', loader=None)
    fake_torch.Tensor = type('Tensor', (), {})
    fake_torch.nn = types.ModuleType('nn')
    fake_torch.nn.functional = types.ModuleType('functional')
    fake_torch.optim = types.ModuleType('optim')

    fake_agent_forge = types.ModuleType('agent_forge')
    fake_adas = types.ModuleType('agent_forge.adas')
    fake_tech = types.ModuleType('agent_forge.adas.technique_archive')

    class ChainOfThought:
        def process(self, text):
            return text

    fake_tech.ChainOfThought = ChainOfThought
    fake_adas.technique_archive = fake_tech
    fake_agent_forge.adas = fake_adas

    modules = {
        'torch': fake_torch,
        'torch.nn': fake_torch.nn,
        'torch.nn.functional': fake_torch.nn.functional,
        'torch.optim': fake_torch.optim,
        'agent_forge': fake_agent_forge,
        'agent_forge.adas': fake_adas,
        'agent_forge.adas.technique_archive': fake_tech,
    }

    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'research_capabilities',
            Path(__file__).resolve().parents[1] / 'agents' / 'sage' / 'research_capabilities.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class DummyAgent:
    def __init__(self):
        self.research_capabilities = ['web_search', 'data_analysis']


class TestResearchCapabilitiesEvolution(unittest.IsolatedAsyncioTestCase):
    async def test_evolve_research_capabilities(self):
        rc_module = load_module()
        ResearchCapabilities = rc_module.ResearchCapabilities
        agent = DummyAgent()
        rc = ResearchCapabilities(agent)
        for _ in range(5):
            rc.record_result('web_search', True)
            rc.record_result('data_analysis', False)
            rc.record_result('exploration_mode', True)
        await rc.evolve_research_capabilities()
        self.assertIn('web_search', agent.research_capabilities)
        self.assertNotIn('data_analysis', agent.research_capabilities)
        self.assertIn('exploration_mode', agent.research_capabilities)


if __name__ == '__main__':
    unittest.main()
