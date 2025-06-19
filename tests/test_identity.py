import importlib
import unittest
import pytest

pytestmark = pytest.mark.requires_gpu
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from agent_forge.training.identity import IdentityFormationSystem, MoralFrameworkBaker


class TestIdentitySystem(unittest.TestCase):
    def test_identity_creation(self):
        system = IdentityFormationSystem()
        identity = system.create_agent_identity()
        self.assertIn("Agent-", identity.name)
        self.assertEqual(len(identity.personality), 5)
        self.assertEqual(identity.village_context, "AI Village")

    def test_moral_framework(self):
        baker = MoralFrameworkBaker()
        self.assertEqual(len(baker.core_rules), 4)
        compass = baker.tri_part_compass
        self.assertEqual(compass.jesus_archetype.name, "Jesus")
        self.assertEqual(compass.lao_tzu_archetype.name, "Lao Tzu")
        self.assertEqual(compass.diogenes_archetype.name, "Diogenes")


if __name__ == "__main__":
    unittest.main()
