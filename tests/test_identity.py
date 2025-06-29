import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch not installed")

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

    def test_deep_bake_rules(self):
        baker = MoralFrameworkBaker()
        original_texts = [r.text for r in baker.core_rules]
        baked_rules = baker.deep_bake_rules()

        # the returned rules should match what is stored on the baker
        self.assertEqual(baked_rules, baker.core_rules)

        # every rule text should be enriched compared to the original
        for baked, original in zip(baked_rules, original_texts):
            self.assertNotEqual(baked.text, original)


if __name__ == "__main__":
    unittest.main()
