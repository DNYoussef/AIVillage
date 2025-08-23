import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from agent_forge.training.identity import IdentityFormationSystem, MoralFrameworkBaker


class TestIdentitySystem(unittest.TestCase):
    def test_identity_creation(self):
        system = IdentityFormationSystem()
        identity = system.create_agent_identity()
        assert "Agent-" in identity.name
        assert len(identity.personality) == 5
        assert identity.village_context == "AI Village"

    def test_moral_framework(self):
        baker = MoralFrameworkBaker()
        assert len(baker.core_rules) == 4
        compass = baker.tri_part_compass
        assert compass.jesus_archetype.name == "Jesus"
        assert compass.lao_tzu_archetype.name == "Lao Tzu"
        assert compass.diogenes_archetype.name == "Diogenes"

    def test_deep_bake_rules(self):
        baker = MoralFrameworkBaker()
        original_texts = [r.text for r in baker.core_rules]
        baked_rules = baker.deep_bake_rules()

        # the returned rules should match what is stored on the baker
        assert baked_rules == baker.core_rules

        # every rule text should be enriched compared to the original
        for baked, original in zip(baked_rules, original_texts, strict=False):
            assert baked.text != original


if __name__ == "__main__":
    unittest.main()
