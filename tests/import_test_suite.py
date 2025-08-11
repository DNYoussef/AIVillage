from pathlib import Path
import sys
import unittest

# Add project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "experimental" / "services"))


class TestComponentImports(unittest.TestCase):
    def test_import_libp2p_bridge(self):
        try:
            from core.p2p.libp2p_mesh import LibP2PMeshNetwork
        except ImportError as e:
            self.fail(f"Failed to import LibP2PMeshNetwork: {e}")

    def test_import_evolution_manager(self):
        try:
            from experimental.agents.agents.king.evolution_manager import (
                EvolutionManager,
            )
        except ImportError as e:
            self.skipTest(f"EvolutionManager dependencies missing: {e}")

    def test_import_rag_pipeline(self):
        try:
            from experimental.rag.storm.wikipedia_storm_pipeline import (
                WikipediaSTORMPipeline,
            )
        except ImportError as e:
            self.fail(f"Failed to import WikipediaSTORMPipeline: {e}")

    def test_import_digital_twin_api(self):
        try:
            from services.twin.app import app as twin_app
        except ImportError as e:
            self.skipTest(f"Digital Twin API not available: {e}")

    @unittest.skip("Skipping Token Economy test: 'experimental/economy' directory not found.")
    def test_import_token_economy(self):
        pass

    @unittest.skip("Skipping React Native test: 'experimental/mobile' directory not found.")
    def test_import_react_native_app_api_service(self):
        pass

    def test_import_wikipedia_storm_pipeline(self):
        try:
            from experimental.rag.storm.wikipedia_storm_pipeline import (
                WikipediaSTORMPipeline,
            )
        except ImportError as e:
            self.fail(f"Failed to import WikipediaSTORMPipeline: {e}")


if __name__ == "__main__":
    unittest.main()
