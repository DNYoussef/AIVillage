import unittest
import tempfile
import os
import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "adas_system", repo_root / "agent_forge" / "adas" / "system.py"
)
adas_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adas_module)
ADASystem = adas_module.ADASystem


class TestADASystem(unittest.TestCase):
    def test_optimize_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "model")
            os.makedirs(model_dir)
            with open(os.path.join(model_dir, "weights.bin"), "w") as f:
                f.write("0" * 10)

            out_dir = os.path.join(tmp, "out")
            adas = ADASystem(model_dir)
            result = adas.optimize_agent_architecture(out_dir, iterations=2)

            self.assertTrue(os.path.isdir(result))
            self.assertTrue(os.path.isfile(os.path.join(result, "weights.bin")))
            self.assertTrue(os.path.isfile(os.path.join(result, "adas_config.json")))


if __name__ == "__main__":
    unittest.main()
