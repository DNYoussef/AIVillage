import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch not installed")

import torch
from agent_forge.geometry.snapshot import snapshot

class TestSnapshot(unittest.TestCase):
    def test_snapshot_keys(self):
        h = torch.randn(2, 3, 8)
        g = snapshot(h)
        self.assertIn("ID_nl", g)
        self.assertIn("ID_lin", g)
        self.assertGreaterEqual(g["ID_lin"], 1)

if __name__ == "__main__":
    unittest.main()
