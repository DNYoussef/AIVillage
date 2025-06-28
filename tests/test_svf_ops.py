import importlib
import unittest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("svf") is None:
    raise unittest.SkipTest("PyTorch or SVF not installed")

import torch
from agent_forge.training.svf_ops import apply_svf

class TestSVFOps(unittest.TestCase):
    def test_apply_svf(self):
        lin = torch.nn.Linear(4, 4, bias=False)
        before = lin.weight.clone()
        apply_svf(lin, {"": torch.full((4,), 0.01)})
        self.assertFalse(torch.allclose(before, lin.weight))

if __name__ == "__main__":
    unittest.main()
