import importlib
import unittest

if importlib.util.find_spec("torch") is None or importlib.util.find_spec("grokfast") is None:
    msg = "Dependencies not installed"
    raise unittest.SkipTest(msg)

import torch

from agent_forge.training.grokfast_opt import GrokfastAdam


class TestGrokfastOpt(unittest.TestCase):
    def test_slow_power(self):
        lin = torch.nn.Linear(2, 1)
        opt = GrokfastAdam(lin.parameters(), lr=0.01)
        out = lin(torch.randn(4, 2)).sum()
        out.backward()
        opt.step()
        opt.zero_grad()
        assert isinstance(opt.slow_power(), float)


if __name__ == "__main__":
    unittest.main()
