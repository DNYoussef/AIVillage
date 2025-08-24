import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

import torch

from agent_forge.phase3.self_modeling_gate import self_model_cycle


class DummyOpt:
    def zero_grad(self):
        pass

    def step(self, filter=True):
        pass

    def slow_power(self):
        return 0.8


class DummyModel(torch.nn.Module):
    def forward(self, prompt, temp=0.1, return_h=False):
        logits = torch.randn(1, 4)
        h = torch.randn(1, 4)
        if return_h:
            return logits, h
        return logits


class TestGate(unittest.TestCase):
    def test_gate_runs(self):
        state = {"opt": DummyOpt(), "rule": 0.5}
        tasks = [("p", 0.2)]
        self_model_cycle(DummyModel(), tasks, state, δ=0.0, max_iter=2)
        assert state.get("self_grok", False)


if __name__ == "__main__":
    unittest.main()
