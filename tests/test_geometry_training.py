import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

import torch

from agent_forge.training.geometry_pipeline import train_geometry_model
from agent_forge.training.sleep_and_dream import DreamNet, SleepNet


class TestGeometryTraining(unittest.TestCase):
    def test_train_geometry_model_runs(self):
        model = torch.nn.Linear(4, 2)
        dataset = [(torch.randn(1,4), torch.randn(1,2), "x") for _ in range(3)]
        train_geometry_model(model, dataset, epochs=1)

    def test_sleep_and_dream_offline(self):
        sleep = SleepNet(4, 4, 1, pretrained=False)
        dream = DreamNet(4, 4, 1, pretrained=False)
        x = torch.randn(2,4)
        out = dream(sleep(x))
        assert out.shape == (2, 4)

if __name__ == "__main__":
    unittest.main()
