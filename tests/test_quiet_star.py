import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch not installed")

import torch
from agent_forge.training.quiet_star import QuietSTaRModel

class DummyModel:
    def __init__(self, hidden_size=4, vocab_size=6):
        self.config = type("cfg", (), {"hidden_size": hidden_size})
        self.vocab_size = vocab_size
    def __call__(self, input_ids, attention_mask=None):
        b, l = input_ids.shape
        logits = torch.randn(b, l, self.vocab_size)
        return type("Out", (), {"logits": logits})

class TestQuietStar(unittest.TestCase):
    def test_forward_with_thoughts(self):
        model = QuietSTaRModel(DummyModel())
        inp = torch.ones(2, 3, dtype=torch.long)
        logits, thoughts = model(inp)
        self.assertEqual(logits.shape, (2, 3, model.base_model.vocab_size))
        self.assertEqual(thoughts.shape, (2, 3, model.base_model.vocab_size))

    def test_forward_without_thoughts(self):
        model = QuietSTaRModel(DummyModel())
        inp = torch.ones(1, 2, dtype=torch.long)
        logits, thoughts = model(inp, generate_thoughts=False)
        self.assertEqual(logits.shape, (1, 2, model.base_model.vocab_size))
        self.assertIsNone(thoughts)

if __name__ == "__main__":
    unittest.main()
