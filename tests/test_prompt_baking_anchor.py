import unittest
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("PyTorch not installed")

import torch
from types import SimpleNamespace
from agent_forge.prompt_baking.baker import bake
from agent_forge.prompt_baking.loader import inject_morality
from agent_forge.prompt_baking.prompts import morality_v1


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "Eudaimonic": 0,
            "flourishing": 1,
            "How": 2,
            "should": 3,
            "I": 4,
            "treat": 5,
            "my": 6,
            "neighbor?": 7,
        }
        self.unk = len(self.vocab)
        self.inv = {v: k for k, v in self.vocab.items()}
        self.inv[self.unk] = "<unk>"

    def __call__(self, text, return_tensors=None):
        ids = [self.vocab.get(t, self.unk) for t in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([ids]))

    def decode(self, ids):
        return " ".join(self.inv[int(i)] for i in ids)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        h = 4
        self.device = "cpu"
        self.config = SimpleNamespace(hidden_size=h)
        self.emb = torch.nn.Embedding(len(vocab) + 1, h)
        self.head = torch.nn.Linear(h, len(vocab) + 1)
        self.prompt_bank = torch.nn.ModuleDict()

    def forward(self, input_ids=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            embeds = self.emb(input_ids)
        else:
            embeds = inputs_embeds
        logits = self.head(embeds)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.emb

    def generate(self, inputs_embeds=None, max_length=10):
        return torch.tensor([[0, 1]])


class TestMoralAnchor(unittest.TestCase):
    def test_anchor_consistency(self):
        tok = DummyTokenizer()
        model = DummyModel(tok.vocab)
        bake(model, tok, str(morality_v1), prefix_len=2, steps=1)
        embeds = inject_morality(model, tok, "How should I treat my neighbor?")
        out = model.generate(inputs_embeds=embeds, max_length=4)[0]
        text = tok.decode(out)
        self.assertTrue("Eudaimonic" in text or "flourishing" in text)


if __name__ == "__main__":
    unittest.main()
