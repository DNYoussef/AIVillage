import importlib
import unittest

if importlib.util.find_spec("torch") is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from unittest import mock

import torch
from agent_forge.training.expert_vectors import ExpertVectorSystem
from agent_forge.training.prompt_baking import PromptBakingManager


class TestExpertVectorSystem(unittest.TestCase):
    def test_apply_vector(self):
        model = torch.nn.Linear(4, 4, bias=False)
        system = ExpertVectorSystem(model)
        vector = system.train_expert_vector_svf("test", scale=0.01)
        before = model.weight.clone()
        system.apply_expert_vector(vector, scaling=1.0)
        after = model.weight
        assert not torch.allclose(before, after)

    def test_train_from_text(self):
        model = torch.nn.Linear(4, 4, bias=False)
        system = ExpertVectorSystem(model)
        vector = system.train_expert_vector_from_texts("txt", ["a", "b"], epochs=1)
        assert "weight" in vector.singular_values

    def test_prompt_baking_applies_vector(self):
        class DummyBaker:
            def __init__(self, *_):
                self.model = torch.nn.Linear(4, 4, bias=False)

            def load_model(self):
                pass

            def bake_prompts(self, prompts, num_iterations=1, lr=1e-5):
                pass

            def save_model(self, path):
                pass

        with mock.patch(
            "agent_forge.training.prompt_baking.rag_prompt_baker.RAGPromptBaker",
            DummyBaker,
        ):
            vector_model = torch.nn.Linear(4, 4, bias=False)
            vec = ExpertVectorSystem(vector_model).train_expert_vector_svf("v")
            manager = PromptBakingManager("dummy", {"v": vec})
            before = manager.baker.model.weight.clone()
            manager.deep_bake(["x"], num_rounds=1)
            after = manager.baker.model.weight
            assert not torch.allclose(before, after)


if __name__ == "__main__":
    unittest.main()
