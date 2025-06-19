import importlib
import unittest
import pytest

pytestmark = pytest.mark.requires_gpu
if importlib.util.find_spec("torch") is None:
    pytest.skip("PyTorch not installed", allow_module_level=True)

from agent_forge.training.expert_vectors import ExpertVectorSystem, MoralArchetype


class TestExpertVectorSystem(unittest.TestCase):
    def test_apply_vector(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Linear(4, 4, bias=False)
        system = ExpertVectorSystem(model)
        vector = system.train_expert_vector_svf("test", scale=0.01)
        before = model.weight.clone()
        system.apply_expert_vector(vector, scaling=1.0)
        after = model.weight
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
