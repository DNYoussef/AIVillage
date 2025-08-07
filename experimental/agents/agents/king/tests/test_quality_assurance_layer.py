import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import torch


class DummyTok:
    def __call__(self, *args, **kwargs):
        return {"input_ids": torch.tensor([[0]]), "attention_mask": torch.tensor([[1]])}


class DummyModel:
    def __call__(self, *args, **kwargs):
        class Output:
            def __init__(self) -> None:
                self.last_hidden_state = torch.zeros((1, 1, 768))

        return Output()


# Skip if PyTorch is not installed since the QA layer relies on transformer
# models that require torch.
try:
    torch_spec = importlib.util.find_spec("torch")
except ValueError:
    torch_spec = None
if torch_spec is None:
    msg = "PyTorch not installed"
    raise unittest.SkipTest(msg)

from agents.king.quality_assurance_layer import (
    EudaimoniaTriangulator,
    QualityAssuranceLayer,
)
from agents.utils.task import Task as LangroidTask


class TestQualityAssuranceLayer(unittest.TestCase):
    def setUp(self) -> None:
        tok_patch = patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=DummyTok()
        )
        model_patch = patch(
            "transformers.AutoModel.from_pretrained", return_value=DummyModel()
        )
        self.addCleanup(tok_patch.stop)
        self.addCleanup(model_patch.stop)
        tok_patch.start()
        model_patch.start()
        self.qa_layer = QualityAssuranceLayer()

    def test_check_task_safety(self) -> None:
        task = LangroidTask(None, "Test task content")
        is_safe, metrics = self.qa_layer.check_task_safety(task)
        assert isinstance(is_safe, bool)
        assert "uncertainty" in metrics
        assert "eudaimonia_score" in metrics
        assert "rule_compliance" in metrics
        assert "safety_score" in metrics

    def test_estimate_uncertainty(self) -> None:
        task = LangroidTask(None, "Test task content")
        uncertainty = self.qa_layer.estimate_uncertainty(task)
        assert uncertainty >= 0
        assert uncertainty <= 1

    def test_evaluate_rule_compliance(self) -> None:
        task_vector = self.qa_layer.eudaimonia_triangulator.get_embedding(
            "Test task content"
        )
        rule_compliance = self.qa_layer.evaluate_rule_compliance(task_vector)
        assert rule_compliance >= 0
        assert rule_compliance <= 1

    def test_prioritize_entities(self) -> None:
        entities = ["humans", "AI", "animals", "plants"]
        prioritized = self.qa_layer.prioritize_entities(entities)
        assert len(prioritized) == len(entities)
        assert isinstance(prioritized[0], tuple)
        assert isinstance(prioritized[0][1], float)

    def test_update_task_history(self) -> None:
        task = LangroidTask(None, "Test task content")
        self.qa_layer.update_task_history(task, 0.8, 0.2)
        assert len(self.qa_layer.task_history) == 1

    def test_get_info(self) -> None:
        info = self.qa_layer.get_info()
        assert "upo_threshold" in info
        assert "num_samples" in info
        assert "task_history_size" in info
        assert "rules" in info


class TestEudaimoniaTriangulator(unittest.TestCase):
    def setUp(self) -> None:
        tok_patch = patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=DummyTok()
        )
        model_patch = patch(
            "transformers.AutoModel.from_pretrained", return_value=DummyModel()
        )
        self.addCleanup(tok_patch.stop)
        self.addCleanup(model_patch.stop)
        tok_patch.start()
        model_patch.start()
        self.triangulator = EudaimoniaTriangulator()

    def test_get_embedding(self) -> None:
        embedding = self.triangulator.get_embedding("Test content")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)  # Assuming DistilBERT's output size

    def test_triangulate(self) -> None:
        task_vector = self.triangulator.get_embedding("Test task content")
        score = self.triangulator.triangulate(task_vector)
        assert score >= 0
        assert score <= 1


if __name__ == "__main__":
    unittest.main()
