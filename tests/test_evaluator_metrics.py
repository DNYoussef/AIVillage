import importlib.util
import sys
import types
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is None:
    torch_stub = types.ModuleType("torch")
    torch_stub.tensor = lambda x: x
    torch_stub.exp = lambda x: x
    sys.modules["torch"] = torch_stub

spec = importlib.util.spec_from_file_location(
    "evaluator",
    Path(__file__).resolve().parents[1] / "agent_forge" / "evaluation" / "evaluator.py",
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)

measure_coherence = evaluator.measure_coherence
measure_relevance = evaluator.measure_relevance
evaluate_thought_quality = evaluator.evaluate_thought_quality


class DummyModel:
    def generate_thoughts(self, text, mask):
        return text


class TestEvaluationMetrics(unittest.TestCase):
    def test_measure_coherence(self):
        coherent = "The cat sits on the mat. The cat likes the mat."
        incoherent = "The cat sits on the mat. Quantum physics explains universes."
        assert measure_coherence(coherent) > measure_coherence(incoherent)

    def test_measure_relevance(self):
        query = "cat sleeping on the sofa"
        good = "The cat is sleeping on the sofa."
        bad = "Quantum mechanics is hard."
        assert measure_relevance(good, query) > measure_relevance(bad, query)

    def test_evaluate_thought_quality(self):
        eval_data = [
            ("Cats like milk.", None, "cats like milk"),
            ("Physics is fun.", None, "physics fun"),
        ]
        result = evaluate_thought_quality(DummyModel(), eval_data)
        assert "avg_coherence" in result
        assert "avg_relevance" in result
        assert result["avg_coherence"] >= 0.0
        assert result["avg_coherence"] <= 1.0
        assert result["avg_relevance"] >= 0.0
        assert result["avg_relevance"] <= 1.0


if __name__ == "__main__":
    unittest.main()
