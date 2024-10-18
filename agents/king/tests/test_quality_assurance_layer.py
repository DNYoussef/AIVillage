import unittest
import asyncio
from agents.king.quality_assurance_layer import QualityAssuranceLayer, EudaimoniaTriangulator
from agents.utils.task import Task as LangroidTask

class TestQualityAssuranceLayer(unittest.TestCase):
    def setUp(self):
        self.qa_layer = QualityAssuranceLayer()

    def test_check_task_safety(self):
        task = LangroidTask(None, "Test task content")
        is_safe, metrics = self.qa_layer.check_task_safety(task)
        self.assertIsInstance(is_safe, bool)
        self.assertIn("uncertainty", metrics)
        self.assertIn("eudaimonia_score", metrics)
        self.assertIn("rule_compliance", metrics)
        self.assertIn("safety_score", metrics)

    def test_estimate_uncertainty(self):
        task = LangroidTask(None, "Test task content")
        uncertainty = self.qa_layer.estimate_uncertainty(task)
        self.assertGreaterEqual(uncertainty, 0)
        self.assertLessEqual(uncertainty, 1)

    def test_evaluate_rule_compliance(self):
        task_vector = self.qa_layer.eudaimonia_triangulator.get_embedding("Test task content")
        rule_compliance = self.qa_layer.evaluate_rule_compliance(task_vector)
        self.assertGreaterEqual(rule_compliance, 0)
        self.assertLessEqual(rule_compliance, 1)

    def test_prioritize_entities(self):
        entities = ["humans", "AI", "animals", "plants"]
        prioritized = self.qa_layer.prioritize_entities(entities)
        self.assertEqual(len(prioritized), len(entities))
        self.assertIsInstance(prioritized[0], tuple)
        self.assertIsInstance(prioritized[0][1], float)

    def test_update_task_history(self):
        task = LangroidTask(None, "Test task content")
        self.qa_layer.update_task_history(task, 0.8, 0.2)
        self.assertEqual(len(self.qa_layer.task_history), 1)

    def test_get_info(self):
        info = self.qa_layer.get_info()
        self.assertIn("upo_threshold", info)
        self.assertIn("num_samples", info)
        self.assertIn("task_history_size", info)
        self.assertIn("rules", info)

class TestEudaimoniaTriangulator(unittest.TestCase):
    def setUp(self):
        self.triangulator = EudaimoniaTriangulator()

    def test_get_embedding(self):
        embedding = self.triangulator.get_embedding("Test content")
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.shape, (768,))  # Assuming DistilBERT's output size

    def test_triangulate(self):
        task_vector = self.triangulator.get_embedding("Test task content")
        score = self.triangulator.triangulate(task_vector)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

if __name__ == '__main__':
    unittest.main()
