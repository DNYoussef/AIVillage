import unittest
import numpy as np

from nlp.metrics import measure_coherence, measure_relevance


class TestMetrics(unittest.TestCase):
    def test_coherence_higher_for_similar_vectors(self):
        seq1 = np.array([[[1.0, 0.0], [1.0, 0.0]]])
        seq2 = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertGreater(measure_coherence(seq1), measure_coherence(seq2))

    def test_relevance_scores(self):
        thought = np.array([[[1.0, 0.0]]])
        target_same = np.array([[[1.0, 0.0]]])
        target_diff = np.array([[[0.0, 1.0]]])
        self.assertAlmostEqual(measure_relevance(thought, target_same), 1.0, places=5)
        self.assertLess(measure_relevance(thought, target_diff), 0.5)


if __name__ == "__main__":
    unittest.main()
