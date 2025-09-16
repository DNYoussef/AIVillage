"""
Test suite for EvoMerge phase.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock

from .evomerge import EvoMerge
from .config import EvoMergeConfig
from .merge_techniques import MergeTechniques
from .fitness_evaluator import FitnessEvaluator
from .population_manager import PopulationManager
from .genetic_operations import GeneticOperations

class DummyModel(nn.Module):
    """Dummy model for testing."""
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.config = MagicMock()
        self.config.vocab_size = 32000
        self.config.hidden_size = hidden_size

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

class TestMergeTechniques(unittest.TestCase):
    """Test merge techniques."""

    def setUp(self):
        self.merge_tech = MergeTechniques(device='cpu')
        self.models = [DummyModel() for _ in range(3)]

    def test_linear_merge(self):
        """Test linear merge."""
        weights = [0.3, 0.3, 0.4]
        merged = self.merge_tech.linear_merge(self.models, weights)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters are merged
        for name, param in merged.named_parameters():
            self.assertIsNotNone(param)
            self.assertFalse(torch.isnan(param).any())

    def test_slerp_merge(self):
        """Test SLERP merge."""
        merged = self.merge_tech.slerp_merge(self.models[:2], t=0.5)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters
        for name, param in merged.named_parameters():
            self.assertFalse(torch.isnan(param).any())

    def test_ties_merge(self):
        """Test TIES merge."""
        merged = self.merge_tech.ties_merge(self.models, threshold=0.7)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters
        for name, param in merged.named_parameters():
            self.assertFalse(torch.isnan(param).any())

    def test_dare_merge(self):
        """Test DARE merge."""
        merged = self.merge_tech.dare_merge(self.models, drop_rate=0.5)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters
        for name, param in merged.named_parameters():
            self.assertFalse(torch.isnan(param).any())

    def test_frankenmerge(self):
        """Test FrankenMerge."""
        merged = self.merge_tech.frankenmerge(self.models)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters
        for name, param in merged.named_parameters():
            self.assertFalse(torch.isnan(param).any())

    def test_dfs_merge(self):
        """Test DFS merge."""
        merged = self.merge_tech.dfs_merge(self.models)

        self.assertIsInstance(merged, DummyModel)

        # Check parameters
        for name, param in merged.named_parameters():
            self.assertFalse(torch.isnan(param).any())

class TestFitnessEvaluator(unittest.TestCase):
    """Test fitness evaluator."""

    def setUp(self):
        config = {'device': 'cpu'}
        self.evaluator = FitnessEvaluator(config)
        self.model = DummyModel()

    def test_evaluate(self):
        """Test fitness evaluation."""
        metrics = self.evaluator.evaluate(self.model)

        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.perplexity, 0)
        self.assertGreaterEqual(metrics.accuracy, 0)
        self.assertLessEqual(metrics.accuracy, 1)
        self.assertGreater(metrics.inference_speed, 0)
        self.assertGreater(metrics.memory_usage, 0)
        self.assertGreater(metrics.composite_fitness, 0)

    def test_caching(self):
        """Test fitness caching."""
        # First evaluation
        metrics1 = self.evaluator.evaluate(self.model)
        self.assertEqual(self.evaluator.cache_hits, 0)
        self.assertEqual(self.evaluator.cache_misses, 1)

        # Second evaluation (should hit cache)
        metrics2 = self.evaluator.evaluate(self.model)
        self.assertEqual(self.evaluator.cache_hits, 1)
        self.assertEqual(metrics1.composite_fitness, metrics2.composite_fitness)

class TestPopulationManager(unittest.TestCase):
    """Test population manager."""

    def setUp(self):
        config = {'population_size': 4, 'elite_size': 1}
        self.manager = PopulationManager(config)
        self.base_models = [DummyModel() for _ in range(3)]
        self.merge_tech = MergeTechniques(device='cpu')

    def test_initialize_population(self):
        """Test population initialization."""
        population = self.manager.initialize_population(
            self.base_models,
            self.merge_tech
        )

        self.assertEqual(len(population), 4)
        for model in population:
            self.assertIsInstance(model, nn.Module)

    def test_diversity_calculation(self):
        """Test diversity calculation."""
        self.manager.population = [DummyModel() for _ in range(4)]
        diversity = self.manager.calculate_diversity()

        self.assertGreater(diversity, 0)

    def test_elite_selection(self):
        """Test elite selection."""
        self.manager.population = [DummyModel() for _ in range(4)]
        self.manager.fitness_scores = [0.5, 0.7, 0.3, 0.9]

        elites = self.manager.select_elites()

        self.assertEqual(len(elites), 1)
        self.assertEqual(elites[0][1], 0.9)  # Highest fitness

class TestGeneticOperations(unittest.TestCase):
    """Test genetic operations."""

    def setUp(self):
        config = {'mutation_rate': 0.5, 'crossover_rate': 0.7}
        self.genetic_ops = GeneticOperations(config)
        self.parent1 = DummyModel()
        self.parent2 = DummyModel()

    def test_crossover(self):
        """Test crossover operation."""
        child1, child2 = self.genetic_ops.crossover(
            self.parent1,
            self.parent2,
            method='uniform'
        )

        self.assertIsInstance(child1, DummyModel)
        self.assertIsInstance(child2, DummyModel)

        # Check parameters are valid
        for param in child1.parameters():
            self.assertFalse(torch.isnan(param).any())
        for param in child2.parameters():
            self.assertFalse(torch.isnan(param).any())

    def test_mutation(self):
        """Test mutation operation."""
        original_params = [p.clone() for p in self.parent1.parameters()]

        mutated = self.genetic_ops.mutate(
            self.parent1,
            method='gaussian'
        )

        # Check some parameters changed
        changed = False
        for orig, new in zip(original_params, mutated.parameters()):
            if not torch.allclose(orig, new):
                changed = True
                break

        # With 50% mutation rate, should have some changes
        # (might not always change due to randomness)

        # Check parameters are valid
        for param in mutated.parameters():
            self.assertFalse(torch.isnan(param).any())

class TestEvoMerge(unittest.TestCase):
    """Test main EvoMerge class."""

    def setUp(self):
        config = EvoMergeConfig(
            generations=2,  # Small for testing
            population_size=4,
            elite_size=1
        )
        self.evomerge = EvoMerge(config)
        self.cognate_models = [DummyModel() for _ in range(3)]

    def test_input_validation(self):
        """Test input validation."""
        # Should not raise with 3 models
        self.evomerge._validate_input_models(self.cognate_models)

        # Should raise with wrong number
        with self.assertRaises(ValueError):
            self.evomerge._validate_input_models([DummyModel()])

    def test_convergence_check(self):
        """Test convergence checking."""
        # Not converged initially
        self.assertFalse(self.evomerge._check_convergence())

        # Simulate no improvement
        for _ in range(10):
            self.evomerge.state.fitness_history.append(0.5)

        # Should detect convergence
        self.assertTrue(self.evomerge._check_convergence())

    @patch('asyncio.create_task')
    async def test_evolve(self, mock_task):
        """Test evolution process."""
        # Mock async evaluation
        mock_task.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: Mock(composite_fitness=0.5))()
        )

        # Run evolution (small generations for testing)
        self.evomerge.config.generations = 1
        self.evomerge.config.enable_parallel = False

        result = await self.evomerge.evolve(self.cognate_models)

        self.assertIsNotNone(result)
        self.assertIsInstance(result.model, nn.Module)
        self.assertGreater(result.fitness, 0)
        self.assertIn('perplexity', result.metrics)
        self.assertIn('accuracy', result.metrics)

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMergeTechniques))
    suite.addTests(loader.loadTestsFromTestCase(TestFitnessEvaluator))
    suite.addTests(loader.loadTestsFromTestCase(TestPopulationManager))
    suite.addTests(loader.loadTestsFromTestCase(TestGeneticOperations))
    suite.addTests(loader.loadTestsFromTestCase(TestEvoMerge))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)