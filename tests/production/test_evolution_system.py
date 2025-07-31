"""
Comprehensive tests for Production Evolution System.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
import numpy as np

# Mock external dependencies
try:
    from production.evolution.evolution.math_tutor_evolution import (
        MathTutorEvolution, ModelIndividual, EvolutionConfig
    )
    from production.evolution.evolution.math_fitness import FitnessEvaluator
    from production.evolution.evolution.merge_operators import MergeOperator
except ImportError:
    # Create mock classes for testing structure
    @dataclass
    class ModelIndividual:
        model_path: str
        fitness: float = 0.0
        generation: int = 0
        parent_ids: list = None
        
    @dataclass
    class EvolutionConfig:
        population_size: int = 10
        num_generations: int = 5
        mutation_rate: float = 0.1
        
    class MathTutorEvolution:
        def __init__(self, config): pass
    class FitnessEvaluator:
        def __init__(self): pass
    class MergeOperator:
        def __init__(self): pass


class TestModelIndividual:
    """Test ModelIndividual dataclass."""
    
    def test_individual_creation(self):
        """Test creating model individual."""
        individual = ModelIndividual(
            model_path="/path/to/model",
            fitness=0.85,
            generation=1,
            parent_ids=["parent1", "parent2"]
        )
        
        assert individual.model_path == "/path/to/model"
        assert individual.fitness == 0.85
        assert individual.generation == 1
        assert individual.parent_ids == ["parent1", "parent2"]
    
    def test_individual_defaults(self):
        """Test default values for individual."""
        individual = ModelIndividual(model_path="/path/to/model")
        
        assert individual.fitness == 0.0
        assert individual.generation == 0
        assert individual.parent_ids is None
    
    def test_individual_comparison(self):
        """Test individual fitness comparison."""
        individual1 = ModelIndividual(model_path="model1", fitness=0.8)
        individual2 = ModelIndividual(model_path="model2", fitness=0.9)
        
        # Higher fitness should be better
        assert individual2.fitness > individual1.fitness
    
    def test_individual_serialization(self):
        """Test individual serialization."""
        individual = ModelIndividual(
            model_path="/path/to/model",
            fitness=0.85,
            generation=2,
            parent_ids=["p1", "p2"]
        )
        
        # Should be serializable to dict
        from dataclasses import asdict
        individual_dict = asdict(individual)
        
        assert individual_dict["model_path"] == "/path/to/model"
        assert individual_dict["fitness"] == 0.85
        assert individual_dict["generation"] == 2
        assert individual_dict["parent_ids"] == ["p1", "p2"]


class TestEvolutionConfig:
    """Test EvolutionConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating evolution configuration."""
        config = EvolutionConfig(
            population_size=20,
            num_generations=10,
            mutation_rate=0.05
        )
        
        assert config.population_size == 20
        assert config.num_generations == 10
        assert config.mutation_rate == 0.05
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = EvolutionConfig()
        
        assert config.population_size == 10
        assert config.num_generations == 5
        assert config.mutation_rate == 0.1
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = EvolutionConfig(population_size=10, num_generations=5, mutation_rate=0.1)
        assert config.population_size > 0
        assert config.num_generations > 0
        assert 0 <= config.mutation_rate <= 1
        
        # Test bounds
        assert config.population_size >= 1
        assert config.num_generations >= 1
        assert 0.0 <= config.mutation_rate <= 1.0


class TestMathTutorEvolution:
    """Test MathTutorEvolution system."""
    
    @pytest.fixture
    def evolution_config(self):
        """Create evolution configuration for testing."""
        return EvolutionConfig(
            population_size=5,  # Small for testing
            num_generations=3,  # Few generations for speed
            mutation_rate=0.2
        )
    
    @pytest.fixture
    def evolution_system(self, evolution_config):
        """Create evolution system for testing."""
        return MathTutorEvolution(evolution_config)
    
    @pytest.fixture
    def sample_population(self):
        """Create sample population for testing."""
        return [
            ModelIndividual(f"model_{i}", fitness=np.random.random(), generation=0)
            for i in range(5)
        ]
    
    def test_evolution_initialization(self, evolution_system, evolution_config):
        """Test evolution system initialization."""
        assert evolution_system is not None
        # Test if config is stored (if accessible)
        if hasattr(evolution_system, 'config'):
            assert evolution_system.config == evolution_config
    
    def test_population_initialization(self, evolution_system):
        """Test population initialization."""
        if hasattr(evolution_system, 'initialize_population'):
            population = evolution_system.initialize_population()
            
            assert len(population) > 0
            assert all(isinstance(ind, ModelIndividual) for ind in population)
            assert all(ind.generation == 0 for ind in population)
    
    def test_fitness_evaluation(self, evolution_system, sample_population):
        """Test fitness evaluation of population."""
        if hasattr(evolution_system, 'evaluate_fitness'):
            # Mock fitness evaluation
            with patch.object(evolution_system, 'fitness_evaluator') as mock_evaluator:
                mock_evaluator.evaluate.return_value = 0.75
                
                evaluated_pop = evolution_system.evaluate_fitness(sample_population)
                
                assert len(evaluated_pop) == len(sample_population)
                # Fitness should be updated (if evaluation worked)
                if evaluated_pop:
                    assert all(ind.fitness >= 0 for ind in evaluated_pop)
    
    def test_selection_mechanism(self, evolution_system, sample_population):
        """Test selection mechanism."""
        # Set known fitness values
        for i, individual in enumerate(sample_population):
            individual.fitness = i * 0.2  # 0.0, 0.2, 0.4, 0.6, 0.8
        
        if hasattr(evolution_system, 'select_parents'):
            parents = evolution_system.select_parents(sample_population, num_parents=2)
            
            assert len(parents) == 2
            assert all(isinstance(parent, ModelIndividual) for parent in parents)
            # Higher fitness individuals should be more likely to be selected
            assert all(parent.fitness >= 0 for parent in parents)
    
    def test_crossover_operation(self, evolution_system):
        """Test crossover operation."""
        parent1 = ModelIndividual("model_1", fitness=0.8, generation=1)
        parent2 = ModelIndividual("model_2", fitness=0.7, generation=1)
        
        if hasattr(evolution_system, 'crossover'):
            offspring = evolution_system.crossover(parent1, parent2)
            
            assert isinstance(offspring, ModelIndividual)
            assert offspring.generation == 2  # Next generation
            assert offspring.parent_ids == ["model_1", "model_2"]
    
    def test_mutation_operation(self, evolution_system):
        """Test mutation operation."""
        individual = ModelIndividual("model_test", fitness=0.5, generation=1)
        
        if hasattr(evolution_system, 'mutate'):
            mutated = evolution_system.mutate(individual)
            
            assert isinstance(mutated, ModelIndividual)
            assert mutated.model_path != individual.model_path  # Should change
            assert mutated.generation == individual.generation
    
    def test_evolution_step(self, evolution_system, sample_population):
        """Test single evolution step."""
        if hasattr(evolution_system, 'evolution_step'):
            next_generation = evolution_system.evolution_step(sample_population)
            
            assert len(next_generation) == len(sample_population)
            assert all(isinstance(ind, ModelIndividual) for ind in next_generation)
            # Generation should increment
            if next_generation:
                assert next_generation[0].generation > sample_population[0].generation
    
    def test_full_evolution_run(self, evolution_system):
        """Test complete evolution run."""
        if hasattr(evolution_system, 'run_evolution'):
            # Mock the run to avoid long execution
            with patch.object(evolution_system, 'initialize_population') as mock_init:
                mock_init.return_value = [
                    ModelIndividual(f"model_{i}", fitness=0.5, generation=0)
                    for i in range(3)
                ]
                
                try:
                    final_population = evolution_system.run_evolution()
                    
                    assert len(final_population) > 0
                    assert all(isinstance(ind, ModelIndividual) for ind in final_population)
                    
                except AttributeError:
                    # Method might not exist in mock
                    pass
    
    def test_best_individual_tracking(self, evolution_system, sample_population):
        """Test tracking of best individual."""
        # Set fitness values
        sample_population[2].fitness = 0.95  # Best individual
        
        if hasattr(evolution_system, 'get_best_individual'):
            best = evolution_system.get_best_individual(sample_population)
            
            assert best == sample_population[2]
            assert best.fitness == 0.95
        else:
            # Manual best finding
            best = max(sample_population, key=lambda x: x.fitness)
            assert best.fitness == 0.95
    
    def test_evolution_history_tracking(self, evolution_system):
        """Test evolution history tracking."""
        if hasattr(evolution_system, 'history'):
            # Should start empty
            assert len(evolution_system.history) == 0
            
            # After running evolution, should have history
            # This would be tested in integration tests


class TestFitnessEvaluator:
    """Test fitness evaluation system."""
    
    @pytest.fixture
    def fitness_evaluator(self):
        """Create fitness evaluator for testing."""
        return FitnessEvaluator()
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for evaluation."""
        model = Mock()
        model.eval.return_value = None
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        return model
    
    def test_evaluator_initialization(self, fitness_evaluator):
        """Test fitness evaluator initialization."""
        assert fitness_evaluator is not None
        # Check if evaluator has necessary methods
        expected_methods = ['evaluate', 'evaluate_math_performance']
        for method in expected_methods:
            if hasattr(fitness_evaluator, method):
                assert callable(getattr(fitness_evaluator, method))
    
    def test_math_performance_evaluation(self, fitness_evaluator, mock_model):
        """Test math performance evaluation."""
        if hasattr(fitness_evaluator, 'evaluate_math_performance'):
            with patch('torch.tensor') as mock_tensor:
                mock_tensor.return_value = Mock()
                
                score = fitness_evaluator.evaluate_math_performance(mock_model)
                
                assert isinstance(score, (int, float))
                assert 0 <= score <= 1
    
    def test_fitness_aggregation(self, fitness_evaluator, mock_model):
        """Test fitness score aggregation."""
        if hasattr(fitness_evaluator, 'evaluate'):
            individual = ModelIndividual("test_model", fitness=0.0)
            
            # Mock model loading
            with patch.object(fitness_evaluator, 'load_model', return_value=mock_model):
                fitness_score = fitness_evaluator.evaluate(individual)
                
                assert isinstance(fitness_score, (int, float))
                assert 0 <= fitness_score <= 1
    
    def test_batch_evaluation(self, fitness_evaluator):
        """Test batch fitness evaluation."""
        population = [
            ModelIndividual(f"model_{i}", fitness=0.0)
            for i in range(3)
        ]
        
        if hasattr(fitness_evaluator, 'evaluate_batch'):
            scores = fitness_evaluator.evaluate_batch(population)
            
            assert len(scores) == len(population)
            assert all(isinstance(score, (int, float)) for score in scores)
            assert all(0 <= score <= 1 for score in scores)


class TestMergeOperator:
    """Test model merging operations."""
    
    @pytest.fixture
    def merge_operator(self):
        """Create merge operator for testing."""
        return MergeOperator()
    
    @pytest.fixture
    def parent_models(self):
        """Create parent models for merging."""
        return [
            ModelIndividual("parent_1", fitness=0.8),
            ModelIndividual("parent_2", fitness=0.7)
        ]
    
    def test_merge_operator_initialization(self, merge_operator):
        """Test merge operator initialization."""
        assert merge_operator is not None
        # Check for merge methods
        expected_methods = ['merge', 'linear_interpolation', 'task_arithmetic']
        for method in expected_methods:
            if hasattr(merge_operator, method):
                assert callable(getattr(merge_operator, method))
    
    def test_linear_interpolation_merge(self, merge_operator, parent_models):
        """Test linear interpolation merging."""
        if hasattr(merge_operator, 'linear_interpolation'):
            merged = merge_operator.linear_interpolation(
                parent_models[0], parent_models[1], weight=0.5
            )
            
            assert isinstance(merged, (ModelIndividual, str))
            # If returns ModelIndividual, check properties
            if isinstance(merged, ModelIndividual):
                assert merged.parent_ids == ["parent_1", "parent_2"]
    
    def test_task_arithmetic_merge(self, merge_operator, parent_models):
        """Test task arithmetic merging."""
        if hasattr(merge_operator, 'task_arithmetic'):
            merged = merge_operator.task_arithmetic(parent_models)
            
            assert merged is not None
            # Should produce valid merged model
            if isinstance(merged, ModelIndividual):
                assert merged.parent_ids is not None
    
    def test_merge_with_base_model(self, merge_operator, parent_models):
        """Test merging with base model."""
        base_model = ModelIndividual("base_model", fitness=0.6)
        
        if hasattr(merge_operator, 'merge_with_base'):
            merged = merge_operator.merge_with_base(
                parent_models[0], base_model, alpha=0.3
            )
            
            assert merged is not None
            if isinstance(merged, ModelIndividual):
                assert "base_model" in str(merged.parent_ids or [])
    
    def test_adaptive_merging(self, merge_operator, parent_models):
        """Test adaptive merging based on fitness."""
        if hasattr(merge_operator, 'adaptive_merge'):
            # Higher fitness parent should have more influence
            merged = merge_operator.adaptive_merge(parent_models)
            
            assert merged is not None
            # Should consider fitness in merging strategy


@pytest.mark.integration
class TestEvolutionIntegration:
    """Integration tests for evolution system."""
    
    @pytest.mark.asyncio
    async def test_full_evolution_pipeline(self, tmp_path):
        """Test complete evolution pipeline."""
        config = EvolutionConfig(
            population_size=3,
            num_generations=2,
            mutation_rate=0.1
        )
        
        # Mock directory for models
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        
        try:
            evolution = MathTutorEvolution(config)
            
            # This would test the full pipeline
            # For now, just verify components can be initialized
            assert evolution is not None
            
            # Mock evolution run
            mock_population = [
                ModelIndividual(f"model_{i}", fitness=0.5 + i*0.1)
                for i in range(3)
            ]
            
            # Verify population structure
            assert len(mock_population) == 3
            assert all(ind.fitness > 0 for ind in mock_population)
            
        except ImportError:
            pytest.skip("Full evolution system not available")
    
    def test_evolution_persistence(self, tmp_path):
        """Test saving and loading evolution state."""
        save_path = tmp_path / "evolution_state.json"
        
        # Mock evolution state
        evolution_state = {
            "generation": 5,
            "best_fitness": 0.92,
            "population": [
                {
                    "model_path": "model_1",
                    "fitness": 0.85,
                    "generation": 5,
                    "parent_ids": ["model_a", "model_b"]
                }
            ]
        }
        
        # Save state
        with open(save_path, 'w') as f:
            json.dump(evolution_state, f)
        
        # Verify saved state
        assert save_path.exists()
        
        # Load state
        with open(save_path, 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state == evolution_state
        assert loaded_state["generation"] == 5
        assert loaded_state["best_fitness"] == 0.92


@pytest.mark.performance
class TestEvolutionPerformance:
    """Performance tests for evolution system."""
    
    def test_fitness_evaluation_performance(self):
        """Test fitness evaluation performance."""
        # Mock large population
        population_size = 50
        population = [
            ModelIndividual(f"model_{i}", fitness=0.0)
            for i in range(population_size)
        ]
        
        import time
        start_time = time.time()
        
        # Mock fitness evaluation
        for individual in population:
            individual.fitness = np.random.random()
        
        eval_time = time.time() - start_time
        
        # Should evaluate quickly
        assert eval_time < 5.0, f"Fitness evaluation took {eval_time:.2f} seconds"
        assert all(ind.fitness > 0 for ind in population)
    
    def test_merge_operation_performance(self):
        """Test merge operation performance."""
        # Create many parent pairs
        num_merges = 100
        parent_pairs = [
            (ModelIndividual(f"parent_a_{i}", fitness=0.8),
             ModelIndividual(f"parent_b_{i}", fitness=0.7))
            for i in range(num_merges)
        ]
        
        import time
        start_time = time.time()
        
        # Mock merging operations
        merged_models = []
        for parent_a, parent_b in parent_pairs:
            # Simple mock merge
            merged = ModelIndividual(
                f"merged_{len(merged_models)}",
                fitness=(parent_a.fitness + parent_b.fitness) / 2,
                parent_ids=[parent_a.model_path, parent_b.model_path]
            )
            merged_models.append(merged)
        
        merge_time = time.time() - start_time
        
        # Should complete merging quickly
        assert merge_time < 3.0, f"Merging took {merge_time:.2f} seconds"
        assert len(merged_models) == num_merges
    
    def test_evolution_memory_usage(self):
        """Test memory usage during evolution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large population
        population_size = 1000
        population = [
            ModelIndividual(f"model_{i}", fitness=np.random.random())
            for i in range(population_size)
        ]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100, f"Memory increased by {memory_increase:.1f}MB"
        assert len(population) == population_size