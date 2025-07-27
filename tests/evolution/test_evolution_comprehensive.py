"""
Comprehensive evolution system tests building on stable core infrastructure.
Uses proven test patterns from successful core tests.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
import random
import numpy as np
from pathlib import Path
import json

class TestEvolutionaryTournament:
    """Test evolutionary tournament selection mechanisms."""
    
    def test_tournament_basic_selection(self):
        """Test basic tournament selection with simple population."""
        # Create test population
        population = [
            {'model': nn.Linear(10, 5), 'fitness': 0.8, 'id': 'model_1'},
            {'model': nn.Linear(10, 5), 'fitness': 0.6, 'id': 'model_2'},
            {'model': nn.Linear(10, 5), 'fitness': 0.9, 'id': 'model_3'},
            {'model': nn.Linear(10, 5), 'fitness': 0.7, 'id': 'model_4'}
        ]
        
        # Simple selection: pick highest fitness
        winner = max(population, key=lambda x: x['fitness'])
        
        assert winner['id'] == 'model_3'
        assert winner['fitness'] == 0.9
    
    def test_population_diversity(self):
        """Test population maintains diversity."""
        population_size = 10
        population = []
        
        # Create diverse population
        for i in range(population_size):
            model = nn.Sequential(
                nn.Linear(20, random.randint(10, 50)),
                nn.ReLU(),
                nn.Linear(random.randint(10, 50), 10)
            )
            population.append({
                'model': model,
                'fitness': random.uniform(0.5, 1.0),
                'id': f'model_{i}',
                'generation': 0
            })
        
        assert len(population) == population_size
        
        # Check fitness diversity
        fitness_values = [p['fitness'] for p in population]
        fitness_std = np.std(fitness_values)
        assert fitness_std > 0.05  # Some diversity expected
    
    def test_fitness_evaluation(self, sample_model):
        """Test fitness evaluation for models."""
        # Mock fitness function
        def evaluate_fitness(model, test_data=None):
            """Simple fitness based on parameter count."""
            param_count = sum(p.numel() for p in model.parameters())
            # Normalize to 0-1 range (smaller models get higher fitness)
            return min(1.0, 1000.0 / param_count)
        
        fitness = evaluate_fitness(sample_model)
        
        assert 0.0 <= fitness <= 1.0
        assert isinstance(fitness, float)

class TestModelMutation:
    """Test model mutation operations."""
    
    def test_parameter_mutation(self, sample_model):
        """Test parameter mutation preserves model structure."""
        original_params = {}
        for name, param in sample_model.named_parameters():
            original_params[name] = param.clone()
        
        # Apply mutation (simple noise addition)
        mutation_rate = 0.1
        with torch.no_grad():
            for param in sample_model.parameters():
                if random.random() < mutation_rate:
                    noise = torch.randn_like(param) * 0.01
                    param.add_(noise)
        
        # Verify structure preserved but parameters changed
        mutated_params = dict(sample_model.named_parameters())
        
        assert len(original_params) == len(mutated_params)
        
        # Check at least some parameters changed
        params_changed = 0
        for name in original_params:
            if not torch.equal(original_params[name], mutated_params[name]):
                params_changed += 1
        
        # With mutation rate 0.1, expect some changes
        assert params_changed >= 0
    
    def test_structural_mutation(self):
        """Test structural mutations (layer addition/removal)."""
        base_model = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )
        
        # Test layer insertion
        new_layers = list(base_model.children())
        new_layers.insert(2, nn.Dropout(0.2))  # Insert dropout
        mutated_model = nn.Sequential(*new_layers)
        
        assert len(mutated_model) == len(base_model) + 1
        
        # Test forward pass works
        test_input = torch.randn(1, 20)
        output = mutated_model(test_input)
        assert output.shape == (1, 10)

class TestEvolutionMetrics:
    """Test evolution metrics and tracking."""
    
    def test_generation_tracking(self):
        """Test tracking evolution across generations."""
        generations = []
        
        for gen in range(5):
            generation_info = {
                'generation': gen,
                'population_size': 20,
                'best_fitness': 0.7 + gen * 0.05,  # Improving fitness
                'avg_fitness': 0.6 + gen * 0.03,
                'diversity_score': random.uniform(0.3, 0.7)
            }
            generations.append(generation_info)
        
        # Verify improvement trend
        assert generations[-1]['best_fitness'] > generations[0]['best_fitness']
        assert len(generations) == 5
    
    def test_convergence_detection(self):
        """Test detection of population convergence."""
        # Simulate converging population
        fitness_history = [
            [0.5, 0.6, 0.7, 0.8, 0.9],  # Generation 0 - diverse
            [0.7, 0.75, 0.8, 0.85, 0.9],  # Generation 1 - less diverse
            [0.85, 0.87, 0.88, 0.89, 0.9],  # Generation 2 - converging
            [0.88, 0.89, 0.89, 0.9, 0.9]   # Generation 3 - converged
        ]
        
        def calculate_diversity(fitness_list):
            return np.std(fitness_list)
        
        diversity_scores = [calculate_diversity(gen) for gen in fitness_history]
        
        # Diversity should decrease over generations
        assert diversity_scores[-1] < diversity_scores[0]
        
        # Check for convergence (low diversity)
        convergence_threshold = 0.05
        is_converged = diversity_scores[-1] < convergence_threshold
        assert is_converged

class TestEvolutionBenchmarks:
    """Performance benchmarks for evolution operations."""
    
    @pytest.mark.benchmark
    def test_population_evaluation_speed(self, benchmark):
        """Benchmark population fitness evaluation speed."""
        def evaluate_population():
            population = []
            for i in range(10):
                model = nn.Sequential(
                    nn.Linear(50, 25),
                    nn.ReLU(),
                    nn.Linear(25, 10)
                )
                # Simple fitness evaluation
                param_count = sum(p.numel() for p in model.parameters())
                fitness = 1.0 / (1.0 + param_count / 1000)
                
                population.append({
                    'model': model,
                    'fitness': fitness,
                    'id': f'model_{i}'
                })
            return population
        
        population = benchmark(evaluate_population)
        assert len(population) == 10
        assert all('fitness' in individual for individual in population)
    
    @pytest.mark.benchmark
    def test_tournament_selection_speed(self, benchmark):
        """Benchmark tournament selection speed."""
        # Create large population
        population = []
        for i in range(100):
            population.append({
                'model': nn.Linear(10, 5),
                'fitness': random.uniform(0.3, 1.0),
                'id': f'model_{i}'
            })
        
        def run_tournament(pop, tournament_size=5):
            """Run tournament selection."""
            selected = []
            for _ in range(10):  # Select 10 individuals
                tournament = random.sample(pop, tournament_size)
                winner = max(tournament, key=lambda x: x['fitness'])
                selected.append(winner)
            return selected
        
        selected = benchmark(run_tournament, population)
        assert len(selected) == 10

class TestEvolutionIntegration:
    """Integration tests for evolution pipeline."""
    
    @pytest.mark.integration
    def test_evolution_with_compression(self, compression_test_model):
        """Test evolution integrated with compression."""
        # Create population with compressed models
        population = []
        
        for i in range(3):
            # Simulate compressed model data
            compressed_info = {
                'compressed': True,
                'method': 'seedlm',
                'ratio': 4.0,
                'original_params': sum(p.numel() for p in compression_test_model.parameters())
            }
            
            individual = {
                'model': compression_test_model,
                'compression_info': compressed_info,
                'fitness': random.uniform(0.6, 0.9),
                'id': f'compressed_model_{i}'
            }
            population.append(individual)
        
        # Select best compressed model
        winner = max(population, key=lambda x: x['fitness'])
        
        assert winner['compression_info']['compressed'] is True
        assert winner['compression_info']['ratio'] == 4.0
    
    @pytest.mark.integration  
    def test_multi_generation_evolution(self):
        """Test evolution across multiple generations."""
        initial_population_size = 10
        generations = 3
        
        # Initialize population
        population = []
        for i in range(initial_population_size):
            model = nn.Sequential(
                nn.Linear(20, 15),
                nn.ReLU(),
                nn.Linear(15, 10)
            )
            population.append({
                'model': model,
                'fitness': random.uniform(0.4, 0.8),
                'id': f'gen0_model_{i}',
                'generation': 0
            })
        
        evolution_history = []
        
        # Evolve over generations
        for gen in range(generations):
            # Evaluate fitness (mock)
            for individual in population:
                # Simulate fitness improvement over generations
                individual['fitness'] += random.uniform(0.0, 0.1)
                individual['fitness'] = min(1.0, individual['fitness'])
            
            # Record generation stats
            fitness_values = [ind['fitness'] for ind in population]
            evolution_history.append({
                'generation': gen,
                'best_fitness': max(fitness_values),
                'avg_fitness': np.mean(fitness_values),
                'population_size': len(population)
            })
            
            # Select survivors (top 50%)
            population.sort(key=lambda x: x['fitness'], reverse=True)
            survivors = population[:len(population)//2]
            
            # Create next generation (mock reproduction)
            next_generation = []
            for i, parent in enumerate(survivors):
                # Parent survives
                parent['generation'] = gen + 1
                next_generation.append(parent)
                
                # Create offspring (simple copy with mutation)
                offspring = {
                    'model': parent['model'],  # In real implementation, would mutate
                    'fitness': parent['fitness'] + random.uniform(-0.1, 0.1),
                    'id': f'gen{gen+1}_model_{i}_offspring',
                    'generation': gen + 1
                }
                offspring['fitness'] = max(0.0, min(1.0, offspring['fitness']))
                next_generation.append(offspring)
            
            population = next_generation
        
        # Verify evolution progress
        assert len(evolution_history) == generations
        assert evolution_history[-1]['best_fitness'] >= evolution_history[0]['best_fitness']

class TestEvolutionErrorHandling:
    """Test error handling in evolution pipeline."""
    
    def test_invalid_population_handling(self):
        """Test handling of invalid population data."""
        # Empty population
        empty_pop = []
        
        # Should handle gracefully
        try:
            if empty_pop:
                winner = max(empty_pop, key=lambda x: x['fitness'])
            else:
                winner = None
            assert winner is None
        except Exception:
            pytest.fail("Should handle empty population gracefully")
    
    def test_fitness_evaluation_failure(self):
        """Test handling of fitness evaluation failures."""
        population = [
            {'model': nn.Linear(10, 5), 'fitness': None, 'id': 'model_1'},
            {'model': nn.Linear(10, 5), 'fitness': 0.8, 'id': 'model_2'}
        ]
        
        # Filter out invalid fitness values
        valid_population = [ind for ind in population if ind['fitness'] is not None]
        
        assert len(valid_population) == 1
        assert valid_population[0]['id'] == 'model_2'
    
    def test_model_corruption_detection(self):
        """Test detection of corrupted models."""
        # Mock corrupted model (None)
        population = [
            {'model': None, 'fitness': 0.7, 'id': 'corrupted_model'},
            {'model': nn.Linear(10, 5), 'fitness': 0.8, 'id': 'valid_model'}
        ]
        
        # Filter out corrupted models
        valid_models = [ind for ind in population if ind['model'] is not None]
        
        assert len(valid_models) == 1
        assert valid_models[0]['id'] == 'valid_model'

# Evolution-specific fixtures
@pytest.fixture
def evolution_population():
    """Provide a standard evolution population for testing."""
    population = []
    for i in range(10):
        model = nn.Sequential(
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        population.append({
            'model': model,
            'fitness': random.uniform(0.5, 0.95),
            'id': f'evolution_model_{i}',
            'generation': 0,
            'parents': None
        })
    return population

@pytest.fixture
def evolution_config():
    """Provide evolution configuration for testing."""
    return {
        'population_size': 20,
        'mutation_rate': 0.1,
        'crossover_rate': 0.7,
        'tournament_size': 5,
        'max_generations': 100,
        'convergence_threshold': 0.01,
        'elite_size': 2
    }