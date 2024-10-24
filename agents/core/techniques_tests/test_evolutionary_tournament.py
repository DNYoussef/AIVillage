"""Unit tests for Evolutionary Tournament technique."""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from ....magi.techniques.evolutionary_tournament import (
    EvolutionaryTournamentTechnique,
    Individual,
    Population,
    TechniqueResult
)
from ....magi.core.exceptions import ToolError

@pytest.fixture
def technique():
    """Create an Evolutionary Tournament technique instance."""
    return EvolutionaryTournamentTechnique()

@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    mock = AsyncMock()
    mock.llm_response = AsyncMock()
    return mock

@pytest.fixture
def sample_population():
    """Create a sample population of solutions."""
    return Population(
        individuals=[
            Individual(
                id="ind1",
                genes={"param1": 0.5, "param2": 0.8},
                fitness=0.9,
                solution="Use caching with LRU",
                generation=1
            ),
            Individual(
                id="ind2",
                genes={"param1": 0.3, "param2": 0.6},
                fitness=0.7,
                solution="Use simple caching",
                generation=1
            ),
            Individual(
                id="ind3",
                genes={"param1": 0.7, "param2": 0.4},
                fitness=0.8,
                solution="Use distributed cache",
                generation=1
            )
        ],
        generation=1,
        best_fitness=0.9,
        avg_fitness=0.8
    )

@pytest.mark.asyncio
async def test_initialization(technique):
    """Test technique initialization."""
    assert technique.name == "Evolutionary-Tournament"
    assert "evolution" in technique.thought.lower()
    assert technique.population is None
    assert technique.best_individual is None
    assert technique.generation == 0

@pytest.mark.asyncio
async def test_initial_population(technique, mock_agent):
    """Test generation of initial population."""
    mock_agent.llm_response.side_effect = [
        # First individual
        Mock(content="""
        Solution:
        Use hash table
        Parameters:
        param1: 0.5
        param2: 0.8
        Fitness: 0.9
        """),
        # Second individual
        Mock(content="""
        Solution:
        Use binary tree
        Parameters:
        param1: 0.3
        param2: 0.6
        Fitness: 0.7
        """)
    ]
    
    await technique._initialize_population(2)
    
    assert technique.population is not None
    assert len(technique.population.individuals) == 2
    assert all(ind.genes for ind in technique.population.individuals)
    assert all(ind.fitness > 0 for ind in technique.population.individuals)

@pytest.mark.asyncio
async def test_tournament_selection(technique, sample_population):
    """Test tournament selection process."""
    technique.population = sample_population
    technique.tournament_size = 2
    
    selected = technique._tournament_select()
    
    assert selected is not None
    assert selected.fitness >= min(ind.fitness for ind in sample_population.individuals)

@pytest.mark.asyncio
async def test_crossover(technique):
    """Test crossover operation between individuals."""
    parent1 = Individual(
        id="p1",
        genes={"param1": 0.1, "param2": 0.2},
        fitness=0.8,
        solution="Solution 1",
        generation=1
    )
    parent2 = Individual(
        id="p2",
        genes={"param1": 0.8, "param2": 0.9},
        fitness=0.7,
        solution="Solution 2",
        generation=1
    )
    
    child = technique._crossover(parent1, parent2)
    
    assert child is not None
    assert all(0 <= v <= 1 for v in child.genes.values())
    # Child genes should be between parents
    assert all(
        min(parent1.genes[k], parent2.genes[k]) <= child.genes[k] <= max(parent1.genes[k], parent2.genes[k])
        for k in child.genes
    )

@pytest.mark.asyncio
async def test_mutation(technique):
    """Test mutation operation."""
    individual = Individual(
        id="test",
        genes={"param1": 0.5, "param2": 0.5},
        fitness=0.8,
        solution="Test solution",
        generation=1
    )
    
    original_genes = individual.genes.copy()
    mutated = technique._mutate(individual)
    
    assert mutated is not None
    assert any(
        original_genes[k] != mutated.genes[k]
        for k in original_genes
    )
    assert all(0 <= v <= 1 for v in mutated.genes.values())

@pytest.mark.asyncio
async def test_evolution_step(technique, mock_agent, sample_population):
    """Test single evolution step."""
    technique.population = sample_population
    
    mock_agent.llm_response.return_value = Mock(content="""
    Solution:
    Evolved solution
    Parameters:
    param1: 0.6
    param2: 0.7
    Fitness: 0.85
    """)
    
    await technique._evolve_step()
    
    assert technique.generation > sample_population.generation
    assert len(technique.population.individuals) == len(sample_population.individuals)
    assert technique.population.best_fitness >= sample_population.best_fitness

@pytest.mark.asyncio
async def test_error_handling(technique, mock_agent):
    """Test error handling in evolution process."""
    # Mock error in language model
    mock_agent.llm_response.side_effect = Exception("Test error")
    
    with pytest.raises(ToolError) as exc_info:
        await technique.apply(mock_agent, "Test task")
    assert "Test error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_fitness_evaluation(technique, mock_agent):
    """Test fitness evaluation of individuals."""
    individual = Individual(
        id="test",
        genes={"param1": 0.5, "param2": 0.5},
        fitness=0.0,  # Initial fitness
        solution="Test solution",
        generation=1
    )
    
    mock_agent.llm_response.return_value = Mock(content="""
    Evaluation:
    Fitness: 0.85
    Reasoning: Good performance characteristics
    """)
    
    fitness = await technique._evaluate_fitness(individual)
    
    assert 0 <= fitness <= 1
    assert fitness > 0  # Should be updated from initial 0

@pytest.mark.asyncio
async def test_population_diversity(technique, sample_population):
    """Test maintenance of population diversity."""
    technique.population = sample_population
    
    # Calculate initial diversity
    initial_diversity = technique._calculate_diversity()
    
    # Perform evolution step with diversity preservation
    technique.diversity_threshold = 0.5
    await technique._evolve_step()
    
    # Check diversity maintained
    final_diversity = technique._calculate_diversity()
    assert final_diversity >= technique.diversity_threshold

@pytest.mark.asyncio
async def test_elitism(technique, sample_population):
    """Test elitism in evolution process."""
    technique.population = sample_population
    best_individual = max(
        sample_population.individuals,
        key=lambda x: x.fitness
    )
    
    await technique._evolve_step()
    
    # Best individual should be preserved
    assert any(
        ind.fitness >= best_individual.fitness
        for ind in technique.population.individuals
    )

@pytest.mark.asyncio
async def test_convergence_check(technique, sample_population):
    """Test convergence checking."""
    technique.population = sample_population
    technique.convergence_threshold = 0.01
    technique.convergence_generations = 3
    
    # Simulate convergence
    for _ in range(technique.convergence_generations):
        technique.fitness_history.append(0.9)  # Same fitness
    
    assert technique._check_convergence()

@pytest.mark.asyncio
async def test_adaptive_parameters(technique, sample_population):
    """Test adaptation of evolutionary parameters."""
    technique.population = sample_population
    initial_mutation_rate = technique.mutation_rate
    
    # Simulate stagnation
    for _ in range(5):
        technique.fitness_history.append(0.8)  # Same fitness
    
    await technique._adapt_parameters()
    
    # Parameters should be adjusted
    assert technique.mutation_rate != initial_mutation_rate

@pytest.mark.asyncio
async def test_invalid_genes(technique, mock_agent):
    """Test handling of invalid gene values."""
    mock_agent.llm_response.return_value = Mock(content="""
    Solution:
    Invalid solution
    Parameters:
    param1: 1.5  # Invalid: > 1.0
    param2: 0.5
    Fitness: 0.8
    """)
    
    with pytest.raises(ToolError) as exc_info:
        await technique._initialize_population(1)
    assert "gene" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_population_size_maintenance(technique, mock_agent):
    """Test maintenance of population size."""
    initial_size = 5
    await technique._initialize_population(initial_size)
    
    for _ in range(3):  # Multiple evolution steps
        await technique._evolve_step()
        assert len(technique.population.individuals) == initial_size
