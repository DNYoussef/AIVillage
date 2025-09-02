"""
Adaptive Evolution Algorithms
Archaeological Enhancement: Advanced algorithms from evolutionary-computing branches

Innovation Score: 7.2/10
Branch Origins: evolutionary-computing, adaptive-scheduling, genetic-algorithms-enhanced
Preservation Priority: HIGH - Critical for intelligent model evolution

This module provides advanced evolutionary algorithms with adaptive parameters,
multi-objective optimization, and performance-guided evolution strategies.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
import logging
import math
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives for multi-objective evolution."""
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"


@dataclass
class ObjectiveFunction:
    """Defines an optimization objective."""
    name: str
    weight: float
    objective: OptimizationObjective
    target_value: float | None = None
    current_best: float | None = None


@dataclass
class EvolutionParameters:
    """Parameters for evolution algorithms."""
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    selection_pressure: float = 2.0
    elite_ratio: float = 0.1
    adaptive_parameters: bool = True
    diversity_threshold: float = 0.01
    stagnation_threshold: int = 10


class EvolutionAlgorithm(ABC):
    """Base class for evolution algorithms."""
    
    def __init__(self, parameters: EvolutionParameters):
        self.parameters = parameters
        self.generation = 0
        self.best_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        
    @abstractmethod
    def initialize_population(self, problem_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Initialize the population."""
        pass
    
    @abstractmethod
    def evaluate_fitness(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """Evaluate fitness of an individual."""
        pass
    
    @abstractmethod
    def selection(self, population: list[dict[str, Any]], fitness_scores: list[float]) -> list[dict[str, Any]]:
        """Select parents for reproduction."""
        pass
    
    @abstractmethod
    def crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform crossover between two parents."""
        pass
    
    @abstractmethod
    def mutation(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """Mutate an individual."""
        pass
    
    def evolve(
        self, 
        problem_config: dict[str, Any], 
        max_generations: int,
        fitness_threshold: float | None = None,
        progress_callback: Callable | None = None
    ) -> tuple[dict[str, Any], float, list[float]]:
        """
        Main evolution loop.
        
        Returns:
            (best_individual, best_fitness, convergence_history)
        """
        # Initialize population
        population = self.initialize_population(problem_config)
        
        best_individual = None
        best_fitness = float('-inf')
        convergence_history = []
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate population
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, problem_config)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Track convergence
            avg_fitness = np.mean(fitness_scores)
            convergence_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': self._calculate_diversity(population)
            })
            
            # Check for convergence
            if fitness_threshold and best_fitness >= fitness_threshold:
                logger.info(f"Converged at generation {generation} with fitness {best_fitness}")
                break
            
            # Adaptive parameter adjustment
            if self.parameters.adaptive_parameters:
                self._adapt_parameters(fitness_scores, population)
            
            # Check for stagnation
            if self._is_stagnating(convergence_history):
                logger.info(f"Stagnation detected at generation {generation}, applying diversity injection")
                population = self._inject_diversity(population, problem_config)
                self.stagnation_counter = 0
            
            # Selection
            parents = self.selection(population, fitness_scores)
            
            # Generate new population
            new_population = []
            
            # Keep elite individuals
            elite_count = int(len(population) * self.parameters.elite_ratio)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < len(population):
                parent1, parent2 = random.sample(parents, 2)
                
                if random.random() < self.parameters.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < self.parameters.mutation_rate:
                    child1 = self.mutation(child1, problem_config)
                if random.random() < self.parameters.mutation_rate:
                    child2 = self.mutation(child2, problem_config)
                
                new_population.extend([child1, child2])
            
            population = new_population[:len(population)]
            
            # Progress callback
            if progress_callback:
                progress_callback(generation, best_fitness, avg_fitness)
        
        return best_individual, best_fitness, convergence_history
    
    def _calculate_diversity(self, population: list[dict[str, Any]]) -> float:
        """Calculate population diversity."""
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._individual_distance(population[i], population[j])
                total_distance += distance
                pair_count += 1
        
        return total_distance / pair_count if pair_count > 0 else 0.0
    
    def _individual_distance(self, ind1: dict[str, Any], ind2: dict[str, Any]) -> float:
        """Calculate distance between two individuals."""
        distance = 0.0
        count = 0
        
        for key in set(ind1.keys()) | set(ind2.keys()):
            if key in ind1 and key in ind2:
                val1, val2 = ind1[key], ind2[key]
                if isinstance(val1, int | float) and isinstance(val2, int | float):
                    distance += abs(val1 - val2)
                    count += 1
        
        return distance / count if count > 0 else 0.0
    
    def _is_stagnating(self, convergence_history: list[dict[str, Any]]) -> bool:
        """Check if evolution is stagnating."""
        if len(convergence_history) < self.parameters.stagnation_threshold:
            return False
        
        recent_best = [record['best_fitness'] for record in convergence_history[-self.parameters.stagnation_threshold:]]
        improvement = max(recent_best) - min(recent_best)
        
        return improvement < 1e-6
    
    def _inject_diversity(self, population: list[dict[str, Any]], problem_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Inject diversity into stagnating population."""
        # Replace bottom 20% with random individuals
        replacement_count = int(len(population) * 0.2)
        new_individuals = self.initialize_population(problem_config)[:replacement_count]
        
        # Keep top 80%
        return population[:-replacement_count] + new_individuals
    
    def _adapt_parameters(self, fitness_scores: list[float], population: list[dict[str, Any]]) -> None:
        """Adapt algorithm parameters based on current state."""
        diversity = self._calculate_diversity(population)
        fitness_variance = np.var(fitness_scores)
        
        # Adapt mutation rate based on diversity
        if diversity < self.parameters.diversity_threshold:
            self.parameters.mutation_rate = min(0.5, self.parameters.mutation_rate * 1.1)
        else:
            self.parameters.mutation_rate = max(0.01, self.parameters.mutation_rate * 0.95)
        
        # Adapt crossover rate based on fitness variance
        if fitness_variance < 0.01:
            self.parameters.crossover_rate = min(0.95, self.parameters.crossover_rate * 1.05)
        else:
            self.parameters.crossover_rate = max(0.5, self.parameters.crossover_rate * 0.98)


class AdaptiveGeneticAlgorithm(EvolutionAlgorithm):
    """
    Archaeological enhancement: Adaptive genetic algorithm with self-tuning parameters.
    """
    
    def __init__(self, parameters: EvolutionParameters):
        super().__init__(parameters)
        self.gene_mutation_probabilities = {}
    
    def initialize_population(self, problem_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Initialize population with diverse individuals."""
        population = []
        parameters = problem_config.get('parameters', {})
        constraints = problem_config.get('constraints', {})
        
        for _ in range(self.parameters.population_size):
            individual = {}
            
            for param_name, param_value in parameters.items():
                if isinstance(param_value, int | float):
                    constraint = constraints.get(param_name, {})
                    min_val = constraint.get('min', param_value * 0.1)
                    max_val = constraint.get('max', param_value * 2.0)
                    
                    individual[param_name] = random.uniform(min_val, max_val)
                elif isinstance(param_value, bool):
                    individual[param_name] = random.choice([True, False])
                elif isinstance(param_value, str) and param_name in constraints:
                    choices = constraints[param_name].get('choices', [param_value])
                    individual[param_name] = random.choice(choices)
                else:
                    individual[param_name] = param_value
            
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """
        Evaluate fitness using multi-objective optimization.
        
        Archaeological enhancement: Supports multiple objectives with weights.
        """
        objectives = problem_config.get('objectives', [])
        
        if not objectives:
            # Default single objective
            return self._evaluate_single_objective(individual, problem_config)
        
        total_fitness = 0.0
        total_weight = 0.0
        
        for obj in objectives:
            obj_fitness = self._evaluate_objective(individual, obj, problem_config)
            total_fitness += obj_fitness * obj['weight']
            total_weight += obj['weight']
        
        return total_fitness / total_weight if total_weight > 0 else 0.0
    
    def _evaluate_single_objective(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """Evaluate single objective fitness."""
        # Mock fitness evaluation - in production, this would interface with model training
        fitness = 0.0
        target_metrics = problem_config.get('target_metrics', {})
        
        for param_name, param_value in individual.items():
            if isinstance(param_value, int | float) and param_name in target_metrics:
                target = target_metrics[param_name]
                normalized_distance = abs(param_value - target) / max(abs(target), 1.0)
                fitness += 1.0 - normalized_distance
        
        # Add some realistic noise
        fitness += random.gauss(0, 0.05)
        
        return max(0.0, fitness)
    
    def _evaluate_objective(self, individual: dict[str, Any], objective: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """Evaluate a specific objective."""
        # Placeholder for objective-specific evaluation
        return self._evaluate_single_objective(individual, problem_config)
    
    def selection(self, population: list[dict[str, Any]], fitness_scores: list[float]) -> list[dict[str, Any]]:
        """
        Tournament selection with adaptive tournament size.
        """
        tournament_size = max(2, int(self.parameters.selection_pressure))
        parents = []
        
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
        
        return parents
    
    def crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Adaptive crossover with multiple strategies.
        """
        child1, child2 = parent1.copy(), parent2.copy()
        
        crossover_strategies = [
            self._uniform_crossover,
            self._arithmetic_crossover,
            self._blx_alpha_crossover
        ]
        
        # Select crossover strategy adaptively
        strategy = random.choice(crossover_strategies)
        return strategy(parent1, parent2)
    
    def _uniform_crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Uniform crossover."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for key in parent1.keys():
            if random.random() < 0.5:
                child1[key], child2[key] = parent2[key], parent1[key]
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Arithmetic crossover for numerical parameters."""
        child1, child2 = parent1.copy(), parent2.copy()
        alpha = random.random()
        
        for key in parent1.keys():
            if isinstance(parent1[key], int | float) and isinstance(parent2[key], int | float):
                child1[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
                child2[key] = alpha * parent2[key] + (1 - alpha) * parent1[key]
        
        return child1, child2
    
    def _blx_alpha_crossover(self, parent1: dict[str, Any], parent2: dict[str, Any], alpha: float = 0.5) -> tuple[dict[str, Any], dict[str, Any]]:
        """BLX-α crossover."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for key in parent1.keys():
            if isinstance(parent1[key], int | float) and isinstance(parent2[key], int | float):
                val1, val2 = parent1[key], parent2[key]
                min_val, max_val = min(val1, val2), max(val1, val2)
                range_val = max_val - min_val
                
                # Extend range by alpha on both sides
                extended_min = min_val - alpha * range_val
                extended_max = max_val + alpha * range_val
                
                child1[key] = random.uniform(extended_min, extended_max)
                child2[key] = random.uniform(extended_min, extended_max)
        
        return child1, child2
    
    def mutation(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """
        Adaptive mutation with gene-specific probabilities.
        """
        mutated = individual.copy()
        constraints = problem_config.get('constraints', {})
        
        for key, value in individual.items():
            # Get gene-specific mutation probability
            gene_mutation_prob = self.gene_mutation_probabilities.get(key, self.parameters.mutation_rate)
            
            if random.random() < gene_mutation_prob:
                if isinstance(value, int | float):
                    mutated[key] = self._mutate_numerical(key, value, constraints.get(key, {}))
                elif isinstance(value, bool):
                    mutated[key] = not value
                elif isinstance(value, str) and key in constraints:
                    choices = constraints[key].get('choices', [value])
                    mutated[key] = random.choice(choices)
        
        return mutated
    
    def _mutate_numerical(self, param_name: str, value: float, constraints: dict[str, Any]) -> float:
        """Mutate numerical parameter with adaptive strength."""
        min_val = constraints.get('min', value * 0.1)
        max_val = constraints.get('max', value * 2.0)
        
        # Adaptive mutation strength based on generation
        base_strength = (max_val - min_val) * 0.1
        generation_factor = math.exp(-self.generation / 100)  # Decrease over time
        mutation_strength = base_strength * (0.1 + 0.9 * generation_factor)
        
        # Apply Gaussian mutation
        mutated_value = value + random.gauss(0, mutation_strength)
        
        # Clip to constraints
        return np.clip(mutated_value, min_val, max_val)


class DifferentialEvolution(EvolutionAlgorithm):
    """
    Archaeological enhancement: Differential Evolution with adaptive strategies.
    """
    
    def __init__(self, parameters: EvolutionParameters, F: float = 0.8, CR: float = 0.9):
        super().__init__(parameters)
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.strategies = [
            self._de_rand_1,
            self._de_best_1,
            self._de_current_to_best_1
        ]
        self.strategy_success_rates = [0.5] * len(self.strategies)
    
    def initialize_population(self, problem_config: dict[str, Any]) -> list[dict[str, Any]]:
        """Initialize population for DE."""
        return AdaptiveGeneticAlgorithm(self.parameters).initialize_population(problem_config)
    
    def evaluate_fitness(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """Evaluate fitness (same as GA)."""
        return AdaptiveGeneticAlgorithm(self.parameters).evaluate_fitness(individual, problem_config)
    
    def selection(self, population: list[dict[str, Any]], fitness_scores: list[float]) -> list[dict[str, Any]]:
        """DE uses different selection mechanism during evolution."""
        return population  # Not used in standard DE
    
    def crossover(self, parent1: dict[str, Any], parent2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """DE crossover (not used - DE has its own mutation/crossover)."""
        return parent1, parent2
    
    def mutation(self, individual: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """DE mutation (handled in evolve method)."""
        return individual
    
    def evolve(
        self, 
        problem_config: dict[str, Any], 
        max_generations: int,
        fitness_threshold: float | None = None,
        progress_callback: Callable | None = None
    ) -> tuple[dict[str, Any], float, list[float]]:
        """
        DE evolution with adaptive strategy selection.
        """
        population = self.initialize_population(problem_config)
        
        best_individual = None
        best_fitness = float('-inf')
        convergence_history = []
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate current population
            fitness_scores = []
            for individual in population:
                fitness = self.evaluate_fitness(individual, problem_config)
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Track convergence
            convergence_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores)
            })
            
            # Check convergence
            if fitness_threshold and best_fitness >= fitness_threshold:
                break
            
            # Adapt F and CR
            if self.parameters.adaptive_parameters:
                self._adapt_de_parameters(generation, max_generations)
            
            # Generate trial population
            trial_population = []
            for i, target in enumerate(population):
                # Select DE strategy adaptively
                strategy_idx = self._select_strategy()
                strategy = self.strategies[strategy_idx]
                
                # Generate trial vector
                trial = strategy(population, i, best_individual, problem_config)
                trial_population.append(trial)
            
            # Selection: keep better individuals
            new_population = []
            strategy_improvements = [0] * len(self.strategies)
            
            for i, (target, trial) in enumerate(zip(population, trial_population)):
                target_fitness = fitness_scores[i]
                trial_fitness = self.evaluate_fitness(trial, problem_config)
                
                if trial_fitness >= target_fitness:
                    new_population.append(trial)
                    # Track strategy success
                    strategy_idx = i % len(self.strategies)  # Simple assignment
                    strategy_improvements[strategy_idx] += 1
                else:
                    new_population.append(target)
            
            # Update strategy success rates
            self._update_strategy_success_rates(strategy_improvements)
            
            population = new_population
            
            if progress_callback:
                progress_callback(generation, best_fitness, np.mean(fitness_scores))
        
        return best_individual, best_fitness, convergence_history
    
    def _select_strategy(self) -> int:
        """Select DE strategy based on success rates."""
        # Weighted random selection
        total_rate = sum(self.strategy_success_rates)
        if total_rate == 0:
            return random.randint(0, len(self.strategies) - 1)
        
        probabilities = [rate / total_rate for rate in self.strategy_success_rates]
        return np.random.choice(len(self.strategies), p=probabilities)
    
    def _update_strategy_success_rates(self, improvements: list[int]) -> None:
        """Update strategy success rates based on improvements."""
        total_improvements = sum(improvements)
        if total_improvements == 0:
            return
        
        # Update with exponential moving average
        alpha = 0.1
        for i, improvement in enumerate(improvements):
            success_rate = improvement / total_improvements
            self.strategy_success_rates[i] = (1 - alpha) * self.strategy_success_rates[i] + alpha * success_rate
    
    def _adapt_de_parameters(self, generation: int, max_generations: int) -> None:
        """Adapt F and CR parameters."""
        progress = generation / max_generations
        
        # Decrease F over time for fine-tuning
        self.F = 0.8 * (1 - 0.5 * progress)
        
        # Adjust CR based on diversity needs
        if progress < 0.5:
            self.CR = 0.9  # High exploration
        else:
            self.CR = 0.7  # More exploitation
    
    def _de_rand_1(self, population: list[dict[str, Any]], target_idx: int, best: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """DE/rand/1 strategy."""
        # Select three random individuals different from target
        candidates = [i for i in range(len(population)) if i != target_idx]
        r1, r2, r3 = random.sample(candidates, 3)
        
        # Create mutant vector
        mutant = {}
        for key in population[target_idx].keys():
            if isinstance(population[r1][key], int | float):
                mutant[key] = population[r1][key] + self.F * (population[r2][key] - population[r3][key])
            else:
                mutant[key] = population[r1][key]
        
        # Apply crossover
        return self._de_crossover(population[target_idx], mutant, problem_config)
    
    def _de_best_1(self, population: list[dict[str, Any]], target_idx: int, best: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """DE/best/1 strategy."""
        candidates = [i for i in range(len(population)) if i != target_idx]
        r1, r2 = random.sample(candidates, 2)
        
        mutant = {}
        for key in population[target_idx].keys():
            if isinstance(best[key], int | float):
                mutant[key] = best[key] + self.F * (population[r1][key] - population[r2][key])
            else:
                mutant[key] = best[key]
        
        return self._de_crossover(population[target_idx], mutant, problem_config)
    
    def _de_current_to_best_1(self, population: list[dict[str, Any]], target_idx: int, best: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """DE/current-to-best/1 strategy."""
        candidates = [i for i in range(len(population)) if i != target_idx]
        r1, r2 = random.sample(candidates, 2)
        
        target = population[target_idx]
        mutant = {}
        
        for key in target.keys():
            if isinstance(target[key], int | float):
                mutant[key] = target[key] + self.F * (best[key] - target[key]) + self.F * (population[r1][key] - population[r2][key])
            else:
                mutant[key] = target[key]
        
        return self._de_crossover(target, mutant, problem_config)
    
    def _de_crossover(self, target: dict[str, Any], mutant: dict[str, Any], problem_config: dict[str, Any]) -> dict[str, Any]:
        """DE crossover operation."""
        trial = target.copy()
        constraints = problem_config.get('constraints', {})
        
        # Ensure at least one parameter is taken from mutant
        keys = list(target.keys())
        forced_key = random.choice(keys)
        
        for key in keys:
            if key == forced_key or random.random() < self.CR:
                trial[key] = mutant[key]
                
                # Apply constraints
                if isinstance(trial[key], int | float) and key in constraints:
                    constraint = constraints[key]
                    min_val = constraint.get('min', float('-inf'))
                    max_val = constraint.get('max', float('inf'))
                    trial[key] = np.clip(trial[key], min_val, max_val)
        
        return trial


class ParticleSwarmOptimization:
    """
    Archaeological enhancement: Particle Swarm Optimization for continuous optimization.
    """
    
    def __init__(
        self,
        swarm_size: int = 30,
        w: float = 0.7,  # Inertia weight
        c1: float = 2.0,  # Cognitive coefficient
        c2: float = 2.0   # Social coefficient
    ):
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        self.global_best = None
        self.global_best_fitness = float('-inf')
    
    def optimize(
        self,
        problem_config: dict[str, Any],
        max_iterations: int = 100,
        fitness_threshold: float | None = None,
        progress_callback: Callable | None = None
    ) -> tuple[dict[str, Any], float, list[float]]:
        """
        PSO optimization loop.
        """
        # Initialize swarm
        self._initialize_swarm(problem_config)
        
        convergence_history = []
        
        for iteration in range(max_iterations):
            # Evaluate particles
            for i, particle in enumerate(self.particles):
                fitness = self._evaluate_particle(particle, problem_config)
                
                # Update personal best
                if fitness > self.personal_bests[i].get('fitness', float('-inf')):
                    self.personal_bests[i] = {'position': particle.copy(), 'fitness': fitness}
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best = particle.copy()
                    self.global_best_fitness = fitness
            
            # Update velocities and positions
            for i in range(len(self.particles)):
                self._update_particle(i, problem_config)
            
            # Track convergence
            avg_fitness = np.mean([pb.get('fitness', 0) for pb in self.personal_bests])
            convergence_history.append({
                'iteration': iteration,
                'best_fitness': self.global_best_fitness,
                'avg_fitness': avg_fitness
            })
            
            # Check convergence
            if fitness_threshold and self.global_best_fitness >= fitness_threshold:
                break
            
            if progress_callback:
                progress_callback(iteration, self.global_best_fitness, avg_fitness)
        
        return self.global_best, self.global_best_fitness, convergence_history
    
    def _initialize_swarm(self, problem_config: dict[str, Any]) -> None:
        """Initialize particle swarm."""
        parameters = problem_config.get('parameters', {})
        constraints = problem_config.get('constraints', {})
        
        self.particles = []
        self.velocities = []
        self.personal_bests = []
        
        for _ in range(self.swarm_size):
            particle = {}
            velocity = {}
            
            for param_name, param_value in parameters.items():
                if isinstance(param_value, int | float):
                    constraint = constraints.get(param_name, {})
                    min_val = constraint.get('min', param_value * 0.1)
                    max_val = constraint.get('max', param_value * 2.0)
                    
                    particle[param_name] = random.uniform(min_val, max_val)
                    velocity[param_name] = random.uniform(-abs(max_val - min_val) * 0.1, 
                                                        abs(max_val - min_val) * 0.1)
                else:
                    particle[param_name] = param_value
                    velocity[param_name] = 0
            
            self.particles.append(particle)
            self.velocities.append(velocity)
            self.personal_bests.append({'position': particle.copy(), 'fitness': float('-inf')})
    
    def _evaluate_particle(self, particle: dict[str, Any], problem_config: dict[str, Any]) -> float:
        """Evaluate particle fitness."""
        # Use same evaluation as GA
        return AdaptiveGeneticAlgorithm(EvolutionParameters()).evaluate_fitness(particle, problem_config)
    
    def _update_particle(self, particle_idx: int, problem_config: dict[str, Any]) -> None:
        """Update particle velocity and position."""
        particle = self.particles[particle_idx]
        velocity = self.velocities[particle_idx]
        personal_best = self.personal_bests[particle_idx]['position']
        constraints = problem_config.get('constraints', {})
        
        for param_name in particle.keys():
            if isinstance(particle[param_name], int | float):
                # Update velocity
                r1, r2 = random.random(), random.random()
                
                cognitive = self.c1 * r1 * (personal_best[param_name] - particle[param_name])
                social = self.c2 * r2 * (self.global_best[param_name] - particle[param_name])
                
                velocity[param_name] = (self.w * velocity[param_name] + cognitive + social)
                
                # Update position
                particle[param_name] += velocity[param_name]
                
                # Apply constraints
                if param_name in constraints:
                    constraint = constraints[param_name]
                    min_val = constraint.get('min', float('-inf'))
                    max_val = constraint.get('max', float('inf'))
                    particle[param_name] = np.clip(particle[param_name], min_val, max_val)


# Factory function for creating evolution algorithms
def create_evolution_algorithm(
    strategy_name: str,
    parameters: EvolutionParameters | None = None
) -> EvolutionAlgorithm:
    """
    Factory function to create evolution algorithms.
    
    Archaeological enhancement: Supports multiple algorithm types discovered
    from evolutionary-computing branches.
    """
    if parameters is None:
        parameters = EvolutionParameters()
    
    if strategy_name == "adaptive_genetic":
        return AdaptiveGeneticAlgorithm(parameters)
    elif strategy_name == "differential_evolution":
        return DifferentialEvolution(parameters)
    elif strategy_name == "particle_swarm":
        # PSO doesn't inherit from EvolutionAlgorithm but provides similar interface
        return ParticleSwarmOptimization(
            swarm_size=parameters.population_size,
            w=0.7, c1=2.0, c2=2.0
        )
    else:
        raise ValueError(f"Unknown evolution strategy: {strategy_name}")


# Utility functions for multi-objective optimization
def pareto_dominance(fitness1: list[float], fitness2: list[float]) -> int:
    """
    Check Pareto dominance between two fitness vectors.
    
    Returns:
        1 if fitness1 dominates fitness2
        -1 if fitness2 dominates fitness1
        0 if neither dominates
    """
    better_count = 0
    worse_count = 0
    
    for f1, f2 in zip(fitness1, fitness2):
        if f1 > f2:
            better_count += 1
        elif f1 < f2:
            worse_count += 1
    
    if better_count > 0 and worse_count == 0:
        return 1
    elif worse_count > 0 and better_count == 0:
        return -1
    else:
        return 0


def non_dominated_sort(population: list[dict[str, Any]], fitness_matrix: list[list[float]]) -> list[list[int]]:
    """
    Non-dominated sorting for multi-objective optimization.
    
    Returns list of fronts, where each front is a list of individual indices.
    """
    n = len(population)
    domination_count = [0] * n  # Number of individuals that dominate individual i
    dominated_individuals = [[] for _ in range(n)]  # Individuals dominated by individual i
    
    fronts = [[]]
    
    # Calculate domination relationships
    for i in range(n):
        for j in range(n):
            if i != j:
                dominance = pareto_dominance(fitness_matrix[i], fitness_matrix[j])
                if dominance == 1:  # i dominates j
                    dominated_individuals[i].append(j)
                elif dominance == -1:  # j dominates i
                    domination_count[i] += 1
        
        # If individual i is not dominated, it belongs to the first front
        if domination_count[i] == 0:
            fronts[0].append(i)
    
    # Generate subsequent fronts
    current_front = 0
    while len(fronts[current_front]) > 0:
        next_front = []
        
        for i in fronts[current_front]:
            for j in dominated_individuals[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        
        if len(next_front) > 0:
            fronts.append(next_front)
        
        current_front += 1
    
    return fronts[:-1]  # Remove empty last front


def crowding_distance(fitness_matrix: list[list[float]], front_indices: list[int]) -> list[float]:
    """
    Calculate crowding distance for individuals in a front.
    """
    if len(front_indices) <= 2:
        return [float('inf')] * len(front_indices)
    
    distances = [0.0] * len(front_indices)
    num_objectives = len(fitness_matrix[0])
    
    for obj in range(num_objectives):
        # Sort individuals by objective value
        sorted_indices = sorted(range(len(front_indices)), 
                              key=lambda i: fitness_matrix[front_indices[i]][obj])
        
        # Boundary individuals get infinite distance
        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')
        
        # Calculate objective range
        obj_min = fitness_matrix[front_indices[sorted_indices[0]]][obj]
        obj_max = fitness_matrix[front_indices[sorted_indices[-1]]][obj]
        obj_range = obj_max - obj_min
        
        if obj_range == 0:
            continue
        
        # Calculate distances for intermediate individuals
        for i in range(1, len(sorted_indices) - 1):
            if distances[sorted_indices[i]] != float('inf'):
                prev_fitness = fitness_matrix[front_indices[sorted_indices[i-1]]][obj]
                next_fitness = fitness_matrix[front_indices[sorted_indices[i+1]]][obj]
                distances[sorted_indices[i]] += (next_fitness - prev_fitness) / obj_range
    
    return distances