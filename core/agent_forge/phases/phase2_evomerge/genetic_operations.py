"""
Genetic operations for evolutionary optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import copy
import random

class GeneticOperations:
    """Genetic operations for model evolution."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.mutation_strength = config.get('mutation_strength', 0.05)
        self.crossover_rate = config.get('crossover_rate', 0.7)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    def crossover(self, parent1: nn.Module, parent2: nn.Module, method: str = 'uniform') -> Tuple[nn.Module, nn.Module]:
        """Perform crossover between two parent models."""
        if np.random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        if method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        elif method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif method == 'arithmetic':
            return self._arithmetic_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(self, parent1: nn.Module, parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Uniform crossover - each parameter randomly selected from parents."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                child1.named_parameters(),
                child2.named_parameters()
            ):
                # Create random mask
                mask = torch.rand_like(param1) > 0.5

                # Swap parameters based on mask
                temp = param1.data.clone()
                param1.data[mask] = param2.data[mask]
                param2.data[mask] = temp[mask]

        return child1, child2

    def _single_point_crossover(self, parent1: nn.Module, parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Single-point crossover - swap parameters after a random point."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Get all parameters
        params1 = list(child1.named_parameters())
        params2 = list(child2.named_parameters())

        if len(params1) < 2:
            return child1, child2

        # Choose crossover point
        crossover_point = np.random.randint(1, len(params1))

        with torch.no_grad():
            # Swap parameters after crossover point
            for i in range(crossover_point, len(params1)):
                name1, param1 = params1[i]
                name2, param2 = params2[i]

                # Swap
                temp = param1.data.clone()
                param1.data = param2.data.clone()
                param2.data = temp

        return child1, child2

    def _two_point_crossover(self, parent1: nn.Module, parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Two-point crossover - swap parameters between two random points."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Get all parameters
        params1 = list(child1.named_parameters())
        params2 = list(child2.named_parameters())

        if len(params1) < 3:
            return self._single_point_crossover(parent1, parent2)

        # Choose two crossover points
        points = sorted(np.random.choice(range(1, len(params1)), 2, replace=False))
        point1, point2 = points

        with torch.no_grad():
            # Swap parameters between crossover points
            for i in range(point1, point2):
                name1, param1 = params1[i]
                name2, param2 = params2[i]

                # Swap
                temp = param1.data.clone()
                param1.data = param2.data.clone()
                param2.data = temp

        return child1, child2

    def _arithmetic_crossover(self, parent1: nn.Module, parent2: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Arithmetic crossover - weighted average of parameters."""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        # Random weight
        alpha = np.random.uniform(0.3, 0.7)

        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                child1.named_parameters(),
                child2.named_parameters()
            ):
                # Weighted average
                new_param1 = alpha * param1.data + (1 - alpha) * param2.data
                new_param2 = (1 - alpha) * param1.data + alpha * param2.data

                param1.data = new_param1
                param2.data = new_param2

        return child1, child2

    def mutate(self, model: nn.Module, method: str = 'gaussian') -> nn.Module:
        """Apply mutation to a model."""
        if np.random.random() > self.mutation_rate:
            # No mutation
            return model

        mutated = copy.deepcopy(model)

        if method == 'gaussian':
            return self._gaussian_mutation(mutated)
        elif method == 'uniform':
            return self._uniform_mutation(mutated)
        elif method == 'adaptive':
            return self._adaptive_mutation(mutated)
        elif method == 'creep':
            return self._creep_mutation(mutated)
        else:
            return self._gaussian_mutation(mutated)

    def _gaussian_mutation(self, model: nn.Module) -> nn.Module:
        """Gaussian mutation - add Gaussian noise to parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if np.random.random() < self.mutation_rate:
                    # Add Gaussian noise
                    noise = torch.randn_like(param) * self.mutation_strength
                    param.add_(noise)

        return model

    def _uniform_mutation(self, model: nn.Module) -> nn.Module:
        """Uniform mutation - randomly reset some parameters."""
        with torch.no_grad():
            for param in model.parameters():
                if np.random.random() < self.mutation_rate:
                    # Randomly reset some values
                    mask = torch.rand_like(param) < (self.mutation_rate / 10)
                    if mask.any():
                        # Reset to random values in same range
                        std = param.std().item()
                        mean = param.mean().item()
                        random_values = torch.randn_like(param) * std + mean
                        param[mask] = random_values[mask]

        return model

    def _adaptive_mutation(self, model: nn.Module, generation: int = 0) -> nn.Module:
        """Adaptive mutation - mutation strength decreases over time."""
        # Adapt mutation strength based on generation
        adaptive_strength = self.mutation_strength * np.exp(-generation / 100)

        with torch.no_grad():
            for param in model.parameters():
                if np.random.random() < self.mutation_rate:
                    # Add adaptive noise
                    noise = torch.randn_like(param) * adaptive_strength
                    param.add_(noise)

        return model

    def _creep_mutation(self, model: nn.Module) -> nn.Module:
        """Creep mutation - small mutations to few parameters."""
        with torch.no_grad():
            params = list(model.parameters())

            # Select few parameters to mutate
            num_to_mutate = max(1, int(len(params) * 0.1))
            indices = np.random.choice(len(params), num_to_mutate, replace=False)

            for idx in indices:
                param = params[idx]
                # Small mutation
                noise = torch.randn_like(param) * (self.mutation_strength * 0.1)
                param.add_(noise)

        return model

    def create_offspring(self, parents: List[nn.Module], merge_techniques) -> List[nn.Module]:
        """Create offspring from parents using crossover and mutation."""
        offspring = []

        # Pair up parents for crossover
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

            # Crossover
            child1, child2 = self.crossover(parent1, parent2, method='uniform')

            # Mutation
            child1 = self.mutate(child1, method='gaussian')
            child2 = self.mutate(child2, method='gaussian')

            offspring.extend([child1, child2])

        # Additional offspring through merging
        if len(offspring) < self.config.get('population_size', 8):
            # Create additional offspring through different merge techniques
            techniques = ['linear', 'slerp', 'ties', 'dare']

            while len(offspring) < self.config.get('population_size', 8):
                # Random parents
                parent_indices = np.random.choice(len(parents), 2, replace=False)
                selected_parents = [parents[i] for i in parent_indices]

                # Random technique
                technique = np.random.choice(techniques)

                # Merge
                if technique == 'linear':
                    weights = np.random.dirichlet(np.ones(len(selected_parents)))
                    child = merge_techniques.linear_merge(selected_parents, weights.tolist())
                elif technique == 'slerp' and len(selected_parents) == 2:
                    t = np.random.uniform(0.3, 0.7)
                    child = merge_techniques.slerp_merge(selected_parents, t)
                elif technique == 'ties':
                    threshold = np.random.uniform(0.5, 0.9)
                    child = merge_techniques.ties_merge(selected_parents, threshold)
                elif technique == 'dare':
                    drop_rate = np.random.uniform(0.3, 0.7)
                    child = merge_techniques.dare_merge(selected_parents, drop_rate)
                else:
                    child = merge_techniques.linear_merge(selected_parents)

                # Mutate
                child = self.mutate(child)
                offspring.append(child)

        return offspring[:self.config.get('population_size', 8)]

    def repair_model(self, model: nn.Module) -> nn.Module:
        """Repair a model after genetic operations."""
        with torch.no_grad():
            for param in model.parameters():
                # Fix NaN values
                if torch.isnan(param).any():
                    param[torch.isnan(param)] = 0.0

                # Fix Inf values
                if torch.isinf(param).any():
                    param[torch.isinf(param)] = torch.sign(param[torch.isinf(param)]) * 1e6

                # Clip extreme values
                param.clamp_(-1e6, 1e6)

        return model