"""
Population management for evolutionary optimization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import copy
import random

class PopulationManager:
    """Manages population of models during evolution."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.population_size = config.get('population_size', 8)
        self.elite_size = config.get('elite_size', 2)
        self.tournament_size = config.get('tournament_size', 3)
        self.diversity_weight = config.get('diversity_weight', 0.3)
        self.min_diversity = config.get('min_diversity', 0.2)

        # Population storage
        self.population = []
        self.fitness_scores = []
        self.generation = 0

        # Diversity tracking
        self.diversity_history = []

    def initialize_population(self, base_models: List[nn.Module], merge_techniques) -> List[nn.Module]:
        """Initialize diverse population from base models."""
        population = []
        techniques = ['linear', 'slerp', 'ties', 'dare', 'frankenmerge', 'dfs']

        for i in range(self.population_size):
            # Rotate through techniques for diversity
            technique = techniques[i % len(techniques)]

            # Random weights for linear merge
            if technique == 'linear':
                weights = np.random.dirichlet(np.ones(len(base_models)))
                merged = merge_techniques.linear_merge(base_models, weights.tolist())

            # Random interpolation factor for SLERP
            elif technique == 'slerp' and len(base_models) >= 2:
                t = np.random.uniform(0.3, 0.7)
                merged = merge_techniques.slerp_merge(base_models[:2], t)

            # Random threshold for TIES
            elif technique == 'ties':
                threshold = np.random.uniform(0.5, 0.9)
                merged = merge_techniques.ties_merge(base_models, threshold)

            # Random drop rate for DARE
            elif technique == 'dare':
                drop_rate = np.random.uniform(0.3, 0.7)
                merged = merge_techniques.dare_merge(base_models, drop_rate)

            # Random layer assignment for FrankenMerge
            elif technique == 'frankenmerge':
                merged = merge_techniques.frankenmerge(base_models)

            # DFS merge
            elif technique == 'dfs':
                merged = merge_techniques.dfs_merge(base_models)

            else:
                # Fallback to linear merge
                merged = merge_techniques.linear_merge(base_models)

            population.append(merged)

        self.population = population
        return population

    def update_population(self, new_population: List[nn.Module], fitness_scores: List[float]):
        """Update population with new generation."""
        self.population = new_population
        self.fitness_scores = fitness_scores
        self.generation += 1

        # Track diversity
        diversity = self.calculate_diversity()
        self.diversity_history.append(diversity)

    def select_parents(self, num_parents: int) -> List[Tuple[nn.Module, float]]:
        """Select parents for next generation."""
        if not self.fitness_scores:
            # Random selection if no fitness scores
            indices = np.random.choice(len(self.population), num_parents, replace=False)
            return [(self.population[i], 0.5) for i in indices]

        # Tournament selection with diversity consideration
        parents = []

        for _ in range(num_parents):
            # Run tournament
            tournament_indices = np.random.choice(
                len(self.population),
                min(self.tournament_size, len(self.population)),
                replace=False
            )

            # Calculate selection scores (fitness + diversity bonus)
            selection_scores = []
            for idx in tournament_indices:
                fitness = self.fitness_scores[idx]

                # Add diversity bonus
                diversity_bonus = self._calculate_diversity_bonus(idx)
                score = fitness + self.diversity_weight * diversity_bonus

                selection_scores.append((idx, score))

            # Select winner
            winner_idx = max(selection_scores, key=lambda x: x[1])[0]
            parents.append((self.population[winner_idx], self.fitness_scores[winner_idx]))

        return parents

    def select_elites(self) -> List[Tuple[nn.Module, float]]:
        """Select elite individuals to preserve."""
        if not self.fitness_scores:
            return []

        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]

        # Select top individuals
        elites = []
        for i in range(min(self.elite_size, len(self.population))):
            idx = sorted_indices[i]
            elites.append((self.population[idx], self.fitness_scores[idx]))

        return elites

    def calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 1.0

        # Calculate pairwise distances
        distances = []

        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                dist = self._model_distance(self.population[i], self.population[j])
                distances.append(dist)

        # Return average distance as diversity measure
        return np.mean(distances) if distances else 0.0

    def _model_distance(self, model1: nn.Module, model2: nn.Module) -> float:
        """Calculate distance between two models."""
        distance = 0.0
        param_count = 0

        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                model1.named_parameters(),
                model2.named_parameters()
            ):
                # L2 distance between parameters
                diff = (param1 - param2).flatten()
                distance += torch.norm(diff).item()
                param_count += 1

        return distance / max(param_count, 1)

    def _calculate_diversity_bonus(self, individual_idx: int) -> float:
        """Calculate diversity bonus for an individual."""
        if len(self.population) < 2:
            return 0.0

        # Calculate average distance to other individuals
        distances = []

        for i, other_model in enumerate(self.population):
            if i != individual_idx:
                dist = self._model_distance(
                    self.population[individual_idx],
                    other_model
                )
                distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0

        # Normalize to [0, 1]
        max_distance = max(self.diversity_history) if self.diversity_history else 1.0
        normalized_distance = avg_distance / max(max_distance, 1e-6)

        return normalized_distance

    def enforce_diversity(self, population: List[nn.Module]) -> List[nn.Module]:
        """Enforce minimum diversity in population."""
        current_diversity = self.calculate_diversity()

        if current_diversity < self.min_diversity:
            # Add noise to some individuals to increase diversity
            num_to_mutate = len(population) // 4

            for i in range(num_to_mutate):
                idx = np.random.randint(0, len(population))
                population[idx] = self._add_noise_to_model(population[idx])

        return population

    def _add_noise_to_model(self, model: nn.Module, noise_scale: float = 0.01) -> nn.Module:
        """Add small noise to model parameters."""
        noisy_model = copy.deepcopy(model)

        with torch.no_grad():
            for param in noisy_model.parameters():
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)

        return noisy_model

    def get_best_individual(self) -> Tuple[nn.Module, float]:
        """Get the best individual from population."""
        if not self.fitness_scores:
            return self.population[0] if self.population else None, 0.0

        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]

    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about current population."""
        if not self.fitness_scores:
            return {
                'generation': self.generation,
                'population_size': len(self.population),
                'diversity': self.calculate_diversity()
            }

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(self.fitness_scores),
            'average_fitness': np.mean(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores),
            'fitness_std': np.std(self.fitness_scores),
            'diversity': self.calculate_diversity(),
            'diversity_history': self.diversity_history
        }

    def save_checkpoint(self, path: str):
        """Save population checkpoint."""
        checkpoint = {
            'generation': self.generation,
            'population': [model.state_dict() for model in self.population],
            'fitness_scores': self.fitness_scores,
            'diversity_history': self.diversity_history,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, model_class):
        """Load population from checkpoint."""
        checkpoint = torch.load(path)

        self.generation = checkpoint['generation']
        self.fitness_scores = checkpoint['fitness_scores']
        self.diversity_history = checkpoint['diversity_history']
        self.config = checkpoint['config']

        # Recreate models
        self.population = []
        for state_dict in checkpoint['population']:
            model = model_class()
            model.load_state_dict(state_dict)
            self.population.append(model)