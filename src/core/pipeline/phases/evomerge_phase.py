"""
Refactored EvoMerge Phase

A cleaner, more maintainable implementation of the evolutionary model merging
phase with proper separation of concerns and error handling.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import random
from pathlib import Path
from dataclasses import dataclass
import asyncio

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_phase import ModelTrainingPhase
from ...exceptions import PhaseExecutionError, ModelLoadError
from ...interfaces.base_interfaces import BaseModel, PhaseResult
from ...config.agent_forge_config import EvoMergeConfig

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """Represents an individual in the evolutionary population."""
    
    model: nn.Module
    fitness: float = 0.0
    genes: Dict[str, float] = None
    generation: int = 0
    parent_ids: List[str] = None
    
    def __post_init__(self):
        if self.genes is None:
            self.genes = {}
        if self.parent_ids is None:
            self.parent_ids = []


class ModelMerger:
    """Handles different model merging strategies."""
    
    @staticmethod
    def linear_merge(models: List[nn.Module], weights: List[float]) -> nn.Module:
        """Linear interpolation between models."""
        if len(models) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        if not models:
            raise ValueError("At least one model is required")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Create merged model
        merged_model = models[0].__class__(models[0].config)
        merged_state = {}
        
        # Merge state dictionaries
        for key in models[0].state_dict().keys():
            merged_tensor = torch.zeros_like(models[0].state_dict()[key])
            
            for model, weight in zip(models, weights):
                merged_tensor += weight * model.state_dict()[key]
            
            merged_state[key] = merged_tensor
        
        merged_model.load_state_dict(merged_state)
        return merged_model
    
    @staticmethod
    def slerp_merge(model1: nn.Module, model2: nn.Module, t: float) -> nn.Module:
        """Spherical linear interpolation between two models."""
        if not 0 <= t <= 1:
            raise ValueError("Interpolation parameter t must be between 0 and 1")
        
        merged_model = model1.__class__(model1.config)
        merged_state = {}
        
        for key in model1.state_dict().keys():
            tensor1 = model1.state_dict()[key]
            tensor2 = model2.state_dict()[key]
            
            # Flatten tensors for slerp calculation
            flat1 = tensor1.flatten()
            flat2 = tensor2.flatten()
            
            # Calculate angle between vectors
            dot_product = torch.dot(flat1, flat2)
            norm1 = torch.norm(flat1)
            norm2 = torch.norm(flat2)
            
            # Avoid division by zero
            if norm1 == 0 or norm2 == 0:
                merged_tensor = (1 - t) * tensor1 + t * tensor2
            else:
                cos_angle = dot_product / (norm1 * norm2)
                cos_angle = torch.clamp(cos_angle, -1, 1)
                angle = torch.acos(cos_angle)
                
                if angle < 1e-6:  # Vectors are nearly parallel
                    merged_tensor = (1 - t) * tensor1 + t * tensor2
                else:
                    sin_angle = torch.sin(angle)
                    w1 = torch.sin((1 - t) * angle) / sin_angle
                    w2 = torch.sin(t * angle) / sin_angle
                    merged_flat = w1 * flat1 + w2 * flat2
                    merged_tensor = merged_flat.reshape(tensor1.shape)
            
            merged_state[key] = merged_tensor
        
        merged_model.load_state_dict(merged_state)
        return merged_model


class EvolutionEngine:
    """Manages the evolutionary algorithm logic."""
    
    def __init__(self, config: EvoMergeConfig):
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        
    def initialize_population(self, base_models: List[nn.Module]) -> None:
        """Initialize population with base models and random variations."""
        self.population = []
        
        # Add base models as initial population
        for i, model in enumerate(base_models):
            individual = Individual(
                model=model,
                genes={'base_model_id': i},
                generation=0
            )
            self.population.append(individual)
        
        # Add random combinations to reach population size
        while len(self.population) < self.config.population_size:
            # Select random base models for merging
            selected_models = random.sample(base_models, 2)
            weights = [random.random(), random.random()]
            
            merged_model = ModelMerger.linear_merge(selected_models, weights)
            
            individual = Individual(
                model=merged_model,
                genes={'weights': weights, 'merge_type': 'linear'},
                generation=0
            )
            self.population.append(individual)
    
    def evaluate_population(self, evaluation_func: callable) -> None:
        """Evaluate fitness of all individuals in population."""
        for individual in self.population:
            try:
                individual.fitness = evaluation_func(individual.model)
            except Exception as e:
                logger.warning(f"Failed to evaluate individual: {e}")
                individual.fitness = float('-inf')  # Assign worst fitness
        
        # Update best individual
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
    
    def selection(self) -> List[Individual]:
        """Select individuals for reproduction."""
        if self.config.selection_method == "tournament":
            return self._tournament_selection()
        elif self.config.selection_method == "roulette":
            return self._roulette_selection()
        elif self.config.selection_method == "rank":
            return self._rank_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.config.selection_method}")
    
    def _tournament_selection(self, tournament_size: int = 3) -> List[Individual]:
        """Tournament selection."""
        selected = []
        
        for _ in range(len(self.population) // 2):  # Select parents for crossover
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self) -> List[Individual]:
        """Roulette wheel selection."""
        # Ensure all fitness values are positive
        min_fitness = min(ind.fitness for ind in self.population)
        adjusted_fitness = [ind.fitness - min_fitness + 1 for ind in self.population]
        
        total_fitness = sum(adjusted_fitness)
        selection_probs = [f / total_fitness for f in adjusted_fitness]
        
        selected = []
        for _ in range(len(self.population) // 2):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(selection_probs):
                cumulative += prob
                if r <= cumulative:
                    selected.append(self.population[i])
                    break
        
        return selected
    
    def _rank_selection(self) -> List[Individual]:
        """Rank-based selection."""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        ranks = list(range(len(sorted_population), 0, -1))
        total_rank = sum(ranks)
        
        selection_probs = [r / total_rank for r in ranks]
        
        selected = []
        for _ in range(len(self.population) // 2):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(selection_probs):
                cumulative += prob
                if r <= cumulative:
                    selected.append(sorted_population[i])
                    break
        
        return selected
    
    def crossover_and_mutation(self, parents: List[Individual], base_models: List[nn.Module]) -> List[Individual]:
        """Create new generation through crossover and mutation."""
        new_population = []
        
        # Keep best individuals (elitism)
        elite_count = max(1, len(self.population) // 10)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_count]
        new_population.extend(elite)
        
        # Create offspring
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                offspring = self._crossover(parent1, parent2)
            else:
                # Clone a parent
                parent = random.choice(parents if parents else self.population)
                offspring = self._clone_individual(parent)
            
            # Apply mutation
            if random.random() < self.config.mutation_rate:
                offspring = self._mutate(offspring, base_models)
            
            offspring.generation = self.generation + 1
            new_population.append(offspring)
        
        return new_population[:self.config.population_size]
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring from two parents."""
        # Simple crossover: blend the models
        weight = random.random()
        merged_model = ModelMerger.linear_merge([parent1.model, parent2.model], [weight, 1 - weight])
        
        offspring = Individual(
            model=merged_model,
            genes={'crossover_weight': weight},
            parent_ids=[id(parent1), id(parent2)]
        )
        
        return offspring
    
    def _clone_individual(self, parent: Individual) -> Individual:
        """Create a clone of an individual."""
        # Deep copy the model
        cloned_model = parent.model.__class__(parent.model.config)
        cloned_model.load_state_dict(parent.model.state_dict())
        
        return Individual(
            model=cloned_model,
            genes=parent.genes.copy(),
            parent_ids=[id(parent)]
        )
    
    def _mutate(self, individual: Individual, base_models: List[nn.Module]) -> Individual:
        """Apply mutation to an individual."""
        mutation_strength = 0.1
        
        # Mutate model weights
        with torch.no_grad():
            for param in individual.model.parameters():
                if random.random() < 0.1:  # Only mutate some parameters
                    noise = torch.randn_like(param) * mutation_strength
                    param.add_(noise)
        
        return individual
    
    def evolve_generation(self, evaluation_func: callable, base_models: List[nn.Module]) -> None:
        """Evolve to the next generation."""
        # Evaluate current population
        self.evaluate_population(evaluation_func)
        
        # Select parents
        parents = self.selection()
        
        # Create new generation
        self.population = self.crossover_and_mutation(parents, base_models)
        self.generation += 1
        
        logger.info(f"Generation {self.generation}: Best fitness = {self.best_individual.fitness:.4f}")


class EvoMergePhase(ModelTrainingPhase):
    """Refactored EvoMerge phase with clean separation of concerns."""
    
    def __init__(self, phase_name: str, config: Dict[str, Any]):
        super().__init__(phase_name, config)
        self.evomerge_config = EvoMergeConfig.from_dict(config)
        self.evolution_engine: Optional[EvolutionEngine] = None
        self.base_models: List[nn.Module] = []
        self.tokenizer = None
    
    def _validate_phase_specific_config(self) -> None:
        """Validate EvoMerge specific configuration."""
        try:
            self.evomerge_config.validate()
        except Exception as e:
            raise PhaseExecutionError(
                f"Invalid EvoMerge configuration: {e}",
                phase_name=self.phase_name
            )
    
    async def _pre_phase_setup(self, model: BaseModel) -> None:
        """Setup base models and evolution engine."""
        await super()._pre_phase_setup(model)
        
        try:
            # Load base models
            await self._load_base_models()
            
            # Initialize evolution engine
            self.evolution_engine = EvolutionEngine(self.evomerge_config)
            self.evolution_engine.initialize_population(self.base_models)
            
            self.logger.info(f"Initialized evolution with {len(self.base_models)} base models")
            
        except Exception as e:
            raise PhaseExecutionError(
                f"Failed to setup EvoMerge phase: {e}",
                phase_name=self.phase_name,
                cause=e
            )
    
    async def _load_base_models(self) -> None:
        """Load all base models for evolutionary merging."""
        self.base_models = []
        
        for model_path in self.evomerge_config.base_models:
            try:
                self.logger.info(f"Loading base model: {model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.base_models.append(model)
                
                # Load tokenizer from first model
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                
            except Exception as e:
                raise ModelLoadError(
                    f"Failed to load base model {model_path}: {e}",
                    details={"model_path": model_path}
                )
    
    async def _execute_phase(self, model: BaseModel) -> PhaseResult:
        """Execute the evolutionary merging process."""
        self.logger.info("Starting evolutionary model merging")
        
        try:
            # Run evolution for specified generations
            for generation in range(self.evomerge_config.generations):
                self.logger.info(f"Evolution generation {generation + 1}/{self.evomerge_config.generations}")
                
                # Evolve population
                self.evolution_engine.evolve_generation(
                    self._evaluate_model_fitness,
                    self.base_models
                )
                
                # Update metrics
                self.update_metrics(f'generation_{generation}_best_fitness', 
                                   self.evolution_engine.best_individual.fitness)
                
                # Early stopping if fitness doesn't improve
                if self._should_stop_evolution():
                    self.logger.info(f"Early stopping at generation {generation}")
                    break
            
            # Get best model
            best_model = self.evolution_engine.best_individual.model
            
            # Update the input model with the best evolved model
            model._model = best_model
            
            final_metrics = {
                'best_fitness': self.evolution_engine.best_individual.fitness,
                'total_generations': self.evolution_engine.generation,
                'population_size': self.evomerge_config.population_size,
                'base_models_count': len(self.base_models)
            }
            
            self.metrics.update(final_metrics)
            
            return PhaseResult(
                success=True,
                model=model,
                phase_name=self.phase_name,
                metrics=self.metrics,
                artifacts={
                    'best_individual': self.evolution_engine.best_individual,
                    'evolution_history': final_metrics
                }
            )
            
        except Exception as e:
            raise PhaseExecutionError(
                f"EvoMerge execution failed: {e}",
                phase_name=self.phase_name,
                cause=e
            )
    
    def _evaluate_model_fitness(self, model: nn.Module) -> float:
        """Evaluate fitness of a model (simplified version)."""
        try:
            # Simple fitness based on parameter diversity and magnitude
            total_params = 0
            param_variance = 0
            
            with torch.no_grad():
                for param in model.parameters():
                    total_params += param.numel()
                    param_variance += torch.var(param).item()
            
            # Fitness is based on parameter diversity (simple heuristic)
            fitness = param_variance / total_params if total_params > 0 else 0
            return fitness
            
        except Exception as e:
            self.logger.warning(f"Failed to evaluate model fitness: {e}")
            return float('-inf')
    
    def _should_stop_evolution(self, patience: int = 10) -> bool:
        """Check if evolution should stop early."""
        if self.evolution_engine.generation < patience:
            return False
        
        # Check if fitness has improved in last `patience` generations
        recent_fitness = []
        for i in range(patience):
            gen_key = f'generation_{self.evolution_engine.generation - i - 1}_best_fitness'
            if gen_key in self.metrics:
                recent_fitness.append(self.metrics[gen_key])
        
        if len(recent_fitness) < patience:
            return False
        
        # Check if there's been improvement
        improvement = max(recent_fitness) - min(recent_fitness)
        return improvement < 1e-6  # Threshold for improvement
    
    async def _post_phase_cleanup(self, model: BaseModel, result: PhaseResult) -> None:
        """Cleanup resources after phase execution."""
        await super()._post_phase_cleanup(model, result)
        
        # Clean up base models to free memory
        for base_model in self.base_models:
            if hasattr(base_model, 'cpu'):
                base_model.cpu()
        
        self.base_models.clear()
        self.logger.info("Cleaned up base models")