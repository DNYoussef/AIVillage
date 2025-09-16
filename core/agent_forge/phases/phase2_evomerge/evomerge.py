"""
Main EvoMerge implementation - Phase 2 of Agent Forge pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import time
from pathlib import Path
import json

from .config import EvoMergeConfig, MergeResult, EvolutionState
from .merge_techniques import MergeTechniques
from .fitness_evaluator import FitnessEvaluator
from .population_manager import PopulationManager
from .genetic_operations import GeneticOperations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvoMerge:
    """
    Evolutionary Model Merging - Phase 2 of Agent Forge.

    Takes 3 Cognate models (25M parameters each) and evolves them
    through 50 generations to create an optimized merged model.
    """

    def __init__(self, config: Optional[EvoMergeConfig] = None):
        """Initialize EvoMerge with configuration."""
        self.config = config or EvoMergeConfig()

        # Initialize components
        self.merge_techniques = MergeTechniques(device=self.config.device)
        self.fitness_evaluator = FitnessEvaluator(vars(self.config))
        self.population_manager = PopulationManager(vars(self.config))
        self.genetic_operations = GeneticOperations(vars(self.config))

        # Evolution state
        self.state = EvolutionState(
            generation=0,
            best_fitness=0.0,
            average_fitness=0.0,
            diversity=1.0,
            convergence_counter=0,
            population=[],
            fitness_history=[],
            diversity_history=[]
        )

        # Checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WebSocket for progress updates
        self.websocket = None

    async def evolve(self, cognate_models: List[nn.Module]) -> MergeResult:
        """
        Main evolution loop - evolve Cognate models to optimal merge.

        Args:
            cognate_models: List of 3 Cognate models (25M parameters each)

        Returns:
            MergeResult with optimized model and metrics
        """
        logger.info(f"Starting EvoMerge with {len(cognate_models)} Cognate models")

        # Validate input
        self._validate_input_models(cognate_models)

        # Initialize population
        logger.info("Initializing population...")
        population = self.population_manager.initialize_population(
            cognate_models,
            self.merge_techniques
        )

        # Evolution loop
        for generation in range(self.config.generations):
            self.state.generation = generation
            logger.info(f"Generation {generation + 1}/{self.config.generations}")

            # Evaluate fitness
            logger.info("Evaluating population fitness...")
            fitness_scores = await self._evaluate_population(population)

            # Update state
            self.state.best_fitness = max(fitness_scores)
            self.state.average_fitness = np.mean(fitness_scores)
            self.state.fitness_history.append(self.state.best_fitness)

            # Calculate diversity
            diversity = self.population_manager.calculate_diversity()
            self.state.diversity = diversity
            self.state.diversity_history.append(diversity)

            # Check convergence
            if self._check_convergence():
                logger.info(f"Converged at generation {generation}")
                break

            # Selection
            logger.info("Selecting parents...")
            self.population_manager.update_population(population, fitness_scores)
            parents = self.population_manager.select_parents(
                self.config.population_size - self.config.elite_size
            )

            # Get elites
            elites = self.population_manager.select_elites()

            # Create offspring
            logger.info("Creating offspring...")
            parent_models = [p[0] for p in parents]
            offspring = self.genetic_operations.create_offspring(
                parent_models,
                self.merge_techniques
            )

            # Combine elites and offspring
            population = [e[0] for e in elites] + offspring

            # Enforce diversity if needed
            if diversity < self.config.min_diversity:
                logger.info(f"Low diversity ({diversity:.3f}), enforcing diversity...")
                population = self.population_manager.enforce_diversity(population)

            # Checkpoint if needed
            if generation % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(generation)

            # Send progress update
            await self._send_progress_update(generation)

        # Get best model
        best_model, best_fitness = self.population_manager.get_best_individual()

        # Final evaluation
        logger.info("Performing final evaluation...")
        final_metrics = self.fitness_evaluator.evaluate(best_model)

        # Create result
        result = MergeResult(
            model=best_model,
            technique="evolutionary",
            fitness=best_fitness,
            metrics={
                'perplexity': final_metrics.perplexity,
                'accuracy': final_metrics.accuracy,
                'inference_speed': final_metrics.inference_speed,
                'memory_usage': final_metrics.memory_usage,
                'generations': self.state.generation + 1,
                'final_diversity': self.state.diversity
            },
            generation=self.state.generation,
            parent_ids=list(range(len(cognate_models)))
        )

        logger.info(f"Evolution complete! Best fitness: {best_fitness:.4f}")
        logger.info(f"Final metrics: {result.metrics}")

        return result

    async def _evaluate_population(self, population: List[nn.Module]) -> List[float]:
        """Evaluate fitness of entire population."""
        fitness_scores = []

        # Parallel evaluation if enabled
        if self.config.enable_parallel:
            # Create tasks for parallel evaluation
            tasks = []
            for model in population:
                task = asyncio.create_task(self._evaluate_model_async(model))
                tasks.append(task)

            # Wait for all evaluations
            results = await asyncio.gather(*tasks)

            # Extract fitness scores
            for metrics in results:
                fitness_scores.append(metrics.composite_fitness)
        else:
            # Sequential evaluation
            for i, model in enumerate(population):
                logger.debug(f"Evaluating model {i+1}/{len(population)}")
                metrics = self.fitness_evaluator.evaluate(model)
                fitness_scores.append(metrics.composite_fitness)

        return fitness_scores

    async def _evaluate_model_async(self, model: nn.Module):
        """Asynchronous model evaluation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.fitness_evaluator.evaluate,
            model
        )

    def _check_convergence(self) -> bool:
        """Check if evolution has converged."""
        if not self.config.early_stopping:
            return False

        if len(self.state.fitness_history) < self.config.convergence_patience:
            return False

        # Check if improvement is below threshold
        recent_history = self.state.fitness_history[-self.config.convergence_patience:]
        improvement = max(recent_history) - min(recent_history)

        if improvement < self.config.convergence_threshold:
            self.state.convergence_counter += 1
        else:
            self.state.convergence_counter = 0

        return self.state.convergence_counter >= self.config.convergence_patience

    def _validate_input_models(self, models: List[nn.Module]):
        """Validate input Cognate models."""
        if len(models) != 3:
            raise ValueError(f"Expected 3 Cognate models, got {len(models)}")

        for i, model in enumerate(models):
            # Count parameters
            param_count = sum(p.numel() for p in model.parameters())

            # Check if approximately 25M parameters
            expected = 25_000_000
            tolerance = 0.01  # 1% tolerance

            if abs(param_count - expected) / expected > tolerance:
                logger.warning(
                    f"Model {i} has {param_count:,} parameters, "
                    f"expected ~{expected:,}"
                )

    async def _save_checkpoint(self, generation: int):
        """Save checkpoint of current state."""
        checkpoint_path = self.checkpoint_dir / f"generation_{generation}.pt"

        checkpoint = {
            'generation': generation,
            'state': self.state,
            'config': self.config,
            'population': self.population_manager.population,
            'fitness_scores': self.population_manager.fitness_scores
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only recent ones."""
        checkpoints = sorted(self.checkpoint_dir.glob("generation_*.pt"))

        if len(checkpoints) > self.config.keep_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_checkpoints]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint}")

    async def _send_progress_update(self, generation: int):
        """Send progress update via WebSocket."""
        if self.websocket is None:
            return

        progress = {
            'phase': 'evomerge',
            'generation': generation,
            'total_generations': self.config.generations,
            'progress': (generation + 1) / self.config.generations,
            'best_fitness': self.state.best_fitness,
            'average_fitness': self.state.average_fitness,
            'diversity': self.state.diversity,
            'message': f"Generation {generation + 1}/{self.config.generations}"
        }

        try:
            await self.websocket.send_json(progress)
        except Exception as e:
            logger.warning(f"Failed to send progress update: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint."""
        checkpoint = torch.load(checkpoint_path)

        self.state = checkpoint['state']
        self.config = checkpoint['config']
        self.population_manager.population = checkpoint['population']
        self.population_manager.fitness_scores = checkpoint['fitness_scores']

        logger.info(f"Loaded checkpoint from generation {self.state.generation}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current evolution statistics."""
        stats = self.population_manager.get_population_statistics()
        stats.update({
            'best_fitness_history': self.state.fitness_history,
            'diversity_history': self.state.diversity_history,
            'convergence_counter': self.state.convergence_counter,
            'cache_stats': self.fitness_evaluator.get_cache_statistics()
        })
        return stats