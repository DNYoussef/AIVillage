#!/usr/bin/env python3
"""
Simple EvoMerge for Cognate Models
===================================

Direct evolutionary merging of Cognate models without HuggingFace conversion.
This script manually implements the core EvoMerge logic for our custom models.
"""

import os
import sys
import torch
import torch.nn as nn
import json
import logging
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our Cognate model
from scripts.run_cognate_titans_recursive_training import CognateWithTitansMemory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MergedModel:
    """Container for a merged model and its metadata"""
    model: nn.Module
    generation: int
    parents: List[str]
    merge_technique: str
    fitness: float = 0.0
    metrics: Dict[str, float] = None


class CognateEvoMerge:
    """Simple evolutionary merging for Cognate models"""

    def __init__(self, generations: int = 50, population_size: int = 8):
        self.generations = generations
        self.population_size = population_size
        self.elite_size = 2
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.evolution_history = []

    def load_base_models(self) -> List[MergedModel]:
        """Load the 3 trained Cognate models"""
        models = []

        for i in range(1, 4):
            model_path = project_root / f"models/cognate_real_data_model_{i}/model.pt"

            if model_path.exists():
                logger.info(f"Loading Model {i} from {model_path}")
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

                # Create model with config from checkpoint
                config = checkpoint.get('config')
                model = CognateWithTitansMemory(config)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Move to device
                model = model.to(self.device)
                model.eval()

                merged = MergedModel(
                    model=model,
                    generation=0,
                    parents=[],
                    merge_technique="original",
                    fitness=0.0
                )
                models.append(merged)
                logger.info(f"  Model {i} loaded successfully")
            else:
                raise FileNotFoundError(f"Model {i} not found at {model_path}")

        return models

    def create_initial_population(self, base_models: List[MergedModel]) -> List[MergedModel]:
        """Create 8 models from 3 base models using different merge techniques"""
        population = []

        # Keep the original 3 models
        population.extend(base_models)

        # Create 5 more models using different merge techniques
        merge_techniques = ['linear', 'slerp', 'average', 'weighted', 'interpolate']

        for i, technique in enumerate(merge_techniques):
            if len(population) >= self.population_size:
                break

            # Select two random base models
            model1, model2 = random.sample(base_models, 2)

            # Merge them
            merged_model = self.merge_models(model1.model, model2.model, technique)

            merged = MergedModel(
                model=merged_model,
                generation=0,
                parents=[f"base_{base_models.index(model1)}", f"base_{base_models.index(model2)}"],
                merge_technique=technique,
                fitness=0.0
            )
            population.append(merged)

        logger.info(f"Created initial population of {len(population)} models")
        return population

    def merge_models(self, model1: nn.Module, model2: nn.Module, technique: str) -> nn.Module:
        """Merge two models using specified technique"""

        # Create a new model with the same config
        config = model1.config if hasattr(model1, 'config') else model2.config
        merged = CognateWithTitansMemory(config).to(self.device)

        with torch.no_grad():
            if technique == 'linear':
                # Linear interpolation
                alpha = random.uniform(0.3, 0.7)
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    merged.state_dict()[name1].copy_(
                        alpha * param1.data + (1 - alpha) * param2.data
                    )

            elif technique == 'slerp':
                # Spherical linear interpolation
                alpha = random.uniform(0.3, 0.7)
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    # Flatten parameters
                    p1_flat = param1.data.flatten()
                    p2_flat = param2.data.flatten()

                    # Normalize
                    p1_norm = p1_flat / (torch.norm(p1_flat) + 1e-8)
                    p2_norm = p2_flat / (torch.norm(p2_flat) + 1e-8)

                    # Compute angle
                    dot = torch.clamp(torch.dot(p1_norm, p2_norm), -1, 1)
                    theta = torch.acos(dot)

                    # SLERP
                    if theta > 0.01:
                        sin_theta = torch.sin(theta)
                        w1 = torch.sin((1 - alpha) * theta) / sin_theta
                        w2 = torch.sin(alpha * theta) / sin_theta
                        merged_flat = w1 * p1_flat + w2 * p2_flat
                    else:
                        # Fallback to linear for small angles
                        merged_flat = alpha * p1_flat + (1 - alpha) * p2_flat

                    # Reshape and assign
                    merged.state_dict()[name1].copy_(merged_flat.reshape(param1.shape))

            elif technique == 'average':
                # Simple average
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    merged.state_dict()[name1].copy_(
                        0.5 * param1.data + 0.5 * param2.data
                    )

            elif technique == 'weighted':
                # Weighted average with random weights
                w1 = random.random()
                w2 = 1 - w1
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    merged.state_dict()[name1].copy_(
                        w1 * param1.data + w2 * param2.data
                    )

            elif technique == 'interpolate':
                # Layer-wise interpolation with different weights per layer
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    alpha = random.uniform(0.2, 0.8)
                    merged.state_dict()[name1].copy_(
                        alpha * param1.data + (1 - alpha) * param2.data
                    )

        return merged

    def evaluate_model(self, model: MergedModel) -> float:
        """Evaluate model fitness using simple metrics"""
        model.model.eval()
        total_loss = 0.0
        num_samples = 10  # Quick evaluation

        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random data
                x = torch.randint(0, 32000, (4, 128)).to(self.device)
                targets = torch.randint(0, 32000, (4, 128)).to(self.device)

                # Forward pass
                try:
                    outputs = model.model(x, targets)
                    loss = outputs['loss'].item()
                    total_loss += loss
                except Exception as e:
                    logger.warning(f"Error evaluating model: {e}")
                    return 0.0

        # Fitness is inverse of average loss
        avg_loss = total_loss / num_samples
        fitness = 1.0 / (1.0 + avg_loss)

        return fitness

    def tournament_selection(self, population: List[MergedModel], tournament_size: int = 3) -> MergedModel:
        """Select a parent using tournament selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return tournament[0]

    def crossover(self, parent1: MergedModel, parent2: MergedModel, generation: int) -> MergedModel:
        """Create offspring through crossover"""
        technique = random.choice(['linear', 'slerp', 'average', 'weighted', 'interpolate'])
        child_model = self.merge_models(parent1.model, parent2.model, technique)

        return MergedModel(
            model=child_model,
            generation=generation,
            parents=[str(parent1), str(parent2)],
            merge_technique=f"crossover_{technique}",
            fitness=0.0
        )

    def mutate(self, model: MergedModel) -> MergedModel:
        """Apply mutation to a model"""
        with torch.no_grad():
            for param in model.model.parameters():
                if random.random() < self.mutation_rate:
                    noise = torch.randn_like(param) * 0.01
                    param.data.add_(noise)
        return model

    def run_evolution(self):
        """Run the evolutionary merging process"""
        logger.info("=== Starting Cognate EvoMerge Evolution ===")

        # Load base models
        base_models = self.load_base_models()

        # Create initial population
        population = self.create_initial_population(base_models)

        # Evolution loop
        for gen in range(self.generations):
            start_time = time.time()
            logger.info(f"\n=== Generation {gen+1}/{self.generations} ===")

            # Evaluate fitness
            logger.info("Evaluating population fitness...")
            for i, individual in enumerate(population):
                individual.fitness = self.evaluate_model(individual)
                logger.debug(f"  Model {i}: fitness = {individual.fitness:.4f}")

            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            best_fitness = population[0].fitness
            avg_fitness = sum(ind.fitness for ind in population) / len(population)

            logger.info(f"Best fitness: {best_fitness:.4f}")
            logger.info(f"Average fitness: {avg_fitness:.4f}")
            logger.info(f"Best model technique: {population[0].merge_technique}")

            # Record history
            self.evolution_history.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_technique': population[0].merge_technique
            })

            # Check convergence
            if gen > 10:
                recent_history = [h['best_fitness'] for h in self.evolution_history[-5:]]
                if all(abs(f - best_fitness) < 0.0001 for f in recent_history):
                    logger.info("Convergence detected - stopping early")
                    break

            # Create next generation
            if gen < self.generations - 1:
                logger.info("Creating next generation...")

                # Elitism - keep best models
                next_generation = population[:self.elite_size]

                # Generate offspring
                while len(next_generation) < self.population_size:
                    # Selection
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)

                    # Crossover
                    if random.random() < self.crossover_rate:
                        offspring = self.crossover(parent1, parent2, gen + 1)
                    else:
                        # Clone parent
                        offspring = MergedModel(
                            model=parent1.model,
                            generation=gen + 1,
                            parents=[str(parent1)],
                            merge_technique=parent1.merge_technique,
                            fitness=0.0
                        )

                    # Mutation
                    if random.random() < self.mutation_rate:
                        offspring = self.mutate(offspring)

                    next_generation.append(offspring)

                population = next_generation

            elapsed = time.time() - start_time
            logger.info(f"Generation time: {elapsed:.1f}s")

            # Save checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                self.save_checkpoint(population[0], gen + 1)

        # Save final best model
        self.save_final_model(population[0])

        logger.info("\n=== Evolution Complete ===")
        logger.info(f"Final best fitness: {population[0].fitness:.4f}")
        logger.info(f"Final best technique: {population[0].merge_technique}")

        return population[0]

    def save_checkpoint(self, best_model: MergedModel, generation: int):
        """Save a checkpoint"""
        checkpoint_dir = project_root / "checkpoints" / "cognate_evomerge"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"gen_{generation}_best.pt"

        torch.save({
            'generation': generation,
            'model_state_dict': best_model.model.state_dict(),
            'config': best_model.model.config,
            'fitness': best_model.fitness,
            'merge_technique': best_model.merge_technique,
            'parents': best_model.parents
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self, best_model: MergedModel):
        """Save the final best model"""
        final_dir = project_root / "models" / "cognate_evomerge_final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        final_path = final_dir / "best_evolved_model.pt"
        torch.save({
            'model_state_dict': best_model.model.state_dict(),
            'config': best_model.model.config,
            'fitness': best_model.fitness,
            'merge_technique': best_model.merge_technique,
            'parents': best_model.parents,
            'generation': best_model.generation
        }, final_path)

        # Save evolution history
        history_path = final_dir / "evolution_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.evolution_history, f, indent=2)

        logger.info(f"Final model saved to: {final_path}")
        logger.info(f"Evolution history saved to: {history_path}")


def main():
    """Main function"""
    logger.info("=== SIMPLE EVOMERGE FOR COGNATE MODELS ===\n")

    # Create and run evolver
    evolver = CognateEvoMerge(generations=50, population_size=8)
    best_model = evolver.run_evolution()

    logger.info("\n=== SUCCESS ===")
    logger.info("EvoMerge completed successfully!")
    logger.info(f"Best model achieved fitness: {best_model.fitness:.4f}")
    logger.info(f"Used merge technique: {best_model.merge_technique}")

    # Print evolution summary
    if evolver.evolution_history:
        initial_fitness = evolver.evolution_history[0]['best_fitness']
        final_fitness = evolver.evolution_history[-1]['best_fitness']
        improvement = (final_fitness / initial_fitness - 1) * 100 if initial_fitness > 0 else 0
        logger.info(f"Improvement: {improvement:.1f}%")

    return best_model


if __name__ == "__main__":
    model = main()