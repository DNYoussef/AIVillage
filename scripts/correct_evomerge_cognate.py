#!/usr/bin/env python3
"""
Correct EvoMerge for Cognate Models
====================================

Implements the correct evolutionary strategy:
- Top 2 winners each spawn 3 mutated children (6 total)
- Bottom 6 losers grouped into 2 sets of 3, each merged into 1 child (2 total)
- Total: 8 models per generation
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
    model_id: str = ""


class CorrectCognateEvoMerge:
    """Correct evolutionary merging strategy for Cognate models"""

    def __init__(self, generations: int = 50):
        self.generations = generations
        self.population_size = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.evolution_history = []
        self.model_counter = 0

    def get_model_id(self, generation: int, description: str) -> str:
        """Generate a unique model ID"""
        self.model_counter += 1
        return f"gen{generation}_{description}_{self.model_counter}"

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
                    fitness=0.0,
                    model_id=f"base_model_{i}"
                )
                models.append(merged)
                logger.info(f"  Model {i} loaded successfully")
            else:
                raise FileNotFoundError(f"Model {i} not found at {model_path}")

        return models

    def create_initial_population(self, base_models: List[MergedModel]) -> List[MergedModel]:
        """
        Generation 1: Create 8 models from 3 base models using the 2^3 merge recipe combinations
        from the original EvoMerge implementation
        """
        population = []

        # Define the 8 merge recipe combinations (2^3 = 8)
        # Primary: linear or slerp (2 options)
        # Secondary: ties or dare (2 options)
        # Final: frankenmerge or dfs (2 options)
        merge_recipes = [
            {"primary": "linear", "secondary": "ties", "final": "frankenmerge"},
            {"primary": "linear", "secondary": "ties", "final": "dfs"},
            {"primary": "linear", "secondary": "dare", "final": "frankenmerge"},
            {"primary": "linear", "secondary": "dare", "final": "dfs"},
            {"primary": "slerp", "secondary": "ties", "final": "frankenmerge"},
            {"primary": "slerp", "secondary": "ties", "final": "dfs"},
            {"primary": "slerp", "secondary": "dare", "final": "frankenmerge"},
            {"primary": "slerp", "secondary": "dare", "final": "dfs"},
        ]

        logger.info(f"Generation 1: Creating 8 models using 2^3 merge recipe combinations")

        for i, recipe in enumerate(merge_recipes):
            logger.info(f"  Creating model {i+1}/8: {recipe['primary']} → {recipe['secondary']} → {recipe['final']}")

            # Apply 3-stage merging process
            merged_model = self.apply_merge_recipe(base_models, recipe)

            merged = MergedModel(
                model=merged_model,
                generation=1,
                parents=[m.model_id for m in base_models],
                merge_technique=f"{recipe['primary']}_{recipe['secondary']}_{recipe['final']}",
                fitness=0.0,
                model_id=self.get_model_id(1, f"recipe_{i+1}")
            )
            population.append(merged)

        logger.info(f"Generation 1: Created {len(population)} models using 2^3 merge recipes")
        return population

    def apply_merge_recipe(self, base_models: List[MergedModel], recipe: Dict[str, str]) -> nn.Module:
        """
        Apply a 3-stage merge recipe to create a new model
        """
        models = [m.model for m in base_models]

        # Stage 1: Primary merge (linear or slerp)
        if recipe["primary"] == "linear":
            # Linear merge all 3 models with equal weights
            stage1 = self.linear_merge_multiple(models, [1/3, 1/3, 1/3])
        else:  # slerp
            # SLERP merge first two models, then merge with third
            stage1 = self.slerp_merge_multiple(models)

        # Stage 2: Secondary merge (ties or dare)
        if recipe["secondary"] == "ties":
            stage2 = self.ties_merge(stage1, models)
        else:  # dare
            stage2 = self.dare_merge(stage1, models)

        # Stage 3: Final merge (frankenmerge or dfs)
        if recipe["final"] == "frankenmerge":
            final = self.frankenmerge(stage2, models)
        else:  # dfs
            final = self.dfs_merge(stage2, models)

        return final

    def linear_merge_multiple(self, models: List[nn.Module], weights: List[float]) -> nn.Module:
        """Linear interpolation merge of multiple models"""
        config = models[0].config if hasattr(models[0], 'config') else None
        merged = CognateWithTitansMemory(config).to(self.device)

        with torch.no_grad():
            # Initialize with zeros
            for param in merged.parameters():
                param.data.zero_()

            # Add weighted contributions from each model
            for model, weight in zip(models, weights):
                for (name_merged, param_merged), (name_orig, param_orig) in zip(
                    merged.named_parameters(), model.named_parameters()
                ):
                    param_merged.data.add_(param_orig.data * weight)

        return merged

    def slerp_merge_multiple(self, models: List[nn.Module]) -> nn.Module:
        """SLERP merge of multiple models"""
        # First SLERP merge models 0 and 1
        merged_01 = self.merge_models(models[0], models[1], 'slerp')
        # Then SLERP merge result with model 2
        if len(models) > 2:
            final = self.merge_models(merged_01, models[2], 'slerp')
        else:
            final = merged_01
        return final

    def ties_merge(self, current: nn.Module, base_models: List[nn.Module]) -> nn.Module:
        """TIES (Task Interference Elimination Strategy) merge"""
        config = current.config if hasattr(current, 'config') else None
        merged = CognateWithTitansMemory(config).to(self.device)

        with torch.no_grad():
            # TIES: Keep parameters that have consistent signs across models
            for (name, param_merged) in merged.named_parameters():
                # Get corresponding parameters from all models
                current_param = dict(current.named_parameters())[name]
                base_params = [dict(m.named_parameters())[name] for m in base_models]

                # Check sign agreement
                all_params = [current_param] + base_params
                signs = torch.stack([torch.sign(p.data) for p in all_params])
                sign_agreement = torch.abs(torch.mean(signs, dim=0))

                # Keep values where there's strong agreement (>0.5)
                mask = sign_agreement > 0.5

                # Average the agreeing parameters
                avg_param = torch.mean(torch.stack([p.data for p in all_params]), dim=0)
                param_merged.data = avg_param * mask + current_param.data * (~mask)

        return merged

    def dare_merge(self, current: nn.Module, base_models: List[nn.Module]) -> nn.Module:
        """DARE (Drop And REscale) merge"""
        config = current.config if hasattr(current, 'config') else None
        merged = CognateWithTitansMemory(config).to(self.device)

        with torch.no_grad():
            drop_rate = 0.3  # Drop 30% of parameters randomly

            for (name, param_merged) in merged.named_parameters():
                current_param = dict(current.named_parameters())[name]
                base_params = [dict(m.named_parameters())[name] for m in base_models]

                # Create random mask
                mask = torch.rand_like(current_param.data) > drop_rate

                # Rescale and merge
                scale = 1.0 / (1.0 - drop_rate)
                merged_base = torch.mean(torch.stack([p.data for p in base_params]), dim=0)

                param_merged.data = (current_param.data * mask * scale * 0.5 +
                                    merged_base * mask * scale * 0.5)

        return merged

    def frankenmerge(self, current: nn.Module, base_models: List[nn.Module]) -> nn.Module:
        """Frankenmerge: Mix layers from different models"""
        config = current.config if hasattr(current, 'config') else None
        merged = CognateWithTitansMemory(config).to(self.device)

        with torch.no_grad():
            layer_names = list(dict(current.named_parameters()).keys())
            num_layers = len(layer_names)

            # Randomly assign layers from different models
            for i, name in enumerate(layer_names):
                # Choose source model based on layer position
                if i < num_layers // 3:
                    source = base_models[0]
                elif i < 2 * num_layers // 3:
                    source = base_models[1] if len(base_models) > 1 else current
                else:
                    source = base_models[2] if len(base_models) > 2 else current

                source_param = dict(source.named_parameters())[name]
                merged.state_dict()[name].copy_(source_param.data)

        return merged

    def dfs_merge(self, current: nn.Module, base_models: List[nn.Module]) -> nn.Module:
        """DFS (Depth-First Search) merge: Hierarchical merging"""
        # First merge base models 0 and 1
        if len(base_models) >= 2:
            merge_01 = self.merge_models(base_models[0], base_models[1], 'average')
        else:
            merge_01 = base_models[0]

        # Then merge result with base model 2
        if len(base_models) >= 3:
            merge_012 = self.merge_models(merge_01, base_models[2], 'average')
        else:
            merge_012 = merge_01

        # Finally merge with current
        final = self.merge_models(current, merge_012, 'weighted')

        return final

    def mutate_winner(self, winner: MergedModel, generation: int, mutation_strength: float = 0.01) -> List[MergedModel]:
        """
        Mutate a winner model into 3 different children
        Each child gets different mutation patterns
        """
        children = []

        for i in range(3):
            # Create new model with same config
            config = winner.model.config if hasattr(winner.model, 'config') else None
            child = CognateWithTitansMemory(config).to(self.device)
            child.load_state_dict(winner.model.state_dict())

            # Apply different mutation strategies for each child
            with torch.no_grad():
                if i == 0:
                    # Child 1: Small uniform mutations
                    for param in child.parameters():
                        if param.requires_grad:
                            noise = torch.randn_like(param) * mutation_strength
                            param.data.add_(noise)

                elif i == 1:
                    # Child 2: Layer-specific mutations (stronger on later layers)
                    layer_idx = 0
                    for name, param in child.named_parameters():
                        if param.requires_grad:
                            # Increase mutation strength for later layers
                            layer_strength = mutation_strength * (1 + layer_idx * 0.1)
                            noise = torch.randn_like(param) * layer_strength
                            param.data.add_(noise)
                            layer_idx += 1

                else:  # i == 2
                    # Child 3: Sparse mutations (only mutate 30% of parameters)
                    for param in child.parameters():
                        if param.requires_grad:
                            mask = torch.rand_like(param) < 0.3
                            noise = torch.randn_like(param) * mutation_strength * 2
                            param.data.add_(noise * mask)

            child_model = MergedModel(
                model=child,
                generation=generation,
                parents=[winner.model_id],
                merge_technique=f"mutate_type_{i+1}",
                fitness=0.0,
                model_id=self.get_model_id(generation, f"winner_child_{i+1}")
            )
            children.append(child_model)

        return children

    def merge_loser_group(self, losers: List[MergedModel], generation: int) -> MergedModel:
        """
        Merge 3 loser models into 1 child
        Uses weighted average based on their fitness scores
        """
        # Create new model
        config = losers[0].model.config if hasattr(losers[0].model, 'config') else None
        merged = CognateWithTitansMemory(config).to(self.device)

        # Calculate weights based on fitness (better models get more weight even among losers)
        fitnesses = [m.fitness for m in losers]
        total_fitness = sum(fitnesses) if sum(fitnesses) > 0 else 3.0
        weights = [f/total_fitness if total_fitness > 0 else 1.0/3.0 for f in fitnesses]

        # Weighted merge of all 3 models
        with torch.no_grad():
            # Initialize with zeros
            for param in merged.parameters():
                param.data.zero_()

            # Add weighted contributions from each model
            for model, weight in zip(losers, weights):
                for (name_merged, param_merged), (name_orig, param_orig) in zip(
                    merged.named_parameters(),
                    model.model.named_parameters()
                ):
                    param_merged.data.add_(param_orig.data * weight)

        merged_model = MergedModel(
            model=merged,
            generation=generation,
            parents=[m.model_id for m in losers],
            merge_technique="loser_merge",
            fitness=0.0,
            model_id=self.get_model_id(generation, "loser_group")
        )

        return merged_model

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
                        # Linear for small angles
                        merged_flat = alpha * p1_flat + (1 - alpha) * p2_flat

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
                # Random weighted average
                w1 = random.random()
                w2 = 1 - w1
                for (name1, param1), (name2, param2) in zip(
                    model1.named_parameters(), model2.named_parameters()
                ):
                    merged.state_dict()[name1].copy_(
                        w1 * param1.data + w2 * param2.data
                    )

            elif technique == 'interpolate':
                # Layer-wise interpolation
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
                    logger.warning(f"Error evaluating model {model.model_id}: {e}")
                    return 0.0

        # Fitness is inverse of average loss
        avg_loss = total_loss / num_samples
        fitness = 1.0 / (1.0 + avg_loss)

        return fitness

    def create_next_generation(self, population: List[MergedModel], generation: int) -> List[MergedModel]:
        """
        Create next generation following the correct strategy:
        - Top 2 winners each create 3 mutated children (6 total)
        - Bottom 6 losers grouped into 2 sets, each merged into 1 child (2 total)
        """
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Identify winners and losers
        winners = population[:2]
        losers = population[2:]

        new_generation = []

        # Winners create children
        logger.info(f"  Winners: {winners[0].model_id} (fitness: {winners[0].fitness:.4f}), "
                   f"{winners[1].model_id} (fitness: {winners[1].fitness:.4f})")

        for i, winner in enumerate(winners):
            logger.info(f"  Creating 3 children from winner {i+1}")
            children = self.mutate_winner(winner, generation)
            new_generation.extend(children)

        # Losers merge into wildcards
        logger.info(f"  Merging 6 losers into 2 wildcard children")

        # Group 1: First 3 losers
        loser_group1 = losers[:3]
        wildcard1 = self.merge_loser_group(loser_group1, generation)
        new_generation.append(wildcard1)

        # Group 2: Last 3 losers
        loser_group2 = losers[3:6]
        wildcard2 = self.merge_loser_group(loser_group2, generation)
        new_generation.append(wildcard2)

        logger.info(f"  New generation: 6 winner children + 2 loser wildcards = {len(new_generation)} models")

        return new_generation

    def run_evolution(self):
        """Run the correct evolutionary merging process"""
        logger.info("=== Starting Correct Cognate EvoMerge Evolution ===")
        logger.info("Strategy: Winners spawn children, losers merge into wildcards")

        # Load base models
        base_models = self.load_base_models()

        # Generation 1: Create initial 8 models
        logger.info("\n=== GENERATION 1: Creating Initial Population ===")
        population = self.create_initial_population(base_models)

        # Evaluate Generation 1
        logger.info("Benchmarking all 8 models...")
        for model in population:
            model.fitness = self.evaluate_model(model)
            logger.info(f"  {model.model_id}: fitness = {model.fitness:.4f} (technique: {model.merge_technique})")

        # Sort and identify winners/losers
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Record history
        self.evolution_history.append({
            'generation': 1,
            'best_fitness': population[0].fitness,
            'avg_fitness': sum(m.fitness for m in population) / len(population),
            'best_model': population[0].model_id,
            'winner_ids': [population[0].model_id, population[1].model_id]
        })

        # Generations 2-50
        for gen in range(2, self.generations + 1):
            start_time = time.time()

            logger.info(f"\n=== GENERATION {gen}/{self.generations} ===")

            # Create next generation
            logger.info("Creating next generation...")
            population = self.create_next_generation(population, gen)

            # Benchmark all 8 new models
            logger.info("Benchmarking all 8 models...")
            for model in population:
                model.fitness = self.evaluate_model(model)
                logger.info(f"  {model.model_id}: fitness = {model.fitness:.4f}")

            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            best_fitness = population[0].fitness
            avg_fitness = sum(m.fitness for m in population) / len(population)

            logger.info(f"Best fitness: {best_fitness:.4f} (model: {population[0].model_id})")
            logger.info(f"Average fitness: {avg_fitness:.4f}")

            # Record history
            self.evolution_history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_model': population[0].model_id,
                'winner_ids': [population[0].model_id, population[1].model_id]
            })

            elapsed = time.time() - start_time
            logger.info(f"Generation time: {elapsed:.1f}s")

            # Save checkpoint every 10 generations
            if gen % 10 == 0:
                self.save_checkpoint(population[0], gen)

            # Check convergence (disabled to run full 50 generations)
            # if len(self.evolution_history) > 10:
            #     recent_fitness = [h['best_fitness'] for h in self.evolution_history[-5:]]
            #     if all(abs(f - best_fitness) < 0.0001 for f in recent_fitness):
            #         logger.info("Convergence detected - stopping early")
            #         break

            # Log convergence status but continue
            if len(self.evolution_history) > 10:
                recent_fitness = [h['best_fitness'] for h in self.evolution_history[-5:]]
                if all(abs(f - best_fitness) < 0.0001 for f in recent_fitness):
                    logger.info("Note: Fitness has converged but continuing to generation 50")

        # Save final best model
        self.save_final_model(population[0])

        logger.info("\n=== Evolution Complete ===")
        logger.info(f"Final best fitness: {population[0].fitness:.4f}")
        logger.info(f"Final best model: {population[0].model_id}")

        return population[0]

    def save_checkpoint(self, best_model: MergedModel, generation: int):
        """Save a checkpoint"""
        checkpoint_dir = project_root / "checkpoints" / "correct_evomerge"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"gen_{generation}_best.pt"

        torch.save({
            'generation': generation,
            'model_state_dict': best_model.model.state_dict(),
            'config': best_model.model.config,
            'fitness': best_model.fitness,
            'model_id': best_model.model_id,
            'parents': best_model.parents
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self, best_model: MergedModel):
        """Save the final best model"""
        final_dir = project_root / "models" / "correct_evomerge_final"
        final_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        final_path = final_dir / "best_evolved_model.pt"
        torch.save({
            'model_state_dict': best_model.model.state_dict(),
            'config': best_model.model.config,
            'fitness': best_model.fitness,
            'model_id': best_model.model_id,
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
    logger.info("=== CORRECT EVOMERGE FOR COGNATE MODELS ===\n")
    logger.info("Evolution Strategy:")
    logger.info("  - Generation 1: Create 8 models from 3 base models")
    logger.info("  - Generations 2-50:")
    logger.info("    * Top 2 winners → 3 mutated children each (6 total)")
    logger.info("    * Bottom 6 losers → 2 groups of 3, merged into 2 wildcards")
    logger.info("    * Total: 8 new models per generation\n")

    # Create and run evolver
    evolver = CorrectCognateEvoMerge(generations=50)
    best_model = evolver.run_evolution()

    logger.info("\n=== SUCCESS ===")
    logger.info("EvoMerge completed successfully!")
    logger.info(f"Best model achieved fitness: {best_model.fitness:.4f}")

    # Print evolution summary
    if evolver.evolution_history:
        initial_fitness = evolver.evolution_history[0]['best_fitness']
        final_fitness = evolver.evolution_history[-1]['best_fitness']
        improvement = (final_fitness / initial_fitness - 1) * 100 if initial_fitness > 0 else 0
        logger.info(f"Improvement over {len(evolver.evolution_history)} generations: {improvement:.1f}%")

    return best_model


if __name__ == "__main__":
    model = main()