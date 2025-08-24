#!/usr/bin/env python3
"""
Demo 50-Generation EvoMerge Process - Offline Demonstration

This script demonstrates the EvoMerge pipeline with synthetic models
to show the 50-generation evolution process without requiring external
model downloads.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

# Add current directory to Python path
sys.path.insert(0, os.path.abspath("."))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("demo_evomerge_50gen.log")],
)
logger = logging.getLogger(__name__)


class MockModel:
    """Mock model for demonstration purposes."""

    def __init__(self, name: str, fitness_scores: dict = None):
        self.name = name
        self.config = {"model_type": "mock", "name": name}
        self.fitness_scores = fitness_scores or {}
        self.parameters = random.randint(15000000, 17000000)  # ~16M parameters like HRRM

    def save_pretrained(self, path):
        """Mock save function."""
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(Path(path) / "config.json", "w") as f:
            json.dump(self.config, f)
        logger.info(f"Mock model saved to {path}")


class MockEvaluator:
    """Mock evaluator for demonstration."""

    def __init__(self):
        self.domains = ["code", "math", "multilingual", "structured_data"]

    async def evaluate(self, model, tokenizer=None):
        """Mock evaluation with realistic score variations."""
        scores = {}

        # Simulate realistic performance scores with some randomness
        base_performance = 0.6 + random.random() * 0.3  # 0.6 to 0.9

        for domain in self.domains:
            # Add domain-specific variation
            domain_modifier = random.uniform(0.85, 1.15)
            score = min(0.95, max(0.3, base_performance * domain_modifier))
            scores[domain] = score

        return scores


class MockMergeOperators:
    """Mock merge operators for demonstration."""

    @staticmethod
    def linear_merge(models, weights=None):
        """Mock linear merge."""
        if not models:
            return None
        # Create new model with combined fitness
        combined_fitness = {}
        for domain in ["code", "math", "multilingual", "structured_data"]:
            scores = [m.fitness_scores.get(domain, 0.5) for m in models if hasattr(m, "fitness_scores")]
            combined_fitness[domain] = np.mean(scores) if scores else 0.5

        result = MockModel(f"linear_merge_{len(models)}_models")
        result.fitness_scores = combined_fitness
        return result

    @staticmethod
    def slerp_merge(model1, model2, t=0.5):
        """Mock SLERP merge."""
        combined_fitness = {}
        for domain in ["code", "math", "multilingual", "structured_data"]:
            score1 = model1.fitness_scores.get(domain, 0.5)
            score2 = model2.fitness_scores.get(domain, 0.5)
            combined_fitness[domain] = score1 * (1 - t) + score2 * t

        result = MockModel(f"slerp_merge_{model1.name}_{model2.name}")
        result.fitness_scores = combined_fitness
        return result

    @staticmethod
    def ties_merge(models):
        """Mock TIES merge."""
        return MockMergeOperators.linear_merge(models)

    @staticmethod
    def dare_merge(models):
        """Mock DARE merge."""
        return MockMergeOperators.linear_merge(models)

    @staticmethod
    def frankenmerge(models):
        """Mock Frankenmerge."""
        return MockMergeOperators.linear_merge(models)

    @staticmethod
    def dfs_merge(models):
        """Mock DFS merge."""
        return MockMergeOperators.linear_merge(models)


class MergeCandidate:
    """Merge candidate representation."""

    def __init__(self, model_path: str, merge_recipe: dict, generation: int = 0):
        self.model_path = model_path
        self.merge_recipe = merge_recipe
        self.generation = generation
        self.fitness_scores = {}
        self.aggregated_fitness = 0.0
        self.parents = None

    def calculate_aggregated_fitness(self, weights: dict):
        """Calculate weighted fitness."""
        self.aggregated_fitness = sum(self.fitness_scores.get(domain, 0) * weight for domain, weight in weights.items())
        return self.aggregated_fitness


async def demo_50_generation_evomerge():
    """Run a demonstration of 50-generation EvoMerge with correct breeding algorithm."""

    logger.info("[START]" + "=" * 73)
    logger.info("[START] DEMO 50-GENERATION EVOMERGE WITH CORRECT BREEDING")
    logger.info("[START]" + "=" * 73)

    start_time = time.time()

    # Setup
    output_dir = Path("demo_evomerge_output")
    output_dir.mkdir(exist_ok=True)

    merge_ops = MockMergeOperators()
    evaluator = MockEvaluator()

    # Create initial 3 core models (mock HRRM models)
    logger.info("[SEED] Creating initial 3 core HRRM models...")
    core_models = [
        MockModel("hrrm_planner", {"code": 0.7, "math": 0.6, "multilingual": 0.5, "structured_data": 0.8}),
        MockModel("hrrm_reasoner", {"code": 0.6, "math": 0.8, "multilingual": 0.6, "structured_data": 0.7}),
        MockModel("hrrm_memory", {"code": 0.5, "math": 0.7, "multilingual": 0.8, "structured_data": 0.6}),
    ]

    # Generate initial population using all 8 merge technique combinations
    logger.info("[DNA] Generating Generation 0: 3 core models with 8 merge technique combinations...")

    # All 8 merge technique combinations (2^3 pairs: primary/secondary/final)
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

    population = []
    for i, recipe in enumerate(merge_recipes):
        # Apply merge sequence using all 3 core models

        # Step 1: Primary merge on all 3 core models
        if recipe["primary"] == "linear":
            primary_merged = merge_ops.linear_merge(core_models)
        else:  # slerp - combine first two, then merge with third
            temp_merge = merge_ops.slerp_merge(core_models[0], core_models[1])
            primary_merged = merge_ops.slerp_merge(temp_merge, core_models[2])

        # Step 2: Secondary merge - blend primary result with core models
        if recipe["secondary"] == "ties":
            secondary_merged = merge_ops.ties_merge([primary_merged] + core_models)
        else:  # dare
            secondary_merged = merge_ops.dare_merge([primary_merged] + core_models)

        # Step 3: Final merge - apply final technique
        if recipe["final"] == "frankenmerge":
            final_merged = merge_ops.frankenmerge([secondary_merged] + core_models)
        else:  # dfs
            final_merged = merge_ops.dfs_merge([secondary_merged] + core_models)

        # Create candidate
        model_path = output_dir / f"gen0_candidate{i}"
        final_merged.save_pretrained(model_path)

        candidate = MergeCandidate(str(model_path), recipe, generation=0)
        candidate.fitness_scores = final_merged.fitness_scores
        candidate.calculate_aggregated_fitness(
            {"code": 0.25, "math": 0.25, "multilingual": 0.25, "structured_data": 0.25}
        )

        population.append(candidate)
        technique_desc = f"{recipe['primary']}->{recipe['secondary']}->{recipe['final']}"
        logger.info(f"   [OK] Created candidate {i+1}/8 ({technique_desc}): fitness={candidate.aggregated_fitness:.4f}")

    # Quick benchmark - sort initial population by fitness
    population.sort(key=lambda x: x.aggregated_fitness, reverse=True)
    best_fitness = population[0].aggregated_fitness
    fitness_history = [best_fitness]
    diversity_history = []

    logger.info(f"[BENCHMARK] Generation 0 complete - Best fitness: {best_fitness:.4f}")
    logger.info(f"[RANK] Top 2 winners: {population[0].aggregated_fitness:.4f}, {population[1].aggregated_fitness:.4f}")
    logger.info(f"[RANK] Bottom 6 losers: {[p.aggregated_fitness for p in population[2:]]}")

    # Evolution loop for 50 generations
    generations = 50
    generation_history = [population[:]]  # Keep track of generations for cleanup

    for gen in range(generations):
        logger.info(f"\n[GEN] Generation {gen + 1}/{generations}")

        # Calculate diversity (simple variance measure)
        fitnesses = [c.aggregated_fitness for c in population]
        diversity = np.var(fitnesses)
        diversity_history.append(diversity)

        # Sort population by fitness (if not already sorted)
        population.sort(key=lambda x: x.aggregated_fitness, reverse=True)

        # CORRECT BREEDING ALGORITHM:
        # Top 2 winners → 3 children each = 6 winner children
        # Bottom 6 losers → grouped into 2 groups of 3 → 2 loser children
        # Total: 6 + 2 = 8 children for next generation

        next_gen = []

        # WINNER BRANCH: Top 2 models each get 3 mutated children
        logger.info("   [WINNERS] Top 2 models creating 6 children...")
        winners = population[:2]

        for winner_idx, winner in enumerate(winners):
            for child_idx in range(3):  # 3 children per winner
                # Mutate the winner to create child
                child_fitness = {}
                for domain in evaluator.domains:
                    parent_score = winner.fitness_scores.get(domain, 0.5)
                    # Mutation with smaller range for winners
                    mutated_score = parent_score + random.uniform(-0.05, 0.05)
                    mutated_score = max(0.1, min(0.95, mutated_score))
                    child_fitness[domain] = mutated_score

                # Create winner child
                model_path = output_dir / f"gen{gen+1}_winner{winner_idx}_child{child_idx}"
                child_model = MockModel(f"gen{gen+1}_winner{winner_idx}_child{child_idx}")
                child_model.fitness_scores = child_fitness
                child_model.save_pretrained(model_path)

                offspring = MergeCandidate(
                    str(model_path),
                    {"type": "winner_mutation", "parent": winner.model_path, "winner_rank": winner_idx},
                    generation=gen + 1,
                )
                offspring.fitness_scores = child_fitness
                offspring.calculate_aggregated_fitness(
                    {"code": 0.25, "math": 0.25, "multilingual": 0.25, "structured_data": 0.25}
                )

                next_gen.append(offspring)
                logger.info(
                    f"     [CHILD] Winner {winner_idx+1} child {child_idx+1}: fitness={offspring.aggregated_fitness:.4f}"
                )

        # LOSER BRANCH: Bottom 6 models → 2 groups of 3 → 2 merged children
        logger.info("   [LOSERS] Bottom 6 models creating 2 merged children...")
        losers = population[2:]  # Bottom 6

        # Split into 2 groups of 3
        loser_groups = [losers[:3], losers[3:]]

        for group_idx, loser_group in enumerate(loser_groups):
            # Merge the 3 losers into 1 child
            combined_fitness = {}
            for domain in evaluator.domains:
                scores = [loser.fitness_scores.get(domain, 0.5) for loser in loser_group]
                combined_score = np.mean(scores)
                # Add some noise to the merge
                combined_score += random.uniform(-0.03, 0.03)
                combined_score = max(0.1, min(0.95, combined_score))
                combined_fitness[domain] = combined_score

            # Create loser child
            model_path = output_dir / f"gen{gen+1}_loser_group{group_idx}"
            child_model = MockModel(f"gen{gen+1}_loser_group{group_idx}")
            child_model.fitness_scores = combined_fitness
            child_model.save_pretrained(model_path)

            offspring = MergeCandidate(
                str(model_path),
                {"type": "loser_merge", "parents": [loser.model_path for loser in loser_group], "group": group_idx},
                generation=gen + 1,
            )
            offspring.fitness_scores = combined_fitness
            offspring.calculate_aggregated_fitness(
                {"code": 0.25, "math": 0.25, "multilingual": 0.25, "structured_data": 0.25}
            )

            next_gen.append(offspring)
            logger.info(f"     [MERGE] Loser group {group_idx+1}: fitness={offspring.aggregated_fitness:.4f}")

        # Verify we have exactly 8 children
        assert len(next_gen) == 8, f"Expected 8 children, got {len(next_gen)}"

        # Benchmark the new generation
        population = next_gen
        population.sort(key=lambda x: x.aggregated_fitness, reverse=True)
        generation_history.append(population[:])  # Store this generation

        # GENERATION CLEANUP: Delete generation n-2 when we create generation n+1
        # Keep rolling window of 2 generations (current + previous)
        if gen >= 2:  # Starting from generation 3, delete generation 1
            cleanup_gen = gen - 2
            logger.info(f"   [CLEANUP] Deleting generation {cleanup_gen} files...")
            # In real implementation, would delete model files here
            if cleanup_gen < len(generation_history) - 2:
                del generation_history[cleanup_gen]

        # Track best fitness
        current_best = population[0].aggregated_fitness
        if current_best > best_fitness:
            best_fitness = current_best
        fitness_history.append(best_fitness)

        # Log generation stats with breeding breakdown
        avg_fitness = np.mean([c.aggregated_fitness for c in population])
        logger.info(f"   [BENCHMARK] Generation {gen+1} complete - Best: {current_best:.4f}, Avg: {avg_fitness:.4f}")
        logger.info("   [BREEDING] 6 winner children + 2 loser children = 8 total")
        logger.info(f"   [DIVERSITY] Population variance: {diversity:.6f}")

        # Show best scores by domain every 10 generations
        if (gen + 1) % 10 == 0:
            logger.info("   [SCORES] Best model domain scores:")
            for domain, score in population[0].fitness_scores.items():
                logger.info(f"     {domain:15s}: {score:.4f}")

        # Check convergence
        if gen > 5:
            recent_improvement = fitness_history[-1] - fitness_history[-6]
            if recent_improvement < 0.001:
                logger.info(f"   [CONVERGED] Early convergence at generation {gen + 1}")
                break

    end_time = time.time()
    total_time = end_time - start_time

    # Final results
    logger.info("[OK]" + "=" * 73)
    logger.info("[OK] CORRECTED 50-GENERATION EVOMERGE DEMO COMPLETED!")
    logger.info("[OK]" + "=" * 73)

    logger.info(f"[TIME] Total Runtime: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    logger.info(f"[TROPHY] Final Fitness: {best_fitness:.4f}")
    logger.info(f"[CYCLE] Generations Run: {gen + 1}")
    logger.info(f"[CHART] Fitness Improvement: {best_fitness - fitness_history[0]:.4f}")

    # Final domain scores
    logger.info("[TARGET] Final Best Model Domain Scores:")
    for domain, score in population[0].fitness_scores.items():
        logger.info(f"   {domain:20s}: {score:.4f}")

    # Algorithm summary
    logger.info("[ALGORITHM] Correct Breeding Algorithm Used:")
    logger.info("   [DNA] Generation 0: 3 core models → 2^3 = 8 combinations")
    logger.info("   [CYCLE] Each Generation: Top 2 → 6 children, Bottom 6 → 2 children")
    logger.info("   [CLEANUP] Rolling cleanup: Delete generation n-2 when creating n+1")
    logger.info("   [RATIO] 6:2 winner-to-loser breeding ratio maintained")

    # Save results
    results = {
        "demo_completed": True,
        "algorithm_type": "corrected_breeding",
        "total_runtime_seconds": total_time,
        "generations_completed": gen + 1,
        "final_fitness": best_fitness,
        "fitness_improvement": best_fitness - fitness_history[0],
        "final_scores": population[0].fitness_scores,
        "fitness_history": fitness_history,
        "diversity_history": diversity_history,
        "best_model_path": population[0].model_path,
        "merge_recipe": population[0].merge_recipe,
        "population_size": len(population),
        "breeding_algorithm": {
            "initial_combinations": "3_core_models_to_8_via_2^3",
            "winner_children_per_generation": 6,
            "loser_children_per_generation": 2,
            "winners_per_generation": 2,
            "losers_per_generation": 6,
            "winner_mutation_range": "±0.05",
            "loser_merge_noise": "±0.03",
            "cleanup_policy": "delete_generation_n_minus_2",
        },
        "merge_techniques_used": ["linear", "slerp", "ties", "dare", "frankenmerge", "dfs"],
        "evolution_stats": {
            "avg_fitness_final": np.mean([c.aggregated_fitness for c in population]),
            "diversity_final": diversity_history[-1] if diversity_history else 0,
            "convergence_generation": gen + 1,
        },
    }

    results_path = output_dir / "demo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"[FILE] Demo results saved to: {results_path}")

    # Integration next steps
    logger.info("[CYCLE] Next Steps for Agent Forge Integration:")
    logger.info("   1. [OK] EvoMerge Phase Demo Complete")
    logger.info("   2. [CYCLE] Ready for Quiet-STaR Phase")
    logger.info("   3. [CYCLE] Ready for BitNet Compression Phase")
    logger.info("   4. [CYCLE] Ready for Full Agent Forge Pipeline")

    return results


def main():
    """Main execution function."""

    print("[DEMO]" + "=" * 74)
    print("[DEMO] AIVillage EvoMerge - 50 Generation Demo (Offline)")
    print("[DEMO]" + "=" * 74)
    print("[INFO] This demo shows the EvoMerge pipeline with mock models")
    print("[INFO] Demonstrates 6 merge techniques across 50 generations")
    print("[INFO] Expected runtime: 1-2 minutes")
    print("[DEMO]" + "=" * 74)
    print()

    # Run the demo
    results = asyncio.run(demo_50_generation_evomerge())

    if results and results["demo_completed"]:
        print("\n[SUCCESS] EvoMerge demo completed successfully!")
        print(f"[STATS] Fitness improved by {results['fitness_improvement']:.4f}")
        print(f"[STATS] Ran {results['generations_completed']} generations")
        print("[FOLDER] Results saved in: demo_evomerge_output/")
        return 0
    else:
        print("\n[FAILED] EvoMerge demo encountered errors")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
