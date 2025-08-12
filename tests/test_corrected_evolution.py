#!/usr/bin/env python3
"""
Test script to validate the corrected evolution system with 3-5 generations.

This validates:
1. Generation 1 has exactly 8 systematic combinations
2. Breeding produces exactly 8 models per generation (2→6 + 6→2)
3. Population size remains 8 throughout
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_corrected_evolution import CorrectedEvolutionMerger

# Import the corrected evolution system
# Add scripts to path for module resolution
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("D:/AgentForge/test_corrected/validation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class EvolutionValidator:
    """Validator for the corrected evolution system."""

    def __init__(self):
        self.test_output_dir = Path("D:/AgentForge/test_corrected")
        self.test_output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results = []

    def validate_generation_1(
        self, merger: CorrectedEvolutionMerger
    ) -> dict[str, bool]:
        """Validate that Generation 1 is systematic and complete."""
        logger.info("=== VALIDATING GENERATION 1 ===")

        population = merger.population
        validation = {
            "correct_population_size": len(population) == 8,
            "all_systematic_combinations": True,
            "all_generation_1": True,
            "unique_combinations": True,
        }

        # Check population size
        logger.info("Population size: %s (Expected: 8)", len(population))

        # Check that all are from generation 1
        generations = [ind.get("generation", 0) for ind in population]
        all_gen_1 = all(gen == 1 for gen in generations)
        validation["all_generation_1"] = all_gen_1
        logger.info("All from generation 1: %s", all_gen_1)

        # Check systematic combinations
        expected_combinations = []
        for interp in ["slerp", "linear"]:
            for arith in ["task_arithmetic", "ties"]:
                for adv in ["dare_ties", "model_soup"]:
                    expected_combinations.append(tuple(sorted([interp, arith, adv])))

        actual_combinations = []
        for ind in population:
            combo = ind.get("technique_combination", [])
            if combo:
                actual_combinations.append(tuple(sorted(combo)))

        # Check if we have all expected combinations
        expected_set = set(expected_combinations)
        actual_set = set(actual_combinations)

        validation["all_systematic_combinations"] = expected_set == actual_set
        validation["unique_combinations"] = len(actual_combinations) == len(
            set(actual_combinations)
        )

        logger.info("Expected combinations: %s", len(expected_set))
        logger.info("Actual combinations: %s", len(actual_set))
        logger.info(
            "All systematic combinations present: %s",
            validation["all_systematic_combinations"],
        )
        logger.info("All combinations unique: %s", validation["unique_combinations"])

        # Log all combinations for verification
        logger.info("Actual combinations found:")
        for i, combo in enumerate(actual_combinations):
            logger.info("  %s. %s", i + 1, combo)

        return validation

    def validate_breeding_logic(
        self, merger: CorrectedEvolutionMerger, generation: int
    ) -> dict[str, bool]:
        """Validate breeding logic for a specific generation."""
        logger.info("=== VALIDATING BREEDING LOGIC FOR GENERATION %s ===", generation)

        # Get population before breeding
        pre_breeding_pop = merger.population.copy()

        # Sort by fitness to understand breeding inputs
        ranked_population = sorted(
            pre_breeding_pop, key=lambda x: x["fitness"], reverse=True
        )

        logger.info("Pre-breeding population (ranked by fitness):")
        for i, ind in enumerate(ranked_population):
            logger.info(
                "  %s. %s - Fitness: %.4f - Method: %s",
                i + 1,
                ind["id"],
                ind["fitness"],
                ind["primary_method"],
            )

        # Simulate breeding
        next_gen = merger.breed_next_generation(ranked_population)

        validation = {
            "correct_population_size": len(next_gen) == 8,
            "correct_mutant_count": 0,
            "correct_child_count": 0,
            "all_new_generation": True,
        }

        # Count breeding types
        mutant_count = len(
            [ind for ind in next_gen if ind.get("breeding_type") == "mutation"]
        )
        child_count = len(
            [ind for ind in next_gen if ind.get("breeding_type") == "triad_merge"]
        )

        validation["correct_mutant_count"] = mutant_count == 6
        validation["correct_child_count"] = child_count == 2
        validation["correct_population_size"] = len(next_gen) == 8

        # Check generation numbers
        expected_gen = generation + 1
        actual_generations = [ind.get("generation", 0) for ind in next_gen]
        validation["all_new_generation"] = all(
            gen == expected_gen for gen in actual_generations
        )

        logger.info("Next generation size: %s (Expected: 8)", len(next_gen))
        logger.info("Mutants: %s (Expected: 6)", mutant_count)
        logger.info("Children: %s (Expected: 2)", child_count)
        logger.info(
            "All generation %s: %s", expected_gen, validation["all_new_generation"]
        )

        # Log breeding details
        logger.info("Next generation breakdown:")
        for ind in next_gen:
            breeding_type = ind.get("breeding_type", "unknown")
            parent_ids = ind.get("parent_ids", [])
            logger.info(
                "  %s - Type: %s - Parents: %s", ind["id"], breeding_type, parent_ids
            )

        return validation

    def run_mini_evolution_test(self, max_generations: int = 3) -> dict[str, Any]:
        """Run a mini evolution test for validation."""
        logger.info("=" * 80)
        logger.info("RUNNING MINI EVOLUTION TEST (%s GENERATIONS)", max_generations)
        logger.info("=" * 80)

        # Create corrected evolution system
        merger = CorrectedEvolutionMerger(output_dir=str(self.test_output_dir))

        # Override max generations for testing
        merger.max_generations = max_generations

        test_results = {
            "generation_1_validation": None,
            "breeding_validations": [],
            "population_sizes": [],
            "all_tests_passed": True,
        }

        # Validate Generation 1
        gen1_validation = self.validate_generation_1(merger)
        test_results["generation_1_validation"] = gen1_validation

        if not all(gen1_validation.values()):
            logger.error("❌ Generation 1 validation FAILED")
            test_results["all_tests_passed"] = False
        else:
            logger.info("✅ Generation 1 validation PASSED")

        # Run evolution and validate each generation
        start_time = time.time()

        try:
            generation_count = 0
            while merger.evolve_generation() and generation_count < max_generations:
                generation_count += 1

                # Record population size
                test_results["population_sizes"].append(len(merger.population))

                # Validate breeding logic (if not the last generation)
                if generation_count < max_generations:
                    breeding_validation = self.validate_breeding_logic(
                        merger, merger.generation
                    )
                    test_results["breeding_validations"].append(
                        {
                            "generation": merger.generation,
                            "validation": breeding_validation,
                        }
                    )

                    if not all(breeding_validation.values()):
                        logger.error(
                            "❌ Breeding validation FAILED for generation %s",
                            merger.generation,
                        )
                        test_results["all_tests_passed"] = False
                    else:
                        logger.info(
                            "✅ Breeding validation PASSED for generation %s",
                            merger.generation,
                        )

                logger.info("Completed generation %s", merger.generation)

        except Exception as e:
            logger.exception("Evolution test failed with error: %s", e)
            test_results["all_tests_passed"] = False
            test_results["error"] = str(e)

        end_time = time.time()
        duration = end_time - start_time

        # Final validation summary
        logger.info("=" * 80)
        logger.info("MINI EVOLUTION TEST SUMMARY")
        logger.info("=" * 80)
        logger.info("Duration: %.2f seconds", duration)
        logger.info("Generations completed: %s", generation_count)
        logger.info("Population sizes: %s", test_results["population_sizes"])

        # Check population size consistency
        expected_sizes = [8] * len(test_results["population_sizes"])
        consistent_population = test_results["population_sizes"] == expected_sizes
        test_results["consistent_population_size"] = consistent_population

        if not consistent_population:
            logger.error(
                "❌ Population size inconsistent: %s", test_results["population_sizes"]
            )
            test_results["all_tests_passed"] = False
        else:
            logger.info("✅ Population size consistent (8 throughout)")

        # Overall result
        if test_results["all_tests_passed"]:
            logger.info(
                "🎉 ALL TESTS PASSED - Corrected evolution system is working correctly!"
            )
        else:
            logger.error("❌ SOME TESTS FAILED - System needs further correction")

        # Save test results
        test_file = (
            self.test_output_dir / f"validation_results_{max_generations}gen.json"
        )
        with open(test_file, "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        logger.info("Validation results saved to: %s", test_file)

        return test_results


def main():
    """Main validation function."""
    logger.info("Starting corrected evolution validation...")

    # Set random seeds for reproducible testing
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create validator
    validator = EvolutionValidator()

    # Run mini evolution test
    test_results = validator.run_mini_evolution_test(max_generations=3)

    # Return success/failure
    return test_results["all_tests_passed"]


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Validation PASSED - Ready for full 50-generation run!")
        sys.exit(0)
    else:
        print("\n❌ Validation FAILED - System needs correction")
        sys.exit(1)
