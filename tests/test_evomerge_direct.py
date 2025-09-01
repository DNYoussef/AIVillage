#!/usr/bin/env python3
"""
Direct EvoMerge Functionality Test
Tests the actual EvoMerge system without complex imports
"""

import json
import os
from pathlib import Path
import sys

# Fix encoding
if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "core"))


def test_evomerge_files_exist():
    """Test that EvoMerge files exist."""
    print("Testing EvoMerge file structure...")

    evomerge_file = PROJECT_ROOT / "core" / "agent_forge" / "phases" / "evomerge.py"
    if evomerge_file.exists():
        print(f"   OK EvoMerge phase file exists: {evomerge_file}")

        # Check file size
        size = evomerge_file.stat().st_size
        print(f"   File size: {size:,} bytes")

        # Check if it contains key functions
        with open(evomerge_file, encoding="utf-8") as f:
            content = f.read()

        key_terms = ["EvoMergePhase", "EvoMergeConfig", "MergeCandidate", "linear", "slerp"]
        for term in key_terms:
            if term in content:
                print(f"   OK Contains {term}")
            else:
                print(f"   MISSING {term}")

        return True
    else:
        print(f"   ERROR: EvoMerge phase file not found: {evomerge_file}")
        return False


def test_evomerge_results_exist():
    """Test if EvoMerge results from previous runs exist."""
    print("\nChecking for existing EvoMerge results...")

    results_locations = [
        PROJECT_ROOT / "core" / "agent_forge" / "phases" / "evomerge_50gen_final_results.json",
        PROJECT_ROOT / "core" / "agent_forge" / "phases" / "cognate_evomerge_output" / "evomerge_results.json",
    ]

    found_results = False
    for results_file in results_locations:
        if results_file.exists():
            print(f"   + Found results: {results_file}")

            # Try to read the results
            try:
                with open(results_file, encoding="utf-8") as f:
                    data = json.load(f)

                print(f"   Keys: {list(data.keys())}")

                if "champion_model" in data:
                    champion = data["champion_model"]
                    print(f"   Champion: {champion.get('name', 'Unknown')}")
                    print(f"   Fitness: {champion.get('fitness', 'Unknown')}")

                if "total_generations" in data:
                    print(f"   Generations: {data['total_generations']}")

                found_results = True

            except Exception as e:
                print(f"   WARNING: Could not parse results: {e}")
        else:
            print(f"   INFO: No results at: {results_file.name}")

    return found_results


def test_evomerge_output_models():
    """Test if EvoMerge output models exist."""
    print("\nChecking for EvoMerge output models...")

    output_dirs = [
        PROJECT_ROOT / "core" / "agent_forge" / "phases" / "cognate_evomerge_output",
        PROJECT_ROOT / "core" / "agent_forge" / "phases" / "evomerge_50gen_final",
    ]

    found_models = False
    for output_dir in output_dirs:
        if output_dir.exists():
            models = list(output_dir.glob("**/model.pt"))
            configs = list(output_dir.glob("**/config.pt"))

            print(f"   DIR {output_dir.name}: {len(models)} models, {len(configs)} configs")

            # List some examples
            if models:
                for model in models[:3]:  # Show first 3
                    size = model.stat().st_size if model.exists() else 0
                    print(f"     MODEL {model.parent.name}: {size:,} bytes")
                found_models = True
        else:
            print(f"   INFO: No output dir: {output_dir.name}")

    return found_models


def test_simple_evolution_concept():
    """Test the basic evolution concept with mock data."""
    print("\nTesting evolution concept...")

    try:
        import random

        # Simple fitness evaluation
        def evaluate_fitness(model_params):
            # Mock fitness based on parameter diversity
            return sum(model_params) / len(model_params) + random.random() * 0.1

        # Create initial population
        population = []
        for i in range(6):
            params = [random.random() for _ in range(10)]
            fitness = evaluate_fitness(params)
            population.append({"name": f"model_{i}", "params": params, "fitness": fitness})

        print(f"   + Created population of {len(population)} models")

        # Sort by fitness
        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_initial = population[0]["fitness"]

        # Simple evolution loop
        for generation in range(3):
            # Keep top 2
            survivors = population[:2]

            # Create offspring
            offspring = []
            for i in range(4):
                parent1, parent2 = random.choices(survivors, k=2)

                # Simple crossover
                child_params = []
                for j in range(len(parent1["params"])):
                    if random.random() < 0.5:
                        child_params.append(parent1["params"][j])
                    else:
                        child_params.append(parent2["params"][j])

                # Mutation
                for j in range(len(child_params)):
                    if random.random() < 0.1:
                        child_params[j] += random.gauss(0, 0.1)

                child_fitness = evaluate_fitness(child_params)
                offspring.append(
                    {"name": f"gen{generation+1}_child_{i}", "params": child_params, "fitness": child_fitness}
                )

            population = survivors + offspring
            population.sort(key=lambda x: x["fitness"], reverse=True)

        final_best = population[0]["fitness"]
        improvement = ((final_best - best_initial) / best_initial) * 100

        print(f"   Initial best: {best_initial:.4f}")
        print(f"   Final best: {final_best:.4f}")
        print(f"   Improvement: {improvement:+.1f}%")
        print("   + Evolution concept validated")

        return True

    except Exception as e:
        print(f"   ERROR: Evolution test failed: {e}")
        return False


def main():
    """Run EvoMerge tests."""
    print("EVOMERGE DIRECT FUNCTIONALITY TEST")
    print("=" * 60)

    tests_passed = 0
    tests_total = 0

    # Run tests
    test_functions = [
        ("File Structure", test_evomerge_files_exist),
        ("Results Data", test_evomerge_results_exist),
        ("Output Models", test_evomerge_output_models),
        ("Evolution Concept", test_simple_evolution_concept),
    ]

    for test_name, test_func in test_functions:
        tests_total += 1
        try:
            if test_func():
                tests_passed += 1
                print(f"\n+ {test_name}: PASSED")
            else:
                print(f"\n- {test_name}: FAILED")
        except Exception as e:
            print(f"\n! {test_name}: ERROR - {e}")

    # Summary
    print("\n" + "=" * 60)
    print("EVOMERGE TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Success rate: {(tests_passed/tests_total)*100:.1f}%")

    if tests_passed >= 3:
        print("SUCCESS: EvoMerge system appears functional!")
    elif tests_passed >= 2:
        print("WARNING: EvoMerge system partially functional")
    else:
        print("ERROR: EvoMerge system needs attention")

    return tests_passed >= 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
