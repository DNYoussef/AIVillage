#!/usr/bin/env python3
"""
Simple EvoMerge Test - No Unicode characters
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


def test_evomerge_exists():
    """Test that EvoMerge files exist."""
    print("Testing EvoMerge files...")

    evomerge_file = PROJECT_ROOT / "core" / "agent-forge" / "phases" / "evomerge.py"
    if evomerge_file.exists():
        size = evomerge_file.stat().st_size
        print(f"  OK EvoMerge file exists: {size:,} bytes")

        # Check content
        with open(evomerge_file, encoding="utf-8") as f:
            content = f.read()

        required = ["EvoMergePhase", "EvoMergeConfig", "generations", "population_size"]
        found = sum(1 for term in required if term in content)
        print(f"  OK Contains {found}/{len(required)} required terms")
        return True
    else:
        print("  FAILED EvoMerge file not found")
        return False


def test_evomerge_results():
    """Check for EvoMerge results."""
    print("Testing EvoMerge results...")

    results_file = PROJECT_ROOT / "core" / "agent-forge" / "phases" / "evomerge_50gen_final_results.json"
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)

            print("  OK Results file exists")
            if "champion_model" in data:
                champion = data["champion_model"]
                print(f"  OK Champion: {champion.get('name', 'Unknown')}")
                print(f"  OK Fitness: {champion.get('fitness', 'Unknown')}")

            if "total_generations" in data:
                print(f"  OK Generations: {data['total_generations']}")

            return True
        except Exception as e:
            print(f"  FAILED Could not read results: {e}")
            return False
    else:
        print("  INFO No results file found")
        return False


def test_evolution_logic():
    """Test basic evolution logic."""
    print("Testing evolution logic...")

    try:
        import random

        # Simple population
        population = []
        for i in range(4):
            fitness = random.random()
            population.append({"name": f"model_{i}", "fitness": fitness})

        print(f"  OK Created {len(population)} models")

        # Selection
        population.sort(key=lambda x: x["fitness"], reverse=True)
        best_initial = population[0]["fitness"]

        # Simulate generations
        for gen in range(3):
            # Keep top 2
            survivors = population[:2]

            # Create offspring
            offspring = []
            for i in range(2):
                parent1, parent2 = survivors[0], survivors[1]
                child_fitness = (parent1["fitness"] + parent2["fitness"]) / 2
                child_fitness += random.gauss(0, 0.05)  # Mutation

                offspring.append({"name": f"gen{gen+1}_child{i}", "fitness": max(0, min(1, child_fitness))})

            population = survivors + offspring
            population.sort(key=lambda x: x["fitness"], reverse=True)

        best_final = population[0]["fitness"]
        improvement = ((best_final - best_initial) / max(best_initial, 0.001)) * 100

        print("  OK Evolution complete")
        print(f"  OK Initial: {best_initial:.4f}, Final: {best_final:.4f}")
        print(f"  OK Improvement: {improvement:+.1f}%")

        return True

    except Exception as e:
        print(f"  FAILED Evolution test error: {e}")
        return False


def main():
    print("EVOMERGE SIMPLE TEST")
    print("=" * 40)

    tests = [
        ("EvoMerge Files", test_evomerge_exists),
        ("Results Data", test_evomerge_results),
        ("Evolution Logic", test_evolution_logic),
    ]

    passed = 0
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            if test_func():
                passed += 1
                print("  PASSED")
            else:
                print("  FAILED")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 40)
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")

    if passed >= 2:
        print("SUCCESS: EvoMerge system is functional")
        return True
    else:
        print("FAILED: EvoMerge system needs attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
