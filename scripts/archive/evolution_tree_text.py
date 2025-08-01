#!/usr/bin/env python3
"""Create a text-based evolutionary tree representation."""

import json
from pathlib import Path


def create_text_tree() -> None:
    """Create text-based evolutionary tree."""
    # Load results
    results_file = Path("D:/AgentForge/results/evolution_results.json")
    with open(results_file) as f:
        data = json.load(f)

    print("=" * 80)
    print("AGENT FORGE EVOLUTION MERGE - EVOLUTIONARY TREE")
    print("=" * 80)

    generations = data["generation_history"]

    # Method symbols
    method_symbols = {"slerp": "S", "linear": "L", "task_arithmetic": "T"}

    print("\nEvolutionary Tree (Method | Fitness | Best in Gen)")
    print("-" * 60)

    # Track best overall
    best_overall = data["evolution_summary"]["best_configuration"]
    best_id = best_overall["id"]

    for gen_idx, gen_data in enumerate(generations):
        population = gen_data["population"]
        best_in_gen = max(population, key=lambda x: x["fitness"])

        print(f"\nGeneration {gen_idx}:")
        print("    " + "|" + "-" * 50)

        for _i, individual in enumerate(population):
            method = individual["merge_method"]
            fitness = individual["fitness"]
            ind_id = individual["id"]

            symbol = method_symbols.get(method, "?")

            # Mark best in generation and best overall
            marker = ""
            if individual == best_in_gen:
                marker += " [GEN BEST]"
            if ind_id == best_id:
                marker += " [OVERALL BEST]"

            fitness_bar = "█" * int(fitness * 20)

            print(f"    ├── [{symbol}] {fitness:.3f} {fitness_bar}{marker}")

            # Show parameters for best individuals
            if individual == best_in_gen:
                params = individual["parameters"]
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"    │   Parameters: {param_str}")

                if "benchmark_results" in individual:
                    benchmarks = individual["benchmark_results"]
                    bench_str = ", ".join(
                        [f"{k}={v:.3f}" for k, v in benchmarks.items()]
                    )
                    print(f"    │   Benchmarks: {bench_str}")

    print("\n" + "-" * 60)
    print("LEGEND:")
    print("  [S] = SLERP merge")
    print("  [L] = Linear merge")
    print("  [T] = Task Arithmetic merge")
    print("  █ = Fitness visualization (20 chars = 1.0 fitness)")

    print("\nBEST CONFIGURATION DISCOVERED:")
    print(f"  Generation: {best_overall['generation']}")
    print(f"  Method: {best_overall['merge_method']}")
    print(f"  Fitness: {best_overall['fitness']:.3f}")
    print(f"  Parameters: {best_overall['parameters']}")
    print(f"  Benchmarks: {best_overall['benchmark_results']}")

    # Evolution summary
    print("\nEVOLUTION SUMMARY:")
    summary = data["evolution_summary"]
    print(f"  Total Generations: {summary['generations_completed']}")
    print(f"  Duration: {summary['duration_seconds']:.1f} seconds")
    print(f"  Best Fitness Achieved: {summary['best_fitness']:.3f}")

    # Fitness progression
    print("\nFITNESS PROGRESSION:")
    for gen_data in generations:
        gen_num = gen_data["generation"]
        best_fit = gen_data["best_fitness"]
        avg_fit = gen_data["average_fitness"]

        best_bar = "█" * int(best_fit * 15)
        avg_bar = "▓" * int(avg_fit * 15)

        print(f"  Gen {gen_num}: Best={best_fit:.3f} {best_bar}")
        print(f"         Avg ={avg_fit:.3f} {avg_bar}")

    print("=" * 80)


if __name__ == "__main__":
    create_text_tree()
