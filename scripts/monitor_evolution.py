#!/usr/bin/env python3
"""Monitor the 50-generation evolution progress."""

import json
from pathlib import Path
import time


def monitor_progress() -> None:
    """Monitor evolution progress in real-time."""
    results_dir = Path("D:/AgentForge/results_50gen")

    print("Monitoring 50-generation evolution progress...")
    print("Press Ctrl+C to stop monitoring\n")

    last_generation = -1

    try:
        while True:
            # Find latest generation file
            generation_files = list(results_dir.glob("generation_*.json"))
            if generation_files:
                # Sort by generation number
                generation_files.sort(key=lambda x: int(x.stem.split("_")[1]))
                latest_file = generation_files[-1]

                # Extract generation number
                gen_num = int(latest_file.stem.split("_")[1])

                if gen_num > last_generation:
                    # Load and display progress
                    with open(latest_file) as f:
                        gen_data = json.load(f)

                    best_fitness = gen_data["best_fitness"]
                    avg_fitness = gen_data["average_fitness"]
                    diversity = gen_data.get("diversity", 0)
                    stagnation = gen_data.get("stagnation_counter", 0)

                    # Get best individual
                    best_ind = max(gen_data["population"], key=lambda x: x["fitness"])
                    best_method = best_ind["merge_method"]

                    progress_pct = (gen_num + 1) / 50 * 100

                    print(f"Generation {gen_num:2d}/50 ({progress_pct:5.1f}%)")
                    print(f"  Best Fitness:  {best_fitness:.4f}")
                    print(f"  Avg Fitness:   {avg_fitness:.4f}")
                    print(f"  Diversity:     {diversity:.4f}")
                    print(f"  Stagnation:    {stagnation}")
                    print(f"  Best Method:   {best_method}")

                    # Show fitness bar
                    bar_length = 30
                    filled_length = int(bar_length * best_fitness / 1.5)  # Assume max 1.5
                    bar = "█" * filled_length + "░" * (bar_length - filled_length)
                    print(f"  Fitness Bar:   [{bar}] {best_fitness:.3f}")
                    print("-" * 50)

                    last_generation = gen_num

                    # Check if complete
                    if gen_num >= 49:
                        print("\n🎉 50-Generation Evolution Complete!")
                        break

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

    # Show final summary if available
    final_file = results_dir / "evolution_50gen_results.json"
    if final_file.exists():
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)

        with open(final_file) as f:
            final_data = json.load(f)

        summary = final_data["evolution_summary"]
        best_config = summary["best_configuration"]

        print(f"Generations Completed: {summary['generations_completed']}")
        print(f"Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"Best Fitness: {summary['best_fitness']:.4f}")
        print(f"Best Method: {best_config['merge_method']}")
        print(f"Best Parameters: {best_config['parameters']}")
        print(f"Best Benchmarks: {best_config.get('benchmark_results', {})}")


if __name__ == "__main__":
    monitor_progress()
