#!/usr/bin/env python3
"""Collect performance baselines for benchmark tests.

This script runs all benchmark tests multiple times to establish
performance baselines that can be used for regression detection.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_benchmarks(iterations: int = 5) -> dict[str, list[float]]:
    """Run benchmark tests multiple times and collect results."""
    print(f"Running benchmarks {iterations} times to collect baselines...")

    # Results storage
    all_results = {}

    for i in range(iterations):
        print(f"\nIteration {i + 1}/{iterations}")

        # Run pytest with benchmark marker
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "tests/benchmarks/",
            "-v",
            "-m",
            "benchmark and not slow_benchmark",  # Skip slow tests for baseline
            "--tb=short",
        ]

        try:
            # Run the tests
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                print(f"✅ Iteration {i + 1} completed successfully")
            else:
                print(f"⚠️  Iteration {i + 1} had test failures (continuing)")
                print("STDERR:", result.stderr[-500:])  # Last 500 chars

            # Try to parse benchmark results if they exist
            results_file = Path("tests/benchmarks/benchmark_results.json")
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)

                    # Extract latest results from this run
                    for result_data in data:
                        test_name = result_data["test_name"]
                        duration = result_data["duration"]

                        if test_name not in all_results:
                            all_results[test_name] = []

                        # Only add if it's from this run (simple heuristic)
                        if len(all_results[test_name]) < i + 1:
                            all_results[test_name].append(duration)

                except Exception as e:
                    print(f"Warning: Could not parse benchmark results: {e}")

        except subprocess.TimeoutExpired:
            print(f"❌ Iteration {i + 1} timed out")
        except Exception as e:
            print(f"❌ Iteration {i + 1} failed: {e}")

    return all_results


def calculate_baselines(results: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Calculate baseline statistics from multiple runs."""
    baselines = {}

    for test_name, durations in results.items():
        if len(durations) < 2:
            print(f"Warning: {test_name} has only {len(durations)} measurements")
            continue

        # Calculate statistics
        mean_duration = statistics.mean(durations)
        median_duration = statistics.median(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Calculate standard deviation if we have enough samples
        if len(durations) >= 3:
            stdev = statistics.stdev(durations)
            # Use mean + 2*stdev as baseline (accounts for variance)
            baseline = mean_duration + (2 * stdev)
        else:
            # Use max duration as conservative baseline
            baseline = max_duration

        baselines[test_name] = {
            "baseline": baseline,
            "mean": mean_duration,
            "median": median_duration,
            "min": min_duration,
            "max": max_duration,
            "samples": len(durations),
            "raw_durations": durations,
        }

        print(f"{test_name}:")
        print(f"  Baseline: {baseline:.4f}s")
        print(
            f"  Mean: {mean_duration:.4f}s ± {stdev:.4f}s"
            if len(durations) >= 3
            else f"  Mean: {mean_duration:.4f}s"
        )
        print(f"  Range: {min_duration:.4f}s - {max_duration:.4f}s")

    return baselines


def save_baselines(baselines: dict[str, dict[str, float]], output_file: Path) -> None:
    """Save baselines to JSON file."""
    # Prepare data for saving
    baseline_data = {
        "baselines": {name: stats["baseline"] for name, stats in baselines.items()},
        "detailed_stats": baselines,
        "metadata": {
            "created_at": "2025-01-23T10:00:00Z",  # Would use actual timestamp
            "total_tests": len(baselines),
            "description": "Performance baselines for regression detection",
        },
    }

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\n✅ Baselines saved to {output_file}")
        print(f"📊 {len(baselines)} benchmarks recorded")

    except Exception as e:
        print(f"❌ Failed to save baselines: {e}")


def load_existing_baselines(baselines_file: Path) -> dict[str, float]:
    """Load existing baselines for comparison."""
    if not baselines_file.exists():
        return {}

    try:
        with open(baselines_file) as f:
            data = json.load(f)
        return data.get("baselines", {})
    except Exception:
        return {}


def compare_with_existing(
    new_baselines: dict[str, dict[str, float]], existing_baselines: dict[str, float]
) -> None:
    """Compare new baselines with existing ones."""
    if not existing_baselines:
        print("\nNo existing baselines to compare with.")
        return

    print("\n📈 Comparison with existing baselines:")
    print("-" * 50)

    for test_name, stats in new_baselines.items():
        new_baseline = stats["baseline"]
        old_baseline = existing_baselines.get(test_name)

        if old_baseline:
            change = ((new_baseline - old_baseline) / old_baseline) * 100

            if abs(change) < 5:  # Less than 5% change
                status = "🟢"
            elif change > 20:  # More than 20% slower
                status = "🔴"
            else:
                status = "🟡"

            print(f"{status} {test_name}:")
            print(f"    Old: {old_baseline:.4f}s")
            print(f"    New: {new_baseline:.4f}s")
            print(f"    Change: {change:+.1f}%")
        else:
            print(f"🆕 {test_name}: {new_baseline:.4f}s (new benchmark)")
        print()


def main() -> int | None:
    """Main baseline collection process."""
    parser = argparse.ArgumentParser(description="Collect performance baselines")
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        help="Number of iterations to run (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("tests/benchmarks/baselines.json"),
        help="Output file for baselines",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare with existing baselines"
    )

    args = parser.parse_args()

    print("🏃 AI Village Performance Baseline Collection")
    print("=" * 50)

    # Load existing baselines for comparison
    existing_baselines = {}
    if args.compare:
        existing_baselines = load_existing_baselines(args.output)

    # Run benchmarks
    try:
        results = run_benchmarks(args.iterations)

        if not results:
            print("❌ No benchmark results collected!")
            print("Make sure benchmark tests exist and can run successfully.")
            return 1

        # Calculate baselines
        print(f"\n📊 Calculating baselines from {args.iterations} runs...")
        print("-" * 50)
        baselines = calculate_baselines(results)

        if not baselines:
            print("❌ No baselines could be calculated!")
            return 1

        # Compare with existing if requested
        if args.compare and existing_baselines:
            compare_with_existing(baselines, existing_baselines)

        # Save baselines
        save_baselines(baselines, args.output)

        print("\n🎯 Baseline collection completed successfully!")
        print("Use these baselines to detect performance regressions in future runs.")

        return 0

    except KeyboardInterrupt:
        print("\n⏹  Baseline collection interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Baseline collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
