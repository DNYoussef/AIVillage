#!/usr/bin/env python3
"""Aggregate all sprint benchmarks and output performance comparison.
This script runs the creativity, personalization, and repair test benchmarks
and saves their aggregated metrics to ``performance_comparison.json``.
"""

import argparse
import asyncio
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))


async def run_all(output_file: Path) -> dict:
    """Run all benchmarks and write aggregated metrics to *output_file*.

    Parameters
    ----------
    output_file:
        Path to the JSON file where aggregated metrics will be stored.
    """
    from benchmarks.hyperag_creativity import CreativityBenchmark
    from benchmarks.hyperag_personalization import PersonalizationBenchmark
    from benchmarks.hyperag_repair_test_suite import RepairTestSuite

    results: dict[str, object] = {}

    # Creativity benchmark
    creativity = CreativityBenchmark()
    creativity_metrics = await creativity.run_full_benchmark()
    results["creativity"] = asdict(creativity_metrics)

    # Personalization benchmark
    personalization = PersonalizationBenchmark()
    personalization_results = await personalization.run_full_benchmark()
    results["personalization"] = {name: asdict(metrics) for name, metrics in personalization_results.items()}

    # Repair test suite
    repair_suite = RepairTestSuite()
    repair_metrics = await repair_suite.run_comprehensive_repair_tests()
    results["repair"] = asdict(repair_metrics)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run all sprint benchmarks and aggregate results")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("performance_comparison.json"),
        help="File to write aggregated benchmark metrics",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    results = await run_all(args.output)
    print(json.dumps(results, indent=2))
    print(f"\nPerformance comparison written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
