#!/usr/bin/env python3
"""Monitor test performance and track regressions."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class PerformanceMonitor:
    def __init__(self) -> None:
        self.results_file = Path("test_performance_history.json")
        self.history = self._load_history()

    def _load_history(self) -> dict[str, list]:
        """Load performance history."""
        if self.results_file.exists():
            try:
                return json.loads(self.results_file.read_text())
            except json.JSONDecodeError:
                return {"runs": []}
        return {"runs": []}

    def run_benchmarks(self) -> None:
        """Run performance benchmarks."""
        print("Running performance benchmarks...")

        start_time = time.time()

        # Run pytest benchmarks
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--benchmark-only",
                "--benchmark-json=benchmark.json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        elapsed = time.time() - start_time

        # Parse results
        if Path("benchmark.json").exists():
            with open("benchmark.json") as f:
                benchmark_data = json.load(f)

            # Extract key metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": elapsed,
                "benchmarks": {},
            }

            for bench in benchmark_data.get("benchmarks", []):
                metrics["benchmarks"][bench["name"]] = {
                    "mean": bench["stats"]["mean"],
                    "stddev": bench["stats"]["stddev"],
                    "min": bench["stats"]["min"],
                    "max": bench["stats"]["max"],
                }

            # Add to history
            self.history["runs"].append(metrics)

            # Keep only last 50 runs
            self.history["runs"] = self.history["runs"][-50:]

            # Save history
            self.results_file.write_text(json.dumps(self.history, indent=2))

            print(f"[OK] Benchmarks completed in {elapsed:.2f}s")
            self._check_regressions(metrics)
        else:
            print("[WARN] No benchmark results found")

    def _check_regressions(self, current_metrics: dict) -> None:
        """Check for performance regressions."""
        if len(self.history["runs"]) < 2:
            return

        # Compare with previous run
        previous = self.history["runs"][-2]["benchmarks"]
        current = current_metrics["benchmarks"]

        regressions = []

        for name in current:
            if name in previous:
                prev_mean = previous[name]["mean"]
                curr_mean = current[name]["mean"]

                # Check for 20% regression
                if curr_mean > prev_mean * 1.2:
                    regressions.append(
                        {
                            "name": name,
                            "previous": prev_mean,
                            "current": curr_mean,
                            "regression": (curr_mean - prev_mean) / prev_mean * 100,
                        }
                    )

        if regressions:
            print("\n[WARN] Performance Regressions Detected!")
            for reg in regressions:
                print(f"  - {reg['name']}: {reg['regression']:.1f}% slower")
                print(f"    Previous: {reg['previous']:.4f}s")
                print(f"    Current: {reg['current']:.4f}s")
        else:
            print("\n[OK] No performance regressions detected")

    def run_test_performance_summary(self) -> dict:
        """Run test performance summary without benchmarking."""
        print("Running test performance summary...")

        start_time = time.time()

        # Run our working core tests
        test_results = {}

        test_categories = [
            ("core_communication", "tests/core/test_communication.py"),
            ("core_evidencepack", "tests/core/test_evidencepack.py"),
            ("message_tests", "tests/test_message.py"),
            (
                "compression_basic",
                "tests/compression/test_compression_comprehensive.py::TestCompressionPipeline::test_seedlm_compression_basic",
            ),
            (
                "evolution_basic",
                "tests/evolution/test_evolution_comprehensive.py::TestEvolutionaryTournament::test_tournament_basic_selection",
            ),
        ]

        for category, test_path in test_categories:
            category_start = time.time()

            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=no"],
                check=False,
                capture_output=True,
                text=True,
            )

            category_time = time.time() - category_start

            # Parse test results
            passed = result.stdout.count(" PASSED")
            failed = result.stdout.count(" FAILED")
            errors = result.stdout.count(" ERROR")

            test_results[category] = {
                "execution_time": category_time,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": (
                    (passed / (passed + failed + errors))
                    if (passed + failed + errors) > 0
                    else 0
                ),
            }

        total_time = time.time() - start_time

        # Generate performance summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": total_time,
            "test_categories": test_results,
            "overall_metrics": {
                "total_passed": sum(r["passed"] for r in test_results.values()),
                "total_failed": sum(r["failed"] for r in test_results.values()),
                "total_errors": sum(r["errors"] for r in test_results.values()),
                "avg_execution_time": sum(
                    r["execution_time"] for r in test_results.values()
                )
                / len(test_results),
            },
        }

        # Save performance summary
        summary_file = Path("test_performance_summary.json")
        summary_file.write_text(json.dumps(summary, indent=2))

        self._print_performance_summary(summary)

        return summary

    def _print_performance_summary(self, summary: dict) -> None:
        """Print formatted performance summary."""
        print(f"\n{'=' * 60}")
        print("TEST PERFORMANCE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        print()

        print("Test Category Performance:")
        print("-" * 40)
        for category, results in summary["test_categories"].items():
            success_rate = results["success_rate"] * 100
            print(
                f"{category:20} | {results['execution_time']:6.2f}s | {
                    results['passed']:2d}P {results['failed']:2d}F {
                    results['errors']:2d}E | {success_rate:5.1f}%"
            )

        print()
        overall = summary["overall_metrics"]
        print("Overall Metrics:")
        print("-" * 40)
        print(
            f"Total Tests: {overall['total_passed'] + overall['total_failed'] + overall['total_errors']}"
        )
        print(f"Passed: {overall['total_passed']}")
        print(f"Failed: {overall['total_failed']}")
        print(f"Errors: {overall['total_errors']}")
        print(f"Average Category Time: {overall['avg_execution_time']:.2f}s")

        overall_success = (
            overall["total_passed"]
            / (
                overall["total_passed"]
                + overall["total_failed"]
                + overall["total_errors"]
            )
            * 100
        )
        print(f"Overall Success Rate: {overall_success:.1f}%")

        print(f"{'=' * 60}")


if __name__ == "__main__":
    monitor = PerformanceMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        monitor.run_benchmarks()
    else:
        monitor.run_test_performance_summary()
