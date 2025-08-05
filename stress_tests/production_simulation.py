#!/usr/bin/env python3
"""Simple production load simulation for stability testing.

The simulation models a fleet of devices that may fail during a test window.
Users can configure the number of devices, duration of the simulation and the
probability of failure per device per second.  Aggregated stability metrics are
written to a JSON file for reproducibility.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random


@dataclass
class StabilityMetrics:
    """Results produced by :func:`run_simulation`."""

    devices: int
    duration: int
    failure_rate: float
    total_failures: int
    total_events: int
    observed_failure_rate: float
    uptime_percentage: float
    mean_time_between_failures: float


def run_simulation(devices: int, duration: int, failure_rate: float) -> StabilityMetrics:
    """Run a probabilistic failure simulation."""
    failures = 0
    events = devices * duration

    for _ in range(duration):
        for _ in range(devices):
            if random.random() < failure_rate:
                failures += 1

    observed_rate = failures / events if events else 0.0
    uptime = 1 - observed_rate
    mtbf = (events / failures) if failures else float("inf")

    return StabilityMetrics(
        devices=devices,
        duration=duration,
        failure_rate=failure_rate,
        total_failures=failures,
        total_events=events,
        observed_failure_rate=observed_rate,
        uptime_percentage=uptime,
        mean_time_between_failures=mtbf,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Production load simulation")
    parser.add_argument("--devices", type=int, default=100, help="Number of devices")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the simulation in seconds")
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=0.01,
        help="Failure probability per device per second",
    )
    parser.add_argument("--output", type=Path, default=Path("stress_test_results.json"), help="Output file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    metrics = run_simulation(args.devices, args.duration, args.failure_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    print(json.dumps(asdict(metrics), indent=2))
    print(f"\nSimulation results written to {args.output}")


if __name__ == "__main__":
    main()
