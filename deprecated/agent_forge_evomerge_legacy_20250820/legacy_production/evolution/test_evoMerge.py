"""Test evolutionary merge CLI.

Provides a tiny interface exposing a flag used to gate fitness checks in
unit tests.
"""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the evoMerge test module."""
    parser = argparse.ArgumentParser(description="Test evoMerge")
    parser.add_argument(
        "--fitness-check",
        dest="fitness_check",
        action="store_true",
        help="Run a mock fitness check",
    )
    return parser


def main(args: list[str] | None = None) -> bool:
    """Parse arguments and report whether fitness check is requested."""
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    fitness = 0.9 if parsed.fitness_check else 0.0
    report = {
        "fitness_check": bool(parsed.fitness_check),
        "fitness": fitness,
    }
    with open("evolution_fitness.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return bool(parsed.fitness_check)


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    print(main())
