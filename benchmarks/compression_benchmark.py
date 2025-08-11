#!/usr/bin/env python3
"""Reproducible compression benchmark.

This script compresses deterministic sample data and reports
compression ratio, percentage reduction, and time taken.
"""

import argparse
import json
from pathlib import Path
import time
import zlib


def run_benchmark() -> dict:
    """Run compression on sample data and return metrics."""
    data = ("AIVillage benchmark data." * 1024).encode("utf-8")
    start = time.perf_counter()
    compressed = zlib.compress(data)
    elapsed = time.perf_counter() - start

    original_size = len(data)
    compressed_size = len(compressed)
    ratio = compressed_size / original_size
    reduction_percent = (1 - ratio) * 100

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": round(ratio, 3),
        "reduction_percent": round(reduction_percent, 1),
        "time_seconds": round(elapsed, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple compression benchmark and output JSON metrics")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/compression_results.json"),
        help="File to write benchmark results",
    )
    args = parser.parse_args()

    results = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
