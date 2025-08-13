#!/usr/bin/env python3
"""Baseline retrieval-augmented generation latency benchmark.

This benchmark performs deterministic lookups over a small in-memory
corpus and simulates generation time to measure average query latency
and retrieval accuracy.
"""

import argparse
import json
from pathlib import Path
import time

# Simple corpus for deterministic retrieval
CORPUS = {
    "alpha": "alpha document",
    "beta": "beta document",
    "gamma": "gamma document",
    "delta": "delta document",
    "epsilon": "epsilon document",
}

QUERIES = ["alpha", "beta", "gamma", "delta", "epsilon"]


def run_benchmark() -> dict:
    latencies: list[float] = []
    correct = 0
    for query in QUERIES:
        start = time.perf_counter()
        # deterministic dictionary lookup represents retrieval
        doc = CORPUS.get(query)
        # simulate lightweight generation step
        _generated = f"summary:{doc}"
        time.sleep(0.001)  # deterministic generation delay
        latency = time.perf_counter() - start
        latencies.append(latency)
        if doc is not None:
            correct += 1
    avg_latency_ms = sum(latencies) / len(latencies) * 1000
    accuracy_percent = correct / len(QUERIES) * 100
    return {
        "queries": len(QUERIES),
        "avg_latency_ms": round(avg_latency_ms, 3),
        "accuracy_percent": round(accuracy_percent, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple RAG latency benchmark and output JSON metrics")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/rag_latency_results.json"),
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
