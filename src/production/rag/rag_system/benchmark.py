"""Benchmarking utilities for the production RAG vector store.

This script measures latency and basic accuracy for the VectorStore search
across the FAISS and Qdrant backends and writes the results to a CSV file.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
import time

import numpy as np

from .vector_store import VectorStore


@dataclass
class BenchmarkResult:
    """Holds benchmark metrics for a single backend."""

    backend: str
    avg_ms: float
    accuracy: float


def _run_backend(
    use_qdrant: bool,
    docs: np.ndarray,
    ids: list[str],
    payload: list[dict[str, int]],
    queries: np.ndarray,
) -> BenchmarkResult:
    """Benchmark ``VectorStore.search`` for a given backend."""
    os.environ["RAG_USE_QDRANT"] = "1" if use_qdrant else "0"
    store = VectorStore()
    if use_qdrant and store.backend is store.faiss:
        raise RuntimeError("Qdrant backend unavailable")
    store.add(ids, docs, payload)

    durations: list[float] = []
    correct = 0
    for idx, q in enumerate(queries):
        started = time.perf_counter()
        res = store.search(q, k=1)
        durations.append((time.perf_counter() - started) * 1000)
        if res and res[0]["id"] == ids[idx]:
            correct += 1

    avg_ms = float(np.mean(durations)) if durations else 0.0
    accuracy = correct / len(queries) if queries.size else 0.0
    backend_name = "qdrant" if use_qdrant else "faiss"
    return BenchmarkResult(backend_name, avg_ms, accuracy)


def run_benchmark(
    n_docs: int = 100,
    n_queries: int = 10,
    dim: int = 768,
    output: str = "vector_search_metrics.csv",
) -> list[BenchmarkResult]:
    """Run benchmark comparing FAISS and Qdrant backends."""
    rng = np.random.default_rng(0)
    docs = rng.random((n_docs, dim), dtype=np.float32)
    ids = [str(i) for i in range(n_docs)]
    payload = [{"idx": i} for i in range(n_docs)]
    queries = docs[:n_queries] + 0.01 * rng.random((n_queries, dim), dtype=np.float32)

    results: list[BenchmarkResult] = []
    results.append(_run_backend(False, docs, ids, payload, queries))
    try:
        results.append(_run_backend(True, docs, ids, payload, queries))
    except Exception as exc:  # pragma: no cover - network
        print(f"Skipping Qdrant benchmark: {exc}")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "avg_ms", "accuracy"])
        for r in results:
            writer.writerow([r.backend, f"{r.avg_ms:.3f}", f"{r.accuracy:.3f}"])

    return results


if __name__ == "__main__":
    run_benchmark()
