"""Test retrieval-augmented generation pipeline.

This module offers a simple CLI that toggles a mock query performance
check. It is intentionally minimal and is used only by unit tests.
"""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the RAG test pipeline."""
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument(
        "--query-performance",
        dest="query_performance",
        action="store_true",
        help="Evaluate query performance",
    )
    return parser


def main(args: list[str] | None = None) -> str:
    """Run the RAG test pipeline."""
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    if parsed.query_performance:
        result = "performance evaluation complete"
        latency_ms = 42.0
    else:
        result = "performance evaluation skipped"
        latency_ms = 0.0

    report = {
        "query_performance": parsed.query_performance,
        "latency_ms": latency_ms,
    }
    with open("rag_performance.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return result


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    print(main())
