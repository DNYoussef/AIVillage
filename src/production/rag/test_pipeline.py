"""Test retrieval-augmented generation pipeline.

This module offers a simple CLI that toggles a mock query performance
check. It is intentionally minimal and is used only by unit tests.
"""

from __future__ import annotations

import argparse


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
        return "performance evaluation complete"
    return "performance evaluation skipped"


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    print(main())
