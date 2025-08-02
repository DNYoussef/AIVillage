#!/usr/bin/env python3
"""Utility for verifying and counting cleanup actions.

This script reads a log file containing cleanup actions (one per line),
logs each action, counts the total number of actions, and records the
count in a machine-readable JSON metrics file.

Usage:
    python tools/verify_cleanup.py [path_to_log] [--metrics output_json]

The default log file is ``cleanup.log`` and the default metrics file is
``cleanup_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify cleanup actions and record metrics.")
    parser.add_argument(
        "logfile",
        nargs="?",
        default="cleanup.log",
        help="Path to a file listing cleanup actions (one per line).",
    )
    parser.add_argument(
        "--metrics",
        default="cleanup_metrics.json",
        help="Destination JSON file for writing cleanup metrics.",
    )
    return parser.parse_args()


def iter_actions(log_path: Path) -> Iterable[str]:
    """Yield cleanup actions from ``log_path``.

    Empty lines are ignored. The function is separated for easier testing.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            action = line.strip()
            if action:
                yield action


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log_path = Path(args.logfile)
    actions = list(iter_actions(log_path))

    for action in actions:
        logging.info("cleanup action: %s", action)

    metrics = {"cleanup_actions": len(actions)}
    metrics_path = Path(args.metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Recorded %s cleanup actions to %s", len(actions), metrics_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())
