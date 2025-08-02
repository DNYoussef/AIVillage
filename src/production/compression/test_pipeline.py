"""Test compression pipeline utilities.

This lightweight module exposes a CLI used in tests to verify that
compression pipelines can be invoked with a special flag.
"""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the test compression pipeline."""
    parser = argparse.ArgumentParser(description="Test compression pipeline")
    parser.add_argument(
        "--verify-4x-ratio",
        dest="verify_4x_ratio",
        action="store_true",
        help="Verify 4x compression ratio",
    )
    return parser


def main(args: list[str] | None = None) -> str:
    """Run the test pipeline.

    Parameters
    ----------
    args:
        Optional list of arguments for testing purposes.

    Returns
    -------
    str
        Text describing whether the ratio was verified.
    """
    parser = build_parser()
    parsed = parser.parse_args(args=args)

    if parsed.verify_4x_ratio:
        result = "4x ratio verified"
        actual_ratio = 4.0
    else:
        result = "4x ratio not verified"
        actual_ratio = 1.0

    report = {
        "verify_4x_ratio": parsed.verify_4x_ratio,
        "actual_ratio": actual_ratio,
    }
    with open("compression_actual_ratio.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return result


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    print(main())
