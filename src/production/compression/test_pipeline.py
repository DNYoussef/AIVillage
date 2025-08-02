"""Test compression pipeline utilities.

This lightweight module exposes a CLI used in tests to verify that
compression pipelines can be invoked with a special flag.
"""

from __future__ import annotations

import argparse


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
        return "4x ratio verified"
    return "4x ratio not verified"


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    print(main())
