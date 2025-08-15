#!/usr/bin/env python3
"""Utility to list stub code patterns across the repository.

Searches Python files for common stub indicators like ``pass`` bodies,
``raise NotImplementedError`` calls and TODO markers. The script prints
matches in ``<path>:<line>: <pattern>`` format to stdout.
"""
from __future__ import annotations

import argparse
import pathlib
import re
from typing import Iterable

# Patterns to search for and a human readable label
PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*def .*:\s*pass$"), "def-pass"),
    (re.compile(r"^\s*class .*:\s*pass$"), "class-pass"),
    (re.compile(r"raise NotImplementedError"), "raise NotImplementedError"),
    (re.compile(r"# TODO"), "TODO"),
    (re.compile(r"return None\s+# TODO"), "return None TODO"),
]


def iter_python_files(paths: Iterable[str]) -> Iterable[pathlib.Path]:
    """Yield Python files under ``paths``."""
    this_file = pathlib.Path(__file__).resolve()
    for root in paths:
        for path in pathlib.Path(root).rglob("*.py"):
            if path.is_file() and path.resolve() != this_file:
                yield path


def scan_file(path: pathlib.Path) -> Iterable[tuple[int, str, str]]:
    """Scan ``path`` for stub patterns, yielding line number, label and line."""
    try:
        lines = path.read_text().splitlines()
    except Exception:
        return []

    previous = ""
    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "pass" and previous.lstrip().startswith(("def ", "class ")):
            label = "pass-body"
            yield lineno, label, line.rstrip()
            previous = line
            continue

        for pattern, label in PATTERNS:
            if pattern.search(line):
                yield lineno, label, line.rstrip()
                break
        previous = line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["."], help="Paths to scan")
    args = parser.parse_args(argv)

    for file in iter_python_files(args.paths):
        for lineno, label, line in scan_file(file):
            print(f"{file}:{lineno}: {label}: {line.strip()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
