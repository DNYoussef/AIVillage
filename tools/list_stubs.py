#!/usr/bin/env python3
"""List common stub patterns in the repository."""
from __future__ import annotations

import argparse
import pathlib
import re
from typing import Iterable

PATTERNS: dict[str, re.Pattern[str]] = {
    "def-pass": re.compile(r"^\s*def .+:\s*pass\s*$"),
    "class-pass": re.compile(r"^\s*class .+:\s*pass\s*$"),
    "not-implemented": re.compile(r"raise NotImplementedError"),
    "todo": re.compile(r"#\s*TODO", re.IGNORECASE),
    "return-none-todo": re.compile(r"return None\s*#\s*TODO", re.IGNORECASE),
}


def iter_python_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in root.rglob("*.py"):
        yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".", help="root directory")
    args = parser.parse_args()
    root = pathlib.Path(args.path)

    total = 0
    for file in iter_python_files(root):
        try:
            lines = file.read_text().splitlines()
        except Exception:  # pragma: no cover - unreadable file
            continue
        for lineno, line in enumerate(lines, 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    print(f"{file}:{lineno}:{line.strip()}")
                    total += 1
    print(f"Total matches: {total}")


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
