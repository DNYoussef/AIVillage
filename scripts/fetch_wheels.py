#!/usr/bin/env python
"""Download dev wheels and update wheel manifest."""
from __future__ import annotations

import hashlib
import pathlib
import subprocess

REQ_FILE = pathlib.Path("requirements-dev.txt")
WHEEL_DIR = pathlib.Path("vendor/wheels")
MANIFEST = pathlib.Path("docs/build_artifacts/wheel-manifest.txt")


def main() -> None:
    lines = [
        l.strip()
        for l in REQ_FILE.read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    subprocess.check_call(
        [
            "pip",
            "download",
            "-d",
            str(WHEEL_DIR),
            "--only-binary=:all:",
            *lines,
        ]
    )

    manifest_lines = []
    for wheel in sorted(WHEEL_DIR.glob("*.whl")):
        with wheel.open("rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        name, version = wheel.name.split("-")[:2]
        manifest_lines.append(f"{name}=={version} sha256:{digest}")
    MANIFEST.write_text("\n".join(manifest_lines) + "\n")


if __name__ == "__main__":
    main()
