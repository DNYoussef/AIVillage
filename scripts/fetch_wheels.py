#!/usr/bin/env python
"""Validate and fetch wheel dependencies.

Usage:
  python fetch_wheels.py --validate
  python fetch_wheels.py --approve path/to/new.whl
  python fetch_wheels.py [--fetch]

With no arguments (or ``--fetch``), the script downloads wheels listed in
``requirements-dev.txt`` and regenerates the manifest. ``--validate`` checks
existing wheels against the manifest. ``--approve`` records the hash of a new
wheel in the manifest for CI use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import subprocess

REQ_FILE = pathlib.Path("requirements-dev.txt")
WHEEL_DIR = pathlib.Path("vendor/wheels")
MANIFEST = pathlib.Path("docs/build_artifacts/wheel-manifest.txt")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("wheel-guard")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text())


def save_manifest(data: dict) -> None:
    MANIFEST.write_text(json.dumps(data, indent=2))


def validate() -> bool:
    ok = True
    manifest = load_manifest()
    for wheel in WHEEL_DIR.glob("*.whl"):
        digest = sha256(wheel)
        if manifest.get(wheel.name) != digest:
            log.error("Unknown or modified wheel: %s", wheel.name)
            ok = False
    if ok:
        log.info("All wheels validated \N{CHECK MARK}")
    return ok


def approve(path: pathlib.Path) -> None:
    manifest = load_manifest()
    digest = sha256(path)
    manifest[path.name] = digest
    save_manifest(manifest)
    log.info("Approved wheel %s", path.name)


def fetch() -> None:
    """Download wheels and regenerate the manifest."""

    lines = [
        line.strip()
        for line in REQ_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
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

    manifest = {wheel.name: sha256(wheel) for wheel in WHEEL_DIR.glob("*.whl")}
    save_manifest(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--approve", type=pathlib.Path)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()

    if args.validate:
        if not validate():
            raise SystemExit(1)
        return
    if args.approve:
        approve(args.approve)
        return

    fetch()


if __name__ == "__main__":
    main()
