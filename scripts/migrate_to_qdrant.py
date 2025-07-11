"""Utility to migrate existing vectors to Qdrant."""
from __future__ import annotations

import json
from pathlib import Path


def migrate(source: Path, dest_url: str) -> None:
    """Mock migration that prints actions."""
    data = json.loads(source.read_text())
    print(f"Uploading {len(data)} vectors to {dest_url}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest_url")
    args = parser.parse_args()
    migrate(args.source, args.dest_url)
