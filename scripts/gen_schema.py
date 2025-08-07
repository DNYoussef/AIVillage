#!/usr/bin/env python
"""Generate JSON schema for EvidencePack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from core.evidence import EvidencePack  # noqa: E402

SCHEMA_PATH = Path("schemas/evidencepack_v1.json")


def generate_schema() -> dict:
    try:
        return EvidencePack.model_json_schema()
    except AttributeError:
        return EvidencePack.schema()


def main(check: bool = False) -> None:
    schema = generate_schema()
    if check:
        if not SCHEMA_PATH.exists():
            msg = "schema file missing"
            raise SystemExit(msg)
        existing = json.loads(SCHEMA_PATH.read_text())
        if existing != schema:
            print("EvidencePack schema out of date. Run scripts/gen_schema.py")
            raise SystemExit(1)
        print("EvidencePack schema up-to-date")
        return
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"Schema written to {SCHEMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    main(args.check)
