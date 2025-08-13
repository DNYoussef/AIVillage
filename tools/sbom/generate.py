#!/usr/bin/env python3
"""Generate a minimal CycloneDX SBOM for a binary."""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: generate.py <binary> <output>", file=sys.stderr)
        return 1
    binary_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])
    data = binary_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    bom: dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.3",
        "version": 1,
        "components": [
            {
                "type": "file",
                "name": binary_path.name,
                "hashes": [{"alg": "SHA-256", "content": sha256}],
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bom, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
