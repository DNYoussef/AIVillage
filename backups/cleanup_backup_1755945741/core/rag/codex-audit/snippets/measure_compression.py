"""Utility script used by audits to measure compression ratios.

The script leverages :func:`src.production.compression.report.report_compression`
to gather headline compression metrics for the three supported algorithms:
BitNet, SeedLM and VPTQ.  Results are printed and also written to a JSON file
for downstream consumption.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on the path when executed as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.production.compression import report_compression


def main(output_path: str = "compression_report.json") -> None:
    results = {}
    for method in ("bitnet", "seedlm", "vptq"):
        metrics = report_compression(method)
        print(f"{method}: {metrics['ratio']:.2f}x")
        results[method] = metrics

    path = Path(output_path)
    path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":  # pragma: no cover - simple CLI wrapper
    main()
