"""Very small benchmarking helpers used in tests.

The real Agent Forge pipeline performs extensive benchmarking using W&B and a
large evaluation suite.  For the purposes of the open-source subset we
implement a light-weight placeholder benchmark that simply measures how long it
would take to load a model directory.  The function returns a dictionary of
metrics so that callers have a predictable structure to work with.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def benchmark_model(model_path: Path) -> dict[str, Any]:
    """Run a trivial benchmark on ``model_path``.

    The benchmark simply measures how long it takes to traverse the directory
    tree.  This keeps the implementation light-weight while providing a
    deterministic metric for demonstrations and tests.
    """

    start = time.perf_counter()
    total_files = 0
    total_bytes = 0
    for p in model_path.rglob("*"):
        if p.is_file():
            total_files += 1
            total_bytes += p.stat().st_size
    elapsed = time.perf_counter() - start

    return {
        "files": total_files,
        "bytes": total_bytes,
        "elapsed_sec": round(elapsed, 4),
    }
