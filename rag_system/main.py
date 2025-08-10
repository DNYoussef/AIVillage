"""Command-line wrapper for the production RAG pipeline.

This module delegates to ``src.production.rag.rag_system.main`` so the
production RAG pipeline can be invoked from the repository root using
``python rag_system/main.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Compute important paths
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROD_RAG = SRC / "production" / "rag"

# Replace the script's directory on ``sys.path`` with the production RAG path so
# ``import rag_system`` resolves to the production package rather than this
# wrapper package.
sys.path[0] = str(PROD_RAG)

# Ensure additional paths are available for subsequent imports.
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

# Remove any previously loaded stub package
sys.modules.pop("rag_system", None)

from src.production.rag.rag_system.main import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
