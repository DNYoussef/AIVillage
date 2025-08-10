"""Compatibility shim for the production RAG system.

The lightweight STORM implementation has moved to
:mod:`experimental.rag.storm`. This package now exposes the
production RAG modules located under ``src/production/rag/rag_system``.
"""
from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

# Start with existing search path for namespace packages
__path__ = extend_path(__path__, __name__)

# Add the production RAG system path so ``import rag_system.*`` works
_root = Path(__file__).resolve().parents[1]
_prod_path = _root / "src" / "production" / "rag" / "rag_system"
if _prod_path.exists():
    __path__.append(str(_prod_path))
