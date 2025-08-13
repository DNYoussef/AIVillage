"""Minimal tokenizer utility for tests.

Provides a ``get_cl100k_encoding`` function that returns an object with an
``encode`` method.  This is a lightweight stand-in for the real tokenizer used
in production, which is not required for the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _DummyTokenizer:
    """Simple tokenizer returning byte lengths.

    The real implementation relies on ``tiktoken``; however, pulling in that
    dependency is unnecessary for the scope of the tests.  This dummy version
    mimics the interface by exposing an ``encode`` method.
    """

    def encode(self, text: str) -> list[int]:  # pragma: no cover - trivial
        return list(text.encode("utf-8"))


def get_cl100k_encoding() -> _DummyTokenizer:
    """Return a dummy tokenizer used in tests."""
    return _DummyTokenizer()
