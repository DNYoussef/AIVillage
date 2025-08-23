"""Minimal chat runner for the digital twin.

The runner exposes a :func:`chat` generator that first consults the safety
guard before emitting any tokens from a local language model adapter.  The
adapter used here is intentionally tiny – it simply echoes the prompt back to
the caller one token at a time – but the structure mirrors the real system
where more capable models or tools (such as HyperRAG) could be invoked.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from . import guard


def risk_estimator(prompt: str) -> float:
    """Very small heuristic risk estimator.

    Longer prompts are considered slightly riskier but the score is capped at
    ``1.0`` to mimic a probability.
    """

    return min(len(prompt) / 1000.0, 1.0)


def _local_llm(prompt: str) -> Iterator[str]:
    """Simple echo model used for testing.

    Yields the original prompt token by token so that callers can stream the
    response.
    """

    for token in prompt.split():
        yield token + " "


def _merge_domain_lora(prompt: str) -> str:
    """Placeholder for domain LoRA merging.

    The real system would combine the base model with domain-specific LoRA
    weights.  Our tests only require that the function exists and returns the
    prompt unchanged.
    """

    return prompt


def chat(prompt: str, **kw: Any) -> Iterable[str]:
    """Stream a response for ``prompt`` respecting guard decisions."""

    risk = risk_estimator(prompt)
    decision = guard.risk_gate({"content": prompt}, risk)
    if decision != "allow":
        yield f"[{decision}]"
        return

    prompt = _merge_domain_lora(prompt)
    yield from _local_llm(prompt)


__all__ = ["chat", "risk_estimator"]
