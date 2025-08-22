"""
Compatibility wrapper for Agent Forge evaluator.

This module provides backward compatibility for code that imports from the old location.
All functionality has been moved to packages.agent_forge.evaluation.
"""

# Import from the new canonical location
from packages.agent_forge.evaluation import (
    default_evaluator,
    evaluate_hrrm_model,
    evaluate_model,
    evaluate_thought_quality,
)

# Create module-level instance for backward compatibility
_impl = default_evaluator

# re-export helper functions so call sites can import from this location

# Allow tests or callers to monkeypatch evaluate_thought_quality on this module
# and ensure the underlying implementation uses the patched version.


# Direct re-exports from the new location
evaluate_thought_quality = evaluate_thought_quality
evaluate_model = evaluate_model
evaluate_hrrm_model = evaluate_hrrm_model


# expose all other helpers directly
# Define each function explicitly to satisfy F822 checks
def measure_coherence(*args, **kwargs):
    return getattr(_impl, "measure_coherence")(*args, **kwargs)


def measure_relevance(*args, **kwargs):
    return getattr(_impl, "measure_relevance")(*args, **kwargs)


def evaluate_perplexity(*args, **kwargs):
    return getattr(_impl, "evaluate_perplexity")(*args, **kwargs)


def evaluate_coding(*args, **kwargs):
    return getattr(_impl, "evaluate_coding")(*args, **kwargs)


def evaluate_mathematics(*args, **kwargs):
    return getattr(_impl, "evaluate_mathematics")(*args, **kwargs)


def evaluate_writing(*args, **kwargs):
    return getattr(_impl, "evaluate_writing")(*args, **kwargs)


def evaluate_zero_shot_classification(*args, **kwargs):
    return getattr(_impl, "evaluate_zero_shot_classification")(*args, **kwargs)


def evaluate_zero_shot_qa(*args, **kwargs):
    return getattr(_impl, "evaluate_zero_shot_qa")(*args, **kwargs)


def evaluate_story_coherence(*args, **kwargs):
    return getattr(_impl, "evaluate_story_coherence")(*args, **kwargs)


def calculate_overall_score(*args, **kwargs):
    return getattr(_impl, "calculate_overall_score")(*args, **kwargs)


def parallel_evaluate_models(*args, **kwargs):
    return getattr(_impl, "parallel_evaluate_models")(*args, **kwargs)


__all__ = [
    "evaluate_model",
    "evaluate_thought_quality",
    "measure_coherence",
    "measure_relevance",
    "evaluate_perplexity",
    "evaluate_coding",
    "evaluate_mathematics",
    "evaluate_writing",
    "evaluate_zero_shot_classification",
    "evaluate_zero_shot_qa",
    "evaluate_story_coherence",
    "calculate_overall_score",
    "parallel_evaluate_models",
]
