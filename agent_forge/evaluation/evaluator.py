from src.agent_forge.evaluation import evaluator as _impl

# re-export helper functions so call sites can import from this location

# Allow tests or callers to monkeypatch evaluate_thought_quality on this module
# and ensure the underlying implementation uses the patched version.

def evaluate_thought_quality(model, eval_data):
    return _impl.evaluate_thought_quality(model, eval_data)


def evaluate_model(model_or_path, eval_data=None):
    original = _impl.evaluate_thought_quality
    try:
        _impl.evaluate_thought_quality = evaluate_thought_quality
        return _impl.evaluate_model(model_or_path, eval_data)
    finally:
        _impl.evaluate_thought_quality = original

# expose all other helpers directly
for _name in [
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
]:
    globals()[_name] = getattr(_impl, _name)

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
