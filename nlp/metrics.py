import numpy as np


def measure_coherence(thoughts: np.ndarray) -> float:
    """Compute average cosine similarity between consecutive thought vectors."""
    thoughts = np.asarray(thoughts)
    if thoughts.ndim != 3:
        raise ValueError("Thoughts array must be 3-dimensional (batch, seq, dim)")
    if thoughts.shape[1] < 2:
        return 1.0
    a = thoughts[:, :-1, :]
    b = thoughts[:, 1:, :]
    sims = np.sum(a * b, axis=-1) / (
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    )
    return float(np.mean(sims))


def measure_relevance(thoughts: np.ndarray, targets: np.ndarray) -> float:
    """Compute cosine similarity between averaged thoughts and targets."""
    thoughts = np.asarray(thoughts)
    targets = np.asarray(targets)
    if thoughts.ndim == 3:
        thoughts = thoughts.mean(axis=1)
    if targets.ndim == 3:
        targets = targets.mean(axis=1)
    sim = np.sum(thoughts * targets, axis=-1) / (
        np.linalg.norm(thoughts, axis=-1) * np.linalg.norm(targets, axis=-1) + 1e-8
    )
    return float(np.mean(sim))
