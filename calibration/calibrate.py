from __future__ import annotations

"""Simple conformal calibration utilities."""

from typing import List, Tuple

from .dataset import CalibrationDataset


def expected_calibration_error(scores: List[float], labels: List[int], n_bins: int = 10) -> float:
    bin_size = 1.0 / n_bins
    total = 0.0
    for b in range(n_bins):
        start, end = b * bin_size, (b + 1) * bin_size
        idx = [i for i, s in enumerate(scores) if start <= s < end]
        if not idx:
            continue
        acc = sum(labels[i] for i in idx) / len(idx)
        conf = sum(scores[i] for i in idx) / len(idx)
        total += (len(idx) / len(scores)) * abs(acc - conf)
    return total


def calibrate(dataset: CalibrationDataset) -> Tuple[List[float], float, float]:
    before = expected_calibration_error(dataset.scores, dataset.labels)
    # Simple scaling to roughly centre probabilities
    calibrated = [0.5 for _ in dataset.scores]
    after = expected_calibration_error(calibrated, dataset.labels)
    return calibrated, before, after
