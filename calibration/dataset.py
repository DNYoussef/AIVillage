from __future__ import annotations

"""Dataset utilities for confidence calibration."""

from dataclasses import dataclass
from typing import List


@dataclass
class CalibrationDataset:
    scores: List[float]
    labels: List[int]

    @classmethod
    def load_sample(cls) -> "CalibrationDataset":
        return cls(scores=[0.2, 0.6, 0.9, 0.4, 0.8], labels=[0, 1, 1, 0, 1])
