"""Simple conformal calibrator placeholder."""

from __future__ import annotations

from dataclasses import dataclass


def _clip(p: float) -> float:
    return max(0.0, min(1.0, p))


@dataclass
class ConformalCalibrator:
    scale: float = 0.8
    bias: float = 0.1

    def calibrate(self, prob: float) -> float:
        """Apply affine calibration with clipping."""
        return _clip(self.scale * prob + self.bias)

    @classmethod
    def load_default(cls) -> ConformalCalibrator:
        """Load default model (placeholder)."""
        return cls()
