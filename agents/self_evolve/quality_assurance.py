from __future__ import annotations

from dataclasses import dataclass
from typing import List
import random
import numpy as np

try:
    from agents.utils.task import Task as LangroidTask
except Exception:  # pragma: no cover - optional dependency for simple tests
    LangroidTask = object  # type: ignore


@dataclass
class SafetyCheck:
    """Represents the safety score of a single task evaluation."""

    safety_score: float


class BasicUPOChecker:
    """Minimal Uncertainty-based safety checker.

    The checker keeps a history of scores calculated as ``1 - uncertainty``.  The
    ``upo_threshold`` determines the minimum acceptable score.
    """

    def __init__(self, upo_threshold: float = 0.7) -> None:
        self.upo_threshold = upo_threshold
        self._history: List[float] = []

    def estimate_uncertainty(self, task: LangroidTask) -> float:
        """Estimate task uncertainty.

        Current implementation is a stub that returns a random value.
        """
        return random.random()

    async def check_task_safety(self, task: LangroidTask) -> bool:
        """Return ``True`` if the task is considered safe."""
        uncertainty = self.estimate_uncertainty(task)
        score = 1.0 - uncertainty
        self._history.append(score)
        if len(self._history) > 1000:
            self._history.pop(0)
        return score >= self.upo_threshold

    async def get_recent_safety_checks(self) -> List[SafetyCheck]:
        """Return the most recent safety scores."""
        return [SafetyCheck(s) for s in self._history[-5:]]

    async def evolve(self) -> None:
        """Adjust the threshold based on recent history."""
        if not self._history:
            return
        recent = self._history[-10:]
        mean = float(np.mean(recent))
        std = float(np.std(recent))
        new_threshold = mean - 1.5 * std
        self.upo_threshold = max(0.5, min(0.9, new_threshold))
