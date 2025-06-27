from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class DPOConfig:
    """Hyperparameters for Direct Preference Optimization."""

    beta: float = 0.1


class DirectPreferenceOptimizer:
    """Lightweight preference optimizer selecting the highest score."""

    def __init__(self, config: DPOConfig | None = None) -> None:
        self.config = config or DPOConfig()

    def select(self, preferences: Dict[Any, float]) -> Any:
        if not preferences:
            return None
        return max(preferences, key=preferences.get)
