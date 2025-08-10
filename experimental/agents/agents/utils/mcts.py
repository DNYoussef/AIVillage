from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class MCTSConfig:
    """Hyperparameters for Monte Carlo Tree Search."""

    iterations: int = 10
    exploration_weight: float = 1.0
    simulation_depth: int = 10


class MonteCarloTreeSearch:
    """Very small MCTS utility for discrete option selection."""

    def __init__(self, config: MCTSConfig | None = None) -> None:
        self.config = config or MCTSConfig()

    def search(self, options: list[Any], simulate: Callable[[Any], float]) -> Any:
        """Return the best option according to simulated rewards."""
        stats: dict[Any, dict[str, float]] = {
            opt: {"visits": 0, "value": 0.0} for opt in options
        }
        for _ in range(self.config.iterations):
            option = self._select_option(options, stats)
            reward = simulate(option)
            entry = stats[option]
            entry["visits"] += 1
            entry["value"] += reward
        best = max(
            options, key=lambda o: stats[o]["value"] / max(1, stats[o]["visits"])
        )
        return best

    def _select_option(
        self, options: list[Any], stats: dict[Any, dict[str, float]]
    ) -> Any:
        for opt in options:
            if stats[opt]["visits"] == 0:
                return opt
        total = sum(v["visits"] for v in stats.values())
        log_total = math.log(total)

        def uct(opt: Any) -> float:
            s = stats[opt]
            return (
                s["value"] / s["visits"]
            ) + self.config.exploration_weight * math.sqrt(log_total / s["visits"])

        return max(options, key=uct)
