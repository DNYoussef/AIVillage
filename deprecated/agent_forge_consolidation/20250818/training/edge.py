"""
Edge-of-chaos controller for optimal learning curriculum.
Maintains success rate in the productive struggle zone (55-75%).
"""

import random
from collections import deque

import numpy as np


class EdgeController:
    """
    Bandit-based controller that maintains task success rate in the optimal
    learning zone (55-75%) by adjusting difficulty parameters.

    Implements the "Intelligence at the Edge of Chaos" principle.
    """

    def __init__(
        self,
        target_range: tuple[float, float] = (0.55, 0.75),
        window_size: int = 100,
        exploration_rate: float = 0.1,
        difficulty_params: dict[str, tuple[float, float]] | None = None,
    ):
        self.target_min, self.target_max = target_range
        self.target_center = (self.target_min + self.target_max) / 2
        self.window_size = window_size
        self.exploration_rate = exploration_rate

        # Recent success history
        self.success_history = deque(maxlen=window_size)

        # Difficulty parameters and their ranges
        self.difficulty_params = difficulty_params or {
            "max_tokens": (50, 500),
            "complexity_level": (1, 10),
            "bug_depth": (1, 5),
            "api_diversity": (1, 20),
            "constraint_count": (0, 10),
            "time_pressure": (0.5, 5.0),  # relative to baseline
        }

        # Current difficulty settings
        self.current_difficulty = {
            param: (min_val + max_val) / 2
            for param, (min_val, max_val) in self.difficulty_params.items()
        }

        # Multi-armed bandit state
        self.arm_rewards = {}  # difficulty config -> reward history
        self.arm_counts = {}  # difficulty config -> pull count

        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.momentum = 0.9
        self.velocity = dict.fromkeys(self.difficulty_params, 0.0)

    def update(self, recent_scores: list[float]) -> dict[str, float]:
        """
        Update controller with recent task success scores and return
        adjusted difficulty parameters.
        """
        # Update history
        self.success_history.extend(recent_scores)

        if len(self.success_history) < 10:
            # Not enough data yet, return current settings
            return self.current_difficulty

        # Calculate current success rate
        current_rate = np.mean(list(self.success_history))

        # Determine adjustment direction
        if current_rate < self.target_min:
            # Too hard, decrease difficulty
            adjustment = "decrease"
            magnitude = (self.target_min - current_rate) / self.target_min
        elif current_rate > self.target_max:
            # Too easy, increase difficulty
            adjustment = "increase"
            magnitude = (current_rate - self.target_max) / (1 - self.target_max)
        else:
            # In target zone, minor adjustments or exploration
            adjustment = "maintain"
            magnitude = 0.0

        # Apply adjustments
        if adjustment != "maintain" or random.random() < self.exploration_rate:
            self._adjust_difficulty(adjustment, magnitude)

        # Log arm reward for bandit
        config_key = self._get_config_key()
        reward = self._compute_reward(current_rate)

        if config_key not in self.arm_rewards:
            self.arm_rewards[config_key] = deque(maxlen=20)
            self.arm_counts[config_key] = 0

        self.arm_rewards[config_key].append(reward)
        self.arm_counts[config_key] += 1

        return self.current_difficulty.copy()

    def _adjust_difficulty(self, direction: str, magnitude: float):
        """Adjust difficulty parameters based on direction and magnitude."""
        for param, (min_val, max_val) in self.difficulty_params.items():
            current = self.current_difficulty[param]
            range_size = max_val - min_val

            # Compute target value
            if direction == "decrease":
                target = current - magnitude * range_size * self.adaptation_rate
            elif direction == "increase":
                target = current + magnitude * range_size * self.adaptation_rate
            else:  # maintain with exploration
                # Small random perturbation
                target = current + random.gauss(0, 0.05 * range_size)

            # Apply momentum
            self.velocity[param] = self.momentum * self.velocity[param] + (
                1 - self.momentum
            ) * (target - current)

            # Update value with velocity
            new_value = current + self.velocity[param]

            # Clamp to valid range
            self.current_difficulty[param] = np.clip(new_value, min_val, max_val)

    def _compute_reward(self, success_rate: float) -> float:
        """
        Compute reward for current success rate.
        Maximum reward when in target zone, penalty for being outside.
        """
        if self.target_min <= success_rate <= self.target_max:
            # In target zone - high reward
            distance_from_center = abs(success_rate - self.target_center)
            return 1.0 - 0.5 * (
                distance_from_center / (self.target_max - self.target_min)
            )
        elif success_rate < self.target_min:
            # Too hard - negative reward
            return -2 * (self.target_min - success_rate)
        else:
            # Too easy - negative reward
            return -2 * (success_rate - self.target_max)

    def _get_config_key(self) -> str:
        """Get hashable key for current difficulty configuration."""
        # Discretize continuous values for bandit arms
        discretized = {}
        for param, value in self.current_difficulty.items():
            min_val, max_val = self.difficulty_params[param]
            # Discretize into 5 bins
            bin_size = (max_val - min_val) / 5
            bin_idx = int((value - min_val) / bin_size)
            discretized[param] = min(bin_idx, 4)  # Ensure max bin is 4

        return str(sorted(discretized.items()))

    def get_best_config(self) -> dict[str, float]:
        """Get the configuration with highest average reward (UCB selection)."""
        if not self.arm_rewards:
            return self.current_difficulty

        best_config = None
        best_ucb = -float("inf")
        total_pulls = sum(self.arm_counts.values())

        for config_key, rewards in self.arm_rewards.items():
            if not rewards:
                continue

            avg_reward = np.mean(list(rewards))
            pulls = self.arm_counts[config_key]

            # Upper Confidence Bound (UCB1)
            if total_pulls > 0 and pulls > 0:
                exploration_bonus = np.sqrt(2 * np.log(total_pulls) / pulls)
            else:
                exploration_bonus = float("inf")

            ucb = avg_reward + exploration_bonus

            if ucb > best_ucb:
                best_ucb = ucb
                best_config = config_key

        # Parse config key back to difficulty settings
        if best_config:
            # This is simplified - in practice you'd properly deserialize
            return self.current_difficulty

        return self.current_difficulty

    def get_metrics(self) -> dict[str, float]:
        """Get current controller metrics."""
        if not self.success_history:
            return {"success_rate": 0.0, "in_target_zone": False}

        current_rate = np.mean(list(self.success_history))

        return {
            "success_rate": current_rate,
            "in_target_zone": self.target_min <= current_rate <= self.target_max,
            "distance_from_target": min(
                abs(current_rate - self.target_min), abs(current_rate - self.target_max)
            )
            if not (self.target_min <= current_rate <= self.target_max)
            else 0.0,
            "total_samples": len(self.success_history),
            "num_arms_explored": len(self.arm_rewards),
        }


class ComplexityEstimator:
    """Estimates task complexity using various metrics."""

    @staticmethod
    def estimate_entropy(text: str) -> float:
        """Shannon entropy of character distribution."""
        if not text:
            return 0.0

        import math
        from collections import Counter

        counts = Counter(text)
        total = len(text)
        entropy = -sum(
            (count / total) * math.log2(count / total) for count in counts.values()
        )
        return entropy

    @staticmethod
    def estimate_lz_complexity(text: str) -> float:
        """Lempel-Ziv complexity as a measure of compressibility."""
        if not text:
            return 0.0

        import zlib

        compressed = zlib.compress(text.encode())
        return len(compressed) / len(text)

    @staticmethod
    def estimate_ast_complexity(code: str) -> float:
        """AST-based complexity for code."""
        try:
            import ast

            tree = ast.parse(code)

            # Count different node types
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1

            # Weighted complexity based on node types
            weights = {
                "For": 3,
                "While": 3,
                "If": 2,
                "FunctionDef": 2,
                "ClassDef": 3,
                "Try": 2,
                "With": 2,
            }

            complexity = sum(
                node_counts.get(node_type, 0) * weight
                for node_type, weight in weights.items()
            )

            return complexity
        except:
            return 0.0

    @staticmethod
    def estimate_overall(text: str, is_code: bool = True) -> float:
        """Combined complexity estimate."""
        entropy = ComplexityEstimator.estimate_entropy(text)
        lz = ComplexityEstimator.estimate_lz_complexity(text)

        if is_code:
            ast_comp = ComplexityEstimator.estimate_ast_complexity(text)
            return (entropy + lz + ast_comp) / 3
        else:
            return (entropy + lz) / 2
