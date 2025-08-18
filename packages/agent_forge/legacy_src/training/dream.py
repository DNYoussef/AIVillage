"""
Dream buffer and replay system for consolidation and generative replay.
Implements sleep/dream cycles for improved learning and generalization.
"""

import json
import logging
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DreamExample:
    """Single example stored in the dream buffer."""

    # Core content
    prompt: str
    target: str
    model_output: str

    # Performance metrics
    loss: float
    accuracy: float
    confidence: float

    # Context
    temperature: float
    step: int
    stage: str

    # Telemetry at time of generation
    grad_norm: float
    ema_cos: float
    id_value: float

    # Augmentation metadata
    is_augmented: bool = False
    augmentation_type: str | None = None
    parent_id: str | None = None

    # Unique identifier
    example_id: str = ""

    def __post_init__(self):
        if not self.example_id:
            # Generate unique ID based on content and step
            import hashlib

            content = f"{self.prompt}_{self.step}_{random.random()}"
            self.example_id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DreamExample":
        return cls(**data)


class DreamBuffer:
    """
    Stores near-threshold examples for replay during sleep cycles.
    Focuses on examples that are challenging but learnable.
    """

    def __init__(
        self,
        capacity: int = 10000,
        near_threshold_range: tuple[float, float] = (0.3, 0.7),
        prioritize_recent: bool = True,
        save_path: Path | None = None,
    ):
        """
        Args:
            capacity: Maximum number of examples to store
            near_threshold_range: (min, max) accuracy for near-threshold
            prioritize_recent: Whether to prioritize recent examples
            save_path: Path to save/load buffer state
        """
        self.capacity = capacity
        self.near_threshold_min, self.near_threshold_max = near_threshold_range
        self.prioritize_recent = prioritize_recent
        self.save_path = save_path

        # Main storage
        self.buffer = deque(maxlen=capacity)

        # Categorized storage for efficient sampling
        self.categories = {
            "near_success": deque(maxlen=capacity // 3),  # 60-70% accuracy
            "near_failure": deque(maxlen=capacity // 3),  # 30-40% accuracy
            "edge_cases": deque(maxlen=capacity // 3),  # Unusual patterns
        }

        # Statistics
        self.total_pushed = 0
        self.sample_counts = {}

        # Priority weights for sampling
        self.priority_weights = np.ones(capacity)

        if save_path and save_path.exists():
            self.load(save_path)

    def push(self, example: DreamExample):
        """Add an example to the buffer."""
        self.buffer.append(example)
        self.total_pushed += 1

        # Categorize the example
        if self.near_threshold_min <= example.accuracy <= self.near_threshold_max:
            if example.accuracy >= 0.6:
                self.categories["near_success"].append(example)
            else:
                self.categories["near_failure"].append(example)

        # Detect edge cases (high loss but correct, or low loss but wrong)
        is_edge_case = (example.loss > 2.0 and example.accuracy > 0.8) or (
            example.loss < 0.5 and example.accuracy < 0.2
        )
        if is_edge_case:
            self.categories["edge_cases"].append(example)

        # Update priority weights
        if self.prioritize_recent:
            # Exponential decay for older examples
            idx = len(self.buffer) - 1
            self.priority_weights = np.ones(len(self.buffer))
            decay_factor = 0.99
            for i in range(len(self.buffer)):
                age = idx - i
                self.priority_weights[i] = decay_factor**age

    def sample(self, k: int, strategy: str = "balanced", stage_filter: str | None = None) -> list[DreamExample]:
        """
        Sample k examples from the buffer.

        Args:
            k: Number of examples to sample
            strategy: Sampling strategy
                - "balanced": Mix of categories
                - "near_threshold": Focus on near-threshold examples
                - "edge_cases": Focus on unusual patterns
                - "uniform": Random sampling
                - "priority": Weighted by recency/importance
            stage_filter: Only sample from specific training stage

        Returns:
            List of sampled examples
        """
        if len(self.buffer) == 0:
            return []

        # Apply stage filter if specified
        candidates = list(self.buffer)
        if stage_filter:
            candidates = [ex for ex in candidates if ex.stage == stage_filter]

        if len(candidates) == 0:
            return []

        # Limit k to available candidates
        k = min(k, len(candidates))

        if strategy == "balanced":
            # Sample equally from each category
            samples = []
            per_category = k // 3
            remainder = k % 3

            for category, examples in self.categories.items():
                if examples:
                    n = per_category + (1 if remainder > 0 else 0)
                    remainder -= 1

                    cat_samples = random.sample(list(examples), min(n, len(examples)))
                    samples.extend(cat_samples)

            # Fill remaining with uniform sampling
            if len(samples) < k:
                remaining = k - len(samples)
                extra = random.sample(candidates, min(remaining, len(candidates)))
                samples.extend(extra)

            return samples[:k]

        elif strategy == "near_threshold":
            # Focus on near-threshold examples
            near_threshold = [
                ex for ex in candidates if self.near_threshold_min <= ex.accuracy <= self.near_threshold_max
            ]

            if len(near_threshold) >= k:
                return random.sample(near_threshold, k)
            else:
                # Fill with other examples
                others = [ex for ex in candidates if ex not in near_threshold]
                return near_threshold + random.sample(others, min(k - len(near_threshold), len(others)))

        elif strategy == "edge_cases":
            # Prioritize edge cases
            edge_cases = list(self.categories["edge_cases"])
            if len(edge_cases) >= k:
                return random.sample(edge_cases, k)
            else:
                others = [ex for ex in candidates if ex not in edge_cases]
                return edge_cases + random.sample(others, min(k - len(edge_cases), len(others)))

        elif strategy == "priority":
            # Weighted sampling by priority
            weights = self.priority_weights[: len(candidates)]
            weights = weights / weights.sum()

            indices = np.random.choice(len(candidates), size=k, replace=False, p=weights)

            return [candidates[i] for i in indices]

        else:  # uniform
            return random.sample(candidates, k)

    def get_statistics(self) -> dict[str, Any]:
        """Get buffer statistics."""
        if not self.buffer:
            return {
                "total_examples": 0,
                "categories": dict.fromkeys(self.categories, 0),
            }

        accuracies = [ex.accuracy for ex in self.buffer]
        losses = [ex.loss for ex in self.buffer]

        return {
            "total_examples": len(self.buffer),
            "total_pushed": self.total_pushed,
            "capacity_used": len(self.buffer) / self.capacity,
            "avg_accuracy": np.mean(accuracies),
            "avg_loss": np.mean(losses),
            "accuracy_std": np.std(accuracies),
            "categories": {k: len(v) for k, v in self.categories.items()},
            "stage_distribution": self._get_stage_distribution(),
        }

    def _get_stage_distribution(self) -> dict[str, int]:
        """Get distribution of examples by training stage."""
        distribution = {}
        for ex in self.buffer:
            distribution[ex.stage] = distribution.get(ex.stage, 0) + 1
        return distribution

    def save(self, path: Path | None = None):
        """Save buffer to disk."""
        save_path = path or self.save_path
        if not save_path:
            return

        data = {
            "buffer": [ex.to_dict() for ex in self.buffer],
            "statistics": self.get_statistics(),
            "config": {
                "capacity": self.capacity,
                "near_threshold_range": (
                    self.near_threshold_min,
                    self.near_threshold_max,
                ),
                "total_pushed": self.total_pushed,
            },
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Dream buffer saved to {save_path}")

    def load(self, path: Path):
        """Load buffer from disk."""
        if not path.exists():
            logger.warning(f"Dream buffer file not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        self.buffer.clear()
        for ex_dict in data["buffer"]:
            self.buffer.append(DreamExample.from_dict(ex_dict))

        if "config" in data:
            self.total_pushed = data["config"].get("total_pushed", len(self.buffer))

        logger.info(f"Dream buffer loaded from {path}: {len(self.buffer)} examples")


class DreamAugmenter:
    """
    Augments dream examples with variations to improve generalization.
    """

    def __init__(
        self,
        augmentation_types: list[str] = None,
        augmentation_probability: float = 0.3,
    ):
        """
        Args:
            augmentation_types: Types of augmentation to apply
            augmentation_probability: Probability of augmenting each example
        """
        if augmentation_types is None:
            augmentation_types = [
                "api_perturbation",
                "spec_tightening",
                "style_variance",
                "error_injection",
                "context_shift",
            ]

        self.augmentation_types = augmentation_types
        self.augmentation_probability = augmentation_probability

    def augment(self, example: DreamExample) -> list[DreamExample]:
        """
        Generate augmented versions of an example.

        Returns:
            List of augmented examples (may include original)
        """
        augmented = [example]  # Always include original

        if random.random() > self.augmentation_probability:
            return augmented

        # Select augmentation type
        aug_type = random.choice(self.augmentation_types)

        if aug_type == "api_perturbation":
            # Modify API calls in the prompt
            aug_example = self._augment_api(example)

        elif aug_type == "spec_tightening":
            # Add stricter constraints
            aug_example = self._augment_spec(example)

        elif aug_type == "style_variance":
            # Change coding style requirements
            aug_example = self._augment_style(example)

        elif aug_type == "error_injection":
            # Add common errors to fix
            aug_example = self._augment_errors(example)

        elif aug_type == "context_shift":
            # Change context or domain
            aug_example = self._augment_context(example)

        else:
            return augmented

        if aug_example:
            augmented.append(aug_example)

        return augmented

    def _augment_api(self, example: DreamExample) -> DreamExample | None:
        """Perturb API calls in the example."""
        aug = DreamExample(
            prompt=self._modify_api_names(example.prompt),
            target=example.target,
            model_output="",  # Will be regenerated
            loss=example.loss * 1.1,  # Slightly harder
            accuracy=example.accuracy * 0.9,
            confidence=example.confidence * 0.9,
            temperature=example.temperature,
            step=example.step,
            stage=example.stage,
            grad_norm=example.grad_norm,
            ema_cos=example.ema_cos,
            id_value=example.id_value,
            is_augmented=True,
            augmentation_type="api_perturbation",
            parent_id=example.example_id,
        )
        return aug

    def _modify_api_names(self, text: str) -> str:
        """Simple API name modification."""
        # This is a simplified version - in practice would be more sophisticated
        replacements = {
            "print": "display",
            "len": "length",
            "str": "string",
            "int": "integer",
            "list": "array",
            "dict": "mapping",
        }

        modified = text
        for old, new in replacements.items():
            if random.random() < 0.3:  # Don't replace everything
                modified = modified.replace(old, new)

        return modified

    def _augment_spec(self, example: DreamExample) -> DreamExample | None:
        """Add stricter specifications."""
        constraints = [
            "\nAdditional constraint: The solution must use O(1) space.",
            "\nAdditional constraint: The solution must be thread-safe.",
            "\nAdditional constraint: Handle edge cases explicitly.",
            "\nAdditional constraint: Add comprehensive error handling.",
        ]

        aug = DreamExample(
            prompt=example.prompt + random.choice(constraints),
            target=example.target,
            model_output="",
            loss=example.loss * 1.2,
            accuracy=example.accuracy * 0.85,
            confidence=example.confidence * 0.85,
            temperature=example.temperature,
            step=example.step,
            stage=example.stage,
            grad_norm=example.grad_norm,
            ema_cos=example.ema_cos,
            id_value=example.id_value,
            is_augmented=True,
            augmentation_type="spec_tightening",
            parent_id=example.example_id,
        )
        return aug

    def _augment_style(self, example: DreamExample) -> DreamExample | None:
        """Change coding style requirements."""
        styles = [
            "\nUse functional programming style.",
            "\nUse object-oriented design patterns.",
            "\nMinimize the number of lines.",
            "\nOptimize for readability over brevity.",
        ]

        aug = DreamExample(
            prompt=example.prompt + random.choice(styles),
            target=example.target,
            model_output="",
            loss=example.loss,
            accuracy=example.accuracy,
            confidence=example.confidence * 0.95,
            temperature=example.temperature * 1.1,
            step=example.step,
            stage=example.stage,
            grad_norm=example.grad_norm,
            ema_cos=example.ema_cos,
            id_value=example.id_value,
            is_augmented=True,
            augmentation_type="style_variance",
            parent_id=example.example_id,
        )
        return aug

    def _augment_errors(self, example: DreamExample) -> DreamExample | None:
        """Inject common errors to fix."""
        errors = [
            "\nThe provided code has an off-by-one error. Fix it:",
            "\nThe provided code has a null pointer issue. Fix it:",
            "\nThe provided code has incorrect variable scoping. Fix it:",
            "\nThe provided code has a logic error. Fix it:",
        ]

        aug = DreamExample(
            prompt=random.choice(errors) + "\n" + example.prompt,
            target=example.target,
            model_output="",
            loss=example.loss * 1.3,
            accuracy=example.accuracy * 0.8,
            confidence=example.confidence * 0.8,
            temperature=example.temperature,
            step=example.step,
            stage=example.stage,
            grad_norm=example.grad_norm,
            ema_cos=example.ema_cos,
            id_value=example.id_value,
            is_augmented=True,
            augmentation_type="error_injection",
            parent_id=example.example_id,
        )
        return aug

    def _augment_context(self, example: DreamExample) -> DreamExample | None:
        """Shift the context or domain."""
        contexts = [
            "In a distributed system context: ",
            "For a mobile application: ",
            "In a real-time system: ",
            "For a machine learning pipeline: ",
        ]

        aug = DreamExample(
            prompt=random.choice(contexts) + example.prompt,
            target=example.target,
            model_output="",
            loss=example.loss * 1.1,
            accuracy=example.accuracy * 0.9,
            confidence=example.confidence * 0.9,
            temperature=example.temperature,
            step=example.step,
            stage=example.stage,
            grad_norm=example.grad_norm,
            ema_cos=example.ema_cos,
            id_value=example.id_value,
            is_augmented=True,
            augmentation_type="context_shift",
            parent_id=example.example_id,
        )
        return aug


class DreamCycleManager:
    """
    Manages sleep/dream cycles during training.
    Coordinates replay timing and augmentation strategies.
    """

    def __init__(
        self,
        dream_buffer: DreamBuffer,
        augmenter: DreamAugmenter,
        cycle_interval: int = 1000,  # Steps between dream cycles
        dream_duration: int = 50,  # Steps per dream cycle
        replay_batch_size: int = 32,
    ):
        self.buffer = dream_buffer
        self.augmenter = augmenter
        self.cycle_interval = cycle_interval
        self.dream_duration = dream_duration
        self.replay_batch_size = replay_batch_size

        self.steps_since_dream = 0
        self.total_dreams = 0
        self.current_cycle_step = 0
        self.is_dreaming = False

    def should_dream(self, step: int) -> bool:
        """Check if it's time for a dream cycle."""
        self.steps_since_dream += 1
        return self.steps_since_dream >= self.cycle_interval

    def start_dream_cycle(self):
        """Begin a dream cycle."""
        self.is_dreaming = True
        self.current_cycle_step = 0
        self.steps_since_dream = 0
        self.total_dreams += 1
        logger.info(f"Starting dream cycle {self.total_dreams}")

    def get_dream_batch(self, stage: str | None = None) -> list[DreamExample]:
        """Get a batch of examples for dream replay."""
        if not self.is_dreaming:
            return []

        # Sample from buffer
        examples = self.buffer.sample(
            self.replay_batch_size,
            strategy="balanced" if self.current_cycle_step < self.dream_duration // 2 else "edge_cases",
            stage_filter=stage,
        )

        # Augment some examples
        augmented_batch = []
        for ex in examples:
            augmented_batch.extend(self.augmenter.augment(ex))

        self.current_cycle_step += 1

        # End dream cycle if duration reached
        if self.current_cycle_step >= self.dream_duration:
            self.end_dream_cycle()

        return augmented_batch

    def end_dream_cycle(self):
        """End the current dream cycle."""
        self.is_dreaming = False
        self.current_cycle_step = 0
        logger.info(f"Ended dream cycle {self.total_dreams}")

    def get_metrics(self) -> dict[str, Any]:
        """Get dream cycle metrics."""
        return {
            "total_dreams": self.total_dreams,
            "steps_since_dream": self.steps_since_dream,
            "is_dreaming": self.is_dreaming,
            "current_cycle_step": self.current_cycle_step,
            "buffer_stats": self.buffer.get_statistics(),
        }
