"""
Evaluation System for Cogment Training.

Provides stage-specific evaluation metrics, convergence detection,
and performance monitoring for the 4-stage curriculum.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time
from typing import Any

import torch

from .curriculum import CurriculumStage, StageConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for a training step/epoch."""

    # Basic metrics
    loss: float
    accuracy: float
    ponder_cost: float
    step: int
    stage: int

    # Loss breakdown
    deep_supervision_loss: float | None = None
    improvement_loss: float | None = None
    consistency_loss: float | None = None
    ponder_loss: float | None = None

    # Performance metrics
    throughput: float | None = None  # samples/second
    memory_usage: float | None = None  # GB

    # Stage-specific metrics
    convergence_score: float | None = None
    grokking_ratio: float | None = None
    refinement_efficiency: float | None = None

    # Timing
    eval_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ConvergenceDetector:
    """
    Detects convergence and grokking patterns in training metrics.

    Uses multiple signals to determine when a stage has converged:
    - Loss plateau detection
    - Accuracy stability
    - Ponder cost stabilization
    - Grokking onset detection
    """

    def __init__(
        self,
        patience: int = 500,
        min_delta: float = 1e-4,
        smoothing_window: int = 20,
        grokking_threshold: float = 0.1,  # Minimum improvement for grokking
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.smoothing_window = smoothing_window
        self.grokking_threshold = grokking_threshold

        # History tracking
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.ponder_history: list[float] = []

        # Convergence state
        self.steps_without_improvement = 0
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.grokking_detected = False
        self.convergence_detected = False

    def update(self, metrics: EvaluationMetrics) -> dict[str, bool]:
        """
        Update convergence detector with new metrics.

        Args:
            metrics: Latest evaluation metrics

        Returns:
            Dictionary of convergence signals
        """
        # Update history
        self.loss_history.append(metrics.loss)
        self.accuracy_history.append(metrics.accuracy)
        self.ponder_history.append(metrics.ponder_cost)

        # Trim history to window size
        max_history = self.patience * 2
        if len(self.loss_history) > max_history:
            self.loss_history = self.loss_history[-max_history:]
            self.accuracy_history = self.accuracy_history[-max_history:]
            self.ponder_history = self.ponder_history[-max_history:]

        # Check for improvement
        improved = False
        if metrics.loss < self.best_loss - self.min_delta:
            self.best_loss = metrics.loss
            improved = True
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if metrics.accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = metrics.accuracy

        # Detect grokking (sudden improvement after plateau)
        grokking_detected = self._detect_grokking()
        if grokking_detected and not self.grokking_detected:
            self.grokking_detected = True
            logger.info(f"Grokking detected at step {metrics.step}!")

        # Detect convergence
        loss_converged = self._check_loss_convergence()
        accuracy_stable = self._check_accuracy_stability()
        ponder_stable = self._check_ponder_stability()

        # Overall convergence requires loss plateau AND (accuracy OR ponder stability)
        converged = loss_converged and (accuracy_stable or ponder_stable)
        if converged and not self.convergence_detected:
            self.convergence_detected = True
            logger.info(f"Convergence detected at step {metrics.step}")

        return {
            "improved": improved,
            "grokking_detected": self.grokking_detected,
            "converged": self.convergence_detected,
            "loss_plateau": loss_converged,
            "accuracy_stable": accuracy_stable,
            "ponder_stable": ponder_stable,
            "steps_without_improvement": self.steps_without_improvement,
        }

    def _detect_grokking(self) -> bool:
        """Detect grokking pattern: sudden improvement after plateau."""
        if len(self.accuracy_history) < self.smoothing_window * 2:
            return False

        # Look for sudden accuracy jump
        recent_window = self.accuracy_history[-self.smoothing_window :]
        previous_window = self.accuracy_history[-self.smoothing_window * 2 : -self.smoothing_window]

        recent_mean = sum(recent_window) / len(recent_window)
        previous_mean = sum(previous_window) / len(previous_window)

        # Grokking: significant improvement (>threshold) in recent window
        improvement = recent_mean - previous_mean
        return improvement > self.grokking_threshold

    def _check_loss_convergence(self) -> bool:
        """Check if loss has plateaued."""
        return self.steps_without_improvement >= self.patience

    def _check_accuracy_stability(self) -> bool:
        """Check if accuracy has stabilized."""
        if len(self.accuracy_history) < self.smoothing_window:
            return False

        recent = self.accuracy_history[-self.smoothing_window :]
        std_dev = math.sqrt(sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent))
        return std_dev < self.min_delta

    def _check_ponder_stability(self) -> bool:
        """Check if ponder cost has stabilized."""
        if len(self.ponder_history) < self.smoothing_window:
            return False

        recent = self.ponder_history[-self.smoothing_window :]
        std_dev = math.sqrt(sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent))
        return std_dev < 0.1  # Ponder cost stability threshold

    def get_convergence_summary(self) -> dict[str, Any]:
        """Get comprehensive convergence summary."""
        return {
            "converged": self.convergence_detected,
            "grokking_detected": self.grokking_detected,
            "best_loss": self.best_loss,
            "best_accuracy": self.best_accuracy,
            "steps_without_improvement": self.steps_without_improvement,
            "total_evaluations": len(self.loss_history),
            "recent_loss_trend": self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history,
            "recent_accuracy_trend": (
                self.accuracy_history[-10:] if len(self.accuracy_history) >= 10 else self.accuracy_history
            ),
        }

    def reset(self):
        """Reset convergence detector for new stage."""
        self.loss_history.clear()
        self.accuracy_history.clear()
        self.ponder_history.clear()
        self.steps_without_improvement = 0
        self.best_loss = float("inf")
        self.best_accuracy = 0.0
        self.grokking_detected = False
        self.convergence_detected = False


class StageEvaluator:
    """
    Stage-specific evaluator for the 4-stage curriculum.

    Provides specialized evaluation metrics and criteria for each curriculum stage,
    with automatic convergence detection and stage transition recommendations.
    """

    def __init__(self):
        self.convergence_detector = ConvergenceDetector()
        self.current_stage = CurriculumStage.SANITY
        self.stage_start_time = time.time()
        self.evaluation_history: list[EvaluationMetrics] = []

        # Stage-specific thresholds
        self.stage_thresholds = {
            CurriculumStage.SANITY: {"min_accuracy": 0.8, "max_ponder_cost": 2.0, "convergence_patience": 100},
            CurriculumStage.ARC_VISUAL: {
                "min_accuracy": 0.75,
                "max_ponder_cost": 3.0,
                "convergence_patience": 800,
                "grokking_expected": True,
            },
            CurriculumStage.ALGORITHMIC: {
                "min_accuracy": 0.7,
                "max_ponder_cost": 4.0,
                "convergence_patience": 1500,
                "grokking_expected": True,
            },
            CurriculumStage.MATH_TEXT: {"min_accuracy": 0.65, "max_ponder_cost": 5.0, "convergence_patience": 3000},
            CurriculumStage.LONG_CONTEXT: {"min_accuracy": 0.6, "max_ponder_cost": 6.0, "convergence_patience": 5000},
        }

        logger.info("Initialized StageEvaluator")

    def set_stage(self, stage: CurriculumStage):
        """Set current evaluation stage and reset convergence detector."""
        if stage != self.current_stage:
            logger.info(f"StageEvaluator transitioning from {self.current_stage.name} to {stage.name}")
            self.current_stage = stage
            self.stage_start_time = time.time()

            # Update convergence detector parameters for new stage
            thresholds = self.stage_thresholds.get(stage, {})
            self.convergence_detector.patience = thresholds.get("convergence_patience", 500)
            self.convergence_detector.reset()

    def evaluate_batch(
        self, model_output: dict[str, torch.Tensor], targets: torch.Tensor, step: int, return_detailed: bool = False
    ) -> EvaluationMetrics:
        """
        Evaluate a single batch with stage-specific metrics.

        Args:
            model_output: Dictionary containing model outputs
            targets: Target token sequences
            step: Current training step
            return_detailed: Whether to compute detailed metrics

        Returns:
            Comprehensive evaluation metrics
        """
        start_time = time.time()

        # Extract basic outputs
        logits = model_output.get("logits")
        loss = model_output.get("loss", torch.tensor(0.0))
        ponder_cost = model_output.get("ponder_cost", torch.tensor(1.0))

        # Basic metrics
        with torch.no_grad():
            # Accuracy calculation
            predictions = logits.argmax(dim=-1) if logits is not None else torch.zeros_like(targets)
            correct = (predictions == targets).float()
            mask = (targets != -100).float()
            accuracy = (correct * mask).sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

            # Convert to Python values
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            accuracy_val = accuracy.item() if torch.is_tensor(accuracy) else accuracy
            ponder_val = ponder_cost.mean().item() if torch.is_tensor(ponder_cost) else ponder_cost

        # Create base metrics
        metrics = EvaluationMetrics(
            loss=loss_val,
            accuracy=accuracy_val,
            ponder_cost=ponder_val,
            step=step,
            stage=self.current_stage.value,
            eval_time=time.time() - start_time,
        )

        # Add detailed metrics if requested
        if return_detailed:
            metrics = self._add_detailed_metrics(metrics, model_output, targets)

        # Update convergence detector
        convergence_signals = self.convergence_detector.update(metrics)
        metrics.convergence_score = self._compute_convergence_score(convergence_signals)

        # Add to history
        self.evaluation_history.append(metrics)

        return metrics

    def _add_detailed_metrics(
        self, metrics: EvaluationMetrics, model_output: dict[str, torch.Tensor], targets: torch.Tensor
    ) -> EvaluationMetrics:
        """Add detailed stage-specific metrics."""
        # Memory usage (approximate)
        if torch.cuda.is_available():
            metrics.memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB

        # Refinement efficiency (if available)
        if "num_steps" in model_output:
            num_steps = model_output["num_steps"]
            target_steps = self._get_target_steps_for_stage()
            metrics.refinement_efficiency = target_steps / (num_steps.mean().item() + 1e-8)

        # Loss breakdown (if available)
        if "loss_breakdown" in model_output:
            breakdown = model_output["loss_breakdown"]
            metrics.deep_supervision_loss = breakdown.get("deep_supervision", 0.0)
            metrics.improvement_loss = breakdown.get("improvement", 0.0)
            metrics.consistency_loss = breakdown.get("consistency", 0.0)
            metrics.ponder_loss = breakdown.get("ponder", 0.0)

        return metrics

    def _get_target_steps_for_stage(self) -> float:
        """Get target refinement steps for current stage."""
        target_steps = {
            CurriculumStage.SANITY: 1.5,
            CurriculumStage.ARC_VISUAL: 2.5,
            CurriculumStage.ALGORITHMIC: 4.0,
            CurriculumStage.MATH_TEXT: 5.0,
            CurriculumStage.LONG_CONTEXT: 4.0,  # Efficiency focus
        }
        return target_steps.get(self.current_stage, 3.0)

    def _compute_convergence_score(self, signals: dict[str, bool]) -> float:
        """Compute overall convergence score from signals."""
        # Weight different signals
        weights = {
            "improved": -0.2,  # Recent improvement reduces convergence score
            "loss_plateau": 0.4,
            "accuracy_stable": 0.3,
            "ponder_stable": 0.2,
            "grokking_detected": 0.1,
        }

        score = 0.0
        for signal, weight in weights.items():
            if signal in signals and signals[signal]:
                score += weight

        return max(0.0, min(1.0, score))  # Clamp to [0, 1]

    def check_stage_completion(self, config: StageConfig | None = None) -> dict[str, Any]:
        """
        Check if current stage meets completion criteria.

        Args:
            config: Optional stage configuration (uses defaults if None)

        Returns:
            Dictionary with completion status and criteria evaluation
        """
        if not self.evaluation_history:
            return {"complete": False, "reason": "no_evaluations"}

        # Get thresholds
        thresholds = self.stage_thresholds.get(self.current_stage, {})
        if config is not None:
            # Override with provided config
            thresholds.update(
                {
                    "min_accuracy": config.min_accuracy,
                    "max_ponder_cost": config.max_ponder_cost,
                    "convergence_patience": config.convergence_patience,
                }
            )

        # Get recent metrics
        recent_metrics = self.evaluation_history[-10:]  # Last 10 evaluations
        if not recent_metrics:
            return {"complete": False, "reason": "insufficient_data"}

        # Check completion criteria
        avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)
        avg_ponder_cost = sum(m.ponder_cost for m in recent_metrics) / len(recent_metrics)

        criteria = {
            "accuracy_met": avg_accuracy >= thresholds.get("min_accuracy", 0.6),
            "ponder_cost_met": avg_ponder_cost <= thresholds.get("max_ponder_cost", 5.0),
            "converged": self.convergence_detector.convergence_detected,
            "grokking_detected": self.convergence_detector.grokking_detected,
        }

        # Grokking stages require grokking detection
        grokking_expected = thresholds.get("grokking_expected", False)
        if grokking_expected:
            criteria["grokking_required"] = self.convergence_detector.grokking_detected

        # All criteria must be met
        required_criteria = ["accuracy_met", "ponder_cost_met", "converged"]
        if grokking_expected:
            required_criteria.append("grokking_required")

        all_met = all(criteria.get(c, False) for c in required_criteria)

        completion_status = {
            "complete": all_met,
            "criteria": criteria,
            "thresholds": thresholds,
            "recent_metrics": {
                "avg_accuracy": avg_accuracy,
                "avg_ponder_cost": avg_ponder_cost,
                "evaluations_count": len(recent_metrics),
            },
            "stage_duration": time.time() - self.stage_start_time,
            "convergence_summary": self.convergence_detector.get_convergence_summary(),
        }

        return completion_status

    def get_stage_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of current stage evaluation."""
        if not self.evaluation_history:
            return {"status": "no_data"}

        # Calculate aggregate metrics
        metrics_count = len(self.evaluation_history)
        avg_loss = sum(m.loss for m in self.evaluation_history) / metrics_count
        avg_accuracy = sum(m.accuracy for m in self.evaluation_history) / metrics_count
        avg_ponder_cost = sum(m.ponder_cost for m in self.evaluation_history) / metrics_count

        # Best metrics
        best_loss = min(m.loss for m in self.evaluation_history)
        best_accuracy = max(m.accuracy for m in self.evaluation_history)

        # Recent trend (last 20% of evaluations)
        recent_count = max(1, metrics_count // 5)
        recent_metrics = self.evaluation_history[-recent_count:]
        recent_avg_loss = sum(m.loss for m in recent_metrics) / len(recent_metrics)
        recent_avg_accuracy = sum(m.accuracy for m in recent_metrics) / len(recent_metrics)

        return {
            "stage": self.current_stage,
            "duration": time.time() - self.stage_start_time,
            "evaluations_count": metrics_count,
            "aggregate_metrics": {
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "avg_ponder_cost": avg_ponder_cost,
                "best_loss": best_loss,
                "best_accuracy": best_accuracy,
            },
            "recent_trend": {
                "loss": recent_avg_loss,
                "accuracy": recent_avg_accuracy,
                "improvement": {"loss": avg_loss - recent_avg_loss, "accuracy": recent_avg_accuracy - avg_accuracy},
            },
            "convergence": self.convergence_detector.get_convergence_summary(),
            "completion_status": self.check_stage_completion(),
        }

    def reset_for_stage(self, stage: CurriculumStage):
        """Reset evaluator for new stage."""
        self.set_stage(stage)
        self.evaluation_history.clear()
        logger.info(f"Reset StageEvaluator for stage {stage.name}")
