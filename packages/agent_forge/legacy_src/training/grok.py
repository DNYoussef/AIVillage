"""
Grokfast controller for managing gradient amplification during training.
Implements the Grokfast paper's approach to 50x faster grokking.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GrokfastState:
    """State tracking for Grokfast controller."""

    lambda_current: float = 0.05
    ema_grad: torch.Tensor | None = None
    ema_alpha: float = 0.98
    steps_since_increase: int = 0
    grokking_detected: bool = False
    phase: str = "pre_grok"  # pre_grok, onset, active, post_grok


class GrokController:
    """
    Controls Grokfast lambda parameter based on telemetry signals.

    Grokfast amplifies slow-changing gradient components while dampening
    fast-changing ones, accelerating the grokking phenomenon by up to 50x.
    """

    def __init__(
        self,
        ema_alpha: float = 0.98,
        lam_init: float = 0.05,
        lam_max: float = 0.25,
        lam_increase_rate: float = 1.02,
        lam_decrease_rate: float = 0.98,
        cos_threshold: float = 0.5,
        id_drop_threshold: float = 0.2,
        patience: int = 100,
    ):
        """
        Args:
            ema_alpha: Exponential moving average decay factor
            lam_init: Initial lambda value for gradient amplification
            lam_max: Maximum lambda value
            lam_increase_rate: Multiplicative increase rate
            lam_decrease_rate: Multiplicative decrease rate
            cos_threshold: Cosine similarity threshold for activation
            id_drop_threshold: ID drop threshold for grokking detection
            patience: Steps to wait before decreasing lambda
        """
        self.ema_alpha = ema_alpha
        self.lam_init = lam_init
        self.lam_max = lam_max
        self.lam_increase_rate = lam_increase_rate
        self.lam_decrease_rate = lam_decrease_rate
        self.cos_threshold = cos_threshold
        self.id_drop_threshold = id_drop_threshold
        self.patience = patience

        self.state = GrokfastState(lambda_current=lam_init, ema_alpha=ema_alpha)

        # History for analysis
        self.lambda_history = [lam_init]
        self.cos_history = []
        self.activation_history = []

    def step(self, grad: torch.Tensor, telemetry: "TelemetryFrame") -> dict[str, float]:
        """
        Compute Grokfast lambda based on current gradient and telemetry.

        Args:
            grad: Current gradient tensor
            telemetry: Telemetry frame with metrics

        Returns:
            Dict with 'lambda' value for optimizer
        """
        # Update EMA gradient
        if self.state.ema_grad is None:
            self.state.ema_grad = grad.clone()
        else:
            self.state.ema_grad = self.ema_alpha * self.state.ema_grad + (1 - self.ema_alpha) * grad

        # Compute cosine similarity
        cos_sim = self._compute_cosine_similarity(grad, self.state.ema_grad)
        self.cos_history.append(cos_sim)

        # Analyze phase based on telemetry
        phase = self._determine_phase(telemetry, cos_sim)
        self.state.phase = phase

        # Adjust lambda based on phase and signals
        new_lambda = self._adjust_lambda(phase, cos_sim, telemetry)
        self.state.lambda_current = new_lambda
        self.lambda_history.append(new_lambda)

        # Track activation
        is_active = phase in ["onset", "active"] and cos_sim > self.cos_threshold
        self.activation_history.append(is_active)

        return {
            "lambda": new_lambda,
            "phase": phase,
            "cos_sim": cos_sim,
            "active": is_active,
        }

    def _compute_cosine_similarity(self, grad: torch.Tensor, ema_grad: torch.Tensor) -> float:
        """Compute cosine similarity between gradient and EMA gradient."""
        with torch.no_grad():
            grad_flat = grad.flatten()
            ema_flat = ema_grad.flatten()

            cos_sim = torch.nn.functional.cosine_similarity(grad_flat.unsqueeze(0), ema_flat.unsqueeze(0)).item()

        return cos_sim

    def _determine_phase(self, telemetry: "TelemetryFrame", cos_sim: float) -> str:
        """
        Determine current training phase based on signals.

        Phases:
        - pre_grok: Early training, no grokking signals
        - onset: ID dropping, cos_sim increasing
        - active: Grokking in progress
        - post_grok: Grokking complete, stable performance
        """
        # Check ID trends
        id_dropping = False
        if telemetry.id_by_layer:
            # Average ID across layers
            avg_id = np.mean(list(telemetry.id_by_layer.values()))
            if len(self.lambda_history) > 10:
                # Compare to historical average
                # Simplified: check if ID is low
                if avg_id < 10:  # Threshold depends on model
                    id_dropping = True

        # Check performance trends
        performance_improving = telemetry.pass_at_1 > 0.5 and telemetry.loss < 1.0

        # Determine phase
        if self.state.phase == "pre_grok":
            if id_dropping and cos_sim > 0.3:
                return "onset"
            return "pre_grok"

        elif self.state.phase == "onset":
            if cos_sim > self.cos_threshold and performance_improving:
                return "active"
            elif cos_sim < 0.2:
                return "pre_grok"
            return "onset"

        elif self.state.phase == "active":
            if telemetry.pass_at_1 > 0.9 and cos_sim > 0.8:
                return "post_grok"
            elif cos_sim < 0.3:
                return "onset"
            return "active"

        elif self.state.phase == "post_grok":
            if telemetry.pass_at_1 < 0.8:
                return "active"
            return "post_grok"

        return self.state.phase

    def _adjust_lambda(self, phase: str, cos_sim: float, telemetry: "TelemetryFrame") -> float:
        """
        Adjust lambda parameter based on phase and signals.

        Strategy:
        - pre_grok: Keep lambda low
        - onset: Gradually increase lambda
        - active: Maintain or slightly increase lambda
        - post_grok: Gradually decrease lambda
        """
        current_lambda = self.state.lambda_current

        if phase == "pre_grok":
            # Keep lambda low during normal training
            new_lambda = self.lam_init
            self.state.steps_since_increase = 0

        elif phase == "onset":
            # Gradually increase when grokking signals appear
            if cos_sim > 0.3 and telemetry.ema_grad_norm > 0:
                new_lambda = min(current_lambda * self.lam_increase_rate, self.lam_max)
                self.state.steps_since_increase = 0
            else:
                new_lambda = current_lambda
                self.state.steps_since_increase += 1

        elif phase == "active":
            # Maintain or slightly adjust during active grokking
            if cos_sim > self.cos_threshold:
                # Good alignment, can increase slightly
                new_lambda = min(current_lambda * 1.01, self.lam_max)
            else:
                # Alignment dropping, decrease slightly
                new_lambda = current_lambda * 0.99
            self.state.steps_since_increase += 1

        elif phase == "post_grok":
            # Gradually decrease after grokking
            new_lambda = max(current_lambda * self.lam_decrease_rate, self.lam_init)
            self.state.steps_since_increase += 1

        else:
            new_lambda = current_lambda

        # Apply patience mechanism
        if self.state.steps_since_increase > self.patience:
            new_lambda = max(new_lambda * self.lam_decrease_rate, self.lam_init)

        return float(new_lambda)

    def get_statistics(self) -> dict[str, float]:
        """Get controller statistics."""
        stats = {
            "current_lambda": self.state.lambda_current,
            "current_phase": self.state.phase,
            "avg_cos_sim": np.mean(self.cos_history[-100:]) if self.cos_history else 0.0,
            "activation_rate": np.mean(self.activation_history[-100:]) if self.activation_history else 0.0,
            "lambda_range": (min(self.lambda_history), max(self.lambda_history)) if self.lambda_history else (0, 0),
        }
        return stats

    def should_activate_grokfast(self) -> bool:
        """
        Determine if Grokfast should be activated based on current signals.
        """
        if self.state.phase in ["onset", "active"]:
            return True

        # Check recent cosine similarity trend
        if len(self.cos_history) > 10:
            recent_cos = self.cos_history[-10:]
            if np.mean(recent_cos) > 0.4 and np.std(recent_cos) < 0.2:
                return True

        return False

    def reset(self):
        """Reset controller state."""
        self.state = GrokfastState(lambda_current=self.lam_init, ema_alpha=self.ema_alpha)
        self.lambda_history = [self.lam_init]
        self.cos_history = []
        self.activation_history = []


class GrokfastScheduler:
    """
    Advanced scheduler for Grokfast that coordinates with other training components.
    """

    def __init__(
        self,
        controller: GrokController,
        warmup_steps: int = 1000,
        cooldown_steps: int = 500,
    ):
        self.controller = controller
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.step_count = 0
        self.grokking_start_step = None
        self.grokking_end_step = None

    def step(self, grad: torch.Tensor, telemetry: "TelemetryFrame") -> dict[str, float]:
        """
        Step the scheduler and return lambda configuration.
        """
        self.step_count += 1

        # Warmup period - no Grokfast
        if self.step_count < self.warmup_steps:
            return {"lambda": 0.0, "phase": "warmup", "active": False}

        # Normal Grokfast control
        result = self.controller.step(grad, telemetry)

        # Track grokking periods
        if result["phase"] == "onset" and self.grokking_start_step is None:
            self.grokking_start_step = self.step_count
            logger.info(f"Grokking onset detected at step {self.step_count}")

        if result["phase"] == "post_grok" and self.grokking_end_step is None:
            self.grokking_end_step = self.step_count
            logger.info(f"Grokking complete at step {self.step_count}")
            logger.info(f"Grokking duration: {self.grokking_end_step - self.grokking_start_step} steps")

        return result

    def get_metrics(self) -> dict[str, any]:
        """Get scheduler metrics."""
        metrics = self.controller.get_statistics()
        metrics.update(
            {
                "total_steps": self.step_count,
                "grokking_start": self.grokking_start_step,
                "grokking_end": self.grokking_end_step,
                "grokking_duration": (
                    self.grokking_end_step - self.grokking_start_step
                    if self.grokking_end_step and self.grokking_start_step
                    else None
                ),
            }
        )
        return metrics
