#!/usr/bin/env python3
"""
GrokFast Optimizer Integration for Cognate Model

This module implements the GrokFast optimization technique, which has been shown to
significantly accelerate the "grokking" phenomenon where models suddenly improve
after long plateaus. This is particularly relevant for the Cognate model's 
long-term memory and reasoning capabilities.

GrokFast works by maintaining exponential moving averages of gradients and 
applying a momentum-based update that helps overcome local minima.

References:
- GrokFast paper: https://arxiv.org/abs/2405.20233
- Original implementation adaptations for transformer architectures
"""

from dataclasses import dataclass
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class GrokFastConfig:
    """Configuration for GrokFast optimizer."""

    alpha: float = 0.98  # EMA decay factor for gradient averaging
    lamb: float = 2.0  # Amplification factor for gradient updates

    # Advanced parameters
    eps: float = 1e-8  # Numerical stability constant
    weight_decay: float = 0.0  # Weight decay (usually handled by base optimizer)

    # Adaptive parameters
    adaptive_lambda: bool = False  # Whether to adapt lambda based on training dynamics
    lambda_min: float = 1.0  # Minimum lambda value
    lambda_max: float = 5.0  # Maximum lambda value

    # Memory efficiency
    memory_efficient: bool = True  # Use memory-efficient implementation

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"Alpha must be in (0, 1), got {self.alpha}")

        if self.lamb < 0.0:
            raise ValueError(f"Lambda must be >= 0, got {self.lamb}")

        if self.adaptive_lambda and self.lambda_min >= self.lambda_max:
            raise ValueError("lambda_min must be < lambda_max for adaptive lambda")


class GrokFastOptimizer(Optimizer):
    """
    GrokFast optimizer wrapper.

    This optimizer wraps a base optimizer (e.g., AdamW) and applies the GrokFast
    technique to accelerate convergence and help overcome plateaus.

    The key insight of GrokFast is to maintain exponential moving averages of
    gradients and use them to amplify updates in directions where gradients
    have been consistently pointing.
    """

    def __init__(
        self,
        base_optimizer: Optimizer,
        alpha: float = 0.98,
        lamb: float = 2.0,
        eps: float = 1e-8,
        adaptive_lambda: bool = False,
        lambda_min: float = 1.0,
        lambda_max: float = 5.0,
        memory_efficient: bool = True,
    ):
        """
        Initialize GrokFast optimizer.

        Args:
            base_optimizer: Underlying optimizer (e.g., AdamW)
            alpha: EMA decay factor for gradient averaging
            lamb: Amplification factor for gradient updates
            eps: Numerical stability constant
            adaptive_lambda: Whether to adapt lambda based on training dynamics
            lambda_min: Minimum lambda value (for adaptive mode)
            lambda_max: Maximum lambda value (for adaptive mode)
            memory_efficient: Use memory-efficient implementation
        """
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.lamb = lamb
        self.eps = eps
        self.adaptive_lambda = adaptive_lambda
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.memory_efficient = memory_efficient

        # Get parameter groups from base optimizer
        self.param_groups = base_optimizer.param_groups

        # Initialize state
        self.state = {}

        # Tracking for adaptive lambda
        self.step_count = 0
        self.loss_history = []
        self.gradient_norm_history = []

        logger.info(
            f"Initialized GrokFast optimizer: alpha={alpha}, lamb={lamb}, " f"adaptive_lambda={adaptive_lambda}"
        )

    def __getstate__(self):
        """Get state for serialization."""
        return {
            "base_optimizer": self.base_optimizer,
            "alpha": self.alpha,
            "lamb": self.lamb,
            "eps": self.eps,
            "adaptive_lambda": self.adaptive_lambda,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "memory_efficient": self.memory_efficient,
            "state": self.state,
            "step_count": self.step_count,
            "loss_history": self.loss_history,
            "gradient_norm_history": self.gradient_norm_history,
        }

    def __setstate__(self, state):
        """Set state from deserialization."""
        self.__dict__.update(state)

    def state_dict(self):
        """Get state dictionary."""
        return {
            "base_optimizer_state": self.base_optimizer.state_dict(),
            "grokfast_state": {
                "alpha": self.alpha,
                "lamb": self.lamb,
                "eps": self.eps,
                "adaptive_lambda": self.adaptive_lambda,
                "lambda_min": self.lambda_min,
                "lambda_max": self.lambda_max,
                "memory_efficient": self.memory_efficient,
                "state": self.state,
                "step_count": self.step_count,
                "loss_history": self.loss_history[-100:],  # Keep last 100 entries
                "gradient_norm_history": self.gradient_norm_history[-100:],
            },
        }

    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        if "base_optimizer_state" in state_dict:
            self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])

        if "grokfast_state" in state_dict:
            gf_state = state_dict["grokfast_state"]
            self.alpha = gf_state.get("alpha", self.alpha)
            self.lamb = gf_state.get("lamb", self.lamb)
            self.eps = gf_state.get("eps", self.eps)
            self.adaptive_lambda = gf_state.get("adaptive_lambda", self.adaptive_lambda)
            self.lambda_min = gf_state.get("lambda_min", self.lambda_min)
            self.lambda_max = gf_state.get("lambda_max", self.lambda_max)
            self.memory_efficient = gf_state.get("memory_efficient", self.memory_efficient)
            self.state = gf_state.get("state", {})
            self.step_count = gf_state.get("step_count", 0)
            self.loss_history = gf_state.get("loss_history", [])
            self.gradient_norm_history = gf_state.get("gradient_norm_history", [])

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        """Add parameter group to base optimizer."""
        self.base_optimizer.add_param_group(param_group)

    def step(self, closure=None):
        """
        Perform a single optimization step with GrokFast.

        Args:
            closure: Optional closure for re-evaluating the model

        Returns:
            Optional loss value if closure is provided
        """
        # Compute loss if closure provided
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Apply GrokFast gradient modification
        self._apply_grokfast()

        # Step the base optimizer
        base_loss = self.base_optimizer.step()

        # Update step count and tracking
        self.step_count += 1

        # Track loss and gradient norms for adaptive lambda
        if loss is not None and self.adaptive_lambda:
            self.loss_history.append(loss.item() if torch.is_tensor(loss) else loss)

            # Compute gradient norm
            grad_norm = self._compute_gradient_norm()
            self.gradient_norm_history.append(grad_norm)

            # Update lambda adaptively
            self._update_adaptive_lambda()

        return loss if loss is not None else base_loss

    def _apply_grokfast(self):
        """Apply GrokFast gradient transformation."""
        current_lambda = self.lamb

        # Iterate through parameter groups
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                # Get parameter state
                param_key = (group_idx, param_idx)
                if param_key not in self.state:
                    self.state[param_key] = {
                        "grad_ema": torch.zeros_like(param.grad),
                        "step": 0,
                    }

                state = self.state[param_key]
                grad_ema = state["grad_ema"]
                step = state["step"]

                # Update exponential moving average of gradients
                # grad_ema = alpha * grad_ema + (1 - alpha) * grad
                grad_ema.mul_(self.alpha).add_(param.grad, alpha=1 - self.alpha)

                # Apply GrokFast transformation
                # grad = grad + lambda * grad_ema
                if step > 0:  # Skip first step to initialize EMA properly
                    param.grad.add_(grad_ema, alpha=current_lambda)

                # Update state
                state["step"] = step + 1

                # Memory efficient: detach EMA to prevent gradient accumulation
                if self.memory_efficient:
                    state["grad_ema"] = grad_ema.detach()

    def _compute_gradient_norm(self) -> float:
        """Compute L2 norm of gradients across all parameters."""
        total_norm = 0.0

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2

        return total_norm**0.5

    def _update_adaptive_lambda(self):
        """Update lambda based on training dynamics."""
        if not self.adaptive_lambda or len(self.loss_history) < 10:
            return

        # Look at recent loss trend
        recent_losses = self.loss_history[-10:]
        loss_trend = recent_losses[-1] - recent_losses[0]  # Negative = improving

        # Look at recent gradient norm trend
        recent_grad_norms = self.gradient_norm_history[-10:]
        recent_grad_norms[-1] - recent_grad_norms[0]

        # Adaptive strategy:
        # - If loss is plateauing (small negative trend) and gradients are small,
        #   increase lambda to amplify updates
        # - If loss is decreasing well, keep lambda moderate
        # - If gradients are exploding, decrease lambda

        if abs(loss_trend) < 0.01 and recent_grad_norms[-1] < 1.0:
            # Plateauing with small gradients - increase lambda
            self.lamb = min(self.lambda_max, self.lamb * 1.05)
        elif loss_trend < -0.1:
            # Good improvement - keep lambda stable
            pass
        elif recent_grad_norms[-1] > 10.0:
            # Large gradients - decrease lambda
            self.lamb = max(self.lambda_min, self.lamb * 0.95)

        # Periodic logging
        if self.step_count % 100 == 0:
            logger.debug(
                f"Adaptive lambda update: step={self.step_count}, "
                f"lambda={self.lamb:.3f}, loss_trend={loss_trend:.4f}, "
                f"grad_norm={recent_grad_norms[-1]:.2f}"
            )

    def get_grokfast_stats(self) -> dict[str, Any]:
        """Get GrokFast optimizer statistics."""
        stats = {
            "current_lambda": self.lamb,
            "alpha": self.alpha,
            "step_count": self.step_count,
            "adaptive_lambda": self.adaptive_lambda,
        }

        if self.loss_history:
            stats.update(
                {
                    "recent_loss": self.loss_history[-1],
                    "loss_trend_10": (
                        self.loss_history[-1] - self.loss_history[-10] if len(self.loss_history) >= 10 else 0.0
                    ),
                }
            )

        if self.gradient_norm_history:
            stats.update(
                {
                    "recent_grad_norm": self.gradient_norm_history[-1],
                    "avg_grad_norm": sum(self.gradient_norm_history[-10:]) / min(10, len(self.gradient_norm_history)),
                }
            )

        return stats

    def set_lambda(self, new_lambda: float):
        """Manually set lambda value."""
        if new_lambda < 0:
            raise ValueError("Lambda must be >= 0")

        old_lambda = self.lamb
        self.lamb = new_lambda
        logger.info(f"Lambda updated from {old_lambda:.3f} to {new_lambda:.3f}")

    def reset_ema_state(self):
        """Reset the EMA state (useful for transfer learning or fine-tuning)."""
        logger.info("Resetting GrokFast EMA state")
        for param_key in self.state:
            if "grad_ema" in self.state[param_key]:
                self.state[param_key]["grad_ema"].zero_()
                self.state[param_key]["step"] = 0


def create_grokfast_optimizer(
    model: nn.Module,
    base_optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    grokfast_config: GrokFastConfig | None = None,
    **kwargs,
) -> GrokFastOptimizer:
    """
    Factory function to create a GrokFast optimizer with a base optimizer.

    Args:
        model: Model to optimize
        base_optimizer_type: Type of base optimizer ("adamw", "adam", "sgd")
        learning_rate: Learning rate for base optimizer
        weight_decay: Weight decay for base optimizer
        betas: Beta parameters for Adam-based optimizers
        grokfast_config: GrokFast configuration
        **kwargs: Additional arguments for base optimizer

    Returns:
        GrokFastOptimizer: Configured GrokFast optimizer
    """
    if grokfast_config is None:
        grokfast_config = GrokFastConfig()

    # Create base optimizer
    if base_optimizer_type.lower() == "adamw":
        from torch.optim import AdamW

        base_optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, **kwargs)
    elif base_optimizer_type.lower() == "adam":
        from torch.optim import Adam

        base_optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, **kwargs)
    elif base_optimizer_type.lower() == "sgd":
        from torch.optim import SGD

        base_optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    else:
        raise ValueError(f"Unsupported base optimizer type: {base_optimizer_type}")

    # Create GrokFast optimizer
    grokfast_optimizer = GrokFastOptimizer(
        base_optimizer=base_optimizer,
        alpha=grokfast_config.alpha,
        lamb=grokfast_config.lamb,
        eps=grokfast_config.eps,
        adaptive_lambda=grokfast_config.adaptive_lambda,
        lambda_min=grokfast_config.lambda_min,
        lambda_max=grokfast_config.lambda_max,
        memory_efficient=grokfast_config.memory_efficient,
    )

    return grokfast_optimizer


if __name__ == "__main__":
    # Test GrokFast optimizer
    import logging

    import torch.nn as nn

    logging.basicConfig(level=logging.INFO)

    print("Testing GrokFast Optimizer...")

    # Create simple model
    model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 1))

    # Create GrokFast optimizer
    optimizer = create_grokfast_optimizer(
        model, learning_rate=0.01, grokfast_config=GrokFastConfig(alpha=0.95, lamb=1.5, adaptive_lambda=True)
    )

    # Simulate training steps
    for step in range(100):
        # Dummy forward pass
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)

        optimizer.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        optimizer.step()

        # Log progress
        if step % 20 == 0:
            stats = optimizer.get_grokfast_stats()
            print(f"Step {step}: loss={loss:.4f}, lambda={stats['current_lambda']:.3f}")

    print("✅ GrokFast optimizer test completed!")

    # Test state dict save/load
    state_dict = optimizer.state_dict()

    # Create new optimizer and load state
    new_optimizer = create_grokfast_optimizer(model, learning_rate=0.01)
    new_optimizer.load_state_dict(state_dict)

    print("✅ State dict save/load test passed!")
