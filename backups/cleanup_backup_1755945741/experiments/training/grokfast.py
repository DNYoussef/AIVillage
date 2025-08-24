"""
GrokFast Implementation for Accelerated Grokking

Implements the GrokFast algorithm for 50x acceleration of grokking by amplifying
slow gradients while dampening fast gradients. Based on the GrokFast paper.
"""

from collections import defaultdict, deque
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GrokFastOptimizer:
    """
    GrokFast gradient filtering optimizer wrapper.

    Wraps any PyTorch optimizer to apply GrokFast gradient filtering,
    which amplifies slow gradients and dampens fast gradients to accelerate grokking.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        alpha: float = 0.98,
        lamb: float = 2.0,
        window_size: int = 100,
    ):
        """
        Initialize GrokFast optimizer.

        Args:
            base_optimizer: Base PyTorch optimizer (e.g., AdamW)
            model: The model to optimize
            alpha: EMA decay factor for gradient filtering (0.95-0.99)
            lamb: Amplification factor for slow gradients (1.0-5.0)
            window_size: Window size for gradient statistics
        """
        self.base_optimizer = base_optimizer
        self.model = model
        self.alpha = alpha
        self.lamb = lamb
        self.window_size = window_size

        # Gradient history for each parameter
        self.gradient_ema: dict[str, torch.Tensor] = {}
        self.gradient_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Initialize EMA for all parameters
        self._initialize_ema()

        # Statistics tracking
        self.step_count = 0
        self.grokking_detected = False
        self.gradient_stats = {"slow_count": 0, "fast_count": 0, "amplification_ratio": 1.0}

        logger.info(f"Initialized GrokFast optimizer with alpha={alpha}, lambda={lamb}")

    def _initialize_ema(self):
        """Initialize exponential moving average for all parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_ema[name] = torch.zeros_like(param.data)

    def _update_gradient_ema(self, name: str, gradient: torch.Tensor):
        """Update exponential moving average for gradient."""
        if name not in self.gradient_ema:
            self.gradient_ema[name] = torch.zeros_like(gradient)

        # Update EMA: ema = alpha * ema + (1 - alpha) * gradient
        self.gradient_ema[name] = self.alpha * self.gradient_ema[name] + (1 - self.alpha) * gradient

    def _classify_gradient(self, name: str, gradient: torch.Tensor) -> str:
        """
        Classify gradient as slow or fast based on EMA comparison.

        Args:
            name: Parameter name
            gradient: Current gradient

        Returns:
            'slow' or 'fast'
        """
        if name not in self.gradient_ema:
            return "fast"  # Default for first step

        ema_gradient = self.gradient_ema[name]

        # Calculate alignment between current gradient and EMA
        current_norm = torch.norm(gradient)
        ema_norm = torch.norm(ema_gradient)

        if current_norm > 0 and ema_norm > 0:
            # Cosine similarity between current and EMA gradients
            cosine_sim = torch.dot(gradient.flatten(), ema_gradient.flatten()) / (current_norm * ema_norm)

            # Slow gradients are aligned with the trend (high cosine similarity)
            # Fast gradients are orthogonal to the trend (low cosine similarity)
            if cosine_sim > 0.5:  # Threshold for slow/fast classification
                return "slow"
            else:
                return "fast"
        else:
            return "fast"

    def _apply_grokfast_filtering(self):
        """Apply GrokFast gradient filtering to all parameters."""
        slow_count = 0
        fast_count = 0
        total_amplification = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient = param.grad.data

                # Update gradient EMA
                self._update_gradient_ema(name, gradient)

                # Classify gradient as slow or fast
                gradient_type = self._classify_gradient(name, gradient)

                if gradient_type == "slow":
                    # Amplify slow gradients
                    param.grad.data = gradient * self.lamb
                    slow_count += 1
                    total_amplification += self.lamb
                else:
                    # Dampening fast gradients (multiply by 1/lamb)
                    param.grad.data = gradient / self.lamb
                    fast_count += 1
                    total_amplification += 1.0 / self.lamb

                # Store gradient history for analysis
                self.gradient_history[name].append(
                    {"gradient_norm": torch.norm(gradient).item(), "type": gradient_type, "step": self.step_count}
                )

        # Update statistics
        self.gradient_stats["slow_count"] = slow_count
        self.gradient_stats["fast_count"] = fast_count
        if slow_count + fast_count > 0:
            self.gradient_stats["amplification_ratio"] = total_amplification / (slow_count + fast_count)

    def step(self, amplify: bool = True):
        """
        Perform optimization step with optional GrokFast filtering.

        Args:
            amplify: Whether to apply GrokFast amplification
        """
        if amplify:
            self._apply_grokfast_filtering()

        # Perform base optimizer step
        self.base_optimizer.step()
        self.step_count += 1

        # Check for grokking onset
        if self.step_count % 50 == 0:
            self._detect_grokking()

    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()

    def _detect_grokking(self):
        """Detect onset of grokking behavior."""
        # Simple heuristic: grokking when most gradients become "slow"
        total_gradients = self.gradient_stats["slow_count"] + self.gradient_stats["fast_count"]
        if total_gradients > 0:
            slow_ratio = self.gradient_stats["slow_count"] / total_gradients

            if slow_ratio > 0.7 and not self.grokking_detected:
                self.grokking_detected = True
                logger.info(f"Grokking detected at step {self.step_count} (slow ratio: {slow_ratio:.3f})")

    def get_statistics(self) -> dict:
        """Get current GrokFast statistics."""
        return {
            "step_count": self.step_count,
            "grokking_detected": self.grokking_detected,
            "gradient_stats": self.gradient_stats.copy(),
            "alpha": self.alpha,
            "lambda": self.lamb,
        }


class GrokFastTask:
    """
    High-level interface for GrokFast training tasks.

    Provides a simple interface for integrating GrokFast optimization
    into existing training loops with minimal code changes.
    """

    def __init__(
        self,
        agent: Optional = None,  # Langroid ChatAgent (optional)
        model: nn.Module | None = None,
        alpha: float = 0.98,
        lamb: float = 2.0,
    ):
        """
        Initialize GrokFast task.

        Args:
            agent: Optional Langroid ChatAgent for integration
            model: PyTorch model to optimize
            alpha: EMA decay factor
            lamb: Amplification factor
        """
        self.agent = agent
        self.model = model
        self.alpha = alpha
        self.lamb = lamb

        self.grokfast_optimizer: GrokFastOptimizer | None = None
        self.training_metrics = []

        logger.info("Initialized GrokFast task")

    def setup_optimizer(self, base_optimizer: torch.optim.Optimizer) -> GrokFastOptimizer:
        """
        Setup GrokFast optimizer wrapper.

        Args:
            base_optimizer: Base PyTorch optimizer

        Returns:
            GrokFast optimizer wrapper
        """
        if self.model is None:
            raise ValueError("Model must be provided to setup optimizer")

        self.grokfast_optimizer = GrokFastOptimizer(
            base_optimizer=base_optimizer, model=self.model, alpha=self.alpha, lamb=self.lamb
        )

        return self.grokfast_optimizer

    async def filter_gradients(self):
        """Apply GrokFast gradient filtering (async compatible)."""
        if self.grokfast_optimizer is None:
            logger.warning("GrokFast optimizer not initialized")
            return

        # The filtering is applied automatically in the step() method
        # This method is kept for compatibility with existing code
        pass

    def run(self, *args, **kwargs):
        """
        Run GrokFast optimization.

        Returns:
            Status dictionary with metrics
        """
        if self.grokfast_optimizer is None:
            return {"status": "error", "message": "GrokFast optimizer not initialized"}

        stats = self.grokfast_optimizer.get_statistics()

        return {
            "status": "success",
            "message": "GrokFast optimization active",
            "statistics": stats,
            "grokking_detected": stats["grokking_detected"],
        }

    def get_metrics(self) -> dict:
        """Get training metrics and GrokFast statistics."""
        if self.grokfast_optimizer is None:
            return {"error": "Optimizer not initialized"}

        return self.grokfast_optimizer.get_statistics()


# Utility functions for GrokFast integration
def create_grokfast_adamw(
    model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.01, alpha: float = 0.98, lamb: float = 2.0
) -> GrokFastOptimizer:
    """
    Create GrokFast-enabled AdamW optimizer.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        alpha: GrokFast EMA decay
        lamb: GrokFast amplification factor

    Returns:
        GrokFast optimizer wrapper
    """
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    return GrokFastOptimizer(base_optimizer=base_optimizer, model=model, alpha=alpha, lamb=lamb)


def analyze_grokking_dynamics(grokfast_optimizer: GrokFastOptimizer) -> dict:
    """
    Analyze grokking dynamics from GrokFast optimizer.

    Args:
        grokfast_optimizer: GrokFast optimizer with training history

    Returns:
        Analysis of grokking dynamics
    """
    gradient_history = grokfast_optimizer.gradient_history

    if not gradient_history:
        return {"error": "No gradient history available"}

    # Analyze gradient trends across parameters
    slow_trends = []
    fast_trends = []

    for param_name, history in gradient_history.items():
        if len(history) > 10:  # Need sufficient history
            recent_history = list(history)[-50:]  # Last 50 steps

            slow_count = sum(1 for h in recent_history if h["type"] == "slow")
            slow_ratio = slow_count / len(recent_history)

            if slow_ratio > 0.7:
                slow_trends.append(param_name)
            elif slow_ratio < 0.3:
                fast_trends.append(param_name)

    return {
        "total_parameters": len(gradient_history),
        "slow_trending_params": len(slow_trends),
        "fast_trending_params": len(fast_trends),
        "slow_param_names": slow_trends[:5],  # Top 5
        "fast_param_names": fast_trends[:5],  # Top 5
        "grokking_progress": len(slow_trends) / len(gradient_history) if gradient_history else 0.0,
    }
