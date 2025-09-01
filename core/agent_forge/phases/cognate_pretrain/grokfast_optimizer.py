#!/usr/bin/env python3
"""
GrokFast Optimizer - Accelerated gradient descent for Cognate pretraining

Based on the GrokFast paper: Accelerated Grokking by Amplifying Slow Gradients
Implements exponential moving average (EMA) of gradients with amplification.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GrokFastOptimizer:
    """
    GrokFast optimizer that amplifies slow-varying gradient components.

    Key features:
    - Maintains EMA of gradients
    - Amplifies the slow-varying components
    - Accelerates grokking phenomenon in neural networks
    """

    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        alpha: float = 0.98,
        lamb: float = 2.0,
        warmup_steps: int = 100,
    ):
        """
        Initialize GrokFast optimizer.

        Args:
            model: The model to optimize
            base_optimizer: Base optimizer (e.g., AdamW)
            alpha: EMA decay factor (0.98 recommended)
            lamb: Amplification factor for slow gradients (2.0 recommended)
            warmup_steps: Steps before applying GrokFast
        """
        self.model = model
        self.base_optimizer = base_optimizer
        self.alpha = alpha
        self.lamb = lamb
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Initialize gradient EMA storage
        self.grad_ema = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.grad_ema[name] = torch.zeros_like(param.data)

        logger.info(f"GrokFast initialized: alpha={alpha}, lamb={lamb}, warmup={warmup_steps}")

    def step(self, closure=None):
        """Perform a single optimization step with GrokFast."""

        self.step_count += 1

        # During warmup, just use base optimizer
        if self.step_count <= self.warmup_steps:
            return self.base_optimizer.step(closure)

        # Apply GrokFast gradient amplification
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Update EMA of gradients
                    self.grad_ema[name] = self.alpha * self.grad_ema[name] + (1 - self.alpha) * param.grad

                    # Amplify slow-varying component
                    # grad_new = grad + lamb * grad_ema
                    param.grad = param.grad + self.lamb * self.grad_ema[name]

        # Step with base optimizer using amplified gradients
        return self.base_optimizer.step(closure)

    def zero_grad(self):
        """Zero gradients in base optimizer."""
        self.base_optimizer.zero_grad()

    def state_dict(self):
        """Return state dict for checkpointing."""
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "grad_ema": self.grad_ema,
            "step_count": self.step_count,
            "alpha": self.alpha,
            "lamb": self.lamb,
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.grad_ema = state_dict["grad_ema"]
        self.step_count = state_dict["step_count"]
        self.alpha = state_dict.get("alpha", self.alpha)
        self.lamb = state_dict.get("lamb", self.lamb)


def create_grokfast_adamw(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    alpha: float = 0.98,
    lamb: float = 2.0,
    warmup_steps: int = 100,
) -> GrokFastOptimizer:
    """
    Create a GrokFast-enhanced AdamW optimizer.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay for AdamW
        betas: Adam betas
        alpha: GrokFast EMA decay
        lamb: GrokFast amplification factor
        warmup_steps: Warmup before GrokFast

    Returns:
        GrokFastOptimizer instance
    """
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    return GrokFastOptimizer(
        model=model, base_optimizer=base_optimizer, alpha=alpha, lamb=lamb, warmup_steps=warmup_steps
    )


class GrokFastScheduler:
    """Learning rate scheduler compatible with GrokFast."""

    def __init__(
        self, optimizer: GrokFastOptimizer, warmup_steps: int = 100, total_steps: int = 10000, min_lr_ratio: float = 0.1
    ):
        """
        Initialize scheduler for GrokFast.

        Args:
            optimizer: GrokFast optimizer
            warmup_steps: Linear warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum LR as ratio of initial
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # Get initial learning rate
        self.base_lr = optimizer.base_optimizer.param_groups[0]["lr"]
        self.min_lr = self.base_lr * min_lr_ratio

    def step(self, step: int):
        """Update learning rate based on step."""
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine decay after warmup
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        # Update learning rate
        for param_group in self.optimizer.base_optimizer.param_groups:
            param_group["lr"] = lr

        return lr
