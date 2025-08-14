"""
GrokfastAdamW optimizer implementation.
Amplifies slow-changing gradients to accelerate grokking by up to 50x.
"""

import logging
from collections.abc import Callable

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class GrokfastAdamW(Optimizer):
    """
    AdamW optimizer enhanced with Grokfast gradient filtering.

    Grokfast works by maintaining an EMA of gradients and amplifying
    components that change slowly (high alignment with EMA) while
    dampening fast-changing components.

    Based on: "Grokfast: Accelerated Grokking by Amplifying Slow Gradients"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        ema_alpha: float = 0.98,
        grokfast_lambda: float = 0.05,
        grokfast_enabled: bool = True,
    ):
        """
        Args:
            params: Model parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (AdamW)
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient (AdamW)
            ema_alpha: EMA decay factor for Grokfast
            grokfast_lambda: Initial amplification factor
            grokfast_enabled: Whether to apply Grokfast filtering
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError(f"Invalid ema_alpha value: {ema_alpha}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ema_alpha=ema_alpha,
            grokfast_lambda=grokfast_lambda,
            grokfast_enabled=grokfast_enabled,
        )
        super().__init__(params, defaults)

        # Initialize Grokfast EMA buffers
        self._init_grokfast_buffers()

    def _init_grokfast_buffers(self):
        """Initialize EMA gradient buffers for Grokfast."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    # Initialize Grokfast EMA
                    state["ema_grad"] = torch.zeros_like(p.data)
                    state["grokfast_initialized"] = False

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        grokfast_lambda_override: float | None = None,
    ):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss
            grokfast_lambda_override: Override lambda for this step (from controller)

        Returns:
            Optional loss value from closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            ema_alpha = group["ema_alpha"]
            base_lambda = grokfast_lambda_override or group["grokfast_lambda"]
            grokfast_enabled = group["grokfast_enabled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply Grokfast filtering if enabled
                if grokfast_enabled and base_lambda > 0:
                    grad = self._apply_grokfast(grad, p, ema_alpha, base_lambda)

                # Standard AdamW update
                state = self.state[p]

                # State initialization
                if len(state) == 0 or "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute step size
                step_size = group["lr"] / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(group["eps"])

                # Update parameters
                p.data.add_(exp_avg / denom, alpha=-step_size)

                # Add weight decay (AdamW style - decoupled)
                if group["weight_decay"] > 0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss

    def _apply_grokfast(
        self,
        grad: torch.Tensor,
        param: torch.nn.Parameter,
        ema_alpha: float,
        grokfast_lambda: float,
    ) -> torch.Tensor:
        """
        Apply Grokfast gradient filtering.

        The key insight: amplify gradient components that align with the EMA
        (slow-changing) and dampen those that don't (fast-changing).
        """
        state = self.state[param]

        # Update EMA of gradients
        if not state.get("grokfast_initialized", False):
            state["ema_grad"] = grad.clone()
            state["grokfast_initialized"] = True
        else:
            state["ema_grad"].mul_(ema_alpha).add_(grad, alpha=1 - ema_alpha)

        # Compute filtered gradient
        # g_filtered = g + λ * (g · ema_g / ||ema_g||²) * ema_g
        ema_grad = state["ema_grad"]

        # Avoid division by zero
        ema_norm_sq = torch.sum(ema_grad * ema_grad)
        if ema_norm_sq > 1e-10:
            # Project gradient onto EMA direction
            projection_scalar = torch.sum(grad * ema_grad) / ema_norm_sq

            # Amplify component in EMA direction
            filtered_grad = grad + grokfast_lambda * projection_scalar * ema_grad
        else:
            filtered_grad = grad

        return filtered_grad

    def set_grokfast_lambda(self, grokfast_lambda: float):
        """Update Grokfast lambda for all parameter groups."""
        for group in self.param_groups:
            group["grokfast_lambda"] = grokfast_lambda

    def enable_grokfast(self, enabled: bool = True):
        """Enable or disable Grokfast filtering."""
        for group in self.param_groups:
            group["grokfast_enabled"] = enabled

    def get_grokfast_stats(self) -> dict[str, float]:
        """Get statistics about Grokfast operation."""
        total_params = 0
        total_ema_norm = 0.0
        total_grad_ema_alignment = 0.0
        num_params_with_grad = 0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    state = self.state[p]
                    if "ema_grad" in state:
                        ema_grad = state["ema_grad"]
                        grad = p.grad

                        # Compute alignment (cosine similarity)
                        grad_norm = torch.norm(grad)
                        ema_norm = torch.norm(ema_grad)

                        if grad_norm > 1e-10 and ema_norm > 1e-10:
                            alignment = torch.sum(grad * ema_grad) / (grad_norm * ema_norm)
                            total_grad_ema_alignment += alignment.item()
                            num_params_with_grad += 1

                        total_ema_norm += ema_norm.item()

                total_params += p.numel()

        avg_alignment = total_grad_ema_alignment / num_params_with_grad if num_params_with_grad > 0 else 0.0

        return {
            "total_params": total_params,
            "params_with_grad": num_params_with_grad,
            "avg_ema_norm": total_ema_norm / max(num_params_with_grad, 1),
            "avg_grad_ema_alignment": avg_alignment,
            "grokfast_enabled": self.param_groups[0]["grokfast_enabled"],
            "grokfast_lambda": self.param_groups[0]["grokfast_lambda"],
        }


class GrokfastSGD(Optimizer):
    """
    SGD optimizer with Grokfast gradient filtering.
    Simpler alternative to GrokfastAdamW for memory-constrained settings.
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        ema_alpha: float = 0.98,
        grokfast_lambda: float = 0.05,
        grokfast_enabled: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ema_alpha=ema_alpha,
            grokfast_lambda=grokfast_lambda,
            grokfast_enabled=grokfast_enabled,
        )
        super().__init__(params, defaults)

        # Initialize buffers
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    state["ema_grad"] = torch.zeros_like(p.data)
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        grokfast_lambda_override: float | None = None,
    ):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ema_alpha = group["ema_alpha"]
            base_lambda = grokfast_lambda_override or group["grokfast_lambda"]
            grokfast_enabled = group["grokfast_enabled"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Apply Grokfast
                if grokfast_enabled and base_lambda > 0:
                    state = self.state[p]

                    # Update EMA
                    if "ema_grad" not in state:
                        state["ema_grad"] = grad.clone()
                    else:
                        state["ema_grad"].mul_(ema_alpha).add_(grad, alpha=1 - ema_alpha)

                    # Filter gradient
                    ema_grad = state["ema_grad"]
                    ema_norm_sq = torch.sum(ema_grad * ema_grad)

                    if ema_norm_sq > 1e-10:
                        projection = torch.sum(grad * ema_grad) / ema_norm_sq
                        grad = grad + base_lambda * projection * ema_grad

                # Apply momentum
                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.data)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    grad = buf

                # Update parameters
                p.data.add_(grad, alpha=-group["lr"])

        return loss
