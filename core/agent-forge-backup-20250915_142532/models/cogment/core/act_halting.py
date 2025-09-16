"""Adaptive Computation Time (ACT) halting mechanism for Cogment."""

from __future__ import annotations

import torch
import torch.nn as nn


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting mechanism following Graves et al. 2016.

    Accumulates halting probabilities until they exceed threshold (1-ε),
    then computes weighted average of all outputs up to that point.
    """

    def __init__(self, threshold: float = 0.99, epsilon: float = 0.01):
        super().__init__()
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(
        self,
        halt_probs: torch.Tensor,  # [B, T, 1] - halting probabilities at each step
        outputs: torch.Tensor,  # [B, T, ...] - outputs at each step
        step_weights: torch.Tensor | None = None,  # [B, T] - optional step weights
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply ACT halting with weighted averaging.

        Args:
            halt_probs: Halting probabilities [B, T, 1]
            outputs: Model outputs [B, T, D] where D is output dimension
            step_weights: Optional weights for each step [B, T]

        Returns:
            final_output: Weighted average output [B, D]
            ponder_cost: Average number of steps taken [B]
            halt_weights: Actual weights used for averaging [B, T]
        """
        B, T = halt_probs.shape[:2]
        device = halt_probs.device

        # Squeeze halt_probs to [B, T]
        halt_p = halt_probs.squeeze(-1)  # [B, T]

        # Cumulative halt probabilities
        cum_halt = torch.cumsum(halt_p, dim=1)  # [B, T]

        # Find first step where cumulative probability exceeds threshold
        exceeds_threshold = cum_halt >= self.threshold  # [B, T]

        # Find halt step for each batch element (first True position)
        halt_steps = torch.zeros(B, dtype=torch.long, device=device)
        for b in range(B):
            halt_positions = torch.where(exceeds_threshold[b])[0]
            if len(halt_positions) > 0:
                halt_steps[b] = halt_positions[0]
            else:
                halt_steps[b] = T - 1  # Force halt at last step

        # Compute actual weights for weighted averaging
        weights = torch.zeros(B, T, device=device)

        for b in range(B):
            halt_step = halt_steps[b].item()

            # All steps before halt get their halt probability as weight
            if halt_step > 0:
                weights[b, :halt_step] = halt_p[b, :halt_step]

            # Halt step gets remaining probability to sum to 1
            remaining_prob = 1.0 - cum_halt[b, halt_step].item() + halt_p[b, halt_step].item()
            weights[b, halt_step] = remaining_prob

            # Steps after halt get zero weight
            # (already zero by initialization)

        # Apply step weights if provided
        if step_weights is not None:
            weights = weights * step_weights

        # Normalize weights to sum to 1 for each batch element
        weight_sums = weights.sum(dim=1, keepdim=True)  # [B, 1]
        weight_sums = torch.clamp(weight_sums, min=self.epsilon)  # Avoid division by zero
        weights = weights / weight_sums  # [B, T]

        # Compute weighted average output
        # weights: [B, T], outputs: [B, T, D] -> final_output: [B, D]
        weights_expanded = weights.unsqueeze(-1)  # [B, T, 1]
        final_output = torch.sum(weights_expanded * outputs, dim=1)  # [B, D]

        # Compute ponder cost (average number of computation steps)
        ponder_cost = torch.sum(weights * torch.arange(1, T + 1, device=device).float(), dim=1)  # [B]

        return final_output, ponder_cost, weights


class ACTLoss(nn.Module):
    """Loss function combining task loss with ponder cost regularization."""

    def __init__(self, ponder_weight: float = 0.1):
        super().__init__()
        self.ponder_weight = ponder_weight

    def forward(
        self,
        task_loss: torch.Tensor,  # [B] or scalar - primary task loss
        ponder_cost: torch.Tensor,  # [B] - average computation steps
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss with ponder cost regularization.

        Args:
            task_loss: Primary task loss
            ponder_cost: Average number of computation steps

        Returns:
            total_loss: Combined loss
            task_loss_item: Task loss component
            ponder_loss_item: Ponder cost component
        """
        # Ensure both losses have the same shape
        if task_loss.dim() == 0:  # Scalar
            task_loss = task_loss.expand_as(ponder_cost)

        # Ponder loss is just the average number of steps
        ponder_loss = ponder_cost.mean()

        # Combined loss
        total_loss = task_loss.mean() + self.ponder_weight * ponder_loss

        return total_loss, task_loss.mean(), ponder_loss


def compute_halt_mask(
    halt_steps: torch.Tensor,  # [B] - halt step for each batch element
    max_steps: int,  # Maximum number of steps
) -> torch.Tensor:
    """
    Compute mask for valid computation steps.

    Args:
        halt_steps: Step at which each sequence halted [B]
        max_steps: Maximum number of steps

    Returns:
        mask: Binary mask [B, T] where 1 indicates valid computation
    """
    halt_steps.size(0)
    device = halt_steps.device

    # Create step indices
    step_indices = torch.arange(max_steps, device=device).unsqueeze(0)  # [1, T]
    halt_steps_expanded = halt_steps.unsqueeze(1)  # [B, 1]

    # Mask is 1 for steps <= halt_step
    mask = step_indices <= halt_steps_expanded  # [B, T]

    return mask.float()
