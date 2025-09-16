"""
Loss Functions for Cogment Training.

Implements specialized loss functions for iterative refinement training:
1. Deep Supervision: Loss at each refinement step with weight decay
2. Residual Improvement: Penalizes steps that don't improve predictions
3. Consistency Loss: Augmentation-equivariance for ARC-style tasks
4. Ponder Loss: ACT expected steps with scheduled ponder cost
5. Combined Loss: Weighted combination with stage-specific scheduling
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss for iterative refinement.

    Applies supervision at each refinement step with exponential weight decay.
    Encourages early steps to produce reasonable predictions while allowing
    later steps to make fine-grained corrections.
    """

    def __init__(
        self, base_weight: float = 1.0, decay_factor: float = 0.5, min_weight: float = 0.1, ignore_index: int = -100
    ):
        super().__init__()
        self.base_weight = base_weight
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.ignore_index = ignore_index

    def forward(
        self,
        step_logits: list[torch.Tensor],  # List of [B, N, vocab_size] for each step
        targets: torch.Tensor,  # [B, N] target tokens
        step_weights: list[float] | None = None,  # Optional custom weights per step
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute deep supervision loss across all refinement steps.

        Args:
            step_logits: Predictions at each refinement step
            targets: Target token sequences
            step_weights: Optional custom weights per step

        Returns:
            total_loss: Weighted sum of step losses
            step_info: Individual step losses and weights
        """
        if not step_logits:
            raise ValueError("step_logits cannot be empty")

        num_steps = len(step_logits)
        step_losses = []
        weights = []

        for step_idx, logits in enumerate(step_logits):
            # Compute cross-entropy loss for this step
            step_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.ignore_index, reduction="mean"
            )
            step_losses.append(step_loss)

            # Compute weight for this step
            if step_weights is not None and step_idx < len(step_weights):
                weight = step_weights[step_idx]
            else:
                # Exponential decay: l_deep^t = base_weight * decay_factor^t
                weight = self.base_weight * (self.decay_factor**step_idx)
                weight = max(weight, self.min_weight)

            weights.append(weight)

        # Compute weighted total loss
        weighted_losses = [w * loss for w, loss in zip(weights, step_losses)]
        total_loss = sum(weighted_losses)

        # Normalize by sum of weights to maintain loss scale
        weight_sum = sum(weights)
        if weight_sum > 0:
            total_loss = total_loss / weight_sum

        step_info = {
            "step_losses": torch.stack(step_losses),
            "step_weights": torch.tensor(weights, device=step_losses[0].device),
            "num_steps": torch.tensor(num_steps),
            "final_step_loss": step_losses[-1],
        }

        return total_loss, step_info


class ResidualImprovementLoss(nn.Module):
    """
    Residual improvement loss to encourage step-wise progress.

    Penalizes refinement steps that don't improve upon the previous prediction.
    Encourages monotonic improvement in prediction quality.
    """

    def __init__(self, improvement_weight: float = 0.1, temperature: float = 1.0, margin: float = 0.01):
        super().__init__()
        self.improvement_weight = improvement_weight
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        step_logits: list[torch.Tensor],  # [B, N, vocab_size] for each step
        targets: torch.Tensor,  # [B, N] target tokens
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute residual improvement loss.

        Encourages each step t+1 to improve upon step t by having
        higher probability for correct tokens.
        """
        if len(step_logits) < 2:
            # Need at least 2 steps for improvement comparison
            return torch.tensor(0.0, device=step_logits[0].device), {}

        improvement_losses = []
        improvements = []

        for step_idx in range(1, len(step_logits)):
            prev_logits = step_logits[step_idx - 1]  # [B, N, vocab_size]
            curr_logits = step_logits[step_idx]  # [B, N, vocab_size]

            # Get probabilities for target tokens
            prev_probs = F.softmax(prev_logits / self.temperature, dim=-1)
            curr_probs = F.softmax(curr_logits / self.temperature, dim=-1)

            # Extract probabilities for target tokens
            B, N = targets.shape
            valid_mask = targets != -100

            if valid_mask.any():
                # Gather target probabilities
                prev_target_probs = torch.gather(prev_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
                curr_target_probs = torch.gather(curr_probs, -1, targets.unsqueeze(-1)).squeeze(-1)

                # Compute improvement (should be positive)
                improvement = curr_target_probs - prev_target_probs  # [B, N]

                # Loss when improvement is below margin
                improvement_loss = F.relu(self.margin - improvement)  # [B, N]

                # Apply valid mask and average
                masked_loss = improvement_loss * valid_mask.float()
                step_improvement_loss = masked_loss.sum() / valid_mask.float().sum()

                improvement_losses.append(step_improvement_loss)
                improvements.append(improvement[valid_mask].mean())
            else:
                improvement_losses.append(torch.tensor(0.0, device=curr_logits.device))
                improvements.append(torch.tensor(0.0, device=curr_logits.device))

        if improvement_losses:
            total_improvement_loss = sum(improvement_losses) / len(improvement_losses)
            weighted_loss = self.improvement_weight * total_improvement_loss
        else:
            weighted_loss = torch.tensor(0.0, device=step_logits[0].device)

        loss_info = {
            "improvement_losses": torch.stack(improvement_losses) if improvement_losses else torch.tensor([]),
            "mean_improvements": torch.stack(improvements) if improvements else torch.tensor([]),
            "num_comparisons": torch.tensor(len(improvement_losses)),
        }

        return weighted_loss, loss_info


class ConsistencyLoss(nn.Module):
    """
    Consistency loss for augmentation equivariance.

    Ensures model predictions are consistent across augmented versions
    of the same input (rotations, flips for ARC-style tasks).
    """

    def __init__(
        self,
        consistency_weight: float = 0.1,
        temperature: float = 1.0,
        divergence_type: str = "kl",  # 'kl', 'mse', 'cosine'
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.divergence_type = divergence_type

    def forward(
        self,
        original_logits: torch.Tensor,  # [B, N, vocab_size]
        augmented_logits: torch.Tensor,  # [B, N, vocab_size]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute consistency loss between original and augmented predictions.

        Args:
            original_logits: Predictions on original input
            augmented_logits: Predictions on augmented input

        Returns:
            consistency_loss: Weighted consistency loss
            loss_info: Detailed loss information
        """
        # Convert to probabilities
        original_probs = F.softmax(original_logits / self.temperature, dim=-1)
        augmented_probs = F.softmax(augmented_logits / self.temperature, dim=-1)

        if self.divergence_type == "kl":
            # KL divergence between distributions
            consistency_loss = F.kl_div(
                F.log_softmax(augmented_logits / self.temperature, dim=-1), original_probs, reduction="batchmean"
            )
        elif self.divergence_type == "mse":
            # MSE between probability distributions
            consistency_loss = F.mse_loss(augmented_probs, original_probs)
        elif self.divergence_type == "cosine":
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(
                original_probs.view(-1, original_probs.size(-1)),
                augmented_probs.view(-1, augmented_probs.size(-1)),
                dim=-1,
            ).mean()
            consistency_loss = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown divergence type: {self.divergence_type}")

        weighted_loss = self.consistency_weight * consistency_loss

        loss_info = {
            "consistency_loss": consistency_loss,
            "divergence_type": self.divergence_type,
            "mean_original_entropy": -(original_probs * torch.log(original_probs + 1e-8)).sum(-1).mean(),
            "mean_augmented_entropy": -(augmented_probs * torch.log(augmented_probs + 1e-8)).sum(-1).mean(),
        }

        return weighted_loss, loss_info


class PonderLoss(nn.Module):
    """
    Ponder loss for ACT expected computation cost.

    Encourages the model to use minimal computation steps while maintaining
    performance. Includes scheduled ponder cost that increases over training.
    """

    def __init__(
        self,
        initial_ponder_cost: float = 0.005,
        final_ponder_cost: float = 0.02,
        schedule_type: str = "linear",  # 'linear', 'cosine', 'exponential'
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.initial_ponder_cost = initial_ponder_cost
        self.final_ponder_cost = final_ponder_cost
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def get_current_ponder_cost(self, step: int | None = None) -> float:
        """Get current ponder cost based on training schedule."""
        if step is not None:
            self.current_step = step

        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
        else:
            progress = 1.0

        if self.schedule_type == "linear":
            cost = self.initial_ponder_cost + progress * (self.final_ponder_cost - self.initial_ponder_cost)
        elif self.schedule_type == "cosine":
            cost = self.initial_ponder_cost + 0.5 * (self.final_ponder_cost - self.initial_ponder_cost) * (
                1 - math.cos(math.pi * progress)
            )
        elif self.schedule_type == "exponential":
            # Exponential interpolation
            log_initial = math.log(self.initial_ponder_cost)
            log_final = math.log(self.final_ponder_cost)
            log_cost = log_initial + progress * (log_final - log_initial)
            cost = math.exp(log_cost)
        else:
            cost = self.final_ponder_cost

        return cost

    def forward(
        self, ponder_costs: torch.Tensor, step: int | None = None  # [B] or [B, N] expected computation steps
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute ponder loss for ACT computation cost.

        Args:
            ponder_costs: Expected number of computation steps
            step: Current training step for scheduling

        Returns:
            ponder_loss: Weighted ponder cost
            loss_info: Ponder cost statistics
        """
        current_cost_weight = self.get_current_ponder_cost(step)

        # Average ponder cost across batch (and sequence if applicable)
        mean_ponder_cost = ponder_costs.mean()

        # Apply current weight
        weighted_ponder_loss = current_cost_weight * mean_ponder_cost

        loss_info = {
            "mean_ponder_cost": mean_ponder_cost,
            "ponder_cost_weight": torch.tensor(current_cost_weight),
            "min_ponder_cost": ponder_costs.min(),
            "max_ponder_cost": ponder_costs.max(),
            "std_ponder_cost": ponder_costs.std(),
            "current_step": torch.tensor(self.current_step),
        }

        return weighted_ponder_loss, loss_info


class CogmentLoss(nn.Module):
    """
    Combined loss function for Cogment training.

    Integrates all specialized loss components with stage-specific weighting:
    - Deep supervision across refinement steps
    - Residual improvement encouragement
    - Consistency across augmentations
    - Ponder cost for computation efficiency
    """

    def __init__(
        self,
        # Loss component weights
        deep_supervision_weight: float = 1.0,
        improvement_weight: float = 0.1,
        consistency_weight: float = 0.1,
        ponder_weight: float = 1.0,
        # Deep supervision parameters
        deep_decay_factor: float = 0.5,
        deep_min_weight: float = 0.1,
        # Improvement parameters
        improvement_margin: float = 0.01,
        improvement_temperature: float = 1.0,
        # Consistency parameters
        consistency_temperature: float = 1.0,
        consistency_divergence: str = "kl",
        # Ponder parameters
        initial_ponder_cost: float = 0.005,
        final_ponder_cost: float = 0.02,
        ponder_schedule: str = "linear",
        # Stage-specific scaling
        stage_weights: dict[int, dict[str, float]] | None = None,
    ):
        super().__init__()

        # Initialize loss components
        self.deep_supervision = DeepSupervisionLoss(
            base_weight=1.0, decay_factor=deep_decay_factor, min_weight=deep_min_weight
        )

        self.residual_improvement = ResidualImprovementLoss(
            improvement_weight=1.0,  # Will be scaled by main weight
            temperature=improvement_temperature,
            margin=improvement_margin,
        )

        self.consistency = ConsistencyLoss(
            consistency_weight=1.0,  # Will be scaled by main weight
            temperature=consistency_temperature,
            divergence_type=consistency_divergence,
        )

        self.ponder = PonderLoss(
            initial_ponder_cost=initial_ponder_cost, final_ponder_cost=final_ponder_cost, schedule_type=ponder_schedule
        )

        # Main component weights
        self.weights = {
            "deep_supervision": deep_supervision_weight,
            "improvement": improvement_weight,
            "consistency": consistency_weight,
            "ponder": ponder_weight,
        }

        # Stage-specific weight adjustments
        self.stage_weights = stage_weights or {
            1: {
                "deep_supervision": 1.0,
                "improvement": 0.5,
                "consistency": 0.1,
                "ponder": 0.5,
            },  # Stage 1: Focus on basic supervision
            2: {
                "deep_supervision": 1.0,
                "improvement": 1.0,
                "consistency": 0.5,
                "ponder": 1.0,
            },  # Stage 2: Add improvement
            3: {
                "deep_supervision": 0.8,
                "improvement": 1.0,
                "consistency": 1.0,
                "ponder": 1.0,
            },  # Stage 3: Add consistency
            4: {
                "deep_supervision": 0.6,
                "improvement": 1.0,
                "consistency": 1.0,
                "ponder": 1.2,
            },  # Stage 4: Emphasize efficiency
        }

        logger.info("Initialized CogmentLoss with all components")

    def forward(
        self,
        step_logits: list[torch.Tensor],  # Predictions at each refinement step
        targets: torch.Tensor,  # Target tokens [B, N]
        ponder_costs: torch.Tensor,  # ACT ponder costs [B] or [B, N]
        augmented_logits: torch.Tensor | None = None,  # Augmented predictions for consistency
        stage: int = 1,  # Current training stage (1-4)
        step: int | None = None,  # Training step for scheduling
        return_components: bool = False,  # Return individual loss components
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Compute combined Cogment loss.

        Args:
            step_logits: Predictions at each refinement step
            targets: Target token sequences
            ponder_costs: ACT computation costs
            augmented_logits: Optional augmented predictions
            stage: Current training stage for weight adjustment
            step: Training step for scheduling
            return_components: Whether to return detailed component breakdown

        Returns:
            total_loss: Combined weighted loss
            loss_info: Detailed loss breakdown and statistics
        """
        loss_components = {}
        component_info = {}

        # Get stage-specific weights
        stage_scaling = self.stage_weights.get(stage, self.stage_weights[1])

        # 1. Deep supervision loss
        if step_logits:
            deep_loss, deep_info = self.deep_supervision(step_logits, targets)
            scaled_deep_loss = self.weights["deep_supervision"] * stage_scaling["deep_supervision"] * deep_loss
            loss_components["deep_supervision"] = scaled_deep_loss
            component_info["deep_supervision"] = deep_info

        # 2. Residual improvement loss
        if len(step_logits) > 1:
            improvement_loss, improvement_info = self.residual_improvement(step_logits, targets)
            scaled_improvement_loss = self.weights["improvement"] * stage_scaling["improvement"] * improvement_loss
            loss_components["improvement"] = scaled_improvement_loss
            component_info["improvement"] = improvement_info

        # 3. Consistency loss (if augmented data provided)
        if augmented_logits is not None and step_logits:
            original_final = step_logits[-1]  # Use final step predictions
            consistency_loss, consistency_info = self.consistency(original_final, augmented_logits)
            scaled_consistency_loss = self.weights["consistency"] * stage_scaling["consistency"] * consistency_loss
            loss_components["consistency"] = scaled_consistency_loss
            component_info["consistency"] = consistency_info

        # 4. Ponder loss
        ponder_loss, ponder_info = self.ponder(ponder_costs, step)
        scaled_ponder_loss = self.weights["ponder"] * stage_scaling["ponder"] * ponder_loss
        loss_components["ponder"] = scaled_ponder_loss
        component_info["ponder"] = ponder_info

        # Combine all loss components
        total_loss = sum(loss_components.values())

        # Prepare comprehensive loss info
        loss_info = {
            "total_loss": total_loss,
            "stage": stage,
            "stage_weights": stage_scaling,
            "base_weights": self.weights,
            "component_losses": {k: v.item() if torch.is_tensor(v) else v for k, v in loss_components.items()},
            "num_active_components": len(loss_components),
        }

        if return_components:
            loss_info["component_details"] = component_info

        return total_loss, loss_info
