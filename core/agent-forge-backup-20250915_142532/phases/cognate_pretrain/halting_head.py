#!/usr/bin/env python3
"""
ACT Halting Head - Standalone Adaptive Computation Time Module

This module implements the Adaptive Computation Time (ACT) halting mechanism
as a standalone component for the Cognate Refiner. Provides:

- Lightweight halting probability prediction
- Step-aware computation with positional embeddings
- Cumulative probability tracking
- Early stopping decisions based on confidence thresholds

Key Features:
- Efficient MLP architecture for halting decisions
- Step-aware transformations with learned embeddings
- Numerical stability with epsilon handling
- Compatible with both training and inference modes
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTHaltingHead(nn.Module):
    """
    Adaptive Computation Time (ACT) halting head for dynamic computation.

    Predicts when to halt the refinement process based on current hidden states
    and step information. Uses a lightweight MLP with step embeddings to make
    informed halting decisions.

    Architecture:
    - Step embedding layer for positional awareness
    - Multi-layer MLP with SiLU activations
    - Sigmoid output for halting probabilities
    - Dropout for regularization
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.max_steps = getattr(config, "max_act_steps", 16)
        self.epsilon = getattr(config, "act_epsilon", 0.01)
        self.dropout_rate = config.dropout

        # Step embedding for positional awareness
        self.step_embedding = nn.Embedding(self.max_steps + 1, self.d_model)

        # Normalization for step-enhanced states
        self.step_norm = nn.LayerNorm(self.d_model)

        # Halting prediction network
        hidden_dim = max(self.d_model // 4, 64)  # Ensure reasonable minimum size

        self.halting_mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with Xavier uniform strategy."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        step: int | None = 0,
        previous_cumulative_probs: torch.Tensor | None = None,
        return_cumulative: bool = False,
    ) -> torch.Tensor:
        """
        Predict halting probabilities for current step.

        Args:
            hidden_states: Current hidden states (B, d_model) or (B, L, d_model)
            step: Current refinement step (0-indexed)
            previous_cumulative_probs: Previous cumulative halting probabilities (B,)
            return_cumulative: Whether to return cumulative probabilities

        Returns:
            If return_cumulative=False:
                halting_logits: Raw halting logits (B,)
            If return_cumulative=True:
                Tuple of (halting_probs, cumulative_probs, should_halt)
        """
        # Handle different input shapes
        if hidden_states.dim() == 3:
            # (B, L, d_model) -> pool to (B, d_model)
            B, L, _ = hidden_states.shape
            # Use attention-weighted pooling or simple mean
            h_pooled = hidden_states.mean(dim=1)  # Simple mean pooling
        else:
            # (B, d_model)
            B = hidden_states.shape[0]
            h_pooled = hidden_states

        # Add step information
        step_clamped = min(max(step, 0), self.max_steps)  # Clamp to valid range
        step_emb = self.step_embedding(torch.tensor(step_clamped, device=hidden_states.device, dtype=torch.long))
        step_emb = step_emb.unsqueeze(0).expand(B, -1)  # (B, d_model)

        # Combine hidden states with step information
        step_enhanced = self.step_norm(h_pooled + step_emb)

        # Predict halting logits
        halting_logits = self.halting_mlp(step_enhanced).squeeze(-1)  # (B,)

        # Return raw logits if not computing cumulative probabilities
        if not return_cumulative:
            return halting_logits

        # Compute probabilities and cumulative statistics
        current_halt_probs = torch.sigmoid(halting_logits)  # (B,)

        # Initialize or update cumulative probabilities
        if previous_cumulative_probs is None:
            cumulative_probs = current_halt_probs
            adjusted_halt_probs = current_halt_probs
        else:
            # Probability of halting at this step given we haven't halted yet
            remaining_prob = 1.0 - previous_cumulative_probs
            adjusted_halt_probs = current_halt_probs * remaining_prob
            cumulative_probs = previous_cumulative_probs + adjusted_halt_probs

        # Ensure probabilities don't exceed 1.0
        cumulative_probs = torch.clamp(cumulative_probs, 0.0, 1.0)

        # Halting decision based on threshold
        threshold = 1.0 - self.epsilon
        should_halt = cumulative_probs > threshold

        return adjusted_halt_probs, cumulative_probs, should_halt

    def compute_act_loss(
        self, all_halt_probs: list, target_steps: float = 4.0, variance_penalty: float = 0.01
    ) -> torch.Tensor:
        """
        Compute ACT loss for training.

        Args:
            all_halt_probs: List of halting probabilities from each step
            target_steps: Target number of computation steps
            variance_penalty: Penalty weight for high variance in step counts

        Returns:
            ACT loss tensor
        """
        if not all_halt_probs:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        device = all_halt_probs[0].device
        num_steps = len(all_halt_probs)

        # Stack all halting probabilities
        stacked_probs = torch.stack(all_halt_probs, dim=1)  # (B, steps)

        # Compute expected number of steps per sequence
        step_indices = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
        step_indices = step_indices.unsqueeze(0).expand(stacked_probs.shape[0], -1)

        expected_steps = (stacked_probs * step_indices).sum(dim=1)  # (B,)
        mean_expected_steps = expected_steps.mean()

        # Target steps loss - encourage using around target number of steps
        target_loss = F.mse_loss(mean_expected_steps, torch.tensor(target_steps, device=device, dtype=torch.float32))

        # Variance penalty - encourage consistent stepping across sequences
        if variance_penalty > 0:
            steps_variance = expected_steps.var()
            variance_loss = variance_penalty * steps_variance
        else:
            variance_loss = 0.0

        # Completeness loss - ensure probabilities sum to reasonable values
        final_cumulative = stacked_probs.cumsum(dim=1)[:, -1]  # (B,)
        completeness_loss = F.mse_loss(final_cumulative, torch.ones_like(final_cumulative))

        # Combined ACT loss
        total_loss = target_loss + variance_loss + 0.1 * completeness_loss

        return total_loss

    def should_halt(
        self, halting_probs: torch.Tensor, cumulative_probs: torch.Tensor | None = None, threshold: float = 0.99
    ) -> torch.Tensor:
        """
        Determine if computation should halt based on probabilities.

        Args:
            halting_probs: Current step halting probabilities (B,)
            cumulative_probs: Cumulative halting probabilities (B,)
            threshold: Halting threshold

        Returns:
            Boolean tensor indicating which sequences should halt (B,)
        """
        if cumulative_probs is not None:
            return cumulative_probs > threshold
        else:
            return halting_probs > threshold

    def get_expected_steps(self, all_halt_probs: list) -> torch.Tensor:
        """
        Compute expected number of computation steps.

        Args:
            all_halt_probs: List of halting probabilities from each step

        Returns:
            Expected number of steps for each sequence (B,)
        """
        if not all_halt_probs:
            return torch.zeros(1, device=next(self.parameters()).device)

        device = all_halt_probs[0].device
        num_steps = len(all_halt_probs)

        # Stack probabilities and compute expected steps
        stacked_probs = torch.stack(all_halt_probs, dim=1)  # (B, steps)
        step_indices = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32)
        step_indices = step_indices.unsqueeze(0).expand(stacked_probs.shape[0], -1)

        expected_steps = (stacked_probs * step_indices).sum(dim=1)  # (B,)

        return expected_steps


class ACTScheduler:
    """
    Scheduler for managing ACT halting during training and inference.

    Provides utilities for:
    - Dynamic step allocation based on task complexity
    - Early stopping decisions during inference
    - Training supervision for halting behavior
    """

    def __init__(
        self, min_steps: int = 1, max_steps: int = 16, target_steps: float = 4.0, early_stopping_threshold: float = 0.8
    ):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.target_steps = target_steps
        self.early_stopping_threshold = early_stopping_threshold

        # Statistics
        self.step_history = []
        self.halt_history = []

    def should_continue(self, cumulative_probs: torch.Tensor, current_step: int, is_training: bool = False) -> bool:
        """
        Decide whether to continue computation.

        Args:
            cumulative_probs: Cumulative halting probabilities (B,)
            current_step: Current step number
            is_training: Whether in training mode

        Returns:
            Whether to continue computation
        """
        # Always respect minimum and maximum steps
        if current_step < self.min_steps:
            return True

        if current_step >= self.max_steps:
            return False

        # Check halting condition
        halt_ratio = (cumulative_probs > self.early_stopping_threshold).float().mean()

        # More aggressive early stopping during inference
        threshold = 0.5 if is_training else 0.3

        should_continue = halt_ratio < threshold

        # Update statistics
        self.step_history.append(current_step)
        self.halt_history.append(halt_ratio.item())

        return should_continue

    def get_statistics(self) -> dict:
        """Get scheduling statistics."""
        if not self.step_history:
            return {"avg_steps": 0, "avg_halt_ratio": 0, "total_calls": 0}

        return {
            "avg_steps": sum(self.step_history) / len(self.step_history),
            "avg_halt_ratio": sum(self.halt_history) / len(self.halt_history),
            "total_calls": len(self.step_history),
            "max_steps_used": max(self.step_history),
            "min_steps_used": min(self.step_history),
        }


# Factory function
def create_act_halting_head(config, **kwargs):
    """Factory function to create ACT halting head."""
    return ACTHaltingHead(config, **kwargs)


if __name__ == "__main__":
    # Test ACT halting head
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        d_model: int = 512
        max_act_steps: int = 16
        act_epsilon: float = 0.01
        dropout: float = 0.1

    config = TestConfig()
    halting_head = ACTHaltingHead(config)

    # Test forward pass
    batch_size = 4
    hidden_states = torch.randn(batch_size, config.d_model)

    # Test simple forward
    halt_logits = halting_head(hidden_states, step=2)
    print(f"Halting logits shape: {halt_logits.shape}")
    print(f"Halting logits: {halt_logits}")

    # Test cumulative computation
    halt_probs, cum_probs, should_halt = halting_head(hidden_states, step=2, return_cumulative=True)
    print(f"Halting probs: {halt_probs}")
    print(f"Cumulative probs: {cum_probs}")
    print(f"Should halt: {should_halt}")

    # Test ACT loss
    all_probs = [torch.sigmoid(torch.randn(batch_size)) for _ in range(5)]
    act_loss = halting_head.compute_act_loss(all_probs)
    print(f"ACT loss: {act_loss}")

    # Test scheduler
    scheduler = ACTScheduler()
    for step in range(8):
        cum_probs = torch.rand(batch_size)
        should_continue = scheduler.should_continue(cum_probs, step, is_training=True)
        print(f"Step {step}: should_continue={should_continue}")
        if not should_continue:
            break

    print(f"Scheduler stats: {scheduler.get_statistics()}")
    print("✅ ACT Halting Head test passed!")
