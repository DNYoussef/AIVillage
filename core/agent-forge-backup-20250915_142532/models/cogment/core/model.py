"""Cogment: Cognitive Modeling for Enhanced Neural Thinking."""

from __future__ import annotations

from dataclasses import dataclass

from hrrm.common.param_math import count_params
from hrrm.common.transformer_blocks import CausalSelfAttention, RMSNorm, SwiGLU
import torch
import torch.nn as nn
import torch.nn.functional as F

from .act_halting import ACTHalting, ACTLoss
from .config import CogmentConfig
from .refinement_core import RefinementCore, RefinementOutput


@dataclass
class CogmentOutput:
    """Output from Cogment model."""

    logits: torch.Tensor  # Final prediction logits [B, N, vocab_size]
    loss: torch.Tensor | None = None  # Total loss (task + ponder)

    # ACT information
    ponder_cost: torch.Tensor | None = None  # Average computation steps [B]
    halt_weights: torch.Tensor | None = None  # Step weights for averaging [B, T]

    # Refinement information
    refinement_outputs: list[RefinementOutput] | None = None  # All refinement steps
    memory_states: torch.Tensor | None = None  # Final memory states

    # Loss breakdown
    task_loss: torch.Tensor | None = None  # Task-specific loss
    ponder_loss: torch.Tensor | None = None  # Ponder cost loss

    # Debug information
    num_steps: torch.Tensor | None = None  # Actual steps taken [B]
    halt_probs: torch.Tensor | None = None  # All halt probabilities [B, T]


class TransformerBlock(nn.Module):
    """Standard transformer block with RMSNorm and SwiGLU."""

    def __init__(self, config: CogmentConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model

        # Pre-normalization
        self.attn_norm = RMSNorm(d_model, eps=config.layer_norm_eps)
        self.mlp_norm = RMSNorm(d_model, eps=config.layer_norm_eps)

        # Attention with reduced key-value dimension
        self.attn = CausalSelfAttention(d_model=d_model, n_head=config.n_head, rope_base=config.rope_base)

        # MLP with SwiGLU activation
        self.mlp = SwiGLU(d_model, mult=config.d_ff // d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Standard transformer block forward pass."""
        # Attention with residual
        attn_out = self.attn(self.attn_norm(x), attn_mask)
        x = x + self.dropout(attn_out)

        # MLP with residual
        mlp_out = self.mlp(self.mlp_norm(x))
        x = x + self.dropout(mlp_out)

        return x


class CogmentBackbone(nn.Module):
    """Base transformer backbone for Cogment."""

    def __init__(self, config: CogmentConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final normalization
        self.norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass through transformer backbone.

        Args:
            input_ids: Token indices [B, N]
            attn_mask: Attention mask [B, N, N] or [N, N]

        Returns:
            hidden_states: Contextual representations [B, N, d_model]
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # [B, N, d_model]

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Final normalization
        return self.norm(x)


class Cogment(nn.Module):
    """
    Cogment: Cognitive Modeling for Enhanced Neural Thinking.

    Combines a transformer backbone with iterative refinement using ACT halting.
    The RefinementCore processes representations through multiple steps, allowing
    for adaptive computation based on input complexity.
    """

    def __init__(self, config: CogmentConfig):
        super().__init__()
        self.config = config

        # Core components
        self.backbone = CogmentBackbone(config)
        self.refinement_core = RefinementCore(config)

        # ACT halting mechanism
        self.act_halting = ACTHalting(threshold=config.act_threshold, epsilon=config.halt_epsilon)

        # Loss function with ponder cost
        self.act_loss = ACTLoss(ponder_weight=config.ponder_cost_weight)

        # Output projection (tied with embeddings)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_projection.weight = self.backbone.token_embedding.weight  # Weight tying

        print(f"Cogment initialized with {self.count_parameters():,} parameters")

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, N]
        labels: torch.Tensor | None = None,  # [B, N]
        attn_mask: torch.Tensor | None = None,  # [B, N, N]
        memory: torch.Tensor | None = None,  # [B, M, ltm_dim]
        max_refinement_steps: int | None = None,  # Override default steps
        return_refinement_details: bool = False,  # Return detailed step info
    ) -> CogmentOutput:
        """
        Forward pass with iterative refinement and ACT halting.

        Args:
            input_ids: Input token indices [B, N]
            labels: Target labels for training [B, N]
            attn_mask: Attention mask
            memory: Long-term memory states
            max_refinement_steps: Maximum refinement iterations
            return_refinement_details: Whether to return detailed step information

        Returns:
            CogmentOutput with predictions, losses, and optional details
        """
        B, N = input_ids.shape
        device = input_ids.device

        # Get backbone representations
        hidden_states = self.backbone(input_ids, attn_mask)  # [B, N, d_model]

        # Refinement loop with ACT
        max_steps = max_refinement_steps or self.config.max_refinement_steps
        min_steps = self.config.min_refinement_steps

        refinement_outputs = []
        halt_probs_list = []
        step_outputs = []

        current_states = hidden_states

        for step in range(max_steps):
            # Run refinement core
            y_logits, delta_logits, halt_prob, refined_states = self.refinement_core(current_states, memory, step)

            # Combine predictions
            combined_logits = self.refinement_core.compute_prediction(y_logits, delta_logits)

            # Store outputs
            refinement_output = RefinementOutput(
                y_logits=y_logits,
                delta_logits=delta_logits,
                halt_prob=halt_prob,
                refined_states=refined_states,
                combined_logits=combined_logits,
            )
            refinement_outputs.append(refinement_output)
            halt_probs_list.append(halt_prob)
            step_outputs.append(combined_logits)

            # Update states for next iteration
            current_states = refined_states

        # Stack outputs for ACT processing
        halt_probs = torch.stack(halt_probs_list, dim=1)  # [B, T, N, 1] -> [B, N, T, 1]
        halt_probs = halt_probs.transpose(1, 2).squeeze(-1)  # [B, N, T]

        step_logits = torch.stack(step_outputs, dim=1)  # [B, T, N, vocab_size]
        step_logits = step_logits.transpose(1, 2)  # [B, N, T, vocab_size]

        # Apply ACT halting for each position independently
        final_logits = torch.zeros(B, N, self.config.vocab_size, device=device)
        ponder_costs = torch.zeros(B, N, device=device)
        all_weights = torch.zeros(B, N, max_steps, device=device)

        for b in range(B):
            for n in range(N):
                # Apply ACT for this position
                pos_halt_probs = halt_probs[b, n : n + 1, :]  # [1, T]
                pos_outputs = step_logits[b, n : n + 1, :, :]  # [1, T, vocab_size]

                # Ensure minimum steps
                if max_steps > min_steps:
                    pos_halt_probs[:, :min_steps] = 0.0

                final_out, ponder_cost, weights = self.act_halting(
                    pos_halt_probs.unsqueeze(-1),  # [1, T, 1]
                    pos_outputs,  # [1, T, vocab_size]
                )

                final_logits[b, n] = final_out.squeeze(0)  # [vocab_size]
                ponder_costs[b, n] = ponder_cost.squeeze(0)  # scalar
                all_weights[b, n] = weights.squeeze(0)  # [T]

        # Compute losses if labels provided
        total_loss = None
        task_loss = None
        ponder_loss = None

        if labels is not None:
            # Language modeling loss
            task_loss = F.cross_entropy(
                final_logits.view(-1, self.config.vocab_size), labels.view(-1), ignore_index=-100
            )

            # Combine with ponder cost
            avg_ponder_cost = ponder_costs.mean()
            total_loss, task_loss_item, ponder_loss_item = self.act_loss(task_loss, avg_ponder_cost)

            task_loss = task_loss_item
            ponder_loss = ponder_loss_item

        # Prepare output
        output = CogmentOutput(
            logits=final_logits,
            loss=total_loss,
            ponder_cost=ponder_costs.mean(dim=1),  # [B]
            halt_weights=all_weights,
            task_loss=task_loss,
            ponder_loss=ponder_loss,
            num_steps=torch.sum(all_weights > 0, dim=-1).float(),  # [B, N]
            halt_probs=halt_probs,
        )

        if return_refinement_details:
            output.refinement_outputs = refinement_outputs
            output.memory_states = memory

        return output

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = True,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate text using Cogment with adaptive computation.

        Args:
            input_ids: Input prompt [B, N]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            memory: Long-term memory context

        Returns:
            generated_ids: Generated token sequences [B, N + gen_length]
        """
        self.eval()

        with torch.no_grad():
            current_ids = input_ids.clone()

            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(current_ids, memory=memory)
                logits = outputs.logits[:, -1, :]  # [B, vocab_size]

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)

                # Check for EOS token (assuming 2 is EOS)
                if (next_token == 2).all():
                    break

            return current_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return count_params(self)

    def parameter_breakdown(self) -> dict[str, int]:
        """Get detailed parameter breakdown by component."""
        breakdown = {
            "backbone": count_params(self.backbone),
            "refinement_core": count_params(self.refinement_core),
            "act_halting": count_params(self.act_halting),
            "output_projection": 0,  # Tied weights, counted in backbone
            "total": count_params(self),
        }
        return breakdown
