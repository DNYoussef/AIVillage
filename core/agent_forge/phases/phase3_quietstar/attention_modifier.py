"""
Attention Modifier for Quiet-STaR Implementation
Implements thought-aware attention mechanisms for thought injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class ThoughtAwareAttention(nn.Module):
    """
    Modified attention layer that incorporates thought tokens
    into the attention mechanism with configurable blending
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        thought_weight: float = 0.3,
        layer_idx: int = 0,
        total_layers: int = 12
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.thought_weight = thought_weight
        self.layer_idx = layer_idx
        self.total_layers = total_layers

        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Thought-specific projections
        self.thought_q_proj = nn.Linear(d_model, d_model)
        self.thought_k_proj = nn.Linear(d_model, d_model)
        self.thought_v_proj = nn.Linear(d_model, d_model)

        # Attention mixing weights
        self.attention_mixer = nn.Linear(d_model * 2, 1)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

        # Progressive weight calculation
        self.progressive_weight = self._calculate_progressive_weight()

    def _calculate_progressive_weight(self) -> float:
        """Calculate layer-wise progressive modification weight"""
        # Earlier layers have less thought influence, later layers have more
        progress = self.layer_idx / max(1, self.total_layers - 1)
        return self.thought_weight * (0.5 + 0.5 * progress)

    def forward(
        self,
        hidden_states: torch.Tensor,
        thought_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        thought_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with thought-aware attention

        Args:
            hidden_states: [batch_size, seq_len, d_model]
            thought_states: [batch_size, seq_len, thought_len, d_model]
            attention_mask: [batch_size, seq_len]
            thought_mask: [batch_size, seq_len, thought_len]
            output_attentions: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Standard self-attention
        original_output, original_attn = self._compute_standard_attention(
            hidden_states, attention_mask, output_attentions
        )

        # If no thoughts, return standard attention
        if thought_states is None:
            return original_output, original_attn

        # Thought-aware attention
        thought_output, thought_attn = self._compute_thought_attention(
            hidden_states, thought_states, thought_mask, output_attentions
        )

        # Blend attention outputs
        blended_output = self._blend_attention_outputs(
            original_output, thought_output, hidden_states
        )

        # Combine attention weights if requested
        combined_attn = None
        if output_attentions:
            combined_attn = self._combine_attention_weights(
                original_attn, thought_attn
            )

        return blended_output, combined_attn

    def _compute_standard_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        output_attentions: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute standard self-attention"""
        batch_size, seq_len, d_model = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2) == 0, -float('inf')
            )

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device),
            diagonal=1
        ).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, -float('inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.out_proj(attn_output)

        return output, attn_weights if output_attentions else None

    def _compute_thought_attention(
        self,
        hidden_states: torch.Tensor,
        thought_states: torch.Tensor,
        thought_mask: Optional[torch.Tensor],
        output_attentions: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute cross-attention between sequence and thoughts"""
        batch_size, seq_len, d_model = hidden_states.shape
        thought_len = thought_states.shape[2]

        # Flatten thought states for attention computation
        thought_flat = thought_states.view(batch_size, seq_len * thought_len, d_model)

        # Project queries from hidden states, keys/values from thoughts
        q = self.thought_q_proj(hidden_states)
        k = self.thought_k_proj(thought_flat)
        v = self.thought_v_proj(thought_flat)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len * thought_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len * thought_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute cross-attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply thought mask
        if thought_mask is not None:
            thought_mask_flat = thought_mask.view(batch_size, seq_len * thought_len)
            attn_scores = attn_scores.masked_fill(
                thought_mask_flat.unsqueeze(1).unsqueeze(2) == 0, -float('inf')
            )

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.out_proj(attn_output)

        return output, attn_weights if output_attentions else None

    def _blend_attention_outputs(
        self,
        original_output: torch.Tensor,
        thought_output: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Blend original and thought attention outputs"""
        # Concatenate features for mixing decision
        combined_features = torch.cat([original_output, thought_output], dim=-1)

        # Compute mixing weights
        mix_weights = torch.sigmoid(self.attention_mixer(combined_features))

        # Apply progressive weight
        effective_weight = self.progressive_weight * mix_weights

        # Blend outputs
        blended = (1 - effective_weight) * original_output + effective_weight * thought_output

        return blended

    def _combine_attention_weights(
        self,
        original_attn: torch.Tensor,
        thought_attn: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Combine attention weights for visualization"""
        return {
            'original_attention': original_attn,
            'thought_attention': thought_attn,
            'blend_weight': self.progressive_weight
        }


class AttentionModifier(nn.Module):
    """
    Main attention modifier that handles thought injection
    across multiple attention layers
    """

    def __init__(
        self,
        config: Dict[str, Any],
        total_layers: int = 12
    ):
        super().__init__()

        self.config = config
        self.total_layers = total_layers
        self.thought_weight = config.get('thought_weight', 0.3)
        self.progressive_mixing = config.get('progressive_mixing', True)

        # Attention score combiners for each layer
        self.score_combiners = nn.ModuleList([
            nn.Linear(2, 1) for _ in range(total_layers)
        ])

        # Layer-wise thought weights
        self.layer_weights = nn.Parameter(
            torch.linspace(0.1, self.thought_weight, total_layers)
        )

    def modify_attention_scores(
        self,
        original_scores: torch.Tensor,
        thought_scores: torch.Tensor,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Modify attention scores by blending original and thought scores

        Args:
            original_scores: Original attention scores
            thought_scores: Thought-based attention scores
            layer_idx: Current layer index
            attention_mask: Attention mask

        Returns:
            Modified attention scores
        """
        if thought_scores is None:
            return original_scores

        # Get layer-specific weight
        layer_weight = self.layer_weights[layer_idx]

        # Combine scores using learned combination
        score_features = torch.stack([original_scores, thought_scores], dim=-1)
        combination_weights = torch.softmax(
            self.score_combiners[layer_idx](score_features), dim=-1
        )

        # Apply layer weight to thought combination
        modified_weights = combination_weights.clone()
        modified_weights[..., 1] *= layer_weight
        modified_weights = F.normalize(modified_weights, p=1, dim=-1)

        # Combine scores
        modified_scores = (
            modified_weights[..., 0] * original_scores +
            modified_weights[..., 1] * thought_scores
        )

        # Apply attention mask
        if attention_mask is not None:
            modified_scores = modified_scores.masked_fill(
                attention_mask == 0, -float('inf')
            )

        return modified_scores

    def create_thought_attention_mask(
        self,
        sequence_mask: torch.Tensor,
        thought_mask: torch.Tensor
    ) -> torch.Tensor:
        """Create combined attention mask for thoughts and sequence"""
        batch_size, seq_len = sequence_mask.shape
        thought_len = thought_mask.shape[-1]

        # Expand sequence mask for thought dimension
        expanded_seq_mask = sequence_mask.unsqueeze(-1).expand(-1, -1, thought_len)

        # Combine with thought mask
        combined_mask = expanded_seq_mask & thought_mask

        return combined_mask

    def apply_causal_masking(
        self,
        attention_scores: torch.Tensor,
        include_thoughts: bool = True
    ) -> torch.Tensor:
        """Apply causal masking to attention scores"""
        seq_len = attention_scores.shape[-1]

        if include_thoughts:
            # Create causal mask that allows attention to thoughts
            # but maintains causality in sequence
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attention_scores.device),
                diagonal=1
            ).bool()
        else:
            # Standard causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=attention_scores.device),
                diagonal=1
            ).bool()

        attention_scores = attention_scores.masked_fill(causal_mask, -float('inf'))

        return attention_scores


class ThoughtAttentionMixer(nn.Module):
    """
    Specialized module for mixing thought and sequence attention
    """

    def __init__(
        self,
        d_model: int,
        mixing_strategy: str = 'learned',
        temperature: float = 1.0
    ):
        super().__init__()

        self.d_model = d_model
        self.mixing_strategy = mixing_strategy
        self.temperature = temperature

        if mixing_strategy == 'learned':
            self.mixing_network = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
        elif mixing_strategy == 'attention':
            self.attention_mixer = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=0.1
            )

    def forward(
        self,
        sequence_repr: torch.Tensor,
        thought_repr: torch.Tensor,
        mixing_strategy: Optional[str] = None
    ) -> torch.Tensor:
        """
        Mix sequence and thought representations

        Args:
            sequence_repr: Sequence representation
            thought_repr: Thought representation
            mixing_strategy: Override default mixing strategy

        Returns:
            Mixed representation
        """
        strategy = mixing_strategy or self.mixing_strategy

        if strategy == 'learned':
            return self._learned_mixing(sequence_repr, thought_repr)
        elif strategy == 'attention':
            return self._attention_mixing(sequence_repr, thought_repr)
        elif strategy == 'fixed':
            return self._fixed_mixing(sequence_repr, thought_repr)
        else:
            raise ValueError(f"Unknown mixing strategy: {strategy}")

    def _learned_mixing(
        self,
        sequence_repr: torch.Tensor,
        thought_repr: torch.Tensor
    ) -> torch.Tensor:
        """Learned mixing using neural network"""
        combined = torch.cat([sequence_repr, thought_repr], dim=-1)
        mixing_weight = self.mixing_network(combined)

        mixed = (1 - mixing_weight) * sequence_repr + mixing_weight * thought_repr
        return mixed

    def _attention_mixing(
        self,
        sequence_repr: torch.Tensor,
        thought_repr: torch.Tensor
    ) -> torch.Tensor:
        """Attention-based mixing"""
        # Use sequence as query, thought as key/value
        mixed, _ = self.attention_mixer(
            sequence_repr, thought_repr, thought_repr
        )
        return mixed

    def _fixed_mixing(
        self,
        sequence_repr: torch.Tensor,
        thought_repr: torch.Tensor,
        weight: float = 0.3
    ) -> torch.Tensor:
        """Fixed weight mixing"""
        return (1 - weight) * sequence_repr + weight * thought_repr


def create_attention_modifier(
    config: Dict[str, Any],
    layer_idx: int,
    total_layers: int
) -> ThoughtAwareAttention:
    """
    Factory function to create attention modifier for a specific layer

    Args:
        config: Configuration dictionary
        layer_idx: Current layer index
        total_layers: Total number of layers

    Returns:
        ThoughtAwareAttention instance
    """
    return ThoughtAwareAttention(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        dropout=config.get('dropout', 0.1),
        thought_weight=config.get('thought_weight', 0.3),
        layer_idx=layer_idx,
        total_layers=total_layers
    )


def modify_attention_weights(
    attention_weights: torch.Tensor,
    thought_weights: torch.Tensor,
    blend_ratio: float = 0.3
) -> torch.Tensor:
    """
    Utility function to modify attention weights with thought weights

    Args:
        attention_weights: Original attention weights
        thought_weights: Thought-based attention weights
        blend_ratio: Blending ratio for thoughts

    Returns:
        Modified attention weights
    """
    if thought_weights is None:
        return attention_weights

    # Normalize weights to ensure they sum to 1
    attention_weights = F.softmax(attention_weights, dim=-1)
    thought_weights = F.softmax(thought_weights, dim=-1)

    # Blend weights
    modified_weights = (1 - blend_ratio) * attention_weights + blend_ratio * thought_weights

    return modified_weights


# Configuration for attention modification
ATTENTION_CONFIG = {
    'thought_weight': 0.3,
    'progressive_mixing': True,
    'mixing_strategy': 'learned',
    'causal_masking': True,
    'temperature': 1.0,
    'dropout': 0.1
}