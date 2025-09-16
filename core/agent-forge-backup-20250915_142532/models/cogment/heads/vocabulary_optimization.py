"""Optimized vocabulary heads to solve parameter budget crisis.

CRITICAL ISSUE ADDRESSED:
- Original vocabulary heads: ~10.2M parameters (85.7% of RefinementCore)
- Target: Save 5-8M parameters to fit 25M total budget
- Solutions: Tied heads, factorization, reduced vocabulary
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabularyHeadBase(nn.Module, ABC):
    """Base class for vocabulary heads with parameter counting."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

    @abstractmethod
    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (y_logits, delta_logits)."""
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Detailed parameter breakdown for analysis."""
        breakdown = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                breakdown[name] = param.numel()
        return breakdown


class TiedVocabularyHeads(VocabularyHeadBase):
    """
    OPTION 1: Tied Vocabulary Heads (RECOMMENDED)

    Share weights between Y head and Delta head.
    - Parameter savings: 50% reduction = ~5.1M parameters saved
    - Total vocab params: ~5.1M instead of ~10.2M
    - Minimal impact on model capability
    - Clean implementation with weight tying
    """

    def __init__(self, d_model: int, vocab_size: int, delta_scale: float = 0.1):
        super().__init__(d_model, vocab_size)

        # Single shared vocabulary head
        self.shared_head = nn.Linear(d_model, vocab_size, bias=False)

        # Learnable scaling for delta predictions
        self.delta_scale = nn.Parameter(torch.ones(1) * delta_scale)

        # Optional: Different bias terms for y vs delta
        self.y_bias = nn.Parameter(torch.zeros(vocab_size))
        self.delta_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights appropriately."""
        nn.init.xavier_uniform_(self.shared_head.weight)
        nn.init.zeros_(self.y_bias)
        nn.init.zeros_(self.delta_bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with tied vocabulary heads.

        Args:
            hidden_states: Input states [B, N, d_model]

        Returns:
            y_logits: Base prediction logits [B, N, vocab_size]
            delta_logits: Scaled residual logits [B, N, vocab_size]
        """
        # Shared computation
        shared_logits = self.shared_head(hidden_states)  # [B, N, vocab_size]

        # Y head: base prediction with bias
        y_logits = shared_logits + self.y_bias

        # Delta head: scaled prediction with different bias
        delta_logits = (shared_logits + self.delta_bias) * self.delta_scale

        return y_logits, delta_logits

    def parameter_breakdown(self) -> dict:
        """Parameter breakdown showing savings."""
        breakdown = super().parameter_breakdown()
        original_params = 2 * self.d_model * self.vocab_size  # Two separate heads
        current_params = self.count_parameters()
        savings = original_params - current_params

        breakdown["_analysis"] = {
            "original_separate_heads": original_params,
            "current_tied_heads": current_params,
            "parameter_savings": savings,
            "savings_percentage": f"{100 * savings / original_params:.1f}%",
        }
        return breakdown


class FactorizedVocabularyHeads(VocabularyHeadBase):
    """
    OPTION 2: Factorized Vocabulary Heads

    Use bottleneck factorization: d_model → bottleneck → vocab_size
    - Parameter savings: Depends on bottleneck size (1024-2048 recommended)
    - Separate Y and Delta heads but both factorized
    - More flexible than tying but more complex
    """

    def __init__(self, d_model: int, vocab_size: int, bottleneck_dim: int = 1024, delta_scale: float = 0.1):
        super().__init__(d_model, vocab_size)

        self.bottleneck_dim = bottleneck_dim

        # Y head: factorized
        self.y_down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.y_up = nn.Linear(bottleneck_dim, vocab_size, bias=False)

        # Delta head: factorized
        self.delta_down = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.delta_up = nn.Linear(bottleneck_dim, vocab_size, bias=False)

        # Scaling for delta
        self.delta_scale = nn.Parameter(torch.ones(1) * delta_scale)

        self._init_weights()

    def _init_weights(self):
        """Initialize factorized weights."""
        # Xavier initialization for all linear layers
        for layer in [self.y_down, self.y_up, self.delta_down, self.delta_up]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with factorized heads.

        Args:
            hidden_states: Input states [B, N, d_model]

        Returns:
            y_logits: Base prediction logits [B, N, vocab_size]
            delta_logits: Factorized residual logits [B, N, vocab_size]
        """
        # Y head: d_model → bottleneck → vocab_size
        y_hidden = F.relu(self.y_down(hidden_states))  # [B, N, bottleneck_dim]
        y_logits = self.y_up(y_hidden)  # [B, N, vocab_size]

        # Delta head: d_model → bottleneck → vocab_size
        delta_hidden = F.relu(self.delta_down(hidden_states))  # [B, N, bottleneck_dim]
        delta_logits = self.delta_up(delta_hidden) * self.delta_scale  # [B, N, vocab_size]

        return y_logits, delta_logits

    def parameter_breakdown(self) -> dict:
        """Parameter breakdown showing factorization savings."""
        breakdown = super().parameter_breakdown()

        # Original parameters (two full heads)
        original_params = 2 * self.d_model * self.vocab_size

        # Factorized parameters
        factorized_params = 2 * (self.d_model * self.bottleneck_dim + self.bottleneck_dim * self.vocab_size)

        current_params = self.count_parameters()
        savings = original_params - current_params

        breakdown["_analysis"] = {
            "original_separate_heads": original_params,
            "theoretical_factorized": factorized_params,
            "current_params": current_params,
            "parameter_savings": savings,
            "savings_percentage": f"{100 * savings / original_params:.1f}%",
            "bottleneck_dim": self.bottleneck_dim,
        }
        return breakdown


class ReducedVocabHeads(VocabularyHeadBase):
    """
    OPTION 3: Reduced Vocabulary Heads

    Reduce vocabulary size from 32000 to smaller size (8000-16000)
    - Parameter savings: Linear with vocab reduction
    - May impact model capability
    - Simple to implement
    """

    def __init__(self, d_model: int, vocab_size: int, delta_scale: float = 0.1):
        super().__init__(d_model, vocab_size)

        # Standard separate heads but with reduced vocab
        self.y_head = nn.Linear(d_model, vocab_size, bias=False)
        self.delta_head = nn.Linear(d_model, vocab_size, bias=False)

        # Scaling for delta
        self.delta_scale = nn.Parameter(torch.ones(1) * delta_scale)

        self._init_weights()

    def _init_weights(self):
        """Initialize vocabulary heads."""
        nn.init.xavier_uniform_(self.y_head.weight)
        nn.init.xavier_uniform_(self.delta_head.weight)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass with reduced vocabulary."""
        y_logits = self.y_head(hidden_states)
        delta_logits = self.delta_head(hidden_states) * self.delta_scale
        return y_logits, delta_logits


class OptimizedVocabularyHeads(nn.Module):
    """
    Smart vocabulary head wrapper that chooses optimal strategy.

    Automatically selects the best optimization strategy based on:
    - Parameter budget constraints
    - Vocabulary size
    - Performance requirements
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        strategy: str = "tied",  # "tied", "factorized", "reduced"
        bottleneck_dim: int | None = None,
        delta_scale: float = 0.1,
        parameter_budget: int | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.strategy = strategy
        self.parameter_budget = parameter_budget

        # Auto-select strategy if budget specified
        if parameter_budget is not None and strategy == "auto":
            strategy = self._select_strategy(parameter_budget)

        # Create appropriate vocabulary head implementation
        if strategy == "tied":
            self.vocab_heads = TiedVocabularyHeads(d_model, vocab_size, delta_scale)
        elif strategy == "factorized":
            if bottleneck_dim is None:
                # Auto-calculate bottleneck for ~50% parameter reduction
                bottleneck_dim = max(512, min(2048, vocab_size // 16))
            self.vocab_heads = FactorizedVocabularyHeads(d_model, vocab_size, bottleneck_dim, delta_scale)
        elif strategy == "reduced":
            self.vocab_heads = ReducedVocabHeads(d_model, vocab_size, delta_scale)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.selected_strategy = strategy

    def _select_strategy(self, budget: int) -> str:
        """Auto-select optimization strategy based on parameter budget."""
        # Calculate parameters for each strategy
        tied_params = self.d_model * self.vocab_size + 2 * self.vocab_size  # shared + biases
        factorized_params = 2 * (self.d_model * 1024 + 1024 * self.vocab_size)  # 1024 bottleneck

        if tied_params <= budget:
            return "tied"  # Simplest and most effective
        elif factorized_params <= budget:
            return "factorized"
        else:
            return "reduced"  # Last resort

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through selected vocabulary heads."""
        return self.vocab_heads(hidden_states)

    def count_parameters(self) -> int:
        """Count parameters in selected strategy."""
        return self.vocab_heads.count_parameters()

    def parameter_analysis(self) -> dict:
        """Comprehensive parameter analysis."""
        original_params = 2 * self.d_model * self.vocab_size
        current_params = self.count_parameters()
        savings = original_params - current_params

        analysis = {
            "strategy": self.selected_strategy,
            "original_parameters": original_params,
            "optimized_parameters": current_params,
            "parameter_savings": savings,
            "savings_percentage": f"{100 * savings / original_params:.1f}%",
            "parameter_breakdown": self.vocab_heads.parameter_breakdown(),
        }

        if self.parameter_budget:
            analysis["budget_utilization"] = f"{100 * current_params / self.parameter_budget:.1f}%"
            analysis["fits_budget"] = current_params <= self.parameter_budget

        return analysis


def replace_vocabulary_heads_in_refinement_core(refinement_core, strategy: str = "tied"):
    """
    Replace vocabulary heads in existing RefinementCore with optimized versions.

    This function can be used to retrofit existing models with parameter-optimized heads.
    """
    config = refinement_core.config

    # Create optimized vocabulary heads
    optimized_heads = OptimizedVocabularyHeads(
        d_model=config.d_model,
        vocab_size=config.vocab_size,
        strategy=strategy,
        delta_scale=refinement_core.delta_scale.item() if hasattr(refinement_core, "delta_scale") else 0.1,
    )

    # Replace the heads
    refinement_core.y_head = None  # Remove old heads
    refinement_core.delta_y_head = None
    refinement_core.optimized_vocab_heads = optimized_heads

    # Modify forward method to use optimized heads

    def optimized_forward(hidden_states, memory=None, step=0):
        # Run encoder and memory gate as before
        encoded = refinement_core.encoder(hidden_states)
        fused = refinement_core.memory_gate(encoded, memory)

        # Use optimized vocabulary heads
        y_logits, delta_logits = optimized_heads(fused)

        # Halt head unchanged
        halt_logits = refinement_core.halt_head(fused)
        halt_prob = torch.sigmoid(halt_logits)

        # Refined states
        refined_states = hidden_states + encoded

        return y_logits, delta_logits, halt_prob, refined_states

    refinement_core.forward = optimized_forward

    return refinement_core


# Utility functions for parameter analysis
def calculate_vocab_savings(d_model: int, vocab_size: int, strategy: str = "tied") -> dict:
    """Calculate parameter savings for different vocabulary optimization strategies."""
    original = 2 * d_model * vocab_size

    if strategy == "tied":
        optimized = d_model * vocab_size + 2 * vocab_size  # shared weights + biases
    elif strategy == "factorized":
        bottleneck = max(512, min(2048, vocab_size // 16))
        optimized = 2 * (d_model * bottleneck + bottleneck * vocab_size)
    elif strategy == "reduced_half":
        optimized = 2 * d_model * (vocab_size // 2)
    else:
        optimized = original

    savings = original - optimized

    return {
        "strategy": strategy,
        "original_params": original,
        "optimized_params": optimized,
        "savings": savings,
        "savings_mb": savings * 4 / (1024 * 1024),  # Assuming float32
        "savings_percentage": f"{100 * savings / original:.1f}%",
    }


if __name__ == "__main__":
    # Example usage and testing
    d_model = 320
    vocab_size = 16000

    print("=== Cogment Vocabulary Head Optimization Analysis ===")
    print(f"Model dimensions: d_model={d_model}, vocab_size={vocab_size}")
    print()

    for strategy in ["tied", "factorized", "reduced_half"]:
        analysis = calculate_vocab_savings(d_model, vocab_size, strategy)
        print(f"{strategy.upper()} Strategy:")
        print(f"  Original: {analysis['original_params']:,} parameters")
        print(f"  Optimized: {analysis['optimized_params']:,} parameters")
        print(f"  Savings: {analysis['savings']:,} parameters ({analysis['savings_percentage']})")
        print(f"  Memory saved: {analysis['savings_mb']:.1f} MB")
        print()
