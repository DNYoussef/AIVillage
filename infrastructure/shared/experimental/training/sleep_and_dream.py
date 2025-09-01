"""
Sleep and Dream Networks for Self-Modeling

Implements SleepNet and DreamNet architectures for neural consolidation and
creative exploration during model training. Based on memory consolidation
research and neural replay mechanisms.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SleepBlock(nn.Module):
    """
    Sleep block implementing memory consolidation through gradual compression.

    Inspired by slow-wave sleep patterns where information is gradually
    compressed and stored in long-term memory structures.
    """

    def __init__(self, input_size: int, compression_ratio: float = 0.7):
        super().__init__()

        compressed_size = int(input_size * compression_ratio)

        # Compression pathway (encoding)
        self.compress = nn.Sequential(
            nn.Linear(input_size, compressed_size), nn.LayerNorm(compressed_size), nn.GELU(), nn.Dropout(0.1)
        )

        # Reconstruction pathway (decoding)
        self.reconstruct = nn.Sequential(nn.Linear(compressed_size, input_size), nn.LayerNorm(input_size), nn.GELU())

        # Memory consolidation gate
        self.consolidation_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing sleep-like consolidation.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Consolidated tensor with same shape as input
        """
        # Compress information (sleep encoding)
        compressed = self.compress(x)

        # Reconstruct information (consolidation)
        reconstructed = self.reconstruct(compressed)

        # Apply consolidation gate (selective retention)
        gate = self.consolidation_gate(x)
        consolidated = gate * reconstructed + (1 - gate) * x

        return consolidated


class SleepNet(nn.Module):
    """
    SleepNet for neural consolidation during training breaks.

    Implements a series of sleep blocks that gradually consolidate
    learned representations, similar to memory replay during sleep.
    """

    def __init__(
        self, input_size: int, output_size: int, num_sleep_blocks: int = 3, compression_ratios: Optional[list] = None
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_sleep_blocks = num_sleep_blocks

        # Default compression ratios for progressive consolidation
        if compression_ratios is None:
            compression_ratios = [0.8, 0.7, 0.6][:num_sleep_blocks]

        # Sleep blocks for progressive consolidation
        self.sleep_blocks = nn.ModuleList([SleepBlock(input_size, ratio) for ratio in compression_ratios])

        # Output projection (if different size needed)
        if input_size != output_size:
            self.output_projection = nn.Linear(input_size, output_size)
        else:
            self.output_projection = nn.Identity()

        # Temperature scaling for consolidation strength
        self.temperature = nn.Parameter(torch.ones(1))

        logger.info(f"Initialized SleepNet with {num_sleep_blocks} blocks")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply sleep consolidation to hidden states.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Consolidated hidden states [batch_size, seq_len, output_size]
        """
        x = hidden_states

        # Progressive consolidation through sleep blocks
        for sleep_block in self.sleep_blocks:
            x = sleep_block(x)

        # Apply temperature scaling
        x = x * self.temperature

        # Final output projection
        output = self.output_projection(x)

        return output

    def consolidate_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """
        Consolidate gradients using sleep mechanisms.

        Args:
            gradients: Raw gradients to consolidate

        Returns:
            Consolidated gradients
        """
        with torch.no_grad():
            # Apply sleep consolidation to gradients
            consolidated = self.forward(gradients.unsqueeze(0))
            return consolidated.squeeze(0)


class DreamBlock(nn.Module):
    """
    Dream block implementing creative exploration and generation.

    Uses attention mechanisms and random perturbations to generate
    novel combinations of learned patterns.
    """

    def __init__(self, input_size: int, expansion_ratio: float = 1.3):
        super().__init__()

        expanded_size = int(input_size * expansion_ratio)

        # Creative expansion pathway
        self.expand = nn.Sequential(
            nn.Linear(input_size, expanded_size),
            nn.LayerNorm(expanded_size),
            nn.GELU(),
            nn.Dropout(0.15),  # Higher dropout for creativity
        )

        # Self-attention for pattern mixing
        self.attention = nn.MultiheadAttention(embed_dim=expanded_size, num_heads=8, dropout=0.1, batch_first=True)

        # Compression back to original size
        self.compress = nn.Sequential(nn.Linear(expanded_size, input_size), nn.LayerNorm(input_size), nn.GELU())

        # Creativity gate (how much novel content to inject)
        self.creativity_gate = nn.Sequential(nn.Linear(input_size, input_size), nn.Sigmoid())

    def forward(self, x: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
        """
        Forward pass implementing dream-like creativity.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            noise_scale: Scale of random perturbations for creativity

        Returns:
            Creative exploration tensor with same shape as input
        """
        batch_size, seq_len, hidden_size = x.shape

        # Expand to creative space
        expanded = self.expand(x)

        # Add random perturbations for creativity
        if self.training and noise_scale > 0:
            noise = torch.randn_like(expanded) * noise_scale
            expanded = expanded + noise

        # Self-attention for pattern mixing
        attended, _ = self.attention(expanded, expanded, expanded)

        # Compress back to original space
        compressed = self.compress(attended)

        # Apply creativity gate
        gate = self.creativity_gate(x)
        creative_output = gate * compressed + (1 - gate) * x

        return creative_output


class DreamNet(nn.Module):
    """
    DreamNet for creative exploration and pattern generation.

    Implements a series of dream blocks that generate novel combinations
    of learned patterns, similar to REM sleep and dreaming.
    """

    def __init__(
        self, input_size: int, output_size: int, num_dream_blocks: int = 3, creativity_levels: Optional[list] = None
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_dream_blocks = num_dream_blocks

        # Default creativity levels (noise scales)
        if creativity_levels is None:
            creativity_levels = [0.05, 0.1, 0.15][:num_dream_blocks]
        self.creativity_levels = creativity_levels

        # Dream blocks for creative exploration
        self.dream_blocks = nn.ModuleList(
            [DreamBlock(input_size, expansion_ratio=1.2 + 0.1 * i) for i in range(num_dream_blocks)]
        )

        # Output projection
        if input_size != output_size:
            self.output_projection = nn.Linear(input_size, output_size)
        else:
            self.output_projection = nn.Identity()

        # Lucidity parameter (how much creative content to generate)
        self.lucidity = nn.Parameter(torch.tensor(0.5))

        logger.info(f"Initialized DreamNet with {num_dream_blocks} blocks")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply dream-like creative exploration to hidden states.

        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]

        Returns:
            Creative exploration output [batch_size, seq_len, output_size]
        """
        x = hidden_states

        # Progressive creative exploration through dream blocks
        for i, dream_block in enumerate(self.dream_blocks):
            noise_scale = self.creativity_levels[i] * self.lucidity
            x = dream_block(x, noise_scale=noise_scale)

        # Final output projection
        output = self.output_projection(x)

        return output

    def generate_creative_patterns(self, seed_states: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Generate creative patterns through iterative dreaming.

        Args:
            seed_states: Initial states to dream from
            num_steps: Number of dream iterations

        Returns:
            Generated creative patterns
        """
        current_states = seed_states

        for step in range(num_steps):
            # Apply dream transformation
            current_states = self.forward(current_states)

            # Add slight random perturbation between steps
            if step < num_steps - 1:
                noise = torch.randn_like(current_states) * 0.02
                current_states = current_states + noise

        return current_states

    def set_creativity_level(self, level: float):
        """
        Set the overall creativity level (0.0 = conservative, 1.0 = highly creative).

        Args:
            level: Creativity level between 0.0 and 1.0
        """
        with torch.no_grad():
            self.lucidity.data = torch.clamp(torch.tensor(level), 0.0, 1.0)


# Utility functions for sleep/dream cycles
def sleep_dream_cycle(
    hidden_states: torch.Tensor, sleep_net: SleepNet, dream_net: DreamNet, sleep_steps: int = 5, dream_steps: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform a complete sleep-dream cycle for memory consolidation and creativity.

    Args:
        hidden_states: Input hidden states
        sleep_net: SleepNet for consolidation
        dream_net: DreamNet for creativity
        sleep_steps: Number of sleep consolidation steps
        dream_steps: Number of dream generation steps

    Returns:
        Tuple of (consolidated_states, creative_states)
    """
    # Sleep phase: consolidation
    sleep_states = hidden_states
    for _ in range(sleep_steps):
        sleep_states = sleep_net(sleep_states)

    # Dream phase: creative exploration
    dream_states = dream_net.generate_creative_patterns(sleep_states, num_steps=dream_steps)

    return sleep_states, dream_states
