"""
GatedLTM: Titan-style Long-Term Memory for Cogment.

Implements surprise-gated writes and cross-attention reads for efficient
episodic memory with controlled slot competition and decay.
"""

from .gated_ltm import GatedLTMMemory
from .cross_attention import CrossAttentionReader
from .memory_gates import SurpriseGate, MemoryWriter
from .memory_utils import MemoryDecay, surprisal_from_loss

__all__ = [
    'GatedLTMMemory',
    'CrossAttentionReader', 
    'SurpriseGate',
    'MemoryWriter',
    'MemoryDecay',
    'surprisal_from_loss',
]