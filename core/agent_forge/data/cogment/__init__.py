"""
Cogment Data Pipeline

4-Stage Curriculum Dataset Management for replacing HRRM with comprehensive real-world training.

Provides:
- Stage 0: Sanity checks (synthetic tasks, 500 steps)
- Stage 1: ARC visual reasoning (~300 augmentations, 4K steps)  
- Stage 2: Algorithmic puzzles (Sudoku, Mazes, ListOps, 8K steps)
- Stage 3: Math & text reasoning (GSM8K, HotpotQA, 16K steps)
- Stage 4: Long-context (LongBench, SCROLLS, 32K steps)

Replaces HRRM's limited synthetic approach with comprehensive real-world datasets
for accelerated grokking through curriculum-based progressive training.
"""

from .augmentations import ARCAugmentationEngine
from .data_manager import CogmentDataManager
from .stage_0_sanity import SanityCheckDataset
from .stage_1_arc import ARCVisualDataset
from .stage_2_puzzles import AlgorithmicPuzzleDataset
from .stage_3_reasoning import MathTextReasoningDataset
from .stage_4_longcontext import LongContextDataset

__all__ = [
    "CogmentDataManager",
    "SanityCheckDataset",
    "ARCVisualDataset",
    "AlgorithmicPuzzleDataset",
    "MathTextReasoningDataset",
    "LongContextDataset",
    "ARCAugmentationEngine",
]

__version__ = "1.0.0"
__description__ = "Cogment 4-Stage Curriculum Data Pipeline"
