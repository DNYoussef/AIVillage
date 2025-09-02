"""Cogment heads package - input/output heads and task adapters."""

from .image_head import ARCImageHead, ImageHead
from .task_adapters import (
    ARCTaskAdapter,
    ClassificationAdapter,
    MathTaskAdapter,
    RegressionAdapter,
    TaskAdapter,
    TextGenerationAdapter,
)
from .text_head import CogmentTextHead, TextHead
from .vocabulary_optimization import FactorizedVocabularyHeads, OptimizedVocabularyHeads, TiedVocabularyHeads

__all__ = [
    # Vocabulary optimization
    "OptimizedVocabularyHeads",
    "TiedVocabularyHeads",
    "FactorizedVocabularyHeads",
    # Input heads
    "ImageHead",
    "ARCImageHead",
    "TextHead",
    "CogmentTextHead",
    # Task adapters
    "TaskAdapter",
    "ClassificationAdapter",
    "RegressionAdapter",
    "ARCTaskAdapter",
    "TextGenerationAdapter",
    "MathTaskAdapter",
]
