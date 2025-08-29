"""Cogment heads package - input/output heads and task adapters."""

from .vocabulary_optimization import OptimizedVocabularyHeads, TiedVocabularyHeads, FactorizedVocabularyHeads
from .image_head import ImageHead, ARCImageHead
from .text_head import TextHead, CogmentTextHead
from .task_adapters import (
    TaskAdapter,
    ClassificationAdapter,
    RegressionAdapter,
    ARCTaskAdapter,
    TextGenerationAdapter,
    MathTaskAdapter,
)

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