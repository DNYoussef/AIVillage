"""
Cogment Training System.

Implements 4-stage curriculum training with GrokFast integration for accelerated grokking.
Replaces HRRM's 3-phase approach with enhanced loss functions and stage-specific optimization.
"""

from .curriculum import CurriculumStage, FourStageCurriculum, StageConfig
from .evaluator import ConvergenceDetector, EvaluationMetrics, StageEvaluator
from .grokfast_integration import CogmentGrokFastOptimizer, GrokFastConfig, SelectiveGrokFastManager
from .losses import CogmentLoss, ConsistencyLoss, DeepSupervisionLoss, PonderLoss, ResidualImprovementLoss
from .trainer import CogmentTrainer, MultiOptimizerConfig, TrainingConfig

__all__ = [
    # Loss functions
    "DeepSupervisionLoss",
    "ResidualImprovementLoss",
    "ConsistencyLoss",
    "PonderLoss",
    "CogmentLoss",
    # Curriculum
    "CurriculumStage",
    "FourStageCurriculum",
    "StageConfig",
    # GrokFast integration
    "CogmentGrokFastOptimizer",
    "SelectiveGrokFastManager",
    "GrokFastConfig",
    # Training
    "CogmentTrainer",
    "TrainingConfig",
    "MultiOptimizerConfig",
    # Evaluation
    "StageEvaluator",
    "EvaluationMetrics",
    "ConvergenceDetector",
]
