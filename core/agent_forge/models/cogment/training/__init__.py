"""
Cogment Training System.

Implements 4-stage curriculum training with GrokFast integration for accelerated grokking.
Replaces HRRM's 3-phase approach with enhanced loss functions and stage-specific optimization.
"""

from .losses import DeepSupervisionLoss, ResidualImprovementLoss, ConsistencyLoss, PonderLoss, CogmentLoss

from .curriculum import CurriculumStage, FourStageCurriculum, StageConfig

from .grokfast_integration import CogmentGrokFastOptimizer, SelectiveGrokFastManager, GrokFastConfig

from .trainer import CogmentTrainer, TrainingConfig, MultiOptimizerConfig

from .evaluator import StageEvaluator, EvaluationMetrics, ConvergenceDetector

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
