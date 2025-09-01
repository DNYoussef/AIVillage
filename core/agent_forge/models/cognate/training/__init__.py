"""
Cognate Training System

This module provides comprehensive training functionality for the Cognate model,
including GrokFast optimization, train-many/infer-few paradigm support,
and integration with the Agent Forge pipeline.
"""

from .trainer import (
    CognateTrainer,
    CognateTrainingConfig,
    TrainingMetrics,
)

from .grokfast_optimizer import (
    GrokFastOptimizer,
    GrokFastConfig,
)

from .orchestrator import (
    TrainingOrchestrator,
    create_training_pipeline,
)

__all__ = [
    "CognateTrainer",
    "CognateTrainingConfig",
    "TrainingMetrics",
    "GrokFastOptimizer",
    "GrokFastConfig",
    "TrainingOrchestrator",
    "create_training_pipeline",
]
