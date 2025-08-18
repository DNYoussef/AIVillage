"""
Agent Forge: Unified AI Agent Development Pipeline

A comprehensive system for creating, training, and deploying specialized AI agents
using cutting-edge techniques including evolutionary merging, self-modeling,
and advanced compression.

Main Components:
- Unified Pipeline: End-to-end orchestration of all phases
- Phase Controllers: Individual phase implementations
- Training Systems: Forge training loop with grokfast, edge-of-chaos, etc.
- Evolution Engine: Evolutionary optimization and merging
- Compression Pipeline: BitNet, SeedLM, VPTQ compression
- ADAS System: Automated architecture search
"""

from .core.forge_orchestrator import ForgeOrchestrator
from .core.unified_pipeline import UnifiedConfig, UnifiedPipeline
from .phases import (
    ADASPhase,
    CompressionPhase,
    EvoMergePhase,
    GeometryLearningPhase,
    PromptBakingPhase,
    SelfModelingPhase,
)
from .training.forge_trainer import ForgeTrainConfig, ForgeTrainer

__version__ = "1.0.0"

__all__ = [
    "UnifiedPipeline",
    "UnifiedConfig",
    "ForgeOrchestrator",
    "ForgeTrainer",
    "ForgeTrainConfig",
    "EvoMergePhase",
    "GeometryLearningPhase",
    "SelfModelingPhase",
    "PromptBakingPhase",
    "ADASPhase",
    "CompressionPhase",
]
