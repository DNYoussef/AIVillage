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

# Core components
from .core.phase_controller import PhaseController, PhaseResult
from .core.unified_pipeline import UnifiedConfig, UnifiedPipeline

# Available phases - import what exists
from .phases import (
    ADASPhase,
    BitNetCompressionPhase,
    CognatePhase,
    EvoMergePhase,
    FinalCompressionPhase,
    ForgeTrainingPhase,
    QuietSTaRPhase,
    ToolPersonaBakingPhase,
)

# Training components (comment out if missing)
# from .training.forge_trainer import ForgeTrainConfig, ForgeTrainer

__version__ = "1.0.0"

__all__ = [
    "UnifiedPipeline",
    "UnifiedConfig",
    "PhaseController",
    "PhaseResult",
    "ADASPhase",
    "BitNetCompressionPhase",
    "CognatePhase",
    "EvoMergePhase",
    "FinalCompressionPhase",
    "ForgeTrainingPhase",
    "QuietSTaRPhase",
    "ToolPersonaBakingPhase",
]
