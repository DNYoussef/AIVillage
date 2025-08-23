"""
Cogment Integration Package for Agent Forge.

This package provides integration between the single Cogment model (23.7M parameters)
and the existing Agent Forge EvoMerge pipeline, replacing the 3-model HRRM approach
(Planner 50M + Reasoner 50M + Memory 50M = 150M total).

Key benefits of Cogment integration:
- 6x parameter reduction (150M → 23.7M)
- 6x faster evolutionary operations
- Unified architecture with ACT halting and LTM
- Single model deployment vs 3 separate models
- Preserved specialized capabilities through integrated heads

Components:
- EvoMergeAdapter: Adapts EvoMerge for single Cogment model workflow
- PhaseController: Replaces 3-phase HRRM with 4-stage Cogment curriculum
- HFExport: HuggingFace export for unified Cogment model
- ModelCompatibility: ACT and LTM preservation during merging
- DeploymentManager: Production deployment pipeline
"""

from .deployment_manager import CogmentDeploymentManager
from .evomerge_adapter import CogmentEvoMergeAdapter
from .hf_export import CogmentHFExporter
from .model_compatibility import CogmentCompatibilityValidator
from .phase_controller import CogmentPhaseController

__all__ = [
    "CogmentEvoMergeAdapter",
    "CogmentPhaseController",
    "CogmentHFExporter",
    "CogmentCompatibilityValidator",
    "CogmentDeploymentManager",
]

__version__ = "1.0.0"
