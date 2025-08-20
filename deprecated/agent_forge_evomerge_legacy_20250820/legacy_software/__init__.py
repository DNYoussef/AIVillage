"""
Agent Forge - Automated Agent Creation Process

Complete 10-stage process for creating new meta-agents:
1. Model Selection (3 candidates from Hugging Face)
2. EvoMerge Pipeline (50-generation evolution)
3. Model Compression (BitNet + VPTQ)
4. Prompt Baking (Quiet Star + thought bubbles)
5. Two-stage Compression (smaller, trainable)
6. Skill Analysis & Training Problem Generation
7. Geometric Self-Awareness Training
8. Sleep/Dream Cycle Learning
9. Self-Modeling & Temperature Understanding
10. Final Integration & Testing
"""

from .compression import CompressionPipeline
from .evolution import EvoMergeOrchestrator
from .prompt_baking import PromptBaker
from .testing import AgentValidator
from .training import TrainingOrchestrator

__all__ = [
    "EvoMergeOrchestrator",
    "CompressionPipeline",
    "PromptBaker",
    "TrainingOrchestrator",
    "AgentValidator",
]
