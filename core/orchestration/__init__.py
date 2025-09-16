"""
Unified Orchestration System

This module consolidates the 4 overlapping orchestration systems into a unified,
coherent architecture following the Agent 2 architectural blueprint:

- UnifiedPipeline (ML pipeline orchestration)
- CognativeNexusController (agent lifecycle orchestration)  
- CognitiveNexus (cognitive analysis orchestration)
- FogCoordinator (distributed system orchestration)

The unified system provides:
1. Common orchestration interface and base classes
2. Coordination protocol to prevent initialization conflicts
3. Consolidated background process management
4. Standardized result types and error handling
5. Shared configuration management
"""

from .interfaces import (
    OrchestrationInterface,
    OrchestrationResult,
    OrchestrationStatus,
    TaskContext,
    TaskPriority,
    TaskType,
)
from .base import BaseOrchestrator
from .coordinator import OrchestrationCoordinator
from .registry import OrchestratorRegistry

# Specialized orchestrators
from .ml_orchestrator import MLPipelineOrchestrator
from .agent_orchestrator import AgentLifecycleOrchestrator  
from .cognitive_orchestrator import CognitiveAnalysisOrchestrator
from .fog_orchestrator import FogSystemOrchestrator

# Main unified system
from .unified_orchestrator import UnifiedOrchestrator

__all__ = [
    # Core interfaces
    'OrchestrationInterface',
    'OrchestrationResult', 
    'OrchestrationStatus',
    'TaskContext',
    'TaskPriority',
    'TaskType',
    
    # Base classes
    'BaseOrchestrator',
    'OrchestrationCoordinator',
    'OrchestratorRegistry',
    
    # Specialized orchestrators
    'MLPipelineOrchestrator',
    'AgentLifecycleOrchestrator',
    'CognitiveAnalysisOrchestrator', 
    'FogSystemOrchestrator',
    
    # Main unified system
    'UnifiedOrchestrator',
]