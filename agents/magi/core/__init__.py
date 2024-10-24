"""Core MAGI components."""

from agents.magi.core.magi_agent import MagiAgent
from agents.magi.core.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig
from agents.magi.core.magi_planning import MagiPlanning, GraphManager
from agents.magi.core.continuous_learner import ContinuousLearner
from agents.magi.core.evolution_manager import EvolutionManager
from agents.magi.core.quality_assurance_layer import QualityAssuranceLayer
from agents.magi.core.orchestration import TaskQueue, create_agents

__all__ = [
    'MagiAgent',
    'UnifiedBaseAgent',
    'UnifiedAgentConfig',
    'MagiPlanning',
    'GraphManager',
    'ContinuousLearner',
    'EvolutionManager',
    'QualityAssuranceLayer',
    'TaskQueue',
    'create_agents'
]
