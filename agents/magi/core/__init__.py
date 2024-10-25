"""Core MAGI components."""

from .magi_agent import MAGIAgent
from .task_research import TaskResearch
from .quality_assurance_layer import QualityAssuranceLayer
from .evolution_manager import EvolutionManager
from .continuous_learner import ContinuousLearner
from .magi_planning import GraphManager
from .project_planner import ProjectPlanner
from .knowledge_manager import KnowledgeManager

__all__ = [
    'MAGIAgent',
    'TaskResearch',
    'QualityAssuranceLayer',
    'EvolutionManager',
    'ContinuousLearner',
    'GraphManager',
    'ProjectPlanner',
    'KnowledgeManager'
]
