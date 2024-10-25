"""MAGI Agent package."""

from agents.magi.core import (
    MAGIAgent,
    TaskResearch,
    QualityAssuranceLayer,
    EvolutionManager,
    ContinuousLearner,
    GraphManager,
    ProjectPlanner,
    KnowledgeManager
)

from agents.magi.tools import (
    ToolPersistence,
    ToolCreator,
    ToolManager,
    ToolOptimizer
)

__all__ = [
    'MAGIAgent',
    'TaskResearch',
    'QualityAssuranceLayer',
    'EvolutionManager',
    'ContinuousLearner',
    'GraphManager',
    'ProjectPlanner',
    'KnowledgeManager',
    'ToolPersistence',
    'ToolCreator',
    'ToolManager',
    'ToolOptimizer'
]
