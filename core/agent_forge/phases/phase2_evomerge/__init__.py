"""
Phase 2: EvoMerge - Evolutionary Model Optimization
"""

from .evomerge import EvoMerge, EvoMergeConfig
from .merge_techniques import MergeTechniques
from .fitness_evaluator import FitnessEvaluator
from .population_manager import PopulationManager
from .genetic_operations import GeneticOperations

__all__ = [
    'EvoMerge',
    'EvoMergeConfig',
    'MergeTechniques',
    'FitnessEvaluator',
    'PopulationManager',
    'GeneticOperations'
]

__version__ = '1.0.0'