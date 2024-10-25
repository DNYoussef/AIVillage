"""Register core reasoning techniques with the technique registry."""

from .registry import TechniqueRegistry
from .multi_path_exploration import MultiPathExploration
from .scale_aware_solving import ScaleAwareSolving
from .perspective_shifting import PerspectiveShifting
from .progressive_refinement import ProgressiveRefinement
from .pattern_integration import PatternIntegration
from .controlled_disruption import ControlledDisruption
from .solution_unit_manipulation import SolutionUnitManipulation

def register_core_techniques(registry: TechniqueRegistry) -> None:
    """Register all core reasoning techniques."""
    
    # Multi-Path Exploration
    registry.register(
        technique_class=MultiPathExploration,
        name="multi_path_exploration",
        description="Maintains and evolves multiple solution paths simultaneously",
        tags=["exploration", "evolution", "convergence"],
        parameters={
            "max_paths": 5,
            "convergence_threshold": 0.8,
            "cross_pollination_rate": 0.3
        }
    )
    
    # Scale-Aware Problem Solving
    registry.register(
        technique_class=ScaleAwareSolving,
        name="scale_aware_solving",
        description="Analyzes problems at different scales while maintaining consistency",
        tags=["scale", "consistency", "patterns"],
        parameters={
            "scales": ["micro", "meso", "macro"]
        }
    )
    
    # Perspective Shifting
    registry.register(
        technique_class=PerspectiveShifting,
        name="perspective_shifting",
        description="Systematically shifts between different perspectives and domains",
        tags=["perspective", "insight", "assumptions"],
        parameters={
            "perspectives": [
                "technical",
                "human",
                "process",
                "strategic",
                "ethical",
                "creative"
            ]
        }
    )
    
    # Progressive Refinement
    registry.register(
        technique_class=ProgressiveRefinement,
        name="progressive_refinement",
        description="Balances exploration and exploitation in solution refinement",
        tags=["refinement", "exploration", "exploitation"],
        parameters={
            "max_iterations": 10,
            "initial_exploration_rate": 0.7,
            "exploration_decay": 0.9,
            "improvement_threshold": 0.05
        }
    )
    
    # Pattern Integration
    registry.register(
        technique_class=PatternIntegration,
        name="pattern_integration",
        description="Combines bottom-up and top-down analysis to identify patterns",
        tags=["patterns", "integration", "analysis"],
        parameters={
            "confidence_threshold": 0.7,
            "similarity_threshold": 0.6
        }
    )
    
    # Controlled Disruption
    registry.register(
        technique_class=ControlledDisruption,
        name="controlled_disruption",
        description="Introduces controlled errors for insight generation",
        tags=["disruption", "insight", "robustness"],
        parameters={
            "max_magnitude": 0.5,
            "safety_threshold": 0.8,
            "learning_rate": 0.1
        }
    )
    
    # Solution Unit Manipulation
    registry.register(
        technique_class=SolutionUnitManipulation,
        name="solution_unit_manipulation",
        description="Breaks down and recombines solution units with human factors",
        tags=["decomposition", "recombination", "human-factors"],
        parameters={
            "min_unit_size": 3,
            "max_unit_size": 10,
            "human_factor_weights": {
                "comprehensibility": 0.3,
                "usability": 0.3,
                "learnability": 0.2,
                "emotional_impact": 0.2
            }
        }
    )
