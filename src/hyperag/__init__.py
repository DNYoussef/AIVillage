"""HyperAG - Advanced AI Agent System.

This module provides advanced AI agent capabilities including:
- Educational systems (ELI5 chain, curriculum graphs)
- Intelligent reasoning and learning
- Multi-modal agent interactions
"""

__version__ = "1.0.0"
__author__ = "AIVillage Team"

# Import key components
try:
    from .education import (
        ConceptNode,
        CurriculumGraph,
        DifficultyLevel,
        ELI5Chain,
        ELI5Explanation,
        LearningPath,
        LearningStyle,
        add_concept_to_curriculum,
        curriculum_graph,
        eli5_chain,
        explain_concept_eli5,
        generate_learning_path_for_topic,
    )

    EDUCATION_AVAILABLE = True

except ImportError as e:
    # Handle optional dependencies gracefully
    EDUCATION_AVAILABLE = False
    import logging

    logging.warning(f"Education module not fully available: {e}")

# Export public API
__all__ = []
if EDUCATION_AVAILABLE:
    __all__.extend(
        [
            "curriculum_graph",
            "eli5_chain",
            "CurriculumGraph",
            "ELI5Chain",
            "ConceptNode",
            "LearningPath",
            "ELI5Explanation",
            "DifficultyLevel",
            "LearningStyle",
            "add_concept_to_curriculum",
            "generate_learning_path_for_topic",
            "explain_concept_eli5",
        ]
    )
