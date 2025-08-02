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
    from .education import curriculum_graph, eli5_chain
except ImportError:
    # Handle optional dependencies gracefully
    pass