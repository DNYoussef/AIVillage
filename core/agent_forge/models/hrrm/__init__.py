"""
HRRRM: Hierarchical Recurrent Reasoning Memory

Three ~50M parameter models implementing HRM and Titans algorithms:
- Planner: HRM loop + ControllerHead for DSL planning tokens
- Reasoner: HRM loop + ScratchpadSupervisor for reasoning spans
- Memory: Base transformer + Titans test-time learning memory

Designed for fast Agent Forge EvoMerge iteration.
"""

__version__ = "0.1.0"
