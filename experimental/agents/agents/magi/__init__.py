"""
MAGI Agent System

The MAGI (Machine-learning Artificial General Intelligence) agent system
provides specialized AI agents with geometric self-awareness, self-modification
capabilities, and advanced reasoning skills.

Directory Structure:
- core/: Core MAGI agent functionality
- interfaces/: Various interface implementations
- implementations/: Different MAGI agent implementations
- deployment/: Production deployment scripts
- validation/: Testing and validation tools
- research/: Research and experimental features
"""

# Import core MAGI functionality
from .magi_agent import *

# Import specialized modules
from . import interfaces
from . import implementations
from . import deployment
from . import validation

__all__ = [
    # Core exports
    "interfaces",
    "implementations",
    "deployment",
    "validation",
    # Additional exports from magi_agent will be included
]
