"""MAGI Agent Interfaces

This module contains various interface implementations for MAGI agents,
providing different levels of functionality and safety for production use.
"""

from .magi_interface import *
from .safe_magi_interface import *
from .simple_magi_interface import *

__all__ = [
    # Interface modules will define their own exports
]
