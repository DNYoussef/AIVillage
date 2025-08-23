"""
MCP Layer - Model Control Protocol

Standardized interfaces for agent capabilities, memory management,
and knowledge retrieval services.
"""

__version__ = "1.0.0"
__author__ = "AIVillage Team"

# MCP Layer Components
from . import memory, servers, tools

__all__ = ["tools", "memory", "servers"]
