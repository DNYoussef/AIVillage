"""
HypeRAG MCP Server

A Model Context Protocol server for HypeRAG knowledge retrieval and reasoning.
Provides brain-inspired dual-memory architecture with creativity, repair, and safety validation.
"""

__version__ = "1.0.0"
__author__ = "AIVillage Team"

from .server import HypeRAGMCPServer
from .auth import HypeRAGPermissions, PermissionManager
from .models import AgentReasoningModel, ModelRegistry
from .protocol import MCPProtocolHandler

__all__ = [
    "HypeRAGMCPServer",
    "HypeRAGPermissions",
    "PermissionManager",
    "AgentReasoningModel",
    "ModelRegistry",
    "MCPProtocolHandler"
]
