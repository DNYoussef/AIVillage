"""HypeRAG MCP Server.

A Model Context Protocol server for HypeRAG knowledge retrieval and reasoning.
Provides brain-inspired dual-memory architecture with creativity, repair, and safety validation.
"""

__version__ = "1.0.0"
__author__ = "AIVillage Team"


# Delayed imports to avoid dependency issues during testing
def get_server():
    from .server import HypeRAGMCPServer

    return HypeRAGMCPServer


# Only import what's essential
try:
    from .auth import HypeRAGPermissions, PermissionManager
except ImportError:
    HypeRAGPermissions = None
    PermissionManager = None

try:
    from .models import AgentReasoningModel, ModelRegistry
except ImportError:
    AgentReasoningModel = None
    ModelRegistry = None

try:
    from .protocol import MCPProtocolHandler
except ImportError:
    MCPProtocolHandler = None

__all__ = [
    "AgentReasoningModel",
    "HypeRAGPermissions",
    "MCPProtocolHandler",
    "ModelRegistry",
    "PermissionManager",
    "get_server",
]
