"""Agent Components - Focused responsibility classes extracted from BaseAgentTemplate.

This module provides focused, single-responsibility components that replace
the monolithic BaseAgentTemplate with clean, testable, and maintainable classes.

Components follow connascence principles:
- Strong connascence kept local within classes
- Weak connascence for inter-component communication
- Dependency injection for configuration and external services
"""

from .capabilities import AgentCapabilities
from .communication import AgentCommunication
from .configuration import AgentConfiguration
from .metrics import AgentMetrics
from .state_manager import AgentStateManager

__all__ = ["AgentCommunication", "AgentConfiguration", "AgentCapabilities", "AgentMetrics", "AgentStateManager"]
