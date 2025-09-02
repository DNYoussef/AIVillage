"""
Core Business Logic Layer

This module contains the core business logic for AIVillage, organized according to
clean architecture principles and connascence-based coupling management.

Architecture:
- domain/: Core entities, value objects, and business rules
- agents/: Domain-organized agent implementations
- agent_forge/: Training and evolution business logic
- rag/: Knowledge retrieval and reasoning business logic

Principles:
- Weak connascence across module boundaries
- Strong connascence localized within modules
- Dependency injection for infrastructure concerns
- Behavioral contracts over implementation details
"""

# Explicit imports to avoid star import issues
from . import agents, domain, rag

__version__ = "1.0.0"
__all__ = ["domain", "agents", "rag", "agent_forge"]
