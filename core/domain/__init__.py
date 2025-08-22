"""
Domain Layer - Core Business Entities and Rules

Contains the fundamental business concepts, entities, value objects,
and domain policies that drive AIVillage functionality.

Following Domain-Driven Design patterns with connascence awareness:
- Entities have identity and lifecycle
- Value objects are immutable and compared by value
- Services contain domain logic that doesn't belong in entities
- Policies encapsulate business rules and constraints
"""

from .entities import *
from .policies import *
from .services import *

__all__ = ["entities", "services", "policies"]
