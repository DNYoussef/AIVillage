"""Base classes for the Agent Forge system.

Provides the foundational abstractions for the 18-agent ecosystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AgentCapability(Enum):
    """Capabilities that agents can possess."""

    # Core capabilities
    REASONING = "reasoning"
    CODING = "coding"
    RESEARCH = "research"
    COORDINATION = "coordination"

    # Specialized capabilities
    TRANSLATION = "translation"
    MEDICAL_ADVISORY = "medical_advisory"
    LEGAL_COMPLIANCE = "legal_compliance"
    FINANCIAL_ANALYSIS = "financial_analysis"
    CREATIVE_GENERATION = "creative_generation"
    SECURITY_ANALYSIS = "security_analysis"
    PHYSICS_SIMULATION = "physics_simulation"
    EDUCATION = "education"
    LOGISTICS = "logistics"
    SUSTAINABILITY = "sustainability"


class AgentRole(Enum):
    """The 18 meta-agents from Atlantis vision."""

    # Leadership & Coordination
    KING = "king"  # Task orchestration
    STRATEGIST = "strategist"  # Long-range planning

    # Knowledge & Research
    SAGE = "sage"  # Deep research
    ORACLE = "oracle"  # Physics-first emulator
    CURATOR = "curator"  # Privacy & dataset management

    # Creation & Development
    MAGI = "magi"  # Code generation
    MAKER = "maker"  # CAD & 3D printing
    ENSEMBLE = "ensemble"  # Creative generation

    # Operations & Infrastructure
    GARDENER = "gardener"  # Edge infrastructure
    NAVIGATOR = "navigator"  # Supply chain
    SUSTAINER = "sustainer"  # Eco-design

    # Protection & Compliance
    SWORD_SHIELD = "sword_shield"  # Security
    LEGAL = "legal"  # Legal compliance
    AUDITOR = "auditor"  # Financial risk

    # Human Services
    MEDIC = "medic"  # Health advisory
    TUTOR = "tutor"  # Education
    POLYGLOT = "polyglot"  # Translation
    SHAMAN = "shaman"  # Alignment & philosophy


@dataclass
class AgentSpecialization:
    """Defines an agent's specialization."""

    role: AgentRole | str
    primary_capabilities: list[AgentCapability]
    secondary_capabilities: list[AgentCapability]
    performance_metrics: dict[str, float]
    resource_requirements: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        role_value = self.role.value if isinstance(self.role, AgentRole) else self.role
        return {
            "role": role_value,
            "primary_capabilities": [c.value for c in self.primary_capabilities],
            "secondary_capabilities": [c.value for c in self.secondary_capabilities],
            "performance_metrics": self.performance_metrics,
            "resource_requirements": self.resource_requirements,
        }


class BaseMetaAgent(ABC):
    """Abstract base class for all Atlantis meta-agents."""

    def __init__(self, specialization: AgentSpecialization) -> None:
        self.specialization = specialization
        self.performance_history = []
        self.kpi_scores = {}
        self.agent_id = f"{specialization.role}_{id(self)}"
        self.name = getattr(specialization.role, "value", str(specialization.role))

    @abstractmethod
    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a request and return a response."""
        pass

    @abstractmethod
    def evaluate_kpi(self) -> dict[str, float]:
        """Evaluate current Key Performance Indicators."""
        pass

    def update_performance(self, performance_data: dict[str, Any]) -> None:
        """Update performance tracking."""
        self.performance_history.append(performance_data)

        # Keep last 1000 records for performance
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    def get_capabilities(self) -> list[str]:
        """Get list of agent capabilities."""
        primary = [c.value for c in self.specialization.primary_capabilities]
        secondary = [c.value for c in self.specialization.secondary_capabilities]
        return primary + secondary

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": getattr(self.specialization.role, "value", str(self.specialization.role)),
            "capabilities": self.get_capabilities(),
            "performance_records": len(self.performance_history),
            "current_kpis": self.kpi_scores,
        }
