"""
Agent Domain Entity

Represents the core business concept of an AI agent with identity,
capabilities, and lifecycle management. Follows connascence principles
with weak coupling to infrastructure concerns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid


class AgentStatus(Enum):
    """Agent lifecycle status"""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


class AgentCapability(Enum):
    """Core agent capabilities - extensible by domain"""

    REASONING = "reasoning"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    PLANNING = "planning"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    COORDINATION = "coordination"
    CREATIVITY = "creativity"


@dataclass
class AgentId:
    """Agent identifier value object"""

    value: str

    def __post_init__(self):
        if not self.value or not isinstance(self.value, str):
            raise ValueError("AgentId must be a non-empty string")

    @classmethod
    def generate(cls) -> AgentId:
        """Generate new unique agent ID"""
        return cls(str(uuid.uuid4()))

    def __str__(self) -> str:
        return self.value


@dataclass
class Agent:
    """
    Core Agent domain entity

    Represents an AI agent with identity, capabilities, and business logic.
    Infrastructure concerns (RAG, P2P, etc.) are injected as dependencies.
    """

    # Identity
    id: AgentId
    name: str
    agent_type: str

    # Core properties
    capabilities: list[AgentCapability]
    status: AgentStatus = AgentStatus.INITIALIZING
    specialized_role: str | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Performance tracking
    tasks_completed: int = 0
    success_rate: float = 1.0
    average_response_time_ms: float = 0.0

    def __post_init__(self):
        """Validate agent invariants"""
        if not self.name.strip():
            raise ValueError("Agent name cannot be empty")

        if not self.agent_type.strip():
            raise ValueError("Agent type cannot be empty")

        if not self.capabilities:
            raise ValueError("Agent must have at least one capability")

    def activate(self) -> None:
        """Transition agent to active status"""
        if self.status == AgentStatus.TERMINATED:
            raise ValueError("Cannot activate terminated agent")

        self.status = AgentStatus.ACTIVE
        self.last_active = datetime.now()

    def suspend(self) -> None:
        """Suspend agent operations"""
        if self.status == AgentStatus.TERMINATED:
            raise ValueError("Cannot suspend terminated agent")

        self.status = AgentStatus.SUSPENDED

    def terminate(self) -> None:
        """Permanently terminate agent"""
        self.status = AgentStatus.TERMINATED

    def update_performance(self, response_time_ms: float, success: bool) -> None:
        """Update performance metrics"""
        # Update average response time
        if self.tasks_completed > 0:
            self.average_response_time_ms = (
                self.average_response_time_ms * self.tasks_completed + response_time_ms
            ) / (self.tasks_completed + 1)
        else:
            self.average_response_time_ms = response_time_ms

        # Update success rate
        if self.tasks_completed > 0:
            current_successes = self.success_rate * self.tasks_completed
            new_successes = current_successes + (1 if success else 0)
            self.success_rate = new_successes / (self.tasks_completed + 1)
        else:
            self.success_rate = 1.0 if success else 0.0

        self.tasks_completed += 1
        self.last_active = datetime.now()

    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has specific capability"""
        return capability in self.capabilities

    def add_capability(self, capability: AgentCapability) -> None:
        """Add new capability to agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)

    def remove_capability(self, capability: AgentCapability) -> None:
        """Remove capability from agent"""
        if capability in self.capabilities:
            self.capabilities.remove(capability)

    def is_healthy(self) -> bool:
        """Check if agent is in healthy state"""
        return (
            self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
            and self.success_rate >= 0.7
            and self.average_response_time_ms < 10000  # 10 seconds
        )

    def get_specialization_domain(self) -> str:
        """Get agent's specialization domain for organization"""
        domain_mapping = {
            "king": "governance",
            "shield": "governance",
            "sword": "governance",
            "auditor": "governance",
            "legal": "governance",
            "sage": "knowledge",
            "oracle": "knowledge",
            "curator": "knowledge",
            "shaman": "knowledge",
            "strategist": "knowledge",
            "coordinator": "infrastructure",
            "navigator": "infrastructure",
            "gardener": "infrastructure",
            "magi": "infrastructure",
            "sustainer": "infrastructure",
            "ensemble": "culture_making",
            "horticulturist": "culture_making",
            "maker": "culture_making",
            "banker": "economy",
            "merchant": "economy",
            "medic": "language_education_health",
            "polyglot": "language_education_health",
            "tutor": "language_education_health",
        }

        agent_type_lower = self.agent_type.lower()
        for key, domain in domain_mapping.items():
            if key in agent_type_lower:
                return domain

        return "general"

    def to_dict(self) -> dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            "id": str(self.id),
            "name": self.name,
            "agent_type": self.agent_type,
            "specialized_role": self.specialized_role,
            "capabilities": [cap.value for cap in self.capabilities],
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat() if self.last_active else None,
            "tasks_completed": self.tasks_completed,
            "success_rate": self.success_rate,
            "average_response_time_ms": self.average_response_time_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Agent:
        """Create agent from dictionary representation"""
        return cls(
            id=AgentId(data["id"]),
            name=data["name"],
            agent_type=data["agent_type"],
            specialized_role=data.get("specialized_role"),
            capabilities=[AgentCapability(cap) for cap in data["capabilities"]],
            status=AgentStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_active=datetime.fromisoformat(data["last_active"]) if data.get("last_active") else None,
            tasks_completed=data.get("tasks_completed", 0),
            success_rate=data.get("success_rate", 1.0),
            average_response_time_ms=data.get("average_response_time_ms", 0.0),
            metadata=data.get("metadata", {}),
        )
