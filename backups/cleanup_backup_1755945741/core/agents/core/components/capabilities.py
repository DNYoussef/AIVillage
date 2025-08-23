"""Agent Capabilities Component.

Manages agent capabilities, skills, and specialization configuration.
Handles MCP tool registration and capability-based routing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityLevel(Enum):
    """Capability proficiency levels."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class CapabilityDefinition:
    """Definition of a specific agent capability."""

    capability_id: str
    name: str
    description: str
    level: CapabilityLevel
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    acquired_at: datetime = field(default_factory=datetime.now)


@dataclass
class ToolDefinition:
    """Definition of an MCP tool and its capabilities."""

    tool_id: str
    name: str
    description: str
    required_capabilities: list[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: datetime | None = None
    enabled: bool = True


class MCPTool(ABC):
    """Abstract base class for MCP (Model Control Protocol) tools."""

    def __init__(self, tool_name: str, description: str):
        self.tool_name = tool_name
        self.description = description
        self.usage_count = 0
        self.last_used = None

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given parameters."""
        pass

    def log_usage(self) -> None:
        """Log tool usage for monitoring."""
        self.usage_count += 1
        self.last_used = datetime.now()


class AgentCapabilities:
    """Manages agent capabilities, tools, and specialization configuration.

    This component encapsulates capability management, reducing coupling
    between capability concerns and core agent logic.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """Initialize capabilities manager.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/category of the agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Capability tracking (CoN - Connascence of Name for capability IDs)
        self._capabilities: dict[str, CapabilityDefinition] = {}
        self._tools: dict[str, MCPTool] = {}
        self._tool_definitions: dict[str, ToolDefinition] = {}

        # Specialization configuration
        self._specialized_role = "base_template"
        self._skill_matrix: dict[str, float] = {}  # skill_id -> proficiency (0-1)

        logger.debug(f"Capabilities manager initialized for {agent_type} agent {agent_id}")

    def set_specialized_role(self, role: str) -> None:
        """Set the agent's specialized role.

        Args:
            role: Specialized role identifier (e.g., "architect", "coordinator")
        """
        self._specialized_role = role
        logger.info(f"Agent {self.agent_id} specialized role set to: {role}")

    def get_specialized_role(self) -> str:
        """Get the agent's specialized role."""
        return self._specialized_role

    def add_capability(
        self,
        capability_id: str,
        name: str,
        description: str,
        level: CapabilityLevel = CapabilityLevel.BASIC,
        dependencies: list[str] | None = None,
        **metadata,
    ) -> bool:
        """Add a capability to the agent.

        Args:
            capability_id: Unique identifier for the capability
            name: Human-readable capability name
            description: Detailed capability description
            level: Proficiency level for this capability
            dependencies: List of required capability IDs
            **metadata: Additional capability metadata

        Returns:
            True if capability added successfully
        """
        # Check dependencies are satisfied
        deps = dependencies or []
        for dep_id in deps:
            if dep_id not in self._capabilities:
                logger.warning(f"Capability {capability_id} requires missing dependency: {dep_id}")
                return False

        capability = CapabilityDefinition(
            capability_id=capability_id,
            name=name,
            description=description,
            level=level,
            dependencies=deps,
            metadata=metadata,
        )

        self._capabilities[capability_id] = capability
        logger.info(f"Added capability {capability_id} at {level.value} level")
        return True

    def remove_capability(self, capability_id: str) -> bool:
        """Remove a capability from the agent.

        Args:
            capability_id: ID of capability to remove

        Returns:
            True if capability removed successfully
        """
        if capability_id not in self._capabilities:
            return False

        # Check if other capabilities depend on this one
        dependents = [cap_id for cap_id, cap in self._capabilities.items() if capability_id in cap.dependencies]

        if dependents:
            logger.warning(f"Cannot remove capability {capability_id} - required by: {dependents}")
            return False

        del self._capabilities[capability_id]
        logger.info(f"Removed capability: {capability_id}")
        return True

    def has_capability(self, capability_id: str, min_level: CapabilityLevel | None = None) -> bool:
        """Check if agent has specific capability at minimum level.

        Args:
            capability_id: ID of capability to check
            min_level: Minimum required proficiency level

        Returns:
            True if agent has capability at required level
        """
        if capability_id not in self._capabilities:
            return False

        capability = self._capabilities[capability_id]

        if not capability.enabled:
            return False

        if min_level is None:
            return True

        # Check if current level meets minimum requirement
        level_order = [
            CapabilityLevel.BASIC,
            CapabilityLevel.INTERMEDIATE,
            CapabilityLevel.ADVANCED,
            CapabilityLevel.EXPERT,
        ]

        current_index = level_order.index(capability.level)
        required_index = level_order.index(min_level)

        return current_index >= required_index

    def get_capabilities(self, enabled_only: bool = True) -> dict[str, CapabilityDefinition]:
        """Get all agent capabilities.

        Args:
            enabled_only: If True, return only enabled capabilities

        Returns:
            Dict mapping capability IDs to definitions
        """
        if enabled_only:
            return {cap_id: cap for cap_id, cap in self._capabilities.items() if cap.enabled}
        return self._capabilities.copy()

    def upgrade_capability(self, capability_id: str, new_level: CapabilityLevel) -> bool:
        """Upgrade a capability to a higher proficiency level.

        Args:
            capability_id: ID of capability to upgrade
            new_level: Target proficiency level

        Returns:
            True if upgrade successful
        """
        if capability_id not in self._capabilities:
            logger.warning(f"Cannot upgrade unknown capability: {capability_id}")
            return False

        capability = self._capabilities[capability_id]
        old_level = capability.level
        capability.level = new_level

        logger.info(f"Upgraded capability {capability_id}: {old_level.value} -> {new_level.value}")
        return True

    def register_tool(self, tool: MCPTool, required_capabilities: list[str] | None = None) -> bool:
        """Register an MCP tool with the agent.

        Args:
            tool: MCP tool instance to register
            required_capabilities: Capabilities needed to use this tool

        Returns:
            True if tool registered successfully
        """
        tool_id = tool.tool_name

        # Check if agent has required capabilities
        required_caps = required_capabilities or []
        for cap_id in required_caps:
            if not self.has_capability(cap_id):
                logger.warning(f"Cannot register tool {tool_id} - missing capability: {cap_id}")
                return False

        # Register tool and its definition
        self._tools[tool_id] = tool
        self._tool_definitions[tool_id] = ToolDefinition(
            tool_id=tool_id, name=tool.tool_name, description=tool.description, required_capabilities=required_caps
        )

        logger.info(f"Registered MCP tool: {tool_id}")
        return True

    def unregister_tool(self, tool_id: str) -> bool:
        """Unregister an MCP tool.

        Args:
            tool_id: ID of tool to unregister

        Returns:
            True if tool unregistered successfully
        """
        if tool_id not in self._tools:
            return False

        del self._tools[tool_id]
        if tool_id in self._tool_definitions:
            del self._tool_definitions[tool_id]

        logger.info(f"Unregistered MCP tool: {tool_id}")
        return True

    def get_tool(self, tool_id: str) -> MCPTool | None:
        """Get an MCP tool by ID.

        Args:
            tool_id: ID of tool to retrieve

        Returns:
            MCP tool instance or None if not found
        """
        return self._tools.get(tool_id)

    def get_available_tools(self) -> dict[str, ToolDefinition]:
        """Get all available MCP tools.

        Returns:
            Dict mapping tool IDs to definitions
        """
        return {tool_id: defn for tool_id, defn in self._tool_definitions.items() if defn.enabled}

    async def execute_tool(self, tool_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP tool with capability validation.

        Args:
            tool_id: ID of tool to execute
            parameters: Parameters to pass to the tool

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or capabilities insufficient
        """
        if tool_id not in self._tools:
            raise ValueError(f"Tool not found: {tool_id}")

        tool = self._tools[tool_id]
        tool_def = self._tool_definitions.get(tool_id)

        # Validate required capabilities
        if tool_def:
            for cap_id in tool_def.required_capabilities:
                if not self.has_capability(cap_id):
                    raise ValueError(f"Tool {tool_id} requires capability: {cap_id}")

        # Execute tool and update metrics
        try:
            result = await tool.execute(parameters)
            tool.log_usage()

            # Update tool definition metrics
            if tool_def:
                tool_def.usage_count += 1
                tool_def.last_used = datetime.now()

            logger.debug(f"Executed tool {tool_id} successfully")
            return result

        except Exception as e:
            logger.error(f"Tool execution failed for {tool_id}: {e}")
            raise

    def can_handle_task_type(self, task_type: str) -> bool:
        """Check if agent can handle a specific task type based on capabilities.

        Args:
            task_type: Type of task to evaluate

        Returns:
            True if agent has capabilities to handle task type
        """
        # Task type to capability mapping (CoM - Connascence of Meaning)
        task_capability_map = {
            "query": ["query_processing"],
            "generation": ["text_processing", "generation"],
            "analysis": ["reasoning", "analysis"],
            "communication": ["inter_agent_communication"],
            "coordination": ["task_orchestration", "planning"],
            "specialized": [self._specialized_role],  # Role-specific tasks
        }

        required_capabilities = task_capability_map.get(task_type, [])

        # Check if agent has at least one required capability
        for cap_id in required_capabilities:
            if self.has_capability(cap_id):
                return True

        return len(required_capabilities) == 0  # Unknown task types are allowed

    def set_skill_proficiency(self, skill_id: str, proficiency: float) -> None:
        """Set proficiency level for a specific skill.

        Args:
            skill_id: Identifier for the skill
            proficiency: Proficiency level (0.0 to 1.0)
        """
        proficiency = max(0.0, min(1.0, proficiency))  # Clamp to valid range
        self._skill_matrix[skill_id] = proficiency
        logger.debug(f"Set skill {skill_id} proficiency to {proficiency:.2f}")

    def get_skill_proficiency(self, skill_id: str) -> float:
        """Get proficiency level for a specific skill.

        Args:
            skill_id: Identifier for the skill

        Returns:
            Proficiency level (0.0 to 1.0), or 0.0 if skill not found
        """
        return self._skill_matrix.get(skill_id, 0.0)

    def get_capability_metrics(self) -> dict[str, Any]:
        """Get comprehensive capability metrics and statistics.

        Returns:
            Dict containing capability metrics and tool usage statistics
        """
        total_capabilities = len(self._capabilities)
        enabled_capabilities = sum(1 for cap in self._capabilities.values() if cap.enabled)

        # Capability level distribution
        level_distribution = {}
        for cap in self._capabilities.values():
            level = cap.level.value
            level_distribution[level] = level_distribution.get(level, 0) + 1

        # Tool usage statistics
        total_tools = len(self._tools)
        tool_usage = {
            tool_id: {
                "usage_count": defn.usage_count,
                "last_used": defn.last_used.isoformat() if defn.last_used else None,
            }
            for tool_id, defn in self._tool_definitions.items()
        }

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "specialized_role": self._specialized_role,
            "capabilities": {
                "total": total_capabilities,
                "enabled": enabled_capabilities,
                "level_distribution": level_distribution,
                "capability_ids": list(self._capabilities.keys()),
            },
            "tools": {"total": total_tools, "usage_statistics": tool_usage, "tool_ids": list(self._tools.keys())},
            "skills": {
                "skill_count": len(self._skill_matrix),
                "average_proficiency": (
                    sum(self._skill_matrix.values()) / len(self._skill_matrix) if self._skill_matrix else 0.0
                ),
                "skill_matrix": self._skill_matrix.copy(),
            },
        }

    def get_capability_summary(self) -> dict[str, Any]:
        """Get a summary of agent capabilities for external interfaces.

        Returns:
            Dict with essential capability information
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "specialized_role": self._specialized_role,
            "capabilities": [
                {"id": cap.capability_id, "name": cap.name, "level": cap.level.value, "enabled": cap.enabled}
                for cap in self._capabilities.values()
            ],
            "available_tools": list(self._tools.keys()),
            "can_handle_tasks": [
                task_type
                for task_type in ["query", "generation", "analysis", "communication", "coordination"]
                if self.can_handle_task_type(task_type)
            ],
        }
