"""Distributed Agent System for Sprint 7.

This module implements distributed deployment of the 18 AIVillage agents across
the P2P network, enabling cross-device collaboration and agent migration.

Key Components:
- DistributedAgentOrchestrator: Main orchestrator for agent deployment
- AgentMigrationManager: Handles agent migration between devices
- AgentRegistry: Tracks agent locations and capabilities
"""

from .agent_migration_manager import AgentMigrationManager, MigrationReason
from .agent_registry import AgentLocation, DistributedAgentRegistry
from .distributed_agent_orchestrator import AgentDeploymentPlan, DistributedAgentOrchestrator

__all__ = [
    "AgentDeploymentPlan",
    "AgentLocation",
    "AgentMigrationManager",
    "DistributedAgentOrchestrator",
    "DistributedAgentRegistry",
    "MigrationReason",
]
