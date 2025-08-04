#!/usr/bin/env python3
"""Agent Factory for creating specialized agents from templates.
Auto-generated from agent specifications.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.refactor_agent_forge import AgentRole, AgentSpecialization, BaseMetaAgent


class AgentFactory:
    """Factory for creating agents from templates."""

    def __init__(self, template_dir: str = "production/agent_forge/templates") -> None:
        """Initialize the factory and load templates."""
        self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
        self.agent_classes = self._initialize_agent_classes()

    def _load_templates(self) -> dict[str, dict[str, Any]]:
        """Load all agent templates from the agents directory."""
        templates: dict[str, dict[str, Any]] = {}
        agents_dir = self.template_dir / "agents"

        for template_file in agents_dir.glob("*.json"):
            try:
                with open(template_file) as f:
                    template = json.load(f)
                agent_id = template_file.stem
                templates[agent_id] = template
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")

        return templates

    def _initialize_agent_classes(self) -> dict[str, type]:
        """Initialize specialized agent classes."""
        # Import specialized implementations
        agent_classes = {}

        # King Agent
        try:
            from production.agents.king import KingAgent

            agent_classes["king"] = KingAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["king"] = self._create_generic_agent_class("king")

        # Magi Agent
        try:
            from production.agents.magi import MagiAgent

            agent_classes["magi"] = MagiAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["magi"] = self._create_generic_agent_class("magi")

        # Sage Agent
        try:
            from production.agents.sage import SageAgent

            agent_classes["sage"] = SageAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["sage"] = self._create_generic_agent_class("sage")

        # Gardener Agent
        try:
            from production.agents.gardener import GardenerAgent

            agent_classes["gardener"] = GardenerAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["gardener"] = self._create_generic_agent_class("gardener")

        # Sword & Shield Agent
        try:
            from production.agents.sword_shield import SwordAndShieldAgent

            agent_classes["sword_shield"] = SwordAndShieldAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["sword_shield"] = self._create_generic_agent_class("sword_shield")

        # Legal AI Agent
        try:
            from production.agents.legal import LegalAIAgent

            agent_classes["legal"] = LegalAIAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["legal"] = self._create_generic_agent_class("legal")

        # Shaman Agent
        try:
            from production.agents.shaman import ShamanAgent

            agent_classes["shaman"] = ShamanAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["shaman"] = self._create_generic_agent_class("shaman")

        # Oracle Agent
        try:
            from production.agents.oracle import OracleAgent

            agent_classes["oracle"] = OracleAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["oracle"] = self._create_generic_agent_class("oracle")

        # Maker Agent
        try:
            from production.agents.maker import MakerAgent

            agent_classes["maker"] = MakerAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["maker"] = self._create_generic_agent_class("maker")

        # Ensemble Agent
        try:
            from production.agents.ensemble import EnsembleAgent

            agent_classes["ensemble"] = EnsembleAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["ensemble"] = self._create_generic_agent_class("ensemble")

        # Curator Agent
        try:
            from production.agents.curator import CuratorAgent

            agent_classes["curator"] = CuratorAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["curator"] = self._create_generic_agent_class("curator")

        # Auditor Agent
        try:
            from production.agents.auditor import AuditorAgent

            agent_classes["auditor"] = AuditorAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["auditor"] = self._create_generic_agent_class("auditor")

        # Medic Agent
        try:
            from production.agents.medic import MedicAgent

            agent_classes["medic"] = MedicAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["medic"] = self._create_generic_agent_class("medic")

        # Sustainer Agent
        try:
            from production.agents.sustainer import SustainerAgent

            agent_classes["sustainer"] = SustainerAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["sustainer"] = self._create_generic_agent_class("sustainer")

        # Navigator Agent
        try:
            from production.agents.navigator import NavigatorAgent

            agent_classes["navigator"] = NavigatorAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["navigator"] = self._create_generic_agent_class("navigator")

        # Tutor Agent
        try:
            from production.agents.tutor import TutorAgent

            agent_classes["tutor"] = TutorAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["tutor"] = self._create_generic_agent_class("tutor")

        # Polyglot Agent
        try:
            from production.agents.polyglot import PolyglotAgent

            agent_classes["polyglot"] = PolyglotAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["polyglot"] = self._create_generic_agent_class("polyglot")

        # Strategist Agent
        try:
            from production.agents.strategist import StrategistAgent

            agent_classes["strategist"] = StrategistAgent
        except ImportError:
            # Use generic agent if specialized not available
            agent_classes["strategist"] = self._create_generic_agent_class("strategist")

        return agent_classes

    def _create_generic_agent_class(self, agent_id: str) -> type:
        """Create a generic agent class for the given agent ID."""
        template = self.templates.get(agent_id, {})
        default_params = template.get("default_params", {})

        class GenericAgent(BaseMetaAgent):
            def __init__(self, spec: AgentSpecialization) -> None:
                super().__init__(spec)
                self.agent_type = agent_id
                self.config = default_params.copy()

            def process(self, task: dict[str, Any]) -> dict[str, Any]:
                return {
                    "status": "completed",
                    "agent": self.agent_type,
                    "result": f"Processed by {template.get('name', agent_id)}",
                }

            def evaluate_kpi(self) -> dict[str, float]:
                if not self.performance_history:
                    return {"performance": 0.7}

                success_rate = sum(1 for p in self.performance_history if p.get("success", False)) / len(
                    self.performance_history
                )

                return {
                    "success_rate": success_rate,
                    "performance": success_rate * 0.8 + 0.2,
                }

        return GenericAgent

    def create_agent(self, agent_type: str, config: dict[str, Any] | None = None) -> BaseMetaAgent:
        """Create an agent of the specified type."""
        if agent_type not in self.templates:
            msg = f"Unknown agent type: {agent_type}. Available: {list(self.templates.keys())}"
            raise ValueError(msg)

        template = self.templates[agent_type]
        agent_class = self.agent_classes[agent_type]

        try:
            role = AgentRole(agent_type)
        except ValueError:
            role = agent_type

        spec = AgentSpecialization(
            role=role,
            primary_capabilities=[],
            secondary_capabilities=[],
            performance_metrics={},
            resource_requirements={},
        )

        params = template.get("default_params", {}).copy()
        if config:
            params.update(config)

        agent = agent_class(spec)
        # Attach configuration and metadata
        agent.config = getattr(agent, "config", {})
        agent.config.update(params)
        agent.name = template.get("name", agent_type)
        agent.role_description = template.get("role", "")

        return agent

    def list_available_agents(self) -> list[dict[str, str]]:
        """List all available agent types."""
        agents = []

        for agent_id, template in self.templates.items():
            agents.append(
                {
                    "id": agent_id,
                    "name": template.get("name", agent_id),
                    "role": template.get("role", ""),
                }
            )

        return agents

    def get_agent_info(self, agent_type: str) -> dict[str, Any]:
        """Get detailed information about an agent type."""
        if agent_type not in self.templates:
            msg = f"Unknown agent type: {agent_type}"
            raise ValueError(msg)

        return self.templates[agent_type]
