#!/usr/bin/env python3
"""Agent Factory for creating specialized agents from templates.
Auto-generated from agent specifications.
"""

import json
import logging
from importlib import import_module
from pathlib import Path
from typing import Any

from .base import AgentRole, AgentSpecialization, BaseMetaAgent

logger = logging.getLogger(__name__)


class TemplateNotFoundError(Exception):
    """Exception raised when agent template cannot be found."""

    def __init__(
        self, agent_type: str, searched_paths: list[Path], available_types: list[str]
    ):
        self.agent_type = agent_type
        self.searched_paths = searched_paths
        self.available_types = available_types

        paths_str = "\n  - ".join(str(p) for p in searched_paths)
        available_str = ", ".join(available_types)

        super().__init__(
            f"Template for agent type '{agent_type}' not found.\n"
            f"Searched paths:\n  - {paths_str}\n"
            f"Available agent types: {available_str}\n"
            f"\nRemediation hints:\n"
            f"1. Check if template file exists: {agent_type}_template.json\n"
            f"2. Verify template is in templates/ or templates/agents/ directory\n"
            f"3. Ensure template follows the correct JSON structure\n"
            f"4. Check file permissions and accessibility"
        )


class AgentFactory:
    """Factory for creating agents from templates."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        """Initialize the factory and load templates.

        Parameters
        ----------
        template_dir:
            Optional path to the directory containing agent templates.  When
            omitted, the factory uses the ``templates`` directory that resides
            next to this file.  Using an absolute path or a path relative to the
            current working directory is also supported.
        """
        if template_dir is None:
            self.template_dir = Path(__file__).resolve().parent / "templates"
        else:
            self.template_dir = Path(template_dir)
        self.templates = self._load_templates()
        self.agent_classes = self._initialize_agent_classes()

    def _load_templates(self) -> dict[str, dict[str, Any]]:
        """Load all agent templates with backward-compatible search order.

        Search order:
        1. templates/agents/*.json (legacy location)
        2. templates/*_template.json (current location)
        """
        templates: dict[str, dict[str, Any]] = {}

        # Search paths in priority order
        search_paths = [
            self.template_dir / "agents",  # Legacy location
            self.template_dir,  # Current location
        ]

        loaded_templates = set()

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Load from legacy location (agents/*.json)
            if search_dir.name == "agents":
                for template_file in search_dir.glob("*.json"):
                    if template_file.name == "master_config.json":
                        continue
                    agent_id = template_file.stem
                    if agent_id not in loaded_templates:
                        try:
                            with open(template_file) as f:
                                template = json.load(f)
                            templates[agent_id] = template
                            loaded_templates.add(agent_id)
                            msg = (
                                f"Loaded template {agent_id} from legacy location: "
                                f"{template_file}"
                            )
                            print(msg)
                        except Exception as e:
                            print(f"Error loading template {template_file}: {e}")

            # Load from current location (templates/*_template.json)
            else:
                for template_file in search_dir.glob("*_template.json"):
                    # Extract agent_id from filename (remove '_template.json')
                    agent_id = template_file.stem.replace("_template", "")
                    if agent_id not in loaded_templates:
                        try:
                            with open(template_file) as f:
                                template = json.load(f)
                            templates[agent_id] = template
                            loaded_templates.add(agent_id)
                        except Exception as e:
                            print(f"Error loading template {template_file}: {e}")

        # Validate all required templates are present
        self._validate_required_templates(templates)

        return templates

    def _initialize_agent_classes(self) -> dict[str, type]:
        """Initialize specialized agent classes with graceful fallbacks."""
        agent_specs = {
            "king": ("src.production.agents.king", "KingAgent"),
            "magi": ("src.production.agents.magi", "MagiAgent"),
            "sage": ("src.production.agents.sage", "SageAgent"),
            "gardener": ("src.production.agents.gardener", "GardenerAgent"),
            "sword_shield": (
                "src.production.agents.sword_shield",
                "SwordAndShieldAgent",
            ),
            "legal": ("src.production.agents.legal", "LegalAIAgent"),
            "shaman": ("src.production.agents.shaman", "ShamanAgent"),
            "oracle": ("src.production.agents.oracle", "OracleAgent"),
            "maker": ("src.production.agents.maker", "MakerAgent"),
            "ensemble": ("src.production.agents.ensemble", "EnsembleAgent"),
            "curator": ("src.production.agents.curator", "CuratorAgent"),
            "auditor": ("src.production.agents.auditor", "AuditorAgent"),
            "medic": ("src.production.agents.medic", "MedicAgent"),
            "sustainer": ("src.production.agents.sustainer", "SustainerAgent"),
            "navigator": ("src.production.agents.navigator", "NavigatorAgent"),
            "tutor": ("src.production.agents.tutor", "TutorAgent"),
            "polyglot": ("src.production.agents.polyglot", "PolyglotAgent"),
            "strategist": ("src.production.agents.strategist", "StrategistAgent"),
        }

        agent_classes: dict[str, type] = {}
        for agent_id, (module_path, class_name) in agent_specs.items():
            try:
                module = import_module(module_path)
                agent_classes[agent_id] = getattr(module, class_name)
            except Exception as e:  # pragma: no cover - import may fail
                logger.warning(
                    "Falling back to GenericAgent for %s: %s", agent_id, e
                )
                agent_classes[agent_id] = self._create_generic_agent_class(agent_id)

        return agent_classes

    def _validate_required_templates(
        self, templates: dict[str, dict[str, Any]]
    ) -> None:
        """Validate that all required agent templates are present."""
        try:
            required_types = self.required_agent_types()
        except Exception:
            # If master_config is not available, use expected 18 agent types
            required_types = [
                "king",
                "magi",
                "sage",
                "gardener",
                "sword_shield",
                "legal",
                "shaman",
                "oracle",
                "maker",
                "ensemble",
                "curator",
                "auditor",
                "medic",
                "sustainer",
                "navigator",
                "tutor",
                "polyglot",
                "strategist",
            ]

        missing_templates = []
        for agent_type in required_types:
            if agent_type not in templates:
                missing_templates.append(agent_type)

        if missing_templates:
            search_paths = [
                self.template_dir / "agents",
                self.template_dir,
            ]

            available_types = list(templates.keys())

            raise TemplateNotFoundError(
                f"Missing templates: {', '.join(missing_templates)}",
                search_paths,
                available_types,
            )

        msg = (
            f"[PASS] All {len(required_types)} required agent templates "
            f"loaded successfully"
        )
        print(msg)

    def _create_generic_agent_class(self, agent_id: str) -> type:
        """Create a generic agent class for the given agent ID."""
        template = self.templates.get(agent_id, {})
        default_params = template.get("default_params", {})

        class GenericAgent(BaseMetaAgent):
            def __init__(self, spec: AgentSpecialization) -> None:
                super().__init__(spec)
                self.agent_type = agent_id
                self.config = default_params.copy()
                logger.debug("GenericAgent instantiated for %s", agent_id)

            def process(self, task: dict[str, Any]) -> dict[str, Any]:
                return {
                    "status": "completed",
                    "agent": self.agent_type,
                    "result": f"Processed by {template.get('name', agent_id)}",
                }

            def evaluate_kpi(self) -> dict[str, float]:
                if not self.performance_history:
                    return {"performance": 0.7}

                success_rate = sum(
                    1 for p in self.performance_history if p.get("success", False)
                ) / len(self.performance_history)

                return {
                    "success_rate": success_rate,
                    "performance": success_rate * 0.8 + 0.2,
                }

        return GenericAgent

    def create_agent(
        self, agent_spec: str | dict[str, Any], config: dict[str, Any] | None = None
    ) -> BaseMetaAgent:
        """Create an agent of the specified type.

        Parameters
        ----------
        agent_spec:
            Either the string identifier of the agent to create or a
            configuration dictionary containing at minimum the ``agent_type``
            field. When a dictionary is provided it is merged with the template
            defaults and used as the agent's configuration.
        config:
            Optional configuration dictionary. Only used when ``agent_spec`` is
            a string. If provided it must be a mapping.
        """
        # Normalise input and validate types
        if isinstance(agent_spec, dict):
            agent_type = agent_spec.get("agent_type")
            if not isinstance(agent_type, str):
                msg = "agent_spec dictionary must include an 'agent_type' string"
                raise TypeError(msg)
            if config is None:
                config = {k: v for k, v in agent_spec.items() if k != "agent_type"}
        elif isinstance(agent_spec, str):
            agent_type = agent_spec
        else:  # pragma: no cover - defensive programming
            msg = "agent_spec must be a string or a configuration dict"
            raise TypeError(msg)

        if config is not None and not isinstance(config, dict):
            msg = "config must be a dictionary if provided"
            raise TypeError(msg)

        if agent_type not in self.templates:
            msg = (
                f"Unknown agent type: {agent_type}. Available: "
                f"{list(self.templates.keys())}"
            )
            raise ValueError(msg)

        template = self.templates[agent_type]
        agent_class = self.agent_classes[agent_type]

        if config:
            unknown = set(config) - set(template.get("default_params", {}))
            if unknown:
                msg = (
                    f"Unknown config parameters for {agent_type}: "
                    f"{sorted(unknown)}"
                )
                raise ValueError(msg)

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

    def required_agent_types(self) -> list[str]:
        """Return the list of agent types defined by the master configuration.

        The project aims to maintain an ecosystem of exactly ``18`` specialised
        agents.  This helper inspects the :mod:`master_config.json` file shipped
        with the templates and returns the agent identifiers.  If the
        configuration is missing the method falls back to the templates that are
        already loaded in :attr:`self.templates`.

        Returns:
        -------
        list[str]
            All agent identifiers expected by the system.
        """
        config_path = self.template_dir / "master_config.json"
        if config_path.exists():
            with config_path.open() as f:
                data = json.load(f)
            agent_types = data.get("agent_types", [])
            expected_total = data.get("total_agents")
            if expected_total is not None and len(agent_types) != expected_total:
                msg = (
                    f"master_config.json declares {expected_total} agents but "
                    f"{len(agent_types)} types were listed"
                )
                raise ValueError(msg)
            return agent_types

        # Fallback: derive from loaded templates.
        return sorted(getattr(self, "templates", {}).keys())
