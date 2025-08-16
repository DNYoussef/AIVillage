"""AgentForge - Main facade class for the Agent Forge system.

This module composes evolution and compression engines to provide a unified
interface for agent creation, management and evolution.
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hooks import Artifact, apply_compression, evolution_step, on_agent_created

# Import existing engines - with graceful fallbacks
try:
    from ..agent_factory import AgentFactory
except ImportError:
    AgentFactory = None

try:
    from ..evolution.kpi_evolution_engine import AgentKPI, KPIEvolutionEngine
except ImportError:
    KPIEvolutionEngine = None
    AgentKPI = None

try:
    from ..evolution.dual_evolution_system import DualEvolutionSystem
except ImportError:
    DualEvolutionSystem = None

try:
    from ..evolution.resource_constrained_evolution import ResourceConstrainedEvolution
except ImportError:
    ResourceConstrainedEvolution = None

try:
    from ...agent_forge.compression import (
        BITNETCompressor,
        SEEDLMCompressor,
        VPTQCompressor,
    )
except ImportError:
    BITNETCompressor = None
    SEEDLMCompressor = None
    VPTQCompressor = None

logger = logging.getLogger(__name__)


class AgentSpec:
    """Simple agent specification for creation."""

    def __init__(
        self,
        agent_type: str,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        capabilities: list[str] | None = None,
    ):
        self.agent_type = agent_type
        self.name = name or agent_type
        self.config = config or {}
        self.capabilities = capabilities or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_type": self.agent_type,
            "name": self.name,
            "config": self.config,
            "capabilities": self.capabilities,
        }


@dataclass
class AgentManifest:
    """Agent deployment manifest."""

    agents: list[dict[str, Any]] = field(default_factory=list)
    evolution_config: dict[str, Any] = field(default_factory=dict)
    compression_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"version": "1.0.0", "created_at": time.time()}
    )

    def validate(self) -> None:
        if not isinstance(self.agents, list):
            raise TypeError("agents must be a list")
        for agent in self.agents:
            if not isinstance(agent, dict):
                raise TypeError("agent entries must be dictionaries")
            required = {"id", "type", "name", "config", "created_at"}
            missing = required - agent.keys()
            if missing:
                raise ValueError(f"agent entry missing required fields: {missing}")
        if not isinstance(self.evolution_config, dict):
            raise TypeError("evolution_config must be a dict")
        if not isinstance(self.compression_config, dict):
            raise TypeError("compression_config must be a dict")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": self.agents,
            "evolution_config": self.evolution_config,
            "compression_config": self.compression_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentManifest":
        manifest = cls(
            agents=data.get("agents", []),
            evolution_config=data.get("evolution_config", {}),
            compression_config=data.get("compression_config", {}),
            metadata=data.get("metadata", {}),
        )
        manifest.validate()
        return manifest


class AgentForge:
    """Main facade for the Agent Forge system.

    This class provides a unified interface to create agents, manage evolution,
    and handle deployment manifests. It composes existing engines rather than
    reimplementing their functionality.
    """

    def __init__(
        self,
        template_dir: str | Path | None = None,
        evolution_config: dict[str, Any] | None = None,
        compression_config: dict[str, Any] | None = None,
        enable_evolution: bool = True,
        enable_compression: bool = True,
    ):
        """Initialize the Agent Forge system.

        Args:
            template_dir: Directory containing agent templates
            evolution_config: Configuration for evolution systems
            compression_config: Configuration for compression systems
            enable_evolution: Whether to enable evolution systems
            enable_compression: Whether to enable compression systems
        """
        self.template_dir = Path(template_dir) if template_dir else None
        self.evolution_config = evolution_config or {}
        self.compression_config = compression_config or {}
        self.enable_evolution = enable_evolution and KPIEvolutionEngine is not None
        self.enable_compression = enable_compression

        # Initialize components
        self._initialize_components()

        # State tracking
        self.created_agents: dict[str, Any] = {}
        self.evolution_active = False

        logger.info("AgentForge initialized")

    def _initialize_components(self) -> None:
        """Initialize the available components."""
        # Agent Factory
        if AgentFactory is not None:
            try:
                self.agent_factory = AgentFactory(self.template_dir)
                logger.info("AgentFactory initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize AgentFactory: {e}")
                self.agent_factory = None
        else:
            self.agent_factory = None
            logger.warning("AgentFactory not available")

        # Evolution Engines
        self.kpi_engine = None
        self.dual_evolution = None
        self.resource_evolution = None

        if self.enable_evolution:
            # KPI Evolution Engine
            if KPIEvolutionEngine is not None:
                try:
                    self.kpi_engine = KPIEvolutionEngine(
                        **self.evolution_config.get("kpi", {})
                    )
                    logger.info("KPIEvolutionEngine initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize KPIEvolutionEngine: {e}")

            # Dual Evolution System
            if DualEvolutionSystem is not None:
                try:
                    self.dual_evolution = DualEvolutionSystem(
                        self.evolution_config.get("dual", {})
                    )
                    logger.info("DualEvolutionSystem initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize DualEvolutionSystem: {e}")

            # Resource Constrained Evolution would require more setup
            # so we'll stub it for now unless specifically configured

        # Compression Engines
        self.compression_engines = {}
        if self.enable_compression:
            if BITNETCompressor is not None:
                self.compression_engines["bitnet"] = BITNETCompressor
            if SEEDLMCompressor is not None:
                self.compression_engines["seedlm"] = SEEDLMCompressor
            if VPTQCompressor is not None:
                self.compression_engines["vptq"] = VPTQCompressor

        logger.info(f"Initialized {len(self.compression_engines)} compression engines")

    def create_agent(self, spec: AgentSpec | dict[str, Any] | str) -> Any | None:
        """Create an agent from specification.

        Args:
            spec: Agent specification (AgentSpec object, dict, or string)

        Returns:
            Created agent instance or None if creation failed
        """
        if not self.agent_factory:
            logger.error("AgentFactory not available, cannot create agent")
            return None

        try:
            # Normalize spec input
            if isinstance(spec, str):
                agent_spec = AgentSpec(agent_type=spec)
            elif isinstance(spec, dict):
                agent_spec = AgentSpec(
                    agent_type=spec["agent_type"],
                    name=spec.get("name"),
                    config=spec.get("config"),
                    capabilities=spec.get("capabilities"),
                )
            elif isinstance(spec, AgentSpec):
                agent_spec = spec
            else:
                raise ValueError(f"Unsupported spec type: {type(spec)}")

            # Create agent using factory
            agent = self.agent_factory.create_agent(
                agent_spec.agent_type, config=agent_spec.config
            )

            if agent:
                try:
                    on_agent_created(agent)
                except Exception as hook_error:  # pragma: no cover - defensive
                    logger.debug("on_agent_created hook failed: %s", hook_error)

                # Track created agent
                agent_id = f"{agent_spec.agent_type}_{int(time.time())}"
                self.created_agents[agent_id] = {
                    "agent": agent,
                    "spec": agent_spec.to_dict(),
                    "created_at": time.time(),
                }

                # Register with evolution system if available
                if self.kpi_engine:
                    try:
                        self.kpi_engine.register_agent(
                            agent_id,
                            agent_spec.agent_type,
                            agent_spec.config,
                            None,  # template_path
                        )
                        logger.info(
                            f"Registered agent {agent_id} with KPI evolution engine"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to register agent with KPI engine: {e}")

                logger.info(f"Created agent: {agent_id} ({agent_spec.agent_type})")
                return agent
            logger.error(f"Failed to create agent of type: {agent_spec.agent_type}")
            return None

        except Exception as e:
            logger.exception(f"Error creating agent: {e}")
            return None

    def save_manifest(
        self, path: str | Path, manifest: AgentManifest | None = None
    ) -> bool:
        """Save deployment manifest to file.

        Args:
            path: Path to save manifest to
            manifest: Optional manifest to save, creates default if None

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if manifest is None:
                manifest = self._create_default_manifest()

            manifest.validate()

            manifest_path = Path(path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)

            with open(manifest_path, "w") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            logger.info(f"Saved manifest to: {manifest_path}")
            return True

        except Exception as e:
            logger.exception(f"Error saving manifest: {e}")
            return False

    def load_manifest(self, path: str | Path) -> AgentManifest | None:
        """Load deployment manifest from file.

        Args:
            path: Path to load manifest from

        Returns:
            Loaded manifest or None if loading failed
        """
        try:
            manifest_path = Path(path)

            if not manifest_path.exists():
                logger.error(f"Manifest file not found: {manifest_path}")
                return None

            with open(manifest_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise TypeError("Manifest file must contain a JSON object")

            manifest = AgentManifest.from_dict(data)
            logger.info(f"Loaded manifest from: {manifest_path}")
            return manifest

        except Exception as e:
            logger.exception(f"Error loading manifest: {e}")
            return None

    def create_manifest(self) -> AgentManifest:
        """Create a manifest representing current agents."""
        return self._create_default_manifest()

    def run_kpi_cycle(
        self,
        manifest: AgentManifest | None = None,
        kpi_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run KPI evaluation and evolution cycle.

        Args:
            manifest: Optional manifest containing agents to evaluate
            kpi_config: Optional KPI configuration

        Returns:
            Dictionary with cycle results
        """
        if not self.kpi_engine:
            logger.warning("KPI Evolution Engine not available")
            return {
                "success": False,
                "error": "KPI Evolution Engine not available",
                "agents_evaluated": 0,
                "actions_taken": [],
            }

        try:
            # If manifest provided, process its agents
            if manifest and manifest.agents:
                for agent_data in manifest.agents:
                    agent_id = agent_data.get("id")
                    if agent_id and agent_id in self.created_agents:
                        # Update KPI for existing agent
                        agent_info = self.created_agents[agent_id]
                        agent = agent_info["agent"]

                        # Get KPI data from agent if it supports it
                        if hasattr(agent, "evaluate_kpi"):
                            kpi_data = agent.evaluate_kpi()
                            try:
                                patch = evolution_step(agent, kpi_data)
                                if patch:
                                    logger.debug(
                                        "evolution_step returned patch for %s: %s",
                                        agent_id,
                                        patch,
                                    )
                            except Exception as hook_error:  # pragma: no cover
                                logger.debug(
                                    "evolution_step hook failed: %s", hook_error
                                )

                            if AgentKPI is not None:
                                # Create KPI object
                                kpi = AgentKPI(
                                    agent_id=agent_id,
                                    accuracy=kpi_data.get("accuracy", 0.7),
                                    response_time_ms=kpi_data.get(
                                        "response_time_ms", 1000.0
                                    ),
                                    success_rate=kpi_data.get("success_rate", 0.8),
                                    throughput_tps=kpi_data.get("throughput_tps", 1.0),
                                )

                                self.kpi_engine.update_agent_kpi(agent_id, kpi)

            # Run population evaluation
            results = self.kpi_engine.evaluate_population()

            logger.info(f"KPI cycle completed: {results}")
            return {"success": True, "results": results, "timestamp": time.time()}

        except Exception as e:
            logger.exception(f"Error in KPI cycle: {e}")
            return {
                "success": False,
                "error": str(e),
                "agents_evaluated": 0,
                "actions_taken": [],
            }

    def compress_agent(self, agent_id: str, engine_name: str) -> Artifact | None:
        """Compress a created agent using the specified engine.

        The actual compression logic is delegated to the ``apply_compression``
        hook, allowing deployments to provide custom implementations.
        """
        if engine_name not in self.compression_engines:
            raise ValueError(f"Unknown compression engine: {engine_name}")
        agent_info = self.created_agents.get(agent_id)
        if not agent_info:
            raise ValueError(f"Unknown agent id: {agent_id}")
        agent = agent_info["agent"]
        try:
            return apply_compression(agent, engine_name)
        except Exception as hook_error:  # pragma: no cover - defensive
            logger.debug("apply_compression hook failed: %s", hook_error)
            return None

    def _create_default_manifest(self) -> AgentManifest:
        """Create a default manifest from currently created agents."""
        agents = []

        for agent_id, agent_info in self.created_agents.items():
            agents.append(
                {
                    "id": agent_id,
                    "type": agent_info["spec"]["agent_type"],
                    "name": agent_info["spec"]["name"],
                    "config": agent_info["spec"]["config"],
                    "created_at": agent_info["created_at"],
                }
            )

        return AgentManifest(
            agents=agents,
            evolution_config=self.evolution_config,
            compression_config=self.compression_config,
        )

    def get_available_agent_types(self) -> list[str]:
        """Get list of available agent types.

        Returns:
            List of agent type identifiers
        """
        if not self.agent_factory:
            return []

        try:
            available = self.agent_factory.list_available_agents()
            return [agent["id"] for agent in available]
        except Exception as e:
            logger.exception(f"Error getting available agent types: {e}")
            return []

    def get_compression_engines(self) -> list[str]:
        """Get list of available compression engines.

        Returns:
            List of compression engine names
        """
        return list(self.compression_engines.keys())

    def start_evolution(self) -> bool:
        """Start the evolution system.

        Returns:
            True if started successfully, False otherwise
        """
        if not self.enable_evolution:
            logger.warning("Evolution not enabled")
            return False

        try:
            if self.kpi_engine:
                self.kpi_engine.start_evolution_scheduler()
                self.evolution_active = True
                logger.info("Evolution system started")
                return True
            logger.warning("No evolution engines available")
            return False

        except Exception as e:
            logger.exception(f"Error starting evolution: {e}")
            return False

    def stop_evolution(self) -> bool:
        """Stop the evolution system.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.evolution_active:
            return True

        try:
            if self.kpi_engine:
                self.kpi_engine.stop_evolution_scheduler()

            self.evolution_active = False
            logger.info("Evolution system stopped")
            return True

        except Exception as e:
            logger.exception(f"Error stopping evolution: {e}")
            return False

    def get_system_status(self) -> dict[str, Any]:
        """Get current system status.

        Returns:
            Dictionary with system status information
        """
        status = {
            "agent_factory_available": self.agent_factory is not None,
            "evolution_enabled": self.enable_evolution,
            "evolution_active": self.evolution_active,
            "compression_enabled": self.enable_compression,
            "created_agents": len(self.created_agents),
            "available_agent_types": self.get_available_agent_types(),
            "compression_engines": self.get_compression_engines(),
            "components": {
                "kpi_engine": self.kpi_engine is not None,
                "dual_evolution": self.dual_evolution is not None,
                "resource_evolution": self.resource_evolution is not None,
            },
        }

        # Add evolution status if available
        if self.kpi_engine:
            try:
                evolution_status = self.kpi_engine.get_population_status()
                status["evolution_status"] = evolution_status
            except Exception as e:
                logger.debug(f"Could not get evolution status: {e}")

        return status

    def __repr__(self) -> str:
        """String representation of AgentForge instance."""
        return (
            f"AgentForge("
            f"agents={len(self.created_agents)}, "
            f"evolution={'ON' if self.evolution_active else 'OFF'}, "
            f"compression={len(self.compression_engines)} engines"
            f")"
        )


def main() -> None:
    """CLI entry point for basic forge operations."""
    parser = argparse.ArgumentParser(description="Agent Forge CLI")
    parser.add_argument("--list", action="store_true", help="List available agents")
    parser.add_argument("--create", type=str, help="Create agent of given type")
    parser.add_argument("--manifest", type=str, help="Path to save manifest")
    args = parser.parse_args()

    forge = AgentForge()

    if args.list:
        for agent_type in forge.get_available_agent_types():
            print(agent_type)

    if args.create:
        forge.create_agent(args.create)
        if args.manifest:
            manifest = forge.create_manifest()
            forge.save_manifest(args.manifest, manifest)


if __name__ == "__main__":  # pragma: no cover - CLI utility
    main()
