"""Agent Configuration Component.

Manages agent configuration, settings, and external client connections.
Handles dependency injection and configuration validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConfigurationLevel(Enum):
    """Configuration precedence levels."""

    DEFAULT = "default"
    ENVIRONMENT = "environment"
    FILE = "file"
    RUNTIME = "runtime"
    OVERRIDE = "override"


@dataclass
class ConfigurationEntry:
    """Single configuration entry with metadata."""

    key: str
    value: Any
    level: ConfigurationLevel
    description: str = ""
    validation_rule: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentConfig:
    """Core agent configuration settings."""

    # Identity and specialization
    agent_id: str
    agent_type: str
    specialized_role: str = "base_template"

    # Performance settings
    max_concurrent_tasks: int = 10
    task_timeout_seconds: float = 300.0
    health_check_interval_seconds: int = 60

    # Memory and journal settings
    max_journal_entries: int = 1000
    max_memory_entries: int = 5000
    memory_retention_hours: int = 168  # 1 week
    memory_retrieval_threshold: float = 0.3

    # Geometric awareness settings
    geometric_awareness_enabled: bool = True
    self_awareness_update_interval_seconds: int = 30
    auto_adaptation_enabled: bool = True

    # Communication settings
    p2p_enabled: bool = True
    broadcast_enabled: bool = True
    group_channels_enabled: bool = True
    emergency_channel_enabled: bool = True

    # ADAS/self-modification settings
    adas_enabled: bool = True
    adaptation_rate: float = 0.1
    stability_threshold: float = 0.8
    max_modifications_per_hour: int = 5

    # External service connections
    rag_client_config: dict[str, Any] = field(default_factory=dict)
    p2p_client_config: dict[str, Any] = field(default_factory=dict)
    agent_forge_config: dict[str, Any] = field(default_factory=dict)


class AgentConfiguration:
    """Manages agent configuration and external dependencies.

    This component centralizes configuration management and dependency
    injection, reducing coupling between configuration concerns and
    business logic.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """Initialize configuration manager.

        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type/category of the agent
        """
        self.agent_id = agent_id
        self.agent_type = agent_type

        # Configuration storage with precedence levels
        self._config_entries: dict[str, ConfigurationEntry] = {}
        self._core_config = AgentConfig(agent_id=agent_id, agent_type=agent_type)

        # External client dependencies (injected)
        self._external_clients: dict[str, Any] = {}

        # Configuration validation rules
        self._validation_rules: dict[str, callable] = {}

        # Load default configuration
        self._load_defaults()

        logger.debug(f"Configuration manager initialized for {agent_type} agent {agent_id}")

    def set_configuration(
        self,
        key: str,
        value: Any,
        level: ConfigurationLevel = ConfigurationLevel.RUNTIME,
        description: str = "",
        validation_rule: str = "",
    ) -> bool:
        """Set a configuration value with precedence level.

        Args:
            key: Configuration key (dot notation supported)
            value: Configuration value
            level: Precedence level for this setting
            description: Human-readable description
            validation_rule: Validation rule identifier

        Returns:
            True if configuration set successfully
        """
        # Validate the value if rule exists
        if validation_rule and validation_rule in self._validation_rules:
            validator = self._validation_rules[validation_rule]
            if not validator(value):
                logger.warning(f"Configuration validation failed for {key}: {value}")
                return False

        # Check if existing entry has higher precedence
        if key in self._config_entries:
            existing = self._config_entries[key]
            if self._get_precedence(existing.level) > self._get_precedence(level):
                logger.debug(
                    f"Configuration {key} not updated - existing level {existing.level.value} has higher precedence"
                )
                return False

        # Create configuration entry
        entry = ConfigurationEntry(
            key=key, value=value, level=level, description=description, validation_rule=validation_rule
        )

        self._config_entries[key] = entry

        # Update core config if it's a known setting
        self._update_core_config(key, value)

        logger.debug(f"Configuration set: {key} = {value} (level: {level.value})")
        return True

    def get_configuration(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key in self._config_entries:
            return self._config_entries[key].value

        # Try to get from core config using dot notation
        try:
            parts = key.split(".")
            value = self._core_config
            for part in parts:
                value = getattr(value, part)
            return value
        except AttributeError:
            return default

    def get_core_config(self) -> AgentConfig:
        """Get the core agent configuration object.

        Returns:
            Complete agent configuration
        """
        return self._core_config

    def inject_client(self, client_name: str, client_instance: Any) -> None:
        """Inject an external client dependency.

        Args:
            client_name: Name identifier for the client
            client_instance: Client instance to inject
        """
        self._external_clients[client_name] = client_instance
        logger.info(f"Injected external client: {client_name}")

    def get_client(self, client_name: str) -> Any | None:
        """Get an injected external client.

        Args:
            client_name: Name identifier for the client

        Returns:
            Client instance or None if not found
        """
        return self._external_clients.get(client_name)

    def has_client(self, client_name: str) -> bool:
        """Check if a client has been injected.

        Args:
            client_name: Name identifier for the client

        Returns:
            True if client is available
        """
        return client_name in self._external_clients

    def validate_configuration(self) -> dict[str, Any]:
        """Validate all configuration entries.

        Returns:
            Dict with validation results and any errors
        """
        validation_results = {"valid": True, "errors": [], "warnings": [], "checked_entries": 0}

        for key, entry in self._config_entries.items():
            validation_results["checked_entries"] += 1

            # Apply validation rule if specified
            if entry.validation_rule and entry.validation_rule in self._validation_rules:
                validator = self._validation_rules[entry.validation_rule]
                try:
                    if not validator(entry.value):
                        validation_results["valid"] = False
                        validation_results["errors"].append(f"Configuration {key} failed validation: {entry.value}")
                except Exception as e:
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"Validation error for {key}: {str(e)}")

        # Validate core configuration requirements
        core_validation = self._validate_core_config()
        if not core_validation["valid"]:
            validation_results["valid"] = False
            validation_results["errors"].extend(core_validation["errors"])

        return validation_results

    def add_validation_rule(self, rule_name: str, validator: callable) -> None:
        """Add a custom validation rule.

        Args:
            rule_name: Identifier for the validation rule
            validator: Function that takes a value and returns bool
        """
        self._validation_rules[rule_name] = validator
        logger.debug(f"Added validation rule: {rule_name}")

    def get_configuration_summary(self) -> dict[str, Any]:
        """Get a summary of all configuration entries.

        Returns:
            Dict with configuration overview and statistics
        """
        level_counts = {}
        for entry in self._config_entries.values():
            level = entry.level.value
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "total_entries": len(self._config_entries),
            "level_distribution": level_counts,
            "external_clients": list(self._external_clients.keys()),
            "validation_rules": list(self._validation_rules.keys()),
            "core_config": {
                "specialized_role": self._core_config.specialized_role,
                "geometric_awareness_enabled": self._core_config.geometric_awareness_enabled,
                "adas_enabled": self._core_config.adas_enabled,
                "p2p_enabled": self._core_config.p2p_enabled,
            },
            "last_updated": max(
                (entry.last_updated for entry in self._config_entries.values()), default=datetime.now()
            ).isoformat(),
        }

    def export_configuration(self, include_metadata: bool = False) -> dict[str, Any]:
        """Export configuration for backup or transfer.

        Args:
            include_metadata: Whether to include metadata and timestamps

        Returns:
            Dict with exportable configuration data
        """
        if include_metadata:
            return {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "configuration_entries": {
                    key: {
                        "value": entry.value,
                        "level": entry.level.value,
                        "description": entry.description,
                        "validation_rule": entry.validation_rule,
                        "last_updated": entry.last_updated.isoformat(),
                        "metadata": entry.metadata,
                    }
                    for key, entry in self._config_entries.items()
                },
                "external_clients": list(self._external_clients.keys()),
                "export_timestamp": datetime.now().isoformat(),
            }
        else:
            return {key: entry.value for key, entry in self._config_entries.items()}

    def import_configuration(
        self, config_data: dict[str, Any], level: ConfigurationLevel = ConfigurationLevel.FILE
    ) -> dict[str, Any]:
        """Import configuration from external source.

        Args:
            config_data: Configuration data to import
            level: Precedence level for imported settings

        Returns:
            Dict with import results and any errors
        """
        import_results = {"imported": 0, "skipped": 0, "errors": []}

        for key, value in config_data.items():
            if key.startswith("_") or key in ["agent_id", "agent_type"]:
                # Skip private/system settings
                import_results["skipped"] += 1
                continue

            try:
                if self.set_configuration(key, value, level):
                    import_results["imported"] += 1
                else:
                    import_results["skipped"] += 1
            except Exception as e:
                import_results["errors"].append(f"Failed to import {key}: {str(e)}")

        logger.info(
            f"Configuration import completed: {import_results['imported']} imported, {import_results['skipped']} skipped"
        )
        return import_results

    def _load_defaults(self) -> None:
        """Load default configuration values."""
        defaults = {
            "max_concurrent_tasks": 10,
            "task_timeout_seconds": 300.0,
            "health_check_interval_seconds": 60,
            "geometric_awareness_enabled": True,
            "self_awareness_update_interval_seconds": 30,
            "memory_retrieval_threshold": 0.3,
            "adaptation_rate": 0.1,
            "stability_threshold": 0.8,
        }

        for key, value in defaults.items():
            self.set_configuration(key, value, ConfigurationLevel.DEFAULT, f"Default {key}")

        # Add basic validation rules
        self._validation_rules.update(
            {
                "positive_number": lambda x: isinstance(x, int | float) and x > 0,
                "percentage": lambda x: isinstance(x, int | float) and 0 <= x <= 1,
                "boolean": lambda x: isinstance(x, bool),
                "non_empty_string": lambda x: isinstance(x, str) and len(x.strip()) > 0,
            }
        )

    def _update_core_config(self, key: str, value: Any) -> None:
        """Update core configuration object with new value."""
        # Map configuration keys to core config attributes
        key_mapping = {
            "specialized_role": "specialized_role",
            "max_concurrent_tasks": "max_concurrent_tasks",
            "task_timeout_seconds": "task_timeout_seconds",
            "health_check_interval_seconds": "health_check_interval_seconds",
            "geometric_awareness_enabled": "geometric_awareness_enabled",
            "self_awareness_update_interval_seconds": "self_awareness_update_interval_seconds",
            "memory_retrieval_threshold": "memory_retrieval_threshold",
            "adaptation_rate": "adaptation_rate",
            "stability_threshold": "stability_threshold",
            "p2p_enabled": "p2p_enabled",
            "adas_enabled": "adas_enabled",
        }

        if key in key_mapping:
            attr_name = key_mapping[key]
            if hasattr(self._core_config, attr_name):
                setattr(self._core_config, attr_name, value)

    def _validate_core_config(self) -> dict[str, Any]:
        """Validate core configuration requirements."""
        errors = []

        # Check required clients for enabled features
        if self._core_config.p2p_enabled and not self.has_client("p2p_client"):
            errors.append("P2P communication enabled but p2p_client not injected")

        if self._core_config.adas_enabled and not self.has_client("agent_forge_client"):
            errors.append("ADAS enabled but agent_forge_client not injected")

        # Check numeric ranges
        if not (0 < self._core_config.adaptation_rate <= 1.0):
            errors.append("adaptation_rate must be between 0 and 1")

        if not (0 < self._core_config.stability_threshold <= 1.0):
            errors.append("stability_threshold must be between 0 and 1")

        return {"valid": len(errors) == 0, "errors": errors}

    def _get_precedence(self, level: ConfigurationLevel) -> int:
        """Get numeric precedence for configuration level."""
        precedence_map = {
            ConfigurationLevel.DEFAULT: 1,
            ConfigurationLevel.ENVIRONMENT: 2,
            ConfigurationLevel.FILE: 3,
            ConfigurationLevel.RUNTIME: 4,
            ConfigurationLevel.OVERRIDE: 5,
        }
        return precedence_map.get(level, 0)
