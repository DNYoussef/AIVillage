"""Central configuration manager for AIVillage CODEX integration.

This module provides a unified configuration system that loads settings from
multiple sources in priority order and validates all configurations before use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

from .environment_validator import (
    EnvironmentValidator,
    ValidationLevel,
    ValidationReport,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationSource:
    """Configuration source with priority level."""

    name: str
    data: dict[str, Any]
    priority: int  # Higher number = higher priority


@dataclass
class ConfigurationProfile:
    """Configuration profile for different environments."""

    name: str
    description: str
    required_variables: list[str] = field(default_factory=list)
    optional_variables: list[str] = field(default_factory=list)
    security_requirements: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Central configuration manager with priority-based loading."""

    # Default configuration profiles
    PROFILES = {
        "development": ConfigurationProfile(
            name="development",
            description="Development environment with relaxed security",
            defaults={
                "AIVILLAGE_ENV": "development",
                "AIVILLAGE_LOG_LEVEL": "DEBUG",
                "AIVILLAGE_DEBUG_MODE": "true",
                "DEV_AUTO_RELOAD": "true",
                "DEV_MOCK_P2P": "false",
                "DEV_SEED_DATA": "true",
                "TEST_DISABLE_AUTH": "true",
                "API_AUTH_ENABLED": "false",
                "MESH_TLS_ENABLED": "false",
                "MESH_ENCRYPTION_REQUIRED": "false",
                "DIGITAL_TWIN_BACKUP_ENCRYPTION": "false",
                "BACKUP_ENABLED": "false",
                "FEATURE_ADVANCED_ANALYTICS": "true",
                "RATE_LIMIT_ENABLED": "false",
                # Paths
                "AIVILLAGE_DB_PATH": "./data/development/evolution_metrics.db",
                "DIGITAL_TWIN_DB_PATH": "./data/development/digital_twin.db",
                "RAG_FAISS_INDEX_PATH": "./data/development/faiss_index",
                "DIGITAL_TWIN_VAULT_PATH": "./data/development/vault",
                "AIVILLAGE_LOG_DIR": "./logs/development",
                # Ports
                "LIBP2P_PORT": "4001",
                "DIGITAL_TWIN_API_PORT": "8080",
                "EVOLUTION_METRICS_API_PORT": "8081",
                "RAG_PIPELINE_API_PORT": "8082",
                "P2P_STATUS_API_PORT": "8083",
            },
        ),
        "testing": ConfigurationProfile(
            name="testing",
            description="Testing environment for CI/CD",
            defaults={
                "AIVILLAGE_ENV": "testing",
                "AIVILLAGE_LOG_LEVEL": "INFO",
                "TEST_FAST_MODE": "true",
                "TEST_MOCK_EXTERNAL_SERVICES": "true",
                "DEV_MOCK_P2P": "true",
                "BACKUP_ENABLED": "false",
                # In-memory databases for testing
                "AIVILLAGE_DB_PATH": ":memory:",
                "DIGITAL_TWIN_DB_PATH": ":memory:",
                "TEST_DB_PATH": ":memory:",
                # Temporary paths
                "RAG_FAISS_INDEX_PATH": "/tmp/test_faiss_index",
                "DIGITAL_TWIN_VAULT_PATH": "/tmp/test_vault",
                "AIVILLAGE_LOG_DIR": "/tmp/test_logs",
                "MESH_FILE_TRANSPORT_DIR": "/tmp/test_mesh",
                # Test ports (higher range to avoid conflicts)
                "LIBP2P_PORT": "14001",
                "DIGITAL_TWIN_API_PORT": "18080",
                "EVOLUTION_METRICS_API_PORT": "18081",
                "RAG_PIPELINE_API_PORT": "18082",
            },
        ),
        "staging": ConfigurationProfile(
            name="staging",
            description="Staging environment mimicking production",
            defaults={
                "AIVILLAGE_ENV": "staging",
                "AIVILLAGE_LOG_LEVEL": "INFO",
                "API_AUTH_ENABLED": "true",
                "MESH_TLS_ENABLED": "true",
                "MESH_ENCRYPTION_REQUIRED": "true",
                "DIGITAL_TWIN_BACKUP_ENCRYPTION": "true",
                "BACKUP_ENABLED": "true",
                "RATE_LIMIT_ENABLED": "true",
                # Production-like paths
                "AIVILLAGE_DB_PATH": "./data/staging/evolution_metrics.db",
                "DIGITAL_TWIN_DB_PATH": "./data/staging/digital_twin.db",
                "RAG_FAISS_INDEX_PATH": "./data/staging/faiss_index",
                "DIGITAL_TWIN_VAULT_PATH": "./secure/staging/vault",
                "BACKUP_STORAGE_PATH": "./backups/staging",
            },
        ),
        "production": ConfigurationProfile(
            name="production",
            description="Production environment with full security",
            security_requirements={
                "encryption_required": True,
                "tls_required": True,
                "auth_required": True,
                "compliance_required": True,
                "backup_required": True,
            },
            defaults={
                "AIVILLAGE_ENV": "production",
                "AIVILLAGE_LOG_LEVEL": "WARNING",
                "AIVILLAGE_DEBUG_MODE": "false",
                "DEV_AUTO_RELOAD": "false",
                "DEV_MOCK_P2P": "false",
                "TEST_DISABLE_AUTH": "false",
                # Security enabled
                "API_AUTH_ENABLED": "true",
                "MESH_TLS_ENABLED": "true",
                "MESH_ENCRYPTION_REQUIRED": "true",
                "DIGITAL_TWIN_BACKUP_ENCRYPTION": "true",
                "DIGITAL_TWIN_COPPA_COMPLIANT": "true",
                "DIGITAL_TWIN_FERPA_COMPLIANT": "true",
                "DIGITAL_TWIN_GDPR_COMPLIANT": "true",
                "RATE_LIMIT_ENABLED": "true",
                "BACKUP_ENABLED": "true",
                "SECURITY_HSTS_ENABLED": "true",
                "SECURITY_CSP_ENABLED": "true",
                "TLS_VERIFY_PEER": "true",
                # Production paths
                "AIVILLAGE_DB_PATH": "/var/lib/aivillage/evolution_metrics.db",
                "DIGITAL_TWIN_DB_PATH": "/var/lib/aivillage/digital_twin.db",
                "RAG_FAISS_INDEX_PATH": "/var/lib/aivillage/faiss_index",
                "DIGITAL_TWIN_VAULT_PATH": "/var/lib/aivillage/secure/vault",
                "BACKUP_STORAGE_PATH": "/var/backups/aivillage",
                "AIVILLAGE_LOG_DIR": "/var/log/aivillage",
            },
        ),
    }

    def __init__(self, profile: str = None, config_dir: str = None):
        """Initialize configuration manager."""
        self.profile_name = profile or os.environ.get("AIVILLAGE_ENV", "development")
        self.config_dir = Path(config_dir) if config_dir else Path("./config")
        self.sources: list[ConfigurationSource] = []
        self.final_config: dict[str, str] = {}
        self.validation_report: ValidationReport | None = None

        # Ensure profile exists
        if self.profile_name not in self.PROFILES:
            available = ", ".join(self.PROFILES.keys())
            raise ValueError(
                f"Unknown profile '{self.profile_name}'. Available: {available}"
            )

        self.profile = self.PROFILES[self.profile_name]

    def load_configuration(
        self, cli_args: dict[str, Any] | None = None, validate: bool = True
    ) -> dict[str, str]:
        """Load configuration from all sources in priority order.

        Priority order (highest to lowest):
        1. Command line arguments
        2. Environment variables
        3. Profile-specific config file
        4. Main config file
        5. .env file
        6. Profile defaults
        """
        self.sources = []

        # 1. Profile defaults (lowest priority)
        self._load_profile_defaults()

        # 2. Load .env file if available
        self._load_env_file()

        # 3. Load main config file
        self._load_config_file("aivillage_config")

        # 4. Load profile-specific config file
        self._load_config_file(f"aivillage_config_{self.profile_name}")

        # 5. Load environment variables
        self._load_environment_variables()

        # 6. Load command line arguments (highest priority)
        if cli_args:
            self._load_cli_args(cli_args)

        # Merge all sources by priority
        self._merge_sources()

        # Validate configuration if requested
        if validate:
            self._validate_configuration()

        return self.final_config

    def _load_profile_defaults(self) -> None:
        """Load default values for the current profile."""
        self.sources.append(
            ConfigurationSource(
                name=f"profile_defaults_{self.profile_name}",
                data=self.profile.defaults,
                priority=1,
            )
        )
        logger.debug(
            f"Loaded {len(self.profile.defaults)} default values for profile '{self.profile_name}'"
        )

    def _load_env_file(self) -> None:
        """Load .env file if available."""
        env_files = [".env", f".env.{self.profile_name}", ".env.local"]

        for env_file in env_files:
            env_path = Path(env_file)
            if env_path.exists():
                if DOTENV_AVAILABLE:
                    # Load into temporary dict to avoid polluting os.environ
                    env_vars = {}
                    with open(env_path) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                key, value = line.split("=", 1)
                                env_vars[key.strip()] = value.strip().strip("\"'")

                    self.sources.append(
                        ConfigurationSource(
                            name=f"env_file_{env_file}", data=env_vars, priority=2
                        )
                    )
                    logger.debug(f"Loaded {len(env_vars)} variables from {env_file}")
                else:
                    logger.warning(f"Found {env_file} but python-dotenv not available")

    def _load_config_file(self, filename: str) -> None:
        """Load configuration from YAML or JSON file."""
        for ext in ["yaml", "yml", "json"]:
            config_path = self.config_dir / f"{filename}.{ext}"
            if config_path.exists():
                try:
                    with open(config_path) as f:
                        if ext in ["yaml", "yml"]:
                            if YAML_AVAILABLE:
                                data = yaml.safe_load(f)
                            else:
                                logger.warning(
                                    f"Found {config_path} but PyYAML not available"
                                )
                                continue
                        else:  # json
                            data = json.load(f)

                    # Flatten nested config for environment variable compatibility
                    flat_data = self._flatten_config(data)

                    priority = 3 if filename == "aivillage_config" else 4
                    self.sources.append(
                        ConfigurationSource(
                            name=f"config_file_{config_path.name}",
                            data=flat_data,
                            priority=priority,
                        )
                    )
                    logger.debug(
                        f"Loaded {len(flat_data)} variables from {config_path}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Error loading config file {config_path}: {e}")

    def _flatten_config(self, data: dict[str, Any], prefix: str = "") -> dict[str, str]:
        """Flatten nested configuration to environment variable format."""
        result = {}

        for key, value in data.items():
            if prefix:
                full_key = f"{prefix}_{key}".upper()
            else:
                full_key = key.upper()

            if isinstance(value, dict):
                result.update(self._flatten_config(value, full_key))
            else:
                result[full_key] = str(value)

        return result

    def _load_environment_variables(self) -> None:
        """Load configuration from environment variables."""
        # Filter environment variables to only include relevant ones
        relevant_vars = {}
        aivillage_prefixes = [
            "AIVILLAGE_",
            "RAG_",
            "LIBP2P_",
            "MESH_",
            "DIGITAL_TWIN_",
            "API_",
            "MDNS_",
            "REDIS_",
            "WANDB_",
            "OPENAI_",
            "HUGGINGFACE_",
            "TLS_",
            "SECURITY_",
            "RATE_LIMIT_",
            "BACKUP_",
            "RECOVERY_",
            "MAX_",
            "TEST_",
            "DEV_",
            "MOBILE_",
            "FEATURE_",
            "LEGACY_",
            "ANDROID_",
            "IOS_",
            "WIFI_DIRECT_",
            "HEALTH_CHECK_",
        ]

        for key, value in os.environ.items():
            if any(key.startswith(prefix) for prefix in aivillage_prefixes):
                relevant_vars[key] = value

        self.sources.append(
            ConfigurationSource(
                name="environment_variables", data=relevant_vars, priority=5
            )
        )
        logger.debug(f"Loaded {len(relevant_vars)} environment variables")

    def _load_cli_args(self, cli_args: dict[str, Any]) -> None:
        """Load configuration from command line arguments."""
        # Convert CLI args to environment variable format
        env_format_args = {}
        for key, value in cli_args.items():
            if value is not None:
                env_key = key.upper().replace("-", "_")
                if not env_key.startswith("AIVILLAGE_"):
                    env_key = f"AIVILLAGE_{env_key}"
                env_format_args[env_key] = str(value)

        self.sources.append(
            ConfigurationSource(name="cli_arguments", data=env_format_args, priority=6)
        )
        logger.debug(f"Loaded {len(env_format_args)} command line arguments")

    def _merge_sources(self) -> None:
        """Merge all configuration sources by priority."""
        # Sort sources by priority (lowest first, so highest priority overwrites)
        sorted_sources = sorted(self.sources, key=lambda s: s.priority)

        self.final_config = {}
        for source in sorted_sources:
            for key, value in source.data.items():
                if isinstance(value, str):
                    self.final_config[key] = value
                else:
                    self.final_config[key] = str(value)
            logger.debug(f"Applied {len(source.data)} variables from {source.name}")

    def _validate_configuration(self) -> None:
        """Validate the final configuration."""
        validator = EnvironmentValidator(self.profile_name)
        self.validation_report = validator.validate_all(self.final_config)

        if not self.validation_report.is_valid:
            error_count = len(
                [
                    i
                    for i in self.validation_report.issues
                    if i.level == ValidationLevel.ERROR
                ]
            )
            logger.error(f"Configuration validation failed with {error_count} errors")

            # Print validation report
            print(validator.generate_report())

            # In production, fail fast on invalid configuration
            if self.profile_name == "production":
                raise RuntimeError("Invalid configuration in production environment")

    def get(self, key: str, default: Any = None) -> str:
        """Get configuration value with optional default."""
        return self.final_config.get(key, default)

    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(key, str(default))
        try:
            return int(value)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {value}, using default {default}"
            )
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(key, str(default))
        try:
            return float(value)
        except ValueError:
            logger.warning(
                f"Invalid float value for {key}: {value}, using default {default}"
            )
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(key, str(default)).lower()
        return value in ["true", "1", "yes", "on", "enabled"]

    def get_list(
        self, key: str, separator: str = ",", default: list[str] = None
    ) -> list[str]:
        """Get configuration value as list."""
        if default is None:
            default = []

        value = self.get(key, "")
        if not value:
            return default

        return [item.strip() for item in value.split(separator) if item.strip()]

    def get_path(self, key: str, default: str = None) -> Path:
        """Get configuration value as Path object."""
        value = self.get(key, default)
        return Path(value) if value else None

    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.profile_name == "development"

    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.profile_name == "production"

    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.profile_name == "testing"

    def get_database_url(self, component: str) -> str:
        """Get database URL for a component."""
        if component == "evolution":
            db_path = self.get("AIVILLAGE_DB_PATH")
            if db_path == ":memory:":
                return "sqlite:///:memory:"
            return f"sqlite:///{db_path}"
        if component == "digital_twin":
            db_path = self.get("DIGITAL_TWIN_DB_PATH")
            if db_path == ":memory:":
                return "sqlite:///:memory:"
            return f"sqlite:///{db_path}"
        raise ValueError(f"Unknown database component: {component}")

    def get_api_base_url(self, component: str) -> str:
        """Get API base URL for a component."""
        port_map = {
            "digital_twin": "DIGITAL_TWIN_API_PORT",
            "evolution": "EVOLUTION_METRICS_API_PORT",
            "rag": "RAG_PIPELINE_API_PORT",
            "p2p": "P2P_STATUS_API_PORT",
        }

        if component not in port_map:
            available = ", ".join(port_map.keys())
            raise ValueError(
                f"Unknown API component: {component}. Available: {available}"
            )

        host = self.get("LIBP2P_HOST", "localhost")
        port = self.get_int(port_map[component], 8080)

        # Use HTTPS in production
        protocol = (
            "https" if self.is_production() and self.get_bool("TLS_ENABLED") else "http"
        )

        return f"{protocol}://{host}:{port}"

    def export_config(self, include_secrets: bool = False) -> dict[str, str]:
        """Export configuration (optionally excluding secrets)."""
        if include_secrets:
            return self.final_config.copy()

        # Filter out secret/sensitive variables
        sensitive_keywords = ["KEY", "SECRET", "TOKEN", "PASSWORD", "PRIVATE"]
        filtered_config = {}

        for key, value in self.final_config.items():
            if not any(keyword in key for keyword in sensitive_keywords):
                filtered_config[key] = value
            else:
                filtered_config[key] = "[REDACTED]"

        return filtered_config

    def generate_report(self) -> str:
        """Generate configuration report."""
        lines = [
            "=" * 80,
            "AIVILLAGE CONFIGURATION REPORT",
            "=" * 80,
            f"Profile: {self.profile_name} ({self.profile.description})",
            f"Configuration Directory: {self.config_dir}",
            f"Total Variables: {len(self.final_config)}",
            "",
        ]

        # Sources summary
        lines.append("CONFIGURATION SOURCES:")
        lines.append("-" * 40)
        for source in sorted(self.sources, key=lambda s: s.priority, reverse=True):
            lines.append(
                f"  {source.priority}. {source.name} ({len(source.data)} variables)"
            )
        lines.append("")

        # Validation summary
        if self.validation_report:
            lines.extend(
                [
                    "VALIDATION SUMMARY:",
                    "-" * 40,
                    f"Status: {'✅ VALID' if self.validation_report.is_valid else '❌ INVALID'}",
                    f"Errors: {self.validation_report.errors}",
                    f"Warnings: {self.validation_report.warnings}",
                    "",
                ]
            )

        # Component configuration
        components = [
            "evolution_metrics",
            "rag_pipeline",
            "p2p_networking",
            "digital_twin",
            "api_server",
        ]
        lines.append("COMPONENT CONFIGURATION:")
        lines.append("-" * 40)

        for component in components:
            component_vars = [
                k
                for k in self.final_config.keys()
                if self._is_component_variable(k, component)
            ]
            lines.append(f"  {component}: {len(component_vars)} variables configured")

        return "\n".join(lines)

    def _is_component_variable(self, var_name: str, component: str) -> bool:
        """Check if variable belongs to a component."""
        component_prefixes = {
            "evolution_metrics": ["AIVILLAGE_"],
            "rag_pipeline": ["RAG_"],
            "p2p_networking": ["LIBP2P_", "MESH_", "MDNS_"],
            "digital_twin": ["DIGITAL_TWIN_"],
            "api_server": [
                "API_",
                "DIGITAL_TWIN_API_",
                "EVOLUTION_METRICS_API_",
                "RAG_PIPELINE_API_",
                "P2P_STATUS_API_",
            ],
        }

        prefixes = component_prefixes.get(component, [])
        return any(var_name.startswith(prefix) for prefix in prefixes)


# Global configuration instance
_config_manager: ConfigurationManager | None = None


def get_config(profile: str = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager

    if _config_manager is None or (profile and _config_manager.profile_name != profile):
        _config_manager = ConfigurationManager(profile)
        _config_manager.load_configuration()

    return _config_manager


def setup_configuration(
    profile: str = None,
    cli_args: dict[str, Any] = None,
    config_dir: str = None,
    validate: bool = True,
) -> ConfigurationManager:
    """Setup global configuration manager."""
    global _config_manager

    _config_manager = ConfigurationManager(profile, config_dir)
    _config_manager.load_configuration(cli_args, validate)

    return _config_manager


if __name__ == "__main__":
    # CLI interface for configuration management
    parser = argparse.ArgumentParser(description="AIVillage Configuration Manager")
    parser.add_argument(
        "--profile",
        choices=["development", "testing", "staging", "production"],
        help="Configuration profile",
    )
    parser.add_argument("--config-dir", help="Configuration directory path")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--export", help="Export configuration to file")
    parser.add_argument(
        "--include-secrets",
        action="store_true",
        help="Include secrets in export (use with caution)",
    )

    args = parser.parse_args()

    try:
        # Setup configuration
        config = setup_configuration(
            profile=args.profile, config_dir=args.config_dir, validate=args.validate
        )

        # Print report
        print(config.generate_report())

        # Export if requested
        if args.export:
            exported = config.export_config(include_secrets=args.include_secrets)
            with open(args.export, "w") as f:
                json.dump(exported, f, indent=2, sort_keys=True)
            print(f"\nConfiguration exported to {args.export}")

    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
