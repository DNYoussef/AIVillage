#!/usr/bin/env python3
"""Unified configuration management for AIVillage scripts.

This module provides a centralized configuration system that supports:
- Multiple configuration sources (files, environment variables, defaults)
- Environment-aware configuration (dev, staging, prod)
- Configuration validation and schema enforcement
- Dynamic configuration updates
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""


class ConfigManager:
    """Centralized configuration management for AIVillage scripts."""

    def __init__(self, config_dir: Path | None = None, environment: str | None = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing configuration files
            environment: Target environment (dev, staging, prod)
        """
        self.config_dir = config_dir or Path(__file__).parent.parent.parent / "config"
        self.environment = environment or os.getenv("AIVILLAGE_ENV", "dev")
        self.config_cache: dict[str, Any] = {}

        # Default configuration schema
        self.default_config = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
            },
            "monitoring": {
                "enabled": True,
                "interval": 5.0,
                "metrics_file": "metrics.json",
                "thresholds": {
                    "cpu_percent_high": 80.0,
                    "memory_percent_high": 85.0,
                    "disk_usage_high": 90.0,
                },
            },
            "validation": {
                "strict_mode": False,
                "timeout": 300,
                "retry_attempts": 3,
            },
            "benchmarking": {
                "enabled": True,
                "iterations": 5,
                "timeout": 1800,
                "output_dir": "benchmark_results",
            },
            "compression": {
                "method": "SeedLM",
                "compression_level": 0.5,
                "quality_threshold": 0.95,
            },
            "evolution": {
                "generations": 50,
                "population_size": 20,
                "mutation_rate": 0.1,
            },
        }

        logger.info(f"ConfigManager initialized for environment: {self.environment}")

    def load_config(self, config_name: str, required: bool = False) -> dict[str, Any]:
        """Load configuration from file with environment-specific overrides.

        Args:
            config_name: Name of the configuration (without extension)
            required: Whether the configuration file must exist

        Returns:
            Dictionary containing the loaded configuration

        Raises:
            ConfigurationError: If required config is missing or invalid
        """
        # Check cache first
        cache_key = f"{config_name}_{self.environment}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        # Start with default configuration
        config = self.default_config.copy()

        # Load base configuration file
        base_config_file = self.config_dir / f"{config_name}.yaml"
        if base_config_file.exists():
            try:
                with open(base_config_file, encoding="utf-8") as f:
                    base_config = yaml.safe_load(f)
                    if base_config:
                        self._deep_merge(config, base_config)
                logger.debug(f"Loaded base config from {base_config_file}")
            except Exception as e:
                logger.error(f"Failed to load base config {base_config_file}: {e}")
                if required:
                    raise ConfigurationError(f"Failed to load required config: {e}")

        # Load environment-specific overrides
        env_config_file = self.config_dir / f"{config_name}.{self.environment}.yaml"
        if env_config_file.exists():
            try:
                with open(env_config_file, encoding="utf-8") as f:
                    env_config = yaml.safe_load(f)
                    if env_config:
                        self._deep_merge(config, env_config)
                logger.debug(f"Loaded environment config from {env_config_file}")
            except Exception as e:
                logger.warning(
                    f"Failed to load environment config {env_config_file}: {e}"
                )

        # Apply environment variable overrides
        config = self._apply_env_overrides(config, config_name)

        # Validate configuration
        try:
            self._validate_config(config, config_name)
        except Exception as e:
            logger.error(f"Configuration validation failed for {config_name}: {e}")
            if required:
                raise ConfigurationError(f"Configuration validation failed: {e}")

        # Cache the configuration
        self.config_cache[cache_key] = config

        logger.info(f"Successfully loaded configuration: {config_name}")
        return config

    def get_config_value(
        self, config_name: str, key_path: str, default: Any = None
    ) -> Any:
        """Get a specific configuration value using dot notation.

        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value (e.g., 'monitoring.interval')
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        config = self.load_config(config_name)

        keys = key_path.split(".")
        value = config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(
                f"Configuration key not found: {key_path}, using default: {default}"
            )
            return default

    def set_config_value(self, config_name: str, key_path: str, value: Any) -> None:
        """Set a configuration value dynamically.

        Args:
            config_name: Name of the configuration
            key_path: Dot-separated path to the value
            value: Value to set
        """
        cache_key = f"{config_name}_{self.environment}"
        if cache_key not in self.config_cache:
            self.load_config(config_name)

        config = self.config_cache[cache_key]
        keys = key_path.split(".")

        # Navigate to the parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value
        logger.debug(f"Set configuration value: {key_path} = {value}")

    def reload_config(self, config_name: str) -> dict[str, Any]:
        """Reload configuration from disk, clearing cache.

        Args:
            config_name: Name of the configuration to reload

        Returns:
            The reloaded configuration
        """
        cache_key = f"{config_name}_{self.environment}"
        if cache_key in self.config_cache:
            del self.config_cache[cache_key]

        return self.load_config(config_name)

    def save_config(self, config_name: str, config: dict[str, Any]) -> None:
        """Save configuration to file.

        Args:
            config_name: Name of the configuration
            config: Configuration dictionary to save
        """
        config_file = self.config_dir / f"{config_name}.yaml"

        try:
            # Ensure config directory exists
            self.config_dir.mkdir(parents=True, exist_ok=True)

            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=True)

            logger.info(f"Saved configuration to {config_file}")

            # Update cache
            cache_key = f"{config_name}_{self.environment}"
            self.config_cache[cache_key] = config

        except Exception as e:
            logger.error(f"Failed to save configuration {config_name}: {e}")
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def list_available_configs(self) -> list[str]:
        """List all available configuration files.

        Returns:
            List of configuration names (without extensions)
        """
        if not self.config_dir.exists():
            return []

        configs = set()
        for config_file in self.config_dir.glob("*.yaml"):
            name = config_file.stem
            # Remove environment suffix if present
            if "." in name:
                name = name.split(".")[0]
            configs.add(name)

        return sorted(list(configs))

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge override dictionary into base dictionary.

        Args:
            base: Base dictionary to merge into
            override: Override dictionary to merge from
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _apply_env_overrides(
        self, config: dict[str, Any], config_name: str
    ) -> dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Args:
            config: Configuration dictionary
            config_name: Name of the configuration

        Returns:
            Configuration with environment overrides applied
        """
        prefix = f"AIVILLAGE_{config_name.upper()}_"

        for env_var, value in os.environ.items():
            if env_var.startswith(prefix):
                # Extract key path from environment variable
                key_path = env_var[len(prefix) :].lower().replace("_", ".")

                # Convert string value to appropriate type
                converted_value = self._convert_env_value(value)

                # Apply the override
                try:
                    keys = key_path.split(".")
                    current = config
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = converted_value

                    logger.debug(
                        f"Applied environment override: {key_path} = {converted_value}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to apply environment override {env_var}: {e}"
                    )

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value
        """
        # Try boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer
        try:
            return int(value)
        except ValueError:
            pass

        # Try float
        try:
            return float(value)
        except ValueError:
            pass

        # Try JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Return as string
        return value

    def _validate_config(self, config: dict[str, Any], config_name: str) -> None:
        """Validate configuration against schema.

        Args:
            config: Configuration to validate
            config_name: Name of the configuration

        Raises:
            ConfigurationError: If validation fails
        """
        # Basic validation rules
        validation_rules = {
            "logging": {
                "level": lambda x: x
                in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                "format": lambda x: isinstance(x, str) and len(x) > 0,
            },
            "monitoring": {
                "interval": lambda x: isinstance(x, (int, float)) and x > 0,
                "thresholds.cpu_percent_high": lambda x: isinstance(x, (int, float))
                and 0 <= x <= 100,
                "thresholds.memory_percent_high": lambda x: isinstance(x, (int, float))
                and 0 <= x <= 100,
            },
            "validation": {
                "timeout": lambda x: isinstance(x, (int, float)) and x > 0,
                "retry_attempts": lambda x: isinstance(x, int) and x >= 0,
            },
        }

        # Apply validation rules if they exist for this config
        if config_name in validation_rules:
            rules = validation_rules[config_name]
            for key_path, validator in rules.items():
                try:
                    value = self._get_nested_value(config, key_path)
                    if not validator(value):
                        raise ConfigurationError(
                            f"Invalid value for {key_path}: {value}"
                        )
                except KeyError:
                    # Key is optional if not in config
                    pass

    def _get_nested_value(self, config: dict[str, Any], key_path: str) -> Any:
        """Get nested value from configuration using dot notation.

        Args:
            config: Configuration dictionary
            key_path: Dot-separated path to the value

        Returns:
            The nested value

        Raises:
            KeyError: If key path is not found
        """
        keys = key_path.split(".")
        value = config

        for key in keys:
            value = value[key]

        return value


# Global configuration manager instance
config_manager = ConfigManager()
