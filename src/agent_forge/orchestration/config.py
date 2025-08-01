"""Configuration management for multi-model orchestration."""

import json
import os
from pathlib import Path
from typing import Any

import yaml

from .model_config import TaskType


class OrchestrationConfig:
    """Manages configuration for multi-model orchestration."""

    def __init__(self, config_path: str | None = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config_path = config_path or os.getenv("ORCHESTRATION_CONFIG_PATH")
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file or environment."""
        config = {
            # API Configuration
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            "openrouter_enabled": os.getenv("OPENROUTER_ENABLED", "true").lower()
            == "true",
            # Cost Management
            "daily_budget_usd": float(os.getenv("DAILY_BUDGET_USD", "50.0")),
            "cost_tracking_enabled": True,
            "cost_alert_threshold": 0.8,  # Alert at 80% of budget
            # Performance Settings
            "enable_caching": True,
            "cache_ttl_seconds": 3600,
            "parallel_requests": int(os.getenv("PARALLEL_REQUESTS", "5")),
            # Model Selection
            "prefer_opensource": os.getenv("PREFER_OPENSOURCE", "false").lower()
            == "true",
            "quality_threshold": float(os.getenv("QUALITY_THRESHOLD", "0.8")),
            # Monitoring
            "wandb_enabled": os.getenv("WANDB_ENABLED", "true").lower() == "true",
            "wandb_project": os.getenv("WANDB_PROJECT", "agent-forge-orchestration"),
            "metrics_export_interval": 300,  # 5 minutes
            # Fallback Settings
            "local_model_fallback": True,
            "fallback_after_errors": 3,
            "fallback_timeout_seconds": 30,
        }

        # Load from file if provided
        if self.config_path and Path(self.config_path).exists():
            file_config = self._load_from_file(self.config_path)
            config.update(file_config)

        return config

    def _load_from_file(self, path: str) -> dict[str, Any]:
        """Load configuration from JSON or YAML file."""
        path_obj = Path(path)

        with open(path_obj) as f:
            if path_obj.suffix == ".json":
                return json.load(f)
            if path_obj.suffix in [".yml", ".yaml"]:
                return yaml.safe_load(f)
            raise ValueError(f"Unsupported config format: {path_obj.suffix}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value

    def save(self, path: str | None = None):
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path provided for saving configuration")

        path_obj = Path(save_path)

        with open(path_obj, "w") as f:
            if path_obj.suffix == ".json":
                json.dump(self.config, f, indent=2)
            elif path_obj.suffix in [".yml", ".yaml"]:
                yaml.dump(self.config, f, default_flow_style=False)

    def get_task_config(self, task_type: TaskType) -> dict[str, Any]:
        """Get configuration for a specific task type."""
        # Check for task-specific overrides
        task_key = f"task_{task_type.value}"

        if task_key in self.config:
            return self.config[task_key]

        # Return defaults
        return {
            "enabled": True,
            "quality_priority": task_type.value
            in ["problem_generation", "mathematical_reasoning"],
            "cost_sensitive": task_type.value
            in ["evaluation_grading", "content_variation"],
        }

    def validate(self) -> bool:
        """Validate configuration."""
        required_keys = ["openrouter_api_key"]

        for key in required_keys:
            if not self.config.get(key):
                raise ValueError(f"Missing required configuration: {key}")

        return True


# Global configuration instance
_config = None


def get_config() -> OrchestrationConfig:
    """Get global configuration instance."""
    global _config

    if _config is None:
        _config = OrchestrationConfig()

    return _config


def set_config(config: OrchestrationConfig):
    """Set global configuration instance."""
    global _config
    _config = config
