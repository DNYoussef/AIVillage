"""Orchestration service configuration using the shared loader."""

from __future__ import annotations

import os
from typing import Any

from common.config import load_config, save_config

from .model_config import TaskType

DEFAULTS = {
    # API Configuration
    "openrouter_api_key": None,
    "openrouter_enabled": True,
    # Cost Management
    "daily_budget_usd": 50.0,
    "cost_tracking_enabled": True,
    "cost_alert_threshold": 0.8,
    # Performance Settings
    "enable_caching": True,
    "cache_ttl_seconds": 3600,
    "parallel_requests": 5,
    # Model Selection
    "prefer_opensource": False,
    "quality_threshold": 0.8,
    # Monitoring
    "wandb_enabled": True,
    "wandb_project": "agent-forge-orchestration",
    "metrics_export_interval": 300,
    # Fallback Settings
    "local_model_fallback": True,
    "fallback_after_errors": 3,
    "fallback_timeout_seconds": 30,
}


class OrchestrationConfig:
    """Configuration wrapper for orchestration service."""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_path = config_path or os.getenv("ORCHESTRATION_CONFIG_PATH")
        self.config = load_config(
            DEFAULTS,
            self.config_path,
            env_prefix="ORCHESTRATION_",
            required=["openrouter_api_key"],
        )

    def get(self, key: str, default: Any | None = None) -> Any:
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.config[key] = value

    def save(self, path: str | None = None) -> None:
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path provided for saving configuration")
        save_config(self.config, save_path)

    def get_task_config(self, task_type: TaskType) -> dict[str, Any]:
        task_key = f"task_{task_type.value}"
        if task_key in self.config:
            return self.config[task_key]
        return {
            "enabled": True,
            "quality_priority": task_type.value in {"problem_generation", "mathematical_reasoning"},
            "cost_sensitive": task_type.value in {"evaluation_grading", "content_variation"},
        }

    def validate(self) -> bool:
        """Compatibility method; required keys are checked during load."""
        return True


_config: OrchestrationConfig | None = None


def get_config() -> OrchestrationConfig:
    global _config
    if _config is None:
        _config = OrchestrationConfig()
    return _config


def set_config(config: OrchestrationConfig) -> None:
    global _config
    _config = config
