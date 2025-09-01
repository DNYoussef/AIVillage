"""Configuration management system with environment overrides."""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Union

from .task_constants import TaskConstants
from .project_constants import ProjectConstants
from .timing_constants import TimingConstants
from .performance_constants import PerformanceConstants

T = TypeVar("T")


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration overrides."""

    # Task configuration
    batch_size: Optional[int] = None
    default_priority: Optional[int] = None
    max_retries: Optional[int] = None

    # Timing configuration
    batch_processing_interval: Optional[float] = None
    retry_delay: Optional[float] = None
    default_timeout: Optional[int] = None

    # Performance configuration
    learning_rate: Optional[float] = None
    history_length: Optional[int] = None
    max_performance: Optional[float] = None
    min_performance: Optional[float] = None

    # Project configuration
    default_project_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class ConfigManager:
    """Manages configuration with environment variable overrides."""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.

        Args:
            config_file: Optional path to JSON configuration file
        """
        self._config_file = Path(config_file) if config_file else None
        self._environment_config: Optional[EnvironmentConfig] = None
        self._load_configuration()

    def _load_configuration(self) -> None:
        """Load configuration from file and environment variables."""
        config_data = {}

        # Load from configuration file if provided
        if self._config_file and self._config_file.exists():
            with open(self._config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

        # Override with environment variables
        env_overrides = self._load_from_environment()
        config_data.update(env_overrides)

        # Create environment configuration
        self._environment_config = EnvironmentConfig(**config_data)

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Task configuration environment variables
        if batch_size := os.getenv("TASK_BATCH_SIZE"):
            env_config["batch_size"] = int(batch_size)

        if default_priority := os.getenv("TASK_DEFAULT_PRIORITY"):
            env_config["default_priority"] = int(default_priority)

        if max_retries := os.getenv("TASK_MAX_RETRIES"):
            env_config["max_retries"] = int(max_retries)

        # Timing configuration environment variables
        if batch_interval := os.getenv("TIMING_BATCH_PROCESSING_INTERVAL"):
            env_config["batch_processing_interval"] = float(batch_interval)

        if retry_delay := os.getenv("TIMING_RETRY_DELAY"):
            env_config["retry_delay"] = float(retry_delay)

        if timeout := os.getenv("TIMING_DEFAULT_TIMEOUT"):
            env_config["default_timeout"] = int(timeout)

        # Performance configuration environment variables
        if learning_rate := os.getenv("PERFORMANCE_LEARNING_RATE"):
            env_config["learning_rate"] = float(learning_rate)

        if history_length := os.getenv("PERFORMANCE_HISTORY_LENGTH"):
            env_config["history_length"] = int(history_length)

        if max_performance := os.getenv("PERFORMANCE_MAX_MULTIPLIER"):
            env_config["max_performance"] = float(max_performance)

        if min_performance := os.getenv("PERFORMANCE_MIN_MULTIPLIER"):
            env_config["min_performance"] = float(min_performance)

        # Project configuration environment variables
        if project_status := os.getenv("PROJECT_DEFAULT_STATUS"):
            env_config["default_project_status"] = project_status

        return env_config

    def get_task_batch_size(self) -> int:
        """Get configured batch size for task processing."""
        if self._environment_config and self._environment_config.batch_size is not None:
            return self._validate_range(
                self._environment_config.batch_size,
                TaskConstants.MIN_BATCH_SIZE,
                TaskConstants.MAX_BATCH_SIZE,
                "batch_size",
            )
        return TaskConstants.DEFAULT_BATCH_SIZE

    def get_default_priority(self) -> int:
        """Get configured default priority for tasks."""
        if self._environment_config and self._environment_config.default_priority is not None:
            return self._validate_range(
                self._environment_config.default_priority,
                TaskConstants.MIN_PRIORITY,
                TaskConstants.MAX_PRIORITY,
                "default_priority",
            )
        return TaskConstants.DEFAULT_PRIORITY

    def get_max_retries(self) -> int:
        """Get configured maximum retries for tasks."""
        if self._environment_config and self._environment_config.max_retries is not None:
            return max(0, self._environment_config.max_retries)
        return TaskConstants.MAX_RETRIES

    def get_batch_processing_interval(self) -> float:
        """Get configured batch processing interval."""
        if self._environment_config and self._environment_config.batch_processing_interval is not None:
            return max(0.01, self._environment_config.batch_processing_interval)
        return TimingConstants.BATCH_PROCESSING_INTERVAL

    def get_retry_delay(self) -> float:
        """Get configured retry delay."""
        if self._environment_config and self._environment_config.retry_delay is not None:
            return max(0.0, self._environment_config.retry_delay)
        return TimingConstants.RETRY_DELAY

    def get_default_timeout(self) -> int:
        """Get configured default timeout."""
        if self._environment_config and self._environment_config.default_timeout is not None:
            return max(1, self._environment_config.default_timeout)
        return TimingConstants.DEFAULT_TIMEOUT

    def get_learning_rate(self) -> float:
        """Get configured learning rate for performance model."""
        if self._environment_config and self._environment_config.learning_rate is not None:
            return self._validate_range(self._environment_config.learning_rate, 0.001, 1.0, "learning_rate")
        return PerformanceConstants.LEARNING_RATE

    def get_history_length(self) -> int:
        """Get configured history length for performance tracking."""
        if self._environment_config and self._environment_config.history_length is not None:
            return max(10, self._environment_config.history_length)
        return PerformanceConstants.HISTORY_LENGTH

    def get_max_performance_multiplier(self) -> float:
        """Get configured maximum performance multiplier."""
        if self._environment_config and self._environment_config.max_performance is not None:
            return max(1.0, self._environment_config.max_performance)
        return PerformanceConstants.MAX_PERFORMANCE

    def get_min_performance_multiplier(self) -> float:
        """Get configured minimum performance multiplier."""
        if self._environment_config and self._environment_config.min_performance is not None:
            return self._validate_range(self._environment_config.min_performance, 0.1, 1.0, "min_performance")
        return PerformanceConstants.MIN_PERFORMANCE

    def get_default_project_status(self) -> str:
        """Get configured default project status."""
        if self._environment_config and self._environment_config.default_project_status is not None:
            return self._environment_config.default_project_status
        return ProjectConstants.DEFAULT_STATUS

    @staticmethod
    def _validate_range(
        value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float], param_name: str
    ) -> Union[int, float]:
        """Validate that a value is within the specified range."""
        if not min_val <= value <= max_val:
            raise ValueError(
                f"Configuration parameter '{param_name}' must be between {min_val} and {max_val}, " f"but got {value}"
            )
        return value

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate all configuration values and return summary."""
        validation_results = {"valid": True, "errors": [], "warnings": [], "configuration": {}}

        try:
            config = {
                "task_batch_size": self.get_task_batch_size(),
                "default_priority": self.get_default_priority(),
                "max_retries": self.get_max_retries(),
                "batch_processing_interval": self.get_batch_processing_interval(),
                "retry_delay": self.get_retry_delay(),
                "default_timeout": self.get_default_timeout(),
                "learning_rate": self.get_learning_rate(),
                "history_length": self.get_history_length(),
                "max_performance_multiplier": self.get_max_performance_multiplier(),
                "min_performance_multiplier": self.get_min_performance_multiplier(),
                "default_project_status": self.get_default_project_status(),
            }
            validation_results["configuration"] = config

        except ValueError as e:
            validation_results["valid"] = False
            validation_results["errors"].append(str(e))

        return validation_results

    def save_configuration(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file."""
        if self._environment_config:
            config_dict = self._environment_config.to_dict()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

    def reload_configuration(self) -> None:
        """Reload configuration from file and environment."""
        self._load_configuration()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def set_config_file(config_file: Union[str, Path]) -> None:
    """Set configuration file and reload configuration."""
    global _config_manager
    _config_manager = ConfigManager(config_file)
