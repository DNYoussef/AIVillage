"""
Base Configuration Management System

Provides structured configuration handling with validation,
environment variable support, and hierarchical configuration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, Type, Union, get_type_hints
import os
import json
import yaml
from pathlib import Path
import logging

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(ABC):
    """
    Base configuration class that provides:
    - Automatic validation
    - Environment variable support
    - JSON/YAML serialization
    - Hierarchical configuration loading
    """
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    @abstractmethod
    def validate(self) -> None:
        """Validate configuration values. Should raise ConfigurationError if invalid."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if is_dataclass(value):
                result[field_info.name] = value.to_dict()
            elif isinstance(value, (list, tuple)) and value and is_dataclass(value[0]):
                result[field_info.name] = [item.to_dict() for item in value]
            else:
                result[field_info.name] = value
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        try:
            # Get type hints to handle nested configurations
            type_hints = get_type_hints(cls)
            processed_data = {}
            
            for key, value in data.items():
                if key in type_hints:
                    expected_type = type_hints[key]
                    if hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
                        # Handle list of configurations
                        item_type = expected_type.__args__[0]
                        if is_dataclass(item_type):
                            processed_data[key] = [item_type.from_dict(item) for item in value]
                        else:
                            processed_data[key] = value
                    elif is_dataclass(expected_type):
                        # Handle nested configuration
                        processed_data[key] = expected_type.from_dict(value)
                    else:
                        processed_data[key] = value
                else:
                    processed_data[key] = value
            
            return cls(**processed_data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create {cls.__name__} from dictionary",
                details={"data": data, "error": str(e)}
            )
    
    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {file_path}",
                details={"file_path": str(file_path), "error": str(e)}
            )
    
    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML file."""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            return cls.from_dict(data)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration from {file_path}",
                details={"file_path": str(file_path), "error": str(e)}
            )
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'auto') -> None:
        """
        Save configuration to file.
        
        Args:
            file_path: Path to save file
            format: Format ('json', 'yaml', or 'auto' to detect from extension)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'auto':
            format = 'yaml' if file_path.suffix in ['.yml', '.yaml'] else 'json'
        
        try:
            if format == 'yaml':
                with open(file_path, 'w') as f:
                    f.write(self.to_yaml())
            else:
                with open(file_path, 'w') as f:
                    f.write(self.to_json())
        except Exception as e:
            raise ConfigurationError(
                f"Failed to save configuration to {file_path}",
                details={"file_path": str(file_path), "format": format, "error": str(e)}
            )
    
    def update_from_env(self, prefix: str = "") -> None:
        """
        Update configuration values from environment variables.
        
        Args:
            prefix: Environment variable prefix (e.g., "AIVILLAGE_")
        """
        for field_info in fields(self):
            env_key = f"{prefix}{field_info.name.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                try:
                    # Attempt to convert environment variable to appropriate type
                    if field_info.type == bool:
                        value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif field_info.type == int:
                        value = int(env_value)
                    elif field_info.type == float:
                        value = float(env_value)
                    elif field_info.type == str:
                        value = env_value
                    else:
                        # Try JSON parsing for complex types
                        try:
                            value = json.loads(env_value)
                        except json.JSONDecodeError:
                            value = env_value
                    
                    setattr(self, field_info.name, value)
                    logger.info(f"Updated {field_info.name} from environment variable {env_key}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to set {field_info.name} from environment variable {env_key}: {e}")
        
        # Re-validate after updates
        self.validate()


@dataclass
class ResourceConfig(BaseConfig):
    """Configuration for resource management."""
    
    max_memory_gb: float = 8.0
    max_cpu_cores: int = 4
    gpu_memory_fraction: float = 0.8
    temp_dir: str = "/tmp/aivillage"
    
    def validate(self) -> None:
        if self.max_memory_gb <= 0:
            raise ConfigurationError("max_memory_gb must be positive")
        if self.max_cpu_cores <= 0:
            raise ConfigurationError("max_cpu_cores must be positive")
        if not 0 < self.gpu_memory_fraction <= 1:
            raise ConfigurationError("gpu_memory_fraction must be between 0 and 1")


@dataclass 
class LoggingConfig(BaseConfig):
    """Configuration for logging system."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_bytes: int = 10_000_000  # 10MB
    backup_count: int = 3
    
    def validate(self) -> None:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ConfigurationError(
                f"Invalid log level: {self.level}. Must be one of {valid_levels}"
            )


class ConfigManager:
    """
    Centralized configuration manager for the entire application.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path.cwd()
        self.configs: Dict[str, BaseConfig] = {}
        self._config_files: Dict[str, Path] = {}
    
    def register_config(
        self, 
        name: str, 
        config: BaseConfig, 
        file_path: Optional[Path] = None
    ) -> None:
        """Register a configuration object."""
        self.configs[name] = config
        if file_path:
            self._config_files[name] = file_path
    
    def get_config(self, name: str) -> BaseConfig:
        """Get a configuration by name."""
        if name not in self.configs:
            raise ConfigurationError(f"Configuration '{name}' not found")
        return self.configs[name]
    
    def load_from_directory(self, config_dir: Path) -> None:
        """Load all configurations from a directory."""
        if not config_dir.exists():
            logger.warning(f"Configuration directory {config_dir} does not exist")
            return
            
        for config_file in config_dir.glob("*.json"):
            try:
                self._load_config_file(config_file)
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
        
        for config_file in config_dir.glob("*.yaml"):
            try:
                self._load_config_file(config_file)
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    def _load_config_file(self, file_path: Path) -> None:
        """Load a single configuration file."""
        # This would need to be extended with a registry of config classes
        # For now, it's a placeholder for the pattern
        pass
    
    def save_all_configs(self) -> None:
        """Save all registered configurations to their files."""
        for name, config in self.configs.items():
            if name in self._config_files:
                config.save_to_file(self._config_files[name])
    
    def update_all_from_env(self, prefix: str = "AIVILLAGE_") -> None:
        """Update all configurations from environment variables."""
        for config in self.configs.values():
            config.update_from_env(prefix)