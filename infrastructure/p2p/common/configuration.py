"""
Configuration management utilities for P2P infrastructure.

Provides standardized configuration loading, validation,
and management across all P2P components.
"""

import os
import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

try:
    import tomli  # For TOML support
    HAS_TOML = True
except ImportError:
    HAS_TOML = False
    tomli = None

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    ENV = "env"


@dataclass
class ConfigValidationRule:
    """Configuration validation rule."""
    field_path: str  # dot-separated path like "transport.libp2p.enabled"
    required: bool = False
    field_type: Optional[Type] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    default_value: Optional[Any] = None
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Validate configuration field."""
        # Navigate to field using dot notation
        value = config
        field_parts = self.field_path.split('.')
        
        try:
            for part in field_parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    # Field not found
                    if self.required:
                        logger.error(f"Required field missing: {self.field_path}")
                        return False
                    return True  # Optional field not present is OK
            
            # Check type
            if self.field_type and not isinstance(value, self.field_type):
                logger.error(f"Field {self.field_path} has wrong type: expected {self.field_type.__name__}, got {type(value).__name__}")
                return False
            
            # Check allowed values
            if self.allowed_values and value not in self.allowed_values:
                logger.error(f"Field {self.field_path} has invalid value: {value}, allowed: {self.allowed_values}")
                return False
            
            # Check numeric ranges
            if self.min_value is not None and value < self.min_value:
                logger.error(f"Field {self.field_path} below minimum: {value} < {self.min_value}")
                return False
            
            if self.max_value is not None and value > self.max_value:
                logger.error(f"Field {self.field_path} above maximum: {value} > {self.max_value}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating field {self.field_path}: {e}")
            return False


class ConfigSource(ABC):
    """Abstract base class for configuration sources."""
    
    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from source."""
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Get configuration source name."""
        pass


class FileConfig(ConfigSource):
    """File-based configuration source."""
    
    def __init__(self, file_path: Union[str, Path], format: Optional[ConfigFormat] = None):
        self.file_path = Path(file_path)
        
        # Auto-detect format from extension if not specified
        if format is None:
            ext = self.file_path.suffix.lower()
            if ext == '.json':
                format = ConfigFormat.JSON
            elif ext in ['.yaml', '.yml']:
                format = ConfigFormat.YAML
            elif ext == '.toml':
                format = ConfigFormat.TOML
            else:
                raise ValueError(f"Cannot auto-detect format for {file_path}")
        
        self.format = format
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not self.file_path.exists():
            logger.warning(f"Configuration file not found: {self.file_path}")
            return {}
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                if self.format == ConfigFormat.JSON:
                    return json.load(f)
                elif self.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif self.format == ConfigFormat.TOML:
                    if not HAS_TOML:
                        raise ImportError("tomli library required for TOML support")
                    return tomli.load(f)
                else:
                    raise ValueError(f"Unsupported format: {self.format}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.file_path}: {e}")
            return {}
    
    @property
    def source_name(self) -> str:
        return f"file:{self.file_path}"


class EnvironmentConfig(ConfigSource):
    """Environment variable configuration source."""
    
    def __init__(self, prefix: str = "P2P_", separator: str = "_"):
        self.prefix = prefix
        self.separator = separator
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.prefix):
                # Remove prefix and convert to nested dict
                config_key = key[len(self.prefix):].lower()
                
                # Handle nested keys (e.g., TRANSPORT_LIBP2P_ENABLED -> transport.libp2p.enabled)
                if self.separator in config_key:
                    parts = config_key.split(self.separator)
                    current = config
                    
                    # Navigate/create nested structure
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Set final value
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    config[config_key] = self._parse_env_value(value)
        
        return config
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type."""
        # Try to parse as JSON first (for complex values)
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # Parse common types
        value_lower = value.lower()
        if value_lower in ['true', 'yes', '1', 'on']:
            return True
        elif value_lower in ['false', 'no', '0', 'off']:
            return False
        elif value_lower in ['null', 'none', '']:
            return None
        
        # Try numeric parsing
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    @property
    def source_name(self) -> str:
        return f"env:{self.prefix}*"


class ConfigManager:
    """Configuration manager with multiple sources and validation."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.sources: List[ConfigSource] = []
        self.validation_rules: List[ConfigValidationRule] = []
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def add_source(self, source: ConfigSource, priority: int = 0):
        """Add configuration source."""
        self.sources.append((priority, source))
        self.sources.sort(key=lambda x: x[0])  # Sort by priority
        self._config_cache = None  # Invalidate cache
    
    def add_file_source(self, file_path: Union[str, Path], 
                       format: Optional[ConfigFormat] = None, priority: int = 0):
        """Add file configuration source."""
        source = FileConfig(file_path, format)
        self.add_source(source, priority)
    
    def add_env_source(self, prefix: str = None, priority: int = 100):
        """Add environment variable source."""
        if prefix is None:
            prefix = f"{self.component_name.upper()}_"
        source = EnvironmentConfig(prefix)
        self.add_source(source, priority)
    
    def add_validation_rule(self, rule: ConfigValidationRule):
        """Add validation rule."""
        self.validation_rules.append(rule)
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load and merge configuration from all sources."""
        if self._config_cache is not None and not force_reload:
            return self._config_cache
        
        merged_config = {}
        
        # Load from sources in priority order (lower priority first)
        for priority, source in self.sources:
            try:
                source_config = source.load_config()
                logger.debug(f"Loaded config from {source.source_name}: {len(source_config)} keys")
                merged_config = self._deep_merge(merged_config, source_config)
            except Exception as e:
                logger.error(f"Failed to load config from {source.source_name}: {e}")
        
        # Apply defaults from validation rules
        for rule in self.validation_rules:
            if rule.default_value is not None:
                if not self._has_field(merged_config, rule.field_path):
                    self._set_field(merged_config, rule.field_path, rule.default_value)
        
        self._config_cache = merged_config
        return merged_config
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration against rules."""
        if config is None:
            config = self.load_config()
        
        all_valid = True
        for rule in self.validation_rules:
            if not rule.validate(config):
                all_valid = False
        
        return all_valid
    
    def get_value(self, field_path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        config = self.load_config()
        
        value = config
        for part in field_path.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set_value(self, field_path: str, value: Any):
        """Set configuration value (in memory only)."""
        config = self.load_config()
        self._set_field(config, field_path, value)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _has_field(self, config: Dict[str, Any], field_path: str) -> bool:
        """Check if field exists in config."""
        value = config
        for part in field_path.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return False
        return True
    
    def _set_field(self, config: Dict[str, Any], field_path: str, value: Any):
        """Set field value in config."""
        parts = field_path.split('.')
        current = config
        
        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set final value
        current[parts[-1]] = value


# Utility functions
def load_config(*sources: ConfigSource) -> Dict[str, Any]:
    """Load and merge configuration from multiple sources."""
    manager = ConfigManager("p2p")
    for i, source in enumerate(sources):
        manager.add_source(source, i)
    return manager.load_config()


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries."""
    result = {}
    manager = ConfigManager("merge")
    
    for config in configs:
        result = manager._deep_merge(result, config)
    
    return result


def validate_config(config: Dict[str, Any], rules: List[ConfigValidationRule]) -> bool:
    """Validate configuration against rules."""
    all_valid = True
    for rule in rules:
        if not rule.validate(config):
            all_valid = False
    return all_valid


def get_default_p2p_config() -> Dict[str, Any]:
    """Get default P2P configuration."""
    return {
        "transport": {
            "priority": ["libp2p", "bitchat", "betanet", "websocket"],
            "libp2p": {
                "enabled": True,
                "port": 0,  # Auto-assign
                "nat_traversal": True,
                "discovery": {
                    "mdns": True,
                    "dht": True,
                    "bootstrap_peers": []
                }
            },
            "bitchat": {
                "enabled": True,
                "ble_enabled": False,  # Requires hardware support
                "mesh_ttl": 5,
                "discovery_interval": 30
            },
            "betanet": {
                "enabled": False,  # Privacy-focused, opt-in
                "mixnode_count": 3,
                "circuit_timeout": 60,
                "onion_layers": 3
            },
            "websocket": {
                "enabled": True,
                "port": 8080,
                "ssl_enabled": False
            }
        },
        "security": {
            "encryption": {
                "default_algorithm": "aes_256_gcm",
                "key_rotation_interval": 3600  # 1 hour
            },
            "authentication": {
                "required": True,
                "timeout": 30
            }
        },
        "monitoring": {
            "enabled": True,
            "metrics_interval": 60,
            "prometheus_enabled": False,
            "log_level": "INFO"
        },
        "performance": {
            "max_connections": 100,
            "message_timeout": 30,
            "retry_attempts": 3,
            "connection_pool_size": 10
        }
    }


def get_p2p_validation_rules() -> List[ConfigValidationRule]:
    """Get standard P2P configuration validation rules."""
    return [
        # Transport settings
        ConfigValidationRule("transport.priority", required=True, field_type=list, 
                           default_value=["libp2p", "bitchat", "betanet", "websocket"]),
        ConfigValidationRule("transport.libp2p.enabled", field_type=bool, default_value=True),
        ConfigValidationRule("transport.libp2p.port", field_type=int, min_value=0, max_value=65535, default_value=0),
        
        # Security settings
        ConfigValidationRule("security.encryption.default_algorithm", field_type=str,
                           allowed_values=["aes_256_gcm", "chacha20_poly1305"], 
                           default_value="aes_256_gcm"),
        ConfigValidationRule("security.authentication.required", field_type=bool, default_value=True),
        ConfigValidationRule("security.authentication.timeout", field_type=int, min_value=1, max_value=300,
                           default_value=30),
        
        # Performance settings
        ConfigValidationRule("performance.max_connections", field_type=int, min_value=1, max_value=1000,
                           default_value=100),
        ConfigValidationRule("performance.message_timeout", field_type=int, min_value=1, max_value=300,
                           default_value=30),
        ConfigValidationRule("performance.retry_attempts", field_type=int, min_value=1, max_value=10,
                           default_value=3),
        
        # Monitoring settings
        ConfigValidationRule("monitoring.log_level", field_type=str,
                           allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                           default_value="INFO")
    ]


# Create default P2P config manager
def create_p2p_config_manager(component_name: str, 
                             config_files: Optional[List[Union[str, Path]]] = None) -> ConfigManager:
    """Create P2P configuration manager with standard setup."""
    manager = ConfigManager(component_name)
    
    # Add default config as lowest priority
    class DefaultConfigSource(ConfigSource):
        def load_config(self) -> Dict[str, Any]:
            return get_default_p2p_config()
        
        @property
        def source_name(self) -> str:
            return "defaults"
    
    manager.add_source(DefaultConfigSource(), priority=-100)
    
    # Add file sources
    if config_files:
        for i, file_path in enumerate(config_files):
            manager.add_file_source(file_path, priority=i)
    
    # Add environment source (highest priority)
    manager.add_env_source(priority=100)
    
    # Add validation rules
    for rule in get_p2p_validation_rules():
        manager.add_validation_rule(rule)
    
    return manager
