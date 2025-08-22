"""Unified configuration management for services.

This module provides a centralized configuration system that handles
environment variables, YAML files, and provides validation.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from core.error_handling import AIVillageException, ErrorCategory, ErrorSeverity, get_component_logger


class Environment(Enum):
    """Environment types."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "aivillage"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    ssl_mode: str = "prefer"


@dataclass
class RedisConfig:
    """Redis configuration."""

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    database: int = 0
    pool_size: int = 10


@dataclass
class ServiceConfig:
    """Individual service configuration."""

    name: str
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    log_level: str = "INFO"


@dataclass
class SecurityConfig:
    """Security configuration."""

    secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600  # 1 hour
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # 1 minute
    allowed_hosts: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enable_metrics: bool = True
    enable_tracing: bool = False
    metrics_port: int = 9090
    health_check_interval: int = 30
    log_format: str = "json"


@dataclass
class AIConfig:
    """AI/ML configuration."""

    model_name: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"
    max_context_length: int = 8192


@dataclass
class UnifiedConfig:
    """Unified configuration for all services."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Service configurations
    gateway: ServiceConfig = field(default_factory=lambda: ServiceConfig("gateway", port=8000))
    twin: ServiceConfig = field(default_factory=lambda: ServiceConfig("twin", port=8001))

    # Infrastructure
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)

    # Cross-cutting concerns
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    ai: AIConfig = field(default_factory=AIConfig)

    # External service URLs
    external_services: dict[str, str] = field(default_factory=dict)


class ConfigManager:
    """Configuration manager with environment and file support."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.logger = get_component_logger("ConfigManager")
        self.config_path = Path(config_path) if config_path else None
        self._config: UnifiedConfig | None = None

    def load(self) -> UnifiedConfig:
        """Load configuration from environment and files."""
        if self._config is not None:
            return self._config

        try:
            # Start with defaults
            config_data = {}

            # Load from YAML file if provided
            if self.config_path and self.config_path.exists():
                config_data = self._load_from_yaml(self.config_path)

            # Override with environment variables
            env_config = self._load_from_environment()
            config_data = self._merge_configs(config_data, env_config)

            # Create unified config
            self._config = self._create_config_from_dict(config_data)

            # Validate configuration
            self._validate_config(self._config)

            self.logger.info(
                "Configuration loaded successfully",
                extra={
                    "environment": self._config.environment.value,
                    "config_path": str(self.config_path) if self.config_path else None,
                },
            )

            return self._config

        except Exception as e:
            raise AIVillageException(
                message=f"Failed to load configuration: {e!s}",
                category=ErrorCategory.CONFIGURATION,
                severity=ErrorSeverity.CRITICAL,
                operation="load_config",
                context={"config_path": str(self.config_path) if self.config_path else None},
            )

    def _load_from_yaml(self, path: Path) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.exception(f"Failed to load YAML config from {path}: {e}")
            return {}

    def _load_from_environment(self) -> dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Environment
        if env := os.getenv("ENVIRONMENT"):
            config["environment"] = env

        config["debug"] = os.getenv("DEBUG", "false").lower() == "true"

        # Gateway service
        gateway_config = {}
        if host := os.getenv("GATEWAY_HOST"):
            gateway_config["host"] = host
        if port := os.getenv("GATEWAY_PORT"):
            gateway_config["port"] = int(port)
        if cors := os.getenv("GATEWAY_ALLOW_ORIGINS"):
            gateway_config["cors_origins"] = cors.split(",")
        if gateway_config:
            config["gateway"] = gateway_config

        # Twin service
        twin_config = {}
        if host := os.getenv("TWIN_HOST"):
            twin_config["host"] = host
        if port := os.getenv("TWIN_PORT"):
            twin_config["port"] = int(port)
        if twin_config:
            config["twin"] = twin_config

        # Database
        db_config = {}
        if host := os.getenv("DB_HOST"):
            db_config["host"] = host
        if port := os.getenv("DB_PORT"):
            db_config["port"] = int(port)
        if name := os.getenv("DB_NAME"):
            db_config["name"] = name
        if user := os.getenv("DB_USER"):
            db_config["user"] = user
        if password := os.getenv("DB_PASSWORD"):
            db_config["password"] = password
        if db_config:
            config["database"] = db_config

        # Redis
        redis_config = {}
        if host := os.getenv("REDIS_HOST"):
            redis_config["host"] = host
        if port := os.getenv("REDIS_PORT"):
            redis_config["port"] = int(port)
        if password := os.getenv("REDIS_PASSWORD"):
            redis_config["password"] = password
        if redis_config:
            config["redis"] = redis_config

        # Security
        security_config = {}
        if secret := os.getenv("SECRET_KEY"):
            security_config["secret_key"] = secret
        if rate_limit := os.getenv("RATE_LIMIT_REQUESTS"):
            security_config["rate_limit_requests"] = int(rate_limit)
        if rate_window := os.getenv("RATE_LIMIT_WINDOW"):
            security_config["rate_limit_window"] = int(rate_window)
        if security_config:
            config["security"] = security_config

        # AI
        ai_config = {}
        if model := os.getenv("AI_MODEL_NAME"):
            ai_config["model_name"] = model
        if max_tokens := os.getenv("AI_MAX_TOKENS"):
            ai_config["max_tokens"] = int(max_tokens)
        if temperature := os.getenv("AI_TEMPERATURE"):
            ai_config["temperature"] = float(temperature)
        if ai_config:
            config["ai"] = ai_config

        return config

    def _merge_configs(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Merge configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _create_config_from_dict(self, config_data: dict[str, Any]) -> UnifiedConfig:
        """Create UnifiedConfig from dictionary."""
        # Handle environment
        environment = Environment.DEVELOPMENT
        if env_str := config_data.get("environment"):
            try:
                environment = Environment(env_str.lower())
            except ValueError:
                self.logger.warning(f"Invalid environment: {env_str}, using development")

        # Create service configs
        gateway_data = config_data.get("gateway", {})
        gateway_config = ServiceConfig(
            name="gateway",
            **{k: v for k, v in gateway_data.items() if hasattr(ServiceConfig, k)},
        )

        twin_data = config_data.get("twin", {})
        twin_config = ServiceConfig(
            name="twin",
            **{k: v for k, v in twin_data.items() if hasattr(ServiceConfig, k)},
        )

        # Create other configs
        db_data = config_data.get("database", {})
        database_config = DatabaseConfig(**{k: v for k, v in db_data.items() if hasattr(DatabaseConfig, k)})

        redis_data = config_data.get("redis", {})
        redis_config = RedisConfig(**{k: v for k, v in redis_data.items() if hasattr(RedisConfig, k)})

        security_data = config_data.get("security", {})
        security_config = SecurityConfig(**{k: v for k, v in security_data.items() if hasattr(SecurityConfig, k)})

        monitoring_data = config_data.get("monitoring", {})
        monitoring_config = MonitoringConfig(
            **{k: v for k, v in monitoring_data.items() if hasattr(MonitoringConfig, k)}
        )

        ai_data = config_data.get("ai", {})
        ai_config = AIConfig(**{k: v for k, v in ai_data.items() if hasattr(AIConfig, k)})

        return UnifiedConfig(
            environment=environment,
            debug=config_data.get("debug", False),
            gateway=gateway_config,
            twin=twin_config,
            database=database_config,
            redis=redis_config,
            security=security_config,
            monitoring=monitoring_config,
            ai=ai_config,
            external_services=config_data.get("external_services", {}),
        )

    def _validate_config(self, config: UnifiedConfig) -> None:
        """Validate configuration."""
        errors = []

        # Validate required fields for production
        if config.environment == Environment.PRODUCTION:
            if not config.security.secret_key:
                errors.append("SECRET_KEY is required for production")

            if config.database.password == "":
                errors.append("Database password is required for production")

        # Validate port ranges
        for service_name, service_config in [
            ("gateway", config.gateway),
            ("twin", config.twin),
        ]:
            if not (1024 <= service_config.port <= 65535):
                errors.append(f"{service_name} port must be between 1024 and 65535")

        # Validate AI config
        if not (0.0 <= config.ai.temperature <= 2.0):
            errors.append("AI temperature must be between 0.0 and 2.0")

        if config.ai.max_tokens <= 0:
            errors.append("AI max_tokens must be positive")

        if errors:
            raise AIVillageException(
                message="Configuration validation failed",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.CRITICAL,
                operation="validate_config",
                context={"validation_errors": errors},
            )


# Global configuration manager
_config_manager: ConfigManager | None = None


def get_config(config_path: str | Path | None = None) -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config_manager

    if _config_manager is None:
        # Default config paths to check
        default_paths = [
            Path("config.yaml"),
            Path("configs/services.yaml"),
            Path("configs/config.yaml"),
        ]

        # Use provided path or find default
        if config_path:
            _config_manager = ConfigManager(config_path)
        else:
            config_file = None
            for path in default_paths:
                if path.exists():
                    config_file = path
                    break
            _config_manager = ConfigManager(config_file)

    return _config_manager.load()


def reload_config() -> UnifiedConfig:
    """Reload configuration from sources."""
    global _config_manager
    if _config_manager:
        _config_manager._config = None
    return get_config()
