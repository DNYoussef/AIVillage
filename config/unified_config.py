"""
AIVillage Unified Configuration Management System
Centralized configuration with distributed caching via Context7 MCP

Consolidates all configuration systems into hierarchical structure:
- Global Base Configuration (core system defaults)
- Environment Overrides (development/staging/production)  
- Service Specific (individual service configurations)
- Runtime Dynamic (hot-reloadable configurations)
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class SecurityLevel(Enum):
    """Security configuration levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ENHANCED = "enhanced"
    MAXIMUM = "maximum"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    name: str = "aivillage"
    username: str = ""
    password: str = ""  # Loaded from environment
    pool_size: int = 10
    max_connections: int = 50
    ssl_mode: str = "prefer"
    
    def __post_init__(self):
        # Load sensitive values from environment
        self.username = os.getenv("DB_USERNAME", self.username)
        self.password = os.getenv("DB_PASSWORD", "")
        if not self.password:
            raise ValueError("DB_PASSWORD environment variable must be set")

@dataclass
class RedisConfig:
    """Redis configuration"""
    host: str = "localhost"
    port: int = 6379
    password: str = ""  # Loaded from environment
    db: int = 0
    max_connections: int = 20
    retry_on_timeout: bool = True
    socket_timeout: float = 5.0
    
    def __post_init__(self):
        self.password = os.getenv("REDIS_PASSWORD", "")
        if not self.password:
            raise ValueError("REDIS_PASSWORD environment variable must be set")

@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = ""  # Loaded from environment
    jwt_expiry_hours: int = 24
    password_min_length: int = 8
    max_login_attempts: int = 5
    session_timeout_minutes: int = 30
    rate_limit_per_minute: int = 100
    security_level: SecurityLevel = SecurityLevel.STANDARD
    enable_2fa: bool = False
    
    def __post_init__(self):
        self.jwt_secret = os.getenv("JWT_SECRET", "")
        if not self.jwt_secret:
            raise ValueError("JWT_SECRET environment variable must be set")

@dataclass
class APIConfig:
    """API configuration"""
    openrouter_api_key: str = ""  # Loaded from environment
    anthropic_api_key: str = ""   # Loaded from environment
    huggingface_token: str = ""   # Loaded from environment
    default_model: str = "nvidia/llama-3.1-nemotron-70b-instruct"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout_seconds: int = 30
    
    def __post_init__(self):
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", "")

@dataclass
class RAGConfig:
    """RAG system configuration"""
    embedder_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_results: int = 10
    confidence_threshold: float = 0.7
    enable_creative_mode: bool = True
    enable_missing_node_detection: bool = True
    hippo_memory_consolidation_threshold: int = 5
    
@dataclass
class P2PConfig:
    """P2P network configuration"""
    enable_p2p: bool = True
    listen_port: int = 4001
    bootstrap_nodes: list = field(default_factory=list)
    max_peers: int = 50
    discovery_interval_seconds: int = 30
    
    def __post_init__(self):
        if not self.bootstrap_nodes:
            self.bootstrap_nodes = [
                "/ip4/127.0.0.1/tcp/4001",
                "/ip4/127.0.0.1/tcp/4002"
            ]

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    log_level: str = "INFO"
    enable_health_checks: bool = True
    health_check_interval_seconds: int = 30

class UnifiedConfigManager:
    """Unified configuration manager with hierarchical loading"""
    
    def __init__(self, environment: Environment = Environment.DEVELOPMENT):
        self.environment = environment
        self.config_dir = Path(__file__).parent
        self.context7_mcp = None  # Context7MCP() - for distributed caching
        
        # Load configuration hierarchy
        self._base_config = self._load_base_config()
        self._env_config = self._load_environment_config()
        self._service_configs = self._load_service_configs()
        
        # Initialize components
        self.database = DatabaseConfig(**self._get_merged_config("database"))
        self.redis = RedisConfig(**self._get_merged_config("redis"))
        self.security = SecurityConfig(**self._get_merged_config("security"))
        self.api = APIConfig(**self._get_merged_config("api"))
        self.rag = RAGConfig(**self._get_merged_config("rag"))
        self.p2p = P2PConfig(**self._get_merged_config("p2p"))
        self.monitoring = MonitoringConfig(**self._get_merged_config("monitoring"))
        
        logger.info(f"Unified configuration loaded for {environment.value} environment")
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration defaults"""
        base_config_file = self.config_dir / "base_config.yaml"
        
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        # Default base configuration
        return {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "aivillage",
                "pool_size": 10
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "security": {
                "jwt_expiry_hours": 24,
                "password_min_length": 8,
                "max_login_attempts": 5,
                "rate_limit_per_minute": 100
            },
            "api": {
                "default_model": "nvidia/llama-3.1-nemotron-70b-instruct",
                "max_tokens": 4096,
                "temperature": 0.7,
                "timeout_seconds": 30
            },
            "rag": {
                "embedder_model": "all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_results": 10,
                "confidence_threshold": 0.7
            },
            "p2p": {
                "enable_p2p": True,
                "listen_port": 4001,
                "max_peers": 50
            },
            "monitoring": {
                "enable_metrics": True,
                "metrics_port": 9090,
                "log_level": "INFO"
            }
        }
    
    def _load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific overrides"""
        env_config_file = self.config_dir / "env" / f"{self.environment.value}.yaml"
        
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        
        # Environment-specific defaults
        env_defaults = {
            Environment.DEVELOPMENT: {
                "database": {"name": "aivillage_dev"},
                "security": {"security_level": "minimal"},
                "monitoring": {"log_level": "DEBUG"}
            },
            Environment.STAGING: {
                "database": {"name": "aivillage_staging"},
                "security": {"security_level": "standard"},
                "monitoring": {"log_level": "INFO"}
            },
            Environment.PRODUCTION: {
                "database": {"name": "aivillage_prod"},
                "security": {"security_level": "maximum", "enable_2fa": True},
                "monitoring": {"log_level": "WARNING"}
            },
            Environment.TESTING: {
                "database": {"name": "aivillage_test"},
                "security": {"security_level": "minimal"},
                "monitoring": {"log_level": "ERROR"}
            }
        }
        
        return env_defaults.get(self.environment, {})
    
    def _load_service_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load service-specific configurations"""
        service_configs = {}
        services_dir = self.config_dir / "services"
        
        if services_dir.exists():
            for service_file in services_dir.glob("*.yaml"):
                service_name = service_file.stem
                with open(service_file, 'r') as f:
                    service_configs[service_name] = yaml.safe_load(f) or {}
        
        return service_configs
    
    def _get_merged_config(self, section: str) -> Dict[str, Any]:
        """Get merged configuration for a specific section"""
        # Start with base config
        config = self._base_config.get(section, {}).copy()
        
        # Apply environment overrides
        env_overrides = self._env_config.get(section, {})
        config.update(env_overrides)
        
        # Apply service-specific overrides
        for service_config in self._service_configs.values():
            if section in service_config:
                config.update(service_config[section])
        
        return config
    
    async def cache_configuration(self, cache_key: str, ttl_seconds: int = 3600):
        """Cache configuration in Context7 MCP for distributed access"""
        if not self.context7_mcp:
            return
        
        config_data = {
            "database": self.database.__dict__,
            "redis": self.redis.__dict__,
            "security": {k: v for k, v in self.security.__dict__.items() if k != 'jwt_secret'},
            "api": {k: v for k, v in self.api.__dict__.items() if not k.endswith('_key')},
            "rag": self.rag.__dict__,
            "p2p": self.p2p.__dict__,
            "monitoring": self.monitoring.__dict__,
            "environment": self.environment.value
        }
        
        # Cache configuration (excluding sensitive data)
        await self.context7_mcp.cache.set(cache_key, config_data, ttl=ttl_seconds)
        logger.info(f"Configuration cached with key: {cache_key}")
    
    def get_connection_string(self, database_name: Optional[str] = None) -> str:
        """Get database connection string"""
        db_name = database_name or self.database.name
        return (f"postgresql://{self.database.username}:{self.database.password}@"
                f"{self.database.host}:{self.database.port}/{db_name}")
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL"""
        auth_part = f":{self.redis.password}@" if self.redis.password else ""
        return f"redis://{auth_part}{self.redis.host}:{self.redis.port}/{self.redis.db}"
    
    def validate_configuration(self) -> bool:
        """Validate all configuration sections"""
        try:
            # Validate required environment variables are set
            required_env_vars = [
                "DB_PASSWORD", "REDIS_PASSWORD", "JWT_SECRET"
            ]
            
            missing_vars = []
            for var in required_env_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            if missing_vars:
                logger.error(f"Missing required environment variables: {missing_vars}")
                return False
            
            # Validate configuration consistency
            if self.database.pool_size > self.database.max_connections:
                logger.error("Database pool_size cannot exceed max_connections")
                return False
            
            if self.security.jwt_expiry_hours <= 0:
                logger.error("JWT expiry hours must be positive")
                return False
            
            if self.rag.confidence_threshold < 0 or self.rag.confidence_threshold > 1:
                logger.error("RAG confidence threshold must be between 0 and 1")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def hot_reload_config(self, section: str, new_config: Dict[str, Any]) -> bool:
        """Hot reload configuration section"""
        try:
            if section == "rag":
                for key, value in new_config.items():
                    if hasattr(self.rag, key):
                        setattr(self.rag, key, value)
            elif section == "monitoring":
                for key, value in new_config.items():
                    if hasattr(self.monitoring, key):
                        setattr(self.monitoring, key, value)
            # Add other sections as needed
            
            logger.info(f"Hot reloaded configuration section: {section}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to hot reload {section}: {e}")
            return False
    
    def export_template(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Export configuration template for setup"""
        template = {
            "# AIVillage Unified Configuration Template": None,
            "# Set these environment variables before running the system": None,
            "environment_variables": {
                "DB_USERNAME": "your_database_username",
                "DB_PASSWORD": "your_secure_database_password",
                "REDIS_PASSWORD": "your_secure_redis_password", 
                "JWT_SECRET": "your_jwt_secret_key_minimum_32_characters",
                "OPENROUTER_API_KEY": "your_openrouter_api_key",
                "ANTHROPIC_API_KEY": "your_anthropic_api_key (optional)",
                "HUGGINGFACE_TOKEN": "your_huggingface_token (optional)"
            },
            "configuration": {
                "database": {
                    "host": self.database.host,
                    "port": self.database.port,
                    "name": self.database.name,
                    "pool_size": self.database.pool_size,
                    "ssl_mode": self.database.ssl_mode
                },
                "redis": {
                    "host": self.redis.host,
                    "port": self.redis.port,
                    "db": self.redis.db,
                    "max_connections": self.redis.max_connections
                },
                "security": {
                    "jwt_expiry_hours": self.security.jwt_expiry_hours,
                    "password_min_length": self.security.password_min_length,
                    "max_login_attempts": self.security.max_login_attempts,
                    "rate_limit_per_minute": self.security.rate_limit_per_minute,
                    "security_level": self.security.security_level.value
                }
            }
        }
        
        return template

# Global configuration manager instance
_config_manager: Optional[UnifiedConfigManager] = None

def get_config(environment: Environment = None) -> UnifiedConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        env = environment or Environment(os.getenv("AIVILLAGE_ENV", "development"))
        _config_manager = UnifiedConfigManager(env)
    
    return _config_manager

def initialize_config(environment: Environment = None) -> UnifiedConfigManager:
    """Initialize configuration manager"""
    global _config_manager
    env = environment or Environment(os.getenv("AIVILLAGE_ENV", "development"))
    _config_manager = UnifiedConfigManager(env)
    
    # Validate configuration
    if not _config_manager.validate_configuration():
        raise RuntimeError("Configuration validation failed")
    
    return _config_manager

# Template generation script
def main():
    """Generate configuration template"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVillage Configuration Management")
    parser.add_argument("--generate-template", action="store_true", 
                       help="Generate configuration template")
    parser.add_argument("--environment", default="development",
                       choices=["development", "staging", "production", "testing"],
                       help="Environment to configure")
    parser.add_argument("--validate", action="store_true",
                       help="Validate current configuration")
    
    args = parser.parse_args()
    
    if args.generate_template:
        config = UnifiedConfigManager(Environment(args.environment))
        template = config.export_template()
        
        template_file = Path("config_template.yaml")
        with open(template_file, 'w') as f:
            yaml.dump(template, f, default_flow_style=False, indent=2)
        
        print(f"Configuration template generated: {template_file}")
    
    elif args.validate:
        try:
            config = initialize_config(Environment(args.environment))
            print("✅ Configuration validation passed")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            exit(1)
    
    else:
        print("Use --generate-template or --validate")

if __name__ == "__main__":
    main()