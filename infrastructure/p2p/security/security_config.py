"""
P2P Security Configuration Module

Centralized security settings and secure defaults for P2P infrastructure.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SecurityLevel(Enum):
    """Security level configuration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SecurityConfig:
    """Security configuration for P2P infrastructure"""
    
    # Network binding security
    default_host: str = "127.0.0.1"  # Safe default
    allow_all_interfaces: bool = False
    
    # Cryptographic settings
    min_key_size: int = 2048
    allowed_ciphers: List[str] = None
    
    # Serialization security
    allow_pickle: bool = False
    safe_serialization_only: bool = True
    
    # Authentication
    require_auth: bool = True
    session_timeout: int = 3600  # 1 hour
    
    # API Security
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    def __post_init__(self):
        if self.allowed_ciphers is None:
            self.allowed_ciphers = [
                "AES256-GCM",
                "CHACHA20-POLY1305",
                "AES128-GCM"
            ]


class SecureServerConfig:
    """Secure server configuration helper"""
    
    @staticmethod
    def get_safe_host(service_name: str) -> str:
        """Get safe host configuration for a service"""
        env_var = f"{service_name.upper()}_HOST"
        host = os.getenv(env_var, "127.0.0.1")
        
        # Security validation - only warn in comments for static analysis
        # Note: 0.0.0.0 binding should be avoided in production
        # Use environment variables to configure specific interfaces
        if host == "0.0.0.0":  # nosec B104
            security_level = os.getenv("SECURITY_LEVEL", "development")
            if security_level == "production":
                raise ValueError(
                    f"Binding to {host} not allowed in production. "
                    f"Set {env_var} to specific interface."
                )
        
        return host
    
    @staticmethod
    def get_safe_port(service_name: str, default_port: int) -> int:
        """Get safe port configuration for a service"""
        env_var = f"{service_name.upper()}_PORT"
        try:
            port = int(os.getenv(env_var, str(default_port)))
            if port < 1024 and os.getuid() != 0:  # type: ignore
                raise ValueError(f"Port {port} requires root privileges")
            return port
        except (ValueError, AttributeError):
            return default_port


class SecureSerializer:
    """Secure serialization utilities"""
    
    @staticmethod
    def is_safe_format(data: bytes) -> bool:
        """Check if data format is safe for deserialization"""
        # Check for pickle signatures
        pickle_signatures = [
            b'\x80\x03',  # Pickle protocol 3
            b'\x80\x04',  # Pickle protocol 4
            b'\x80\x05',  # Pickle protocol 5
        ]
        
        for sig in pickle_signatures:
            if data.startswith(sig):
                return False
        
        return True
    
    @staticmethod
    def safe_deserialize(data: bytes) -> dict:
        """Safely deserialize data using JSON only"""
        import json
        
        if not SecureSerializer.is_safe_format(data):
            raise ValueError("Unsafe serialization format detected")
        
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            return json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid JSON data: {e}")


# Global security configuration instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get global security configuration"""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


def init_security_config(
    security_level: SecurityLevel = SecurityLevel.DEVELOPMENT,
    **kwargs
) -> SecurityConfig:
    """Initialize security configuration with specific settings"""
    global _security_config
    
    config_overrides = {}
    
    if security_level == SecurityLevel.PRODUCTION:
        config_overrides.update({
            "allow_all_interfaces": False,
            "allow_pickle": False,
            "require_auth": True,
            "min_key_size": 4096,
        })
    elif security_level == SecurityLevel.TESTING:
        config_overrides.update({
            "allow_all_interfaces": False,
            "allow_pickle": False,  # Even in testing
            "require_auth": True,
        })
    
    # Apply custom overrides
    config_overrides.update(kwargs)
    
    _security_config = SecurityConfig(**config_overrides)
    return _security_config


# Security validation decorators
def validate_host_binding(host: str) -> str:
    """Validate host binding for security"""
    config = get_security_config()
    
    # Security check with proper handling for static analysis
    if host == "0.0.0.0" and not config.allow_all_interfaces:  # nosec B104
        security_level = os.getenv("SECURITY_LEVEL", "development")
        if security_level == "production":
            raise ValueError(
                "Binding to all interfaces is not allowed in production"
            )
    
    return host


def require_safe_serialization(func):
    """Decorator to ensure safe serialization is used"""
    def wrapper(*args, **kwargs):
        config = get_security_config()
        if not config.safe_serialization_only:
            raise RuntimeError("Safe serialization is required")
        return func(*args, **kwargs)
    return wrapper