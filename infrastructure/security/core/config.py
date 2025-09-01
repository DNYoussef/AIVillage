"""Security Configuration Management.

Provides centralized configuration management for all security components
with environment-based overrides and validation.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .interfaces import SecurityLevel, AuthenticationMethod
from .exceptions import SecurityConfigurationError


@dataclass
class EncryptionConfig:
    """Encryption configuration."""

    algorithm: str = "AES-256-GCM"
    key_size: int = 32
    key_rotation_interval: int = 3600  # 1 hour
    max_key_age: int = 7200  # 2 hours
    enable_perfect_forward_secrecy: bool = True

    def __post_init__(self):
        """Validate encryption configuration."""
        if self.key_size < 16:
            raise SecurityConfigurationError("Key size must be at least 16 bytes")

        if self.key_rotation_interval <= 0:
            raise SecurityConfigurationError("Key rotation interval must be positive")


@dataclass
class AuthenticationConfig:
    """Authentication configuration."""

    methods: List[AuthenticationMethod] = field(default_factory=lambda: [AuthenticationMethod.TOKEN])
    token_expiry_hours: int = 24
    refresh_token_expiry_days: int = 30
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    require_mfa: bool = False
    mfa_methods: List[str] = field(default_factory=lambda: ["TOTP"])
    session_timeout_minutes: int = 60

    def __post_init__(self):
        """Validate authentication configuration."""
        if self.token_expiry_hours <= 0:
            raise SecurityConfigurationError("Token expiry must be positive")

        if self.max_failed_attempts <= 0:
            raise SecurityConfigurationError("Max failed attempts must be positive")


@dataclass
class AuthorizationConfig:
    """Authorization configuration."""

    enable_rbac: bool = True
    default_role: str = "user"
    admin_roles: List[str] = field(default_factory=lambda: ["admin", "super_admin"])
    permission_cache_ttl: int = 300  # 5 minutes
    enable_resource_isolation: bool = True
    tenant_isolation: bool = True

    def __post_init__(self):
        """Validate authorization configuration."""
        if not self.default_role:
            raise SecurityConfigurationError("Default role cannot be empty")


@dataclass
class ThreatDetectionConfig:
    """Threat detection configuration."""

    enable_anomaly_detection: bool = True
    enable_rate_limiting: bool = True
    max_requests_per_minute: int = 60
    ban_duration_minutes: int = 60
    trust_threshold: float = 0.3
    reputation_decay_rate: float = 0.01
    enable_ip_blocking: bool = True
    enable_geo_blocking: bool = False
    blocked_countries: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate threat detection configuration."""
        if not 0 <= self.trust_threshold <= 1:
            raise SecurityConfigurationError("Trust threshold must be between 0 and 1")


@dataclass
class AuditConfig:
    """Audit configuration."""

    enable_audit_logging: bool = True
    log_all_requests: bool = False
    log_failed_attempts: bool = True
    log_admin_actions: bool = True
    retention_days: int = 90
    enable_real_time_alerts: bool = True
    alert_threshold_critical: int = 5
    alert_threshold_high: int = 10

    def __post_init__(self):
        """Validate audit configuration."""
        if self.retention_days <= 0:
            raise SecurityConfigurationError("Retention days must be positive")


@dataclass
class NetworkSecurityConfig:
    """Network security configuration."""

    enable_tls: bool = True
    tls_version: str = "1.3"
    enable_hsts: bool = True
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["https://localhost:3000"])
    enable_csp: bool = True
    csp_policy: str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"

    def __post_init__(self):
        """Validate network security configuration."""
        if self.enable_tls and self.tls_version not in ["1.2", "1.3"]:
            raise SecurityConfigurationError("TLS version must be 1.2 or 1.3")


class SecurityConfiguration:
    """Main security configuration manager."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize security configuration."""
        self.config_dict = config_dict or {}

        # Load configuration from environment and config dict
        self._load_base_config()
        self._load_component_configs()

        # Validate overall configuration
        self._validate_configuration()

    def _load_base_config(self):
        """Load base configuration."""
        self.security_level = SecurityLevel(
            os.getenv("SECURITY_LEVEL", self.config_dict.get("security_level", "standard"))
        )

        self.debug_mode = os.getenv("SECURITY_DEBUG", "false").lower() == "true"
        self.environment = os.getenv("ENVIRONMENT", self.config_dict.get("environment", "production"))

        # Feature flags
        self.enable_authentication = self._get_bool_config("ENABLE_AUTHENTICATION", True)
        self.enable_authorization = self._get_bool_config("ENABLE_AUTHORIZATION", True)
        self.enable_encryption = self._get_bool_config("ENABLE_ENCRYPTION", True)
        self.enable_threat_detection = self._get_bool_config("ENABLE_THREAT_DETECTION", True)
        self.enable_auditing = self._get_bool_config("ENABLE_AUDITING", True)

    def _load_component_configs(self):
        """Load component-specific configurations."""
        # Encryption
        encryption_config = self.config_dict.get("encryption", {})
        self.encryption = EncryptionConfig(
            algorithm=encryption_config.get("algorithm", "AES-256-GCM"),
            key_size=int(os.getenv("ENCRYPTION_KEY_SIZE", encryption_config.get("key_size", 32))),
            key_rotation_interval=int(
                os.getenv("KEY_ROTATION_INTERVAL", encryption_config.get("key_rotation_interval", 3600))
            ),
            max_key_age=int(os.getenv("MAX_KEY_AGE", encryption_config.get("max_key_age", 7200))),
            enable_perfect_forward_secrecy=self._get_bool_config(
                "ENABLE_PFS", encryption_config.get("enable_perfect_forward_secrecy", True)
            ),
        )

        # Authentication
        auth_config = self.config_dict.get("authentication", {})
        self.authentication = AuthenticationConfig(
            token_expiry_hours=int(os.getenv("TOKEN_EXPIRY_HOURS", auth_config.get("token_expiry_hours", 24))),
            refresh_token_expiry_days=int(
                os.getenv("REFRESH_TOKEN_EXPIRY_DAYS", auth_config.get("refresh_token_expiry_days", 30))
            ),
            max_failed_attempts=int(os.getenv("MAX_FAILED_ATTEMPTS", auth_config.get("max_failed_attempts", 5))),
            lockout_duration_minutes=int(
                os.getenv("LOCKOUT_DURATION_MINUTES", auth_config.get("lockout_duration_minutes", 30))
            ),
            require_mfa=self._get_bool_config("REQUIRE_MFA", auth_config.get("require_mfa", False)),
            session_timeout_minutes=int(
                os.getenv("SESSION_TIMEOUT_MINUTES", auth_config.get("session_timeout_minutes", 60))
            ),
        )

        # Authorization
        authz_config = self.config_dict.get("authorization", {})
        self.authorization = AuthorizationConfig(
            enable_rbac=self._get_bool_config("ENABLE_RBAC", authz_config.get("enable_rbac", True)),
            default_role=os.getenv("DEFAULT_ROLE", authz_config.get("default_role", "user")),
            permission_cache_ttl=int(os.getenv("PERMISSION_CACHE_TTL", authz_config.get("permission_cache_ttl", 300))),
            enable_resource_isolation=self._get_bool_config(
                "ENABLE_RESOURCE_ISOLATION", authz_config.get("enable_resource_isolation", True)
            ),
            tenant_isolation=self._get_bool_config("TENANT_ISOLATION", authz_config.get("tenant_isolation", True)),
        )

        # Threat Detection
        threat_config = self.config_dict.get("threat_detection", {})
        self.threat_detection = ThreatDetectionConfig(
            enable_anomaly_detection=self._get_bool_config(
                "ENABLE_ANOMALY_DETECTION", threat_config.get("enable_anomaly_detection", True)
            ),
            enable_rate_limiting=self._get_bool_config(
                "ENABLE_RATE_LIMITING", threat_config.get("enable_rate_limiting", True)
            ),
            max_requests_per_minute=int(
                os.getenv("MAX_REQUESTS_PER_MINUTE", threat_config.get("max_requests_per_minute", 60))
            ),
            ban_duration_minutes=int(os.getenv("BAN_DURATION_MINUTES", threat_config.get("ban_duration_minutes", 60))),
            trust_threshold=float(os.getenv("TRUST_THRESHOLD", threat_config.get("trust_threshold", 0.3))),
        )

        # Audit
        audit_config = self.config_dict.get("audit", {})
        self.audit = AuditConfig(
            enable_audit_logging=self._get_bool_config(
                "ENABLE_AUDIT_LOGGING", audit_config.get("enable_audit_logging", True)
            ),
            log_all_requests=self._get_bool_config("LOG_ALL_REQUESTS", audit_config.get("log_all_requests", False)),
            log_failed_attempts=self._get_bool_config(
                "LOG_FAILED_ATTEMPTS", audit_config.get("log_failed_attempts", True)
            ),
            retention_days=int(os.getenv("AUDIT_RETENTION_DAYS", audit_config.get("retention_days", 90))),
        )

        # Network Security
        network_config = self.config_dict.get("network_security", {})
        self.network_security = NetworkSecurityConfig(
            enable_tls=self._get_bool_config("ENABLE_TLS", network_config.get("enable_tls", True)),
            tls_version=os.getenv("TLS_VERSION", network_config.get("tls_version", "1.3")),
            enable_hsts=self._get_bool_config("ENABLE_HSTS", network_config.get("enable_hsts", True)),
            enable_cors=self._get_bool_config("ENABLE_CORS", network_config.get("enable_cors", True)),
            enable_csp=self._get_bool_config("ENABLE_CSP", network_config.get("enable_csp", True)),
        )

    def _get_bool_config(self, env_key: str, default: bool) -> bool:
        """Get boolean configuration from environment or config dict."""
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value.lower() in ["true", "1", "yes", "on"]

        config_key = env_key.lower()
        return self.config_dict.get(config_key, default)

    def _validate_configuration(self):
        """Validate overall configuration consistency."""
        # Security level adjustments
        if self.security_level == SecurityLevel.CRITICAL:
            if not self.enable_encryption:
                raise SecurityConfigurationError("Encryption required for CRITICAL security level")

            if not self.authentication.require_mfa:
                self.authentication.require_mfa = True

            if self.authentication.token_expiry_hours > 8:
                self.authentication.token_expiry_hours = 8

        elif self.security_level == SecurityLevel.HIGH:
            if self.authentication.token_expiry_hours > 12:
                self.authentication.token_expiry_hours = 12

        elif self.security_level == SecurityLevel.NONE:
            # Disable security features for testing
            if self.environment != "development":
                raise SecurityConfigurationError("NONE security level only allowed in development")

            self.enable_authentication = False
            self.enable_authorization = False
            self.enable_encryption = False

    def get_security_level(self) -> SecurityLevel:
        """Get current security level."""
        return self.security_level

    def get_encryption_config(self) -> Dict[str, Any]:
        """Get encryption configuration."""
        return {
            "algorithm": self.encryption.algorithm,
            "key_size": self.encryption.key_size,
            "key_rotation_interval": self.encryption.key_rotation_interval,
            "max_key_age": self.encryption.max_key_age,
            "enable_perfect_forward_secrecy": self.encryption.enable_perfect_forward_secrecy,
        }

    def get_authentication_config(self) -> Dict[str, Any]:
        """Get authentication configuration."""
        return {
            "methods": [method.value for method in self.authentication.methods],
            "token_expiry_hours": self.authentication.token_expiry_hours,
            "refresh_token_expiry_days": self.authentication.refresh_token_expiry_days,
            "max_failed_attempts": self.authentication.max_failed_attempts,
            "lockout_duration_minutes": self.authentication.lockout_duration_minutes,
            "require_mfa": self.authentication.require_mfa,
            "mfa_methods": self.authentication.mfa_methods,
            "session_timeout_minutes": self.authentication.session_timeout_minutes,
        }

    def get_authorization_config(self) -> Dict[str, Any]:
        """Get authorization configuration."""
        return {
            "enable_rbac": self.authorization.enable_rbac,
            "default_role": self.authorization.default_role,
            "admin_roles": self.authorization.admin_roles,
            "permission_cache_ttl": self.authorization.permission_cache_ttl,
            "enable_resource_isolation": self.authorization.enable_resource_isolation,
            "tenant_isolation": self.authorization.tenant_isolation,
        }

    def is_feature_enabled(self, feature: str) -> bool:
        """Check if security feature is enabled."""
        feature_map = {
            "authentication": self.enable_authentication,
            "authorization": self.enable_authorization,
            "encryption": self.enable_encryption,
            "threat_detection": self.enable_threat_detection,
            "auditing": self.enable_auditing,
            "mfa": self.authentication.require_mfa,
            "rate_limiting": self.threat_detection.enable_rate_limiting,
            "anomaly_detection": self.threat_detection.enable_anomaly_detection,
            "audit_logging": self.audit.enable_audit_logging,
            "tls": self.network_security.enable_tls,
            "cors": self.network_security.enable_cors,
        }

        return feature_map.get(feature, False)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "security_level": self.security_level.value,
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "features": {
                "authentication": self.enable_authentication,
                "authorization": self.enable_authorization,
                "encryption": self.enable_encryption,
                "threat_detection": self.enable_threat_detection,
                "auditing": self.enable_auditing,
            },
            "encryption": self.get_encryption_config(),
            "authentication": self.get_authentication_config(),
            "authorization": self.get_authorization_config(),
            "threat_detection": {
                "enable_anomaly_detection": self.threat_detection.enable_anomaly_detection,
                "enable_rate_limiting": self.threat_detection.enable_rate_limiting,
                "max_requests_per_minute": self.threat_detection.max_requests_per_minute,
                "ban_duration_minutes": self.threat_detection.ban_duration_minutes,
                "trust_threshold": self.threat_detection.trust_threshold,
            },
            "audit": {
                "enable_audit_logging": self.audit.enable_audit_logging,
                "log_all_requests": self.audit.log_all_requests,
                "retention_days": self.audit.retention_days,
            },
            "network_security": {
                "enable_tls": self.network_security.enable_tls,
                "tls_version": self.network_security.tls_version,
                "enable_cors": self.network_security.enable_cors,
                "enable_csp": self.network_security.enable_csp,
            },
        }


def create_default_config(security_level: SecurityLevel = SecurityLevel.STANDARD) -> SecurityConfiguration:
    """Create default security configuration for specified level."""
    config_dict = {
        "security_level": security_level.value,
        "environment": "production",
    }

    # Adjust defaults based on security level
    if security_level == SecurityLevel.CRITICAL:
        config_dict.update(
            {
                "authentication": {
                    "require_mfa": True,
                    "token_expiry_hours": 4,
                    "max_failed_attempts": 3,
                },
                "encryption": {
                    "key_rotation_interval": 1800,  # 30 minutes
                },
                "threat_detection": {
                    "trust_threshold": 0.7,
                    "ban_duration_minutes": 120,
                },
            }
        )

    elif security_level == SecurityLevel.HIGH:
        config_dict.update(
            {
                "authentication": {
                    "token_expiry_hours": 8,
                    "max_failed_attempts": 3,
                },
                "encryption": {
                    "key_rotation_interval": 3600,  # 1 hour
                },
            }
        )

    elif security_level == SecurityLevel.NONE:
        config_dict.update(
            {
                "environment": "development",
                "enable_authentication": False,
                "enable_authorization": False,
                "enable_encryption": False,
            }
        )

    return SecurityConfiguration(config_dict)
