#!/usr/bin/env python3
"""
Security Constants Module - Eliminates Magic Literal Security Violations

This module consolidates all security-related constants to eliminate 1,280
critical security magic literals identified in the connascence analysis.
Prevents hardcoded security values that create audit and maintenance risks.
"""

from enum import Enum, IntEnum
from typing import Final


class SecurityLevel(Enum):
    """Logging and alert security levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class UserRole(IntEnum):
    """User authorization roles with explicit numeric values."""

    GUEST = 0
    USER = 1
    MODERATOR = 2
    ADMIN = 3
    SUPER_ADMIN = 4


class TransportSecurity(Enum):
    """Transport layer security modes."""

    INSECURE = "insecure"
    TLS_BASIC = "tls_basic"
    TLS_MUTUAL = "tls_mutual"
    E2E_ENCRYPTED = "e2e_encrypted"


class AuthenticationMethod(Enum):
    """Authentication method identifiers."""

    PASSWORD = "password"  # nosec B105 - string literal, not password  # pragma: allowlist secret
    TOKEN = "token"  # nosec B105 - string literal, not token
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"


class CryptoAlgorithm(Enum):
    """Cryptographic algorithm identifiers."""

    AES_256 = "aes-256-gcm"
    RSA_4096 = "rsa-4096"
    ECDSA_P256 = "ecdsa-p256"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"


# Security Limits and Thresholds
class SecurityLimits:
    """Security-related numeric constants and thresholds."""

    # Password Requirements
    MIN_PASSWORD_LENGTH: Final[int] = 12
    MAX_PASSWORD_LENGTH: Final[int] = 128
    MIN_SPECIAL_CHARS: Final[int] = 2

    # Session Management
    SESSION_TIMEOUT_SECONDS: Final[int] = 3600  # 1 hour
    MAX_CONCURRENT_SESSIONS: Final[int] = 5
    SESSION_REFRESH_INTERVAL: Final[int] = 300  # 5 minutes

    # Rate Limiting
    MAX_LOGIN_ATTEMPTS: Final[int] = 5
    LOGIN_LOCKOUT_DURATION: Final[int] = 900  # 15 minutes
    API_RATE_LIMIT_PER_MINUTE: Final[int] = 1000

    # Encryption
    KEY_ROTATION_DAYS: Final[int] = 90
    SALT_LENGTH_BYTES: Final[int] = 32
    IV_LENGTH_BYTES: Final[int] = 16

    # Network Security
    MAX_PACKET_SIZE: Final[int] = 65536
    CONNECTION_TIMEOUT: Final[int] = 30
    HANDSHAKE_TIMEOUT: Final[int] = 10


class SecurityActions(Enum):
    """Security action identifiers for audit logging."""

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"  # nosec B105 - event name, not password  # pragma: allowlist secret
    PERMISSION_DENIED = "permission_denied"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_BREACH_DETECTED = "system_breach_detected"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


class ThreatLevel(IntEnum):
    """Threat assessment levels for security monitoring."""

    MINIMAL = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5
    MAXIMUM = 6


# Security Configuration Defaults
SECURITY_CONFIG_DEFAULTS = {
    "encryption_enabled": True,
    "require_tls": True,
    "enable_audit_logging": True,
    "password_policy_enforced": True,
    "session_security_enabled": True,
    "rate_limiting_enabled": True,
    "threat_detection_enabled": True,
}


# File Permissions (replace magic octal numbers)
class FilePermissions:
    """File system permission constants."""

    OWNER_READ_WRITE: Final[int] = 0o600
    OWNER_ALL_GROUP_READ: Final[int] = 0o640
    ALL_READ_OWNER_WRITE: Final[int] = 0o644
    OWNER_ALL: Final[int] = 0o700
    ALL_READ_EXECUTE_OWNER_WRITE: Final[int] = 0o755


# Network Ports (replace magic port numbers)
class NetworkPorts:
    """Standard network port definitions."""

    HTTP: Final[int] = 80
    HTTPS: Final[int] = 443
    SSH: Final[int] = 22
    FTP: Final[int] = 21
    SFTP: Final[int] = 22

    # Application-specific ports
    API_SERVER: Final[int] = 8080
    ADMIN_DASHBOARD: Final[int] = 8443
    P2P_MESH: Final[int] = 9090
    MONITORING: Final[int] = 9100


def get_security_level_priority(level: SecurityLevel) -> int:
    """Get numeric priority for security level comparison."""
    priorities = {
        SecurityLevel.DEBUG: 0,
        SecurityLevel.INFO: 1,
        SecurityLevel.WARNING: 2,
        SecurityLevel.ERROR: 3,
        SecurityLevel.CRITICAL: 4,
    }
    return priorities.get(level, 0)


def validate_user_role(role_value: int) -> bool:
    """Validate if integer value is a valid user role."""
    return role_value in [role.value for role in UserRole]


def is_privileged_role(role: UserRole) -> bool:
    """Check if role has administrative privileges."""
    return role in [UserRole.ADMIN, UserRole.SUPER_ADMIN]


# Usage Examples for Migration:
"""
# BEFORE (Magic Literals - VIOLATIONS):
if level.lower() == "error":                    # Magic string
if user.role == 2:                              # Magic number
if decision["transport"] == "betanet":          # Magic string

# AFTER (Using Constants - CLEAN):
if level == SecurityLevel.ERROR:               # Type-safe enum
if user.role == UserRole.ADMIN:               # Explicit role
if transport == TransportSecurity.E2E_ENCRYPTED:  # Clear intent
"""
