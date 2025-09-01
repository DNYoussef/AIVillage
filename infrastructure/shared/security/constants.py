"""Security-related constants for AIVillage.

This module centralizes all security-related magic literals to eliminate
connascence of meaning and ensure consistent security configurations
across the system.
"""

from enum import Enum
from typing import Final

# JWT Configuration
JWT_ALGORITHM: Final[str] = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS: Final[int] = 7
JWT_SECRET_KEY_LENGTH: Final[int] = 32

# Password Security
PASSWORD_MIN_LENGTH: Final[int] = 8
PASSWORD_MAX_LENGTH: Final[int] = 128
PASSWORD_HASH_ROUNDS: Final[int] = 12
PASSWORD_RESET_TOKEN_EXPIRE_HOURS: Final[int] = 1

# Session Management
SESSION_TIMEOUT_MINUTES: Final[int] = 60
SESSION_CLEANUP_INTERVAL_HOURS: Final[int] = 24
MAX_CONCURRENT_SESSIONS: Final[int] = 5
SESSION_TOKEN_LENGTH: Final[int] = 64

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 60
RATE_LIMIT_BURST_SIZE: Final[int] = 100
RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
AUTH_FAILURE_LOCKOUT_ATTEMPTS: Final[int] = 5
AUTH_FAILURE_LOCKOUT_DURATION_MINUTES: Final[int] = 15

# Encryption
AES_KEY_SIZE_BITS: Final[int] = 256
RSA_KEY_SIZE_BITS: Final[int] = 2048
SALT_LENGTH_BYTES: Final[int] = 32
IV_LENGTH_BYTES: Final[int] = 16
PBKDF2_ITERATIONS: Final[int] = 100_000

# Digital Twin Security
DIGITAL_TWIN_ENCRYPTION_ALGORITHM: Final[str] = "AES-256-GCM"
DIGITAL_TWIN_KEY_ROTATION_DAYS: Final[int] = 30
DIGITAL_TWIN_BACKUP_ENCRYPTION: Final[bool] = True

# Multi-tenant Security
TENANT_ISOLATION_LEVEL: Final[str] = "strict"
TENANT_DATA_ENCRYPTION: Final[bool] = True
CROSS_TENANT_ACCESS_DENIED: Final[bool] = True
TENANT_AUDIT_LOG_RETENTION_DAYS: Final[int] = 365

# API Security
API_KEY_LENGTH: Final[int] = 64
API_RATE_LIMIT_PER_HOUR: Final[int] = 1000
API_SIGNATURE_ALGORITHM: Final[str] = "HMAC-SHA256"
API_TIMESTAMP_TOLERANCE_SECONDS: Final[int] = 300

# File Upload Security
MAX_FILE_SIZE_MB: Final[int] = 100
ALLOWED_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".txt", ".json", ".yaml", ".yml", ".csv", ".py", ".md")
QUARANTINE_SCAN_TIMEOUT_SECONDS: Final[int] = 30
FILE_HASH_ALGORITHM: Final[str] = "sha256"

# Audit and Logging
AUDIT_LOG_RETENTION_DAYS: Final[int] = 730  # 2 years
SECURITY_EVENT_LOG_LEVEL: Final[str] = "WARNING"
FAILED_LOGIN_LOG_THRESHOLD: Final[int] = 3
SENSITIVE_DATA_MASK_CHAR: Final[str] = "*"

# Network Security
TLS_MIN_VERSION: Final[str] = "1.2"
CERTIFICATE_VALIDATION: Final[bool] = True
SECURITY_HEADERS_ENABLED: Final[bool] = True
CORS_ALLOW_CREDENTIALS: Final[bool] = False
CORS_MAX_AGE_SECONDS: Final[int] = 86400


class SecurityLevel(Enum):
    """Security levels for different operations."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"  # nosec B105 - classification level constant, not password


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "AES-256-GCM"
    AES_256_CBC = "AES-256-CBC"
    CHACHA20_POLY1305 = "ChaCha20-Poly1305"
    RSA_OAEP = "RSA-OAEP"


class HashAlgorithm(Enum):
    """Supported hash algorithms."""

    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"
    ARGON2 = "argon2"


class AuthenticationMethod(Enum):
    """Authentication methods."""

    PASSWORD = "password"  # nosec B105 - field name constant, not password
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"


class PermissionScope(Enum):
    """Permission scopes for access control."""

    GLOBAL = "global"
    TENANT = "tenant"
    PROJECT = "project"
    RESOURCE = "resource"
    PERSONAL = "personal"


# Security Headers
SECURITY_HEADERS: Final[dict[str, str]] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


# Error Messages (Security-focused)
class SecurityMessages:
    """Standardized security-related messages."""

    ACCESS_DENIED: Final[str] = "Access denied"
    AUTHENTICATION_REQUIRED: Final[str] = "Authentication required"
    INVALID_CREDENTIALS: Final[str] = "Invalid credentials"
    SESSION_EXPIRED: Final[str] = "Session has expired"
    PERMISSION_DENIED: Final[str] = "Insufficient permissions"
    RATE_LIMIT_EXCEEDED: Final[str] = "Rate limit exceeded"
    INVALID_TOKEN: Final[str] = "Invalid or expired token"
    ACCOUNT_LOCKED: Final[str] = "Account temporarily locked"
    SECURITY_VIOLATION: Final[str] = "Security policy violation detected"
    ENCRYPTION_FAILED: Final[str] = "Encryption operation failed"
    DECRYPTION_FAILED: Final[str] = "Decryption operation failed"
    SIGNATURE_INVALID: Final[str] = "Digital signature verification failed"


# Compliance and Standards
class ComplianceStandards:
    """Compliance-related constants."""

    GDPR_DATA_RETENTION_DAYS: Final[int] = 1095  # 3 years
    PCI_DSS_LOG_RETENTION_MONTHS: Final[int] = 12
    SOX_AUDIT_TRAIL_YEARS: Final[int] = 7
    HIPAA_ACCESS_LOG_YEARS: Final[int] = 6

    # Data classification
    PII_DETECTION_ENABLED: Final[bool] = True
    PHI_ENCRYPTION_REQUIRED: Final[bool] = True
    FINANCIAL_DATA_ISOLATION: Final[bool] = True


# Zero-Knowledge Proof Constants
ZKP_PROOF_SIZE_BYTES: Final[int] = 256
ZKP_VERIFICATION_TIMEOUT_MS: Final[int] = 1000
ZKP_CHALLENGE_SIZE_BITS: Final[int] = 256
ZKP_COMMITMENT_SCHEME: Final[str] = "Pedersen"

# P2P Security
P2P_HANDSHAKE_TIMEOUT_SECONDS: Final[int] = 30
P2P_KEY_EXCHANGE_ALGORITHM: Final[str] = "ECDH-P256"
P2P_CIPHER_SUITE: Final[str] = "TLS_AES_256_GCM_SHA384"
P2P_CERTIFICATE_LIFETIME_DAYS: Final[int] = 90
