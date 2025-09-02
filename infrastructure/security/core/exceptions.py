"""Security Exceptions.

Defines security-related exceptions for the modular security architecture.
"""

from typing import Any


class SecurityError(Exception):
    """Base security error."""

    def __init__(self, message: str, error_code: str | None = None, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.error_code = error_code
        self.metadata = metadata or {}


class AuthenticationError(SecurityError):
    """Authentication-related error."""

    def __init__(
        self,
        message: str = "Authentication failed",
        error_code: str = "AUTH_FAILED",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)


class AuthorizationError(SecurityError):
    """Authorization-related error."""

    def __init__(
        self,
        message: str = "Access denied",
        error_code: str = "ACCESS_DENIED",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)


class CryptographicError(SecurityError):
    """Cryptographic operation error."""

    def __init__(
        self,
        message: str = "Cryptographic operation failed",
        error_code: str = "CRYPTO_ERROR",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)


class SecurityConfigurationError(SecurityError):
    """Security configuration error."""

    def __init__(
        self,
        message: str = "Security configuration error",
        error_code: str = "CONFIG_ERROR",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)


class SessionError(SecurityError):
    """Session management error."""

    def __init__(
        self,
        message: str = "Session error",
        error_code: str = "SESSION_ERROR",
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)


class TokenError(SecurityError):
    """Token-related error."""

    def __init__(
        self, message: str = "Token error", error_code: str = "TOKEN_ERROR", metadata: dict[str, Any] | None = None
    ):
        super().__init__(message, error_code, metadata)


class ThreatDetectedError(SecurityError):
    """Threat detection error."""

    def __init__(
        self,
        message: str = "Security threat detected",
        error_code: str = "THREAT_DETECTED",
        threat_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)
        self.threat_type = threat_type


class RateLimitError(SecurityError):
    """Rate limiting error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        error_code: str = "RATE_LIMIT",
        retry_after: int | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)
        self.retry_after = retry_after


class MFARequiredError(AuthenticationError):
    """Multi-factor authentication required."""

    def __init__(
        self,
        message: str = "Multi-factor authentication required",
        error_code: str = "MFA_REQUIRED",
        available_methods: list | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)
        self.available_methods = available_methods or []


class SecurityValidationError(SecurityError):
    """Security validation error."""

    def __init__(
        self,
        message: str = "Security validation failed",
        error_code: str = "VALIDATION_ERROR",
        validation_errors: list | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(message, error_code, metadata)
        self.validation_errors = validation_errors or []
