"""Unified service error handling for AIVillage services.

This module provides standardized error handling for FastAPI services
with consistent logging, error categorization, and API responses.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import logging
import traceback
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.error_handling import (
    AIVillageException,
    ErrorCategory,
    ErrorSeverity,
)

# Service-specific error codes
SERVICE_ERROR_CODES = {
    ErrorCategory.NETWORK: {
        ErrorSeverity.LOW: 503,
        ErrorSeverity.MEDIUM: 503,
        ErrorSeverity.HIGH: 503,
        ErrorSeverity.CRITICAL: 503,
    },
    ErrorCategory.VALIDATION: {
        ErrorSeverity.LOW: 400,
        ErrorSeverity.MEDIUM: 400,
        ErrorSeverity.HIGH: 422,
        ErrorSeverity.CRITICAL: 422,
    },
    ErrorCategory.SECURITY: {
        ErrorSeverity.LOW: 401,
        ErrorSeverity.MEDIUM: 401,
        ErrorSeverity.HIGH: 401,
        ErrorSeverity.CRITICAL: 401,
    },
    ErrorCategory.EXTERNAL_SERVICE: {
        ErrorSeverity.LOW: 503,
        ErrorSeverity.MEDIUM: 503,
        ErrorSeverity.HIGH: 503,
        ErrorSeverity.CRITICAL: 503,
    },
    ErrorCategory.SYSTEM: {
        ErrorSeverity.LOW: 500,
        ErrorSeverity.MEDIUM: 500,
        ErrorSeverity.HIGH: 500,
        ErrorSeverity.CRITICAL: 500,
    },
    ErrorCategory.DATABASE: {
        ErrorSeverity.LOW: 500,
        ErrorSeverity.MEDIUM: 500,
        ErrorSeverity.HIGH: 500,
        ErrorSeverity.CRITICAL: 500,
    },
    ErrorCategory.BUSINESS_LOGIC: {
        ErrorSeverity.LOW: 400,
        ErrorSeverity.MEDIUM: 400,
        ErrorSeverity.HIGH: 422,
        ErrorSeverity.CRITICAL: 422,
    },
    ErrorCategory.UNKNOWN: {
        ErrorSeverity.LOW: 500,
        ErrorSeverity.MEDIUM: 500,
        ErrorSeverity.HIGH: 500,
        ErrorSeverity.CRITICAL: 500,
    },
    ErrorCategory.RESOURCE: {
        ErrorSeverity.LOW: 404,
        ErrorSeverity.MEDIUM: 404,
        ErrorSeverity.HIGH: 404,
        ErrorSeverity.CRITICAL: 404,
    },
    ErrorCategory.RATE_LIMIT: {
        ErrorSeverity.LOW: 429,
        ErrorSeverity.MEDIUM: 429,
        ErrorSeverity.HIGH: 429,
        ErrorSeverity.CRITICAL: 429,
    },
}


class ServiceErrorHandler:
    """Centralized error handler for FastAPI services."""

    def __init__(self, service_name: str, logger: logging.Logger | None = None):
        self.service_name = service_name
        self.logger = logger or logging.getLogger(service_name)

    def create_error_response(
        self,
        exception: Exception,
        request: Request | None = None,
        include_stacktrace: bool = False,
    ) -> dict[str, Any]:
        """Create standardized error response."""
        if isinstance(exception, AIVillageException):
            error_info = exception.to_dict()
            status_code = SERVICE_ERROR_CODES.get(
                exception.category, SERVICE_ERROR_CODES[ErrorCategory.UNKNOWN]
            )[exception.severity]
        else:
            # Convert generic exception to AIVillageException
            av_exception = AIVillageException(
                message=str(exception),
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.MEDIUM,
                operation="service_operation",
                context={"original_error": str(exception)},
            )
            error_info = av_exception.to_dict()
            status_code = 500

        response = {
            "error": {
                "type": error_info["type"],
                "message": error_info["message"],
                "code": error_info["code"],
                "category": error_info["category"],
                "severity": error_info["severity"],
                "timestamp": error_info["timestamp"],
                "service": self.service_name,
            }
        }

        if include_stacktrace and request:
            response["error"]["stacktrace"] = traceback.format_exc()

        if request:
            response["error"]["request_id"] = getattr(request.state, "request_id", None)
            response["error"]["path"] = str(request.url.path)

        return response

    def http_exception_handler(self, request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for FastAPI."""
        # Handle FastAPI validation errors
        if isinstance(exc, RequestValidationError):
            return self._handle_validation_error(request, exc)

        # Log the error with context
        if isinstance(exc, AIVillageException):
            self.logger.error(
                f"Service error: {exc.message}",
                extra={
                    "error": exc.to_dict(),
                    "service": self.service_name,
                    "request": {
                        "method": request.method,
                        "path": str(request.url.path),
                        "query": str(request.url.query),
                    },
                },
            )
        else:
            self.logger.error(
                f"Unhandled exception: {exc!s}",
                extra={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "service": self.service_name,
                    "request": {
                        "method": request.method,
                        "path": str(request.url.path),
                        "query": str(request.url.query),
                    },
                },
                exc_info=True,
            )

        # Create response
        error_response = self.create_error_response(exc, request)
        status_code = 500

        if isinstance(exc, AIVillageException):
            status_code = SERVICE_ERROR_CODES.get(
                exc.category, SERVICE_ERROR_CODES[ErrorCategory.UNKNOWN]
            )[exc.severity]
        elif isinstance(exc, HTTPException):
            status_code = exc.status_code

        return JSONResponse(
            status_code=status_code,
            content=error_response,
        )

    def _handle_validation_error(
        self, request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle FastAPI validation errors with standardized format."""
        # Create standardized validation error
        validation_errors = []
        for error in exc.errors():
            field_path = " -> ".join(str(loc) for loc in error["loc"])
            validation_errors.append(
                {
                    "field": field_path,
                    "message": error["msg"],
                    "type": error["type"],
                    "input": error.get("input"),
                }
            )

        # Create AIVillageException for consistent logging
        av_exception = validation_error(
            f"Validation failed for {len(validation_errors)} field(s)",
            details={
                "validation_errors": validation_errors,
                "request_path": str(request.url.path),
                "request_method": request.method,
            },
        )

        # Log validation error
        self.logger.warning(
            f"Validation error in {self.service_name}",
            extra={
                "error": av_exception.to_dict(),
                "validation_errors": validation_errors,
                "service": self.service_name,
                "request": {
                    "method": request.method,
                    "path": str(request.url.path),
                    "query": str(request.url.query),
                },
            },
        )

        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "type": "AIVillageException",
                    "message": av_exception.message,
                    "category": "VALIDATION",
                    "severity": "MEDIUM",
                    "code": "VALIDATION_MEDIUM",
                    "service": self.service_name,
                    "timestamp": av_exception.timestamp,
                    "context": av_exception.context,
                    "validation_errors": validation_errors,
                }
            },
        )

    def service_error_handler(self, operation: str):
        """Decorator for service operations with standardized error handling."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    # Convert to AIVillageException if needed
                    if not isinstance(exc, AIVillageException):
                        exc = AIVillageException(
                            message=str(exc),
                            category=ErrorCategory.SYSTEM,
                            severity=ErrorSeverity.MEDIUM,
                            operation=operation,
                            context={"original_error": str(exc)},
                        )

                    # Log and re-raise
                    self.logger.error(
                        f"Service operation failed: {exc.message}",
                        extra={
                            "error": exc.to_dict(),
                            "operation": operation,
                            "service": self.service_name,
                        },
                    )
                    raise exc

            return wrapper

        return decorator


# Global service error handler instances
gateway_error_handler = ServiceErrorHandler("gateway")
twin_error_handler = ServiceErrorHandler("twin")


def create_service_error(
    message: str,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    operation: str = "service_operation",
    details: dict[str, Any] | None = None,
) -> AIVillageException:
    """Create a standardized service error."""
    return AIVillageException(
        message=message,
        category=category,
        severity=severity,
        operation=operation,
        context=details or {},
    )


# Common service error factories
def validation_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create validation error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.MEDIUM,
        operation="validation",
        details=details,
    )


def network_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create network error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.NETWORK,
        severity=ErrorSeverity.HIGH,
        operation="network_request",
        details=details,
    )


def database_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create database error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.DATABASE,
        severity=ErrorSeverity.HIGH,
        operation="database_operation",
        details=details,
    )


def security_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create security error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.SECURITY,
        severity=ErrorSeverity.HIGH,
        operation="security_check",
        details=details,
    )


def external_service_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create external service error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.EXTERNAL_SERVICE,
        severity=ErrorSeverity.HIGH,
        operation="external_service",
        details=details,
    )


def resource_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create resource error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.RESOURCE,
        severity=ErrorSeverity.MEDIUM,
        operation="resource_access",
        details=details,
    )


def rate_limit_error(
    message: str, details: dict[str, Any] | None = None
) -> AIVillageException:
    """Create rate limit error."""
    return create_service_error(
        message=message,
        category=ErrorCategory.RATE_LIMIT,
        severity=ErrorSeverity.MEDIUM,
        operation="rate_limiting",
        details=details,
    )
