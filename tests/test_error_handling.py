"""Tests for the unified error handling system."""

import asyncio

import pytest

from core.error_handling import (
    AIVillageException,
    ErrorCategory,
    ErrorContextManager,
    ErrorSeverity,
    get_component_logger,
    migrate_from_legacy_exception,
    with_error_handling,
)


class TestAIVillageException:
    """Test cases for AIVillageException."""

    def test_exception_creation(self):
        """Test basic exception creation."""
        exc = AIVillageException(
            "Test error",
            component="test",
            operation="test_operation",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            context={"key": "value"},
        )

        assert str(exc) == "Test error"
        assert exc.component == "test"
        assert exc.operation == "test_operation"
        assert exc.severity == ErrorSeverity.HIGH
        assert exc.category == ErrorCategory.BUSINESS_LOGIC
        assert exc.context == {"key": "value"}

    def test_exception_with_cause(self):
        """Test exception with underlying cause."""
        original_error = ValueError("Original error")
        exc = AIVillageException(
            "Wrapped error",
            component="test",
            operation="test_operation",
            cause=original_error,
        )

        assert exc.__cause__ is original_error

    def test_exception_serialization(self):
        """Test exception serialization for logging."""
        exc = AIVillageException(
            "Test error",
            component="test",
            operation="test_operation",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
        )

        serialized = exc.to_dict()
        assert serialized["message"] == "Test error"
        assert serialized["component"] == "test"
        assert serialized["operation"] == "test_operation"
        assert serialized["severity"] == "MEDIUM"
        assert serialized["category"] == "NETWORK"


class TestErrorContextManager:
    """Test cases for ErrorContextManager."""

    def test_context_manager_success(self):
        """Test context manager with successful operation."""
        with ErrorContextManager("test", "operation") as ctx:
            assert ctx.context.component == "test"
            assert ctx.context.operation == "operation"

    def test_context_manager_exception(self):
        """Test context manager with exception."""
        with pytest.raises(ValueError):
            with ErrorContextManager("test", "operation", context={"test": True}):
                raise ValueError("Test error")

    def test_context_manager_custom_category(self):
        """Test context manager with custom error category."""
        with pytest.raises(RuntimeError):
            with ErrorContextManager(
                "test",
                "operation",
                category=ErrorCategory.DATABASE,
                context={"query": "SELECT * FROM test"},
            ):
                raise RuntimeError("Database error")


class TestWithErrorHandlingDecorator:
    """Test cases for with_error_handling decorator."""

    def test_decorator_success(self):
        """Test decorator with successful function."""

        @with_error_handling(component="test", operation="success_operation")
        def success_function():
            return "success"

        result = success_function()
        assert result == "success"

    def test_decorator_exception(self):
        """Test decorator with exception."""

        @with_error_handling(
            component="test", operation="error_operation", severity=ErrorSeverity.HIGH
        )
        def error_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            error_function()

    def test_decorator_async(self):
        """Test decorator with async function."""

        @with_error_handling(component="test", operation="async_success")
        async def async_success():
            return "async success"

        result = asyncio.run(async_success())
        assert result == "async success"

    def test_decorator_async_exception(self):
        """Test decorator with async exception."""

        @with_error_handling(component="test", operation="async_error")
        async def async_error():
            raise ValueError("Async error")

        with pytest.raises(ValueError):
            asyncio.run(async_error())

    def test_decorator_with_retries(self):
        """Test decorator with retry functionality."""
        attempts = 0

        @with_error_handling(
            component="test", operation="retry_operation", max_retries=2
        )
        def retry_function():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ValueError("Need retry")
            return "success"

        result = retry_function()
        assert result == "success"
        assert attempts == 2

    def test_decorator_with_retries_exhausted(self):
        """Test decorator when retries are exhausted."""
        attempts = 0

        @with_error_handling(
            component="test", operation="retry_exhausted", max_retries=1
        )
        def retry_exhausted():
            nonlocal attempts
            attempts += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            retry_exhausted()
        assert attempts == 2  # Initial + 1 retry


class TestMigrateFromLegacyException:
    """Test cases for legacy exception migration."""

    def test_migration_from_legacy(self):
        """Test migrating from legacy exception."""
        legacy_exc = ValueError("Legacy error")

        new_exc = migrate_from_legacy_exception(
            legacy_exc, component="test", operation="migration"
        )

        assert isinstance(new_exc, AIVillageException)
        assert str(new_exc) == "Legacy error"
        assert new_exc.component == "test"
        assert new_exc.operation == "migration"
        assert new_exc.__cause__ is legacy_exc

    def test_migration_with_context(self):
        """Test migration with additional context."""
        legacy_exc = RuntimeError("Runtime error")
        context = {"user_id": 123, "request_id": "abc123"}

        new_exc = migrate_from_legacy_exception(
            legacy_exc,
            component="test",
            operation="migration",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            context=context,
        )

        assert new_exc.severity == ErrorSeverity.HIGH
        assert new_exc.category == ErrorCategory.SYSTEM
        assert new_exc.context["user_id"] == 123
        assert new_exc.context["request_id"] == "abc123"
        assert new_exc.context["legacy_exception_type"] == "RuntimeError"


class TestGetComponentLogger:
    """Test cases for get_component_logger."""

    def test_logger_creation(self):
        """Test logger creation."""
        logger = get_component_logger("test_component")
        assert logger.name == "aivillage.test_component"

    def test_logger_with_context(self):
        """Test logger with additional context."""
        extra = {"user_id": 123, "session_id": "abc123"}
        logger = get_component_logger("test_component", extra=extra)
        assert logger.name == "aivillage.test_component"
