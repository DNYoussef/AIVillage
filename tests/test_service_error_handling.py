"""
Tests for service error handling implementation.

This module tests the unified error handling system for FastAPI services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.error_handling import (
    AIVillageException,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
)
from services.core.service_error_handler import (
    ServiceErrorHandler,
    create_service_error,
    network_error,
    rate_limit_error,
    resource_error,
    validation_error,
)

# Import test fixtures


class TestServiceErrorHandler:
    """Test the ServiceErrorHandler class."""

    def test_create_error_response_with_aivillage_exception(self):
        """Test creating error response from AIVillageException."""
        handler = ServiceErrorHandler("test-service")

        exception = AIVillageException(
            message="Test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.INFO,
            context=ErrorContext(
                component="test-service",
                operation="test_operation",
                details={"field": "test"},
            ),
        )

        response = handler.create_error_response(exception)

        assert response["error"]["type"] == "AIVillageException"
        assert response["error"]["message"] == "Test error"
        assert response["error"]["category"] == "VALIDATION"
        assert response["error"]["severity"] == "INFO"
        assert response["error"]["service"] == "test-service"
        assert "timestamp" in response["error"]
        assert "code" in response["error"]

    def test_create_error_response_with_generic_exception(self):
        """Test creating error response from generic exception."""
        handler = ServiceErrorHandler("test-service")

        exception = ValueError("Generic error")
        response = handler.create_error_response(exception)

        assert response["error"]["type"] == "AIVillageException"
        assert response["error"]["message"] == "Generic error"
        assert response["error"]["category"] == "CONFIGURATION"
        assert response["error"]["severity"] == "INFO"

    def test_create_error_response_with_request_context(self):
        """Test creating error response with request context."""
        handler = ServiceErrorHandler("test-service")

        exception = AIVillageException(
            message="Test error",
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            context=ErrorContext(
                component="test-service", operation="test_operation", details={}
            ),
        )

        # Mock request
        mock_request = MagicMock()
        mock_request.url.path = "/test/path"
        mock_request.state.request_id = "test-123"

        response = handler.create_error_response(
            exception, request=mock_request, include_stacktrace=True
        )

        assert response["error"]["path"] == "/test/path"
        assert response["error"]["request_id"] == "test-123"
        assert "stacktrace" in response["error"]


class TestErrorFactories:
    """Test the error factory functions."""

    def test_create_service_error(self):
        """Test create_service_error factory."""
        error = create_service_error(
            message="Test service error",
            category=ErrorCategory.ACCESS,
            severity=ErrorSeverity.ERROR,
            operation="login",
            details={"username": "test"},
        )

        assert isinstance(error, AIVillageException)
        assert error.message == "Test service error"
        assert error.category == ErrorCategory.ACCESS
        assert error.severity == ErrorSeverity.ERROR
        assert error.context.operation == "login"
        assert error.context.details["username"] == "test"

    def test_validation_error(self):
        """Test validation_error factory."""
        error = validation_error("Invalid input", {"field": "email"})

        assert isinstance(error, AIVillageException)
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.INFO
        assert error.context.operation == "validation"
        assert error.context.details["field"] == "email"

    def test_network_error(self):
        """Test network_error factory."""
        error = network_error("Connection failed", {"url": "http://test.com"})

        assert isinstance(error, AIVillageException)
        assert error.category == ErrorCategory.NETWORK
        assert error.severity == ErrorSeverity.ERROR
        assert error.context.operation == "network_request"

    def test_resource_error(self):
        """Test resource_error factory."""
        error = resource_error("Resource not found", {"id": "123"})

        assert isinstance(error, AIVillageException)
        assert error.category == ErrorCategory.ACCESS
        assert error.severity == ErrorSeverity.INFO

    def test_rate_limit_error(self):
        """Test rate_limit_error factory."""
        error = rate_limit_error("Too many requests", {"limit": 100})

        assert isinstance(error, AIVillageException)
        assert error.category == ErrorCategory.TIMEOUT
        assert error.severity == ErrorSeverity.INFO


class TestServiceIntegration:
    """Test service integration with FastAPI."""

    def test_gateway_service_error_handling(self):
        """Test gateway service error handling."""
        from services.gateway.app import app

        client = TestClient(app)

        # Test rate limit error
        with patch("services.gateway.app.rl_cache") as mock_cache:
            mock_cache.get.return_value = 101  # Exceed limit
            response = client.post("/v1/chat", json={"message": "test"})

            assert response.status_code == 429
            assert response.json()["error"]["category"] == "RATE_LIMIT"
            assert "Rate limit exceeded" in response.json()["error"]["message"]

    def test_twin_service_validation_error(self):
        """Test twin service validation error."""
        from services.twin.app import app

        client = TestClient(app)

        # Test empty message validation
        response = client.post(
            "/v1/chat", json={"message": "", "conversation_id": "test"}
        )

        assert response.status_code == 422
        # Check standardized error format
        assert "error" in response.json()
        error = response.json()["error"]
        assert error["type"] == "AIVillageException"
        assert error["category"] == "VALIDATION"
        assert error["severity"] == "INFO"
        assert "validation_errors" in error

    def test_twin_service_explain_validation(self):
        """Test twin service explain endpoint validation."""
        from services.twin.app import app

        client = TestClient(app)

        # Test same source and destination
        response = client.post("/explain", json={"src": "node1", "dst": "node1"})

        assert response.status_code == 422
        # Check standardized error format
        assert "error" in response.json()
        error = response.json()["error"]
        assert error["type"] == "AIVillageException"
        assert error["category"] == "VALIDATION"

    def test_twin_service_explain_resource_error(self):
        """Test twin service explain endpoint resource error."""
        from services.twin.app import app

        client = TestClient(app)

        # Test non-existent path - FastAPI will return 422 for missing required fields
        response = client.post("/explain", json={"src": "nonexistent", "dst": "target"})
        assert response.status_code == 200  # Twin service returns 200 with found: false

    def test_twin_service_upload_validation(self):
        """Test twin service upload endpoint validation."""
        from services.twin.app import app

        client = TestClient(app)

        # Test empty filename - FastAPI will handle this with 422
        response = client.post("/v1/upload", files={"file": ("", b"content")})

        assert response.status_code == 422
        # Check standardized error format
        assert "error" in response.json()
        error = response.json()["error"]
        assert error["type"] == "AIVillageException"
        assert error["category"] == "VALIDATION"

    def test_twin_service_upload_size_limit(self):
        """Test twin service upload size limit."""
        from services.twin.app import app

        client = TestClient(app)

        # Test file size limit - FastAPI will handle this with 413
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        response = client.post(
            "/v1/upload", files={"file": ("test.txt", large_content)}
        )

        assert response.status_code == 413  # Request Entity Too Large


class TestErrorResponseFormat:
    """Test error response format consistency."""

    def test_error_response_structure(self):
        """Test that all error responses have consistent structure."""
        handler = ServiceErrorHandler("test")

        exception = AIVillageException(
            message="Test error",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
        )

        response = handler.create_error_response(exception)

        # Check required fields
        assert "error" in response
        error = response["error"]

        required_fields = {
            "type",
            "message",
            "code",
            "category",
            "severity",
            "timestamp",
            "service",
        }
        assert required_fields.issubset(error.keys())

        # Check field types
        assert isinstance(error["type"], str)
        assert isinstance(error["message"], str)
        assert isinstance(error["code"], str)
        assert isinstance(error["category"], str)
        assert isinstance(error["severity"], str)
        assert isinstance(error["timestamp"], str)
        assert isinstance(error["service"], str)


class TestServiceErrorHandlingWithFixtures:
    """Test error handling using mock services."""

    @pytest.mark.asyncio
    async def test_chat_service_error_handling(self, mock_chat_service):
        """Test error handling in chat service."""
        # Make the service raise an exception
        from core.error_handling import (
            AIVillageException,
            ErrorCategory,
            ErrorContext,
            ErrorSeverity,
        )

        test_exception = AIVillageException(
            message="Chat processing failed",
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.ERROR,
            context=ErrorContext(
                component="chat-service", operation="process_chat", details={}
            ),
        )

        mock_chat_service.process_chat.side_effect = test_exception

        # Test that the exception propagates correctly
        with pytest.raises(AIVillageException) as exc_info:
            from services.core.interfaces import ChatRequest

            request = ChatRequest(message="test", conversation_id="test-123")
            await mock_chat_service.process_chat(request)

        assert exc_info.value.message == "Chat processing failed"
        assert exc_info.value.category == ErrorCategory.CONFIGURATION

    @pytest.mark.asyncio
    async def test_query_service_error_handling(self, mock_query_service):
        """Test error handling in query service."""
        from core.error_handling import (
            AIVillageException,
            ErrorCategory,
            ErrorContext,
            ErrorSeverity,
        )

        test_exception = AIVillageException(
            message="Query execution failed",
            category=ErrorCategory.ACCESS,
            severity=ErrorSeverity.INFO,
            context=ErrorContext(
                component="query-service", operation="execute_query", details={}
            ),
        )

        mock_query_service.execute_query.side_effect = test_exception

        with pytest.raises(AIVillageException) as exc_info:
            from services.core.interfaces import QueryRequest

            request = QueryRequest(query="test query", limit=5)
            await mock_query_service.execute_query(request)

        assert exc_info.value.message == "Query execution failed"
        assert exc_info.value.category == ErrorCategory.ACCESS

    def test_error_handler_with_test_config(self, test_config):
        """Test error handler creation with test configuration."""
        handler = ServiceErrorHandler(test_config.gateway.name)

        exception = AIVillageException(
            message="Configuration test error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.DEBUG,
        )

        response = handler.create_error_response(exception)

        assert response["error"]["service"] == test_config.gateway.name
        assert response["error"]["message"] == "Configuration test error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
