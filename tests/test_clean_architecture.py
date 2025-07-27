"""Tests for the clean architecture implementation.

This module tests the business logic, adapters, and interfaces
using the new test fixtures for complete isolation.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile
import pytest

from services.core.business_logic import (
    ChatBusinessLogic,
    HealthCheckLogic,
    QueryBusinessLogic,
    UploadBusinessLogic,
)
from services.core.http_adapters import (
    ChatHTTPAdapter,
    HealthHTTPAdapter,
    QueryHTTPAdapter,
    UploadHTTPAdapter,
)
from services.core.interfaces import (
    ChatRequest,
    ChatResponse,
    HealthCheckResponse,
    QueryRequest,
    QueryResponse,
    UploadRequest,
    UploadResponse,
)

# Import fixtures from our fixtures module


class TestBusinessLogic:
    """Test business logic components."""

    @pytest.mark.asyncio
    async def test_chat_business_logic(self, mock_chat_engine, test_config):
        """Test chat business logic."""
        chat_logic = ChatBusinessLogic(
            chat_engine=mock_chat_engine,
            max_message_length=test_config.gateway.max_request_size,
        )

        request = ChatRequest(
            message="Hello, how are you?", conversation_id="test-conv-123"
        )

        response = await chat_logic.process_chat(request)

        assert isinstance(response, ChatResponse)
        assert response.success is True
        assert response.response == "Test response"
        assert response.conversation_id == "test-conv-123"
        assert response.processing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_query_business_logic(self, mock_rag_pipeline, test_config):
        """Test query business logic."""
        query_logic = QueryBusinessLogic(rag_pipeline=mock_rag_pipeline)

        request = QueryRequest(query="What is AI?", limit=5)

        response = await query_logic.execute_query(request)

        assert isinstance(response, QueryResponse)
        assert response.success is True
        assert len(response.results) == 2
        assert response.results[0]["text"] == "Test result 1"
        assert response.total_count == 2

    @pytest.mark.asyncio
    async def test_upload_business_logic(self, mock_vector_store, test_config):
        """Test upload business logic."""
        upload_logic = UploadBusinessLogic(
            vector_store=mock_vector_store,
            max_file_size=test_config.gateway.max_request_size,
        )

        request = UploadRequest(
            filename="test.txt", content=b"Test file content", content_type="text/plain"
        )

        response = await upload_logic.process_upload(request)

        assert isinstance(response, UploadResponse)
        assert response.success is True
        assert response.filename == "test.txt"
        assert response.size == len(b"Test file content")
        assert response.status == "uploaded"

    @pytest.mark.asyncio
    async def test_health_check_logic(self, test_config):
        """Test health check logic."""
        health_logic = HealthCheckLogic(
            service_name=test_config.gateway.name, version=test_config.gateway.version
        )

        response = await health_logic.check_health()

        assert isinstance(response, HealthCheckResponse)
        assert response.success is True
        assert response.status == "ok"
        assert response.version == test_config.gateway.version
        assert "self" in response.services


class TestHTTPAdapters:
    """Test HTTP adapter components."""

    @pytest.mark.asyncio
    async def test_chat_adapter(self, mock_chat_service):
        """Test chat HTTP adapter."""
        adapter = ChatHTTPAdapter(mock_chat_service)

        request_data = {
            "message": "Hello, how are you?",
            "conversation_id": "test-conv-123",
        }

        response = await adapter.handle_chat_request(request_data)

        assert response["response"] == "Echo: Hello, how are you?"
        assert response["conversation_id"] == "test-conv-123"
        assert response["processing_time_ms"] == 100

    @pytest.mark.asyncio
    async def test_query_adapter(self, mock_query_service):
        """Test query HTTP adapter."""
        adapter = QueryHTTPAdapter(mock_query_service)

        request_data = {"query": "What is AI?", "limit": 5}

        response = await adapter.handle_query_request(request_data)

        assert response["results"][0]["text"] == "Result for: What is AI?"
        assert response["total_count"] == 1
        assert response["processing_time_ms"] == 50

    @pytest.mark.asyncio
    async def test_upload_adapter(self, mock_upload_service):
        """Test upload HTTP adapter."""
        adapter = UploadHTTPAdapter(mock_upload_service)

        # Create mock file
        file_mock = MagicMock(spec=UploadFile)
        file_mock.filename = "test.txt"
        file_mock.content_type = "text/plain"
        file_mock.read = AsyncMock(return_value=b"Test file content")

        response = await adapter.handle_upload_request(file_mock)

        assert response["status"] == "uploaded"
        assert response["filename"] == "test.txt"
        assert response["size"] == len(b"Test file content")
        assert response["file_id"] == "test-file-123"

    @pytest.mark.asyncio
    async def test_health_adapter(self, mock_health_service):
        """Test health HTTP adapter."""
        adapter = HealthHTTPAdapter(mock_health_service)

        response = await adapter.handle_health_check()

        assert response["status"] == "ok"
        assert response["version"] == "0.0.1"
        assert response["services"]["self"] == "ok"


class TestIntegration:
    """Test integration between components."""

    @pytest.mark.asyncio
    async def test_end_to_end_chat_flow(
        self,
        mock_business_logic_factory,
        mock_http_adapter_factory,
        isolated_test_environment,
    ):
        """Test complete chat flow using factories."""
        # Get adapters from factory
        chat_adapter = mock_http_adapter_factory.create_chat_adapter.return_value

        # Test chat request
        request_data = {
            "message": "Hello, how are you?",
            "conversation_id": "test-conv-123",
        }

        response = await chat_adapter.handle_chat_request(request_data)

        # Verify factory was called
        mock_http_adapter_factory.create_chat_adapter.assert_called_once()

        # Verify response
        assert response["response"] == "Test response"
        assert response["conversation_id"] == "test-conv-123"

    @pytest.mark.asyncio
    async def test_error_handling_in_adapters(self, mock_chat_service):
        """Test error handling in adapters."""
        # Make the service raise an exception
        mock_chat_service.process_chat.side_effect = Exception("Service error")

        adapter = ChatHTTPAdapter(mock_chat_service)

        with pytest.raises(Exception, match="Service error"):
            await adapter.handle_chat_request(
                {"message": "test", "conversation_id": "test-123"}
            )

    @pytest.mark.asyncio
    async def test_validation_in_business_logic(self, mock_chat_engine, test_config):
        """Test input validation in business logic."""
        chat_logic = ChatBusinessLogic(
            chat_engine=mock_chat_engine,
            max_message_length=test_config.gateway.max_request_size,
        )

        # Test empty message
        request = ChatRequest(message="", conversation_id="test-123")

        with pytest.raises(Exception, match="Message cannot be empty"):
            await chat_logic.process_chat(request)

    @pytest.mark.asyncio
    async def test_configuration_usage(self, mock_chat_engine, test_config):
        """Test that business logic uses configuration properly."""
        chat_logic = ChatBusinessLogic(
            chat_engine=mock_chat_engine,
            max_message_length=1000,  # Small limit for testing
        )

        # Test message length limit
        long_message = "x" * 1001  # Exceed the limit
        request = ChatRequest(message=long_message, conversation_id="test-123")

        with pytest.raises(Exception, match="Message exceeds maximum length"):
            await chat_logic.process_chat(request)


class TestServiceInterfaces:
    """Test that services implement interfaces correctly."""

    def test_chat_service_interface_compliance(self, mock_chat_service):
        """Test that chat service implements the interface."""

        # Check that all required methods exist
        assert hasattr(mock_chat_service, "process_chat")
        assert hasattr(mock_chat_service, "delete_conversation")
        assert hasattr(mock_chat_service, "delete_user_data")

    def test_query_service_interface_compliance(self, mock_query_service):
        """Test that query service implements the interface."""

        # Check that all required methods exist
        assert hasattr(mock_query_service, "execute_query")

    def test_upload_service_interface_compliance(self, mock_upload_service):
        """Test that upload service implements the interface."""

        # Check that all required methods exist
        assert hasattr(mock_upload_service, "process_upload")
        assert hasattr(mock_upload_service, "validate_file")


class TestFactoryPattern:
    """Test the factory pattern implementation."""

    def test_business_logic_factory_creation(self, mock_business_logic_factory):
        """Test business logic factory creates services correctly."""
        # Test that factory can create different types of logic
        chat_logic = mock_business_logic_factory.create_chat_logic()
        query_logic = mock_business_logic_factory.create_query_logic()
        upload_logic = mock_business_logic_factory.create_upload_logic()
        health_logic = mock_business_logic_factory.create_health_logic()

        # Verify all services were created
        assert chat_logic is not None
        assert query_logic is not None
        assert upload_logic is not None
        assert health_logic is not None

    def test_adapter_factory_creation(self, mock_http_adapter_factory):
        """Test HTTP adapter factory creates adapters correctly."""
        # Test that factory can create different types of adapters
        chat_adapter = mock_http_adapter_factory.create_chat_adapter()
        query_adapter = mock_http_adapter_factory.create_query_adapter()
        upload_adapter = mock_http_adapter_factory.create_upload_adapter()
        health_adapter = mock_http_adapter_factory.create_health_adapter()

        # Verify all adapters were created
        assert chat_adapter is not None
        assert query_adapter is not None
        assert upload_adapter is not None
        assert health_adapter is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
