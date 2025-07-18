"""Test fixtures for service testing.

This module provides comprehensive test fixtures for isolated testing
of services without external dependencies.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from services.core.business_logic import ServiceBusinessLogicFactory
from services.core.config import AIConfig, SecurityConfig, ServiceConfig, UnifiedConfig
from services.core.http_adapters import HTTPAdapterFactory
from services.core.interfaces import (
    ChatRequest,
    ChatResponse,
    ChatServiceInterface,
    HealthCheckInterface,
    HealthCheckResponse,
    QueryRequest,
    QueryResponse,
    QueryServiceInterface,
    ServiceResponse,
    UploadRequest,
    UploadResponse,
    UploadServiceInterface,
)


@pytest.fixture
def mock_chat_engine():
    """Mock chat engine."""
    engine = Mock()
    engine.process_chat.return_value = {
        "response": "Test response",
        "conversation_id": "test-conv-123",
        "chunks": [],
    }
    return engine


@pytest.fixture
def mock_rag_pipeline():
    """Mock RAG pipeline."""
    pipeline = AsyncMock()
    pipeline.process_query.return_value = {
        "chunks": [
            {"text": "Test result 1", "score": 0.9},
            {"text": "Test result 2", "score": 0.8},
        ]
    }
    pipeline.get_embedding.return_value = [0.1] * 768
    pipeline.add_document = AsyncMock()
    return pipeline


@pytest.fixture
def mock_vector_store():
    """Mock vector store."""
    store = Mock()
    store.add_texts = AsyncMock()
    store.similarity_search = AsyncMock(
        return_value=[{"text": "Similar text", "metadata": {}}]
    )
    return store


@pytest.fixture
def test_config():
    """Test configuration."""
    return UnifiedConfig(
        gateway=ServiceConfig(
            name="test-gateway",
            version="0.0.1",
            port=8888,
            max_request_size=1024 * 1024,  # 1MB for testing
        ),
        twin=ServiceConfig(
            name="test-twin",
            version="0.0.1",
            port=8889,
            max_request_size=1024 * 1024,  # 1MB for testing
        ),
        security=SecurityConfig(
            secret_key="test-secret",
            rate_limit_requests=10,
            rate_limit_window=60,
        ),
        ai=AIConfig(
            model_name="test-model",
            max_tokens=100,
            temperature=0.5,
            max_context_length=1000,
        ),
    )


@pytest.fixture
def mock_business_logic_factory(
    mock_chat_engine, mock_rag_pipeline, mock_vector_store, test_config
):
    """Mock business logic factory with all dependencies."""
    factory = ServiceBusinessLogicFactory(
        {
            "service_name": "test-service",
            "version": "0.0.1",
            "max_message_length": 1000,
            "max_file_size": 1024 * 1024,  # 1MB
            "dependencies": {
                "chat_engine": mock_chat_engine,
                "rag_pipeline": mock_rag_pipeline,
                "vector_store": mock_vector_store,
            },
        }
    )

    # Inject mocks
    factory.chat_engine = mock_chat_engine
    factory.rag_pipeline = mock_rag_pipeline
    factory.vector_store = mock_vector_store

    return factory


@pytest.fixture
def mock_chat_service():
    """Mock chat service."""
    service = Mock(spec=ChatServiceInterface)

    async def mock_process_chat(request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            success=True,
            response=f"Echo: {request.message}",
            conversation_id=request.conversation_id or "test-conv-123",
            processing_time_ms=100,
        )

    async def mock_delete_conversation(conv_id: str) -> ServiceResponse:
        return ServiceResponse(success=True, data={"deleted": True})

    async def mock_delete_user_data(user_id: str) -> ServiceResponse:
        return ServiceResponse(success=True, data={"deleted_conversations": 2})

    service.process_chat = AsyncMock(side_effect=mock_process_chat)
    service.delete_conversation = AsyncMock(side_effect=mock_delete_conversation)
    service.delete_user_data = AsyncMock(side_effect=mock_delete_user_data)

    return service


@pytest.fixture
def mock_query_service():
    """Mock query service."""
    service = Mock(spec=QueryServiceInterface)

    async def mock_execute_query(request: QueryRequest) -> QueryResponse:
        return QueryResponse(
            success=True,
            results=[{"text": f"Result for: {request.query}", "score": 0.9}],
            total_count=1,
            processing_time_ms=50,
        )

    service.execute_query = AsyncMock(side_effect=mock_execute_query)

    return service


@pytest.fixture
def mock_upload_service():
    """Mock upload service."""
    service = Mock(spec=UploadServiceInterface)

    async def mock_process_upload(request: UploadRequest) -> UploadResponse:
        return UploadResponse(
            success=True,
            file_id="test-file-123",
            filename=request.filename,
            size=len(request.content),
            status="uploaded",
        )

    async def mock_validate_file(filename: str, content: bytes) -> ServiceResponse:
        if filename.endswith(".txt"):
            return ServiceResponse(success=True)
        return ServiceResponse(
            success=False, error={"message": "Unsupported file type"}
        )

    service.process_upload = AsyncMock(side_effect=mock_process_upload)
    service.validate_file = AsyncMock(side_effect=mock_validate_file)

    return service


@pytest.fixture
def mock_health_service():
    """Mock health service."""
    service = Mock(spec=HealthCheckInterface)

    async def mock_check_health() -> HealthCheckResponse:
        return HealthCheckResponse(
            success=True,
            status="ok",
            version="0.0.1",
            services={"self": "ok", "database": "ok"},
            timestamp=datetime.utcnow(),
        )

    service.check_health = AsyncMock(side_effect=mock_check_health)

    return service


@pytest.fixture
def mock_http_adapter_factory(
    mock_chat_service, mock_query_service, mock_upload_service, mock_health_service
):
    """Mock HTTP adapter factory."""
    factory = Mock(spec=HTTPAdapterFactory)

    # Create mock adapters
    chat_adapter = Mock()
    chat_adapter.handle_chat_request = AsyncMock(
        return_value={
            "response": "Test response",
            "conversation_id": "test-conv-123",
            "processing_time_ms": 100,
        }
    )
    chat_adapter.handle_delete_conversation = AsyncMock(return_value={"deleted": True})
    chat_adapter.handle_delete_user_data = AsyncMock(
        return_value={"deleted_conversations": 2}
    )

    query_adapter = Mock()
    query_adapter.handle_query_request = AsyncMock(
        return_value={
            "results": [{"text": "Test result", "score": 0.9}],
            "total_count": 1,
            "processing_time_ms": 50,
        }
    )

    upload_adapter = Mock()
    upload_adapter.handle_upload_request = AsyncMock(
        return_value={
            "status": "uploaded",
            "filename": "test.txt",
            "size": 100,
            "file_id": "test-file-123",
            "message": "File uploaded successfully",
        }
    )

    health_adapter = Mock()
    health_adapter.handle_health_check = AsyncMock(
        return_value={
            "status": "ok",
            "version": "0.0.1",
            "services": {"self": "ok"},
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    factory.create_chat_adapter.return_value = chat_adapter
    factory.create_query_adapter.return_value = query_adapter
    factory.create_upload_adapter.return_value = upload_adapter
    factory.create_health_adapter.return_value = health_adapter

    return factory


@pytest.fixture
def sample_chat_request():
    """Sample chat request data."""
    return {"message": "Hello, how are you?", "conversation_id": "test-conv-123"}


@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {"query": "What is AI?", "limit": 5}


@pytest.fixture
def sample_upload_file():
    """Sample upload file data."""
    file_mock = Mock()
    file_mock.filename = "test.txt"
    file_mock.content_type = "text/plain"
    file_mock.read = AsyncMock(return_value=b"Test file content")
    return file_mock


@pytest.fixture
def mock_external_dependencies():
    """Mock all external dependencies."""
    return {
        "torch": Mock(),
        "numpy": Mock(),
        "sklearn": Mock(),
        "faiss": Mock(),
        "transformers": Mock(),
        "openai": Mock(),
        "langroid": Mock(),
    }


@pytest.fixture
def isolated_test_environment(monkeypatch, mock_external_dependencies):
    """Completely isolated test environment."""
    # Mock all heavy dependencies
    for module_name, mock_module in mock_external_dependencies.items():
        monkeypatch.setattr(f"sys.modules['{module_name}']", mock_module, raising=False)

    # Mock environment variables
    test_env = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "SECRET_KEY": "test-secret-key",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
    }

    for key, value in test_env.items():
        monkeypatch.setenv(key, value)

    return test_env
