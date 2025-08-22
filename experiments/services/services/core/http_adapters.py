"""HTTP adapters for converting between HTTP requests and service interfaces.

This module provides adapters that handle HTTP-specific concerns while
delegating business logic to the service interfaces.
"""

from typing import Any

from fastapi import UploadFile
from services.core.interfaces import ChatRequest as ServiceChatRequest
from services.core.interfaces import ChatServiceInterface, HealthCheckInterface
from services.core.interfaces import QueryRequest as ServiceQueryRequest
from services.core.interfaces import QueryServiceInterface
from services.core.interfaces import UploadRequest as ServiceUploadRequest
from services.core.interfaces import UploadServiceInterface
from services.core.service_error_handler import ServiceErrorHandler

from core.error_handling import get_component_logger


class HTTPAdapter:
    """Base HTTP adapter class."""

    def __init__(self, service_name: str) -> None:
        self.logger = get_component_logger(f"HTTPAdapter.{service_name}")
        self.error_handler = ServiceErrorHandler(service_name)


class ChatHTTPAdapter(HTTPAdapter):
    """HTTP adapter for chat service."""

    def __init__(self, chat_service: ChatServiceInterface) -> None:
        super().__init__("ChatService")
        self.chat_service = chat_service

    async def handle_chat_request(self, chat_request: dict[str, Any]) -> dict[str, Any]:
        """Convert HTTP chat request to service request and process."""
        # Convert HTTP request to service request
        service_request = ServiceChatRequest(
            message=chat_request.get("message", ""),
            conversation_id=chat_request.get("conversation_id"),
            user_id=chat_request.get("user_id"),
            context=chat_request.get("context"),
        )

        # Process with business logic
        response = await self.chat_service.process_chat(service_request)

        # Convert service response to HTTP response
        return {
            "response": response.response,
            "conversation_id": response.conversation_id,
            "processing_time_ms": response.processing_time_ms,
            "calibrated_prob": response.calibrated_prob,
        }

    async def handle_delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        """Handle conversation deletion."""
        response = await self.chat_service.delete_conversation(conversation_id)
        return response.data if response.success else {"error": response.error}

    async def handle_delete_user_data(self, user_id: str) -> dict[str, Any]:
        """Handle user data deletion."""
        response = await self.chat_service.delete_user_data(user_id)
        return response.data if response.success else {"error": response.error}


class QueryHTTPAdapter(HTTPAdapter):
    """HTTP adapter for query service."""

    def __init__(self, query_service: QueryServiceInterface) -> None:
        super().__init__("QueryService")
        self.query_service = query_service

    async def handle_query_request(self, query_request: dict[str, Any]) -> dict[str, Any]:
        """Convert HTTP query request to service request and process."""
        # Convert HTTP request to service request
        service_request = ServiceQueryRequest(
            query=query_request.get("query", ""),
            filters=query_request.get("filters"),
            limit=query_request.get("limit", 10),
        )

        # Process with business logic
        response = await self.query_service.execute_query(service_request)

        # Convert service response to HTTP response
        return {
            "results": response.results,
            "total_count": response.total_count,
            "processing_time_ms": response.processing_time_ms,
        }


class UploadHTTPAdapter(HTTPAdapter):
    """HTTP adapter for upload service."""

    def __init__(self, upload_service: UploadServiceInterface) -> None:
        super().__init__("UploadService")
        self.upload_service = upload_service

    async def handle_upload_request(self, file: UploadFile) -> dict[str, Any]:
        """Convert HTTP upload request to service request and process."""
        # Read file content
        content = await file.read()

        # Convert HTTP request to service request
        service_request = ServiceUploadRequest(
            filename=file.filename or "",
            content=content,
            content_type=file.content_type or "application/octet-stream",
        )

        # Process with business logic
        response = await self.upload_service.process_upload(service_request)

        # Convert service response to HTTP response
        return {
            "status": response.status,
            "filename": response.filename,
            "size": response.size,
            "file_id": response.file_id,
            "message": f"File {response.filename} uploaded successfully",
        }


class HealthHTTPAdapter(HTTPAdapter):
    """HTTP adapter for health check service."""

    def __init__(self, health_service: HealthCheckInterface) -> None:
        super().__init__("HealthService")
        self.health_service = health_service

    async def handle_health_check(self) -> dict[str, Any]:
        """Handle health check request."""
        response = await self.health_service.check_health()

        return {
            "status": response.status,
            "version": response.version,
            "services": response.services,
            "timestamp": response.timestamp.isoformat(),
        }


class HTTPAdapterFactory:
    """Factory for creating HTTP adapters."""

    def __init__(self, business_logic_factory) -> None:
        self.business_logic_factory = business_logic_factory

    def create_chat_adapter(self) -> ChatHTTPAdapter:
        """Create chat HTTP adapter."""
        chat_service = self.business_logic_factory.create_chat_service()
        return ChatHTTPAdapter(chat_service)

    def create_query_adapter(self) -> QueryHTTPAdapter:
        """Create query HTTP adapter."""
        query_service = self.business_logic_factory.create_query_service()
        return QueryHTTPAdapter(query_service)

    def create_upload_adapter(self) -> UploadHTTPAdapter:
        """Create upload HTTP adapter."""
        upload_service = self.business_logic_factory.create_upload_service()
        return UploadHTTPAdapter(upload_service)

    def create_health_adapter(self) -> HealthHTTPAdapter:
        """Create health HTTP adapter."""
        health_service = self.business_logic_factory.create_health_service()
        return HealthHTTPAdapter(health_service)
