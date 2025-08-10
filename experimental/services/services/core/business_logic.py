"""Business logic implementations for services.

This module contains the actual business logic separated from HTTP concerns,
making it easier to test and maintain.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from core.error_handling import (
    AIVillageException,
    ErrorCategory,
    ErrorContext,
    ErrorSeverity,
    get_component_logger,
)
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


class ChatBusinessLogic(ChatServiceInterface):
    """Business logic for chat service."""

    def __init__(self, chat_engine=None, max_message_length: int = 5000) -> None:
        self.logger = get_component_logger("ChatBusinessLogic")
        self.chat_engine = chat_engine
        self.max_message_length = max_message_length
        self.conversations = {}  # In-memory storage for demo

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request."""
        start_time = time.time()

        # Validate message
        if not request.message or not request.message.strip():
            raise AIVillageException(
                message="Message cannot be empty",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.INFO,
                context=ErrorContext(
                    component="ChatBusinessLogic",
                    operation="process_chat",
                    details={"message": request.message},
                ),
            )

        if len(request.message) > self.max_message_length:
            raise AIVillageException(
                message=f"Message exceeds maximum length of {self.max_message_length}",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.INFO,
                context=ErrorContext(
                    component="ChatBusinessLogic",
                    operation="process_chat",
                    details={"message_length": len(request.message)},
                ),
            )

        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())

        # Process with chat engine if available
        if self.chat_engine:
            try:
                result = self.chat_engine.process_chat(request.message, conversation_id)
                response_text = result.get("response", "I understand your message.")
            except Exception as e:
                self.logger.exception(f"Chat engine error: {e}")
                response_text = "I'm having trouble processing your request."
        else:
            # Fallback response
            response_text = f"Echo: {request.message}"

        # Store conversation
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].append(
            {
                "user": request.message,
                "assistant": response_text,
                "timestamp": datetime.now(timezone.utc),
            }
        )

        processing_time = int((time.time() - start_time) * 1000)

        return ChatResponse(
            success=True,
            response=response_text,
            conversation_id=conversation_id,
            processing_time_ms=processing_time,
            calibrated_prob=None,
        )

    async def delete_conversation(self, conversation_id: str) -> ServiceResponse:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return ServiceResponse(success=True, data={"deleted": True})
        return ServiceResponse(
            success=False, error={"message": "Conversation not found"}
        )

    async def delete_user_data(self, user_id: str) -> ServiceResponse:
        """Delete all user data."""
        # In a real system, this would delete from persistent storage
        deleted_count = 0
        for conv_id in list(self.conversations.keys()):
            if conv_id.startswith(user_id):
                del self.conversations[conv_id]
                deleted_count += 1

        return ServiceResponse(
            success=True, data={"deleted_conversations": deleted_count}
        )

    async def process(self, request: ChatRequest) -> ChatResponse:
        """Generic process method."""
        return await self.process_chat(request)


class QueryBusinessLogic(QueryServiceInterface):
    """Business logic for query service."""

    def __init__(self, rag_pipeline=None) -> None:
        self.logger = get_component_logger("QueryBusinessLogic")
        self.rag_pipeline = rag_pipeline

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query."""
        start_time = time.time()

        # Validate query
        if not request.query or not request.query.strip():
            raise AIVillageException(
                message="Query cannot be empty",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.INFO,
                context=ErrorContext(
                    component="QueryBusinessLogic",
                    operation="execute_query",
                    details={},
                ),
            )

        # Process with RAG pipeline if available
        if self.rag_pipeline:
            try:
                result = await self.rag_pipeline.process_query(request.query)
                results = result.get("chunks", [])
            except Exception as e:
                self.logger.exception(f"RAG pipeline error: {e}")
                results = []
        else:
            # Fallback results
            results = [{"text": f"Result for: {request.query}", "score": 0.9}]

        processing_time = int((time.time() - start_time) * 1000)

        return QueryResponse(
            success=True,
            results=results[: request.limit],
            total_count=len(results),
            processing_time_ms=processing_time,
        )

    async def process(self, request: QueryRequest) -> QueryResponse:
        """Generic process method."""
        return await self.execute_query(request)


class UploadBusinessLogic(UploadServiceInterface):
    """Business logic for upload service."""

    def __init__(
        self, vector_store=None, max_file_size: int = 10 * 1024 * 1024
    ) -> None:
        self.logger = get_component_logger("UploadBusinessLogic")
        self.vector_store = vector_store
        self.max_file_size = max_file_size
        self.allowed_extensions = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}

    async def process_upload(self, request: UploadRequest) -> UploadResponse:
        """Process a file upload."""
        # Validate file
        validation_result = await self.validate_file(request.filename, request.content)

        if not validation_result.success:
            raise AIVillageException(
                message=validation_result.error["message"],
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.INFO,
                context=ErrorContext(
                    component="UploadBusinessLogic",
                    operation="process_upload",
                    details=validation_result.error,
                ),
            )

        # Generate file ID
        file_id = str(uuid.uuid4())

        # Process with vector store if available
        if self.vector_store:
            try:
                # In a real system, we'd process and store the file
                self.logger.info(f"Storing file {request.filename} with ID {file_id}")
            except Exception as e:
                self.logger.exception(f"Vector store error: {e}")

        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=request.filename,
            size=len(request.content),
            status="uploaded",
        )

    async def validate_file(self, filename: str, content: bytes) -> ServiceResponse:
        """Validate uploaded file."""
        # Check filename
        if not filename or not filename.strip():
            return ServiceResponse(
                success=False, error={"message": "Filename cannot be empty"}
            )

        # Check extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in self.allowed_extensions:
            return ServiceResponse(
                success=False,
                error={
                    "message": f"Unsupported file type: {file_ext}",
                    "allowed_extensions": list(self.allowed_extensions),
                },
            )

        # Check size
        if len(content) > self.max_file_size:
            return ServiceResponse(
                success=False,
                error={
                    "message": f"File size exceeds {self.max_file_size // (1024 * 1024)}MB limit",
                    "file_size": len(content),
                    "max_size": self.max_file_size,
                },
            )

        return ServiceResponse(success=True)

    async def process(self, request: UploadRequest) -> UploadResponse:
        """Generic process method."""
        return await self.process_upload(request)


class HealthCheckLogic(HealthCheckInterface):
    """Business logic for health checks."""

    def __init__(
        self,
        service_name: str,
        version: str,
        dependencies: dict[str, Any] | None = None,
    ) -> None:
        self.logger = get_component_logger("HealthCheckLogic")
        self.service_name = service_name
        self.version = version
        self.dependencies = dependencies or {}

    async def check_health(self) -> HealthCheckResponse:
        """Check service health."""
        services_status = {"self": "ok"}

        # Check dependencies
        for name, dep in self.dependencies.items():
            try:
                if hasattr(dep, "health_check"):
                    status = await dep.health_check()
                    services_status[name] = "ok" if status else "degraded"
                else:
                    services_status[name] = "unknown"
            except Exception as e:
                self.logger.exception(f"Health check failed for {name}: {e}")
                services_status[name] = "error"

        overall_status = (
            "ok" if all(s == "ok" for s in services_status.values()) else "degraded"
        )

        return HealthCheckResponse(
            success=True,
            status=overall_status,
            version=self.version,
            services=services_status,
            timestamp=datetime.now(timezone.utc),
        )

    async def process(self, request: Any) -> HealthCheckResponse:
        """Generic process method."""
        return await self.check_health()


class ServiceBusinessLogicFactory:
    """Factory for creating business logic instances."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.chat_engine = None  # Would be injected in real system
        self.rag_pipeline = None  # Would be injected in real system
        self.vector_store = None  # Would be injected in real system

    def create_chat_service(self) -> ChatServiceInterface:
        """Create chat service instance."""
        return ChatBusinessLogic(
            chat_engine=self.chat_engine,
            max_message_length=self.config.get("max_message_length", 5000),
        )

    def create_query_service(self) -> QueryServiceInterface:
        """Create query service instance."""
        return QueryBusinessLogic(rag_pipeline=self.rag_pipeline)

    def create_upload_service(self) -> UploadServiceInterface:
        """Create upload service instance."""
        return UploadBusinessLogic(
            vector_store=self.vector_store,
            max_file_size=self.config.get("max_file_size", 10 * 1024 * 1024),
        )

    def create_health_service(self) -> HealthCheckInterface:
        """Create health check service instance."""
        return HealthCheckLogic(
            service_name=self.config.get("service_name", "unknown"),
            version=self.config.get("version", "0.0.0"),
            dependencies=self.config.get("dependencies", {}),
        )
