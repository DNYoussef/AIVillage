"""Service interface definitions for clean architecture.

This module provides abstract interfaces that separate business logic
from HTTP/transport concerns, enabling better testability and maintainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol


@dataclass
class ServiceRequest:
    """Base class for service requests."""

    request_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ServiceResponse:
    """Base class for service responses."""

    success: bool
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ServiceInterface(ABC):
    """Abstract interface for service business logic."""

    @abstractmethod
    async def process(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request and return a response."""


# Chat Service Interfaces


@dataclass
class ChatRequest:
    """Chat service request."""

    message: str
    conversation_id: str | None = None
    user_id: str | None = None
    context: dict[str, Any] | None = None
    request_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ChatResponse:
    """Chat service response."""

    success: bool
    response: str
    conversation_id: str
    processing_time_ms: int
    calibrated_prob: float | None = None
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ChatServiceInterface(ServiceInterface):
    """Interface for chat service business logic."""

    @abstractmethod
    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request."""

    @abstractmethod
    async def delete_conversation(self, conversation_id: str) -> ServiceResponse:
        """Delete a conversation."""

    @abstractmethod
    async def delete_user_data(self, user_id: str) -> ServiceResponse:
        """Delete all user data."""


# Query Service Interfaces


@dataclass
class QueryRequest:
    """Query service request."""

    query: str
    filters: dict[str, Any] | None = None
    limit: int = 10
    request_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class QueryResponse:
    """Query service response."""

    success: bool
    results: list[dict[str, Any]]
    total_count: int
    processing_time_ms: int
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class QueryServiceInterface(ServiceInterface):
    """Interface for query service business logic."""

    @abstractmethod
    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute a query."""


# Upload Service Interfaces


@dataclass
class UploadRequest:
    """Upload service request."""

    filename: str
    content: bytes
    content_type: str
    user_id: str | None = None
    request_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class UploadResponse:
    """Upload service response."""

    success: bool
    file_id: str
    filename: str
    size: int
    status: str
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class UploadServiceInterface(ServiceInterface):
    """Interface for upload service business logic."""

    @abstractmethod
    async def process_upload(self, request: UploadRequest) -> UploadResponse:
        """Process a file upload."""

    @abstractmethod
    async def validate_file(self, filename: str, content: bytes) -> ServiceResponse:
        """Validate uploaded file."""


# Health Check Interface


@dataclass
class HealthCheckResponse:
    """Health check response."""

    success: bool
    status: str
    version: str
    services: dict[str, str]
    timestamp: datetime
    data: Any | None = None
    error: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class HealthCheckInterface(ServiceInterface):
    """Interface for health check service."""

    @abstractmethod
    async def check_health(self) -> HealthCheckResponse:
        """Check service health."""


# Service Factory Protocol


class ServiceFactory(Protocol):
    """Protocol for service factories."""

    def create_chat_service(self) -> ChatServiceInterface:
        """Create chat service instance."""
        ...

    def create_query_service(self) -> QueryServiceInterface:
        """Create query service instance."""
        ...

    def create_upload_service(self) -> UploadServiceInterface:
        """Create upload service instance."""
        ...

    def create_health_service(self) -> HealthCheckInterface:
        """Create health check service instance."""
        ...
