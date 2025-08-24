"""
Query Handler with Idempotency Support

This module demonstrates how to implement idempotency keys for mutating operations
and integrate with the resilient HTTP client for external calls.
"""

import logging
import time
from typing import Any
import uuid

from fastapi import HTTPException, Request, status
from pydantic import BaseModel

from packages.core.common import get_http_client, is_enabled

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request model for query operations."""

    query: str
    mode: str = "balanced"
    user_id: str | None = None
    context: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    """Response model for query operations."""

    query_id: str
    response: str
    metadata: dict[str, Any]
    processing_time_ms: int


class IdempotentQueryHandler:
    """Handler for query operations with idempotency support."""

    def __init__(self):
        self.http_client = get_http_client()
        self.query_cache: dict[str, QueryResponse] = {}

    def _generate_query_id(self) -> str:
        """Generate unique query ID."""
        return str(uuid.uuid4())

    def _extract_idempotency_key(self, request: Request) -> str | None:
        """Extract idempotency key from request headers."""
        return request.headers.get("idempotency-key") or request.headers.get("Idempotency-Key")

    def _validate_idempotency_key(self, key: str) -> bool:
        """Validate idempotency key format."""
        # Simple validation - should be at least 16 characters
        return len(key) >= 16 and key.replace("-", "").replace("_", "").isalnum()

    async def handle_query(self, request: Request, query_request: QueryRequest) -> QueryResponse:
        """Handle query with idempotency support."""
        start_time = time.time()

        # Extract idempotency key
        idempotency_key = self._extract_idempotency_key(request)

        # Check for cached response if idempotency key provided
        if idempotency_key:
            if not self._validate_idempotency_key(idempotency_key):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid idempotency key format")

            # Check cache
            if idempotency_key in self.query_cache:
                logger.info(f"Returning cached response for idempotency key: {idempotency_key[:16]}...")
                return self.query_cache[idempotency_key]

        # Generate query ID
        query_id = self._generate_query_id()

        try:
            # Process query based on enabled features
            if is_enabled("advanced_rag_features", query_request.user_id):
                response_text = await self._process_advanced_query(query_request)
            else:
                response_text = await self._process_standard_query(query_request)

            # Simulate external API call with resilient client
            if is_enabled("rag_cognitive_nexus", query_request.user_id):
                await self._enhance_with_external_service(query_request, response_text)

            # Create response
            processing_time = int((time.time() - start_time) * 1000)

            query_response = QueryResponse(
                query_id=query_id,
                response=response_text,
                metadata={
                    "mode": query_request.mode,
                    "user_id": query_request.user_id,
                    "features_enabled": {
                        "advanced_rag": is_enabled("advanced_rag_features", query_request.user_id),
                        "cognitive_nexus": is_enabled("rag_cognitive_nexus", query_request.user_id),
                    },
                    "timestamp": time.time(),
                },
                processing_time_ms=processing_time,
            )

            # Cache response if idempotency key provided
            if idempotency_key:
                self.query_cache[idempotency_key] = query_response
                logger.info(f"Cached response for idempotency key: {idempotency_key[:16]}...")

            return query_response

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query processing failed: {str(e)}"
            )

    async def _process_standard_query(self, query_request: QueryRequest) -> str:
        """Process query with standard RAG system."""
        # Simulate standard processing
        return f"Standard response to: {query_request.query}"

    async def _process_advanced_query(self, query_request: QueryRequest) -> str:
        """Process query with advanced RAG features."""
        # Simulate advanced processing with Bayesian trust
        return f"Advanced response with Bayesian trust to: {query_request.query}"

    async def _enhance_with_external_service(self, query_request: QueryRequest, response: str):
        """Enhance response using external cognitive nexus service."""
        try:
            # Example external API call with resilient client
            external_response = await self.http_client.post(
                "https://cognitive-nexus.example.com/enhance",
                json={"query": query_request.query, "response": response, "context": query_request.context or {}},
                idempotency_key=f"enhance-{query_request.query[:16]}-{hash(response) % 10000}",
            )

            if external_response.status_code == 200:
                enhancement = external_response.json()
                logger.info(f"Enhanced response with external service: {enhancement.get('confidence', 0)}")

        except Exception as e:
            logger.warning(f"External enhancement failed, continuing with standard response: {e}")


# Message deduplication handler
class MessageDeduplicator:
    """Handle message deduplication for distributed systems."""

    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.seen_messages: dict[str, float] = {}

    def is_duplicate(self, message_id: str) -> bool:
        """Check if message is a duplicate within the time window."""
        current_time = time.time()

        # Cleanup old messages
        cutoff_time = current_time - self.window_seconds
        self.seen_messages = {
            msg_id: timestamp for msg_id, timestamp in self.seen_messages.items() if timestamp > cutoff_time
        }

        # Check for duplicate
        if message_id in self.seen_messages:
            return True

        # Record new message
        self.seen_messages[message_id] = current_time
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get deduplication statistics."""
        return {
            "active_messages": len(self.seen_messages),
            "window_seconds": self.window_seconds,
            "oldest_message_age": (time.time() - min(self.seen_messages.values()) if self.seen_messages else 0),
        }


# Global instances
_query_handler: IdempotentQueryHandler | None = None
_message_deduplicator: MessageDeduplicator | None = None


def get_query_handler() -> IdempotentQueryHandler:
    """Get global query handler instance."""
    global _query_handler
    if _query_handler is None:
        _query_handler = IdempotentQueryHandler()
    return _query_handler


def get_message_deduplicator() -> MessageDeduplicator:
    """Get global message deduplicator instance."""
    global _message_deduplicator
    if _message_deduplicator is None:
        _message_deduplicator = MessageDeduplicator()
    return _message_deduplicator


# Usage example for FastAPI
"""
from fastapi import FastAPI, Request
from .query import QueryRequest, get_query_handler

app = FastAPI()
query_handler = get_query_handler()

@app.post("/v1/query", response_model=QueryResponse)
async def handle_query_endpoint(request: Request, query_request: QueryRequest):
    return await query_handler.handle_query(request, query_request)

@app.get("/v1/health/idempotency")
async def get_idempotency_stats():
    return {
        "query_cache_size": len(query_handler.query_cache),
        "message_deduplication": get_message_deduplicator().get_stats(),
        "circuit_breakers": query_handler.http_client.get_circuit_breaker_status()
    }
"""
