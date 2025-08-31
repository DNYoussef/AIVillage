"""
Base Handler Template for AI Village Gateway Request Handlers

This template provides a foundation for all gateway request handlers with:
- Async request/response processing
- Middleware pipeline support
- Input validation and sanitization
- Comprehensive error handling
- Request/response logging and metrics
- Authentication and authorization
- Rate limiting support
"""

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum
import uuid
import traceback

from .base_service import BaseService, ServiceError


class HTTPMethod(Enum):
    """HTTP methods supported by handlers"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class HTTPStatus(Enum):
    """Common HTTP status codes"""
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    UNPROCESSABLE_ENTITY = 422
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


@dataclass
class RequestContext:
    """Request context information"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    method: str = ""
    path: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    path_params: Dict[str, str] = field(default_factory=dict)
    user_id: Optional[str] = None
    auth_token: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        return (datetime.utcnow() - self.start_time).total_seconds() * 1000


@dataclass
class ResponseContext:
    """Response context information"""
    status_code: int = HTTPStatus.OK.value
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    content_type: str = "application/json"
    
    def set_json_response(self, data: Any, status_code: int = HTTPStatus.OK.value) -> None:
        """Set JSON response data"""
        self.body = data
        self.status_code = status_code
        self.content_type = "application/json"
        self.headers["Content-Type"] = "application/json"
    
    def set_error_response(self, message: str, status_code: int, error_code: str = None, details: Dict[str, Any] = None) -> None:
        """Set error response data"""
        error_data = {
            "error": {
                "message": message,
                "code": error_code or "HANDLER_ERROR",
                "status": status_code
            }
        }
        if details:
            error_data["error"]["details"] = details
        
        self.set_json_response(error_data, status_code)


class HandlerError(Exception):
    """Base exception for handler errors"""
    
    def __init__(self, message: str, status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value, 
                 error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "HANDLER_ERROR"
        self.details = details or {}


class ValidationError(HandlerError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, HTTPStatus.BAD_REQUEST.value, "VALIDATION_ERROR", details)
        self.field = field


class AuthenticationError(HandlerError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, HTTPStatus.UNAUTHORIZED.value, "AUTHENTICATION_ERROR")


class AuthorizationError(HandlerError):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, HTTPStatus.FORBIDDEN.value, "AUTHORIZATION_ERROR")


class RateLimitError(HandlerError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(message, HTTPStatus.TOO_MANY_REQUESTS.value, "RATE_LIMIT_ERROR", details)


# Middleware types
MiddlewareFunction = Callable[[RequestContext, ResponseContext], Any]
AsyncMiddlewareFunction = Callable[[RequestContext, ResponseContext], Any]


@dataclass
class MiddlewareConfig:
    """Middleware configuration"""
    name: str
    function: Union[MiddlewareFunction, AsyncMiddlewareFunction]
    priority: int = 0  # Higher priority runs first
    enabled: bool = True


class BaseHandler(ABC):
    """
    Abstract base class for all AI Village gateway request handlers.
    
    Provides:
    - Request/response lifecycle management
    - Middleware pipeline execution
    - Input validation and sanitization
    - Error handling and logging
    - Authentication and authorization hooks
    - Rate limiting support
    - Metrics collection
    """
    
    def __init__(
        self,
        name: str,
        service_aggregator: 'ServiceAggregator' = None,
        config: Dict[str, Any] = None,
        logger: logging.Logger = None
    ):
        self.name = name
        self.service_aggregator = service_aggregator
        self.config = config or {}
        self.logger = logger or self._setup_logger()
        
        # Middleware pipeline
        self._middleware: List[MiddlewareConfig] = []
        self._middleware_sorted = False
        
        # Handler state
        self._initialized = False
        
        # Metrics
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        
        self.logger.info(f"Handler '{self.name}' created")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging for the handler"""
        logger = logging.getLogger(f"gateway.handler.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Handler Lifecycle
    
    async def initialize(self) -> None:
        """Initialize the handler"""
        if self._initialized:
            return
        
        try:
            self.logger.info(f"Initializing handler '{self.name}'...")
            
            # Setup default middleware
            await self._setup_default_middleware()
            
            # Handler-specific initialization
            await self._initialize_handler()
            
            # Sort middleware by priority
            self._sort_middleware()
            
            self._initialized = True
            self.logger.info(f"Handler '{self.name}' initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize handler '{self.name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise HandlerError(error_msg)
    
    # Request Processing
    
    async def handle_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str] = None,
        query_params: Dict[str, Any] = None,
        path_params: Dict[str, str] = None,
        body: Any = None
    ) -> Tuple[int, Dict[str, str], Any]:
        """
        Handle an incoming request.
        
        Returns:
            Tuple of (status_code, headers, body)
        """
        if not self._initialized:
            await self.initialize()
        
        # Create request and response contexts
        request_ctx = RequestContext(
            method=method.upper(),
            path=path,
            headers=headers or {},
            query_params=query_params or {},
            path_params=path_params or {}
        )
        
        response_ctx = ResponseContext()
        
        try:
            # Extract additional context from headers
            self._extract_request_context(request_ctx)
            
            # Log request start
            self.logger.info(
                f"Request started: {request_ctx.request_id} {method} {path}",
                extra={"request_id": request_ctx.request_id}
            )
            
            # Execute middleware pipeline and handler
            async with self._request_context(request_ctx, response_ctx):
                await self._execute_middleware_pipeline(request_ctx, response_ctx, body)
            
            # Log successful response
            self.logger.info(
                f"Request completed: {request_ctx.request_id} "
                f"status={response_ctx.status_code} "
                f"elapsed={request_ctx.elapsed_ms:.2f}ms",
                extra={"request_id": request_ctx.request_id}
            )
            
            return (
                response_ctx.status_code,
                response_ctx.headers,
                response_ctx.body
            )
            
        except HandlerError as e:
            # Handle known handler errors
            self.logger.warning(
                f"Handler error: {request_ctx.request_id} {str(e)}",
                extra={"request_id": request_ctx.request_id}
            )
            
            response_ctx.set_error_response(
                message=e.message,
                status_code=e.status_code,
                error_code=e.error_code,
                details=e.details
            )
            
            return (
                response_ctx.status_code,
                response_ctx.headers,
                response_ctx.body
            )
            
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(
                f"Unexpected error: {request_ctx.request_id} {str(e)}",
                extra={"request_id": request_ctx.request_id},
                exc_info=True
            )
            
            response_ctx.set_error_response(
                message="Internal server error",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                error_code="INTERNAL_ERROR",
                details={"traceback": traceback.format_exc()} if self.config.get("debug", False) else {}
            )
            
            return (
                response_ctx.status_code,
                response_ctx.headers,
                response_ctx.body
            )
    
    # Middleware Management
    
    def add_middleware(
        self,
        name: str,
        function: Union[MiddlewareFunction, AsyncMiddlewareFunction],
        priority: int = 0,
        enabled: bool = True
    ) -> None:
        """Add middleware to the pipeline"""
        middleware = MiddlewareConfig(
            name=name,
            function=function,
            priority=priority,
            enabled=enabled
        )
        
        self._middleware.append(middleware)
        self._middleware_sorted = False
        
        self.logger.info(f"Added middleware '{name}' with priority {priority}")
    
    def remove_middleware(self, name: str) -> None:
        """Remove middleware from the pipeline"""
        self._middleware = [m for m in self._middleware if m.name != name]
        self.logger.info(f"Removed middleware '{name}'")
    
    def enable_middleware(self, name: str) -> None:
        """Enable middleware"""
        for middleware in self._middleware:
            if middleware.name == name:
                middleware.enabled = True
                self.logger.info(f"Enabled middleware '{name}'")
                return
        
        self.logger.warning(f"Middleware '{name}' not found")
    
    def disable_middleware(self, name: str) -> None:
        """Disable middleware"""
        for middleware in self._middleware:
            if middleware.name == name:
                middleware.enabled = False
                self.logger.info(f"Disabled middleware '{name}'")
                return
        
        self.logger.warning(f"Middleware '{name}' not found")
    
    # Service Access
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """Get a service from the aggregator"""
        if self.service_aggregator:
            return self.service_aggregator.get_service(name)
        return None
    
    def require_service(self, name: str) -> BaseService:
        """Get a service from the aggregator, raise error if not found"""
        if self.service_aggregator:
            return self.service_aggregator.require_service(name)
        raise HandlerError(f"Service '{name}' not available - no service aggregator configured")
    
    # Context Managers
    
    @asynccontextmanager
    async def _request_context(self, request_ctx: RequestContext, response_ctx: ResponseContext):
        """Context manager for request processing"""
        start_time = time.time()
        
        try:
            yield
            
            # Update success metrics
            self._request_count += 1
            response_time = time.time() - start_time
            self._total_response_time += response_time
            
        except Exception as e:
            # Update error metrics
            self._request_count += 1
            self._error_count += 1
            response_time = time.time() - start_time
            self._total_response_time += response_time
            raise
    
    # Private Methods
    
    def _extract_request_context(self, request_ctx: RequestContext) -> None:
        """Extract additional context from request headers"""
        headers = request_ctx.headers
        
        # Extract common headers
        request_ctx.client_ip = headers.get("X-Forwarded-For", "").split(",")[0].strip()
        request_ctx.user_agent = headers.get("User-Agent", "")
        request_ctx.auth_token = headers.get("Authorization", "").replace("Bearer ", "")
        
        # Extract custom headers
        request_ctx.user_id = headers.get("X-User-ID")
    
    async def _execute_middleware_pipeline(
        self,
        request_ctx: RequestContext,
        response_ctx: ResponseContext,
        body: Any
    ) -> None:
        """Execute the middleware pipeline and handler"""
        # Ensure middleware is sorted
        if not self._middleware_sorted:
            self._sort_middleware()
        
        # Execute pre-processing middleware
        for middleware in self._middleware:
            if middleware.enabled:
                try:
                    if asyncio.iscoroutinefunction(middleware.function):
                        await middleware.function(request_ctx, response_ctx)
                    else:
                        middleware.function(request_ctx, response_ctx)
                except Exception as e:
                    self.logger.error(f"Middleware '{middleware.name}' error: {str(e)}")
                    raise HandlerError(f"Middleware error: {str(e)}")
        
        # Execute main handler
        result = await self._execute_handler(request_ctx, response_ctx, body)
        
        # Set response if handler returned data
        if result is not None:
            response_ctx.set_json_response(result)
    
    async def _execute_handler(
        self,
        request_ctx: RequestContext,
        response_ctx: ResponseContext,
        body: Any
    ) -> Any:
        """Execute the main handler logic"""
        method = request_ctx.method
        
        # Route to appropriate method handler
        if method == HTTPMethod.GET.value:
            return await self.handle_get(request_ctx, response_ctx)
        elif method == HTTPMethod.POST.value:
            return await self.handle_post(request_ctx, response_ctx, body)
        elif method == HTTPMethod.PUT.value:
            return await self.handle_put(request_ctx, response_ctx, body)
        elif method == HTTPMethod.DELETE.value:
            return await self.handle_delete(request_ctx, response_ctx)
        elif method == HTTPMethod.PATCH.value:
            return await self.handle_patch(request_ctx, response_ctx, body)
        else:
            raise HandlerError(
                f"Method {method} not allowed",
                HTTPStatus.METHOD_NOT_ALLOWED.value,
                "METHOD_NOT_ALLOWED"
            )
    
    def _sort_middleware(self) -> None:
        """Sort middleware by priority (higher priority first)"""
        self._middleware.sort(key=lambda m: m.priority, reverse=True)
        self._middleware_sorted = True
    
    async def _setup_default_middleware(self) -> None:
        """Setup default middleware"""
        # Request logging middleware
        async def request_logger(req_ctx: RequestContext, resp_ctx: ResponseContext):
            self.logger.debug(
                f"Processing request: {req_ctx.method} {req_ctx.path}",
                extra={"request_id": req_ctx.request_id}
            )
        
        self.add_middleware("request_logger", request_logger, priority=1000)
        
        # CORS middleware (if enabled)
        if self.config.get("cors", {}).get("enabled", False):
            def cors_middleware(req_ctx: RequestContext, resp_ctx: ResponseContext):
                cors_config = self.config.get("cors", {})
                resp_ctx.headers.update({
                    "Access-Control-Allow-Origin": cors_config.get("origin", "*"),
                    "Access-Control-Allow-Methods": cors_config.get("methods", "GET,POST,PUT,DELETE,OPTIONS"),
                    "Access-Control-Allow-Headers": cors_config.get("headers", "Content-Type,Authorization"),
                })
            
            self.add_middleware("cors", cors_middleware, priority=900)
    
    # Abstract Methods (Override in subclasses)
    
    async def _initialize_handler(self) -> None:
        """Handler-specific initialization (optional override)"""
        pass
    
    async def handle_get(self, request_ctx: RequestContext, response_ctx: ResponseContext) -> Any:
        """Handle GET requests (optional override)"""
        raise HandlerError("GET method not implemented", HTTPStatus.METHOD_NOT_ALLOWED.value)
    
    async def handle_post(self, request_ctx: RequestContext, response_ctx: ResponseContext, body: Any) -> Any:
        """Handle POST requests (optional override)"""
        raise HandlerError("POST method not implemented", HTTPStatus.METHOD_NOT_ALLOWED.value)
    
    async def handle_put(self, request_ctx: RequestContext, response_ctx: ResponseContext, body: Any) -> Any:
        """Handle PUT requests (optional override)"""
        raise HandlerError("PUT method not implemented", HTTPStatus.METHOD_NOT_ALLOWED.value)
    
    async def handle_delete(self, request_ctx: RequestContext, response_ctx: ResponseContext) -> Any:
        """Handle DELETE requests (optional override)"""
        raise HandlerError("DELETE method not implemented", HTTPStatus.METHOD_NOT_ALLOWED.value)
    
    async def handle_patch(self, request_ctx: RequestContext, response_ctx: ResponseContext, body: Any) -> Any:
        """Handle PATCH requests (optional override)"""
        raise HandlerError("PATCH method not implemented", HTTPStatus.METHOD_NOT_ALLOWED.value)
    
    # Metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get handler metrics"""
        avg_response_time = (
            self._total_response_time / max(self._request_count, 1)
        )
        
        return {
            "name": self.name,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "success_count": self._request_count - self._error_count,
            "error_rate": (self._error_count / max(self._request_count, 1)) * 100.0,
            "average_response_time_seconds": avg_response_time,
            "middleware_count": len([m for m in self._middleware if m.enabled])
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"initialized={self._initialized}, "
            f"middleware_count={len(self._middleware)}"
            f")"
        )


# Common middleware functions

async def authentication_middleware(request_ctx: RequestContext, response_ctx: ResponseContext) -> None:
    """Middleware for authentication validation"""
    # Skip authentication for OPTIONS requests
    if request_ctx.method == HTTPMethod.OPTIONS.value:
        return
    
    auth_token = request_ctx.auth_token
    if not auth_token:
        raise AuthenticationError("Authentication token required")
    
    # TODO: Validate token with auth service
    # This is a placeholder - implement actual token validation
    if auth_token == "invalid":
        raise AuthenticationError("Invalid authentication token")


def rate_limiting_middleware(
    max_requests: int = 100,
    window_seconds: int = 60
) -> MiddlewareFunction:
    """Create rate limiting middleware"""
    
    # Simple in-memory rate limiter (use Redis in production)
    request_counts: Dict[str, List[float]] = {}
    
    def rate_limiter(request_ctx: RequestContext, response_ctx: ResponseContext) -> None:
        client_id = request_ctx.client_ip or "unknown"
        current_time = time.time()
        
        # Clean old entries
        if client_id in request_counts:
            request_counts[client_id] = [
                req_time for req_time in request_counts[client_id]
                if current_time - req_time < window_seconds
            ]
        else:
            request_counts[client_id] = []
        
        # Check rate limit
        if len(request_counts[client_id]) >= max_requests:
            raise RateLimitError(
                f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds",
                retry_after=window_seconds
            )
        
        # Add current request
        request_counts[client_id].append(current_time)
    
    return rate_limiter


def validation_middleware(schema: Dict[str, Any]) -> AsyncMiddlewareFunction:
    """Create input validation middleware"""
    
    async def validator(request_ctx: RequestContext, response_ctx: ResponseContext) -> None:
        # TODO: Implement schema validation
        # This is a placeholder - implement actual validation logic
        pass
    
    return validator


# Export public interface
__all__ = [
    'BaseHandler',
    'RequestContext',
    'ResponseContext',
    'HandlerError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'HTTPMethod',
    'HTTPStatus',
    'MiddlewareFunction',
    'AsyncMiddlewareFunction',
    'MiddlewareConfig',
    'authentication_middleware',
    'rate_limiting_middleware',
    'validation_middleware'
]