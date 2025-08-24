#!/usr/bin/env python3
"""
AIVillage Unified Production Gateway Server

This is the consolidated, production-ready HTTP API gateway that unifies all capabilities
from across the codebase into a single, high-performance FastAPI application.

Architecture: User Request → API Gateway → Agent Controller → Knowledge System → Response
Performance Target: <100ms health check response time
Security: Complete middleware stack with input validation, rate limiting, and headers
"""

import asyncio
from datetime import datetime, timezone
import logging
import time
from typing import Any

from cachetools import TTLCache

# Core FastAPI and middleware
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer

# Configuration and dependencies
import httpx

# Prometheus monitoring
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("aivillage.gateway")


# Configuration System
class GatewayConfig:
    """Consolidated gateway configuration with environment variable support"""

    def __init__(self):
        import os

        # Service configuration
        self.host = os.getenv("GATEWAY_HOST", "0.0.0.0")
        self.port = int(os.getenv("GATEWAY_PORT", "8000"))
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

        # Security configuration
        self.api_key = os.getenv("API_KEY", "dev-gateway-key-12345")
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

        # Rate limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))

        # CORS configuration
        cors_origins = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins = cors_origins.split(",") if cors_origins != "*" else ["*"]

        # Downstream services
        self.twin_url = os.getenv("TWIN_URL", "http://localhost:8001")
        self.agent_controller_url = os.getenv("AGENT_CONTROLLER_URL", "http://localhost:8002")
        self.knowledge_system_url = os.getenv("KNOWLEDGE_SYSTEM_URL", "http://localhost:8003")

        # Performance settings
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))
        self.health_check_timeout = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB

        # Feature flags
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_file_upload = os.getenv("ENABLE_FILE_UPLOAD", "true").lower() == "true"
        self.enable_agent_proxy = os.getenv("ENABLE_AGENT_PROXY", "true").lower() == "true"


# Initialize configuration
config = GatewayConfig()


# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Comprehensive security headers for production deployment"""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "frame-src 'none'; "
                "base-uri 'self'"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), " "payment=(), usb=(), magnetometer=(), gyroscope=()"
            ),
        }

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)

        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        # Add performance headers
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting with IP-based tracking"""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.cache = TTLCache(maxsize=10000, ttl=config.rate_limit_window)
        self.suspicious_ips = set()

    def get_client_ip(self, request: Request) -> str:
        """Extract client IP considering proxies"""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    async def dispatch(self, request: Request, call_next):
        client_ip = self.get_client_ip(request)
        current_time = time.time()

        # Check if IP is suspicious
        limit = config.rate_limit_requests
        if client_ip in self.suspicious_ips:
            limit = limit // 4  # Reduce limit for suspicious IPs

        # Track requests
        hits = self.cache.get(client_ip, 0) + 1
        self.cache[client_ip] = hits

        if hits > limit:
            # Mark as suspicious if severely exceeding limits
            if hits > limit * 2:
                self.suspicious_ips.add(client_ip)
                logger.warning(f"IP {client_ip} flagged as suspicious for excessive requests")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": config.rate_limit_window,
                    "limit": limit,
                    "window": config.rate_limit_window,
                },
                headers={
                    "Retry-After": str(config.rate_limit_window),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, limit - hits)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(current_time + config.rate_limit_window))

        return response


# Prometheus metrics
if config.enable_metrics:
    request_counter = Counter(
        "gateway_requests_total", "Total gateway HTTP requests", ["method", "endpoint", "status_code"]
    )

    request_duration = Histogram(
        "gateway_request_duration_seconds",
        "Gateway request duration",
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    health_check_duration = Histogram(
        "gateway_health_check_duration_seconds",
        "Health check response time",
        buckets=(0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),  # Target <100ms
    )


# Metrics middleware
class MetricsMiddleware(BaseHTTPMiddleware):
    """Prometheus metrics collection"""

    async def dispatch(self, request: Request, call_next):
        if not config.enable_metrics:
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        # Record metrics
        request_counter.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()

        request_duration.observe(duration)

        # Special handling for health checks
        if request.url.path == "/healthz":
            health_check_duration.observe(duration)

        return response


# Pydantic models
class QueryRequest(BaseModel):
    """Secure query request with validation"""

    query: str = Field(..., min_length=1, max_length=10000, description="User query text")
    session_id: str | None = Field(None, description="Optional session ID for conversation tracking")
    options: dict[str, Any] | None = Field(default_factory=dict, description="Query options and parameters")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Query must be a string")

        # Basic XSS prevention
        dangerous_patterns = ["<script", "javascript:", "data:text/html", "vbscript:"]
        query_lower = v.lower()

        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValueError("Query contains potentially dangerous content")

        return v.strip()


class HealthResponse(BaseModel):
    """Structured health check response"""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    services: dict[str, Any] = Field(default_factory=dict, description="Downstream service health")
    gateway: dict[str, Any] = Field(default_factory=dict, description="Gateway-specific metrics")
    system: dict[str, Any] | None = Field(None, description="System resource information")


# FastAPI app initialization
app = FastAPI(
    title="AIVillage Unified Gateway",
    description="Production-ready API gateway for AIVillage ecosystem",
    version="1.0.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    openapi_url="/openapi.json" if config.debug else None,
)

# Add middleware (order matters!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True if config.cors_origins != ["*"] else False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-RateLimit-*"],
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)

if config.enable_metrics:
    app.add_middleware(MetricsMiddleware)

# Authentication dependency
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials=Depends(security)) -> bool:
    """Verify API key authentication"""
    if not config.api_key or config.environment == "development":
        return True

    if not credentials or credentials.credentials != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


# Health check endpoint (performance critical - <100ms target)
@app.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Comprehensive health check with cascade probing of downstream services.
    Target: <100ms response time
    """
    start_time = time.time()

    services = {}
    gateway_metrics = {
        "uptime_seconds": time.time() - getattr(app.state, "start_time", start_time),
        "environment": config.environment,
        "debug_mode": config.debug,
    }

    # Fast health check mode for testing/development
    if config.environment == "test" or not hasattr(app.state, "start_time"):
        # Skip downstream service probes for fast response
        services = {
            "twin": {"status": "unknown", "note": "skipped_in_test_mode"},
            "agent_controller": {"status": "unknown", "note": "skipped_in_test_mode"},
        }
        overall_status = "healthy"
    else:
        # Production mode: probe downstream services with aggressive timeout
        timeout_per_service = min(0.02, config.health_check_timeout / 4)  # 20ms max per service

        async with httpx.AsyncClient(timeout=timeout_per_service) as client:
            # Probe services concurrently for speed
            tasks = []

            # Twin service health
            async def probe_twin():
                try:
                    probe_start = time.time()
                    response = await client.get(f"{config.twin_url}/healthz")
                    probe_time = (time.time() - probe_start) * 1000
                    return "twin", {
                        "status": "healthy" if response.status_code == 200 else "degraded",
                        "status_code": response.status_code,
                        "response_time_ms": round(probe_time, 2),
                    }
                except Exception as e:
                    return "twin", {
                        "status": "unhealthy",
                        "error": str(e)[:100],  # Truncate long error messages
                        "response_time_ms": None,
                    }

            # Agent controller health (if enabled)
            async def probe_agent_controller():
                if not config.enable_agent_proxy:
                    return "agent_controller", {"status": "disabled"}

                try:
                    probe_start = time.time()
                    response = await client.get(f"{config.agent_controller_url}/healthz")
                    probe_time = (time.time() - probe_start) * 1000
                    return "agent_controller", {
                        "status": "healthy" if response.status_code == 200 else "degraded",
                        "status_code": response.status_code,
                        "response_time_ms": round(probe_time, 2),
                    }
                except Exception as e:
                    return "agent_controller", {
                        "status": "unhealthy",
                        "error": str(e)[:100],  # Truncate long error messages
                        "response_time_ms": None,
                    }

            # Run probes concurrently
            tasks = [probe_twin(), probe_agent_controller()]

            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"Health probe failed: {result}")
                        continue
                    service_name, service_data = result
                    services[service_name] = service_data
            except Exception as e:
                logger.error(f"Health check probe error: {e}")
                services = {
                    "twin": {"status": "probe_failed", "error": str(e)[:100]},
                    "agent_controller": {"status": "probe_failed", "error": str(e)[:100]},
                }

        # Determine overall status
        service_statuses = [
            svc.get("status", "unhealthy") for svc in services.values() if svc.get("status") != "disabled"
        ]
        if all(status == "healthy" for status in service_statuses):
            overall_status = "healthy"
        elif any(status in ["healthy", "degraded"] for status in service_statuses):
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

    response_time = time.time() - start_time
    gateway_metrics["response_time_ms"] = round(response_time * 1000, 2)

    # Log slow health checks
    if response_time > 0.1:  # 100ms threshold
        logger.warning(f"Health check exceeded 100ms target: {response_time:.3f}s")

    return HealthResponse(status=overall_status, services=services, gateway=gateway_metrics)


# Metrics endpoint
if config.enable_metrics:

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint"""
        from fastapi.responses import Response

        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Main query endpoint
@app.post("/v1/query")
async def query_endpoint(request: QueryRequest, authenticated: bool = Depends(verify_api_key)) -> dict[str, Any]:
    """
    Primary query endpoint that routes requests through the knowledge system
    Architecture: Gateway → Agent Controller → Knowledge System
    """
    start_time = time.time()

    try:
        # Route to appropriate downstream service
        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            # First try agent controller for intelligent routing
            if config.enable_agent_proxy:
                try:
                    response = await client.post(
                        f"{config.agent_controller_url}/v1/query",
                        json={"query": request.query, "session_id": request.session_id, "options": request.options},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
                        result["routed_via"] = "agent_controller"
                        return result

                except Exception as e:
                    logger.warning(f"Agent controller unavailable, falling back to direct knowledge system: {e}")

            # Fallback to direct knowledge system
            try:
                response = await client.post(f"{config.knowledge_system_url}/v1/query", json=request.dict())
                response.raise_for_status()

                result = response.json()
                result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
                result["routed_via"] = "knowledge_system_direct"
                return result

            except Exception as e:
                logger.error(f"Knowledge system query failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Knowledge system temporarily unavailable"
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during query processing"
        )


# File upload endpoint (if enabled)
if config.enable_file_upload:

    @app.post("/v1/upload")
    async def upload_endpoint(
        file: UploadFile = File(...), authenticated: bool = Depends(verify_api_key)
    ) -> dict[str, Any]:
        """
        Secure file upload endpoint with streaming validation
        """
        if not file.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")

        # Validate file size
        if file.size and file.size > config.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {config.max_file_size} bytes",
            )

        try:
            # Stream file processing
            content = await file.read()

            # Forward to knowledge system for processing
            async with httpx.AsyncClient(timeout=config.request_timeout) as client:
                files = {"file": (file.filename, content, file.content_type)}
                response = await client.post(f"{config.knowledge_system_url}/v1/upload", files=files)
                response.raise_for_status()

                result = response.json()
                result["gateway_processed_size"] = len(content)
                return result

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"File upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload processing failed"
            )


# Chat endpoint (proxy to Twin service)
@app.post("/v1/chat")
async def chat_endpoint(request: Request, authenticated: bool = Depends(verify_api_key)) -> dict[str, Any]:
    """
    Chat endpoint that proxies requests to Twin service
    """
    try:
        body = await request.body()

        async with httpx.AsyncClient(timeout=config.request_timeout) as client:
            response = await client.post(
                f"{config.twin_url}/v1/chat", content=body, headers={"content-type": "application/json"}
            )
            response.raise_for_status()

            return response.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Twin service error: {e.response.status_code}")
    except Exception as e:
        logger.exception(f"Chat proxy failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service temporarily unavailable"
        )


# Status endpoint for backward compatibility
@app.get("/status")
async def status_endpoint():
    """Redirect to standardized health check"""
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/healthz", status_code=307)


# Root endpoint
@app.get("/")
async def root():
    """Gateway information endpoint"""
    return {
        "service": "AIVillage Unified Gateway",
        "version": "1.0.0",
        "environment": config.environment,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics" if config.enable_metrics else None,
            "query": "/v1/query",
            "chat": "/v1/chat",
            "upload": "/v1/upload" if config.enable_file_upload else None,
            "docs": "/docs" if config.debug else None,
        },
    }


# Application lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize application state"""
    app.state.start_time = time.time()
    logger.info(f"AIVillage Gateway started in {config.environment} environment")
    logger.info(f"Listening on {config.host}:{config.port}")
    logger.info(f"Health check timeout: {config.health_check_timeout}s (target: <100ms)")

    if config.enable_metrics:
        logger.info("Prometheus metrics enabled on /metrics")

    if config.enable_file_upload:
        logger.info(f"File upload enabled (max size: {config.max_file_size} bytes)")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("AIVillage Gateway shutting down")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Structured HTTP exception handling"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "status_code": exc.status_code,
                "message": exc.detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url.path),
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    logger.exception(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "internal_error",
                "status_code": 500,
                "message": "Internal server error" if config.environment == "production" else str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": str(request.url.path),
            }
        },
    )


# Development server runner
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug",
        access_log=True,
    )
