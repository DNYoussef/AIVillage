"""
Security Middleware for AIVillage Gateway Service

This module provides comprehensive security middleware including:
- CORS configuration
- CSP (Content Security Policy) headers
- Input validation and sanitization
- Rate limiting enhancements
- Security headers
- Request/response filtering
"""

import json
import logging
import re
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add comprehensive security headers."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.security_headers = {
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # XSS protection (legacy browsers)
            "X-XSS-Protection": "1; mode=block",
            # Force HTTPS (configure based on environment)
            "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",
            # Content Security Policy
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "connect-src 'self'; "
                "font-src 'self'; "
                "object-src 'none'; "
                "media-src 'self'; "
                "frame-src 'none'; "
                "base-uri 'self'"
            ),
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Feature policy (experimental)
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "speaker=()"
            ),
        }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add all security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value

        # Add request processing time
        if hasattr(request.state, "start_time"):
            process_time = time.time() - request.state.start_time
            response.headers["X-Process-Time"] = str(process_time)

        return response


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive input validation and sanitization."""

    def __init__(self, app: FastAPI):
        super().__init__(app)

        # Dangerous patterns to detect
        self.sql_injection_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b)",
            r"(\binsert\b.*\binto\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(\bupdate\b.*\bset\b)",
            r"(--|\/\*|\*\/)",
            r"(\bor\b.*=.*\bor\b)",
            r"(1=1|1\s*=\s*1)",
        ]

        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe",
            r"<object",
            r"<embed",
            r"<link",
            r"<meta",
        ]

        self.command_injection_patterns = [
            r"[;&|`$(){}[\]]",
            r"(wget|curl|nc|netcat|bash|sh|cmd|powershell)",
        ]

        # Compile patterns for performance
        self.compiled_sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.sql_injection_patterns]
        self.compiled_xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.xss_patterns]
        self.compiled_cmd_patterns = [re.compile(p, re.IGNORECASE) for p in self.command_injection_patterns]

    def _scan_for_threats(self, text: str) -> list[str]:
        """Scan text for security threats."""
        threats = []

        # SQL Injection detection
        for pattern in self.compiled_sql_patterns:
            if pattern.search(text):
                threats.append("sql_injection")
                break

        # XSS detection
        for pattern in self.compiled_xss_patterns:
            if pattern.search(text):
                threats.append("xss")
                break

        # Command injection detection
        for pattern in self.compiled_cmd_patterns:
            if pattern.search(text):
                threats.append("command_injection")
                break

        return threats

    def _validate_json_payload(self, payload: dict[str, Any]) -> list[str]:
        """Recursively validate JSON payload for threats."""
        threats = []

        def scan_value(value):
            if isinstance(value, str):
                threats.extend(self._scan_for_threats(value))
            elif isinstance(value, dict):
                for v in value.values():
                    scan_value(v)
            elif isinstance(value, list):
                for item in value:
                    scan_value(item)

        scan_value(payload)
        return threats

    async def dispatch(self, request: Request, call_next):
        # Record start time for processing
        request.state.start_time = time.time()

        # Validate request headers
        user_agent = request.headers.get("user-agent", "")
        if len(user_agent) > 500:  # Unusually long user agent
            logger.warning(f"Suspicious user agent length: {len(user_agent)}")

        # Validate content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Request entity too large")

        # Validate query parameters
        for key, value in request.query_params.items():
            if len(str(value)) > 1000:  # Limit query param length
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Query parameter '{key}' too long")

            threats = self._scan_for_threats(str(value))
            if threats:
                logger.warning(f"Potential security threat in query param '{key}': {threats}")
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request format")

        # For POST/PUT requests, validate JSON payload
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            if "application/json" in content_type:
                try:
                    body = await request.body()
                    if body:
                        payload = json.loads(body)
                        threats = self._validate_json_payload(payload)
                        if threats:
                            logger.warning(f"Potential security threats in payload: {threats}")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid request format"
                            )

                        # Re-create request with validated body
                        async def receive():
                            return {"type": "http.request", "body": body}

                        request._receive = receive

                except json.JSONDecodeError:
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON format")

        response = await call_next(request)
        return response


class EnhancedRateLimitMiddleware(BaseHTTPMiddleware):
    """Enhanced rate limiting with multiple tiers and threat detection."""

    def __init__(self, app: FastAPI):
        super().__init__(app)

        # Rate limits per tier (requests per minute)
        self.rate_limits = {
            "default": 60,  # Default users
            "authenticated": 100,  # Authenticated users
            "premium": 200,  # Premium users
            "suspicious": 10,  # Flagged IPs
        }

        # Storage for rate limiting (in production, use Redis)
        self.request_counts = {}
        self.suspicious_ips = set()
        self.last_cleanup = time.time()

    def _cleanup_old_requests(self):
        """Clean up old request data (runs every 5 minutes)."""
        now = time.time()
        if now - self.last_cleanup > 300:  # 5 minutes
            cutoff = now - 3600  # Keep last hour of data

            to_remove = []
            for key, timestamps in self.request_counts.items():
                # Remove old timestamps
                self.request_counts[key] = [ts for ts in timestamps if ts > cutoff]
                if not self.request_counts[key]:
                    to_remove.append(key)

            # Remove empty entries
            for key in to_remove:
                del self.request_counts[key]

            self.last_cleanup = now

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering proxies."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _get_rate_limit_tier(self, request: Request, client_ip: str) -> tuple[str, int]:
        """Determine rate limit tier for the request."""

        # Check if IP is suspicious
        if client_ip in self.suspicious_ips:
            return "suspicious", self.rate_limits["suspicious"]

        # Check for authentication (API key, JWT, etc.)
        auth_header = request.headers.get("authorization")
        api_key = request.query_params.get("api_key")

        if auth_header or api_key:
            # In production, validate the token/key here
            # For now, assume authenticated users get higher limits
            return "authenticated", self.rate_limits["authenticated"]

        return "default", self.rate_limits["default"]

    async def dispatch(self, request: Request, call_next):
        self._cleanup_old_requests()

        client_ip = self._get_client_ip(request)
        tier, limit = self._get_rate_limit_tier(request, client_ip)

        # Create unique key for this client
        key = f"{client_ip}:{tier}"
        now = time.time()

        # Initialize or get existing request timestamps
        if key not in self.request_counts:
            self.request_counts[key] = []

        # Remove requests older than 1 minute
        minute_ago = now - 60
        self.request_counts[key] = [ts for ts in self.request_counts[key] if ts > minute_ago]

        # Check if rate limit exceeded
        if len(self.request_counts[key]) >= limit:
            # Flag IP as suspicious if severely exceeding limits
            if len(self.request_counts[key]) > limit * 2:
                self.suspicious_ips.add(client_ip)
                logger.warning(f"IP {client_ip} flagged as suspicious for excessive requests")

            retry_after = 60 - (now - self.request_counts[key][0])

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": int(retry_after),
                    "limit": limit,
                    "window": 60,
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + retry_after)),
                },
            )

        # Record this request
        self.request_counts[key].append(now)

        response = await call_next(request)

        # Add rate limit headers
        remaining = max(0, limit - len(self.request_counts[key]))
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + 60))

        return response


def configure_cors(app: FastAPI, environment: str = "development"):
    """Configure CORS middleware with environment-specific settings."""

    if environment == "production":
        # Strict CORS for production
        allowed_origins = ["https://yourdomain.com", "https://api.yourdomain.com", "https://app.yourdomain.com"]
        allow_credentials = True
        allow_methods = ["GET", "POST", "PUT", "DELETE"]
        allow_headers = ["Authorization", "Content-Type", "X-Requested-With"]

    elif environment == "staging":
        # Moderate CORS for staging
        allowed_origins = [
            "https://staging.yourdomain.com",
            "https://dev.yourdomain.com",
            "http://localhost:3000",
            "http://localhost:8080",
        ]
        allow_credentials = True
        allow_methods = ["*"]
        allow_headers = ["*"]

    else:
        # Permissive CORS for development
        allowed_origins = ["*"]
        allow_credentials = False  # Can't use credentials with wildcard origins
        allow_methods = ["*"]
        allow_headers = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=["X-Process-Time", "X-RateLimit-*"],
    )


def add_security_middleware(app: FastAPI, environment: str = "development"):
    """Add all security middleware to the FastAPI application."""

    # Configure CORS first (order matters)
    configure_cors(app, environment)

    # Add security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Add input validation
    app.add_middleware(InputValidationMiddleware)

    # Add enhanced rate limiting
    app.add_middleware(EnhancedRateLimitMiddleware)

    logger.info(f"Security middleware configured for {environment} environment")


class SecurityConfig:
    """Security configuration class."""

    def __init__(self):
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_query_param_length = 1000
        self.max_header_length = 8192
        self.rate_limit_default = 60
        self.rate_limit_authenticated = 100
        self.rate_limit_premium = 200
        self.suspicious_threshold_multiplier = 2

        # CSP configuration
        self.csp_default_src = ["'self'"]
        self.csp_script_src = ["'self'", "'unsafe-inline'"]
        self.csp_style_src = ["'self'", "'unsafe-inline'"]
        self.csp_img_src = ["'self'", "data:", "https:"]

        # HSTS configuration
        self.hsts_max_age = 63072000  # 2 years
        self.hsts_include_subdomains = True
        self.hsts_preload = True


# Example usage for integration
def create_secure_app(environment: str = "development") -> FastAPI:
    """Create a FastAPI app with security middleware pre-configured."""

    app = FastAPI(
        title="Secure AIVillage Gateway",
        description="AIVillage Gateway with comprehensive security middleware",
        version="1.0.0",
        docs_url="/docs" if environment != "production" else None,
        redoc_url="/redoc" if environment != "production" else None,
    )

    # Add security middleware
    add_security_middleware(app, environment)

    return app


if __name__ == "__main__":
    # Example usage
    app = create_secure_app("development")

    @app.get("/test")
    async def test_endpoint():
        return {"message": "Security middleware is active!"}

    # Run with: uvicorn security_middleware:app --reload
