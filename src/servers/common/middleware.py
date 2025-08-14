"""Reusable middleware and helpers for server implementations."""

from __future__ import annotations

# isort: skip_file

from collections import defaultdict
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Simple in-memory rate limiter using a sliding window."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Return True if the request is within limits."""
        now = time.time()
        window = self.requests[client_id]
        self.requests[client_id] = [t for t in window if now - t < self.window_seconds]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

    def get_stats(self, client_id: str) -> dict[str, Any]:
        """Return statistics for the given client."""
        now = time.time()
        window = [
            t for t in self.requests.get(client_id, []) if now - t < self.window_seconds
        ]
        remaining = max(0, self.max_requests - len(window))
        return {
            "requests": len(window),
            "remaining": remaining,
            "reset_time": now + self.window_seconds,
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware providing basic API key auth and rate limiting."""

    def __init__(
        self,
        app,
        api_key: str | None = None,
        rate_limiter: RateLimiter | None = None,
        protected_paths: tuple[str, ...] = ("/query", "/upload"),
    ) -> None:
        super().__init__(app)
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.protected_paths = protected_paths

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        client_ip = request.client.host if request.client else "unknown"
        if self.rate_limiter and request.url.path.startswith(self.protected_paths):
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429, content={"detail": "Rate limit exceeded"}
                )
        if self.api_key and request.url.path not in {"/", "/healthz", "/status"}:
            key = request.headers.get("x-api-key")
            if key != self.api_key:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        return await call_next(request)
