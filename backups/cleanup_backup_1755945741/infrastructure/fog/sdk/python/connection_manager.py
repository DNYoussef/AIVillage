"""
Connection Management for Fog Client

Handles HTTP connections, authentication, and request/response processing.
Focused on network communication concerns following single responsibility.
"""

import logging
from typing import Any

import aiohttp

from .client_types import (
    AuthenticationError,
    FogClientError,
    JobNotFoundError,
    QuotaExceededError,
    SandboxNotFoundError,
)

logger = logging.getLogger(__name__)


class AuthenticationConfig:
    """Configuration for authentication - immutable config object."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @property
    def headers(self) -> dict[str, str]:
        """Get authentication headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class ConnectionConfig:
    """Configuration for connection settings."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)


class HTTPConnectionManager:
    """
    Manages HTTP connections and request/response handling.

    Focuses solely on network communication concerns.
    Uses dependency injection for configuration.
    """

    def __init__(self, connection_config: ConnectionConfig, auth_config: AuthenticationConfig):
        self._connection_config = connection_config
        self._auth_config = auth_config
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=self._connection_config.timeout, headers=self._auth_config.headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
            self._session = None

    async def request(
        self, method: str, path: str, json_data: dict | None = None, params: dict | None = None
    ) -> dict[str, Any]:
        """
        Make authenticated HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (starting with /)
            json_data: JSON payload for request body
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            FogClientError: For various API and network errors
        """
        if not self._session:
            raise FogClientError("Connection manager must be used as async context manager")

        url = f"{self._connection_config.base_url}{path}"

        try:
            async with self._session.request(method, url, json=json_data, params=params) as response:
                await self._handle_http_errors(response, path)
                return await response.json()

        except aiohttp.ClientError as e:
            raise FogClientError(f"Request failed: {e}")

    async def _handle_http_errors(self, response: aiohttp.ClientResponse, path: str) -> None:
        """
        Handle HTTP error responses with appropriate exceptions.

        Args:
            response: HTTP response object
            path: Request path for context

        Raises:
            Specific FogClientError subclasses based on status code
        """
        if response.status < 400:
            return  # Success response

        # Map status codes to specific exceptions
        if response.status == 401:
            raise AuthenticationError("Invalid API key or unauthorized")
        elif response.status == 404:
            if "/jobs/" in path:
                raise JobNotFoundError("Job not found")
            elif "/sandboxes/" in path:
                raise SandboxNotFoundError("Sandbox not found")
            else:
                raise FogClientError("Resource not found")
        elif response.status == 429:
            raise QuotaExceededError("Rate limit or quota exceeded")
        else:
            error_text = await response.text()
            raise FogClientError(f"API error {response.status}: {error_text}")

    def is_connected(self) -> bool:
        """Check if connection manager is ready."""
        return self._session is not None and not self._session.closed
