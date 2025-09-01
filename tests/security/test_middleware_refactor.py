"""Comprehensive Security Middleware Test Suite for Refactoring.

Tests for the extracted security middleware classes including:
- Security headers middleware
- Localhost validation middleware 
- Audit logging middleware
- Rate limiting middleware
- MFA enforcement middleware
"""

import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedSecureAPIServer,
)
from infrastructure.shared.security.redis_session_manager import DeviceInfo


class TestSecurityHeadersMiddleware:
    """Test security headers middleware functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.tls_enabled = True

    @pytest.mark.asyncio
    async def test_security_headers_application(self):
        """Test that security headers are properly applied."""
        # Mock request and handler
        request = make_mocked_request("GET", "/test", headers={"User-Agent": "Test Browser"})
        request.remote = "192.168.1.100"

        async def mock_handler(request):
            return web.Response(text="Success")

        # Apply security middleware
        response = await self.server._security_middleware(request, mock_handler)

        # Verify security headers
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "X-Security-Level": "B+",
            "X-Encryption-Algorithm": "AES-256-GCM",
        }

        for header, value in expected_headers.items():
            assert response.headers.get(header) == value

    @pytest.mark.asyncio
    async def test_hsts_header_with_tls(self):
        """Test HSTS header when TLS is enabled."""
        request = make_mocked_request("GET", "/test")
        request.remote = "127.0.0.1"

        async def mock_handler(request):
            return web.Response(text="Success")

        # Enable TLS
        self.server.tls_enabled = True

        response = await self.server._security_middleware(request, mock_handler)

        # Verify HSTS header
        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
        assert "includeSubDomains" in response.headers["Strict-Transport-Security"]
        assert "preload" in response.headers["Strict-Transport-Security"]

    @pytest.mark.asyncio
    async def test_security_logging(self):
        """Test security event logging."""
        request = make_mocked_request("GET", "/sensitive", headers={"User-Agent": "Suspicious Browser"})
        request.remote = "192.168.1.200"

        async def mock_handler(request):
            return web.Response(text="Success")

        with patch("infrastructure.shared.security.enhanced_secure_api_server.logger") as mock_logger:
            await self.server._security_middleware(request, mock_handler)

            # Verify security logging
            mock_logger.info.assert_called_once()
            log_call_args = mock_logger.info.call_args[0][0]
            assert "192.168.1.200" in log_call_args
            assert "GET" in log_call_args
            assert "/sensitive" in log_call_args


class TestLocalhostValidationMiddleware:
    """Test localhost validation middleware."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_localhost_admin_access_allowed(self):
        """Test that localhost access to admin endpoints is allowed."""
        # Mock localhost request
        request = make_mocked_request("GET", "/admin/status")
        request.remote = "127.0.0.1"

        # Mock user context
        request["user"] = {"user_id": "admin_user", "roles": ["admin"]}

        async def mock_handler(request):
            return web.Response(text="Admin Success")

        # Should pass through without blocking
        response = await self.server._auth_middleware(request, mock_handler)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_remote_admin_access_with_proper_auth(self):
        """Test remote admin access requires proper authentication."""
        # Mock remote request
        request = make_mocked_request("GET", "/admin/users")
        request.remote = "192.168.1.100"
        request.headers = {"Authorization": "Bearer valid_admin_token"}

        # Mock user context with admin role
        request["user"] = {"user_id": "admin_user", "roles": ["admin"], "mfa_verified": True}

        async def mock_handler(request):
            return web.Response(text="Admin Success")

        # Should require additional validation for remote admin access
        with patch.object(self.server, "authenticator") as mock_auth:
            mock_auth.verify_token_with_session = AsyncMock(
                return_value={"user_id": "admin_user", "roles": ["admin"], "mfa_verified": True}
            )

            response = await self.server._auth_middleware(request, mock_handler)
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_localhost_validation_security_bypass_protection(self):
        """Test protection against localhost validation bypass attempts."""
        # Test various localhost spoofing attempts
        spoofed_requests = [
            ("127.0.0.1", {"X-Forwarded-For": "192.168.1.100"}),
            ("127.0.0.1", {"X-Real-IP": "10.0.0.50"}),
            ("::1", {"X-Forwarded-For": "192.168.1.100"}),
        ]

        for ip, headers in spoofed_requests:
            request = make_mocked_request("GET", "/admin/emergency", headers=headers)
            request.remote = ip

            async def mock_handler(request):
                return web.Response(text="Admin Success")

            # Should validate true origin, not forwarded headers
            response = await self.server._auth_middleware(request, mock_handler)

            # Without proper admin token, should be blocked
            assert response.status == 401


class TestAuditLoggingMiddleware:
    """Test audit logging middleware."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_sensitive_endpoint_audit_logging(self):
        """Test audit logging for sensitive endpoints."""
        request = make_mocked_request("POST", "/profiles/user123/export")
        request.remote = "192.168.1.50"
        request["user"] = {"user_id": "user123", "roles": ["user"]}

        async def mock_handler(request):
            return web.Response(text="Export completed")

        with patch("infrastructure.shared.security.enhanced_secure_api_server.logger") as mock_logger:
            await self.server._security_middleware(request, mock_handler)

            # Verify audit logging occurred
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_failed_authentication_audit_logging(self):
        """Test audit logging for failed authentication attempts."""
        request = make_mocked_request("POST", "/auth/login")
        request.remote = "192.168.1.200"

        # Mock failed authentication
        mock_auth_data = json.dumps({"username": "attacker", "password": "wrong_password"}).encode()

        async def mock_read():
            return mock_auth_data

        request.read = mock_read

        async def mock_handler(request):
            return web.json_response({"error": "Invalid credentials"}, status=401)

        with patch("infrastructure.shared.security.enhanced_secure_api_server.logger") as mock_logger:
            response = await self.server._enhanced_login(request)

            # Verify failed authentication was logged
            assert response.status == 401
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_admin_action_audit_logging(self):
        """Test comprehensive audit logging for admin actions."""
        request = make_mocked_request("DELETE", "/admin/users/user123")
        request.remote = "127.0.0.1"
        request["user"] = {"user_id": "admin_user", "roles": ["admin"]}

        async def mock_handler(request):
            return web.Response(text="User deleted")

        with patch("infrastructure.shared.security.enhanced_secure_api_server.logger") as mock_logger:
            await self.server._security_middleware(request, mock_handler)

            # Verify admin action logging
            mock_logger.info.assert_called()
            log_message = mock_logger.info.call_args[0][0]
            assert "admin_user" in log_message or "DELETE" in log_message


class TestRateLimitingMiddleware:
    """Test rate limiting middleware."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.rate_limits = {}  # Reset rate limits

    @pytest.mark.asyncio
    async def test_authenticated_user_rate_limiting(self):
        """Test rate limiting for authenticated users."""
        request = make_mocked_request("GET", "/api/data")
        request.remote = "192.168.1.100"
        request["user"] = {"user_id": "test_user"}

        async def mock_handler(request):
            return web.Response(text="Success")

        # First 120 requests should pass (authenticated user limit)
        for i in range(120):
            response = await self.server._rate_limit_middleware(request, mock_handler)
            assert response.status == 200

        # 121st request should be rate limited
        response = await self.server._rate_limit_middleware(request, mock_handler)
        assert response.status == 429
        assert "Rate limit exceeded" in response.text

    @pytest.mark.asyncio
    async def test_unauthenticated_user_rate_limiting(self):
        """Test rate limiting for unauthenticated users."""
        request = make_mocked_request("GET", "/api/public")
        request.remote = "192.168.1.200"
        # No user context = unauthenticated

        async def mock_handler(request):
            return web.Response(text="Success")

        # First 60 requests should pass (unauthenticated limit)
        for i in range(60):
            response = await self.server._rate_limit_middleware(request, mock_handler)
            assert response.status == 200

        # 61st request should be rate limited
        response = await self.server._rate_limit_middleware(request, mock_handler)
        assert response.status == 429

    @pytest.mark.asyncio
    async def test_rate_limit_reset_after_window(self):
        """Test rate limit reset after time window."""
        request = make_mocked_request("GET", "/api/test")
        request.remote = "127.0.0.1"

        async def mock_handler(request):
            return web.Response(text="Success")

        # Fill rate limit
        for i in range(60):
            await self.server._rate_limit_middleware(request, mock_handler)

        # Should be rate limited
        response = await self.server._rate_limit_middleware(request, mock_handler)
        assert response.status == 429

        # Mock time passage (61 seconds later)
        with patch("time.time", return_value=time.time() + 61):
            # Should be allowed again
            response = await self.server._rate_limit_middleware(request, mock_handler)
            assert response.status == 200

    @pytest.mark.asyncio
    async def test_health_check_rate_limit_bypass(self):
        """Test that health check endpoints bypass rate limiting."""
        request = make_mocked_request("GET", "/health")
        request.remote = "192.168.1.100"

        async def mock_handler(request):
            return web.Response(text="Healthy")

        # Health checks should always pass regardless of rate limits
        for i in range(200):  # Way over normal limits
            response = await self.server._rate_limit_middleware(request, mock_handler)
            assert response.status == 200


class TestMFAEnforcementMiddleware:
    """Test MFA enforcement middleware."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_mfa_required_for_sensitive_operations(self):
        """Test MFA requirement for sensitive operations."""
        # Test sensitive endpoints that require MFA
        sensitive_endpoints = ["/profiles/user123/export", "/admin/users", "/profiles/user123/delete"]

        for endpoint in sensitive_endpoints:
            request = make_mocked_request("GET", endpoint)
            request["user"] = {"user_id": "test_user", "mfa_verified": False}

            async def mock_handler(request):
                return web.Response(text="Sensitive operation")

            response = await self.server._mfa_middleware(request, mock_handler)

            # Should require MFA
            assert response.status == 403
            response_data = json.loads(response.text)
            assert response_data["mfa_required"] is True

    @pytest.mark.asyncio
    async def test_mfa_verified_user_access(self):
        """Test access for MFA-verified users."""
        request = make_mocked_request("GET", "/profiles/user123/export")
        request["user"] = {"user_id": "test_user", "mfa_verified": True}

        async def mock_handler(request):
            return web.Response(text="Export completed")

        response = await self.server._mfa_middleware(request, mock_handler)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_mfa_token_header_validation(self):
        """Test MFA token header validation."""
        request = make_mocked_request("GET", "/admin/critical")
        request.headers = {"X-MFA-Token": "123456"}
        request["user"] = {"user_id": "admin_user", "mfa_verified": False}

        async def mock_handler(request):
            return web.Response(text="Critical operation")

        # With MFA token header, should pass to handler
        response = await self.server._mfa_middleware(request, mock_handler)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_public_endpoint_mfa_bypass(self):
        """Test that public endpoints bypass MFA requirements."""
        public_endpoints = ["/health", "/auth/login", "/auth/register", "/auth/mfa/setup", "/auth/mfa/verify"]

        for endpoint in public_endpoints:
            request = make_mocked_request("GET", endpoint)
            request["user"] = {"user_id": "test_user", "mfa_verified": False}

            async def mock_handler(request):
                return web.Response(text="Public access")

            response = await self.server._mfa_middleware(request, mock_handler)
            assert response.status == 200


class TestMiddlewareIntegration:
    """Test integration between middleware components."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_middleware_chain_execution_order(self):
        """Test that middleware executes in correct order."""
        request = make_mocked_request("GET", "/api/test")
        request.remote = "192.168.1.100"

        execution_order = []

        async def tracking_security_middleware(request, handler):
            execution_order.append("security")
            return await self.server._security_middleware(request, handler)

        async def tracking_rate_limit_middleware(request, handler):
            execution_order.append("rate_limit")
            return await self.server._rate_limit_middleware(request, handler)

        async def mock_handler(request):
            execution_order.append("handler")
            return web.Response(text="Success")

        # Chain middleware manually to test order
        await tracking_security_middleware(request, lambda req: tracking_rate_limit_middleware(req, mock_handler))

        assert execution_order == ["security", "rate_limit", "handler"]

    @pytest.mark.asyncio
    async def test_middleware_error_propagation(self):
        """Test error propagation through middleware chain."""
        request = make_mocked_request("GET", "/api/error")
        request.remote = "192.168.1.100"

        async def error_handler(request):
            raise ValueError("Test error")

        # Should propagate errors properly
        with pytest.raises(ValueError):
            await self.server._security_middleware(request, error_handler)

    @pytest.mark.asyncio
    async def test_middleware_request_modification(self):
        """Test that middleware can modify request context."""
        request = make_mocked_request("GET", "/api/test")
        request.remote = "127.0.0.1"

        # Session middleware should add device info
        DeviceInfo("Test Browser", "127.0.0.1")

        async def mock_handler(request):
            # Check if device info was added
            assert "device_info" in request
            return web.Response(text="Success")

        await self.server._session_middleware(request, mock_handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
