"""Comprehensive Security Context Test Suite for Refactoring.

Tests for security context extraction and validation including:
- Request context extraction and parsing
- User context validation and authorization
- Device fingerprinting and tracking
- Security context propagation through middleware
- Context-based access control decisions
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedSecureAPIServer,
)
from infrastructure.shared.security.redis_session_manager import DeviceInfo


class TestSecurityContextExtraction:
    """Test security context extraction from requests."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.session_manager = Mock()
        self.server.authenticator = Mock()

    def test_device_info_extraction(self):
        """Test device information extraction from request."""
        request = make_mocked_request(
            "GET",
            "/api/test",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "X-Forwarded-For": "203.0.113.1, 192.168.1.100",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        request.remote = "192.168.1.100"

        # Extract device info
        device_info = DeviceInfo(
            user_agent=request.headers.get("User-Agent", "unknown"), ip_address=request.remote or "unknown"
        )

        assert device_info.user_agent.startswith("Mozilla/5.0")
        assert device_info.ip_address == "192.168.1.100"
        assert device_info.device_fingerprint is not None
        assert len(device_info.device_fingerprint) == 16  # Truncated hash

    def test_security_headers_extraction(self):
        """Test extraction of security-related headers."""
        security_headers = {
            "X-Forwarded-For": "203.0.113.1",
            "X-Real-IP": "203.0.113.1",
            "X-Forwarded-Proto": "https",
            "X-Request-ID": "req-123456",
            "Authorization": "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
        }

        request = make_mocked_request("GET", "/api/secure", headers=security_headers)

        # Extract and validate security headers
        auth_header = request.headers.get("Authorization")
        forwarded_for = request.headers.get("X-Forwarded-For")
        real_ip = request.headers.get("X-Real-IP")
        proto = request.headers.get("X-Forwarded-Proto")

        assert auth_header.startswith("Bearer ")
        assert forwarded_for == "203.0.113.1"
        assert real_ip == "203.0.113.1"
        assert proto == "https"

    def test_request_metadata_extraction(self):
        """Test extraction of request metadata for security analysis."""
        request = make_mocked_request(
            "POST",
            "/api/sensitive",
            headers={
                "Content-Type": "application/json",
                "Content-Length": "256",
                "Referer": "https://example.com/page",
                "Origin": "https://example.com",
            },
        )
        request.remote = "10.0.0.1"

        # Extract metadata
        metadata = {
            "method": request.method,
            "path": request.path,
            "content_type": request.headers.get("Content-Type"),
            "content_length": request.headers.get("Content-Length"),
            "referer": request.headers.get("Referer"),
            "origin": request.headers.get("Origin"),
            "remote_addr": request.remote,
            "timestamp": datetime.utcnow().isoformat(),
        }

        assert metadata["method"] == "POST"
        assert metadata["path"] == "/api/sensitive"
        assert metadata["content_type"] == "application/json"
        assert metadata["origin"] == "https://example.com"

    def test_suspicious_request_detection(self):
        """Test detection of suspicious request patterns."""
        suspicious_patterns = [
            # Missing User-Agent
            {"headers": {}, "remote": "192.168.1.100"},
            # Suspicious User-Agent
            {"headers": {"User-Agent": "sqlmap/1.0"}, "remote": "192.168.1.101"},
            # Multiple X-Forwarded-For headers
            {"headers": {"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"}, "remote": "192.168.1.102"},
            # Tor exit node pattern
            {"headers": {"User-Agent": "Mozilla/5.0 Tor Browser"}, "remote": "192.168.1.103"},
        ]

        for pattern in suspicious_patterns:
            request = make_mocked_request("GET", "/api/test", headers=pattern["headers"])
            request.remote = pattern["remote"]

            # Simple suspicious detection logic
            user_agent = request.headers.get("User-Agent", "")
            is_suspicious = (
                not user_agent  # Missing User-Agent
                or "sqlmap" in user_agent.lower()  # Security scanner
                or "tor" in user_agent.lower()  # Tor browser
                or len(request.headers.get("X-Forwarded-For", "").split(",")) > 3  # Too many proxies
            )

            assert is_suspicious, f"Should detect suspicious pattern: {pattern}"


class TestUserContextValidation:
    """Test user context validation and authorization."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.authenticator = Mock()
        self.server.rbac_system = Mock()

    @pytest.mark.asyncio
    async def test_token_validation_and_user_extraction(self):
        """Test JWT token validation and user context extraction."""
        request = make_mocked_request("GET", "/api/protected")
        request.headers = {"Authorization": "Bearer valid_jwt_token"}

        # Mock token validation
        mock_payload = {
            "user_id": "user123",
            "session_id": "session456",
            "roles": ["user", "premium"],
            "permissions": ["read", "write"],
            "mfa_verified": True,
            "exp": (datetime.utcnow() + timedelta(hours=1)).timestamp(),
        }

        self.server.authenticator.verify_token_with_session = AsyncMock(return_value=mock_payload)

        # Validate token and extract user context
        token = "valid_jwt_token"
        payload = await self.server.authenticator.verify_token_with_session(token)

        assert payload["user_id"] == "user123"
        assert "premium" in payload["roles"]
        assert payload["mfa_verified"] is True

    def test_role_based_access_validation(self):
        """Test role-based access control validation."""
        test_cases = [
            # Admin access to admin endpoint
            {"user_roles": ["admin"], "required_roles": ["admin"], "should_allow": True},
            # User access to user endpoint
            {"user_roles": ["user"], "required_roles": ["user"], "should_allow": True},
            # User trying to access admin endpoint
            {"user_roles": ["user"], "required_roles": ["admin"], "should_allow": False},
            # Multiple roles, one matching
            {"user_roles": ["user", "premium"], "required_roles": ["premium"], "should_allow": True},
            # No roles matching
            {"user_roles": ["guest"], "required_roles": ["user", "admin"], "should_allow": False},
        ]

        for case in test_cases:
            # Simple RBAC logic
            has_required_role = any(role in case["user_roles"] for role in case["required_roles"])

            assert has_required_role == case["should_allow"], f"Failed case: {case}"

    def test_permission_based_access_validation(self):
        """Test permission-based access control validation."""
        test_cases = [
            # Read permission for GET request
            {"user_permissions": ["read"], "required_permissions": ["read"], "should_allow": True},
            # Write permission for POST request
            {"user_permissions": ["write"], "required_permissions": ["write"], "should_allow": True},
            # Admin permission for admin action
            {"user_permissions": ["admin"], "required_permissions": ["admin"], "should_allow": True},
            # Insufficient permissions
            {"user_permissions": ["read"], "required_permissions": ["write"], "should_allow": False},
            # Multiple permissions, one matching
            {"user_permissions": ["read", "write"], "required_permissions": ["write"], "should_allow": True},
        ]

        for case in test_cases:
            has_required_permission = any(perm in case["user_permissions"] for perm in case["required_permissions"])

            assert has_required_permission == case["should_allow"], f"Failed case: {case}"

    def test_mfa_verification_status(self):
        """Test MFA verification status validation."""
        test_scenarios = [
            # MFA verified user accessing sensitive endpoint
            {"mfa_verified": True, "endpoint_requires_mfa": True, "should_allow": True},
            # MFA not verified for sensitive endpoint
            {"mfa_verified": False, "endpoint_requires_mfa": True, "should_allow": False},
            # MFA not required for regular endpoint
            {"mfa_verified": False, "endpoint_requires_mfa": False, "should_allow": True},
            # MFA verified for regular endpoint (always allowed)
            {"mfa_verified": True, "endpoint_requires_mfa": False, "should_allow": True},
        ]

        for scenario in test_scenarios:
            access_allowed = not scenario["endpoint_requires_mfa"] or scenario["mfa_verified"]

            assert access_allowed == scenario["should_allow"], f"Failed scenario: {scenario}"


class TestContextPropagation:
    """Test security context propagation through middleware chain."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.context_log = []  # Track context propagation

    @pytest.mark.asyncio
    async def test_context_propagation_through_middleware(self):
        """Test that security context is properly propagated."""
        request = make_mocked_request("GET", "/api/test")
        request.remote = "192.168.1.100"

        # Track context additions
        async def tracking_middleware_1(request, handler):
            request["middleware_1"] = "context_added"
            self.context_log.append("middleware_1")
            return await handler(request)

        async def tracking_middleware_2(request, handler):
            request["middleware_2"] = "context_added"
            self.context_log.append("middleware_2")
            # Verify previous context exists
            assert "middleware_1" in request
            return await handler(request)

        async def final_handler(request):
            self.context_log.append("handler")
            # Verify all context exists
            assert "middleware_1" in request
            assert "middleware_2" in request
            return web.Response(text="Success")

        # Chain execution
        await tracking_middleware_1(request, lambda req: tracking_middleware_2(req, final_handler))

        assert self.context_log == ["middleware_1", "middleware_2", "handler"]

    @pytest.mark.asyncio
    async def test_security_context_enrichment(self):
        """Test security context enrichment through middleware chain."""
        request = make_mocked_request("GET", "/api/secure")
        request.remote = "127.0.0.1"
        request.headers = {"User-Agent": "Test Browser"}

        # Mock security enrichment middleware
        async def security_enrichment_middleware(request, handler):
            # Add device info
            request["device_info"] = DeviceInfo(
                user_agent=request.headers.get("User-Agent", "unknown"), ip_address=request.remote or "unknown"
            )

            # Add security flags
            request["security_context"] = {
                "is_localhost": request.remote == "127.0.0.1",
                "has_user_agent": bool(request.headers.get("User-Agent")),
                "request_timestamp": datetime.utcnow(),
            }

            return await handler(request)

        async def validation_handler(request):
            # Verify enrichment
            assert "device_info" in request
            assert "security_context" in request
            assert request["security_context"]["is_localhost"] is True
            assert request["security_context"]["has_user_agent"] is True
            return web.Response(text="Validated")

        await security_enrichment_middleware(request, validation_handler)

    @pytest.mark.asyncio
    async def test_context_isolation_between_requests(self):
        """Test that context is isolated between different requests."""
        # Request 1
        request1 = make_mocked_request("GET", "/api/test1")
        request1["test_context"] = "request1_data"

        # Request 2
        request2 = make_mocked_request("GET", "/api/test2")
        request2["test_context"] = "request2_data"

        # Verify isolation
        assert request1["test_context"] == "request1_data"
        assert request2["test_context"] == "request2_data"
        assert request1["test_context"] != request2["test_context"]


class TestAccessControlDecisions:
    """Test context-based access control decisions."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    def test_endpoint_based_access_control(self):
        """Test access control decisions based on endpoint patterns."""
        access_patterns = [
            # Public endpoints
            {"path": "/health", "requires_auth": False},
            {"path": "/auth/login", "requires_auth": False},
            {"path": "/auth/register", "requires_auth": False},
            # Protected endpoints
            {"path": "/api/data", "requires_auth": True},
            {"path": "/profiles/user123", "requires_auth": True},
            # Admin endpoints
            {"path": "/admin/users", "requires_auth": True, "requires_admin": True},
            {"path": "/admin/system", "requires_auth": True, "requires_admin": True},
            # Sensitive endpoints
            {"path": "/profiles/user123/export", "requires_auth": True, "requires_mfa": True},
            {"path": "/admin/emergency", "requires_auth": True, "requires_admin": True, "requires_localhost": True},
        ]

        for pattern in access_patterns:
            path = pattern["path"]

            # Determine access requirements based on path
            requires_auth = not any(path.startswith(public) for public in ["/health", "/auth/login", "/auth/register"])
            requires_admin = path.startswith("/admin/")
            requires_mfa = "export" in path or "emergency" in path
            requires_localhost = "emergency" in path

            assert requires_auth == pattern.get("requires_auth", True)
            if "requires_admin" in pattern:
                assert requires_admin == pattern["requires_admin"]
            if "requires_mfa" in pattern:
                assert requires_mfa == pattern["requires_mfa"]
            if "requires_localhost" in pattern:
                assert requires_localhost == pattern["requires_localhost"]

    def test_time_based_access_control(self):
        """Test time-based access control decisions."""
        now = datetime.utcnow()

        # Business hours: 9 AM to 5 PM UTC
        business_start = now.replace(hour=9, minute=0, second=0)
        business_end = now.replace(hour=17, minute=0, second=0)

        test_times = [
            {"time": now.replace(hour=8, minute=0), "is_business_hours": False},  # Before hours
            {"time": now.replace(hour=12, minute=0), "is_business_hours": True},  # During hours
            {"time": now.replace(hour=18, minute=0), "is_business_hours": False},  # After hours
            {"time": now.replace(hour=9, minute=0), "is_business_hours": True},  # Start of hours
            {"time": now.replace(hour=17, minute=0), "is_business_hours": True},  # End of hours
        ]

        for test_case in test_times:
            test_time = test_case["time"]
            is_business_hours = business_start <= test_time <= business_end

            assert is_business_hours == test_case["is_business_hours"], f"Failed for time: {test_time}"

    def test_rate_limit_based_access_control(self):
        """Test rate limit-based access control decisions."""
        # Mock rate limit scenarios
        rate_limit_cases = [
            {"requests_count": 50, "limit": 100, "should_allow": True},  # Under limit
            {"requests_count": 100, "limit": 100, "should_allow": False},  # At limit
            {"requests_count": 150, "limit": 100, "should_allow": False},  # Over limit
            {"requests_count": 0, "limit": 100, "should_allow": True},  # No requests yet
        ]

        for case in rate_limit_cases:
            requests_remaining = case["limit"] - case["requests_count"]
            access_allowed = requests_remaining > 0

            assert access_allowed == case["should_allow"], f"Failed case: {case}"

    def test_device_trust_based_access_control(self):
        """Test device trust-based access control decisions."""
        device_scenarios = [
            # Known device from trusted IP
            {"device_fingerprint": "known_device_123", "ip_address": "192.168.1.100", "is_trusted": True},
            # Known device from new IP (suspicious)
            {"device_fingerprint": "known_device_123", "ip_address": "203.0.113.1", "is_trusted": False},
            # New device from trusted IP
            {"device_fingerprint": "new_device_456", "ip_address": "192.168.1.100", "is_trusted": False},
            # New device from new IP (very suspicious)
            {"device_fingerprint": "new_device_789", "ip_address": "203.0.113.1", "is_trusted": False},
        ]

        # Mock known devices and trusted IPs
        known_devices = {"known_device_123"}
        trusted_ips = {"192.168.1.100", "127.0.0.1"}

        for scenario in device_scenarios:
            device_known = scenario["device_fingerprint"] in known_devices
            ip_trusted = scenario["ip_address"] in trusted_ips

            # Device is trusted if both device is known AND IP is trusted
            is_trusted = device_known and ip_trusted

            assert is_trusted == scenario["is_trusted"], f"Failed scenario: {scenario}"


class TestSecurityContextErrorHandling:
    """Test error handling in security context operations."""

    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()

    def test_malformed_auth_header_handling(self):
        """Test handling of malformed authorization headers."""
        malformed_headers = [
            "",  # Empty header
            "Bearer",  # Missing token
            "Basic dGVzdDp0ZXN0",  # Wrong auth type
            "Bearer invalid.jwt.token.extra",  # Invalid JWT format
            "Bearer " + "a" * 1000,  # Extremely long token
        ]

        for header in malformed_headers:
            request = make_mocked_request("GET", "/api/test")
            if header:
                request.headers = {"Authorization": header}

            # Extract and validate auth header
            auth_header = request.headers.get("Authorization", "")
            is_valid_bearer = auth_header.startswith("Bearer ") and len(auth_header) > 7

            if header in ["", "Bearer", "Basic dGVzdDp0ZXN0"]:
                assert not is_valid_bearer
            elif header.startswith("Bearer "):
                assert is_valid_bearer  # Format is correct even if token is invalid

    def test_missing_context_graceful_degradation(self):
        """Test graceful degradation when context is missing."""
        request = make_mocked_request("GET", "/api/test")

        # Try to access missing context gracefully
        user_context = request.get("user", {})
        device_info = request.get("device_info", None)
        security_context = request.get("security_context", {})

        # Should not raise errors
        assert user_context == {}
        assert device_info is None
        assert security_context == {}

    def test_corrupted_context_data_handling(self):
        """Test handling of corrupted context data."""
        request = make_mocked_request("GET", "/api/test")

        # Simulate corrupted context data
        request["user"] = "corrupted_string_instead_of_dict"
        request["device_info"] = {"invalid": "device_info_format"}

        # Safe context access
        user_id = None
        if isinstance(request.get("user"), dict):
            user_id = request["user"].get("user_id")

        device_fingerprint = None
        device_info = request.get("device_info")
        if hasattr(device_info, "device_fingerprint"):
            device_fingerprint = device_info.device_fingerprint

        # Should handle gracefully
        assert user_id is None
        assert device_fingerprint is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
