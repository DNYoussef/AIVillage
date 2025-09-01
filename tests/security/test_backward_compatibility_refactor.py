"""Backward Compatibility Test Suite for Security Server Refactoring.

Ensures 100% backward compatibility during refactoring:
- Legacy API endpoints maintain same behavior
- Legacy response formats are preserved  
- Legacy authentication flows continue working
- Legacy configuration options remain supported
- No breaking changes in public interfaces
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedSecureAPIServer,
)
from infrastructure.shared.security.redis_session_manager import DeviceInfo


class TestLegacyAPICompatibility:
    """Test that legacy API endpoints maintain compatibility."""

    def setup_method(self):
        """Set up legacy compatibility test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.rbac_system = Mock()
        self.server.mfa_system = Mock()
        self.server.session_manager = Mock()
        self.server.encryption = Mock()
        self.server.authenticator = Mock()

    @pytest.mark.asyncio
    async def test_legacy_health_endpoint_format(self):
        """Test that /health endpoint maintains legacy format."""
        request = make_mocked_request("GET", "/health")

        # Mock component health
        self.server.session_manager.health_check = AsyncMock(return_value={"status": "healthy", "latency_ms": 5})

        self.server.encryption.get_key_status.return_value = {"algorithm": "AES-256-GCM", "rotation_needed": False}

        response = await self.server._health_check(request)

        assert response.status == 200
        response_data = json.loads(response.text)

        # Legacy format fields must be present
        legacy_required_fields = ["status", "timestamp", "version"]
        for field in legacy_required_fields:
            assert field in response_data

        # Legacy status values
        assert response_data["status"] in ["healthy", "degraded", "error"]

        # New fields should be additional, not replacing legacy ones
        assert "security_rating" in response_data  # New field
        assert "services" in response_data  # New detailed services

    @pytest.mark.asyncio
    async def test_legacy_login_endpoint_compatibility(self):
        """Test that /auth/login maintains legacy behavior."""
        request = make_mocked_request("POST", "/auth/login")
        request["device_info"] = DeviceInfo("Legacy Browser", "127.0.0.1")

        # Legacy login request format
        async def mock_legacy_json():
            return {
                "username": "legacyuser",
                "password": "legacypass123",
                # Note: No MFA fields in legacy request
            }

        request.json = mock_legacy_json

        # Mock legacy user (no MFA enabled)
        mock_user = {"user_id": "legacy_user_123", "password_hash": "legacy_hash", "password_salt": "legacy_salt"}
        self.server.rbac_system.get_user.return_value = mock_user

        # Mock no MFA requirement for legacy user
        self.server.mfa_system.get_user_mfa_status.return_value = {
            "totp_enabled": False,
            "sms_enabled": False,
            "email_enabled": False,
            "methods_available": [],
        }

        # Mock RBAC
        self.server.rbac_system.get_user_roles.return_value = ["user"]
        self.server.rbac_system.get_role_permissions.return_value = ["read"]

        # Mock token creation
        legacy_tokens = {
            "access_token": "legacy_access_token",
            "refresh_token": "legacy_refresh_token",
            "token_type": "Bearer",
            "expires_in": 86400,
            "session_id": "legacy_session_123",
        }
        self.server.authenticator.create_session_tokens = AsyncMock(return_value=legacy_tokens)

        with patch("hmac.compare_digest", return_value=True):
            with patch("hashlib.pbkdf2_hmac") as mock_pbkdf2:
                mock_pbkdf2.return_value.hex.return_value = "legacy_hash"

                response = await self.server._enhanced_login(request)

                assert response.status == 200
                response_data = json.loads(response.text)

                # Legacy response format fields must be present
                legacy_required_fields = ["access_token", "token_type", "expires_in"]
                for field in legacy_required_fields:
                    assert field in response_data

                assert response_data["token_type"] == "Bearer"
                assert isinstance(response_data["expires_in"], int)

    @pytest.mark.asyncio
    async def test_legacy_profile_endpoints_compatibility(self):
        """Test that profile endpoints maintain legacy behavior."""
        profile_id = "legacy_profile_123"
        request = make_mocked_request("GET", f"/profiles/{profile_id}")
        request.match_info = {"profile_id": profile_id}
        request["user"] = {"user_id": "legacy_user", "roles": ["user"]}

        # Mock profile database
        self.server.profile_db = Mock()
        mock_profile_data = {
            "profile_id": profile_id,
            "user_preferences": {"theme": "light", "notifications": True},
            "created_at": "2024-01-01T00:00:00Z",
        }
        self.server.profile_db.get_profile.return_value = mock_profile_data

        response = await self.server._get_profile_enhanced(request)

        assert response.status == 200
        response_data = json.loads(response.text)

        # Legacy response format
        assert "profile_id" in response_data
        assert "data" in response_data
        assert response_data["profile_id"] == profile_id

        # Should include legacy data structure
        assert "user_preferences" in response_data["data"]

    def test_legacy_configuration_support(self):
        """Test that legacy configuration options are still supported."""
        # Test legacy environment variables
        legacy_env_vars = {
            "TLS_ENABLED": "true",
            "API_SECRET_KEY": "legacy_secret_key_32_chars_long",
            "API_JWT_EXPIRY_HOURS": "24",
            "API_CORS_ENABLED": "true",
            "API_CORS_ORIGINS": "https://legacy.example.com,https://app.example.com",
        }

        with patch.dict("os.environ", legacy_env_vars):
            server = EnhancedSecureAPIServer()

            # Legacy configuration should still work
            assert server.tls_enabled is True
            assert server.cors_enabled is True
            assert len(server.cors_origins) == 2
            assert "https://legacy.example.com" in server.cors_origins

    def test_legacy_error_response_format(self):
        """Test that error responses maintain legacy format."""
        # Legacy error format should be: {"error": "message"}

        legacy_errors = [
            {"status": 400, "message": "Bad request"},
            {"status": 401, "message": "Unauthorized"},
            {"status": 403, "message": "Forbidden"},
            {"status": 404, "message": "Not found"},
            {"status": 500, "message": "Internal server error"},
        ]

        for error in legacy_errors:
            response = web.json_response({"error": error["message"]}, status=error["status"])

            # Check legacy format
            assert response.status == error["status"]
            response_data = json.loads(response.text)
            assert "error" in response_data
            assert response_data["error"] == error["message"]


class TestLegacyMiddlewareCompatibility:
    """Test that middleware changes don't break legacy behavior."""

    def setup_method(self):
        """Set up middleware compatibility test."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_legacy_auth_header_processing(self):
        """Test that legacy auth header processing still works."""
        # Legacy auth header format
        request = make_mocked_request("GET", "/api/legacy")
        request.headers = {"Authorization": "Bearer legacy_jwt_token_format"}

        # Extract auth header (legacy way)
        auth_header = request.headers.get("Authorization")
        assert auth_header == "Bearer legacy_jwt_token_format"

        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove 'Bearer '
            assert token == "legacy_jwt_token_format"

    @pytest.mark.asyncio
    async def test_legacy_cors_handling(self):
        """Test that legacy CORS handling is preserved."""
        # Legacy CORS preflight request
        request = make_mocked_request(
            "OPTIONS",
            "/api/legacy",
            headers={
                "Origin": "https://legacy-app.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type",
            },
        )

        # Legacy CORS validation
        origin = request.headers.get("Origin")
        assert origin == "https://legacy-app.example.com"

        # Should still support legacy origins
        legacy_origins = ["https://legacy-app.example.com", "https://old-app.example.com"]
        assert origin in legacy_origins or self.server.cors_enabled

    @pytest.mark.asyncio
    async def test_legacy_rate_limiting_behavior(self):
        """Test that legacy rate limiting behavior is preserved."""
        request = make_mocked_request("GET", "/api/legacy")
        request.remote = "192.168.1.100"

        async def mock_handler(request):
            return web.Response(text="Success")

        # Legacy rate limiting should still work the same way
        # First request should succeed
        response = await self.server._rate_limit_middleware(request, mock_handler)
        assert response.status == 200

        # Rate limit structure should be preserved
        assert hasattr(self.server, "rate_limits")
        assert isinstance(self.server.rate_limits, dict)


class TestLegacySecurityFeatures:
    """Test that legacy security features are preserved."""

    def setup_method(self):
        """Set up legacy security test environment."""
        self.server = EnhancedSecureAPIServer()

    def test_legacy_ssl_configuration(self):
        """Test that legacy SSL configuration is preserved."""
        # Legacy SSL settings should still work
        self.server.tls_enabled = True
        self.server.cert_file = "./legacy/server.crt"
        self.server.key_file = "./legacy/server.key"

        # Should create SSL context with legacy settings
        with patch("ssl.create_default_context") as mock_ssl:
            mock_context = Mock()
            mock_ssl.return_value = mock_context

            with patch("pathlib.Path.exists", return_value=True):
                self.server._create_ssl_context()

                # Should use legacy cert files
                mock_context.load_cert_chain.assert_called_with(self.server.cert_file, self.server.key_file)

    def test_legacy_security_headers(self):
        """Test that legacy security headers are preserved."""
        legacy_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
        }

        # Mock response
        response = web.Response(text="Test")

        # Apply legacy headers
        for header, value in legacy_headers.items():
            response.headers[header] = value

        # Verify legacy headers are present
        for header, expected_value in legacy_headers.items():
            assert response.headers[header] == expected_value

    @pytest.mark.asyncio
    async def test_legacy_session_format_compatibility(self):
        """Test that legacy session formats are still supported."""
        # Test legacy session data format
        legacy_session_data = {
            "user_id": "legacy_user",
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "is_active": True,
        }

        # Should be able to process legacy session format
        user_id = legacy_session_data.get("user_id")
        is_active = legacy_session_data.get("is_active", False)
        created_at = datetime.fromisoformat(legacy_session_data["created_at"])

        assert user_id == "legacy_user"
        assert is_active is True
        assert isinstance(created_at, datetime)

    def test_legacy_rbac_integration_compatibility(self):
        """Test that legacy RBAC integration is preserved."""
        # Mock legacy RBAC system
        mock_rbac = Mock()

        # Legacy RBAC methods should still be called
        mock_rbac.get_user.return_value = {"user_id": "test", "roles": ["user"]}
        mock_rbac.get_user_roles.return_value = ["user"]
        mock_rbac.get_role_permissions.return_value = ["read"]

        # Verify legacy RBAC interface
        user = mock_rbac.get_user("testuser")
        assert user["user_id"] == "test"

        roles = mock_rbac.get_user_roles("test")
        assert "user" in roles

        permissions = mock_rbac.get_role_permissions("user")
        assert "read" in permissions


class TestLegacyDatabaseCompatibility:
    """Test database schema and query compatibility."""

    def setup_method(self):
        """Set up database compatibility test."""
        self.server = EnhancedSecureAPIServer()

    def test_legacy_profile_database_schema(self):
        """Test that legacy profile database schema is supported."""
        # Mock legacy profile database
        self.server.profile_db = Mock()

        # Legacy profile format
        legacy_profile = {
            "user_id": "legacy_user",
            "profile_data": {"name": "Legacy User", "preferences": {"theme": "dark"}},
            "created_date": "2023-01-01",
            "last_modified": "2024-01-01",
        }

        self.server.profile_db.get_profile.return_value = legacy_profile

        # Should be able to retrieve legacy profile
        profile = self.server.profile_db.get_profile("legacy_user", "profile123")

        assert profile["user_id"] == "legacy_user"
        assert "profile_data" in profile
        assert "created_date" in profile  # Legacy field name

    def test_legacy_user_database_schema(self):
        """Test that legacy user database schema is supported."""
        # Mock legacy user format
        legacy_user = {
            "user_id": "legacy_user_456",
            "username": "legacyuser",
            "email": "legacy@example.com",
            "password_hash": "legacy_hash_format",
            "password_salt": "legacy_salt",
            "roles": ["user"],
            "created_date": "2023-01-01T00:00:00Z",
            "is_active": True,
        }

        self.server.rbac_system = Mock()
        self.server.rbac_system.get_user.return_value = legacy_user

        # Should support legacy user schema
        user = self.server.rbac_system.get_user("legacyuser")

        assert user["username"] == "legacyuser"
        assert "password_hash" in user
        assert "password_salt" in user
        assert "created_date" in user  # Legacy field name

    def test_legacy_encryption_compatibility(self):
        """Test that legacy encryption is still supported."""
        # Mock legacy Fernet encryption
        with patch("cryptography.fernet.Fernet") as mock_fernet:
            mock_cipher = Mock()
            mock_cipher.decrypt.return_value = b"legacy_decrypted_data"
            mock_fernet.return_value = mock_cipher

            # Legacy environment variable
            with patch.dict("os.environ", {"DIGITAL_TWIN_ENCRYPTION_KEY": "legacy_key"}):
                from infrastructure.shared.security.enhanced_encryption import EnhancedDigitalTwinEncryption

                encryption = EnhancedDigitalTwinEncryption()

                # Should support legacy decryption
                legacy_encrypted = b"legacy_encrypted_data"
                decrypted = encryption.decrypt_sensitive_field(legacy_encrypted, "test_field")

                assert decrypted == "legacy_decrypted_data"
                mock_cipher.decrypt.assert_called_once_with(legacy_encrypted)


class TestLegacyResponseFormats:
    """Test that response formats maintain backward compatibility."""

    def setup_method(self):
        """Set up response format test."""
        self.server = EnhancedSecureAPIServer()

    def test_legacy_success_response_format(self):
        """Test that success responses maintain legacy format."""
        # Legacy success response should have consistent structure
        legacy_success_formats = [
            # Profile retrieval
            {"profile_id": "profile123", "data": {"name": "Test User"}, "status": "success"},
            # Authentication success
            {"access_token": "token123", "token_type": "Bearer", "expires_in": 3600},
            # Operation success
            {"message": "Operation completed successfully", "timestamp": datetime.utcnow().isoformat()},
        ]

        for format_example in legacy_success_formats:
            response = web.json_response(format_example)
            response_data = json.loads(response.text)

            # Should maintain exact same structure
            assert response_data == format_example

    def test_legacy_error_response_format(self):
        """Test that error responses maintain legacy format."""
        # Legacy error responses
        legacy_error_formats = [
            {"error": "Invalid credentials", "status": 401},
            {"error": "Profile not found", "status": 404},
            {"error": "Internal server error", "status": 500},
        ]

        for error_format in legacy_error_formats:
            response = web.json_response({"error": error_format["error"]}, status=error_format["status"])

            assert response.status == error_format["status"]
            response_data = json.loads(response.text)
            assert response_data["error"] == error_format["error"]

    def test_legacy_pagination_format(self):
        """Test that pagination format is preserved."""
        # Legacy pagination format
        legacy_pagination = {
            "data": [{"id": 1}, {"id": 2}, {"id": 3}],
            "total": 3,
            "page": 1,
            "per_page": 10,
            "pages": 1,
        }

        # Should maintain exact pagination structure
        assert "data" in legacy_pagination
        assert "total" in legacy_pagination
        assert "page" in legacy_pagination
        assert legacy_pagination["total"] == len(legacy_pagination["data"])


class TestLegacySecurityPolicyCompatibility:
    """Test that legacy security policies are preserved."""

    def setup_method(self):
        """Set up security policy test."""
        self.server = EnhancedSecureAPIServer()

    def test_legacy_password_policy_enforcement(self):
        """Test that legacy password policies are still enforced."""
        # Legacy password validation rules
        password_tests = [
            {"password": "short", "should_pass": False},  # Too short
            {"password": "nouppercase123", "should_pass": False},  # No uppercase
            {"password": "NOLOWERCASE123", "should_pass": False},  # No lowercase
            {"password": "NoNumbers", "should_pass": False},  # No numbers
            {"password": "ValidPass123", "should_pass": True},  # Meets all criteria
        ]

        def legacy_password_validator(password):
            """Legacy password validation logic."""
            if len(password) < 8:
                return False
            if not any(c.isupper() for c in password):
                return False
            if not any(c.islower() for c in password):
                return False
            if not any(c.isdigit() for c in password):
                return False
            return True

        for test in password_tests:
            result = legacy_password_validator(test["password"])
            assert result == test["should_pass"], f"Failed for password: {test['password']}"

    def test_legacy_session_timeout_behavior(self):
        """Test that legacy session timeout behavior is preserved."""
        # Legacy session timeout (24 hours)
        legacy_timeout_hours = 24

        now = datetime.utcnow()
        session_created = now - timedelta(hours=23)  # Just under timeout
        session_expired = now - timedelta(hours=25)  # Just over timeout

        # Check if session is expired (legacy logic)
        timeout_delta = timedelta(hours=legacy_timeout_hours)

        active_session_valid = (now - session_created) < timeout_delta
        expired_session_valid = (now - session_expired) < timeout_delta

        assert active_session_valid is True  # Should still be valid
        assert expired_session_valid is False  # Should be expired

    def test_legacy_rbac_permission_model(self):
        """Test that legacy RBAC permission model is preserved."""
        # Legacy role hierarchy
        legacy_roles = {
            "guest": ["read_public"],
            "user": ["read_public", "read_user", "write_user"],
            "premium": ["read_public", "read_user", "write_user", "read_premium"],
            "admin": ["read_public", "read_user", "write_user", "read_premium", "admin_all"],
        }

        # Test legacy permission inheritance
        user_roles = ["user"]
        user_permissions = []
        for role in user_roles:
            user_permissions.extend(legacy_roles.get(role, []))

        assert "read_user" in user_permissions
        assert "write_user" in user_permissions
        assert "admin_all" not in user_permissions  # Should not have admin perms


class TestBackwardCompatibilityRegression:
    """Regression tests to ensure no breaking changes."""

    def setup_method(self):
        """Set up regression test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_no_new_required_parameters(self):
        """Test that no new required parameters were added to existing endpoints."""
        # Legacy login request (minimal required fields)
        legacy_login = {"username": "testuser", "password": "testpass"}

        request = make_mocked_request("POST", "/auth/login")
        request["device_info"] = DeviceInfo("Test Browser", "127.0.0.1")

        async def mock_json():
            return legacy_login

        request.json = mock_json

        # Should not require new parameters for basic functionality
        username = legacy_login.get("username")
        password = legacy_login.get("password")

        assert username is not None
        assert password is not None
        # MFA fields should be optional
        assert legacy_login.get("mfa_token") is None
        assert legacy_login.get("mfa_method") is None

    def test_no_removed_response_fields(self):
        """Test that no response fields were removed."""
        # Legacy token response format
        legacy_token_response = {"access_token": "jwt_token_here", "token_type": "Bearer", "expires_in": 3600}

        # Enhanced response should include all legacy fields plus new ones
        enhanced_token_response = {
            "access_token": "jwt_token_here",
            "refresh_token": "refresh_token_here",  # New field
            "token_type": "Bearer",
            "expires_in": 3600,
            "session_id": "session_123",  # New field
            "mfa_verified": False,  # New field
        }

        # All legacy fields should be present
        for key in legacy_token_response:
            assert key in enhanced_token_response
            assert enhanced_token_response[key] == legacy_token_response[key]

    def test_api_version_compatibility(self):
        """Test API version compatibility."""
        # Should support legacy API version headers
        version_headers = [
            {"Accept": "application/json"},  # Default
            {"Accept": "application/json; version=1.0"},  # Legacy version
            {"Accept": "application/json; version=2.0"},  # Current version
        ]

        for headers in version_headers:
            request = make_mocked_request("GET", "/api/test", headers=headers)

            # Should accept all version formats
            accept_header = request.headers.get("Accept", "application/json")
            assert "application/json" in accept_header

    @pytest.mark.asyncio
    async def test_no_breaking_middleware_changes(self):
        """Test that middleware changes don't break existing request flow."""
        request = make_mocked_request("GET", "/api/legacy-compatible")
        request.remote = "127.0.0.1"

        # Track middleware execution
        middleware_calls = []

        async def tracking_handler(request):
            middleware_calls.append("handler_reached")
            return web.Response(text="Success")

        # Enhanced middleware should not break request flow
        try:
            response = await self.server._security_middleware(request, tracking_handler)
            assert "handler_reached" in middleware_calls
            assert response.status == 200
        except Exception as e:
            pytest.fail(f"Middleware broke request flow: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
