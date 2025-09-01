"""Comprehensive SessionManager Test Suite for Security Server Refactoring.

Tests for the extracted SessionManager class including:
- Authentication and token management
- MFA integration
- Session lifecycle management
- Device tracking and security monitoring
- Rate limiting and session enforcement
"""

import asyncio
from datetime import datetime, timedelta
import json
import secrets
from unittest.mock import AsyncMock, patch

import pytest

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedJWTAuthenticator,
)
from infrastructure.shared.security.redis_session_manager import (
    DeviceInfo,
    RedisSessionManager,
    SessionData,
)


class TestSessionManager:
    """Test SessionManager class functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()

        # Mock JWT authenticator
        self.authenticator = EnhancedJWTAuthenticator(self.session_manager)
        self.authenticator.secret_key = secrets.token_urlsafe(32)

    @pytest.mark.asyncio
    async def test_session_creation_with_authentication(self):
        """Test session creation integrated with authentication."""
        user_id = "test_user_123"
        device_info = DeviceInfo(user_agent="Mozilla/5.0 Test Browser", ip_address="192.168.1.100")

        # Mock Redis operations
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        # Create session
        session_id = await self.session_manager.create_session(user_id, device_info)

        # Verify session creation
        assert session_id.startswith("sess_")
        assert len(session_id) > 20

        # Verify Redis operations were called properly
        self.session_manager.redis_client.pipeline.assert_called()

    @pytest.mark.asyncio
    async def test_mfa_integration_with_session(self):
        """Test MFA integration with session management."""
        user_id = "mfa_test_user"
        device_info = DeviceInfo("Test Browser", "127.0.0.1")

        # Mock session creation
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        # Create session with MFA requirement
        session_id = await self.session_manager.create_session(user_id, device_info)

        # Test MFA status tracking in session
        mock_session_data = {
            "user_id": user_id,
            "session_id": session_id,
            "device_info": json.dumps(device_info.to_dict()),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "access_tokens": json.dumps([]),
            "refresh_tokens": json.dumps([]),
            "is_active": True,
            "security_flags": json.dumps(["mfa_required"]),
        }

        self.session_manager.redis_client.hgetall.return_value = mock_session_data

        # Get session and check MFA requirement
        session = await self.session_manager.get_session(session_id)
        assert session is not None
        assert "mfa_required" in session.security_flags

    @pytest.mark.asyncio
    async def test_session_lifecycle_management(self):
        """Test complete session lifecycle from creation to termination."""
        user_id = "lifecycle_user"
        device_info = DeviceInfo("Lifecycle Browser", "10.0.0.1")

        # 1. Create session
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        session_id = await self.session_manager.create_session(user_id, device_info)

        # 2. Add tokens to session
        jti_access = "access_token_123"
        jti_refresh = "refresh_token_456"

        await self.session_manager.add_token_to_session(session_id, jti_access, "access")
        await self.session_manager.add_token_to_session(session_id, jti_refresh, "refresh")

        # 3. Verify session activity
        self.session_manager.redis_client.hgetall.return_value = {
            "user_id": user_id,
            "session_id": session_id,
            "device_info": json.dumps(device_info.to_dict()),
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "access_tokens": json.dumps([jti_access]),
            "refresh_tokens": json.dumps([jti_refresh]),
            "is_active": True,
            "security_flags": json.dumps([]),
        }

        session = await self.session_manager.get_session(session_id)
        assert session.is_active
        assert jti_access in session.access_tokens
        assert jti_refresh in session.refresh_tokens

        # 4. Revoke session
        revoked = await self.session_manager.revoke_session(session_id)
        assert revoked

    @pytest.mark.asyncio
    async def test_device_tracking_security(self):
        """Test device tracking and security monitoring."""
        user_id = "device_test_user"

        # Multiple device attempts
        devices = [
            DeviceInfo("Browser1", "192.168.1.1"),
            DeviceInfo("Browser2", "192.168.1.2"),
            DeviceInfo("Browser3", "10.0.0.1"),
            DeviceInfo("Suspicious Browser", "192.168.1.100"),
        ]

        # Create sessions for each device
        sessions = []
        for i, device in enumerate(devices):
            self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
            self.session_manager.redis_client.smembers.return_value = set()

            session_id = await self.session_manager.create_session(user_id, device)
            sessions.append(SessionData(user_id, session_id, device))

        # Mock get_user_sessions to return all sessions
        with patch.object(self.session_manager, "get_user_sessions", return_value=sessions):
            # Test suspicious activity detection
            new_device = DeviceInfo("New Suspicious Browser", "192.168.1.200")
            suspicious = await self.session_manager.detect_suspicious_activity(user_id, new_device)

            # Should detect suspicious activity due to multiple IPs
            assert suspicious

    @pytest.mark.asyncio
    async def test_session_limits_enforcement(self):
        """Test session limits and automatic cleanup."""
        user_id = "limits_test_user"
        max_sessions = self.session_manager.max_sessions_per_user

        # Create sessions up to the limit
        existing_sessions = []
        for i in range(max_sessions):
            device = DeviceInfo(f"Browser{i}", f"10.0.0.{i}")
            session = SessionData(user_id, f"sess_{i}", device)
            session.last_activity = datetime.utcnow() - timedelta(minutes=i * 5)
            existing_sessions.append(session)

        # Mock get_user_sessions
        with patch.object(self.session_manager, "get_user_sessions", return_value=existing_sessions):
            with patch.object(self.session_manager, "revoke_session", return_value=True) as mock_revoke:
                # Try to create another session (should trigger cleanup)
                await self.session_manager._enforce_session_limits(user_id)

                # Should revoke oldest session
                mock_revoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with session manager."""
        user_id = "rate_limit_user"

        # Test rate limiting for session operations
        for i in range(5):  # Within limit
            allowed = self.session_manager._check_operation_rate_limit(user_id, "login")
            assert allowed

        # 6th attempt should be blocked
        blocked = self.session_manager._check_operation_rate_limit(user_id, "login")
        assert not blocked

    @pytest.mark.asyncio
    async def test_session_analytics_tracking(self):
        """Test session analytics and monitoring."""
        user_id = "analytics_user"

        # Mock analytics data
        mock_analytics = {
            "user_id": user_id,
            "total_sessions": 5,
            "active_sessions": 2,
            "devices_count": 3,
            "last_login": datetime.utcnow().isoformat(),
            "suspicious_activity_count": 0,
        }

        with patch.object(self.session_manager, "_calculate_user_analytics", return_value=mock_analytics):
            analytics = await self.session_manager.get_session_analytics(user_id)

            assert analytics["user_id"] == user_id
            assert analytics["total_sessions"] == 5
            assert analytics["active_sessions"] == 2


class TestEnhancedJWTAuthenticator:
    """Test JWT authenticator with session integration."""

    def setup_method(self):
        """Set up test environment."""
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()
        self.authenticator = EnhancedJWTAuthenticator(self.session_manager)

    @pytest.mark.asyncio
    async def test_token_creation_with_session(self):
        """Test token creation integrated with session management."""
        user_id = "token_test_user"
        device_info = DeviceInfo("Test Browser", "127.0.0.1")
        roles = ["user", "premium"]
        permissions = ["read", "write"]

        # Mock session creation
        self.session_manager.create_session = AsyncMock(return_value="sess_12345")
        self.session_manager.add_token_to_session = AsyncMock(return_value=True)

        # Create tokens
        tokens = await self.authenticator.create_session_tokens(
            user_id=user_id, device_info=device_info, roles=roles, permissions=permissions, mfa_verified=True
        )

        # Verify token structure
        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert "session_id" in tokens
        assert tokens["mfa_verified"] is True

        # Verify session was created
        self.session_manager.create_session.assert_called_once_with(user_id, device_info)

    @pytest.mark.asyncio
    async def test_token_verification_with_session_validation(self):
        """Test token verification with session validation."""
        user_id = "verify_test_user"
        session_id = "sess_verify_123"
        jti = "token_jti_456"

        # Create test token
        import jwt

        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "roles": ["user"],
            "permissions": ["read"],
            "mfa_verified": True,
            "type": "access_token",
            "jti": jti,
            "exp": datetime.utcnow() + timedelta(hours=1),
        }
        token = jwt.encode(payload, self.authenticator.secret_key, algorithm="HS256")

        # Mock session validation
        mock_session = SessionData(user_id, session_id, DeviceInfo("Browser", "127.0.0.1"))
        mock_session.is_active = True

        self.session_manager.is_token_revoked = AsyncMock(return_value=False)
        self.session_manager.get_session = AsyncMock(return_value=mock_session)
        self.session_manager.update_session = AsyncMock(return_value=True)

        # Verify token
        verified_payload = await self.authenticator.verify_token_with_session(token)

        assert verified_payload["user_id"] == user_id
        assert verified_payload["session_id"] == session_id
        assert verified_payload["mfa_verified"] is True

    @pytest.mark.asyncio
    async def test_token_revocation_cascade(self):
        """Test token revocation cascading through session."""
        jti = "revoke_test_jti"

        # Mock token-to-session mapping
        self.session_manager.revoke_token = AsyncMock(return_value=True)

        # Revoke token
        revoked = await self.authenticator.revoke_token(jti)
        assert revoked

        # Verify session manager was called
        self.session_manager.revoke_token.assert_called_once_with(jti)

    @pytest.mark.asyncio
    async def test_session_revocation_cascade(self):
        """Test session revocation cascading to all tokens."""
        session_id = "sess_cascade_123"

        # Mock session revocation
        self.session_manager.revoke_session = AsyncMock(return_value=True)

        # Revoke session
        revoked = await self.authenticator.revoke_session(session_id)
        assert revoked

        # Verify session manager was called
        self.session_manager.revoke_session.assert_called_once_with(session_id)


class TestSessionManagerErrorHandling:
    """Test error handling and edge cases for SessionManager."""

    def setup_method(self):
        """Set up test environment."""
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()

    @pytest.mark.asyncio
    async def test_redis_connection_failure_handling(self):
        """Test handling of Redis connection failures."""
        # Mock Redis connection failure
        self.session_manager.redis_client.pipeline.side_effect = ConnectionError("Redis unavailable")

        user_id = "error_test_user"
        device_info = DeviceInfo("Test Browser", "127.0.0.1")

        # Should handle gracefully
        with pytest.raises(ConnectionError):
            await self.session_manager.create_session(user_id, device_info)

    @pytest.mark.asyncio
    async def test_invalid_session_data_handling(self):
        """Test handling of invalid session data."""
        session_id = "invalid_session_123"

        # Mock invalid session data
        self.session_manager.redis_client.hgetall.return_value = {
            "user_id": "test_user",
            "device_info": "invalid_json_data",  # Invalid JSON
            "created_at": "invalid_date",  # Invalid date
        }

        # Should handle gracefully
        session = await self.session_manager.get_session(session_id)
        assert session is None  # Should return None for invalid data

    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self):
        """Test concurrent session operations."""
        user_id = "concurrent_user"
        device_info = DeviceInfo("Concurrent Browser", "127.0.0.1")

        # Mock Redis operations
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        # Create multiple sessions concurrently
        tasks = [self.session_manager.create_session(user_id, device_info) for _ in range(5)]

        session_ids = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify all sessions were created (or handled gracefully)
        for session_id in session_ids:
            if isinstance(session_id, str):
                assert session_id.startswith("sess_")

    @pytest.mark.asyncio
    async def test_session_data_corruption_recovery(self):
        """Test recovery from session data corruption."""
        session_id = "corrupted_session_123"

        # Mock corrupted session data
        self.session_manager.redis_client.hgetall.return_value = {"corrupted_field": "corrupted_value"}

        # Should handle corruption gracefully
        session = await self.session_manager.get_session(session_id)
        assert session is None  # Should return None for corrupted data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
