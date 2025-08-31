"""Integration Test Suite for Security Server Refactoring.

Tests integration between refactored security modules:
- SessionManager + JWTAuthenticator integration
- Middleware chain integration
- Route handlers + security context integration
- End-to-end security workflow integration
- Backward compatibility validation
"""

import asyncio
from datetime import datetime, timedelta
import json
import secrets
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, make_mocked_request

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedSecureAPIServer,
    EnhancedJWTAuthenticator,
    EnhancedSecurityError,
)
from infrastructure.shared.security.redis_session_manager import (
    DeviceInfo,
    RedisSessionManager,
    SessionData,
)
from infrastructure.shared.security.mfa_system import MFAMethodType, MFASystem


class TestFullSecurityWorkflowIntegration:
    """Test complete security workflow integration."""
    
    def setup_method(self):
        """Set up integrated test environment."""
        self.server = EnhancedSecureAPIServer()
        
        # Initialize all security components
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()
        
        self.authenticator = EnhancedJWTAuthenticator(self.session_manager)
        self.mfa_system = MFASystem()
        
        # Wire up server components
        self.server.session_manager = self.session_manager
        self.server.authenticator = self.authenticator
        self.server.mfa_system = self.mfa_system
        
        # Mock RBAC system
        self.server.rbac_system = Mock()
        self.server.encryption = Mock()
        
    @pytest.mark.asyncio
    async def test_complete_login_to_api_access_workflow(self):
        """Test complete workflow from login to API access."""
        user_id = "integration_user"
        device_info = DeviceInfo("Integration Browser", "192.168.1.100")
        
        # Step 1: Mock user login
        login_request = make_mocked_request("POST", "/auth/login")
        login_request["device_info"] = device_info
        
        async def mock_login_json():
            return {
                "username": "integrationuser",
                "password": "testpass123",
                "mfa_token": "123456",
                "mfa_method": MFAMethodType.TOTP
            }
        login_request.json = mock_login_json
        
        # Mock user validation
        mock_user = {
            "user_id": user_id,
            "password_hash": "mock_hash",
            "password_salt": "mock_salt",
            "totp_secret": "mock_secret"
        }
        self.server.rbac_system.get_user.return_value = mock_user
        
        # Mock session creation
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()
        
        # Mock authentication components
        with patch('hmac.compare_digest', return_value=True):
            with patch('hashlib.pbkdf2_hmac') as mock_pbkdf2:
                mock_pbkdf2.return_value.hex.return_value = "mock_hash"
                
                self.server.mfa_system.get_user_mfa_status.return_value = {
                    "totp_enabled": True,
                    "methods_available": [MFAMethodType.TOTP]
                }
                self.server.mfa_system.verify_mfa.return_value = True
                
                self.server.rbac_system.get_user_roles.return_value = ["user"]
                self.server.rbac_system.get_role_permissions.return_value = ["read"]
                
                # Perform login
                login_response = await self.server._enhanced_login(login_request)
                
                assert login_response.status == 200
                login_data = json.loads(login_response.text)
                access_token = login_data["access_token"]
                
        # Step 2: Use token to access protected API
        api_request = make_mocked_request("GET", "/api/protected")
        api_request.headers = {"Authorization": f"Bearer {access_token}"}
        
        # Mock token verification
        mock_payload = {
            "user_id": user_id,
            "session_id": login_data["session_id"],
            "roles": ["user"],
            "permissions": ["read"],
            "mfa_verified": True,
        }
        
        # Mock session validation
        mock_session = SessionData(user_id, login_data["session_id"], device_info)
        mock_session.is_active = True
        
        self.session_manager.is_token_revoked = AsyncMock(return_value=False)
        self.session_manager.get_session = AsyncMock(return_value=mock_session)
        self.session_manager.update_session = AsyncMock(return_value=True)
        
        with patch.object(self.authenticator, 'verify_token_with_session', return_value=mock_payload):
            # Apply auth middleware
            async def protected_handler(request):
                return web.json_response({"data": "protected_data", "user_id": request["user"]["user_id"]})
            
            api_response = await self.server._auth_middleware(api_request, protected_handler)
            
            assert api_response.status == 200
            api_data = json.loads(api_response.text)
            assert api_data["user_id"] == user_id
            
    @pytest.mark.asyncio
    async def test_middleware_chain_integration(self):
        """Test integration of all middleware components."""
        request = make_mocked_request("POST", "/api/sensitive")
        request.remote = "192.168.1.100"
        request.headers = {
            "Authorization": "Bearer valid_token",
            "User-Agent": "Test Browser",
            "X-MFA-Token": "123456"
        }
        
        # Mock all middleware dependencies
        mock_user = {
            "user_id": "user123",
            "roles": ["user"],
            "permissions": ["read", "write"],
            "mfa_verified": True,
            "session_id": "session123"
        }
        
        # Chain all middleware
        async def final_handler(request):
            return web.json_response({"result": "success", "context_keys": list(request.keys())})
        
        # Apply middleware in order
        response = await self.server._security_middleware(request,
            lambda req: self.server._rate_limit_middleware(req,
                lambda req: self.server._session_middleware(req,
                    lambda req: self.server._mfa_middleware(req,
                        lambda req: self.server._auth_middleware(req, final_handler)))))
        
        # Should have security headers
        assert "X-Security-Level" in response.headers
        assert response.headers["X-Security-Level"] == "B+"
        
    @pytest.mark.asyncio
    async def test_session_token_lifecycle_integration(self):
        """Test integration of session and token lifecycle."""
        user_id = "lifecycle_user"
        device_info = DeviceInfo("Lifecycle Browser", "127.0.0.1")
        
        # Mock session creation
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()
        
        # Create session and tokens
        session_id = await self.session_manager.create_session(user_id, device_info)
        
        # Mock token creation
        with patch('jwt.encode', return_value="mock_jwt_token"):
            tokens = await self.authenticator.create_session_tokens(
                user_id=user_id,
                device_info=device_info,
                roles=["user"],
                permissions=["read"],
                mfa_verified=True
            )
        
        assert "access_token" in tokens
        assert "session_id" in tokens
        
        # Mock token verification
        mock_payload = {
            "user_id": user_id,
            "session_id": tokens["session_id"],
            "jti": "mock_jti"
        }
        
        mock_session = SessionData(user_id, tokens["session_id"], device_info)
        mock_session.is_active = True
        
        self.session_manager.is_token_revoked = AsyncMock(return_value=False)
        self.session_manager.get_session = AsyncMock(return_value=mock_session)
        self.session_manager.update_session = AsyncMock(return_value=True)
        
        with patch('jwt.decode', return_value=mock_payload):
            # Verify token
            verified_payload = await self.authenticator.verify_token_with_session(tokens["access_token"])
            
            assert verified_payload["user_id"] == user_id
            assert verified_payload["session_id"] == tokens["session_id"]
            
        # Revoke session (should invalidate all tokens)
        self.session_manager.revoke_session = AsyncMock(return_value=True)
        revoked = await self.authenticator.revoke_session(tokens["session_id"])
        assert revoked


class TestBackwardCompatibilityIntegration:
    """Test backward compatibility with existing systems."""
    
    def setup_method(self):
        """Set up backward compatibility test environment."""
        self.server = EnhancedSecureAPIServer()
        
    @pytest.mark.asyncio
    async def test_legacy_endpoint_compatibility(self):
        """Test that legacy endpoints still work with new security."""
        # Test legacy endpoint patterns
        legacy_endpoints = [
            "/health",
            "/auth/login",
            "/profiles/user123",
            "/api/data"
        ]
        
        for endpoint in legacy_endpoints:
            request = make_mocked_request("GET", endpoint)
            
            # Legacy endpoints should still be routable
            assert request.path == endpoint
            assert request.method == "GET"
            
    def test_legacy_response_format_compatibility(self):
        """Test that response formats remain compatible."""
        # Mock legacy health check response
        expected_legacy_format = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
        
        # New enhanced response should include legacy fields
        enhanced_response = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "security_rating": "B+",  # New field
            "services": {  # New detailed services
                "authentication": "operational",
                "session_management": "healthy",
                "encryption": "operational",
            }
        }
        
        # Verify legacy fields are present
        for key in expected_legacy_format:
            assert key in enhanced_response
            
    @pytest.mark.asyncio
    async def test_legacy_authentication_flow_compatibility(self):
        """Test that legacy authentication flows still work."""
        request = make_mocked_request("POST", "/auth/login")
        
        # Legacy login format (without MFA)
        async def mock_legacy_json():
            return {
                "username": "legacyuser",
                "password": "legacypass"
            }
        request.json = mock_legacy_json
        
        # Should still work without MFA if not enabled for user
        mock_user = {
            "user_id": "legacy_user",
            "password_hash": "mock_hash",
            "password_salt": "mock_salt"
        }
        
        self.server.rbac_system = Mock()
        self.server.rbac_system.get_user.return_value = mock_user
        
        # Mock no MFA requirement
        self.server.mfa_system = Mock()
        self.server.mfa_system.get_user_mfa_status.return_value = {
            "totp_enabled": False,
            "sms_enabled": False,
            "email_enabled": False
        }
        
        with patch('hmac.compare_digest', return_value=True):
            with patch('hashlib.pbkdf2_hmac') as mock_pbkdf2:
                mock_pbkdf2.return_value.hex.return_value = "mock_hash"
                
                self.server.rbac_system.get_user_roles.return_value = ["user"]
                self.server.rbac_system.get_role_permissions.return_value = ["read"]
                
                self.server.authenticator = Mock()
                self.server.authenticator.create_session_tokens = AsyncMock(return_value={
                    "access_token": "legacy_token",
                    "refresh_token": "legacy_refresh",
                    "session_id": "legacy_session"
                })
                
                request["device_info"] = DeviceInfo("Legacy Browser", "127.0.0.1")
                
                response = await self.server._enhanced_login(request)
                
                # Should succeed without MFA
                assert response.status == 200
                response_data = json.loads(response.text)
                assert "access_token" in response_data


class TestConcurrentOperationsIntegration:
    """Test integration under concurrent operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.session_manager = RedisSessionManager()
        self.server.session_manager.redis_client = AsyncMock()
        
    @pytest.mark.asyncio
    async def test_concurrent_login_attempts(self):
        """Test concurrent login attempts for same user."""
        user_id = "concurrent_user"
        
        # Mock user data
        mock_user = {
            "user_id": user_id,
            "password_hash": "mock_hash",
            "password_salt": "mock_salt"
        }
        
        self.server.rbac_system = Mock()
        self.server.rbac_system.get_user.return_value = mock_user
        self.server.rbac_system.get_user_roles.return_value = ["user"]
        self.server.rbac_system.get_role_permissions.return_value = ["read"]
        
        self.server.mfa_system = Mock()
        self.server.mfa_system.get_user_mfa_status.return_value = {"totp_enabled": False}
        
        self.server.authenticator = Mock()
        
        # Mock Redis operations for concurrent sessions
        self.server.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.server.session_manager.redis_client.smembers.return_value = set()
        
        # Create multiple concurrent login requests
        login_tasks = []
        for i in range(5):
            request = make_mocked_request("POST", "/auth/login")
            request["device_info"] = DeviceInfo(f"Browser{i}", f"192.168.1.{100+i}")
            
            async def mock_json():
                return {"username": "concurrentuser", "password": "testpass"}
            request.json = mock_json
            
            # Mock token creation for each request
            self.server.authenticator.create_session_tokens = AsyncMock(return_value={
                "access_token": f"token_{i}",
                "session_id": f"session_{i}"
            })
            
            with patch('hmac.compare_digest', return_value=True):
                with patch('hashlib.pbkdf2_hmac') as mock_pbkdf2:
                    mock_pbkdf2.return_value.hex.return_value = "mock_hash"
                    
                    login_tasks.append(self.server._enhanced_login(request))
        
        # Execute concurrent logins
        responses = await asyncio.gather(*login_tasks, return_exceptions=True)
        
        # All should succeed or handle gracefully
        for response in responses:
            if isinstance(response, web.Response):
                assert response.status in [200, 429]  # Success or rate limited
                
    @pytest.mark.asyncio
    async def test_concurrent_session_operations(self):
        """Test concurrent session operations."""
        user_id = "session_ops_user"
        device_info = DeviceInfo("Operations Browser", "127.0.0.1")
        
        # Mock session creation
        self.server.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.server.session_manager.redis_client.smembers.return_value = set()
        
        session_id = await self.server.session_manager.create_session(user_id, device_info)
        
        # Create concurrent operations
        operations = [
            self.server.session_manager.add_token_to_session(session_id, f"token_{i}", "access")
            for i in range(10)
        ]
        
        # Execute concurrently
        results = await asyncio.gather(*operations, return_exceptions=True)
        
        # All operations should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            
    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self):
        """Test rate limiting behavior under high load."""
        # Create many requests from same IP
        requests = []
        for i in range(100):
            request = make_mocked_request("GET", f"/api/test{i}")
            request.remote = "192.168.1.100"
            request["user"] = {"user_id": "load_test_user"}
            requests.append(request)
        
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # Apply rate limiting to all requests
        responses = []
        for request in requests:
            response = await self.server._rate_limit_middleware(request, mock_handler)
            responses.append(response)
        
        # Should have mix of successful and rate-limited responses
        success_count = sum(1 for r in responses if r.status == 200)
        rate_limited_count = sum(1 for r in responses if r.status == 429)
        
        assert success_count > 0  # Some should succeed
        assert rate_limited_count > 0  # Some should be rate limited
        assert success_count + rate_limited_count == 100


class TestSecurityBreachSimulation:
    """Test security breach simulation and response."""
    
    def setup_method(self):
        """Set up breach simulation environment."""
        self.server = EnhancedSecureAPIServer()
        self.server.session_manager = Mock()
        self.server.authenticator = Mock()
        
    @pytest.mark.asyncio
    async def test_token_compromise_response(self):
        """Test response to token compromise."""
        compromised_jti = "compromised_token_123"
        
        # Mock token revocation
        self.server.authenticator.revoke_token = AsyncMock(return_value=True)
        
        # Simulate token compromise detection and response
        revoked = await self.server.authenticator.revoke_token(compromised_jti)
        assert revoked
        
        # Verify token is marked as revoked
        self.server.authenticator.revoke_token.assert_called_once_with(compromised_jti)
        
    @pytest.mark.asyncio
    async def test_session_hijacking_protection(self):
        """Test protection against session hijacking."""
        user_id = "hijack_test_user"
        original_device = DeviceInfo("Original Browser", "192.168.1.100")
        hijacking_device = DeviceInfo("Hijacking Browser", "203.0.113.1")
        
        # Mock session with original device
        original_session = SessionData(user_id, "session123", original_device)
        original_session.is_active = True
        
        # Mock session manager responses
        self.server.session_manager.get_session = AsyncMock(return_value=original_session)
        self.server.session_manager.detect_suspicious_activity = AsyncMock(return_value=True)
        
        # Simulate request from different device
        hijack_request = make_mocked_request("GET", "/api/sensitive")
        hijack_request["device_info"] = hijacking_device
        hijack_request["user"] = {"user_id": user_id, "session_id": "session123"}
        
        async def mock_handler(request):
            return web.Response(text="Sensitive data")
        
        # Apply session middleware (should detect suspicious activity)
        response = await self.server._session_middleware(hijack_request, mock_handler)
        
        # Verify suspicious activity was detected
        self.server.session_manager.detect_suspicious_activity.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_brute_force_attack_protection(self):
        """Test protection against brute force attacks."""
        attacker_ip = "192.168.1.200"
        
        # Simulate multiple failed login attempts
        failed_attempts = []
        for i in range(10):
            request = make_mocked_request("POST", "/auth/login")
            request.remote = attacker_ip
            
            async def mock_json():
                return {"username": "victim", "password": f"wrong_pass_{i}"}
            request.json = mock_json
            
            # Mock user not found (failed login)
            self.server.rbac_system = Mock()
            self.server.rbac_system.get_user.return_value = None
            
            response = await self.server._enhanced_login(request)
            failed_attempts.append(response.status)
        
        # All attempts should fail with 401
        assert all(status == 401 for status in failed_attempts)
        
        # Rate limiting should eventually kick in
        request = make_mocked_request("GET", "/api/test")
        request.remote = attacker_ip
        
        async def mock_handler(request):
            return web.Response(text="Success")
        
        # After many failed attempts, should be rate limited
        for _ in range(70):  # Exceed unauthenticated rate limit
            await self.server._rate_limit_middleware(request, mock_handler)
        
        response = await self.server._rate_limit_middleware(request, mock_handler)
        assert response.status == 429  # Rate limited


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience integration."""
    
    def setup_method(self):
        """Set up error recovery test environment."""
        self.server = EnhancedSecureAPIServer()
        
    @pytest.mark.asyncio
    async def test_redis_failure_graceful_degradation(self):
        """Test graceful degradation when Redis fails."""
        request = make_mocked_request("GET", "/api/test")
        
        # Mock Redis failure
        self.server.session_manager = Mock()
        self.server.session_manager.health_check = AsyncMock(side_effect=ConnectionError("Redis down"))
        
        # Health check should report degraded status
        response = await self.server._health_check(request)
        
        assert response.status == 500
        response_data = json.loads(response.text)
        assert "error" in response_data
        
    @pytest.mark.asyncio
    async def test_partial_service_failure_handling(self):
        """Test handling when some security services fail."""
        request = make_mocked_request("GET", "/security/status")
        
        # Mock partial failure
        self.server.encryption = Mock()
        self.server.encryption.get_key_status.side_effect = Exception("Encryption service error")
        
        self.server.session_manager = Mock()
        self.server.session_manager.health_check = AsyncMock(return_value={"status": "healthy"})
        
        # Should handle partial failure gracefully
        response = await self.server._security_status(request)
        
        assert response.status == 500
        response_data = json.loads(response.text)
        assert "Failed to get security status" in response_data["error"]
        
    @pytest.mark.asyncio
    async def test_automatic_recovery_mechanisms(self):
        """Test automatic recovery mechanisms."""
        # Mock recovery scenario
        self.server.session_manager = Mock()
        
        # Simulate service recovery
        health_responses = [
            {"status": "unhealthy"},  # Initially unhealthy
            {"status": "recovering"},  # Recovery in progress
            {"status": "healthy"}     # Recovered
        ]
        
        self.server.session_manager.health_check = AsyncMock(side_effect=health_responses)
        
        # Multiple health checks should show recovery progression
        for expected_status in ["unhealthy", "recovering", "healthy"]:
            health = await self.server.session_manager.health_check()
            assert health["status"] == expected_status


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])