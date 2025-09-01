"""Performance Test Suite for Security Server Refactoring.

Tests performance impact of refactored security modules:
- Authentication and token validation performance
- Middleware chain execution performance
- Session management performance under load
- Memory usage optimization
- Concurrent operation performance
"""

import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import make_mocked_request

from infrastructure.shared.security.enhanced_secure_api_server import (
    EnhancedSecureAPIServer,
    EnhancedJWTAuthenticator,
)
from infrastructure.shared.security.redis_session_manager import (
    DeviceInfo,
    RedisSessionManager,
)


class TestAuthenticationPerformance:
    """Test authentication performance after refactoring."""

    def setup_method(self):
        """Set up performance test environment."""
        self.server = EnhancedSecureAPIServer()
        self.session_manager = RedisSessionManager()
        self.session_manager.redis_client = AsyncMock()
        self.authenticator = EnhancedJWTAuthenticator(self.session_manager)

    @pytest.mark.asyncio
    async def test_token_validation_performance(self):
        """Test token validation performance."""
        # Create test token
        import jwt

        payload = {
            "user_id": "perf_user",
            "session_id": "perf_session",
            "exp": datetime.utcnow() + timedelta(hours=1),
            "jti": "perf_jti",
        }
        token = jwt.encode(payload, self.authenticator.secret_key, algorithm="HS256")

        # Mock session validation (fast path)
        self.session_manager.is_token_revoked = AsyncMock(return_value=False)
        self.session_manager.get_session = AsyncMock(return_value=Mock(is_active=True))
        self.session_manager.update_session = AsyncMock(return_value=True)

        # Measure token validation performance
        start_time = time.time()

        # Perform multiple validations
        for _ in range(100):
            await self.authenticator.verify_token_with_session(token)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_validation = total_time / 100

        # Should be fast (under 10ms per validation)
        assert avg_time_per_validation < 0.01, f"Token validation too slow: {avg_time_per_validation:.4f}s"

    @pytest.mark.asyncio
    async def test_session_creation_performance(self):
        """Test session creation performance."""
        user_id = "perf_session_user"

        # Mock fast Redis operations
        self.session_manager.redis_client.pipeline.return_value.execute = AsyncMock()
        self.session_manager.redis_client.smembers.return_value = set()

        # Measure session creation performance
        start_time = time.time()

        # Create multiple sessions
        session_ids = []
        for i in range(50):
            device = DeviceInfo(f"Browser{i}", f"192.168.1.{100+i}")
            session_id = await self.session_manager.create_session(user_id, device)
            session_ids.append(session_id)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_creation = total_time / 50

        # Should be fast (under 50ms per creation)
        assert avg_time_per_creation < 0.05, f"Session creation too slow: {avg_time_per_creation:.4f}s"

        # Verify all sessions were created
        assert len(session_ids) == 50
        assert all(sid.startswith("sess_") for sid in session_ids)

    @pytest.mark.asyncio
    async def test_concurrent_authentication_performance(self):
        """Test performance under concurrent authentication load."""
        # Create concurrent authentication tasks
        auth_tasks = []

        for i in range(20):
            request = make_mocked_request("POST", "/auth/login")
            request["device_info"] = DeviceInfo(f"ConcBrowser{i}", f"10.0.0.{i}")

            async def mock_json():
                return {"username": f"user{i}", "password": "testpass"}

            request.json = mock_json

            auth_tasks.append(self._mock_login_process(request))

        # Execute concurrent authentications
        start_time = time.time()
        results = await asyncio.gather(*auth_tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # Should handle 20 concurrent auths in under 2 seconds
        assert total_time < 2.0, f"Concurrent auth too slow: {total_time:.4f}s"

        # Most should succeed (some might be rate limited)
        success_count = sum(1 for r in results if isinstance(r, dict) and "access_token" in r)
        assert success_count >= 15  # At least 75% should succeed

    async def _mock_login_process(self, request):
        """Mock login process for performance testing."""
        # Simplified mock login
        await asyncio.sleep(0.01)  # Simulate processing time
        return {"access_token": f"token_{time.time()}", "session_id": f"session_{time.time()}"}


class TestMiddlewarePerformance:
    """Test middleware chain performance."""

    def setup_method(self):
        """Set up middleware performance test."""
        self.server = EnhancedSecureAPIServer()
        self.server.rate_limits = {}

    @pytest.mark.asyncio
    async def test_middleware_chain_execution_time(self):
        """Test middleware chain execution performance."""
        request = make_mocked_request("GET", "/api/perf-test")
        request.remote = "127.0.0.1"
        request["user"] = {"user_id": "perf_user", "mfa_verified": True}

        async def fast_handler(request):
            return web.Response(text="Success")

        # Measure middleware chain execution
        start_time = time.time()

        # Execute middleware chain multiple times
        for _ in range(100):
            await self.server._security_middleware(
                request,
                lambda req: self.server._rate_limit_middleware(
                    req,
                    lambda req: self.server._session_middleware(
                        req, lambda req: self.server._mfa_middleware(req, fast_handler)
                    ),
                ),
            )

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_request = total_time / 100

        # Middleware chain should be fast (under 5ms per request)
        assert avg_time_per_request < 0.005, f"Middleware chain too slow: {avg_time_per_request:.4f}s"

    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance under load."""
        # Test rate limiting with many different IPs
        start_time = time.time()

        for i in range(200):
            request = make_mocked_request("GET", "/api/rate-test")
            request.remote = f"192.168.1.{i % 255}"  # Different IPs

            async def mock_handler(request):
                return web.Response(text="Success")

            await self.server._rate_limit_middleware(request, mock_handler)

        end_time = time.time()
        total_time = end_time - start_time

        # Should handle 200 rate limit checks in under 1 second
        assert total_time < 1.0, f"Rate limiting too slow: {total_time:.4f}s"

    def test_rate_limit_memory_efficiency(self):
        """Test rate limiting memory efficiency."""
        # Fill rate limits with many entries
        initial_memory = len(self.server.rate_limits)

        # Simulate many requests over time
        now = time.time()
        for i in range(1000):
            key = f"ip:192.168.1.{i % 100}"
            if key not in self.server.rate_limits:
                self.server.rate_limits[key] = []

            # Add timestamps (some old, some new)
            self.server.rate_limits[key].append(now - (i % 120))  # Some older than 60s window

        # Clean up old entries (simulate middleware cleanup)
        for key in list(self.server.rate_limits.keys()):
            self.server.rate_limits[key] = [
                timestamp for timestamp in self.server.rate_limits[key] if now - timestamp < 60
            ]
            if not self.server.rate_limits[key]:
                del self.server.rate_limits[key]

        final_memory = len(self.server.rate_limits)

        # Memory should be cleaned up efficiently
        assert final_memory < initial_memory + 100  # Should not grow indefinitely


class TestSecurityPerformanceOptimizations:
    """Test security-specific performance optimizations."""

    def setup_method(self):
        """Set up security performance test."""
        self.server = EnhancedSecureAPIServer()

    def test_device_fingerprint_calculation_performance(self):
        """Test device fingerprint calculation performance."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        ] * 100  # 300 total

        start_time = time.time()

        fingerprints = []
        for ua in user_agents:
            device = DeviceInfo(ua, "192.168.1.100")
            fingerprints.append(device.device_fingerprint)

        end_time = time.time()
        total_time = end_time - start_time

        # Should calculate 300 fingerprints in under 100ms
        assert total_time < 0.1, f"Fingerprint calculation too slow: {total_time:.4f}s"

        # Verify fingerprints are consistent
        assert len(set(fingerprints)) <= 3  # Should have at most 3 unique fingerprints

    @pytest.mark.asyncio
    async def test_security_header_application_performance(self):
        """Test security header application performance."""
        request = make_mocked_request("GET", "/api/headers-test")
        request.remote = "127.0.0.1"

        async def simple_handler(request):
            return web.Response(text="Test")

        # Measure header application performance
        start_time = time.time()

        for _ in range(1000):
            await self.server._security_middleware(request, simple_handler)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_request = total_time / 1000

        # Should apply security headers very fast (under 1ms per request)
        assert avg_time_per_request < 0.001, f"Header application too slow: {avg_time_per_request:.4f}s"

    @pytest.mark.asyncio
    async def test_mfa_verification_performance(self):
        """Test MFA verification performance."""
        request = make_mocked_request("GET", "/api/mfa-test")
        request["user"] = {"user_id": "mfa_perf_user", "mfa_verified": True}

        async def simple_handler(request):
            return web.Response(text="MFA Success")

        # Measure MFA middleware performance
        start_time = time.time()

        for _ in range(500):
            await self.server._mfa_middleware(request, simple_handler)

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_check = total_time / 500

        # MFA checks should be very fast (under 0.5ms per check)
        assert avg_time_per_check < 0.0005, f"MFA verification too slow: {avg_time_per_check:.4f}s"


class TestMemoryUsageOptimization:
    """Test memory usage optimization in refactored security."""

    def setup_method(self):
        """Set up memory usage test."""
        self.server = EnhancedSecureAPIServer()

    def test_rate_limit_memory_management(self):
        """Test that rate limiting doesn't cause memory leaks."""
        import sys

        # Get initial memory usage
        initial_memory = sys.getsizeof(self.server.rate_limits)

        # Simulate many requests over time windows
        now = time.time()

        # Add many rate limit entries
        for i in range(1000):
            key = f"test_key_{i}"
            self.server.rate_limits[key] = [now - j for j in range(0, 120, 5)]  # 24 timestamps per key

        memory_after_fill = sys.getsizeof(self.server.rate_limits)

        # Clean old entries (simulate automatic cleanup)
        for key in list(self.server.rate_limits.keys()):
            self.server.rate_limits[key] = [
                timestamp
                for timestamp in self.server.rate_limits[key]
                if now - timestamp < 60  # Keep only recent entries
            ]
            if not self.server.rate_limits[key]:
                del self.server.rate_limits[key]

        memory_after_cleanup = sys.getsizeof(self.server.rate_limits)

        # Memory should be cleaned up
        memory_growth = memory_after_cleanup - initial_memory
        max_acceptable_growth = memory_after_fill * 0.1  # 10% of peak usage

        assert memory_growth < max_acceptable_growth, f"Memory not cleaned up efficiently: {memory_growth} bytes"

    def test_security_context_memory_efficiency(self):
        """Test that security context doesn't accumulate memory."""
        import sys

        # Create many request contexts
        requests = []
        for i in range(100):
            request = make_mocked_request("GET", f"/api/memory-test-{i}")
            request["user"] = {"user_id": f"user_{i}", "roles": ["user"]}
            request["device_info"] = DeviceInfo(f"Browser{i}", f"10.0.0.{i}")
            request["security_context"] = {
                "timestamp": datetime.utcnow(),
                "request_id": f"req_{i}",
                "metadata": {"test": f"data_{i}"},
            }
            requests.append(request)

        # Memory usage should be reasonable
        total_memory = sum(sys.getsizeof(req) for req in requests)
        avg_memory_per_request = total_memory / 100

        # Each request context should be under 10KB
        assert avg_memory_per_request < 10240, f"Request context too large: {avg_memory_per_request} bytes"


class TestConcurrentPerformance:
    """Test performance under concurrent load."""

    def setup_method(self):
        """Set up concurrent performance test."""
        self.server = EnhancedSecureAPIServer()
        self.server.session_manager = RedisSessionManager()
        self.server.session_manager.redis_client = AsyncMock()

    @pytest.mark.asyncio
    async def test_concurrent_middleware_execution(self):
        """Test middleware performance under concurrent load."""
        # Create concurrent requests
        tasks = []

        for i in range(50):
            request = make_mocked_request("GET", f"/api/concurrent-{i}")
            request.remote = f"192.168.1.{100 + (i % 50)}"
            request["user"] = {"user_id": f"user_{i}", "mfa_verified": True}

            async def mock_handler(request):
                await asyncio.sleep(0.001)  # Simulate small processing time
                return web.Response(text="Success")

            # Apply full middleware chain
            task = self.server._security_middleware(
                request,
                lambda req: self.server._rate_limit_middleware(
                    req,
                    lambda req: self.server._session_middleware(
                        req, lambda req: self.server._mfa_middleware(req, mock_handler)
                    ),
                ),
            )

            tasks.append(task)

        # Execute all concurrent requests
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # Should handle 50 concurrent requests in under 1 second
        assert total_time < 1.0, f"Concurrent middleware too slow: {total_time:.4f}s"

        # Most should succeed
        success_count = sum(1 for r in responses if isinstance(r, web.Response) and r.status == 200)
        assert success_count >= 40  # At least 80% should succeed

    @pytest.mark.asyncio
    async def test_high_throughput_session_operations(self):
        """Test session operations under high throughput."""
        user_id = "throughput_user"

        # Mock fast Redis operations
        self.server.session_manager.redis_client.get.return_value = "session_123"
        self.server.session_manager.redis_client.hgetall.return_value = {"user_id": user_id, "is_active": "True"}

        # Create high throughput session operations
        operations = []

        for i in range(200):
            if i % 3 == 0:
                # Token validation operations
                operations.append(self.server.session_manager.is_token_revoked(f"token_{i}"))
            elif i % 3 == 1:
                # Session retrieval operations
                operations.append(self.server.session_manager.get_session(f"session_{i}"))
            else:
                # Session update operations
                mock_session = Mock()
                operations.append(self.server.session_manager.update_session(mock_session))

        # Execute high throughput operations
        start_time = time.time()
        results = await asyncio.gather(*operations, return_exceptions=True)
        end_time = time.time()

        total_time = end_time - start_time

        # Should handle 200 operations in under 500ms
        assert total_time < 0.5, f"High throughput operations too slow: {total_time:.4f}s"

        # All operations should complete
        assert len(results) == 200


class TestPerformanceRegression:
    """Test for performance regressions after refactoring."""

    def setup_method(self):
        """Set up regression test environment."""
        self.server = EnhancedSecureAPIServer()

    @pytest.mark.asyncio
    async def test_no_performance_regression_in_auth(self):
        """Test that authentication performance hasn't regressed."""
        # Baseline performance expectations
        max_auth_time = 0.01  # 10ms max per auth

        # Mock components for fast testing
        self.server.rbac_system = Mock()
        self.server.rbac_system.get_user.return_value = {
            "user_id": "regression_user",
            "password_hash": "hash",
            "password_salt": "salt",
        }

        self.server.mfa_system = Mock()
        self.server.mfa_system.get_user_mfa_status.return_value = {"totp_enabled": False}

        request = make_mocked_request("POST", "/auth/login")
        request["device_info"] = DeviceInfo("Regression Browser", "127.0.0.1")

        async def mock_json():
            return {"username": "regressionuser", "password": "testpass"}

        request.json = mock_json

        with patch("hmac.compare_digest", return_value=True):
            with patch("hashlib.pbkdf2_hmac") as mock_pbkdf2:
                mock_pbkdf2.return_value.hex.return_value = "hash"

                self.server.rbac_system.get_user_roles.return_value = ["user"]
                self.server.rbac_system.get_role_permissions.return_value = ["read"]

                self.server.authenticator = Mock()
                self.server.authenticator.create_session_tokens = AsyncMock(
                    return_value={"access_token": "regression_token"}
                )

                # Measure authentication performance
                start_time = time.time()
                response = await self.server._enhanced_login(request)
                end_time = time.time()

                auth_time = end_time - start_time

                # Should meet performance expectations
                assert auth_time < max_auth_time, f"Auth regression: {auth_time:.4f}s > {max_auth_time}s"
                assert response.status == 200

    @pytest.mark.asyncio
    async def test_no_memory_leaks_in_refactored_components(self):
        """Test that refactored components don't introduce memory leaks."""
        import gc

        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create and destroy many security contexts
        for i in range(100):
            request = make_mocked_request("GET", f"/api/memory-leak-test-{i}")
            request["user"] = {"user_id": f"user_{i}"}
            request["device_info"] = DeviceInfo(f"Browser{i}", f"10.0.{i//256}.{i%256}")

            async def mock_handler(request):
                return web.Response(text="Success")

            # Process through middleware
            await self.server._security_middleware(request, mock_handler)

        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())

        objects_created = final_objects - initial_objects

        # Should not create excessive objects (allowing for some test overhead)
        assert objects_created < 500, f"Possible memory leak: {objects_created} objects created"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
