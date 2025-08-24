"""
Integration tests for retries, circuit breakers, and idempotency.

These tests verify that external call reliability features work correctly
under various failure scenarios.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from packages.core.common import (
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    DependencyOutageSimulator,
    IdempotencyConfig,
    ResilientHttpClient,
    RetryConfig,
)


class TestRetryLogic:
    """Test retry logic with exponential backoff and jitter."""

    @pytest.mark.asyncio
    async def test_successful_request_no_retry(self):
        """Test that successful requests don't trigger retries."""
        client = ResilientHttpClient()

        # Mock successful response
        client.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        client.client.request.return_value = mock_response

        response = await client.get("https://api.example.com/test")

        assert response.status_code == 200
        assert client.client.request.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Test retry behavior on 5xx errors."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        client = ResilientHttpClient(retry_config=retry_config)

        # Mock responses: first two fail, third succeeds
        client.client = AsyncMock()
        mock_responses = [
            MagicMock(status_code=503),  # Service unavailable
            MagicMock(status_code=502),  # Bad gateway
            MagicMock(status_code=200),  # Success
        ]
        client.client.request.side_effect = mock_responses

        response = await client.get("https://api.example.com/test")

        assert response.status_code == 200
        assert client.client.request.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_exhaust_all_retries(self):
        """Test behavior when all retries are exhausted."""
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        client = ResilientHttpClient(retry_config=retry_config)

        # Mock responses: all fail
        client.client = AsyncMock()
        mock_response = MagicMock(status_code=500)
        client.client.request.return_value = mock_response

        response = await client.get("https://api.example.com/test")

        assert response.status_code == 500  # Last response returned
        assert client.client.request.call_count == 2

        await client.close()

    @pytest.mark.asyncio
    async def test_retry_with_timeout_exception(self):
        """Test retry behavior with timeout exceptions."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        client = ResilientHttpClient(retry_config=retry_config)

        # Mock timeout then success
        client.client = AsyncMock()
        client.client.request.side_effect = [
            httpx.TimeoutException("Request timeout"),
            httpx.TimeoutException("Request timeout"),
            MagicMock(status_code=200),
        ]

        response = await client.get("https://api.example.com/test")

        assert response.status_code == 200
        assert client.client.request.call_count == 3

        await client.close()

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Test that 4xx errors don't trigger retries."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        client = ResilientHttpClient(retry_config=retry_config)

        client.client = AsyncMock()
        mock_response = MagicMock(status_code=404)  # Not found
        client.client.request.return_value = mock_response

        response = await client.get("https://api.example.com/test")

        assert response.status_code == 404
        assert client.client.request.call_count == 1  # No retries

        await client.close()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        circuit_config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        client = ResilientHttpClient(circuit_config=circuit_config)

        # Mock failing requests
        client.client = AsyncMock()
        client.client.request.side_effect = httpx.RequestError("Connection failed")

        # First two requests should fail but go through
        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")

        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")

        # Third request should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerError):
            await client.get("https://api.example.com/test")

        # Check circuit breaker state
        status = client.get_circuit_breaker_status()
        service_status = next(iter(status.values()))
        assert service_status["state"] == CircuitState.OPEN.value

        await client.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        circuit_config = CircuitBreakerConfig(failure_threshold=2, success_threshold=1, timeout=0.1)
        client = ResilientHttpClient(circuit_config=circuit_config)

        # Mock failures then success
        client.client = AsyncMock()
        failure_response = httpx.RequestError("Connection failed")
        success_response = MagicMock(status_code=200)

        # Trip circuit breaker
        client.client.request.side_effect = [failure_response, failure_response]

        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")
        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")

        # Circuit should be open
        with pytest.raises(CircuitBreakerError):
            await client.get("https://api.example.com/test")

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Should transition to half-open and succeed
        client.client.request.side_effect = [success_response]
        response = await client.get("https://api.example.com/test")
        assert response.status_code == 200

        # Circuit should now be closed
        status = client.get_circuit_breaker_status()
        service_status = next(iter(status.values()))
        assert service_status["state"] == CircuitState.CLOSED.value

        await client.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_per_service(self):
        """Test that circuit breakers are isolated per service."""
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        client = ResilientHttpClient(circuit_config=circuit_config)

        # Mock different behaviors for different services
        client.client = AsyncMock()

        def mock_request(method, url, **kwargs):
            if "service-a" in url:
                raise httpx.RequestError("Service A down")
            else:
                return MagicMock(status_code=200)

        client.client.request.side_effect = mock_request

        # Trip circuit breaker for service A
        with pytest.raises(httpx.RequestError):
            await client.get("https://service-a.example.com/test")

        # Service A should be blocked
        with pytest.raises(CircuitBreakerError):
            await client.get("https://service-a.example.com/test")

        # Service B should still work
        response = await client.get("https://service-b.example.com/test")
        assert response.status_code == 200

        # Verify separate circuit breaker states
        status = client.get_circuit_breaker_status()
        assert len(status) == 2  # Two different services

        await client.close()


class TestIdempotency:
    """Test idempotency key handling."""

    @pytest.mark.asyncio
    async def test_idempotency_key_caching(self):
        """Test that responses are cached by idempotency key."""
        client = ResilientHttpClient()

        # Mock response
        client.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123, "name": "test"}
        mock_response.headers = {"Content-Type": "application/json"}
        client.client.request.return_value = mock_response

        # First request
        response1 = await client.post(
            "https://api.example.com/users", json={"name": "test"}, idempotency_key="create-user-123"
        )

        # Second request with same key should return cached response
        response2 = await client.post(
            "https://api.example.com/users", json={"name": "test"}, idempotency_key="create-user-123"
        )

        # Should only call the API once
        assert client.client.request.call_count == 1
        assert response1.status_code == 201
        assert response2.status_code == 201

        await client.close()

    @pytest.mark.asyncio
    async def test_auto_generated_idempotency_key(self):
        """Test automatic generation of idempotency keys."""
        idempotency_config = IdempotencyConfig(auto_generate=True)
        client = ResilientHttpClient(idempotency_config=idempotency_config)

        # Mock response
        client.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123}
        mock_response.headers = {}
        client.client.request.return_value = mock_response

        # Same request data should generate same key
        await client.post("https://api.example.com/users", json={"name": "test", "email": "test@example.com"})

        await client.post("https://api.example.com/users", json={"name": "test", "email": "test@example.com"})

        # Should be deduplicated
        assert client.client.request.call_count == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_different_requests_different_keys(self):
        """Test that different requests generate different keys."""
        idempotency_config = IdempotencyConfig(auto_generate=True)
        client = ResilientHttpClient(idempotency_config=idempotency_config)

        client.client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 123}
        mock_response.headers = {}
        client.client.request.return_value = mock_response

        # Different request data
        await client.post("https://api.example.com/users", json={"name": "user1"})
        await client.post("https://api.example.com/users", json={"name": "user2"})

        # Should not be deduplicated
        assert client.client.request.call_count == 2

        await client.close()


class TestDependencyOutageSimulation:
    """Test dependency outage simulation for testing."""

    @pytest.mark.asyncio
    async def test_dependency_outage_simulator(self):
        """Test that dependency outage simulator works correctly."""
        client = ResilientHttpClient()

        # Normal request should work
        client.client = AsyncMock()
        client.client.request.return_value = MagicMock(status_code=200)

        response = await client.get("https://api.example.com/test")
        assert response.status_code == 200

        # Simulate outage for specific service
        async with DependencyOutageSimulator(client, "api.example.com"):
            with pytest.raises(httpx.RequestError):
                await client.get("https://api.example.com/test")

            # Other services should still work
            response = await client.get("https://other-api.example.com/test")
            assert response.status_code == 200

        # After simulation, service should work again
        response = await client.get("https://api.example.com/test")
        assert response.status_code == 200

        await client.close()


class TestIntegrationScenarios:
    """Integration tests combining multiple features."""

    @pytest.mark.asyncio
    async def test_retry_then_circuit_breaker(self):
        """Test that retries work before circuit breaker opens."""
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        circuit_config = CircuitBreakerConfig(failure_threshold=3, timeout=0.2)

        client = ResilientHttpClient(retry_config=retry_config, circuit_config=circuit_config)

        # Mock failing requests
        client.client = AsyncMock()
        client.client.request.side_effect = httpx.RequestError("Connection failed")

        # First call: 2 attempts (1 original + 1 retry)
        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")
        assert client.client.request.call_count == 2

        # Second call: 1 more failure reaches threshold
        client.client.request.reset_mock()
        client.client.request.side_effect = httpx.RequestError("Connection failed")

        with pytest.raises(httpx.RequestError):
            await client.get("https://api.example.com/test")
        assert client.client.request.call_count == 1  # Circuit opens after 1st failure

        # Third call: circuit breaker blocks immediately
        client.client.request.reset_mock()
        with pytest.raises(CircuitBreakerError):
            await client.get("https://api.example.com/test")
        assert client.client.request.call_count == 0  # No actual call made

        await client.close()

    @pytest.mark.asyncio
    async def test_idempotency_with_retries(self):
        """Test idempotency works correctly with retries."""
        retry_config = RetryConfig(max_attempts=3, base_delay=0.1)
        client = ResilientHttpClient(retry_config=retry_config)

        # Mock responses: first call fails then succeeds on retry
        client.client = AsyncMock()
        responses = [
            MagicMock(status_code=503),  # Fail
            MagicMock(status_code=201, json=lambda: {"id": 123}, headers={}),  # Success
        ]
        client.client.request.side_effect = responses

        # First request with idempotency key
        response1 = await client.post(
            "https://api.example.com/users", json={"name": "test"}, idempotency_key="create-user-456"
        )
        assert response1.status_code == 201
        assert client.client.request.call_count == 2  # Initial fail + retry success

        # Second request should return cached response
        client.client.request.reset_mock()
        response2 = await client.post(
            "https://api.example.com/users", json={"name": "test"}, idempotency_key="create-user-456"
        )
        assert response2.status_code == 201
        assert client.client.request.call_count == 0  # No API call, returned cached

        await client.close()

    @pytest.mark.asyncio
    async def test_end_to_end_reliability_scenario(self):
        """End-to-end test of all reliability features."""
        retry_config = RetryConfig(max_attempts=2, base_delay=0.1)
        circuit_config = CircuitBreakerConfig(failure_threshold=2, timeout=0.2)
        idempotency_config = IdempotencyConfig(auto_generate=True)

        client = ResilientHttpClient(
            retry_config=retry_config, circuit_config=circuit_config, idempotency_config=idempotency_config
        )

        # Scenario: Service initially fails, then recovers
        client.client = AsyncMock()

        # Phase 1: Service fails consistently (opens circuit breaker)
        client.client.request.side_effect = httpx.RequestError("Service down")

        with pytest.raises(httpx.RequestError):
            await client.post("https://api.example.com/orders", json={"item": "widget"})

        with pytest.raises(httpx.RequestError):
            await client.post("https://api.example.com/orders", json={"item": "gadget"})

        # Circuit should be open now
        with pytest.raises(CircuitBreakerError):
            await client.post("https://api.example.com/orders", json={"item": "tool"})

        # Phase 2: Wait for circuit breaker timeout
        await asyncio.sleep(0.25)

        # Phase 3: Service recovers
        success_response = MagicMock()
        success_response.status_code = 201
        success_response.json.return_value = {"order_id": "12345"}
        success_response.headers = {"Content-Type": "application/json"}

        client.client.request.side_effect = [success_response]

        # Should succeed and close circuit breaker
        response = await client.post("https://api.example.com/orders", json={"item": "widget"})
        assert response.status_code == 201

        # Verify circuit is closed
        status = client.get_circuit_breaker_status()
        service_status = next(iter(status.values()))
        assert service_status["state"] == CircuitState.CLOSED.value

        # Phase 4: Duplicate request should be deduplicated
        client.client.request.reset_mock()
        response2 = await client.post("https://api.example.com/orders", json={"item": "widget"})
        assert response2.status_code == 201
        assert client.client.request.call_count == 0  # Deduplicated

        await client.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
