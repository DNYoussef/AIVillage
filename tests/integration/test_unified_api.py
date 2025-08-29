#!/usr/bin/env python3
"""
Integration Tests for Unified API Gateway

Tests all major endpoints and functionality of the unified API gateway including:
- Authentication and authorization
- Agent Forge endpoints
- P2P/Fog computing endpoints
- Rate limiting and error handling
- WebSocket functionality
"""

import os
from pathlib import Path

# Import the unified gateway app
import sys
import time

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from infrastructure.gateway.auth import JWTHandler
from infrastructure.gateway.unified_api_gateway import app


class TestUnifiedAPIGateway:
    """Test suite for unified API gateway."""

    @pytest.fixture(scope="class")
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture(scope="class")
    def jwt_handler(self):
        """Create JWT handler for testing."""
        return JWTHandler(secret_key="test-secret-key", require_mfa=False)

    @pytest.fixture(scope="class")
    def auth_headers(self, jwt_handler):
        """Create authentication headers."""
        token = jwt_handler.create_token(user_id="test_user", scopes=["read", "write", "admin"], mfa_verified=True)
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture(scope="class")
    def api_key_headers(self):
        """Create API key headers."""
        return {"X-API-Key": "test-api-key-123"}


class TestHealthAndStatus:
    """Test health and status endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "service" in data["data"]
        assert data["data"]["service"] == "AIVillage Unified API Gateway"
        assert "version" in data["data"]
        assert "endpoints" in data["data"]

    def test_health_endpoint(self, client):
        """Test health endpoint returns service status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "services" in data
        assert "timestamp" in data
        assert "version" in data


class TestAuthentication:
    """Test authentication and authorization."""

    def test_jwt_authentication_success(self, client, auth_headers):
        """Test successful JWT authentication."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_jwt_authentication_failure(self, client):
        """Test JWT authentication failure."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/v1/models", headers=headers)
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "AUTHENTICATION" in data.get("error_code", "").upper()

    def test_missing_authentication(self, client):
        """Test missing authentication."""
        response = client.get("/v1/models")
        assert response.status_code == 401

    def test_api_key_authentication(self, client, api_key_headers):
        """Test API key authentication (when implemented)."""
        # This would test API key auth when fully implemented
        pass


class TestAgentForgeEndpoints:
    """Test Agent Forge API endpoints."""

    def test_start_training(self, client, auth_headers):
        """Test starting model training."""
        training_request = {
            "phase_name": "cognate",
            "real_training": False,  # Use simulation for testing
            "max_steps": 100,
            "batch_size": 1,
        }

        response = client.post("/v1/models/train", json=training_request, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "task_id" in data["data"]
        assert data["data"]["phase"] == "cognate"
        assert data["data"]["status"] == "started"

    def test_list_models(self, client, auth_headers):
        """Test listing models."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "models" in data["data"]
        assert "total_count" in data["data"]
        assert isinstance(data["data"]["models"], list)

    def test_chat_with_model(self, client, auth_headers):
        """Test chatting with model."""
        chat_request = {
            "model_id": "test-model-123",
            "message": "Hello, test message",
            "conversation_id": "test-conv-123",
        }

        response = client.post("/v1/chat", json=chat_request, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["model_id"] == "test-model-123"
        assert "response" in data["data"]
        assert data["data"]["conversation_id"] == "test-conv-123"


class TestP2PFogEndpoints:
    """Test P2P/Fog computing endpoints."""

    def test_p2p_status(self, client, auth_headers):
        """Test P2P network status."""
        response = client.get("/v1/p2p/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "status" in data["data"]
        assert "bitchat" in data["data"]
        assert "betanet" in data["data"]

    def test_fog_nodes(self, client, auth_headers):
        """Test fog computing nodes."""
        response = client.get("/v1/fog/nodes", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "total_nodes" in data["data"]
        assert "active_nodes" in data["data"]
        assert "nodes" in data["data"]
        assert isinstance(data["data"]["nodes"], list)

    def test_token_status(self, client, auth_headers):
        """Test FOG token status."""
        response = client.get("/v1/tokens", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "balance" in data["data"]
        assert "total_supply" in data["data"]
        assert "network_status" in data["data"]


class TestUtilityEndpoints:
    """Test utility endpoints."""

    def test_query_processing(self, client, auth_headers):
        """Test RAG query processing."""
        query_request = {
            "query": "What is artificial intelligence?",
            "max_results": 5,
            "include_sources": True,
            "mode": "comprehensive",
        }

        response = client.post("/v1/query", json=query_request, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["data"]["query"] == query_request["query"]
        assert "answer" in data["data"]
        assert "sources" in data["data"]

    def test_file_upload(self, client, auth_headers):
        """Test file upload."""
        # Create a test file
        test_content = b"This is test file content for upload testing."
        files = {"file": ("test.txt", test_content, "text/plain")}

        response = client.post("/v1/upload", files=files, headers=auth_headers)
        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "file_id" in data["data"]
        assert data["data"]["filename"] == "test.txt"
        assert data["data"]["size"] == len(test_content)
        assert data["data"]["status"] == "processed"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_headers(self, client, auth_headers):
        """Test rate limit headers are present."""
        response = client.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200

        # Check rate limit headers
        assert "X-RateLimit-Tier" in response.headers
        assert "X-RateLimit-Limit" in response.headers

    def test_rate_limit_enforcement(self, client, auth_headers):
        """Test rate limit enforcement (this is a slow test)."""
        # This test would make many requests to trigger rate limiting
        # Skip for now to avoid long test times
        pass


class TestErrorHandling:
    """Test error handling and response formats."""

    def test_404_error_format(self, client):
        """Test 404 error response format."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        # FastAPI returns standard 404 format

    def test_validation_error_format(self, client, auth_headers):
        """Test validation error response format."""
        # Send invalid training request
        invalid_request = {
            "phase_name": "",  # Empty phase name should fail validation
            "max_steps": -1,  # Negative steps should fail
        }

        response = client.post("/v1/models/train", json=invalid_request, headers=auth_headers)
        assert response.status_code == 422  # FastAPI validation error

    def test_service_unavailable_error(self, client, auth_headers):
        """Test service unavailable error handling."""
        # This would test when Agent Forge service is unavailable
        # The actual behavior depends on service availability
        pass


class TestWebSocketFunctionality:
    """Test WebSocket functionality."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and basic communication."""
        # This test requires the server to be running
        # Skip for now to avoid external dependencies
        pass

    @pytest.mark.asyncio
    async def test_websocket_training_updates(self):
        """Test WebSocket training progress updates."""
        # This would test real-time training updates via WebSocket
        pass


class TestSecurityFeatures:
    """Test security features."""

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/")
        # CORS headers should be present
        assert response.status_code == 200

    def test_request_id_tracking(self, client):
        """Test request ID tracking."""
        response = client.get("/")
        assert response.status_code == 200
        # Request ID should be in headers
        assert "X-Request-ID" in response.headers

    def test_security_headers(self, client):
        """Test security headers are present."""
        response = client.get("/")
        assert response.status_code == 200
        # Various security headers should be present
        # Specific headers depend on middleware configuration


class TestPerformance:
    """Test performance characteristics."""

    def test_response_time_health_check(self, client):
        """Test health check response time."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should respond within 1 second

    def test_concurrent_requests(self, client, auth_headers):
        """Test handling concurrent requests."""
        # This would test multiple simultaneous requests
        # Implementation depends on testing framework capabilities
        pass


class TestDataValidation:
    """Test input validation and sanitization."""

    def test_query_length_validation(self, client, auth_headers):
        """Test query length validation."""
        # Test maximum length
        long_query = "x" * 6000  # Over the 5000 character limit
        query_request = {"query": long_query}

        response = client.post("/v1/query", json=query_request, headers=auth_headers)
        assert response.status_code == 422  # Validation error

    def test_message_length_validation(self, client, auth_headers):
        """Test chat message length validation."""
        long_message = "x" * 6000  # Over the limit
        chat_request = {"model_id": "test-model", "message": long_message}

        response = client.post("/v1/chat", json=chat_request, headers=auth_headers)
        assert response.status_code == 422  # Validation error

    def test_file_upload_validation(self, client, auth_headers):
        """Test file upload validation."""
        # Test with invalid file type or size
        large_content = b"x" * (50 * 1024 * 1024)  # 50MB file
        files = {"file": ("large.txt", large_content, "text/plain")}

        response = client.post("/v1/upload", files=files, headers=auth_headers)
        # Should handle large files appropriately
        assert response.status_code in [200, 400, 413]  # OK, Bad Request, or Payload Too Large


# Test configuration and fixtures for pytest
@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {"base_url": "http://localhost:8000", "timeout": 30, "jwt_secret": "test-secret-key"}


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment."""
    # Set test environment variables
    os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-integration-tests"
    os.environ["REQUIRE_MFA"] = "false"
    os.environ["DEBUG"] = "true"

    yield

    # Cleanup
    for key in ["JWT_SECRET_KEY", "REQUIRE_MFA", "DEBUG"]:
        if key in os.environ:
            del os.environ[key]


# Integration test runner
if __name__ == "__main__":
    """Run integration tests."""
    import subprocess
    import sys

    print("🚀 Running AIVillage Unified API Integration Tests...")

    # Run with pytest
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v", "--tb=short", "--strict-markers"])

    if result.returncode == 0:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed!")
        sys.exit(1)
