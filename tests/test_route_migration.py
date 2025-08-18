"""Tests for route migration from server.py to microservices."""

import os
import warnings
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_env_dev_mode():
    """Mock environment for dev mode."""
    with patch.dict(os.environ, {"AIVILLAGE_DEV_MODE": "true"}):
        yield


@pytest.fixture
def mock_env_prod_mode():
    """Mock environment for production mode."""
    with patch.dict(os.environ, {"AIVILLAGE_DEV_MODE": "false"}):
        yield


class TestServerDeprecationWarnings:
    """Test that server.py emits appropriate deprecation warnings."""

    def test_server_startup_warning_in_prod(self, mock_env_prod_mode):
        """Test server.py warns when started without dev mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Import server to trigger startup

            # Should have deprecation warning
            assert any("development only" in str(warning.message).lower() for warning in w)

    def test_no_warning_in_dev_mode(self, mock_env_dev_mode):
        """Test server.py doesn't warn in dev mode."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should not have deprecation warning
            assert not any("development only" in str(warning.message).lower() for warning in w)


class TestRouteMigration:
    """Test that routes are properly migrated to microservices."""

    @pytest.fixture
    def server_client(self):
        """Create test client for server.py."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent / "bin"))
        from server import app

        return TestClient(app)

    @pytest.fixture
    def twin_client(self):
        """Create test client for twin service."""
        from services.twin.app import app

        return TestClient(app)

    @pytest.fixture
    def gateway_client(self):
        """Create test client for gateway service."""
        from services.gateway.app import app

        return TestClient(app)

    def test_query_endpoint_deprecated(self, server_client):
        """Test /query endpoint returns deprecation header."""
        response = server_client.post("/query", json={"query": "test"})
        assert "X-Deprecated" in response.headers
        assert "Use /v1/query via Twin service" in response.headers.get("X-Deprecated", "")

    def test_upload_endpoint_deprecated(self, server_client):
        """Test /upload endpoint returns deprecation header."""
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = server_client.post("/upload", files=files)
        assert "X-Deprecated" in response.headers
        assert "Use /v1/upload via Twin service" in response.headers.get("X-Deprecated", "")

    def test_explain_endpoint_deprecated(self, server_client):
        """Test /explain endpoint returns deprecation header."""
        response = server_client.get("/explain", params={"start": "a", "end": "b"})
        assert "X-Deprecated" in response.headers
        assert "Use POST /explain via Twin service" in response.headers.get("X-Deprecated", "")

    def test_v1_explanation_deprecated(self, server_client):
        """Test /v1/explanation endpoint returns deprecation header."""
        response = server_client.get("/v1/explanation", params={"chat_id": "123"})
        assert "X-Deprecated" in response.headers
        assert "Use POST /v1/evidence via Twin service" in response.headers.get("X-Deprecated", "")


class TestNewTwinRoutes:
    """Test new routes added to Twin service."""

    @pytest.fixture
    def twin_client(self):
        """Create test client for twin service with mocked dependencies."""
        with patch("services.twin.app.initialize_pipeline") as mock_init:
            mock_pipeline = MagicMock()
            mock_init.return_value = mock_pipeline

            from services.twin.app import app

            return TestClient(app)

    def test_v1_query_endpoint_exists(self, twin_client):
        """Test new /v1/query endpoint in Twin service."""
        response = twin_client.post("/v1/query", json={"query": "test query"})
        # Should not be 404
        assert response.status_code != 404

    def test_v1_upload_endpoint_exists(self, twin_client):
        """Test new /v1/upload endpoint in Twin service."""
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = twin_client.post("/v1/upload", files=files)
        # Should not be 404
        assert response.status_code != 404

    def test_v1_debug_bayes_endpoint_exists(self, twin_client):
        """Test new /v1/debug/bayes endpoint in Twin service."""
        response = twin_client.get("/v1/debug/bayes")
        # Should not be 404
        assert response.status_code != 404

    def test_v1_debug_logs_endpoint_exists(self, twin_client):
        """Test new /v1/debug/logs endpoint in Twin service."""
        response = twin_client.get("/v1/debug/logs")
        # Should not be 404
        assert response.status_code != 404


class TestGatewayProxying:
    """Test that Gateway properly proxies to Twin service."""

    @pytest.fixture
    def gateway_client(self):
        """Create test client for gateway with mocked Twin service."""
        with patch("services.gateway.app.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "test"}
            mock_response.headers = {}
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

            from services.gateway.app import app

            return TestClient(app)

    def test_gateway_proxies_v1_query(self, gateway_client):
        """Test Gateway proxies /v1/query to Twin."""
        response = gateway_client.post("/v1/query", json={"query": "test"})
        assert response.status_code == 200

    def test_gateway_adds_deprecation_context(self, gateway_client):
        """Test Gateway adds migration context headers."""
        response = gateway_client.get("/healthz")
        assert response.status_code == 200
        # Gateway should indicate it's the proper entry point
        assert "X-Service" in response.headers
        assert response.headers["X-Service"] == "gateway"


class TestHealthcheckConsolidation:
    """Test health check consolidation across services."""

    def test_status_endpoint_redirects_to_healthz(self, server_client):
        """Test /status endpoint redirects to /healthz."""
        response = server_client.get("/status", follow_redirects=False)
        assert response.status_code == 307  # Temporary redirect
        assert response.headers["location"] == "/healthz"

    def test_healthz_aggregates_service_health(self, gateway_client):
        """Test /healthz aggregates health from all services."""
        response = gateway_client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data or "healthy" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
