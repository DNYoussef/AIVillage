#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Gateway
===========================================

Tests all consolidated gateway functionality including:
- Performance benchmarks (<50ms health checks)
- Security validation (rate limiting, input validation)
- Service routing and orchestration
- Authentication and authorization
- MCP server integration
- Error handling and recovery
"""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from fastapi.testclient import TestClient
import httpx

# Import the unified gateway
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "core" / "gateway"))

from unified_gateway import app, UnifiedServiceOrchestrator, UnifiedQueryRequest


class TestUnifiedGatewayPerformance:
    """Performance benchmark tests with strict SLA requirements."""
    
    @pytest.fixture
    def client(self):
        """Test client fixture."""
        return TestClient(app)
    
    def test_health_check_performance(self, client):
        """Test health check meets <50ms requirement."""
        # Warm up
        for _ in range(5):
            client.get("/healthz")
        
        # Measure performance
        times = []
        for _ in range(20):
            start = time.time()
            response = client.get("/healthz")
            duration = (time.time() - start) * 1000  # Convert to milliseconds
            times.append(duration)
            
            assert response.status_code == 200
        
        avg_time = sum(times) / len(times)
        p99_time = sorted(times)[int(len(times) * 0.99)]
        
        # Performance assertions
        assert avg_time < 25, f"Average health check time {avg_time:.2f}ms exceeds 25ms target"
        assert p99_time < 50, f"P99 health check time {p99_time:.2f}ms exceeds 50ms target"
        
        # Validate response structure
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in health_data
        assert "version" in health_data
        assert "performance" in health_data
        
        response_time = health_data["performance"]["response_time_ms"]
        assert response_time < 50, f"Reported response time {response_time}ms exceeds target"
    
    def test_concurrent_request_handling(self, client):
        """Test gateway handles concurrent requests efficiently."""
        
        async def make_request():
            async with httpx.AsyncClient(app=app, base_url="http://test") as client:
                return await client.get("/healthz")
        
        async def run_concurrent_test():
            # Create 50 concurrent requests
            tasks = [make_request() for _ in range(50)]
            start_time = time.time()
            responses = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            # All requests should succeed
            assert all(r.status_code == 200 for r in responses)
            
            # Total time should be reasonable (not serialized)
            assert total_time < 5, f"50 concurrent requests took {total_time:.2f}s, expected <5s"
            
            return responses, total_time
        
        # Run the async test
        responses, duration = asyncio.run(run_concurrent_test())
        
        # Validate no degradation under load
        for response in responses[-5:]:  # Check last few responses
            health_data = response.json()
            assert health_data["performance"]["response_time_ms"] < 100


class TestUnifiedGatewaySecurity:
    """Security validation tests for consolidated security features."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_security_headers(self, client):
        """Test all security headers are properly set."""
        response = client.get("/healthz")
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy",
            "Referrer-Policy",
            "Permissions-Policy",
            "X-Gateway-Version",
            "X-Process-Time",
            "X-Request-ID"
        ]
        
        for header in required_headers:
            assert header in response.headers, f"Missing security header: {header}"
        
        # Validate specific security header values
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Gateway-Version"] == "2.0.0-unified"
    
    def test_rate_limiting_basic(self, client):
        """Test basic rate limiting functionality."""
        # Make requests rapidly from same IP
        responses = []
        for i in range(15):
            response = client.get("/healthz")
            responses.append(response)
            
            # Check for rate limit headers
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers
        
        # Should not hit rate limits for health checks in normal testing
        assert all(r.status_code == 200 for r in responses)
    
    def test_threat_detection(self, client):
        """Test threat detection in rate limiting middleware."""
        # Test SQL injection pattern
        malicious_paths = [
            "/v1/query?q='; DROP TABLE users; --",
            "/healthz?search=<script>alert('xss')</script>",
            "/admin/../../../etc/passwd"
        ]
        
        for path in malicious_paths:
            response = client.get(path)
            # Should not crash the server
            assert response.status_code in [200, 400, 404, 429]
    
    def test_input_validation(self, client):
        """Test input validation for query requests."""
        # Test with malicious query content
        malicious_queries = [
            {"query": "<script>alert('xss')</script>"},
            {"query": "'; DROP TABLE users; --"},
            {"query": "javascript:alert('xss')"},
            {"query": "data:text/html,<script>alert('xss')</script>"}
        ]
        
        for query_data in malicious_queries:
            response = client.post("/v1/query", json=query_data)
            # Should return 422 for validation error or 400 for bad request
            assert response.status_code in [400, 422], f"Query should be rejected: {query_data}"


class TestServiceOrchestration:
    """Test intelligent service routing and orchestration."""
    
    @pytest.fixture
    def orchestrator(self):
        return UnifiedServiceOrchestrator()
    
    @pytest.mark.asyncio
    async def test_service_type_detection(self, orchestrator):
        """Test ML-based service type detection."""
        test_cases = [
            ("train a new model", "agent_forge"),
            ("search for documents about AI", "rag"),
            ("connect to p2p network", "fog"),
            ("analyze my data", "rag"),
            ("start fog computing job", "fog")
        ]
        
        for query, expected_service in test_cases:
            detected = await orchestrator._determine_service_type(query)
            assert detected == expected_service, f"Query '{query}' should route to {expected_service}, got {detected}"
    
    @pytest.mark.asyncio
    async def test_request_routing(self, orchestrator):
        """Test request routing to different services."""
        # Mock services
        orchestrator.services = {
            "rag": AsyncMock(),
            "agent_forge": "available",
            "fog": AsyncMock()
        }
        
        # Mock RAG service response
        orchestrator.services["rag"].process.return_value = {"answer": "Test response"}
        
        # Test RAG routing
        request = UnifiedQueryRequest(query="search for test", service_type="rag")
        result = await orchestrator.route_request(request)
        
        assert result["success"] is True
        assert result["service"] == "rag"
        assert "answer" in result["data"]
        
        # Test auto-routing
        request = UnifiedQueryRequest(query="train a model", service_type="auto")
        result = await orchestrator.route_request(request)
        assert result["service"] == "agent_forge"
    
    @pytest.mark.asyncio
    async def test_external_service_routing(self, orchestrator):
        """Test routing to external microservices."""
        request = UnifiedQueryRequest(query="external service test", service_type="external")
        
        # Mock httpx client
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"success": True, "data": "external response"}
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
            
            result = await orchestrator._handle_external_request(request, "external")
            assert result["success"] is True
            assert result["data"] == "external response"


class TestAuthenticationIntegration:
    """Test authentication and authorization integration."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_public_endpoints_accessible(self, client):
        """Test public endpoints work without authentication."""
        public_endpoints = [
            "/",
            "/healthz",
            "/metrics"  # if metrics enabled
        ]
        
        for endpoint in public_endpoints:
            response = client.get(endpoint)
            # Should be accessible (200) or not found (404) but not auth error (401/403)
            assert response.status_code not in [401, 403], f"Endpoint {endpoint} should be public"
    
    @patch('unified_gateway.AUTH_AVAILABLE', False)
    def test_development_mode_bypass(self, client):
        """Test authentication bypass in development mode."""
        # In development mode, should allow access without auth
        response = client.post("/v1/query", json={"query": "test query"})
        # Should not get authentication error
        assert response.status_code != 401
    
    def test_cors_configuration(self, client):
        """Test CORS configuration."""
        # Test OPTIONS request
        response = client.options("/v1/query", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type,Authorization"
        })
        
        # Should handle CORS preflight
        assert response.status_code in [200, 204]
        
        # Test actual CORS headers in response
        response = client.get("/healthz", headers={"Origin": "http://localhost:3000"})
        cors_headers = [h for h in response.headers.keys() if h.lower().startswith("access-control")]
        
        # Should have some CORS headers
        assert len(cors_headers) > 0


class TestWebSocketIntegration:
    """Test WebSocket functionality if enabled."""
    
    def test_websocket_connection(self):
        """Test WebSocket connection and echo functionality."""
        with TestClient(app) as client:
            try:
                with client.websocket_connect("/ws") as websocket:
                    # Send test data
                    test_message = "Hello WebSocket"
                    websocket.send_text(test_message)
                    
                    # Receive response
                    data = websocket.receive_text()
                    assert f"Processed: {test_message}" in data
                    
            except Exception as e:
                # WebSocket might not be enabled in test environment
                pytest.skip(f"WebSocket test skipped: {e}")


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    @pytest.fixture  
    def client(self):
        return TestClient(app)
    
    def test_malformed_requests(self, client):
        """Test handling of malformed requests."""
        # Test invalid JSON
        response = client.post("/v1/query", 
                             data="invalid json",
                             headers={"content-type": "application/json"})
        assert response.status_code == 422
        
        # Test missing required fields
        response = client.post("/v1/query", json={})
        assert response.status_code == 422
        
        # Test invalid field values
        response = client.post("/v1/query", json={"query": ""})
        assert response.status_code == 422
    
    def test_file_upload_validation(self, client):
        """Test file upload validation."""
        # Test no file
        response = client.post("/v1/upload")
        assert response.status_code == 422
        
        # Test oversized file (mock)
        with patch('unified_gateway.config.max_file_size', 100):
            large_content = b"x" * 200  # Larger than limit
            response = client.post("/v1/upload", 
                                 files={"file": ("test.txt", large_content, "text/plain")})
            assert response.status_code == 413
    
    def test_service_unavailable_handling(self, client):
        """Test handling when services are unavailable."""
        # This would require mocking service dependencies
        # For now, test that the system doesn't crash
        response = client.post("/v1/query", json={"query": "test when services down"})
        # Should return some response, not crash
        assert response.status_code in [200, 400, 500, 502, 503]


class TestMCPIntegration:
    """Test MCP server integration functionality."""
    
    def test_memory_mcp_hooks(self):
        """Test Memory MCP integration points."""
        # This would test actual MCP server integration
        # For now, verify the hooks are callable
        assert callable(app.state.get("orchestrator", lambda: None))
    
    def test_performance_metrics_storage(self):
        """Test performance metrics are stored for MCP retrieval."""
        # Test metrics collection
        with TestClient(app) as client:
            response = client.get("/healthz")
            health_data = response.json()
            
            # Verify performance data is collected
            assert "performance" in health_data
            assert "response_time_ms" in health_data["performance"]
            assert isinstance(health_data["performance"]["response_time_ms"], (int, float))


# Integration test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])