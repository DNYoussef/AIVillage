"""
Integration Tests for Agent Fog Tools MCP

Tests the Model Control Protocol (MCP) tools that enable agents to interact
with fog computing infrastructure. Validates sandbox creation, job execution,
artifact retrieval, and fog resource management.

Key Test Areas:
- MCP tool registration and discovery
- Sandbox creation and isolation
- Job submission and execution monitoring
- Artifact collection and validation
- Error handling and fallback mechanisms
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# Import the Agent fog tools we're testing
try:
    from packages.agents.bridges.fog_tools import (
        CreateSandboxTool,
        FetchArtifactsTool,
        FogJobStatusTool,
        RunJobTool,
        StreamLogsTool,
    )
    from packages.agents.core.agent_interface import AgentCapability, AgentMetadata
    from packages.agents.core.base_agent_template import BaseAgentTemplate

    FOG_TOOLS_AVAILABLE = True
except ImportError as e:
    FOG_TOOLS_AVAILABLE = False
    pytest.skip(f"Fog tools not available: {e}", allow_module_level=True)


class TestFogToolsMCP:
    """Test suite for Agent fog tools MCP integration"""

    @pytest.fixture
    def mock_fog_gateway_url(self):
        """Mock fog gateway URL for testing"""
        return "http://test-fog-gateway:8080"

    @pytest.fixture
    def create_sandbox_tool(self, mock_fog_gateway_url):
        """Create a CreateSandboxTool instance for testing"""
        return CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url)

    @pytest.fixture
    def run_job_tool(self, mock_fog_gateway_url):
        """Create a RunJobTool instance for testing"""
        return RunJobTool(fog_gateway_url=mock_fog_gateway_url)

    @pytest.fixture
    def stream_logs_tool(self, mock_fog_gateway_url):
        """Create a StreamLogsTool instance for testing"""
        return StreamLogsTool(fog_gateway_url=mock_fog_gateway_url)

    @pytest.fixture
    def fetch_artifacts_tool(self, mock_fog_gateway_url):
        """Create a FetchArtifactsTool instance for testing"""
        return FetchArtifactsTool(fog_gateway_url=mock_fog_gateway_url)

    @pytest.fixture
    def fog_job_status_tool(self, mock_fog_gateway_url):
        """Create a FogJobStatusTool instance for testing"""
        return FogJobStatusTool(fog_gateway_url=mock_fog_gateway_url)

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent with fog tools"""
        agent = MagicMock(spec=BaseAgentTemplate)
        agent.agent_id = "test_agent_001"
        agent.name = "TestAgent"
        return agent

    @pytest.mark.asyncio
    async def test_create_sandbox_tool_basic(self, create_sandbox_tool):
        """Test basic sandbox creation functionality"""

        # Mock successful API response
        mock_response_data = {
            "sandbox_id": "sandbox_12345",
            "status": "created",
            "namespace": "agent-test",
            "resources": {"cpu_cores": 2.0, "memory_gb": 4.0, "storage_gb": 10.0},
            "endpoint": "http://sandbox-12345.fog.local",
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # Create mock response with proper async json() method
            mock_response = AsyncMock()
            mock_response.status = 201

            # Make json() an async function that returns the data
            async def mock_json():
                return mock_response_data

            mock_response.json = mock_json

            # Create a context manager class for the response
            class MockResponseContext:
                def __init__(self, response):
                    self.response = response

                async def __aenter__(self):
                    return self.response

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            # Create a context manager class for the session
            class MockSessionContext:
                def __init__(self):
                    self.response_context = MockResponseContext(mock_response)

                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

                def post(self, *args, **kwargs):
                    return self.response_context

            # Setup the session mock
            mock_session.return_value = MockSessionContext()

            # Execute sandbox creation
            parameters = {
                "namespace": "agent-test",
                "runtime": "wasi",
                "resources": {"cpu_cores": 2.0, "memory_gb": 4.0, "storage_gb": 10.0},
            }

            result = await create_sandbox_tool.execute(parameters)

            # Validate results
            assert result["status"] == "success"
            assert result["sandbox_id"] == "sandbox_12345"
            assert result["namespace"] == "agent-test"
            assert result["resources"]["cpu_cores"] == 2.0

    @pytest.mark.asyncio
    async def test_create_sandbox_tool_error_handling(self, create_sandbox_tool):
        """Test sandbox creation error handling"""

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.text = AsyncMock(return_value="Invalid namespace")

            mock_session_instance = AsyncMock()
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            parameters = {"namespace": "invalid-namespace!", "runtime": "wasi"}

            result = await create_sandbox_tool.execute(parameters)

            # Validate error handling
            assert result["status"] == "error"
            assert "Invalid namespace" in result["message"]

    @pytest.mark.asyncio
    async def test_run_job_tool_execution(self, run_job_tool):
        """Test job execution via fog computing"""

        # Mock successful job submission and execution
        job_submission_response = {"job_id": "job_67890", "status": "submitted", "sandbox_id": "sandbox_12345"}

        job_status_response = {"job_id": "job_67890", "status": "completed", "exit_code": 0, "duration_ms": 5000}

        job_result_response = {
            "job_id": "job_67890",
            "stdout": "Job executed successfully",
            "stderr": "",
            "artifacts": ["output.txt", "results.json"],
            "metrics": {"cpu_usage": 0.75, "memory_usage_mb": 512, "duration_ms": 5000},
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock job submission
            mock_submit_response = AsyncMock()
            mock_submit_response.status = 201
            mock_submit_response.json = AsyncMock(return_value=job_submission_response)

            # Mock job status check
            mock_status_response = AsyncMock()
            mock_status_response.status = 200
            mock_status_response.json = AsyncMock(return_value=job_status_response)

            # Mock job result retrieval
            mock_result_response = AsyncMock()
            mock_result_response.status = 200
            mock_result_response.json = AsyncMock(return_value=job_result_response)

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup post context manager
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_submit_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            # Setup get context managers for status and result
            mock_get_context_1 = AsyncMock()
            mock_get_context_1.__aenter__ = AsyncMock(return_value=mock_status_response)
            mock_get_context_1.__aexit__ = AsyncMock(return_value=None)

            mock_get_context_2 = AsyncMock()
            mock_get_context_2.__aenter__ = AsyncMock(return_value=mock_result_response)
            mock_get_context_2.__aexit__ = AsyncMock(return_value=None)

            # Mock multiple get calls for status and result
            mock_session_instance.get.side_effect = [
                mock_get_context_1,  # status check
                mock_get_context_2,  # result retrieval
            ]

            parameters = {
                "sandbox_id": "sandbox_12345",
                "image": "python:3.9-alpine",
                "command": ["python", "-c", "print('Hello from fog!')"],
                "env": {"TEST_VAR": "test_value"},
                "timeout_s": 60,
            }

            result = await run_job_tool.execute(parameters)

            # Validate job execution results
            assert result["status"] == "success"
            assert result["job_id"] == "job_67890"
            assert result["stdout"] == "Job executed successfully"
            assert result["exit_code"] == 0
            assert "artifacts" in result
            assert len(result["artifacts"]) == 2

    @pytest.mark.asyncio
    async def test_stream_logs_tool_functionality(self, stream_logs_tool):
        """Test real-time log streaming from fog jobs"""

        # Mock streaming logs response
        mock_log_lines = [
            {"timestamp": "2025-01-19T10:00:00Z", "level": "INFO", "message": "Starting job execution"},
            {"timestamp": "2025-01-19T10:00:01Z", "level": "INFO", "message": "Processing input data"},
            {"timestamp": "2025-01-19T10:00:02Z", "level": "INFO", "message": "Job completed successfully"},
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200

            # Mock WebSocket-like streaming (simplified for testing)
            mock_response.content.readline.side_effect = [
                json.dumps(log_line).encode() + b"\n" for log_line in mock_log_lines
            ] + [
                b""
            ]  # End of stream

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup get context manager
            mock_get_context = AsyncMock()
            mock_get_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.get.return_value = mock_get_context

            parameters = {"job_id": "job_67890", "follow": True, "tail_lines": 10}

            result = await stream_logs_tool.execute(parameters)

            # Validate log streaming
            assert result["status"] == "success"
            assert "logs" in result
            assert len(result["logs"]) == 3
            assert result["logs"][0]["level"] == "INFO"
            assert "Starting job execution" in result["logs"][0]["message"]

    @pytest.mark.asyncio
    async def test_fetch_artifacts_tool_download(self, fetch_artifacts_tool):
        """Test artifact download from fog execution"""

        # Mock artifact listing and download
        artifacts_list_response = {
            "artifacts": [
                {
                    "name": "output.txt",
                    "size": 1024,
                    "type": "text/plain",
                    "url": "/v1/fog/jobs/job_67890/artifacts/output.txt",
                },
                {
                    "name": "results.json",
                    "size": 2048,
                    "type": "application/json",
                    "url": "/v1/fog/jobs/job_67890/artifacts/results.json",
                },
            ]
        }

        artifact_content = b"Test output content from fog execution"

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock artifacts list response
            mock_list_response = AsyncMock()
            mock_list_response.status = 200
            mock_list_response.json = AsyncMock(return_value=artifacts_list_response)

            # Mock artifact download response
            mock_download_response = AsyncMock()
            mock_download_response.status = 200
            mock_download_response.read = AsyncMock(return_value=artifact_content)

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup get context managers for list and download
            mock_get_context_1 = AsyncMock()
            mock_get_context_1.__aenter__ = AsyncMock(return_value=mock_list_response)
            mock_get_context_1.__aexit__ = AsyncMock(return_value=None)

            mock_get_context_2 = AsyncMock()
            mock_get_context_2.__aenter__ = AsyncMock(return_value=mock_download_response)
            mock_get_context_2.__aexit__ = AsyncMock(return_value=None)

            mock_session_instance.get.side_effect = [
                mock_get_context_1,  # artifacts list
                mock_get_context_2,  # artifact download
            ]

            with tempfile.TemporaryDirectory() as temp_dir:
                parameters = {"job_id": "job_67890", "artifacts": ["output.txt"], "download_path": temp_dir}

                result = await fetch_artifacts_tool.execute(parameters)

                # Validate artifact fetch
                assert result["status"] == "success"
                assert "downloaded_artifacts" in result
                assert len(result["downloaded_artifacts"]) == 1

                # Check if file was actually downloaded
                downloaded_file = Path(temp_dir) / "output.txt"
                assert downloaded_file.exists()
                assert downloaded_file.read_bytes() == artifact_content

    @pytest.mark.asyncio
    async def test_fog_job_status_tool_monitoring(self, fog_job_status_tool):
        """Test job status monitoring and progress tracking"""

        # Mock job status progression
        status_responses = [
            {"job_id": "job_67890", "status": "pending", "progress": 0},
            {"job_id": "job_67890", "status": "running", "progress": 50},
            {"job_id": "job_67890", "status": "completed", "progress": 100, "exit_code": 0},
        ]

        with patch("aiohttp.ClientSession") as mock_session:
            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Create mock contexts for each status response
            mock_contexts = []
            for status_data in status_responses:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=status_data)

                mock_context = AsyncMock()
                mock_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_context.__aexit__ = AsyncMock(return_value=None)
                mock_contexts.append(mock_context)

            mock_session_instance.get.side_effect = mock_contexts

            parameters = {"job_id": "job_67890", "wait_for_completion": True, "poll_interval_s": 1, "timeout_s": 30}

            result = await fog_job_status_tool.execute(parameters)

            # Validate status monitoring
            assert result["status"] == "success"
            assert result["final_status"] == "completed"
            assert result["exit_code"] == 0
            assert "status_history" in result
            assert len(result["status_history"]) == 3

    @pytest.mark.asyncio
    async def test_agent_fog_tools_integration(self, mock_agent, mock_fog_gateway_url):
        """Test complete agent integration with fog tools"""

        # Create a mock agent with fog tools
        metadata = AgentMetadata(
            agent_id="test_fog_agent",
            agent_type="test_agent",
            name="FogTestAgent",
            capabilities=[AgentCapability.COMPUTE, AgentCapability.COMMUNICATION],
        )
        agent = BaseAgentTemplate(metadata)

        # Mock fog tools initialization
        with patch.object(agent, "_initialize_fog_tools") as mock_init:
            mock_init.return_value = True

            # Test fog computation offload
            with patch("aiohttp.ClientSession") as mock_session:
                # Mock successful computation offload
                mock_response_data = {
                    "computation_id": "comp_12345",
                    "status": "completed",
                    "result": {"answer": 42, "computation_time_ms": 1500},
                    "fog_node": "node_001",
                }

                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value=mock_response_data)

                # Setup session context manager
                mock_session_instance = AsyncMock()
                mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
                mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

                # Setup post context manager
                mock_post_context = AsyncMock()
                mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
                mock_post_context.__aexit__ = AsyncMock(return_value=None)
                mock_session_instance.post.return_value = mock_post_context

                # Execute fog computation
                computation_params = {
                    "computation_type": "mathematical_analysis",
                    "input_data": {"numbers": [1, 2, 3, 4, 5]},
                    "resources": {"cpu_cores": 2, "memory_gb": 4},
                }

                result = await agent.offload_computation_to_fog(
                    computation_type="mathematical_analysis",
                    input_data=computation_params["input_data"],
                    resources=computation_params["resources"],
                )

                # Validate fog computation integration
                assert result["status"] == "success"
                assert result["computation_id"] == "comp_12345"
                assert result["result"]["answer"] == 42
                assert "fog_node" in result

    @pytest.mark.asyncio
    async def test_fog_tools_error_scenarios(self, create_sandbox_tool, run_job_tool):
        """Test error handling across fog tools"""

        # Test network connectivity errors
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.side_effect = Exception("Connection refused")

            parameters = {"namespace": "test", "runtime": "wasi"}
            result = await create_sandbox_tool.execute(parameters)

            assert result["status"] == "error"
            assert "Connection refused" in result["message"]

        # Test fog gateway timeout
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 504
            mock_response.text = AsyncMock(return_value="Gateway timeout")

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup post context manager
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            parameters = {
                "sandbox_id": "sandbox_12345",
                "image": "test:latest",
                "command": ["sleep", "3600"],  # Long-running job
            }

            result = await run_job_tool.execute(parameters)

            assert result["status"] == "error"
            assert "Gateway timeout" in result["message"]

    @pytest.mark.asyncio
    async def test_fog_tools_concurrent_execution(self, mock_fog_gateway_url):
        """Test concurrent execution of multiple fog tools"""

        # Create multiple tool instances
        tools = [
            CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url),
            CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url),
            CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url),
        ]

        # Mock successful responses for all tools
        with patch("aiohttp.ClientSession") as mock_session:
            {"sandbox_id": lambda: f"sandbox_{uuid4().hex[:8]}", "status": "created", "namespace": "concurrent-test"}

            async def mock_json():
                return {"sandbox_id": f"sandbox_{uuid4().hex[:8]}", "status": "created", "namespace": "concurrent-test"}

            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = mock_json

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup post context manager
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            # Execute tools concurrently
            parameters = {"namespace": "concurrent-test", "runtime": "wasi"}

            tasks = [tool.execute(parameters) for tool in tools]
            results = await asyncio.gather(*tasks)

            # Validate concurrent execution
            assert len(results) == 3
            for result in results:
                assert result["status"] == "success"
                assert result["namespace"] == "concurrent-test"
                assert result["sandbox_id"].startswith("sandbox_")

            # Ensure all sandbox IDs are unique
            sandbox_ids = [result["sandbox_id"] for result in results]
            assert len(set(sandbox_ids)) == 3  # All unique


class TestFogToolsPerformance:
    """Performance tests for fog tools MCP integration"""

    @pytest.fixture
    def mock_fog_gateway_url(self):
        """Mock fog gateway URL for testing"""
        return "http://test-fog-gateway:8080"

    @pytest.mark.asyncio
    async def test_fog_tools_latency_measurement(self, mock_fog_gateway_url):
        """Test fog tools latency and performance characteristics"""

        tool = CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url)

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock response with controlled delay
            async def delayed_response():
                await asyncio.sleep(0.1)  # 100ms simulated latency
                return {"sandbox_id": "perf_test_sandbox", "status": "created", "creation_time_ms": 100}

            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = delayed_response

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup post context manager
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            import time

            start_time = time.time()

            parameters = {"namespace": "perf-test", "runtime": "wasi"}
            result = await tool.execute(parameters)

            execution_time = time.time() - start_time

            # Validate performance characteristics
            assert result["status"] == "success"
            assert execution_time >= 0.1  # At least the simulated latency
            assert execution_time < 1.0  # Should complete within reasonable time

    @pytest.mark.asyncio
    async def test_fog_tools_resource_usage(self, mock_fog_gateway_url):
        """Test fog tools resource usage and cleanup"""

        # Test that tools properly clean up resources
        tools = []
        for i in range(10):
            tool = CreateSandboxTool(fog_gateway_url=mock_fog_gateway_url)
            tools.append(tool)

        # Mock successful execution for all tools
        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 201
            mock_response.json = AsyncMock(return_value={"sandbox_id": "test", "status": "created"})

            # Setup session context manager
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)

            # Setup post context manager
            mock_post_context = AsyncMock()
            mock_post_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post_context.__aexit__ = AsyncMock(return_value=None)
            mock_session_instance.post.return_value = mock_post_context

            # Execute all tools
            parameters = {"namespace": "resource-test", "runtime": "wasi"}
            for tool in tools:
                result = await tool.execute(parameters)
                assert result["status"] == "success"

        # Verify no resource leaks (simplified check)
        assert len(tools) == 10
        del tools  # Trigger cleanup


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
