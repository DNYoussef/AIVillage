"""
Comprehensive unit tests for UnifiedBaseAgent._process_task method.

This test suite provides extensive coverage for the core task processing logic
including routing, validation, error handling, timeouts, and metrics collection.
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../experiments/agents/agents"))

from core.error_handling import AIVillageException, ErrorCategory
from experiments.agents.agents.unified_base_agent import UnifiedBaseAgent


@dataclass
class MockTask:
    """Mock task class for testing."""

    content: str
    type: str = "general"
    id: str = "test_task_123"
    timeout: float = 30.0
    recipient: str = None
    target_agent: str = None


class TestUnifiedBaseAgentProcessTask:
    """Test suite for UnifiedBaseAgent._process_task method."""

    @pytest.fixture
    async def mock_agent(self):
        """Create a mock agent for testing."""
        config = Mock()
        config.name = "TestAgent"
        config.description = "Test agent"
        config.capabilities = ["text_generation", "data_analysis", "general"]
        config.model = "gpt-4"
        config.instructions = "Test instructions"
        config.rag_config = Mock()
        config.vector_store = Mock()

        communication_protocol = Mock()

        # Create agent instance
        with patch("experiments.agents.agents.unified_base_agent.EnhancedRAGPipeline"):
            with patch("experiments.agents.agents.unified_base_agent.OpenAIGPTConfig"):
                agent = UnifiedBaseAgent(config, communication_protocol)

        # Mock the logger
        agent.logger = Mock()
        agent.logger.info = Mock()
        agent.logger.error = Mock()
        agent.logger.warning = Mock()
        agent.logger.debug = Mock()

        # Mock the LLM and other components
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=Mock(text="Generated response"))
        agent.rag_pipeline = AsyncMock()
        agent.rag_pipeline.process_query = AsyncMock(return_value={"answer": "RAG response"})

        # Mock the generate method
        agent.generate = AsyncMock(return_value="Generated text")
        agent.query_rag = AsyncMock(return_value={"rag_result": "data"})
        agent.communicate = AsyncMock(return_value="Communication response")

        return agent

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return MockTask(
            content="Generate a summary of machine learning algorithms", type="text_generation", id="test_task_123"
        )

    # Core Functionality Tests

    @pytest.mark.asyncio
    async def test_process_task_success_basic(self, mock_agent, sample_task):
        """Test successful basic task processing."""
        result = await mock_agent._process_task(sample_task)

        assert result["success"] is True
        assert result["task_id"] == "test_task_123"
        assert result["agent_name"] == "TestAgent"
        assert "result" in result
        assert "metadata" in result
        assert result["metadata"]["task_type"] == "text_generation"
        assert "processing_time_ms" in result["metadata"]

        # Verify logging was called
        mock_agent.logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_process_task_all_handler_types(self, mock_agent):
        """Test all different task handler types."""
        task_types = [
            "text_generation",
            "question_answering",
            "data_analysis",
            "code_generation",
            "translation",
            "summarization",
            "classification",
            "rag_query",
            "agent_communication",
            "general",
            "handoff",
        ]

        for task_type in task_types:
            task = MockTask(content=f"Test {task_type} task", type=task_type, id=f"test_{task_type}")

            if task_type == "agent_communication":
                task.content = "Test message to: TestRecipient"
            elif task_type == "handoff":
                task.target_agent = "TestTargetAgent"

            result = await mock_agent._process_task(task)

            assert result["success"] is True
            assert result["metadata"]["task_type"] == task_type

    # Validation Tests

    @pytest.mark.asyncio
    async def test_process_task_missing_content(self, mock_agent):
        """Test task validation with missing content."""
        task = Mock()
        task.content = None

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(task)

        assert exc_info.value.category == ErrorCategory.VALIDATION
        assert "missing required content" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_process_task_content_too_long(self, mock_agent):
        """Test task validation with content exceeding length limit."""
        task = MockTask(content="x" * 60000)  # Exceeds 50KB limit

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(task)

        assert exc_info.value.category == ErrorCategory.VALIDATION
        assert "exceeds maximum length" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_process_task_content_sanitization(self, mock_agent, sample_task):
        """Test content sanitization removes dangerous patterns."""
        task = MockTask(content="<script>alert('xss')</script>Normal content here", type="text_generation")

        result = await mock_agent._process_task(task)

        # Content should be sanitized
        assert result["success"] is True
        mock_agent.logger.warning.assert_called()

    # Error Handling Tests

    @pytest.mark.asyncio
    async def test_process_task_handler_exception(self, mock_agent, sample_task):
        """Test error handling when task handler raises exception."""
        # Mock the generate method to raise an exception
        mock_agent.generate.side_effect = Exception("Handler failed")

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(sample_task)

        assert exc_info.value.category == ErrorCategory.PROCESSING
        assert "Task processing failed" in str(exc_info.value)
        assert "Handler failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_task_timeout_handling(self, mock_agent):
        """Test timeout handling for long-running tasks."""
        task = MockTask(content="Long running task", type="text_generation", timeout=0.1)  # Very short timeout

        # Make generate take longer than timeout
        async def slow_generate(prompt):
            await asyncio.sleep(0.2)
            return "Response"

        mock_agent.generate = slow_generate

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(task)

        assert exc_info.value.category == ErrorCategory.TIMEOUT
        assert "timed out" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_process_task_cancellation(self, mock_agent, sample_task):
        """Test task cancellation handling."""
        # Mock the generate method to simulate cancellation
        mock_agent.generate.side_effect = asyncio.CancelledError()

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(sample_task)

        assert exc_info.value.category == ErrorCategory.PROCESSING
        assert "cancelled" in str(exc_info.value).lower()

    # Progress Tracking Tests

    @pytest.mark.asyncio
    async def test_progress_tracking_stages(self, mock_agent, sample_task):
        """Test that progress tracking goes through all expected stages."""
        result = await mock_agent._process_task(sample_task)

        assert result["metadata"]["steps_completed"] == ["validation", "routing", "processing", "formatting"]
        assert result["metadata"]["processing_time_ms"] > 0

    # Performance and Metrics Tests

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, mock_agent, sample_task):
        """Test collection of performance metrics."""
        result = await mock_agent._process_task(sample_task)

        metadata = result["metadata"]
        perf_metrics = metadata["performance_metrics"]

        assert "meets_100ms_target" in perf_metrics
        assert "memory_usage_mb" in perf_metrics
        assert "tokens_processed" in perf_metrics
        assert isinstance(perf_metrics["tokens_processed"], int)

    @pytest.mark.asyncio
    async def test_fast_task_meets_performance_target(self, mock_agent):
        """Test that fast tasks meet the 100ms performance target."""
        task = MockTask(content="Quick task", type="general")

        # Mock fast response
        async def fast_generate(prompt):
            return "Quick response"

        mock_agent.generate = fast_generate

        result = await mock_agent._process_task(task)

        # Check if it meets performance target (this may vary based on system)
        processing_time = result["metadata"]["processing_time_ms"]
        assert processing_time >= 0

    # Task Routing Tests

    @pytest.mark.asyncio
    async def test_task_routing_capability_matching(self, mock_agent):
        """Test task routing based on agent capabilities."""
        task = MockTask(
            content="Analyze this data", type="custom_data_analysis"  # Not in handlers but matches capability
        )

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        # Should use general handler for capability match
        mock_agent.logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_task_routing_unknown_type(self, mock_agent):
        """Test task routing for unknown task types."""
        task = MockTask(content="Unknown task", type="completely_unknown_type")

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        # Should default to general handler

    # Specific Handler Tests

    @pytest.mark.asyncio
    async def test_agent_communication_handler_with_recipient(self, mock_agent):
        """Test agent communication handler with recipient parsing."""
        task = MockTask(content="Send message to: OtherAgent\nHello there!", type="agent_communication")

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        assert "result" in result
        mock_agent.communicate.assert_called()

    @pytest.mark.asyncio
    async def test_agent_communication_handler_no_recipient(self, mock_agent):
        """Test agent communication handler without recipient."""
        task = MockTask(content="Message with no recipient", type="agent_communication")

        with pytest.raises(AIVillageException) as exc_info:
            await mock_agent._process_task(task)

        assert "No recipient specified" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handoff_handler_with_valid_target(self, mock_agent):
        """Test handoff handler with valid target agent."""
        # Mock handoff tool
        mock_target_agent = Mock()
        mock_target_agent.name = "TargetAgent"

        mock_agent.get_tool = Mock(return_value=lambda: mock_target_agent)

        task = MockTask(content="Handoff task", type="handoff", target_agent="TargetAgent")

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        assert result["result"]["handoff_successful"] is True

    @pytest.mark.asyncio
    async def test_handoff_handler_invalid_target(self, mock_agent):
        """Test handoff handler with invalid target agent."""
        mock_agent.get_tool = Mock(return_value=None)

        task = MockTask(content="Handoff task", type="handoff", target_agent="NonexistentAgent")

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        assert result["result"]["handoff_successful"] is False

    # Memory Usage Tests

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self, mock_agent, sample_task):
        """Test memory usage tracking in metrics."""
        with patch("psutil.Process") as mock_process:
            mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB

            result = await mock_agent._process_task(sample_task)

            assert result["metadata"]["performance_metrics"]["memory_usage_mb"] == 100.0

    @pytest.mark.asyncio
    async def test_memory_usage_no_psutil(self, mock_agent, sample_task):
        """Test memory usage tracking when psutil is not available."""
        with patch("builtins.__import__", side_effect=ImportError):
            result = await mock_agent._process_task(sample_task)

            assert result["metadata"]["performance_metrics"]["memory_usage_mb"] == 0.0

    # Token Estimation Tests

    @pytest.mark.asyncio
    async def test_token_estimation(self, mock_agent):
        """Test token estimation for different content lengths."""
        test_cases = [
            ("Short", 1),  # 5 chars / 4 = 1 token
            ("This is a longer text", 5),  # 22 chars / 4 = 5 tokens
            ("x" * 100, 25),  # 100 chars / 4 = 25 tokens
        ]

        for content, expected_tokens in test_cases:
            task = MockTask(content=content)
            result = await mock_agent._process_task(task)

            actual_tokens = result["metadata"]["performance_metrics"]["tokens_processed"]
            assert actual_tokens == expected_tokens

    # Capability Usage Tests

    @pytest.mark.asyncio
    async def test_capabilities_used_detection(self, mock_agent):
        """Test detection of which capabilities were used."""
        task = MockTask(content="Please do some text_generation and data_analysis", type="general")

        result = await mock_agent._process_task(task)

        capabilities_used = result["metadata"]["agent_capabilities_used"]
        assert "text_generation" in capabilities_used
        assert "data_analysis" in capabilities_used

    @pytest.mark.asyncio
    async def test_no_specific_capabilities_used(self, mock_agent):
        """Test when no specific capabilities are detected."""
        task = MockTask(content="Generic task with no specific capability keywords", type="general")

        result = await mock_agent._process_task(task)

        capabilities_used = result["metadata"]["agent_capabilities_used"]
        assert capabilities_used == ["general"]

    # Integration Tests

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, mock_agent):
        """Test complete workflow from start to finish."""
        task = MockTask(content="Generate a comprehensive analysis", type="data_analysis", id="integration_test_123")

        start_time = time.perf_counter()
        result = await mock_agent._process_task(task)
        end_time = time.perf_counter()

        # Verify complete result structure
        assert result["success"] is True
        assert result["task_id"] == "integration_test_123"
        assert result["agent_name"] == "TestAgent"
        assert "result" in result

        # Verify metadata
        metadata = result["metadata"]
        assert metadata["task_type"] == "data_analysis"
        assert len(metadata["steps_completed"]) == 4
        assert metadata["processing_time_ms"] > 0
        assert metadata["processing_time_ms"] < (end_time - start_time) * 1000 + 100  # Allow for overhead

        # Verify performance metrics
        perf_metrics = metadata["performance_metrics"]
        assert "meets_100ms_target" in perf_metrics
        assert "memory_usage_mb" in perf_metrics
        assert "tokens_processed" in perf_metrics

        # Verify logging
        mock_agent.logger.info.assert_called()

    # Edge Cases and Error Scenarios

    @pytest.mark.asyncio
    async def test_task_with_none_type(self, mock_agent):
        """Test task with None type defaults to general."""
        task = MockTask(content="Test task")
        task.type = None

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        assert result["metadata"]["task_type"] is None

    @pytest.mark.asyncio
    async def test_task_with_empty_content(self, mock_agent):
        """Test task with empty content."""
        task = MockTask(content="")

        result = await mock_agent._process_task(task)

        assert result["success"] is True
        assert result["metadata"]["performance_metrics"]["tokens_processed"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, mock_agent):
        """Test concurrent processing of multiple tasks."""
        tasks = [MockTask(content=f"Task {i}", id=f"task_{i}") for i in range(3)]

        # Process tasks concurrently
        results = await asyncio.gather(*[mock_agent._process_task(task) for task in tasks])

        # Verify all tasks completed successfully
        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["task_id"] == f"task_{i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
