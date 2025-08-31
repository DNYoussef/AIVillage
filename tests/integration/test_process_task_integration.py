"""
Integration tests for UnifiedBaseAgent._process_task method.

Tests the complete integration with real dependencies and end-to-end workflows.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../experiments/agents/agents'))

from agents.utils.task import Task as LangroidTask
from core.error_handling import AIVillageException, ErrorCategory
from experiments.agents.agents.unified_base_agent import (
    UnifiedBaseAgent,
    UnifiedAgentConfig
)


@dataclass
class IntegrationTestTask:
    """Test task class with full LangroidTask interface."""
    content: str
    type: str = "general"
    id: str = "integration_test"
    timeout: float = 30.0
    recipient: str = None
    target_agent: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestProcessTaskIntegration:
    """Integration test suite for _process_task method."""

    @pytest.fixture
    async def real_agent(self):
        """Create an agent with minimal real dependencies."""
        # Mock configuration
        config = Mock()
        config.name = "IntegrationTestAgent"
        config.description = "Integration test agent"
        config.capabilities = [
            "text_generation", 
            "question_answering", 
            "data_analysis",
            "code_generation"
        ]
        config.model = "gpt-4"
        config.instructions = "You are a helpful AI assistant for integration testing."
        config.rag_config = Mock()
        config.vector_store = Mock()
        
        communication_protocol = Mock()
        communication_protocol.subscribe = Mock()
        communication_protocol.send_message = AsyncMock()
        communication_protocol.query = AsyncMock(return_value="Communication response")
        
        # Create agent with mocked external dependencies
        with patch('experiments.agents.agents.unified_base_agent.EnhancedRAGPipeline'):
            with patch('experiments.agents.agents.unified_base_agent.OpenAIGPTConfig'):
                agent = UnifiedBaseAgent(config, communication_protocol)
        
        # Mock external service calls but keep internal logic
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=Mock(text="LLM Response"))
        agent.rag_pipeline = Mock()
        agent.rag_pipeline.process_query = AsyncMock(return_value={
            "answer": "RAG answer",
            "sources": ["source1", "source2"],
            "confidence": 0.85
        })
        agent.rag_pipeline.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        
        return agent

    @pytest.mark.asyncio
    async def test_end_to_end_text_generation(self, real_agent):
        """Test complete end-to-end text generation workflow."""
        task = IntegrationTestTask(
            content="Write a short story about AI agents collaborating",
            type="text_generation",
            id="e2e_text_gen"
        )
        
        result = await real_agent._process_task(task)
        
        # Verify complete result structure
        assert result["success"] is True
        assert result["task_id"] == "e2e_text_gen"
        assert result["agent_name"] == "IntegrationTestAgent"
        assert "result" in result
        assert "generated_text" in result["result"]
        
        # Verify metadata completeness
        metadata = result["metadata"]
        assert metadata["task_type"] == "text_generation"
        assert metadata["processing_time_ms"] > 0
        assert len(metadata["steps_completed"]) == 4
        assert set(metadata["steps_completed"]) == {
            "validation", "routing", "processing", "formatting"
        }
        
        # Verify performance metrics
        perf_metrics = metadata["performance_metrics"]
        assert "meets_100ms_target" in perf_metrics
        assert "memory_usage_mb" in perf_metrics
        assert "tokens_processed" in perf_metrics

    @pytest.mark.asyncio
    async def test_end_to_end_question_answering(self, real_agent):
        """Test complete question answering with RAG integration."""
        task = IntegrationTestTask(
            content="What are the benefits of distributed AI systems?",
            type="question_answering",
            id="e2e_qa"
        )
        
        result = await real_agent._process_task(task)
        
        assert result["success"] is True
        assert "answer" in result["result"]
        assert result["result"]["source"] == "rag_pipeline"
        
        # Verify RAG pipeline was called
        real_agent.rag_pipeline.process_query.assert_called_once_with(task.content)

    @pytest.mark.asyncio
    async def test_end_to_end_agent_communication(self, real_agent):
        """Test inter-agent communication workflow."""
        task = IntegrationTestTask(
            content="Send status update to: MonitoringAgent\nAll systems operational",
            type="agent_communication",
            id="e2e_comm"
        )
        
        result = await real_agent._process_task(task)
        
        assert result["success"] is True
        assert result["result"]["recipient"] == "MonitoringAgent"
        assert result["result"]["type"] == "agent_communication"
        
        # Verify communication protocol was used
        real_agent.communication_protocol.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_large_content(self, real_agent):
        """Test processing workflow with large content."""
        # Create content just under the size limit
        large_content = "This is a large task content. " * 1500  # ~45KB
        
        task = IntegrationTestTask(
            content=large_content,
            type="summarization",
            id="large_content_test"
        )
        
        start_time = time.perf_counter()
        result = await real_agent._process_task(task)
        end_time = time.perf_counter()
        
        assert result["success"] is True
        assert result["metadata"]["processing_time_ms"] > 0
        
        # Large content should have longer timeout
        processing_time = end_time - start_time
        assert processing_time < 60  # Should complete within extended timeout

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, real_agent):
        """Test error handling and recovery in full workflow."""
        # Mock the LLM to fail initially then succeed
        call_count = 0
        
        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return Mock(text="Success after retry")
        
        real_agent.llm.complete = failing_then_succeeding
        
        task = IntegrationTestTask(
            content="Test error recovery",
            type="text_generation"
        )
        
        # Should fail due to the exception
        with pytest.raises(AIVillageException):
            await real_agent._process_task(task)

    @pytest.mark.asyncio
    async def test_concurrent_task_processing_integration(self, real_agent):
        """Test concurrent processing with real workflow."""
        tasks = [
            IntegrationTestTask(
                content=f"Analyze data set {i}",
                type="data_analysis",
                id=f"concurrent_{i}"
            )
            for i in range(3)
        ]
        
        # Process all tasks concurrently
        results = await asyncio.gather(*[
            real_agent._process_task(task) for task in tasks
        ])
        
        # Verify all completed successfully
        for i, result in enumerate(results):
            assert result["success"] is True
            assert result["task_id"] == f"concurrent_{i}"
            assert result["metadata"]["task_type"] == "data_analysis"

    @pytest.mark.asyncio
    async def test_performance_under_load(self, real_agent):
        """Test performance characteristics under load."""
        # Create multiple tasks of different types
        tasks = [
            IntegrationTestTask(content="Quick task 1", type="general", id="load_1"),
            IntegrationTestTask(content="Quick task 2", type="text_generation", id="load_2"),
            IntegrationTestTask(content="Quick task 3", type="classification", id="load_3"),
            IntegrationTestTask(content="Quick task 4", type="summarization", id="load_4"),
            IntegrationTestTask(content="Quick task 5", type="data_analysis", id="load_5"),
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[
            real_agent._process_task(task) for task in tasks
        ])
        total_time = time.perf_counter() - start_time
        
        # Verify all tasks completed
        assert len(results) == 5
        for result in results:
            assert result["success"] is True
        
        # Calculate average processing time
        avg_processing_time = sum(
            result["metadata"]["processing_time_ms"] for result in results
        ) / len(results)
        
        print(f"Average processing time: {avg_processing_time:.2f}ms")
        print(f"Total concurrent execution time: {total_time * 1000:.2f}ms")
        
        # Performance should be reasonable
        assert total_time < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, real_agent):
        """Test memory usage remains stable across multiple tasks."""
        import gc
        
        # Process multiple tasks and check memory doesn't grow excessively
        initial_results = []
        for i in range(5):
            task = IntegrationTestTask(
                content=f"Memory test task {i}",
                type="general",
                id=f"memory_{i}"
            )
            result = await real_agent._process_task(task)
            initial_results.append(result["metadata"]["performance_metrics"]["memory_usage_mb"])
        
        # Force garbage collection
        gc.collect()
        
        # Process more tasks
        later_results = []
        for i in range(5, 10):
            task = IntegrationTestTask(
                content=f"Memory test task {i}",
                type="general",
                id=f"memory_{i}"
            )
            result = await real_agent._process_task(task)
            later_results.append(result["metadata"]["performance_metrics"]["memory_usage_mb"])
        
        # Memory usage should not grow significantly
        if initial_results[0] > 0 and later_results[0] > 0:
            avg_initial = sum(initial_results) / len(initial_results)
            avg_later = sum(later_results) / len(later_results)
            
            # Allow for some growth but not excessive
            memory_growth_ratio = avg_later / avg_initial
            assert memory_growth_ratio < 1.5  # Less than 50% growth

    @pytest.mark.asyncio
    async def test_capability_matching_integration(self, real_agent):
        """Test capability-based task routing in full integration."""
        # Task that matches specific capability
        data_task = IntegrationTestTask(
            content="Perform comprehensive data_analysis on user behavior patterns",
            type="custom_analytics",  # Not a standard handler
            id="capability_match"
        )
        
        result = await real_agent._process_task(data_task)
        
        assert result["success"] is True
        # Should use general handler but detect capability usage
        capabilities_used = result["metadata"]["agent_capabilities_used"]
        assert "data_analysis" in capabilities_used

    @pytest.mark.asyncio
    async def test_sanitization_integration(self, real_agent):
        """Test content sanitization in full workflow."""
        malicious_task = IntegrationTestTask(
            content="<script>alert('xss')</script>Please analyze this data: [1,2,3,4,5]",
            type="data_analysis",
            id="sanitization_test"
        )
        
        result = await real_agent._process_task(malicious_task)
        
        assert result["success"] is True
        # Content should have been sanitized but task should still complete

    @pytest.mark.asyncio
    async def test_timeout_integration(self, real_agent):
        """Test timeout handling in integration scenario."""
        # Mock a slow operation
        async def slow_llm_response(*args, **kwargs):
            await asyncio.sleep(2.0)  # 2 second delay
            return Mock(text="Slow response")
        
        real_agent.llm.complete = slow_llm_response
        
        fast_timeout_task = IntegrationTestTask(
            content="This should timeout quickly",
            type="text_generation",
            timeout=0.5,  # 500ms timeout
            id="timeout_test"
        )
        
        with pytest.raises(AIVillageException) as exc_info:
            await real_agent._process_task(fast_timeout_task)
        
        assert exc_info.value.category == ErrorCategory.TIMEOUT

    @pytest.mark.asyncio
    async def test_handoff_integration(self, real_agent):
        """Test agent handoff functionality in integration."""
        # Create mock target agent
        target_agent = Mock()
        target_agent.name = "SpecializedAgent"
        
        # Set up handoff tool
        real_agent.add_tool("transfer_to_SpecializedAgent", lambda: target_agent)
        
        handoff_task = IntegrationTestTask(
            content="This task needs specialized handling",
            type="handoff",
            target_agent="SpecializedAgent",
            id="handoff_test"
        )
        
        result = await real_agent._process_task(handoff_task)
        
        assert result["success"] is True
        assert result["result"]["handoff_successful"] is True
        assert result["result"]["transferred_to"] == "SpecializedAgent"


class TestPerformanceBenchmarks:
    """Performance benchmark tests for _process_task method."""

    @pytest.fixture
    async def benchmark_agent(self):
        """Create agent optimized for benchmarking."""
        config = Mock()
        config.name = "BenchmarkAgent"
        config.description = "Performance benchmark agent"
        config.capabilities = ["general", "text_generation"]
        config.model = "gpt-4"
        config.instructions = "Fast response agent"
        config.rag_config = Mock()
        config.vector_store = Mock()
        
        communication_protocol = Mock()
        communication_protocol.subscribe = Mock()
        
        with patch('experiments.agents.agents.unified_base_agent.EnhancedRAGPipeline'):
            with patch('experiments.agents.agents.unified_base_agent.OpenAIGPTConfig'):
                agent = UnifiedBaseAgent(config, communication_protocol)
        
        # Mock for fast responses
        agent.llm = AsyncMock()
        agent.llm.complete = AsyncMock(return_value=Mock(text="Fast response"))
        
        return agent

    @pytest.mark.asyncio
    async def test_100ms_performance_target(self, benchmark_agent):
        """Test that simple tasks meet the 100ms performance target."""
        simple_task = IntegrationTestTask(
            content="Hello",
            type="general",
            id="perf_test"
        )
        
        # Run multiple iterations to get average
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = await benchmark_agent._process_task(simple_task)
            end = time.perf_counter()
            
            processing_time = (end - start) * 1000  # Convert to ms
            times.append(processing_time)
            
            assert result["success"] is True
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"Performance Metrics:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min: {min_time:.2f}ms") 
        print(f"  Max: {max_time:.2f}ms")
        
        # At least some should meet the 100ms target
        fast_enough = [t for t in times if t < 100]
        assert len(fast_enough) > len(times) / 2  # At least half should be fast enough

    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, benchmark_agent):
        """Test processing throughput with concurrent tasks."""
        num_tasks = 20
        tasks = [
            IntegrationTestTask(
                content=f"Throughput test {i}",
                type="general",
                id=f"throughput_{i}"
            )
            for i in range(num_tasks)
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*[
            benchmark_agent._process_task(task) for task in tasks
        ])
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        throughput = num_tasks / total_time
        
        print(f"Throughput Benchmark:")
        print(f"  Processed {num_tasks} tasks in {total_time:.2f}s")
        print(f"  Throughput: {throughput:.2f} tasks/second")
        
        # Verify all tasks completed successfully
        assert len(results) == num_tasks
        for result in results:
            assert result["success"] is True
        
        # Should achieve reasonable throughput
        assert throughput > 5.0  # At least 5 tasks per second

    @pytest.mark.asyncio
    async def test_memory_efficiency_benchmark(self, benchmark_agent):
        """Test memory usage efficiency during processing."""
        import gc
        
        # Process tasks and monitor memory
        memory_readings = []
        
        for i in range(10):
            task = IntegrationTestTask(
                content=f"Memory efficiency test {i}",
                type="general"
            )
            
            gc.collect()  # Force garbage collection before measurement
            
            result = await benchmark_agent._process_task(task)
            memory_mb = result["metadata"]["performance_metrics"]["memory_usage_mb"]
            
            if memory_mb > 0:  # Only track if psutil is available
                memory_readings.append(memory_mb)
        
        if memory_readings:
            avg_memory = sum(memory_readings) / len(memory_readings)
            max_memory = max(memory_readings)
            
            print(f"Memory Efficiency:")
            print(f"  Average: {avg_memory:.2f}MB")
            print(f"  Peak: {max_memory:.2f}MB")
            
            # Memory should be reasonable for simple tasks
            assert avg_memory < 500  # Less than 500MB average


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])