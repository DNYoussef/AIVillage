"""Integration tests for agent abstract method implementations.

Tests the interaction between unified_base_agent, base_analytics, and 
processing_interface components in realistic scenarios.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.king.analytics.base_analytics import BaseAnalytics
from agents.interfaces.processing_interface import ProcessingInterface, ProcessorCapability
from core.communication import StandardCommunicationProtocol
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore


class IntegrationAnalytics(BaseAnalytics):
    """Analytics implementation for integration testing."""

    def generate_analytics_report(self) -> Dict[str, Any]:
        return {
            "total_metrics": len(self.metrics),
            "agent_performance": {
                metric: sum(values) / len(values) if values else 0 for metric, values in self.metrics.items()
            },
            "integration_health": "healthy" if len(self.metrics) > 0 else "no_data",
            "timestamp": datetime.now().isoformat(),
        }


class IntegrationProcessor(ProcessingInterface[Dict[str, Any], Dict[str, Any]]):
    """Processing interface implementation for integration testing."""

    def __init__(self, processor_id: str = "integration_processor"):
        super().__init__(processor_id)
        self.add_capability(ProcessorCapability.TEXT_PROCESSING)
        self.add_capability(ProcessorCapability.BATCH_PROCESSING)
        self.processed_tasks = []
        self.analytics = IntegrationAnalytics()

    async def initialize(self) -> bool:
        await asyncio.sleep(0.01)
        return True

    async def shutdown(self) -> bool:
        await asyncio.sleep(0.01)
        return True

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        from agents.base import ProcessResult, ProcessStatus

        start_time = datetime.now()

        # Simulate processing
        await asyncio.sleep(0.02)

        # Record analytics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.analytics.record_metric("processing_time", processing_time)
        self.analytics.record_metric("tasks_processed", 1.0)

        task_content = input_data.get("content", "")
        self.processed_tasks.append(task_content)

        return ProcessResult(
            status=ProcessStatus.COMPLETED,
            data={
                "original": input_data,
                "processed": f"processed_{task_content}",
                "processing_time": processing_time,
                "processor_id": self.processor_id,
            },
        )

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict) and "content" in input_data

    async def estimate_processing_time(self, input_data: Dict[str, Any]) -> float:
        return 0.02  # 20ms estimate


@pytest.mark.integration
class TestAgentAnalyticsIntegration:
    """Test integration between agents and analytics."""

    @pytest.fixture
    async def agent_with_analytics(self, sample_agent_config, mock_communication_protocol):
        """Create agent with integrated analytics."""
        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.analytics = IntegrationAnalytics()

            # Mock the _process_task to use analytics
            async def mock_process_with_analytics(task):
                start_time = datetime.now()

                # Simulate processing
                await asyncio.sleep(0.01)

                # Record metrics in analytics
                processing_time = (datetime.now() - start_time).total_seconds()
                agent.analytics.record_metric("task_processing_time", processing_time)
                agent.analytics.record_metric("task_success_rate", 1.0)

                return {"processed": True, "analytics_recorded": True}

            agent._process_task = mock_process_with_analytics
            return agent

    async def test_agent_analytics_integration(self, agent_with_analytics, sample_langroid_task):
        """Test agent processing with analytics recording."""
        # Mock layer methods
        agent_with_analytics.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent_with_analytics.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        agent_with_analytics.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent_with_analytics.decision_making_layer.make_decision = AsyncMock(return_value="decision_made")
        agent_with_analytics.continuous_learning_layer.update = AsyncMock()

        # Execute task
        result = await agent_with_analytics.execute_task(sample_langroid_task)

        # Verify task processing
        assert result["result"] == "decision_made"

        # Verify analytics were recorded
        analytics_report = agent_with_analytics.analytics.generate_analytics_report()
        assert analytics_report["total_metrics"] == 2
        assert "task_processing_time" in analytics_report["agent_performance"]
        assert "task_success_rate" in analytics_report["agent_performance"]
        assert analytics_report["integration_health"] == "healthy"

    async def test_multi_task_analytics_aggregation(self, agent_with_analytics, test_data_generator):
        """Test analytics aggregation across multiple tasks."""
        tasks = test_data_generator.generate_task_batch(5)

        # Setup mocks
        agent_with_analytics.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent_with_analytics.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        agent_with_analytics.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent_with_analytics.decision_making_layer.make_decision = AsyncMock(return_value="decision")
        agent_with_analytics.continuous_learning_layer.update = AsyncMock()

        # Process all tasks
        results = []
        for task in tasks:
            result = await agent_with_analytics.execute_task(task)
            results.append(result)

        # Verify all tasks processed
        assert len(results) == 5

        # Verify analytics aggregation
        analytics_report = agent_with_analytics.analytics.generate_analytics_report()
        assert analytics_report["total_metrics"] == 2

        # Should have recorded 5 processing times and success rates
        assert len(agent_with_analytics.analytics.metrics["task_processing_time"]) == 5
        assert len(agent_with_analytics.analytics.metrics["task_success_rate"]) == 5

        # All success rates should be 1.0
        success_rates = agent_with_analytics.analytics.metrics["task_success_rate"]
        assert all(rate == 1.0 for rate in success_rates)


@pytest.mark.integration
class TestAgentProcessorIntegration:
    """Test integration between agents and processing interfaces."""

    @pytest.fixture
    async def agent_with_processor(self, sample_agent_config, mock_communication_protocol):
        """Create agent with integrated processor."""
        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.processor = IntegrationProcessor("agent_processor")
            await agent.processor.initialize()

            # Override _process_task to use processor
            async def mock_process_with_processor(task):
                task_data = {"content": task.content, "type": task.type}

                # Validate input
                is_valid = await agent.processor.validate_input(task_data)
                if not is_valid:
                    raise ValueError("Invalid task data")

                # Process through processor
                result = await agent.processor.process(task_data)

                return result.data if result.is_success else {"error": result.error}

            agent._process_task = mock_process_with_processor
            return agent

    async def test_agent_processor_integration(self, agent_with_processor, sample_langroid_task):
        """Test agent task processing through processor."""
        # Setup mocks
        agent_with_processor.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent_with_processor.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        agent_with_processor.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent_with_processor.decision_making_layer.make_decision = AsyncMock(return_value="processor_decision")
        agent_with_processor.continuous_learning_layer.update = AsyncMock()

        # Execute task
        result = await agent_with_processor.execute_task(sample_langroid_task)

        # Verify processing chain
        assert result["result"] == "processor_decision"

        # Verify processor recorded the task
        assert len(agent_with_processor.processor.processed_tasks) == 1
        assert sample_langroid_task.content in agent_with_processor.processor.processed_tasks

        # Verify analytics were recorded in processor
        analytics_report = agent_with_processor.processor.analytics.generate_analytics_report()
        assert analytics_report["total_metrics"] == 2
        assert analytics_report["integration_health"] == "healthy"

    async def test_batch_processing_through_agent(self, agent_with_processor, test_data_generator):
        """Test batch processing through agent-processor integration."""
        tasks = test_data_generator.generate_task_batch(3)

        # Setup mocks
        agent_with_processor.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent_with_processor.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        agent_with_processor.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent_with_processor.decision_making_layer.make_decision = AsyncMock(return_value="batch_decision")
        agent_with_processor.continuous_learning_layer.update = AsyncMock()

        # Process tasks concurrently
        results = await asyncio.gather(*[agent_with_processor.execute_task(task) for task in tasks])

        # Verify all results
        assert len(results) == 3
        assert all(result["result"] == "batch_decision" for result in results)

        # Verify processor handled all tasks
        assert len(agent_with_processor.processor.processed_tasks) == 3

        # Verify processor analytics
        metrics = agent_with_processor.processor.analytics.metrics
        assert len(metrics["processing_time"]) == 3
        assert len(metrics["tasks_processed"]) == 3


@pytest.mark.integration
class TestMultiAgentSystemIntegration:
    """Test integration of multiple agents with analytics and processing."""

    @pytest.fixture
    async def multi_agent_system(self, mock_communication_protocol):
        """Create multi-agent system for testing."""
        agents = []

        for i in range(3):
            # Create agent config
            vector_store = Mock(spec=VectorStore)
            vector_store.add_texts = AsyncMock()
            vector_store.similarity_search = AsyncMock(return_value=[])

            config = UnifiedAgentConfig(
                name=f"Agent{i}",
                description=f"Test agent {i}",
                capabilities=[f"capability_{i}", "shared_capability"],
                rag_config=UnifiedConfig(),
                vector_store=vector_store,
                model="gpt-4",
                instructions=f"Agent {i} instructions",
            )

            with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
                agent = UnifiedBaseAgent(config, mock_communication_protocol)
                agent.analytics = IntegrationAnalytics()
                agent.processor = IntegrationProcessor(f"processor_{i}")
                await agent.processor.initialize()

                # Mock _process_task for integration
                async def make_process_task(agent_id):
                    async def mock_process_task(task):
                        # Record in analytics
                        agent.analytics.record_metric("tasks_handled", 1.0)
                        agent.analytics.record_metric(f"agent_{agent_id}_performance", 0.9)

                        # Process through processor
                        task_data = {"content": task.content, "agent_id": agent_id}
                        result = await agent.processor.process(task_data)

                        return result.data if result.is_success else {"error": result.error}

                    return mock_process_task

                agent._process_task = await make_process_task(i)
                agents.append(agent)

        return SelfEvolvingSystem(agents)

    async def test_multi_agent_task_distribution(self, multi_agent_system, test_data_generator):
        """Test task distribution across multiple agents."""
        tasks = test_data_generator.generate_task_batch(6)  # 2 tasks per agent

        # Assign task types to match agent capabilities
        for i, task in enumerate(tasks):
            task.type = f"capability_{i % 3}"

        # Mock all agent layers
        for agent in multi_agent_system.agents:
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(return_value=f"decision_from_{agent.name}")
            agent.continuous_learning_layer.update = AsyncMock()

        # Process tasks through system
        results = []
        for task in tasks:
            result = await multi_agent_system.process_task(task)
            results.append(result)

        # Verify all tasks processed
        assert len(results) == 6

        # Verify each agent handled tasks
        for i, agent in enumerate(multi_agent_system.agents):
            agent_metrics = agent.analytics.metrics
            assert "tasks_handled" in agent_metrics
            assert f"agent_{i}_performance" in agent_metrics

            # Each agent should have handled 2 tasks
            assert len(agent_metrics["tasks_handled"]) == 2

    async def test_system_wide_analytics_aggregation(self, multi_agent_system):
        """Test system-wide analytics aggregation."""
        # Generate different workloads for each agent
        workloads = [
            [Mock(content=f"task_{i}_for_agent_0", type="capability_0") for i in range(3)],
            [Mock(content=f"task_{i}_for_agent_1", type="capability_1") for i in range(2)],
            [Mock(content=f"task_{i}_for_agent_2", type="capability_2") for i in range(4)],
        ]

        # Setup mocks for all agents
        for agent in multi_agent_system.agents:
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(return_value="decision")
            agent.continuous_learning_layer.update = AsyncMock()

        # Process workloads
        for i, workload in enumerate(workloads):
            for task in workload:
                await multi_agent_system.process_task(task)

        # Aggregate analytics across all agents
        system_analytics = {
            "total_agents": len(multi_agent_system.agents),
            "agent_reports": [],
            "system_metrics": {"total_tasks_processed": 0, "average_performance": 0.0},
        }

        for agent in multi_agent_system.agents:
            report = agent.analytics.generate_analytics_report()
            system_analytics["agent_reports"].append({"agent_name": agent.name, "report": report})

            # Aggregate metrics
            tasks_handled = len(agent.analytics.metrics.get("tasks_handled", []))
            system_analytics["system_metrics"]["total_tasks_processed"] += tasks_handled

        # Calculate average performance
        total_agents = len(multi_agent_system.agents)
        if total_agents > 0:
            system_analytics["system_metrics"]["average_performance"] = (
                sum(0.9 for _ in range(total_agents)) / total_agents
            )

        # Verify system-wide metrics
        assert system_analytics["total_agents"] == 3
        assert len(system_analytics["agent_reports"]) == 3
        assert system_analytics["system_metrics"]["total_tasks_processed"] == 9  # 3+2+4
        assert system_analytics["system_metrics"]["average_performance"] == 0.9


@pytest.mark.integration
class TestCommunicationIntegration:
    """Test integration of communication protocols with agents and processing."""

    @pytest.fixture
    async def communicating_agents(self):
        """Create agents that can communicate with each other."""
        communication_protocol = Mock(spec=StandardCommunicationProtocol)
        communication_protocol.subscribe = Mock()
        communication_protocol.send_message = AsyncMock()
        communication_protocol.query = AsyncMock()

        agents = []

        for i in range(2):
            vector_store = Mock(spec=VectorStore)
            vector_store.add_texts = AsyncMock()

            config = UnifiedAgentConfig(
                name=f"CommunicatingAgent{i}",
                description=f"Agent {i} for communication testing",
                capabilities=["communication", f"specialty_{i}"],
                rag_config=UnifiedConfig(),
                vector_store=vector_store,
                model="gpt-4",
                instructions=f"Communication agent {i}",
            )

            with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
                agent = UnifiedBaseAgent(config, communication_protocol)
                agent.analytics = IntegrationAnalytics()

                # Mock _process_task to simulate communication needs
                async def make_communicating_process_task(agent_id):
                    async def mock_process_task(task):
                        # Record communication metrics
                        agent.analytics.record_metric("communication_attempts", 1.0)

                        if task.content.startswith("communicate"):
                            # Simulate need to communicate with other agent
                            other_agent = f"CommunicatingAgent{1-agent_id}"
                            response = await agent.communicate("collaboration request", other_agent)

                            agent.analytics.record_metric("successful_communications", 1.0)

                            return {"result": "collaborative_result", "communication_response": response}
                        else:
                            return {"result": f"processed_by_agent_{agent_id}"}

                    return mock_process_task

                agent._process_task = await make_communicating_process_task(i)
                agents.append(agent)

        return agents, communication_protocol

    async def test_inter_agent_communication_integration(self, communicating_agents):
        """Test communication between agents during task processing."""
        agents, comm_protocol = communicating_agents

        # Setup communication protocol mock
        comm_protocol.query.return_value = {"collaboration": "successful"}

        # Create communication task
        comm_task = Mock()
        comm_task.content = "communicate_with_peer"
        comm_task.type = "communication"

        # Setup layer mocks for first agent
        agent = agents[0]
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=comm_task)
        agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent.decision_making_layer.make_decision = AsyncMock(return_value="communication_decision")
        agent.continuous_learning_layer.update = AsyncMock()

        # Execute communication task
        result = await agent.execute_task(comm_task)

        # Verify communication occurred
        assert result["result"] == "communication_decision"
        comm_protocol.query.assert_called()

        # Verify analytics recorded communication
        analytics_report = agent.analytics.generate_analytics_report()
        agent_performance = analytics_report["agent_performance"]
        assert "communication_attempts" in agent_performance
        assert "successful_communications" in agent_performance
        assert agent_performance["communication_attempts"] == 1.0
        assert agent_performance["successful_communications"] == 1.0

    async def test_communication_failure_handling(self, communicating_agents):
        """Test handling of communication failures."""
        agents, comm_protocol = communicating_agents

        # Setup communication failure
        comm_protocol.query.side_effect = Exception("Communication failed")

        # Create communication task
        comm_task = Mock()
        comm_task.content = "communicate_with_peer"
        comm_task.type = "communication"

        # Setup layer mocks
        agent = agents[0]
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=comm_task)
        agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent.decision_making_layer.make_decision = AsyncMock(return_value="fallback_decision")
        agent.continuous_learning_layer.update = AsyncMock()

        # Execute task - should handle communication failure gracefully
        with pytest.raises(Exception):  # Communication failure should propagate
            await agent.execute_task(comm_task)

        # Verify attempt was recorded
        analytics_report = agent.analytics.generate_analytics_report()
        agent_performance = analytics_report["agent_performance"]
        assert "communication_attempts" in agent_performance


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    @pytest.fixture
    async def error_prone_agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent with error-prone components for testing."""
        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.analytics = IntegrationAnalytics()
            agent.processor = IntegrationProcessor("error_processor")
            await agent.processor.initialize()

            # Override processor to simulate errors
            original_process = agent.processor.process

            async def error_prone_process(input_data, **kwargs):
                if "error" in input_data.get("content", ""):
                    raise ValueError("Processing error occurred")
                return await original_process(input_data, **kwargs)

            agent.processor.process = error_prone_process

            # Override _process_task to use error-prone processor
            async def mock_process_with_errors(task):
                agent.analytics.record_metric("processing_attempts", 1.0)

                try:
                    task_data = {"content": task.content}
                    result = await agent.processor.process(task_data)

                    agent.analytics.record_metric("successful_processing", 1.0)
                    return result.data if result.is_success else {"error": result.error}

                except Exception as e:
                    agent.analytics.record_metric("processing_errors", 1.0)
                    return {"error": str(e)}

            agent._process_task = mock_process_with_errors
            return agent

    async def test_error_handling_with_analytics(self, error_prone_agent):
        """Test error handling with analytics recording."""
        # Create tasks - some will cause errors
        tasks = [
            Mock(content="normal_task", type="normal"),
            Mock(content="error_task", type="error"),  # Will cause error
            Mock(content="another_normal", type="normal"),
        ]

        # Setup layer mocks
        error_prone_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        error_prone_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        error_prone_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        error_prone_agent.decision_making_layer.make_decision = AsyncMock(return_value="error_handled_decision")
        error_prone_agent.continuous_learning_layer.update = AsyncMock()

        # Process tasks
        results = []
        for task in tasks:
            result = await error_prone_agent.execute_task(task)
            results.append(result)

        # Verify all tasks were processed (even with errors)
        assert len(results) == 3

        # Verify analytics recorded errors and successes
        analytics_report = error_prone_agent.analytics.generate_analytics_report()
        agent_performance = analytics_report["agent_performance"]

        assert "processing_attempts" in agent_performance
        assert agent_performance["processing_attempts"] == 3.0

        assert "successful_processing" in agent_performance
        assert agent_performance["successful_processing"] == 2.0  # 2 successful, 1 error

        assert "processing_errors" in agent_performance
        assert agent_performance["processing_errors"] == 1.0  # 1 error

    async def test_cascading_error_recovery(self, error_prone_agent):
        """Test recovery from cascading errors across components."""
        # Create a task that will cause multiple component failures
        error_task = Mock(content="cascade_error_task", type="cascade_error")

        # Setup layers to fail in sequence
        error_prone_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        error_prone_agent.foundational_layer.process_task = AsyncMock(return_value=error_task)

        # Agent architecture layer will fail
        error_prone_agent.agent_architecture_layer.process_result = AsyncMock(
            side_effect=Exception("Architecture layer failed")
        )

        # Decision layer should still be called for recovery
        error_prone_agent.decision_making_layer.make_decision = AsyncMock(return_value="recovery_decision")
        error_prone_agent.continuous_learning_layer.update = AsyncMock()

        # Process task - should handle architecture layer failure
        with pytest.raises(Exception):  # Architecture layer failure should propagate
            await error_prone_agent.execute_task(error_task)

        # Verify error was recorded
        analytics = error_prone_agent.analytics.metrics
        assert "processing_attempts" in analytics
        assert "processing_errors" in analytics


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    @pytest.fixture
    async def performance_agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent optimized for performance testing."""
        with patch.multiple("agents.unified_base_agent", EnhancedRAGPipeline=Mock(), OpenAIGPTConfig=Mock()):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.analytics = IntegrationAnalytics()
            agent.processor = IntegrationProcessor("performance_processor")
            await agent.processor.initialize()

            # Optimized _process_task for performance testing
            async def performance_process_task(task):
                start_time = datetime.now()

                # Minimal processing
                task_data = {"content": task.content}
                result = await agent.processor.process(task_data)

                # Record performance metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                agent.analytics.record_metric("execution_time", processing_time)

                return result.data if result.is_success else {"error": result.error}

            agent._process_task = performance_process_task
            return agent

    async def test_high_throughput_processing(self, performance_agent, performance_thresholds):
        """Test high-throughput task processing."""
        num_tasks = 100
        tasks = [Mock(content=f"task_{i}", type="performance") for i in range(num_tasks)]

        # Setup minimal mocks for speed
        performance_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        performance_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        performance_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        performance_agent.decision_making_layer.make_decision = AsyncMock(return_value="fast_decision")
        performance_agent.continuous_learning_layer.update = AsyncMock()

        # Process tasks concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*[performance_agent.execute_task(task) for task in tasks])
        end_time = datetime.now()

        # Calculate throughput
        total_time = (end_time - start_time).total_seconds()
        throughput = num_tasks / total_time

        # Verify results
        assert len(results) == num_tasks
        assert all(result["result"] == "fast_decision" for result in results)

        # Verify performance metrics
        analytics = performance_agent.analytics.metrics
        assert len(analytics["execution_time"]) == num_tasks

        # Check throughput threshold
        min_throughput = performance_thresholds.get("batch_processing_throughput", 10)
        assert throughput >= min_throughput, f"Throughput {throughput} below threshold {min_throughput}"

    async def test_memory_efficiency_integration(self, performance_agent):
        """Test memory efficiency across integrated components."""
        import gc

        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Process many tasks
        num_tasks = 50
        tasks = [Mock(content=f"memory_test_{i}", type="memory") for i in range(num_tasks)]

        # Setup mocks
        performance_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        performance_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        performance_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        performance_agent.decision_making_layer.make_decision = AsyncMock(return_value="memory_decision")
        performance_agent.continuous_learning_layer.update = AsyncMock()

        # Process tasks in batches to simulate real usage
        batch_size = 10
        for i in range(0, num_tasks, batch_size):
            batch = tasks[i : i + batch_size]
            await asyncio.gather(*[performance_agent.execute_task(task) for task in batch])

            # Force cleanup between batches
            gc.collect()

        # Final cleanup and measurement
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable
        object_growth = final_objects - initial_objects
        growth_per_task = object_growth / num_tasks

        # Should not create excessive objects per task
        assert growth_per_task < 100, f"Memory growth {growth_per_task} objects per task is too high"
