"""Comprehensive unit tests for UnifiedBaseAgent implementation.

Uses TDD London School methodology with extensive mocking and behavior verification.
Focuses on testing all abstract method implementations with various scenarios.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Any, Dict

from agents.unified_base_agent import (
    UnifiedBaseAgent,
    UnifiedAgentConfig,
    QualityAssuranceLayer,
    FoundationalLayer,
    ContinuousLearningLayer,
    AgentArchitectureLayer,
    DecisionMakingLayer,
    SelfEvolvingSystem,
    create_agent
)
from agents.utils.task import Task as LangroidTask
from core.communication import Message, MessageType, Priority
from core.error_handling import AIVillageException, ErrorCategory, ErrorSeverity


@pytest.mark.unit
class TestUnifiedBaseAgentInitialization:
    """Test agent initialization with various configurations."""
    
    async def test_successful_initialization(
        self, sample_agent_config, mock_communication_protocol
    ):
        """Test successful agent initialization."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(
                sample_agent_config, 
                mock_communication_protocol
            )
            
            assert agent.name == "TestAgent"
            assert agent.description == "Agent for testing"
            assert "test_capability" in agent.capabilities
            assert agent.model == "gpt-4"
            assert agent.instructions == "Test instructions"
            
            # Verify layer initialization
            assert hasattr(agent, 'quality_assurance_layer')
            assert hasattr(agent, 'foundational_layer')
            assert hasattr(agent, 'continuous_learning_layer')
            assert hasattr(agent, 'agent_architecture_layer')
            assert hasattr(agent, 'decision_making_layer')
    
    async def test_initialization_with_knowledge_tracker(
        self, sample_agent_config, mock_communication_protocol
    ):
        """Test initialization with knowledge tracker."""
        mock_tracker = Mock()
        
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(
                sample_agent_config,
                mock_communication_protocol,
                knowledge_tracker=mock_tracker
            )
            
            assert agent is not None
    
    async def test_initialization_failure_handling(
        self, sample_agent_config, mock_communication_protocol
    ):
        """Test initialization failure handling."""
        with patch('agents.unified_base_agent.EnhancedRAGPipeline', 
                   side_effect=Exception("RAG pipeline failed")):
            
            with pytest.raises(AIVillageException) as exc_info:
                UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            
            assert exc_info.value.category == ErrorCategory.INITIALIZATION
            assert exc_info.value.severity == ErrorSeverity.CRITICAL


@pytest.mark.unit
class TestUnifiedBaseAgentTaskExecution:
    """Test task execution with comprehensive scenarios."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            
            # Mock the _process_task method
            agent._process_task = AsyncMock(return_value={"processed": True})
            
            return agent
    
    async def test_successful_task_execution(self, agent, sample_langroid_task):
        """Test successful task execution flow."""
        # Mock all layer methods
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        agent.agent_architecture_layer.process_result = AsyncMock(return_value="processed_result")
        agent.decision_making_layer.make_decision = AsyncMock(return_value="decision_made")
        agent.continuous_learning_layer.update = AsyncMock()
        
        result = await agent.execute_task(sample_langroid_task)
        
        assert result["result"] == "decision_made"
        
        # Verify all layers were called
        agent.quality_assurance_layer.check_task_safety.assert_called_once()
        agent.foundational_layer.process_task.assert_called_once()
        agent.agent_architecture_layer.process_result.assert_called_once()
        agent.decision_making_layer.make_decision.assert_called_once()
        agent.continuous_learning_layer.update.assert_called_once()
    
    async def test_task_safety_rejection(self, agent, sample_langroid_task):
        """Test task rejection by quality assurance layer."""
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=False)
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.execute_task(sample_langroid_task)
        
        assert exc_info.value.category == ErrorCategory.VALIDATION
        assert exc_info.value.severity == ErrorSeverity.WARNING
    
    async def test_foundational_layer_failure(self, agent, sample_langroid_task):
        """Test foundational layer processing failure."""
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(
            side_effect=Exception("Foundation failed")
        )
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.execute_task(sample_langroid_task)
        
        assert exc_info.value.category == ErrorCategory.PROCESSING
    
    async def test_not_implemented_process_task(self, agent, sample_langroid_task):
        """Test handling of not implemented _process_task."""
        agent._process_task = AsyncMock(side_effect=NotImplementedError())
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.execute_task(sample_langroid_task)
        
        assert exc_info.value.category == ErrorCategory.NOT_IMPLEMENTED
    
    async def test_decision_making_failure(self, agent, sample_langroid_task):
        """Test decision making layer failure handling."""
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        agent.agent_architecture_layer.process_result = AsyncMock(return_value="result")
        agent.decision_making_layer.make_decision = AsyncMock(
            side_effect=Exception("Decision failed")
        )
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.execute_task(sample_langroid_task)
        
        assert exc_info.value.category == ErrorCategory.DECISION
    
    async def test_continuous_learning_failure_graceful(self, agent, sample_langroid_task):
        """Test graceful handling of continuous learning failure."""
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(return_value=sample_langroid_task)
        agent.agent_architecture_layer.process_result = AsyncMock(return_value="result")
        agent.decision_making_layer.make_decision = AsyncMock(return_value="decision")
        agent.continuous_learning_layer.update = AsyncMock(
            side_effect=Exception("Learning failed")
        )
        
        # Should not raise exception, just log warning
        result = await agent.execute_task(sample_langroid_task)
        assert result["result"] == "decision"


@pytest.mark.unit
class TestUnifiedBaseAgentCommunication:
    """Test agent communication capabilities."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            return UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
    
    async def test_message_handling_task_type(self, agent):
        """Test handling of task-type messages."""
        agent.process_message = AsyncMock(return_value={"result": "success"})
        
        message = Message(
            type=MessageType.TASK,
            sender="TestSender",
            receiver=agent.name,
            content={"content": "test task"}
        )
        
        await agent.handle_message(message)
        
        agent.process_message.assert_called_once_with({"content": "test task"})
        agent.communication_protocol.send_message.assert_called_once()
    
    async def test_message_handling_non_task_type(self, agent):
        """Test handling of non-task messages."""
        message = Message(
            type=MessageType.QUERY,
            sender="TestSender",
            receiver=agent.name,
            content={"query": "test"}
        )
        
        await agent.handle_message(message)
        
        # Should not call process_message or send response
        agent.communication_protocol.send_message.assert_not_called()
    
    async def test_process_message_success(self, agent):
        """Test successful message processing."""
        agent.execute_task = AsyncMock(return_value={"result": "success"})
        
        message_content = {"content": "test message"}
        result = await agent.process_message(message_content)
        
        assert result == {"result": "success"}
        agent.execute_task.assert_called_once()
    
    async def test_process_message_invalid_format(self, agent):
        """Test handling of invalid message format."""
        message_content = {"invalid": "format"}  # Missing 'content' key
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.process_message(message_content)
        
        assert exc_info.value.category == ErrorCategory.VALIDATION
    
    async def test_inter_agent_communication(self, agent):
        """Test communication with other agents."""
        agent.communication_protocol.query.return_value = {"response": "test_response"}
        
        result = await agent.communicate("Hello", "OtherAgent")
        
        assert "Sent: Hello" in result
        assert "Received: {'response': 'test_response'}" in result
        agent.communication_protocol.query.assert_called_once()
    
    async def test_communication_failure(self, agent):
        """Test communication failure handling."""
        agent.communication_protocol.query.side_effect = Exception("Network error")
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.communicate("Hello", "OtherAgent")
        
        assert exc_info.value.category == ErrorCategory.COMMUNICATION


@pytest.mark.unit
class TestUnifiedBaseAgentRAGIntegration:
    """Test RAG integration methods."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol, mock_rag_pipeline):
        """Create agent with mocked RAG pipeline."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(return_value=mock_rag_pipeline),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.rag_pipeline = mock_rag_pipeline
            return agent
    
    async def test_generate_response(self, agent, mock_llm):
        """Test response generation."""
        agent.llm = mock_llm
        
        result = await agent.generate("Test prompt")
        
        assert result == "Generated response"
        mock_llm.complete.assert_called_once_with("Test prompt")
    
    async def test_generate_response_failure(self, agent, mock_llm):
        """Test response generation failure."""
        agent.llm = mock_llm
        mock_llm.complete.side_effect = Exception("LLM error")
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.generate("Test prompt")
        
        assert exc_info.value.category == ErrorCategory.EXTERNAL_SERVICE
    
    async def test_get_embedding(self, agent):
        """Test embedding retrieval."""
        result = await agent.get_embedding("Test text")
        
        assert result == [0.1] * 384
        agent.rag_pipeline.get_embedding.assert_called_once_with("Test text")
    
    async def test_get_embedding_failure(self, agent):
        """Test embedding retrieval failure."""
        agent.rag_pipeline.get_embedding.side_effect = Exception("Embedding error")
        
        with pytest.raises(AIVillageException) as exc_info:
            await agent.get_embedding("Test text")
        
        assert exc_info.value.category == ErrorCategory.PROCESSING
    
    async def test_rerank_results(self, agent):
        """Test result reranking."""
        results = [{"content": "item1"}, {"content": "item2"}]
        
        reranked = await agent.rerank("query", results, 1)
        
        assert reranked == [{"content": "ranked"}]
        agent.rag_pipeline.rerank.assert_called_once_with("query", results, 1)
    
    async def test_query_rag(self, agent):
        """Test RAG query processing."""
        result = await agent.query_rag("Test query")
        
        assert result == {"answer": "test answer"}
        agent.rag_pipeline.process_query.assert_called_once_with("Test query")
    
    async def test_add_document(self, agent):
        """Test document addition to RAG."""
        await agent.add_document("Test content", "test.txt")
        
        agent.rag_pipeline.add_document.assert_called_once_with("Test content", "test.txt")


@pytest.mark.unit
class TestUnifiedBaseAgentCapabilityManagement:
    """Test capability and tool management."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            return UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
    
    def test_add_capability(self, agent):
        """Test adding new capability."""
        initial_count = len(agent.capabilities)
        agent.add_capability("new_capability")
        
        assert len(agent.capabilities) == initial_count + 1
        assert "new_capability" in agent.capabilities
    
    def test_add_existing_capability(self, agent):
        """Test adding existing capability (should not duplicate)."""
        initial_capabilities = agent.capabilities.copy()
        agent.add_capability("test_capability")  # Already exists
        
        assert agent.capabilities == initial_capabilities
    
    def test_remove_capability(self, agent):
        """Test removing capability."""
        agent.remove_capability("test_capability")
        
        assert "test_capability" not in agent.capabilities
    
    def test_remove_nonexistent_capability(self, agent):
        """Test removing non-existent capability."""
        initial_capabilities = agent.capabilities.copy()
        agent.remove_capability("nonexistent_capability")
        
        assert agent.capabilities == initial_capabilities
    
    def test_add_tool(self, agent):
        """Test adding tool."""
        def test_tool():
            return "tool_result"
        
        agent.add_tool("test_tool", test_tool)
        
        assert "test_tool" in agent.tools
        assert agent.get_tool("test_tool") == test_tool
    
    def test_remove_tool(self, agent):
        """Test removing tool."""
        def test_tool():
            return "tool_result"
        
        agent.add_tool("test_tool", test_tool)
        agent.remove_tool("test_tool")
        
        assert "test_tool" not in agent.tools
        assert agent.get_tool("test_tool") is None
    
    def test_agent_info_property(self, agent):
        """Test agent info property."""
        info = agent.info
        
        assert info["name"] == agent.name
        assert info["description"] == agent.description
        assert info["capabilities"] == agent.capabilities
        assert info["model"] == agent.model
        assert "tools" in info


@pytest.mark.unit
class TestUnifiedBaseAgentAdvancedFeatures:
    """Test advanced agent features."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol, mock_llm):
        """Create agent for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
            agent.llm = mock_llm
            return agent
    
    async def test_latent_space_activation(self, agent, mock_llm):
        """Test latent space activation."""
        mock_llm.complete.return_value.text = (
            "Background knowledge about the topic.\n"
            "Refined Query: Enhanced query with background knowledge"
        )
        
        background, refined_query = await agent.activate_latent_space("Original query")
        
        assert "Background knowledge" in background
        assert refined_query == "Enhanced query with background knowledge"
    
    async def test_latent_space_activation_no_refinement(self, agent, mock_llm):
        """Test latent space activation without refined query."""
        mock_llm.complete.return_value.text = "Just background knowledge"
        
        background, refined_query = await agent.activate_latent_space("Original query")
        
        assert background == "Just background knowledge"
        assert refined_query == "Original query"  # Falls back to original
    
    async def test_create_handoff(self, agent):
        """Test handoff tool creation."""
        target_agent = Mock()
        target_agent.name = "TargetAgent"
        
        agent.create_handoff(target_agent)
        
        assert f"transfer_to_{target_agent.name}" in agent.tools
        handoff_tool = agent.get_tool(f"transfer_to_{target_agent.name}")
        assert handoff_tool() == target_agent
    
    async def test_update_instructions(self, agent):
        """Test instruction updates."""
        new_instructions = "Updated instructions"
        
        await agent.update_instructions(new_instructions)
        
        assert agent.instructions == new_instructions
    
    async def test_introspect(self, agent):
        """Test agent introspection."""
        result = await agent.introspect()
        
        assert result == agent.info
    
    async def test_evolve(self, agent):
        """Test agent evolution."""
        # Mock all layer evolve methods
        agent.quality_assurance_layer.evolve = AsyncMock()
        agent.foundational_layer.evolve = AsyncMock()
        agent.continuous_learning_layer.evolve = AsyncMock()
        agent.agent_architecture_layer.evolve = AsyncMock()
        agent.decision_making_layer.evolve = AsyncMock()
        
        await agent.evolve()
        
        # Verify all layers evolved
        agent.quality_assurance_layer.evolve.assert_called_once()
        agent.foundational_layer.evolve.assert_called_once()
        agent.continuous_learning_layer.evolve.assert_called_once()
        agent.agent_architecture_layer.evolve.assert_called_once()
        agent.decision_making_layer.evolve.assert_called_once()


@pytest.mark.unit
class TestAgentLayers:
    """Test individual agent layers."""
    
    def test_quality_assurance_layer_initialization(self):
        """Test quality assurance layer initialization."""
        layer = QualityAssuranceLayer(upo_threshold=0.8)
        
        assert layer.upo_threshold == 0.8
    
    def test_quality_assurance_layer_safety_check(self):
        """Test safety check functionality."""
        layer = QualityAssuranceLayer(upo_threshold=0.7)
        task = Mock()
        
        with patch.object(layer, 'estimate_uncertainty', return_value=0.5):
            assert layer.check_task_safety(task) is True
        
        with patch.object(layer, 'estimate_uncertainty', return_value=0.8):
            assert layer.check_task_safety(task) is False
    
    async def test_quality_assurance_layer_evolve(self):
        """Test quality assurance layer evolution."""
        layer = QualityAssuranceLayer(upo_threshold=0.7)
        initial_threshold = layer.upo_threshold
        
        await layer.evolve()
        
        # Threshold should change (within bounds)
        assert 0.5 <= layer.upo_threshold <= 0.9
    
    def test_foundational_layer_initialization(self, mock_vector_store):
        """Test foundational layer initialization."""
        layer = FoundationalLayer(mock_vector_store)
        
        assert layer.vector_store == mock_vector_store
        assert layer.bake_strength == 1.0
        assert layer._history == []
    
    async def test_foundational_layer_process_task(self, mock_vector_store):
        """Test foundational layer task processing."""
        layer = FoundationalLayer(mock_vector_store)
        task = Mock()
        task.content = "Original content"
        
        processed_task = await layer.process_task(task)
        
        assert "Baked Knowledge:" in processed_task.content
        assert "Original content" in processed_task.content
        assert len(layer._history) == 1
    
    async def test_foundational_layer_evolve(self, mock_vector_store):
        """Test foundational layer evolution."""
        layer = FoundationalLayer(mock_vector_store)
        layer._history = [100] * 10  # Short content history
        initial_strength = layer.bake_strength
        
        await layer.evolve()
        
        # Should increase strength for short content
        assert layer.bake_strength >= initial_strength
        assert 0.5 <= layer.bake_strength <= 2.0
    
    def test_continuous_learning_layer_initialization(self, mock_vector_store):
        """Test continuous learning layer initialization."""
        layer = ContinuousLearningLayer(mock_vector_store)
        
        assert layer.vector_store == mock_vector_store
        assert layer.learning_rate == 0.05
        assert layer.performance_history == []
    
    async def test_continuous_learning_layer_update(self, mock_vector_store):
        """Test continuous learning layer update."""
        layer = ContinuousLearningLayer(mock_vector_store)
        task = Mock()
        task.content = "Test task"
        result = {"performance": 0.8}
        
        await layer.update(task, result)
        
        assert len(layer.performance_history) == 1
        assert layer.performance_history[0] == 0.8
        mock_vector_store.add_texts.assert_called_once()


@pytest.mark.unit  
class TestAgentFactory:
    """Test agent factory function."""
    
    async def test_create_agent_success(self, sample_agent_config, mock_communication_protocol):
        """Test successful agent creation."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = create_agent(
                "TestAgent",
                sample_agent_config,
                mock_communication_protocol
            )
            
            assert isinstance(agent, UnifiedBaseAgent)
            assert agent.name == "TestAgent"
    
    async def test_create_agent_failure(self, sample_agent_config, mock_communication_protocol):
        """Test agent creation failure."""
        with patch('agents.unified_base_agent.UnifiedBaseAgent',
                   side_effect=Exception("Creation failed")):
            
            with pytest.raises(AIVillageException) as exc_info:
                create_agent(
                    "TestAgent",
                    sample_agent_config,
                    mock_communication_protocol
                )
            
            assert exc_info.value.category == ErrorCategory.CREATION


@pytest.mark.unit
class TestSelfEvolvingSystem:
    """Test self-evolving system functionality."""
    
    @pytest.fixture
    def mock_agents(self, sample_agent_config, mock_communication_protocol):
        """Create mock agents for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agents = []
            for i in range(3):
                config = sample_agent_config
                config.name = f"Agent{i}"
                agent = UnifiedBaseAgent(config, mock_communication_protocol)
                agents.append(agent)
            return agents
    
    def test_self_evolving_system_initialization(self, mock_agents):
        """Test system initialization."""
        system = SelfEvolvingSystem(mock_agents)
        
        assert system.agents == mock_agents
        assert hasattr(system, 'sage_framework')
        assert hasattr(system, 'dpo')
        assert hasattr(system, 'quality_assurance')
    
    async def test_process_task_success(self, mock_agents):
        """Test successful task processing."""
        system = SelfEvolvingSystem(mock_agents)
        task = Mock()
        task.type = "test_capability"
        
        # Mock execute_task for matching agent
        mock_agents[0].execute_task = AsyncMock(return_value={"result": "success"})
        
        result = await system.process_task(task)
        
        assert result == {"result": "success"}
    
    async def test_process_task_no_suitable_agent(self, mock_agents):
        """Test task processing with no suitable agent."""
        system = SelfEvolvingSystem(mock_agents)
        task = Mock()
        task.type = "unknown_capability"
        
        with pytest.raises(AIVillageException) as exc_info:
            await system.process_task(task)
        
        assert exc_info.value.category == ErrorCategory.VALIDATION
    
    async def test_system_evolve(self, mock_agents):
        """Test system-wide evolution."""
        system = SelfEvolvingSystem(mock_agents)
        
        # Mock required methods
        system.analyze_agent_performance = AsyncMock(return_value={"test_capability": 0.8})
        system.generate_new_capabilities = AsyncMock(return_value=["new_cap"])
        system.optimize_upo_threshold = AsyncMock(return_value=0.75)
        system.evolve_decision_maker = AsyncMock()
        
        for agent in mock_agents:
            agent.evolve = AsyncMock()
        
        await system.evolve()
        
        # Verify all agents evolved
        for agent in mock_agents:
            agent.evolve.assert_called_once()


# Edge case and error condition tests
@pytest.mark.unit
class TestUnifiedBaseAgentEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    async def agent(self, sample_agent_config, mock_communication_protocol):
        """Create agent for testing."""
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            return UnifiedBaseAgent(sample_agent_config, mock_communication_protocol)
    
    async def test_empty_content_handling(self, agent):
        """Test handling of empty content."""
        message_content = {"content": ""}
        
        # Should not raise exception for empty content
        with patch.object(agent, 'execute_task', return_value={"result": "empty"}):
            result = await agent.process_message(message_content)
            assert result == {"result": "empty"}
    
    async def test_very_long_content_handling(self, agent, mock_llm):
        """Test handling of very long content."""
        agent.llm = mock_llm
        long_content = "x" * 10000
        
        result = await agent.generate(long_content)
        
        # Should truncate in error context if fails, but process if succeeds
        assert result == "Generated response"
    
    async def test_null_values_handling(self, agent):
        """Test handling of null values."""
        with pytest.raises(AIVillageException):
            await agent.process_message(None)
    
    async def test_concurrent_task_execution(self, agent, test_data_generator):
        """Test concurrent task execution."""
        tasks = test_data_generator.generate_task_batch(5)
        
        # Mock task processing
        agent._process_task = AsyncMock(return_value={"processed": True})
        agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        agent.decision_making_layer.make_decision = AsyncMock(return_value="decision")
        agent.continuous_learning_layer.update = AsyncMock()
        
        # Execute tasks concurrently
        results = await asyncio.gather(*[agent.execute_task(task) for task in tasks])
        
        assert len(results) == 5
        assert all(result["result"] == "decision" for result in results)
    
    async def test_memory_cleanup_on_failure(self, agent, isolation_manager):
        """Test memory cleanup when operations fail."""
        # This test ensures no memory leaks on failures
        initial_refs = len(agent.__dict__)
        
        try:
            # Force a failure that might leave references
            with patch.object(agent, '_process_task', side_effect=MemoryError("Out of memory")):
                await agent.execute_task(Mock())
        except:
            pass
        
        # Memory structure should remain consistent
        final_refs = len(agent.__dict__)
        assert final_refs == initial_refs