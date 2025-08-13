"""Tests for the unified base agent with error handling integration."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from agents.unified_base_agent import (
    SelfEvolvingSystem,
    UnifiedAgentConfig,
    UnifiedBaseAgent,
    _DPOModule,
    _SageFramework,
    create_agent,
)
from agents.utils.task import Task as LangroidTask
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore

from core.error_handling import (
    AIVillageException,
    ErrorCategory,
    ErrorSeverity,
    StandardCommunicationProtocol,
)


class TestUnifiedBaseAgent:
    """Test suite for UnifiedBaseAgent with error handling."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent configuration."""
        return UnifiedAgentConfig(
            name="TestAgent",
            description="A test agent",
            capabilities=["test_capability"],
            rag_config=MagicMock(spec=UnifiedConfig),
            vector_store=MagicMock(spec=VectorStore),
            model="gpt-4",
            instructions="Test instructions",
        )

    @pytest.fixture
    def mock_communication_protocol(self):
        """Create a mock communication protocol."""
        protocol = MagicMock(spec=StandardCommunicationProtocol)
        protocol.subscribe = AsyncMock()
        protocol.send_message = AsyncMock()
        return protocol

    @pytest.fixture
    def agent(self, mock_config, mock_communication_protocol):
        """Create a test agent instance."""
        with patch(
            "agents.language_models.openai_gpt.OpenAIGPTConfig"
        ) as mock_llm_config:
            mock_llm = MagicMock()
            mock_llm.complete = AsyncMock(return_value=MagicMock(text="test response"))
            mock_llm_config.return_value.create.return_value = mock_llm

            agent = UnifiedBaseAgent(mock_config, mock_communication_protocol)
            yield agent

    def test_agent_initialization_success(
        self, mock_config, mock_communication_protocol
    ):
        """Test successful agent initialization."""
        with patch(
            "agents.language_models.openai_gpt.OpenAIGPTConfig"
        ) as mock_llm_config:
            mock_llm = MagicMock()
            mock_llm_config.return_value.create.return_value = mock_llm

            agent = UnifiedBaseAgent(mock_config, mock_communication_protocol)
            assert agent.name == "TestAgent"
            assert agent.description == "A test agent"
            assert "test_capability" in agent.capabilities

    def test_agent_initialization_failure(
        self, mock_config, mock_communication_protocol
    ):
        """Test agent initialization failure with proper error handling."""
        with patch(
            "agents.language_models.openai_gpt.OpenAIGPTConfig"
        ) as mock_llm_config:
            mock_llm_config.return_value.create.side_effect = Exception(
                "LLM init failed"
            )

            with pytest.raises(AIVillageException) as exc_info:
                UnifiedBaseAgent(mock_config, mock_communication_protocol)

            assert exc_info.value.category == ErrorCategory.INITIALIZATION
            assert exc_info.value.severity == ErrorSeverity.CRITICAL
            assert "Failed to initialize agent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_task_success(self, agent):
        """Test successful task execution."""
        # Mock the _process_task method
        agent._process_task = AsyncMock(return_value={"result": "success"})

        task = LangroidTask(agent, "test task content")
        result = await agent.execute_task(task)

        assert result["result"] == "success"

    @pytest.mark.asyncio
    async def test_execute_task_quality_assurance_failure(self, agent):
        """Test task execution failure due to quality assurance."""
        with patch.object(
            agent.quality_assurance_layer, "check_task_safety", return_value=False
        ):
            task = LangroidTask(agent, "unsafe task")

            with pytest.raises(AIVillageException) as exc_info:
                await agent.execute_task(task)

            assert exc_info.value.category == ErrorCategory.VALIDATION
            assert "Task deemed unsafe" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_task_processing_failure(self, agent):
        """Test task execution failure during processing."""
        agent._process_task = AsyncMock(side_effect=Exception("Processing failed"))

        task = LangroidTask(agent, "test task")

        with pytest.raises(AIVillageException) as exc_info:
            await agent.execute_task(task)

        assert exc_info.value.category == ErrorCategory.PROCESSING
        assert exc_info.value.severity == ErrorSeverity.ERROR

    @pytest.mark.asyncio
    async def test_generate_success(self, agent):
        """Test successful text generation."""
        agent.llm.complete = AsyncMock(return_value=MagicMock(text="generated text"))

        result = await agent.generate("test prompt")
        assert result == "generated text"

    @pytest.mark.asyncio
    async def test_generate_failure(self, agent):
        """Test text generation failure."""
        agent.llm.complete = AsyncMock(side_effect=Exception("LLM error"))

        with pytest.raises(AIVillageException) as exc_info:
            await agent.generate("test prompt")

        assert exc_info.value.category == ErrorCategory.EXTERNAL_SERVICE
        assert "Failed to generate response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_embedding_success(self, agent):
        """Test successful embedding retrieval."""
        agent.rag_pipeline.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        result = await agent.get_embedding("test text")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_get_embedding_failure(self, agent):
        """Test embedding retrieval failure."""
        agent.rag_pipeline.get_embedding = AsyncMock(
            side_effect=Exception("Embedding error")
        )

        with pytest.raises(AIVillageException) as exc_info:
            await agent.get_embedding("test text")

        assert exc_info.value.category == ErrorCategory.PROCESSING
        assert "Failed to get embedding" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rerank_success(self, agent):
        """Test successful reranking."""
        agent.rag_pipeline.rerank = AsyncMock(return_value=[{"doc": "result"}])

        result = await agent.rerank("query", [{"doc": "doc1"}], 1)
        assert result == [{"doc": "result"}]

    @pytest.mark.asyncio
    async def test_rerank_failure(self, agent):
        """Test reranking failure."""
        agent.rag_pipeline.rerank = AsyncMock(side_effect=Exception("Rerank error"))

        with pytest.raises(AIVillageException) as exc_info:
            await agent.rerank("query", [{"doc": "doc1"}], 1)

        assert exc_info.value.category == ErrorCategory.PROCESSING
        assert "Failed to rerank results" in str(exc_info.value)

    def test_add_remove_capabilities(self, agent):
        """Test adding and removing capabilities."""
        agent.add_capability("new_capability")
        assert "new_capability" in agent.capabilities

        agent.remove_capability("new_capability")
        assert "new_capability" not in agent.capabilities

    def test_add_remove_tools(self, agent):
        """Test adding and removing tools."""

        def test_tool():
            return "tool result"

        agent.add_tool("test_tool", test_tool)
        assert agent.get_tool("test_tool") == test_tool

        agent.remove_tool("test_tool")
        assert agent.get_tool("test_tool") is None

    def test_info_property(self, agent):
        """Test the info property."""
        info = agent.info
        assert info["name"] == "TestAgent"
        assert info["description"] == "A test agent"
        assert "test_capability" in info["capabilities"]


class TestSelfEvolvingSystem:
    """Test suite for SelfEvolvingSystem with error handling."""

    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing."""
        agent1 = MagicMock(spec=UnifiedBaseAgent)
        agent1.name = "Agent1"
        agent1.capabilities = ["cap1", "cap2"]
        agent1.execute_task = AsyncMock(return_value={"result": "success"})
        agent1.evolve = AsyncMock()

        agent2 = MagicMock(spec=UnifiedBaseAgent)
        agent2.name = "Agent2"
        agent2.capabilities = ["cap3", "cap4"]
        agent2.execute_task = AsyncMock(return_value={"result": "success"})
        agent2.evolve = AsyncMock()

        return [agent1, agent2]

    @pytest.fixture
    def system(self, mock_agents):
        """Create a test SelfEvolvingSystem instance."""
        with (
            patch("agents.unified_base_agent._SageFramework"),
            patch("agents.unified_base_agent.MCTSConfig"),
            patch("agents.unified_base_agent._DPOModule"),
            patch("agents.unified_base_agent.BasicUPOChecker"),
        ):
            system = SelfEvolvingSystem(mock_agents)
            yield system

    @pytest.mark.asyncio
    async def test_system_initialization_success(self, mock_agents):
        """Test successful system initialization."""
        system = SelfEvolvingSystem(mock_agents)
        assert len(system.agents) == 2
        assert system.agents[0].name == "Agent1"

    @pytest.mark.asyncio
    async def test_process_task_success(self, system, mock_agents):
        """Test successful task processing."""
        task = LangroidTask(mock_agents[0], "test task")
        task.type = "cap1"

        result = await system.process_task(task)
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_process_task_no_suitable_agent(self, system):
        """Test task processing when no suitable agent is found."""
        task = MagicMock()
        task.type = "nonexistent_capability"

        with pytest.raises(AIVillageException) as exc_info:
            await system.process_task(task)

        assert exc_info.value.category == ErrorCategory.VALIDATION
        assert "No suitable agent found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evolve_system_success(self, system):
        """Test successful system evolution."""
        with (
            patch.object(
                system, "analyze_agent_performance", return_value={"cap1": 0.8}
            ),
            patch.object(system, "generate_new_capabilities", return_value=["new_cap"]),
        ):
            await system.evolve()

    @pytest.mark.asyncio
    async def test_analyze_agent_performance(self, system, mock_agents):
        """Test agent performance analysis."""
        performance = await system.analyze_agent_performance(mock_agents[0])
        assert isinstance(performance, dict)
        assert all(0.4 <= score <= 1.0 for score in performance.values())

    @pytest.mark.asyncio
    async def test_generate_new_capabilities(self, system, mock_agents):
        """Test new capability generation."""
        performance = {"cap1": 0.5, "cap2": 0.8}
        new_caps = await system.generate_new_capabilities(mock_agents[0], performance)
        assert isinstance(new_caps, list)

    @pytest.mark.asyncio
    async def test_evolve_decision_maker(self, system):
        """Test decision maker evolution."""
        await system.evolve_decision_maker()

    @pytest.mark.asyncio
    async def test_optimize_upo_threshold(self, system):
        """Test UPO threshold optimization."""
        threshold = await system.optimize_upo_threshold()
        assert 0.5 <= threshold <= 0.9

    def test_get_recent_decisions(self, system):
        """Test getting recent decisions."""
        system.recent_decisions = [(np.array([1, 2]), 1), (np.array([3, 4]), 0)]
        decisions = system.get_recent_decisions()
        assert len(decisions) == 2

    @pytest.mark.asyncio
    async def test_add_decision(self, system):
        """Test adding a decision."""
        features = np.array([1, 2, 3])
        await system.add_decision(features, 1)
        assert len(system.recent_decisions) == 1


class TestCreateAgent:
    """Test suite for create_agent factory function."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent configuration."""
        return UnifiedAgentConfig(
            name="FactoryAgent",
            description="Factory created agent",
            capabilities=["factory_capability"],
            rag_config=MagicMock(spec=UnifiedConfig),
            vector_store=MagicMock(spec=VectorStore),
            model="gpt-4",
            instructions="Factory instructions",
        )

    @pytest.fixture
    def mock_communication_protocol(self):
        """Create a mock communication protocol."""
        return MagicMock(spec=StandardCommunicationProtocol)

    def test_create_agent_success(self, mock_config, mock_communication_protocol):
        """Test successful agent creation via factory."""
        with patch(
            "agents.language_models.openai_gpt.OpenAIGPTConfig"
        ) as mock_llm_config:
            mock_llm = MagicMock()
            mock_llm_config.return_value.create.return_value = mock_llm

            agent = create_agent("TestAgent", mock_config, mock_communication_protocol)
            assert agent.name == "FactoryAgent"

    def test_create_agent_failure(self, mock_config, mock_communication_protocol):
        """Test agent creation failure via factory."""
        with patch(
            "agents.language_models.openai_gpt.OpenAIGPTConfig"
        ) as mock_llm_config:
            mock_llm_config.return_value.create.side_effect = Exception(
                "Creation failed"
            )

            with pytest.raises(AIVillageException) as exc_info:
                create_agent("TestAgent", mock_config, mock_communication_protocol)

            assert exc_info.value.category == ErrorCategory.CREATION
            assert "Failed to create agent" in str(exc_info.value)


class TestHelperClasses:
    """Test suite for helper classes."""

    def test_sage_framework(self):
        """Test _SageFramework functionality."""
        sage = _SageFramework()
        response = asyncio.run(sage.assistant_response("test prompt"))
        assert isinstance(response, str)

    def test_dpo_module(self):
        """Test _DPOModule functionality."""
        dpo = _DPOModule()

        # Test adding records
        features = np.array([1.0, 2.0, 3.0])
        dpo.add_record(features, 1)
        assert len(dpo.X) == 1
        assert len(dpo.y) == 1

        # Test fitting
        dpo.fit()

        # Test record limit
        for _i in range(1001):
            dpo.add_record(features, 1)
        assert len(dpo.X) <= 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
