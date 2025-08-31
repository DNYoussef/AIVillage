"""
Test suite for AgentForgeTrainingService

Tests the agent-specific training capabilities including:
- Agent behavior training configurations
- Multi-agent coordination training
- Task-specific fine-tuning workflows
- Skill acquisition and adaptation
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
import sys
from pathlib import Path

# Add the infrastructure directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent / "infrastructure"))

from gateway.services.agent_forge_training_service import (
    AgentForgeTrainingService,
    AgentTrainingConfig,
    AgentTrainingMode,
    AgentArchitecture,
    MockAgentProgressEmitter,
    MockAgentEnvironmentLoader,
    MockAgentModelTrainer
)


class TestAgentForgeTrainingService:
    """Test suite for agent training service."""
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return AgentTrainingConfig(
            max_episodes=100,
            batch_size=4,
            learning_rate=1e-4,
            training_mode=AgentTrainingMode.BEHAVIOR_ADAPTATION,
            agent_architecture=AgentArchitecture.HIERARCHICAL_AGENT,
            hidden_size=64,
            num_layers=2,
            max_agents=3,
            coordination_strategy="hierarchical"
        )
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for testing."""
        return {
            "progress_emitter": MockAgentProgressEmitter(),
            "environment_loader": MockAgentEnvironmentLoader(),
            "agent_trainer": MockAgentModelTrainer()
        }
    
    @pytest.fixture
    def training_service(self, mock_dependencies, training_config):
        """Create training service with mocked dependencies."""
        return AgentForgeTrainingService(
            progress_emitter=mock_dependencies["progress_emitter"],
            environment_loader=mock_dependencies["environment_loader"],
            agent_trainer=mock_dependencies["agent_trainer"],
            config=training_config
        )
    
    @pytest.mark.asyncio
    async def test_start_agent_training_session(self, training_service):
        """Test starting an agent training session."""
        task_id = "test_task_001"
        training_parameters = {
            "focus": "coordination",
            "complexity": "intermediate"
        }
        
        session_info = await training_service.start_agent_training_session(
            task_id=task_id,
            training_parameters=training_parameters
        )
        
        assert session_info["task_id"] == task_id
        assert session_info["status"] == "initializing"
        assert session_info["training_mode"] == AgentTrainingMode.BEHAVIOR_ADAPTATION.value
        assert len(session_info["agent_specifications"]) == 3  # Default count
        assert session_info in training_service.active_training_sessions.values()
    
    @pytest.mark.asyncio
    async def test_execute_agent_training_pipeline(self, training_service):
        """Test complete agent training pipeline execution."""
        task_id = "test_task_002"
        
        # Start session
        await training_service.start_agent_training_session(
            task_id=task_id,
            training_parameters={"focus": "multi_agent"}
        )
        
        # Execute pipeline
        trained_agents = await training_service.execute_agent_training_pipeline(task_id)
        
        assert len(trained_agents) >= 1  # At least one agent should be trained
        assert all(agent.training_status == "completed" for agent in trained_agents)
        assert all(agent.training_mode == AgentTrainingMode.BEHAVIOR_ADAPTATION.value for agent in trained_agents)
        
        # Check session status
        session = await training_service.get_agent_training_status(task_id)
        assert session["status"] == "completed"
        assert session["trained_agents"] == [agent.agent_id for agent in trained_agents]
    
    def test_agent_parameter_count_calculation(self, training_service):
        """Test agent parameter count calculation (non-Cognate)."""
        param_count = training_service._calculate_agent_parameter_count()
        
        # Should be reasonable for agent architecture (not 25M like Cognate)
        assert 1000 < param_count < 1000000  # Agent-sized, not large model
        assert param_count != 25083528  # Should not match Cognate model size
    
    @pytest.mark.asyncio
    async def test_agent_artifacts_creation(self, training_service):
        """Test creation of agent artifacts."""
        agent_spec = {
            "name": "test_coordinator",
            "architecture": "hierarchical_agent",
            "role": "coordination"
        }
        
        training_stats = {
            "total_episodes": 100,
            "final_reward": 0.85,
            "success_rate": 0.78,
            "coordination_score": 0.82,
            "adaptation_speed": 0.75
        }
        
        environment_results = {"simulation": True, "interactive": True}
        
        artifacts = await training_service._create_agent_artifacts(
            agent_spec, 0, training_stats, environment_results
        )
        
        assert artifacts.agent_name == "Trained Test Coordinator"
        assert artifacts.training_mode == AgentTrainingMode.BEHAVIOR_ADAPTATION.value
        assert artifacts.agent_architecture == AgentArchitecture.HIERARCHICAL_AGENT.value
        assert artifacts.specialization == "coordination"
        assert "behavior_policy.json" in artifacts.artifacts["behavior_policy"]
        assert artifacts.performance_metrics["reward"] == 0.85
        assert "Hierarchical decision making" in artifacts.capabilities
    
    @pytest.mark.asyncio
    async def test_different_training_modes(self, mock_dependencies, training_config):
        """Test different agent training modes."""
        modes_to_test = [
            AgentTrainingMode.TASK_SPECIALIZATION,
            AgentTrainingMode.COORDINATION_TRAINING,
            AgentTrainingMode.SKILL_ACQUISITION,
            AgentTrainingMode.MULTI_AGENT_COLLABORATION
        ]
        
        for mode in modes_to_test:
            training_config.training_mode = mode
            service = AgentForgeTrainingService(
                progress_emitter=mock_dependencies["progress_emitter"],
                environment_loader=mock_dependencies["environment_loader"],
                agent_trainer=mock_dependencies["agent_trainer"],
                config=training_config
            )
            
            task_id = f"test_{mode.value}"
            session_info = await service.start_agent_training_session(
                task_id=task_id,
                training_parameters={"mode": mode.value}
            )
            
            assert session_info["training_mode"] == mode.value
    
    @pytest.mark.asyncio
    async def test_different_agent_architectures(self, mock_dependencies, training_config):
        """Test different agent architectures."""
        architectures_to_test = [
            AgentArchitecture.PLANNING_AGENT,
            AgentArchitecture.REACTIVE_AGENT,
            AgentArchitecture.HYBRID_AGENT
        ]
        
        for arch in architectures_to_test:
            training_config.agent_architecture = arch
            service = AgentForgeTrainingService(
                progress_emitter=mock_dependencies["progress_emitter"],
                environment_loader=mock_dependencies["environment_loader"],
                agent_trainer=mock_dependencies["agent_trainer"],
                config=training_config
            )
            
            # Test that configuration is properly set
            assert service.config.agent_architecture == arch
            
            # Test parameter calculation varies by architecture
            param_count = service._calculate_agent_parameter_count()
            assert param_count > 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, training_service):
        """Test training session management capabilities."""
        task_id = "test_session_mgmt"
        
        # Test session creation
        await training_service.start_agent_training_session(
            task_id=task_id,
            training_parameters={"test": "session_mgmt"}
        )
        
        # Test session status retrieval
        status = await training_service.get_agent_training_status(task_id)
        assert status is not None
        assert status["task_id"] == task_id
        
        # Test session cancellation
        cancelled = await training_service.cancel_agent_training_session(task_id)
        assert cancelled is True
        
        # Verify cancellation
        status = await training_service.get_agent_training_status(task_id)
        assert status["status"] == "cancelled"
    
    @pytest.mark.asyncio
    async def test_mock_implementations(self, mock_dependencies):
        """Test that mock implementations work correctly."""
        # Test progress emitter
        progress_emitter = mock_dependencies["progress_emitter"]
        assert len(progress_emitter.progress_events) == 0
        assert len(progress_emitter.agent_events) == 0
        
        # Test environment loader
        env_loader = mock_dependencies["environment_loader"]
        env_path = await env_loader.setup_environment("simulation", {})
        assert "simulation_environment" in env_path
        
        scenario_path = await env_loader.create_multi_agent_scenario(3)
        assert "multi_agent_scenario_3" in scenario_path
        
        # Test agent trainer
        trainer = mock_dependencies["agent_trainer"]
        config = AgentTrainingConfig(max_episodes=10)
        
        async def mock_callback(ep, total, reward, success, coord):
            pass
        
        stats = await trainer.train_agent("test_agent", config, mock_callback)
        assert stats["agent_name"] == "test_agent"
        assert stats["total_episodes"] == 10
        assert 0 <= stats["success_rate"] <= 1
        assert 0 <= stats["coordination_score"] <= 1


@pytest.mark.asyncio
async def test_integration_example():
    """Test complete integration example."""
    # Create service with mock dependencies
    service = AgentForgeTrainingService(
        progress_emitter=MockAgentProgressEmitter(),
        environment_loader=MockAgentEnvironmentLoader(),
        agent_trainer=MockAgentModelTrainer(),
        config=AgentTrainingConfig(
            max_episodes=50,
            training_mode=AgentTrainingMode.MULTI_AGENT_COLLABORATION,
            agent_architecture=AgentArchitecture.HYBRID_AGENT,
            coordination_strategy="mesh"
        )
    )
    
    # Define custom agent specifications
    agent_specifications = [
        {"name": "leader_agent", "architecture": "hierarchical_agent", "role": "leadership"},
        {"name": "worker_agent", "architecture": "reactive_agent", "role": "execution"},
        {"name": "advisor_agent", "architecture": "planning_agent", "role": "strategy"}
    ]
    
    # Start training session
    task_id = "integration_test"
    session_info = await service.start_agent_training_session(
        task_id=task_id,
        training_parameters={"integration": True},
        agent_specifications=agent_specifications
    )
    
    # Execute training pipeline
    trained_agents = await service.execute_agent_training_pipeline(task_id)
    
    # Verify results
    assert len(trained_agents) >= 1
    assert all(agent.training_mode == AgentTrainingMode.MULTI_AGENT_COLLABORATION.value for agent in trained_agents)
    assert any("coordination" in cap.lower() for agent in trained_agents for cap in agent.capabilities)
    
    # Check trained agents storage
    all_agents = await service.list_trained_agents()
    assert len(all_agents) >= len(trained_agents)
    
    # Verify each agent has proper artifacts
    for agent in trained_agents:
        artifacts = await service.get_agent_artifacts(agent.agent_id)
        assert artifacts is not None
        assert artifacts.agent_id == agent.agent_id
        assert "behavior_policy" in artifacts.artifacts
        assert len(artifacts.performance_metrics) > 0


if __name__ == "__main__":
    # Run a simple test
    asyncio.run(test_integration_example())
    print("AgentForgeTrainingService integration test passed!")