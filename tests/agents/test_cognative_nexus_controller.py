"""
Comprehensive Test Suite for CognativeNexusController

Tests all critical functionality including:
- Agent creation with <500ms performance target
- NoneType error prevention and validation
- Task processing with ACT halting
- >95% task completion rate validation
- System performance metrics and reporting
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch

# Import the system under test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.agents.cognative_nexus_controller import (
    CognativeNexusController,
    create_cognative_nexus_controller,
    AgentType,
    AgentStatus,
    TaskPriority,
    CognativeTask,
)


class TestCognativeNexusController:
    """Comprehensive test suite for the unified agent controller"""
    
    @pytest.fixture
    async def controller(self):
        """Create controller for testing"""
        # Mock cognitive nexus to avoid complex dependencies
        with patch('core.agents.cognative_nexus_controller.CognitiveNexus', None):
            controller = CognativeNexusController(enable_cognitive_nexus=False)
            await controller.initialize()
            yield controller
            await controller.shutdown()
    
    @pytest.mark.asyncio
    async def test_controller_initialization(self):
        """Test controller initializes successfully"""
        controller = CognativeNexusController(enable_cognitive_nexus=False)
        
        # Should initialize successfully
        success = await controller.initialize()
        assert success is True
        assert controller.is_initialized is True
        
        # Should have performance metrics initialized
        assert 'total_agents_created' in controller.performance_metrics
        assert controller.performance_metrics['total_agents_created'] == 0
        
        await controller.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_creation_performance(self, controller):
        """Test agent creation meets <500ms performance target"""
        
        # Test multiple agent creations
        creation_times = []
        
        for i in range(5):
            start_time = time.perf_counter()
            
            agent_id = await controller.create_agent(
                AgentType.SAGE, 
                f"test_sage_{i}",
                capabilities=['knowledge_retrieval', 'reasoning']
            )
            
            creation_time = (time.perf_counter() - start_time) * 1000
            creation_times.append(creation_time)
            
            # Validate successful creation
            assert agent_id is not None
            assert agent_id in controller.agents
            
            # Validate performance target
            assert creation_time < 500, f"Agent creation took {creation_time:.1f}ms, exceeds 500ms target"
        
        # Validate average performance
        avg_creation_time = sum(creation_times) / len(creation_times)
        assert avg_creation_time < 500, f"Average creation time {avg_creation_time:.1f}ms exceeds target"
    
    @pytest.mark.asyncio
    async def test_agent_creation_success_rate(self, controller):
        """Test 100% agent creation success rate"""
        
        total_attempts = 20
        successful_creations = 0
        
        for i in range(total_attempts):
            agent_id = await controller.create_agent(
                AgentType.MAGI,
                f"test_magi_{i}"
            )
            
            if agent_id is not None:
                successful_creations += 1
        
        success_rate = (successful_creations / total_attempts) * 100
        assert success_rate == 100.0, f"Creation success rate {success_rate}% below 100% target"
    
    @pytest.mark.asyncio
    async def test_nonetype_error_prevention(self, controller):
        """Test that NoneType errors are prevented during agent creation"""
        
        # Test with various configurations that previously caused NoneType errors
        test_cases = [
            {'agent_type': AgentType.KING, 'agent_id': 'test_king'},
            {'agent_type': AgentType.SHIELD, 'agent_id': 'test_shield', 'capabilities': []},
            {'agent_type': AgentType.ORACLE, 'agent_id': 'test_oracle', 'capabilities': None},
        ]
        
        for test_case in test_cases:
            agent_id = await controller.create_agent(**test_case)
            
            # Should never return None due to proper dependency injection
            assert agent_id is not None, f"Agent creation returned None for {test_case}"
            
            # Agent should be properly registered
            assert agent_id in controller.agents
            
            # Agent should have all required services initialized
            agent = controller.agents[agent_id].agent
            assert hasattr(agent, '_embedding_service')
            assert hasattr(agent, '_communication_service')
            assert hasattr(agent, '_introspection_service')
            assert agent._embedding_service is not None
            assert agent._communication_service is not None
            assert agent._introspection_service is not None
    
    @pytest.mark.asyncio
    async def test_act_halting_system(self, controller):
        """Test ACT halting with iterative refinement"""
        
        # Create test agent
        agent_id = await controller.create_agent(AgentType.STRATEGIST, "test_strategist")
        assert agent_id is not None
        
        # Create task with ACT halting configuration
        task = CognativeTask(
            task_id="test_act_halting",
            description="Analyze the strategic implications of AI development",
            priority=TaskPriority.HIGH,
            max_iterations=3,
            halt_on_confidence=0.8,
            iterative_refinement=True
        )
        
        # Process task
        result = await controller.process_task_with_act_halting(task)
        
        # Validate results
        assert result['status'] == 'success'
        assert 'confidence' in result
        assert 'iterations_used' in result
        assert result['iterations_used'] >= 1
        assert result['iterations_used'] <= task.max_iterations
    
    @pytest.mark.asyncio
    async def test_task_completion_rate(self, controller):
        """Test >95% task completion rate target"""
        
        # Create multiple agents for task processing
        for i in range(3):
            agent_id = await controller.create_agent(AgentType.SAGE, f"sage_{i}")
            assert agent_id is not None
        
        # Process multiple tasks
        total_tasks = 20
        successful_tasks = 0
        
        for i in range(total_tasks):
            task = CognativeTask(
                task_id=f"test_task_{i}",
                description=f"Process task number {i}",
                priority=TaskPriority.NORMAL,
                max_iterations=2
            )
            
            result = await controller.process_task_with_act_halting(task)
            
            if result['status'] == 'success':
                successful_tasks += 1
        
        completion_rate = (successful_tasks / total_tasks) * 100
        assert completion_rate >= 95.0, f"Task completion rate {completion_rate}% below 95% target"
    
    @pytest.mark.asyncio
    async def test_agent_types_comprehensive(self, controller):
        """Test creation of all agent types"""
        
        created_agents = []
        
        # Test creation of each agent type
        for agent_type in AgentType:
            agent_id = await controller.create_agent(agent_type, f"test_{agent_type.value}")
            
            assert agent_id is not None, f"Failed to create agent of type {agent_type.value}"
            created_agents.append((agent_type, agent_id))
            
            # Validate agent registration
            assert agent_id in controller.agents
            assert controller.agents[agent_id].agent_type == agent_type
        
        # Validate type indexing
        for agent_type, agent_id in created_agents:
            assert agent_id in controller.agent_types_index[agent_type]
    
    @pytest.mark.asyncio
    async def test_performance_reporting(self, controller):
        """Test system performance reporting"""
        
        # Create some agents and process some tasks
        await controller.create_agent(AgentType.MAGI, "test_magi")
        await controller.create_agent(AgentType.SAGE, "test_sage")
        
        task = CognativeTask(
            task_id="test_reporting",
            description="Test task for performance reporting",
            priority=TaskPriority.NORMAL
        )
        
        await controller.process_task_with_act_halting(task)
        
        # Get performance report
        report = await controller.get_system_performance_report()
        
        # Validate report structure
        assert 'system_status' in report
        assert 'agent_performance' in report
        assert 'task_performance' in report
        assert 'targets_status' in report
        
        # Validate system status
        assert report['system_status']['initialized'] is True
        assert 'uptime_seconds' in report['system_status']
        
        # Validate agent performance metrics
        agent_perf = report['agent_performance']
        assert agent_perf['total_agents'] >= 2
        assert agent_perf['creation_success_rate_percent'] == 100.0
        assert agent_perf['instantiation_target_met'] is True
        
        # Validate task performance metrics
        task_perf = report['task_performance']
        assert task_perf['total_tasks_processed'] >= 1
        assert task_perf['task_completion_rate_percent'] >= 95.0
    
    @pytest.mark.asyncio
    async def test_cognitive_reasoning_integration(self):
        """Test cognitive reasoning integration when available"""
        
        # Test with cognitive nexus disabled (our default test case)
        controller = CognativeNexusController(enable_cognitive_nexus=False)
        await controller.initialize()
        
        assert controller.cognitive_nexus is None
        
        # Create task with reasoning requirements
        task = CognativeTask(
            task_id="test_reasoning",
            description="Complex reasoning task",
            priority=TaskPriority.HIGH,
            requires_reasoning=True
        )
        
        # Should still process successfully without cognitive nexus
        agent_id = await controller.create_agent(AgentType.ORACLE, "test_oracle")
        result = await controller.process_task_with_act_halting(task)
        
        assert result['status'] == 'success'
        
        await controller.shutdown()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, controller):
        """Test system error handling and recovery"""
        
        # Test with invalid agent type (should handle gracefully)
        with patch('core.agents.cognative_nexus_controller.BaseAgent') as mock_base_agent:
            # Simulate agent creation failure
            mock_base_agent.side_effect = Exception("Simulated creation failure")
            
            agent_id = await controller.create_agent(AgentType.KING, "test_error_handling")
            
            # Should return None but not crash
            assert agent_id is None
            assert controller.performance_metrics['total_agent_creation_failures'] > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, controller):
        """Test concurrent agent operations"""
        
        async def create_agent_task(agent_type, agent_id):
            return await controller.create_agent(agent_type, agent_id)
        
        # Create multiple agents concurrently
        tasks = [
            create_agent_task(AgentType.SAGE, f"concurrent_sage_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        for result in results:
            assert result is not None
        
        # Validate all agents are registered
        assert len(controller.agents) >= 5
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self, controller):
        """Test clean system shutdown"""
        
        # Create some agents
        await controller.create_agent(AgentType.MAGI, "test_shutdown")
        
        initial_agent_count = len(controller.agents)
        assert initial_agent_count > 0
        
        # Shutdown should complete without errors
        await controller.shutdown()
        
        # System should be marked as not initialized
        assert controller.is_initialized is False
        
        # All agents should be offline
        for registration in controller.agents.values():
            assert registration.status == AgentStatus.OFFLINE


class TestPerformanceValidation:
    """Dedicated performance validation tests"""
    
    @pytest.mark.asyncio
    async def test_large_scale_agent_creation(self):
        """Test performance with large number of agents"""
        
        controller = CognativeNexusController(enable_cognitive_nexus=False)
        await controller.initialize()
        
        try:
            # Create 50 agents and measure performance
            start_time = time.perf_counter()
            
            created_agents = []
            for i in range(50):
                agent_type = list(AgentType)[i % len(AgentType)]
                agent_id = await controller.create_agent(agent_type, f"scale_test_{i}")
                if agent_id:
                    created_agents.append(agent_id)
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Validate performance targets
            assert len(created_agents) == 50, "Not all agents created successfully"
            assert total_time < 25000, f"Large scale creation took {total_time:.1f}ms, target <25s"
            
            # Validate average instantiation time
            avg_time = controller.performance_metrics['average_instantiation_time_ms']
            assert avg_time < 500, f"Average instantiation time {avg_time:.1f}ms exceeds 500ms target"
            
        finally:
            await controller.shutdown()


if __name__ == "__main__":
    # Run basic smoke test
    async def smoke_test():
        print("Running CognativeNexusController smoke test...")
        
        controller = CognativeNexusController(enable_cognitive_nexus=False)
        success = await controller.initialize()
        
        if success:
            print("✅ Controller initialization successful")
            
            # Create test agent
            agent_id = await controller.create_agent(AgentType.SAGE, "smoke_test_sage")
            if agent_id:
                print(f"✅ Agent creation successful: {agent_id}")
                
                # Test task processing
                task = CognativeTask(
                    task_id="smoke_test",
                    description="Simple test task",
                    priority=TaskPriority.NORMAL
                )
                
                result = await controller.process_task_with_act_halting(task)
                if result['status'] == 'success':
                    print("✅ Task processing successful")
                else:
                    print(f"❌ Task processing failed: {result}")
            else:
                print("❌ Agent creation failed")
            
            # Get performance report
            report = await controller.get_system_performance_report()
            print(f"📊 Performance Report: {report['targets_status']}")
            
            await controller.shutdown()
            print("✅ System shutdown successful")
        else:
            print("❌ Controller initialization failed")
    
    # Run smoke test
    asyncio.run(smoke_test())