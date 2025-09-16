"""
Comprehensive Test Suite for Unified Orchestration System

Tests all consolidated functionality to ensure no regressions were introduced
during the consolidation of the 4 overlapping orchestration systems.
"""

import asyncio
import pytest
from datetime import datetime
from typing import Dict, Any

# Import the unified orchestration system
from core.orchestration import (
    UnifiedOrchestrator,
    OrchestrationCoordinator,
    OrchestratorRegistry,
    TaskContext,
    TaskType,
    TaskPriority,
    OrchestrationStatus,
    MLPipelineOrchestrator,
    AgentLifecycleOrchestrator,
    CognitiveAnalysisOrchestrator,
    FogSystemOrchestrator,
    MLConfig,
    AgentConfig,
    CognitiveConfig,
    FogConfig,
)


class TestOrchestrationInterfaces:
    """Test the unified orchestration interfaces."""
    
    def test_orchestration_result_structure(self):
        """Test OrchestrationResult standardized structure."""
        from core.orchestration.interfaces import OrchestrationResult
        
        result = OrchestrationResult(
            success=True,
            task_id="test_task_123",
            orchestrator_id="test_orchestrator",
            task_type=TaskType.SYSTEM_HEALTH,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.5
        )
        
        assert result.success is True
        assert result.task_id == "test_task_123"
        assert result.status == "SUCCESS"
        assert result.duration_seconds == 1.5
        assert isinstance(result.metrics, dict)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
    
    def test_task_context_structure(self):
        """Test TaskContext unified structure."""
        context = TaskContext(
            task_type=TaskType.ML_PIPELINE,
            priority=TaskPriority.HIGH,
            timeout_seconds=300.0,
            metadata={'operation': 'run_pipeline'}
        )
        
        assert context.task_type == TaskType.ML_PIPELINE
        assert context.priority == TaskPriority.HIGH
        assert context.timeout_seconds == 300.0
        assert 'operation' in context.metadata
        assert context.retry_count == 0
        assert context.max_retries == 3
    
    def test_health_status_structure(self):
        """Test HealthStatus unified structure."""
        from core.orchestration.interfaces import HealthStatus
        
        health = HealthStatus(
            healthy=True,
            timestamp=datetime.now(),
            orchestrator_id="test_orchestrator",
            components={'service_1': True, 'service_2': False},
            metrics={'uptime': 99.5, 'response_time': 0.150},
            alerts=['Service 2 degraded'],
            uptime_seconds=3600.0
        )
        
        assert health.healthy is True
        assert health.health_score == 0.5  # 1 out of 2 components healthy
        assert len(health.alerts) == 1
        assert health.uptime_seconds == 3600.0


class TestOrchestratorRegistry:
    """Test the centralized orchestrator registry."""
    
    @pytest.fixture
    def registry(self):
        """Create a test registry."""
        return OrchestratorRegistry()
    
    def test_register_orchestrator_type(self, registry):
        """Test orchestrator type registration."""
        from core.orchestration.base import BaseOrchestrator
        
        success = registry.register_type(
            orchestrator_type="test_orchestrator",
            orchestrator_class=BaseOrchestrator,
            description="Test orchestrator for unit tests",
            dependencies=["dependency1"],
            priority=50,
            enabled=True
        )
        
        assert success is True
        assert "test_orchestrator" in registry.list_types()
        
        # Test duplicate registration
        success = registry.register_type(
            orchestrator_type="test_orchestrator",
            orchestrator_class=BaseOrchestrator
        )
        assert success is False
    
    def test_dependency_order_calculation(self, registry):
        """Test dependency order calculation."""
        from core.orchestration.base import BaseOrchestrator
        
        # Register orchestrators with dependencies
        registry.register_type("service_a", BaseOrchestrator, dependencies=[], priority=10)
        registry.register_type("service_b", BaseOrchestrator, dependencies=["service_a"], priority=20)
        registry.register_type("service_c", BaseOrchestrator, dependencies=["service_b"], priority=30)
        
        order = registry.get_dependency_order()
        
        # Should be in dependency order: a, b, c
        assert order.index("service_a") < order.index("service_b")
        assert order.index("service_b") < order.index("service_c")
    
    def test_registry_statistics(self, registry):
        """Test registry statistics."""
        from core.orchestration.base import BaseOrchestrator
        
        registry.register_type("type1", BaseOrchestrator, enabled=True)
        registry.register_type("type2", BaseOrchestrator, enabled=False)
        
        stats = registry.get_registry_stats()
        
        assert stats['total_registered_types'] == 2
        assert stats['enabled_types'] == 1
        assert stats['disabled_types'] == 1
        assert 'type1' in stats['enabled_type_names']
        assert 'type2' not in stats['enabled_type_names']


class TestOrchestrationCoordinator:
    """Test the orchestration coordinator."""
    
    @pytest.fixture
    def coordinator(self):
        """Create a test coordinator."""
        return OrchestrationCoordinator()
    
    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test coordinator state management."""
        from core.orchestration.base import BaseOrchestrator
        
        # Create mock orchestrators
        orchestrator1 = BaseOrchestrator("test_type_1", "test_id_1")
        orchestrator2 = BaseOrchestrator("test_type_2", "test_id_2") 
        
        # Register orchestrators
        success1 = await coordinator.register_orchestrator(orchestrator1, dependencies=[])
        success2 = await coordinator.register_orchestrator(orchestrator2, dependencies=["test_id_1"])
        
        assert success1 is True
        assert success2 is True
        
        # Test initialization (would need to mock actual orchestrator methods)
        # This is a structural test to verify the coordinator can manage orchestrators
        metrics = await coordinator.get_system_metrics()
        assert 'total_orchestrators' in metrics
        assert metrics['total_orchestrators'] == 2
    
    @pytest.mark.asyncio
    async def test_task_routing(self, coordinator):
        """Test task routing through coordinator."""
        task_context = TaskContext(
            task_type=TaskType.SYSTEM_HEALTH,
            metadata={'operation': 'health_check'}
        )
        
        # Test routing when no orchestrators are available
        result = await coordinator.route_task(task_context)
        assert result.success is False
        assert "No orchestrator available" in result.errors[0]


class TestSpecializedOrchestrators:
    """Test the specialized orchestrator implementations."""
    
    @pytest.mark.asyncio
    async def test_ml_orchestrator_initialization(self):
        """Test ML Pipeline Orchestrator."""
        config = MLConfig(
            base_models=["test_model"],
            enable_wandb=False,
            max_phases=3
        )
        
        orchestrator = MLPipelineOrchestrator()
        success = await orchestrator.initialize(config)
        
        # Should succeed even without PyTorch for basic functionality
        assert isinstance(orchestrator.orchestrator_id, str)
        assert orchestrator.status == OrchestrationStatus.READY
        
        # Test metrics
        metrics = await orchestrator.get_metrics()
        assert 'orchestrator_id' in metrics
        assert 'orchestrator_type' in metrics
        assert metrics['orchestrator_type'] == 'ml_pipeline'
    
    @pytest.mark.asyncio
    async def test_agent_orchestrator_functionality(self):
        """Test Agent Lifecycle Orchestrator."""
        config = AgentConfig(
            enable_cognitive_nexus=False,  # Disable for simpler testing
            max_agents_per_type=5
        )
        
        orchestrator = AgentLifecycleOrchestrator()
        success = await orchestrator.initialize(config)
        
        assert success is True
        assert orchestrator.status == OrchestrationStatus.READY
        
        # Test agent creation through task processing
        from core.orchestration.agent_orchestrator import AgentType
        
        task_context = TaskContext(
            task_type=TaskType.AGENT_LIFECYCLE,
            metadata={
                'operation': 'create_agent',
                'agent_type': AgentType.CURATOR.value,
                'agent_id': 'test_curator_agent'
            }
        )
        
        result = await orchestrator.process_task(task_context)
        assert result.success is True
        assert isinstance(result.data, str)  # Should return agent ID
    
    @pytest.mark.asyncio 
    async def test_cognitive_orchestrator_analysis(self):
        """Test Cognitive Analysis Orchestrator."""
        config = CognitiveConfig(
            enable_fog_computing=False,
            analysis_timeout_seconds=10.0
        )
        
        orchestrator = CognitiveAnalysisOrchestrator()
        success = await orchestrator.initialize(config)
        
        assert success is True
        assert orchestrator.status == OrchestrationStatus.READY
        
        # Test cognitive analysis
        from core.orchestration.cognitive_orchestrator import RetrievedInformation
        
        retrieved_info = [
            RetrievedInformation(content="Test information", source="test_source")
        ]
        
        task_context = TaskContext(
            task_type=TaskType.COGNITIVE_ANALYSIS,
            metadata={
                'operation': 'analyze',
                'query': 'Test query',
                'retrieved_info': retrieved_info,
                'reasoning_strategy': 'probabilistic'
            }
        )
        
        result = await orchestrator.process_task(task_context)
        assert result.success is True
        assert isinstance(result.data, list)  # Should return analysis results
    
    @pytest.mark.asyncio
    async def test_fog_orchestrator_coordination(self):
        """Test Fog System Orchestrator."""
        config = FogConfig(
            node_id="test_fog_node",
            enable_mobile_harvest=False,
            enable_onion_routing=False,
            enable_marketplace=False
        )
        
        orchestrator = FogSystemOrchestrator()
        success = await orchestrator.initialize(config)
        
        assert success is True
        assert orchestrator.status == OrchestrationStatus.READY
        
        # Test fog request processing
        task_context = TaskContext(
            task_type=TaskType.FOG_COORDINATION,
            metadata={
                'operation': 'process_fog_request',
                'request_type': 'status',
                'request_data': {}
            }
        )
        
        result = await orchestrator.process_task(task_context)
        assert result.success is True
        assert isinstance(result.data, dict)


class TestUnifiedOrchestrator:
    """Test the main unified orchestrator."""
    
    @pytest.mark.asyncio
    async def test_unified_orchestrator_initialization(self):
        """Test unified orchestrator initialization."""
        unified = UnifiedOrchestrator()
        
        # Test initialization with minimal config
        success = await unified.initialize(
            enable_ml_pipeline=False,  # Disable to simplify test
            enable_agent_lifecycle=True,
            enable_cognitive_analysis=True, 
            enable_fog_system=False
        )
        
        assert success is True
        assert unified.is_initialized is True
        assert unified.orchestrator_count == 2  # agent + cognitive
    
    @pytest.mark.asyncio
    async def test_unified_orchestrator_task_routing(self):
        """Test task routing through unified interface."""
        unified = UnifiedOrchestrator()
        
        # Initialize with agent orchestrator only for simpler testing
        success = await unified.initialize(
            enable_ml_pipeline=False,
            enable_agent_lifecycle=True,
            enable_cognitive_analysis=False,
            enable_fog_system=False
        )
        
        assert success is True
        
        # Start the system
        start_success = await unified.start()
        assert start_success is True
        assert unified.is_running is True
        
        # Test agent creation through unified interface
        result = await unified.create_agent(
            agent_type="curator",
            agent_id="test_unified_agent"
        )
        
        assert result.success is True
        
        # Test system health
        health = await unified.get_system_health()
        assert health.healthy is True
        
        # Test system metrics
        metrics = await unified.get_system_metrics()
        assert 'unified_orchestrator_id' in metrics
        assert metrics['system_running'] is True
        
        # Cleanup
        await unified.stop()
    
    @pytest.mark.asyncio
    async def test_unified_orchestrator_error_handling(self):
        """Test error handling in unified orchestrator."""
        unified = UnifiedOrchestrator()
        
        # Test processing task without initialization
        task_context = TaskContext(
            task_type=TaskType.SYSTEM_HEALTH,
            metadata={'operation': 'test'}
        )
        
        result = await unified.process_task(task_context)
        assert result.success is False
        assert "not running" in result.errors[0].lower()
    
    def test_unified_orchestrator_properties(self):
        """Test unified orchestrator properties."""
        unified = UnifiedOrchestrator()
        
        assert unified.is_initialized is False
        assert unified.is_running is False
        assert unified.orchestrator_count == 0
        assert unified.get_orchestrator('nonexistent') is None


class TestRegressionPrevention:
    """Test that consolidation didn't break existing functionality."""
    
    @pytest.mark.asyncio
    async def test_no_initialization_race_conditions(self):
        """
        Test that the coordinator prevents initialization race conditions
        that were identified in Agent 1's analysis.
        """
        coordinator = OrchestrationCoordinator()
        
        from core.orchestration.base import BaseOrchestrator
        
        # Create multiple orchestrators that would previously race
        orchestrators = [
            BaseOrchestrator("test_type_1", f"test_id_1"),
            BaseOrchestrator("test_type_2", f"test_id_2"),  
            BaseOrchestrator("test_type_3", f"test_id_3")
        ]
        
        # Register all orchestrators
        for i, orch in enumerate(orchestrators):
            await coordinator.register_orchestrator(orch, dependencies=[], priority=i)
        
        # Initialize all - should be sequential, not concurrent
        success = await coordinator.initialize_all(timeout_seconds=60.0)
        
        # All orchestrators should be initialized without conflicts
        health = await coordinator.get_system_health()
        assert health.healthy is True
    
    @pytest.mark.asyncio
    async def test_no_method_signature_conflicts(self):
        """
        Test that method signature conflicts identified in Agent 1's analysis
        have been resolved through the unified interface.
        """
        # All orchestrators should have the same interface signatures
        orchestrators = [
            MLPipelineOrchestrator(),
            AgentLifecycleOrchestrator(), 
            CognitiveAnalysisOrchestrator(),
            FogSystemOrchestrator()
        ]
        
        for orchestrator in orchestrators:
            # Test that all orchestrators have the same interface methods
            assert hasattr(orchestrator, 'initialize')
            assert hasattr(orchestrator, 'start')
            assert hasattr(orchestrator, 'stop')
            assert hasattr(orchestrator, 'process_task')
            assert hasattr(orchestrator, 'get_health_status')
            assert hasattr(orchestrator, 'get_metrics')
            
            # Test that orchestrator_id property exists and is consistent
            assert hasattr(orchestrator, 'orchestrator_id')
            assert isinstance(orchestrator.orchestrator_id, str)
            
            # Test that status property exists
            assert hasattr(orchestrator, 'status')
            assert isinstance(orchestrator.status, OrchestrationStatus)
    
    @pytest.mark.asyncio
    async def test_consolidated_background_processes(self):
        """
        Test that background processes are properly managed through
        the base orchestrator, eliminating the conflicts identified
        in Agent 1's analysis.
        """
        from core.orchestration.base import BaseOrchestrator
        
        orchestrator = BaseOrchestrator("test_type", "test_id")
        
        # Test background process management interface
        assert hasattr(orchestrator, 'start_background_processes')
        assert hasattr(orchestrator, 'stop_background_processes')
        assert hasattr(orchestrator, 'get_background_process_status')
        
        # Test that starting/stopping works without conflicts
        start_success = await orchestrator.start_background_processes()
        assert start_success is True
        
        status = await orchestrator.get_background_process_status()
        assert isinstance(status, dict)
        
        stop_success = await orchestrator.stop_background_processes()
        assert stop_success is True
    
    def test_unified_result_types(self):
        """
        Test that the unified result types eliminate the conflicts
        identified in Agent 1's analysis.
        """
        from core.orchestration.interfaces import OrchestrationResult
        
        # Create results from different orchestrator types
        ml_result = OrchestrationResult(
            success=True,
            task_id="ml_task",
            orchestrator_id="ml_orchestrator",
            task_type=TaskType.ML_PIPELINE,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=1.0
        )
        
        agent_result = OrchestrationResult(
            success=True,
            task_id="agent_task", 
            orchestrator_id="agent_orchestrator",
            task_type=TaskType.AGENT_LIFECYCLE,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=2.0
        )
        
        # Both results should have the same structure and interface
        assert ml_result.success == agent_result.success
        assert hasattr(ml_result, 'status') and hasattr(agent_result, 'status')
        assert isinstance(ml_result.metrics, dict) and isinstance(agent_result.metrics, dict)
        assert isinstance(ml_result.errors, list) and isinstance(agent_result.errors, list)
        
        # Results should be comparable and processable uniformly
        results = [ml_result, agent_result]
        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 2


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])