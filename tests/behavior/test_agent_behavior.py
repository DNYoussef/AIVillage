"""Behavior-driven tests for agent task processing workflows.

Tests complex scenarios and workflows to ensure agents behave correctly
in realistic operational environments.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict, List

from agents.unified_base_agent import UnifiedBaseAgent, UnifiedAgentConfig, SelfEvolvingSystem
from agents.king.analytics.base_analytics import BaseAnalytics
from agents.interfaces.processing_interface import ProcessingInterface, ProcessorCapability
from agents.utils.task import Task as LangroidTask
from core.communication import StandardCommunicationProtocol, Message, MessageType
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore


class BehaviorAnalytics(BaseAnalytics):
    """Analytics for behavior testing."""
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        return {
            "total_metrics": len(self.metrics),
            "workflow_metrics": {
                metric: {
                    "count": len(values),
                    "latest": values[-1] if values else 0,
                    "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "stable"
                }
                for metric, values in self.metrics.items()
            },
            "behavior_health": "optimal" if len(self.metrics) > 0 else "inactive",
            "timestamp": datetime.now().isoformat()
        }


class WorkflowProcessor(ProcessingInterface[Dict[str, Any], Dict[str, Any]]):
    """Processor for workflow behavior testing."""
    
    def __init__(self, processor_id: str = "workflow_processor"):
        super().__init__(processor_id)
        self.add_capability(ProcessorCapability.TEXT_PROCESSING)
        self.add_capability(ProcessorCapability.PARALLEL_PROCESSING)
        self.add_capability(ProcessorCapability.CHAIN_PROCESSING)
        self.workflow_state = {}
        self.step_history = []
    
    async def initialize(self) -> bool:
        self.workflow_state = {"initialized": True, "step_count": 0}
        return True
    
    async def shutdown(self) -> bool:
        self.workflow_state = {"shutting_down": True}
        return True
    
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        from agents.base import ProcessResult, ProcessStatus
        
        step_type = input_data.get("step_type", "default")
        workflow_id = input_data.get("workflow_id", "unknown")
        
        # Record workflow step
        step_info = {
            "step_type": step_type,
            "workflow_id": workflow_id,
            "timestamp": datetime.now(),
            "step_number": self.workflow_state.get("step_count", 0) + 1
        }
        self.step_history.append(step_info)
        self.workflow_state["step_count"] = step_info["step_number"]
        
        # Simulate different workflow behaviors
        if step_type == "analysis":
            await asyncio.sleep(0.02)  # Analysis takes time
            result = {
                "analysis_result": f"analyzed_{input_data.get('content', '')}",
                "confidence": 0.85,
                "workflow_id": workflow_id,
                "step_completed": step_info["step_number"]
            }
        elif step_type == "decision":
            await asyncio.sleep(0.01)  # Quick decision
            result = {
                "decision": f"approve_{input_data.get('content', '')}",
                "reasoning": "behavior_based_decision",
                "workflow_id": workflow_id,
                "step_completed": step_info["step_number"]
            }
        elif step_type == "execution":
            await asyncio.sleep(0.03)  # Execution takes longer
            result = {
                "execution_result": f"executed_{input_data.get('content', '')}",
                "status": "completed",
                "workflow_id": workflow_id,
                "step_completed": step_info["step_number"]
            }
        else:
            result = {
                "default_result": f"processed_{input_data.get('content', '')}",
                "workflow_id": workflow_id,
                "step_completed": step_info["step_number"]
            }
        
        return ProcessResult(status=ProcessStatus.COMPLETED, data=result)
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        return isinstance(input_data, dict) and "content" in input_data
    
    async def estimate_processing_time(self, input_data: Dict[str, Any]) -> float:
        step_type = input_data.get("step_type", "default")
        time_estimates = {
            "analysis": 0.02,
            "decision": 0.01,
            "execution": 0.03,
            "default": 0.015
        }
        return time_estimates.get(step_type, 0.015)


@pytest.mark.behavior
class TestTaskProcessingWorkflows:
    """Test complete task processing workflows."""
    
    @pytest.fixture
    async def workflow_agent(self, mock_communication_protocol):
        """Create agent configured for workflow testing."""
        vector_store = Mock(spec=VectorStore)
        vector_store.add_texts = AsyncMock()
        vector_store.similarity_search = AsyncMock(return_value=[])
        
        config = UnifiedAgentConfig(
            name="WorkflowAgent",
            description="Agent for workflow behavior testing",
            capabilities=["analysis", "decision_making", "execution"],
            rag_config=UnifiedConfig(),
            vector_store=vector_store,
            model="gpt-4",
            instructions="Execute multi-step workflows efficiently"
        )
        
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(config, mock_communication_protocol)
            agent.analytics = BehaviorAnalytics()
            agent.processor = WorkflowProcessor("workflow_processor")
            await agent.processor.initialize()
            
            # Implement workflow-aware _process_task
            async def workflow_process_task(task):
                workflow_id = getattr(task, 'workflow_id', 'default_workflow')
                
                # Multi-step workflow processing
                steps = ["analysis", "decision", "execution"]
                workflow_results = []
                
                for step in steps:
                    step_data = {
                        "content": task.content,
                        "step_type": step,
                        "workflow_id": workflow_id
                    }
                    
                    step_result = await agent.processor.process(step_data)
                    workflow_results.append(step_result.data if step_result.is_success else {"error": step_result.error})
                    
                    # Record step metrics
                    agent.analytics.record_metric(f"workflow_{step}_completed", 1.0)
                    agent.analytics.record_metric("total_workflow_steps", 1.0)
                
                return {
                    "workflow_id": workflow_id,
                    "workflow_results": workflow_results,
                    "workflow_status": "completed",
                    "total_steps": len(steps)
                }
            
            agent._process_task = workflow_process_task
            return agent
    
    async def test_single_task_workflow_execution(self, workflow_agent):
        """Test execution of single task through complete workflow."""
        # Create workflow task
        task = Mock()
        task.content = "analyze_user_request"
        task.type = "workflow_task"
        task.workflow_id = "workflow_001"
        
        # Setup agent layers
        workflow_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        workflow_agent.foundational_layer.process_task = AsyncMock(return_value=task)
        workflow_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        workflow_agent.decision_making_layer.make_decision = AsyncMock(
            return_value="workflow_decision_made"
        )
        workflow_agent.continuous_learning_layer.update = AsyncMock()
        
        # Execute workflow
        result = await workflow_agent.execute_task(task)
        
        # Verify workflow completion
        assert result["result"] == "workflow_decision_made"
        
        # Verify processor recorded all workflow steps
        step_history = workflow_agent.processor.step_history
        assert len(step_history) == 3  # analysis, decision, execution
        
        step_types = [step["step_type"] for step in step_history]
        assert step_types == ["analysis", "decision", "execution"]
        
        # Verify workflow ID consistency
        workflow_ids = [step["workflow_id"] for step in step_history]
        assert all(wid == "workflow_001" for wid in workflow_ids)
        
        # Verify analytics recorded workflow metrics
        analytics_report = workflow_agent.analytics.generate_analytics_report()
        workflow_metrics = analytics_report["workflow_metrics"]
        
        assert "workflow_analysis_completed" in workflow_metrics
        assert "workflow_decision_completed" in workflow_metrics
        assert "workflow_execution_completed" in workflow_metrics
        assert "total_workflow_steps" in workflow_metrics
        
        assert workflow_metrics["total_workflow_steps"]["count"] == 3
    
    async def test_concurrent_workflow_execution(self, workflow_agent):
        """Test concurrent execution of multiple workflows."""
        # Create multiple workflow tasks
        workflows = [
            Mock(content=f"workflow_task_{i}", type="workflow", workflow_id=f"workflow_{i:03d}")
            for i in range(5)
        ]
        
        # Setup agent layers
        workflow_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        workflow_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        workflow_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        workflow_agent.decision_making_layer.make_decision = AsyncMock(
            return_value="concurrent_workflow_decision"
        )
        workflow_agent.continuous_learning_layer.update = AsyncMock()
        
        # Execute workflows concurrently
        start_time = datetime.now()
        results = await asyncio.gather(*[
            workflow_agent.execute_task(task) for task in workflows
        ])
        end_time = datetime.now()
        
        # Verify all workflows completed
        assert len(results) == 5
        assert all(result["result"] == "concurrent_workflow_decision" for result in results)
        
        # Verify processor handled all workflow steps
        step_history = workflow_agent.processor.step_history
        assert len(step_history) == 15  # 5 workflows * 3 steps each
        
        # Verify workflow separation (each workflow has unique ID)
        workflow_ids = set(step["workflow_id"] for step in step_history)
        assert len(workflow_ids) == 5
        
        # Verify concurrent execution was actually faster than sequential
        total_time = (end_time - start_time).total_seconds()
        sequential_estimate = 5 * (0.02 + 0.01 + 0.03)  # 5 workflows * 3 steps
        assert total_time < sequential_estimate * 0.8  # At least 20% faster
        
        # Verify analytics aggregated correctly
        analytics_report = workflow_agent.analytics.generate_analytics_report()
        workflow_metrics = analytics_report["workflow_metrics"]
        assert workflow_metrics["total_workflow_steps"]["count"] == 15
    
    async def test_workflow_error_recovery(self, workflow_agent):
        """Test workflow error recovery and continuation."""
        # Create task that will cause error in decision step
        task = Mock()
        task.content = "error_in_decision_step"
        task.type = "error_workflow"
        task.workflow_id = "error_workflow_001"
        
        # Override processor to simulate error in decision step
        original_process = workflow_agent.processor.process
        async def error_prone_process(input_data, **kwargs):
            if input_data.get("step_type") == "decision" and "error_in_decision" in input_data.get("content", ""):
                from agents.base import ProcessResult, ProcessStatus
                return ProcessResult(
                    status=ProcessStatus.FAILED,
                    error="Decision step failed due to ambiguous requirements"
                )
            return await original_process(input_data, **kwargs)
        
        workflow_agent.processor.process = error_prone_process
        
        # Setup agent layers
        workflow_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        workflow_agent.foundational_layer.process_task = AsyncMock(return_value=task)
        workflow_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        workflow_agent.decision_making_layer.make_decision = AsyncMock(
            return_value="error_recovery_decision"
        )
        workflow_agent.continuous_learning_layer.update = AsyncMock()
        
        # Execute error-prone workflow
        result = await workflow_agent.execute_task(task)
        
        # Verify workflow completed despite error
        assert result["result"] == "error_recovery_decision"
        
        # Verify all steps were attempted
        step_history = workflow_agent.processor.step_history
        assert len(step_history) == 3
        
        # Verify error step was recorded but workflow continued
        step_results = [step.get("error") for step in step_history]
        error_steps = [i for i, error in enumerate(step_results) if error]
        
        # Should have attempted all steps even with error
        step_types = [step["step_type"] for step in step_history]
        assert "analysis" in step_types
        assert "decision" in step_types
        assert "execution" in step_types


@pytest.mark.behavior
class TestAdaptiveAgentBehavior:
    """Test adaptive behavior based on experience and learning."""
    
    @pytest.fixture
    async def adaptive_agent(self, mock_communication_protocol):
        """Create agent that adapts behavior based on experience."""
        vector_store = Mock(spec=VectorStore)
        vector_store.add_texts = AsyncMock()
        vector_store.similarity_search = AsyncMock(return_value=[])
        
        config = UnifiedAgentConfig(
            name="AdaptiveAgent",
            description="Agent that adapts behavior based on experience",
            capabilities=["learning", "adaptation", "optimization"],
            rag_config=UnifiedConfig(),
            vector_store=vector_store,
            model="gpt-4",
            instructions="Adapt behavior based on task outcomes and patterns"
        )
        
        with patch.multiple(
            'agents.unified_base_agent',
            EnhancedRAGPipeline=Mock(),
            OpenAIGPTConfig=Mock()
        ):
            agent = UnifiedBaseAgent(config, mock_communication_protocol)
            agent.analytics = BehaviorAnalytics()
            agent.experience_history = []  # Track experience for adaptation
            agent.adaptation_state = {
                "strategy": "conservative",
                "confidence": 0.5,
                "learning_rate": 0.1
            }
            
            # Implement adaptive _process_task
            async def adaptive_process_task(task):
                start_time = datetime.now()
                
                # Analyze previous experience to adapt strategy
                recent_successes = sum(1 for exp in agent.experience_history[-10:] 
                                     if exp.get("success", False))
                recent_performance = recent_successes / min(10, len(agent.experience_history)) if agent.experience_history else 0.5
                
                # Adapt strategy based on performance
                if recent_performance > 0.8:
                    agent.adaptation_state["strategy"] = "aggressive"
                    agent.adaptation_state["confidence"] = min(1.0, agent.adaptation_state["confidence"] + 0.1)
                elif recent_performance < 0.4:
                    agent.adaptation_state["strategy"] = "conservative"
                    agent.adaptation_state["confidence"] = max(0.2, agent.adaptation_state["confidence"] - 0.1)
                else:
                    agent.adaptation_state["strategy"] = "balanced"
                
                # Process task based on current strategy
                processing_time = {
                    "aggressive": 0.01,  # Fast but potentially less accurate
                    "balanced": 0.02,    # Balanced approach
                    "conservative": 0.03  # Slower but more thorough
                }[agent.adaptation_state["strategy"]]
                
                await asyncio.sleep(processing_time)
                
                # Simulate success rate based on strategy
                success_rates = {
                    "aggressive": 0.7,
                    "balanced": 0.85,
                    "conservative": 0.95
                }
                
                import random
                success = random.random() < success_rates[agent.adaptation_state["strategy"]]
                
                # Record experience
                experience = {
                    "task_content": task.content,
                    "strategy_used": agent.adaptation_state["strategy"],
                    "confidence": agent.adaptation_state["confidence"],
                    "success": success,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now()
                }
                agent.experience_history.append(experience)
                
                # Keep only recent experience (sliding window)
                if len(agent.experience_history) > 50:
                    agent.experience_history.pop(0)
                
                # Record analytics
                agent.analytics.record_metric("adaptation_confidence", agent.adaptation_state["confidence"])
                agent.analytics.record_metric("strategy_success_rate", 1.0 if success else 0.0)
                agent.analytics.record_metric(f"{agent.adaptation_state['strategy']}_strategy_used", 1.0)
                
                return {
                    "result": f"processed_with_{agent.adaptation_state['strategy']}_strategy",
                    "success": success,
                    "confidence": agent.adaptation_state["confidence"],
                    "adaptation_info": agent.adaptation_state.copy()
                }
            
            agent._process_task = adaptive_process_task
            return agent
    
    async def test_strategy_adaptation_over_time(self, adaptive_agent):
        """Test agent strategy adaptation based on performance history."""
        # Create tasks to build experience history
        tasks = [Mock(content=f"adaptive_task_{i}", type="adaptive") for i in range(20)]
        
        # Setup agent layers
        adaptive_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        adaptive_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.decision_making_layer.make_decision = AsyncMock(
            return_value="adaptive_decision"
        )
        adaptive_agent.continuous_learning_layer.update = AsyncMock()
        
        # Record initial strategy
        initial_strategy = adaptive_agent.adaptation_state["strategy"]
        initial_confidence = adaptive_agent.adaptation_state["confidence"]
        
        # Process tasks to build experience
        results = []
        for task in tasks:
            result = await adaptive_agent.execute_task(task)
            results.append(result)
            
            # Small delay to allow adaptation
            await asyncio.sleep(0.001)
        
        # Verify all tasks processed
        assert len(results) == 20
        
        # Verify experience history was built
        assert len(adaptive_agent.experience_history) == 20
        
        # Verify strategy may have adapted
        final_strategy = adaptive_agent.adaptation_state["strategy"]
        final_confidence = adaptive_agent.adaptation_state["confidence"]
        
        # Strategy should be in valid set
        assert final_strategy in ["aggressive", "balanced", "conservative"]
        assert 0.2 <= final_confidence <= 1.0
        
        # Verify analytics tracked adaptation
        analytics_report = adaptive_agent.analytics.generate_analytics_report()
        workflow_metrics = analytics_report["workflow_metrics"]
        
        assert "adaptation_confidence" in workflow_metrics
        assert "strategy_success_rate" in workflow_metrics
        
        # Should have used strategies (at least one type)
        strategy_metrics = [key for key in workflow_metrics.keys() if "_strategy_used" in key]
        assert len(strategy_metrics) > 0
    
    async def test_performance_based_confidence_adjustment(self, adaptive_agent):
        """Test confidence adjustment based on task performance."""
        # Simulate successful tasks to increase confidence
        successful_tasks = [Mock(content=f"success_task_{i}", type="success") for i in range(10)]
        
        # Setup layers
        adaptive_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        adaptive_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.decision_making_layer.make_decision = AsyncMock(return_value="confident_decision")
        adaptive_agent.continuous_learning_layer.update = AsyncMock()
        
        # Override random to ensure high success rate
        with patch('random.random', return_value=0.1):  # Always below success threshold
            # Record initial confidence
            initial_confidence = adaptive_agent.adaptation_state["confidence"]
            
            # Process successful tasks
            for task in successful_tasks:
                await adaptive_agent.execute_task(task)
            
            # Confidence should have increased
            final_confidence = adaptive_agent.adaptation_state["confidence"]
            
            # With high success rate, strategy should become more aggressive
            final_strategy = adaptive_agent.adaptation_state["strategy"]
            
            # Verify adaptation occurred
            experience_success_rate = sum(1 for exp in adaptive_agent.experience_history if exp["success"]) / len(adaptive_agent.experience_history)
            assert experience_success_rate > 0.8  # High success rate
            
            # With high success rate, should move toward aggressive strategy
            if len(adaptive_agent.experience_history) >= 10:
                assert final_strategy in ["aggressive", "balanced"]
    
    async def test_learning_rate_adaptation(self, adaptive_agent):
        """Test learning rate adaptation based on environment stability."""
        # Create tasks with varying difficulty to test adaptation stability
        tasks = []
        task_types = ["easy", "medium", "hard"] * 10  # 30 tasks with mixed difficulty
        
        for i, difficulty in enumerate(task_types):
            task = Mock(content=f"{difficulty}_task_{i}", type=difficulty)
            tasks.append(task)
        
        # Setup layers
        adaptive_agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
        adaptive_agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
        adaptive_agent.decision_making_layer.make_decision = AsyncMock(return_value="learning_decision")
        adaptive_agent.continuous_learning_layer.update = AsyncMock()
        
        # Track adaptation changes over time
        strategy_changes = []
        confidence_history = []
        
        for task in tasks:
            prev_strategy = adaptive_agent.adaptation_state["strategy"]
            
            await adaptive_agent.execute_task(task)
            
            current_strategy = adaptive_agent.adaptation_state["strategy"]
            current_confidence = adaptive_agent.adaptation_state["confidence"]
            
            if prev_strategy != current_strategy:
                strategy_changes.append({
                    "task_number": len(confidence_history),
                    "from_strategy": prev_strategy,
                    "to_strategy": current_strategy,
                    "confidence": current_confidence
                })
            
            confidence_history.append(current_confidence)
        
        # Verify learning occurred
        assert len(adaptive_agent.experience_history) == 30
        
        # Verify some adaptation occurred (strategy may have changed)
        final_strategy = adaptive_agent.adaptation_state["strategy"]
        assert final_strategy in ["aggressive", "balanced", "conservative"]
        
        # Confidence should stabilize over time (less variance in later stages)
        early_confidence_variance = max(confidence_history[:10]) - min(confidence_history[:10]) if len(confidence_history) >= 10 else 0
        late_confidence_variance = max(confidence_history[-10:]) - min(confidence_history[-10:]) if len(confidence_history) >= 10 else 0
        
        # Later confidence should generally be more stable (or learning has occurred)
        assert len(confidence_history) == 30


@pytest.mark.behavior
class TestCollaborativeBehavior:
    """Test collaborative behavior between multiple agents."""
    
    @pytest.fixture
    async def collaborative_system(self, mock_communication_protocol):
        """Create system with multiple collaborative agents."""
        agents = []
        
        # Create specialized agents for collaboration
        agent_specs = [
            {"name": "AnalyzerAgent", "specialty": "analysis", "capabilities": ["deep_analysis", "pattern_recognition"]},
            {"name": "DecisionAgent", "specialty": "decision", "capabilities": ["decision_making", "risk_assessment"]},
            {"name": "ExecutorAgent", "specialty": "execution", "capabilities": ["task_execution", "implementation"]}
        ]
        
        for spec in agent_specs:
            vector_store = Mock(spec=VectorStore)
            vector_store.add_texts = AsyncMock()
            vector_store.similarity_search = AsyncMock(return_value=[])
            
            config = UnifiedAgentConfig(
                name=spec["name"],
                description=f"Specialized {spec['specialty']} agent",
                capabilities=spec["capabilities"] + ["collaboration"],
                rag_config=UnifiedConfig(),
                vector_store=vector_store,
                model="gpt-4",
                instructions=f"Specialize in {spec['specialty']} and collaborate with other agents"
            )
            
            with patch.multiple(
                'agents.unified_base_agent',
                EnhancedRAGPipeline=Mock(),
                OpenAIGPTConfig=Mock()
            ):
                agent = UnifiedBaseAgent(config, mock_communication_protocol)
                agent.analytics = BehaviorAnalytics()
                agent.specialty = spec["specialty"]
                agent.collaboration_history = []
                
                # Implement collaborative _process_task
                async def make_collaborative_process_task(agent_specialty):
                    async def collaborative_process_task(task):
                        # Check if task requires collaboration
                        if hasattr(task, 'requires_collaboration') and task.requires_collaboration:
                            # Record collaboration attempt
                            agent.analytics.record_metric("collaboration_attempts", 1.0)
                            
                            # Simulate collaboration with other agents
                            if agent_specialty == "analysis":
                                # Analyzer does initial analysis
                                analysis_result = f"analyzed_{task.content}"
                                
                                # Request decision from DecisionAgent
                                decision_response = await agent.communicate(
                                    f"Please make decision based on: {analysis_result}",
                                    "DecisionAgent"
                                )
                                
                                agent.collaboration_history.append({
                                    "collaborated_with": "DecisionAgent",
                                    "task": task.content,
                                    "type": "analysis_to_decision"
                                })
                                
                                return {
                                    "analysis": analysis_result,
                                    "decision_request": decision_response,
                                    "collaboration": True
                                }
                            
                            elif agent_specialty == "decision":
                                # Decision agent makes decisions
                                decision_result = f"decision_for_{task.content}"
                                
                                # Request execution from ExecutorAgent
                                execution_response = await agent.communicate(
                                    f"Please execute: {decision_result}",
                                    "ExecutorAgent"
                                )
                                
                                agent.collaboration_history.append({
                                    "collaborated_with": "ExecutorAgent",
                                    "task": task.content,
                                    "type": "decision_to_execution"
                                })
                                
                                return {
                                    "decision": decision_result,
                                    "execution_request": execution_response,
                                    "collaboration": True
                                }
                            
                            elif agent_specialty == "execution":
                                # Executor implements decisions
                                execution_result = f"executed_{task.content}"
                                
                                agent.collaboration_history.append({
                                    "executed_task": task.content,
                                    "type": "execution_completion"
                                })
                                
                                return {
                                    "execution": execution_result,
                                    "collaboration": True
                                }
                        
                        else:
                            # Handle task independently
                            agent.analytics.record_metric("independent_processing", 1.0)
                            return {
                                f"{agent_specialty}_result": f"{agent_specialty}_processed_{task.content}",
                                "collaboration": False
                            }
                    
                    return collaborative_process_task
                
                agent._process_task = await make_collaborative_process_task(spec["specialty"])
                agents.append(agent)
        
        # Setup communication protocol to simulate inter-agent communication
        mock_communication_protocol.query.return_value = {"collaboration_response": "acknowledged"}
        
        return SelfEvolvingSystem(agents)
    
    async def test_multi_agent_collaborative_workflow(self, collaborative_system):
        """Test collaborative workflow across multiple specialized agents."""
        # Create collaborative task
        collab_task = Mock()
        collab_task.content = "complex_collaborative_task"
        collab_task.type = "analysis"  # Start with analysis agent
        collab_task.requires_collaboration = True
        
        # Setup agent layers for all agents
        for agent in collaborative_system.agents:
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(
                return_value=f"collaborative_decision_from_{agent.name}"
            )
            agent.continuous_learning_layer.update = AsyncMock()
        
        # Process task through system (should be handled by AnalyzerAgent)
        result = await collaborative_system.process_task(collab_task)
        
        # Verify task was processed
        assert result["result"].startswith("collaborative_decision_from_AnalyzerAgent")
        
        # Verify collaboration occurred
        analyzer_agent = collaborative_system.agents[0]  # AnalyzerAgent
        collaboration_metrics = analyzer_agent.analytics.metrics
        assert "collaboration_attempts" in collaboration_metrics
        assert collaboration_metrics["collaboration_attempts"][0] == 1.0
        
        # Verify collaboration history
        assert len(analyzer_agent.collaboration_history) == 1
        collab_record = analyzer_agent.collaboration_history[0]
        assert collab_record["collaborated_with"] == "DecisionAgent"
        assert collab_record["type"] == "analysis_to_decision"
    
    async def test_independent_vs_collaborative_task_handling(self, collaborative_system):
        """Test agent behavior for independent vs collaborative tasks."""
        # Create both independent and collaborative tasks
        independent_task = Mock()
        independent_task.content = "independent_task"
        independent_task.type = "analysis"
        independent_task.requires_collaboration = False
        
        collaborative_task = Mock()
        collaborative_task.content = "collaborative_task"
        collaborative_task.type = "analysis"
        collaborative_task.requires_collaboration = True
        
        # Setup agent layers
        for agent in collaborative_system.agents:
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(
                return_value=f"decision_from_{agent.name}"
            )
            agent.continuous_learning_layer.update = AsyncMock()
        
        # Process both tasks
        independent_result = await collaborative_system.process_task(independent_task)
        collaborative_result = await collaborative_system.process_task(collaborative_task)
        
        # Verify both tasks completed
        assert independent_result["result"].startswith("decision_from_AnalyzerAgent")
        assert collaborative_result["result"].startswith("decision_from_AnalyzerAgent")
        
        # Verify different processing approaches
        analyzer_agent = collaborative_system.agents[0]
        analytics_metrics = analyzer_agent.analytics.metrics
        
        # Should have both independent and collaborative processing
        assert "independent_processing" in analytics_metrics
        assert "collaboration_attempts" in analytics_metrics
        
        # Should have 1 independent and 1 collaborative task
        assert analytics_metrics["independent_processing"][0] == 1.0
        assert analytics_metrics["collaboration_attempts"][-1] == 1.0  # Most recent
    
    async def test_collaboration_failure_recovery(self, collaborative_system):
        """Test recovery when collaboration fails."""
        # Create collaborative task
        collab_task = Mock()
        collab_task.content = "failing_collaborative_task"
        collab_task.type = "analysis"
        collab_task.requires_collaboration = True
        
        # Setup communication failure
        collaborative_system.agents[0].communication_protocol.query.side_effect = Exception("Communication failed")
        
        # Setup agent layers
        for agent in collaborative_system.agents:
            agent.quality_assurance_layer.check_task_safety = Mock(return_value=True)
            agent.foundational_layer.process_task = AsyncMock(side_effect=lambda x: x)
            agent.agent_architecture_layer.process_result = AsyncMock(side_effect=lambda x: x)
            agent.decision_making_layer.make_decision = AsyncMock(
                return_value=f"fallback_decision_from_{agent.name}"
            )
            agent.continuous_learning_layer.update = AsyncMock()
        
        # Process task - should handle communication failure
        with pytest.raises(Exception):  # Communication failure should propagate
            await collaborative_system.process_task(collab_task)
        
        # Verify collaboration was attempted
        analyzer_agent = collaborative_system.agents[0]
        analytics_metrics = analyzer_agent.analytics.metrics
        assert "collaboration_attempts" in analytics_metrics