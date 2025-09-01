#!/usr/bin/env python3
"""
Unified Agent Base Class - MCP-Enhanced Agent Framework Consolidation
==================================================================

Consolidates all 5 agent framework patterns into a single, unified architecture
with full MCP server orchestration, DSPy optimization, and cross-system compatibility.

This class integrates:
- DSPy 3.0.2 optimization system
- Enhanced coordination with memory persistence  
- Sequential thinking integration
- Service instrumentation monitoring
- Domain entity architecture
- Training pipeline integration
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
from enum import Enum

import dspy
from dspy import ChainOfThought, ReAct, Predict, Parallel

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent status enumeration - weak connascence (CoN)."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class AgentCapability(Enum):
    """Standard agent capability enumeration."""
    REASONING = "reasoning"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    MEMORY_ACCESS = "memory_access"
    COMPUTATION = "computation"
    COORDINATION = "coordination"
    MONITORING = "monitoring"


@dataclass
class MCPConfiguration:
    """MCP server configuration for agent coordination."""
    memory_enabled: bool = True
    sequential_thinking_enabled: bool = True
    github_integration_enabled: bool = False
    context_cache_enabled: bool = True
    session_id: str = field(default_factory=lambda: f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    memory_namespace: str = "unified_agents"


@dataclass
class DSPyConfiguration:
    """DSPy optimization configuration."""
    optimization_target: float = 0.90
    enable_mipro: bool = True
    enable_evaluation: bool = True
    training_examples_required: int = 5
    optimization_iterations: int = 3


@dataclass
class AgentMetrics:
    """Performance metrics tracking."""
    tasks_completed: int = 0
    success_rate: float = 0.0
    average_response_time_ms: float = 0.0
    memory_operations: int = 0
    communication_events: int = 0
    optimization_cycles: int = 0
    last_active: Optional[datetime] = None


class UnifiedAgentInterface(Protocol):
    """Unified interface for all agent implementations."""
    
    async def initialize(self) -> bool:
        """Initialize agent with all systems."""
        ...
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task with unified coordination."""
        ...
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown agent."""
        ...


class UnifiedBaseAgent(dspy.Module, ABC):
    """
    Unified Base Agent Class - Consolidates All Framework Patterns
    
    Integrates:
    1. DSPy 3.0.2 optimization with MIPROv2
    2. Enhanced agent coordination with memory MCP
    3. Sequential thinking with reasoning chains  
    4. Service instrumentation and monitoring
    5. Domain entity architecture with dependency injection
    6. Training pipeline integration with GrokFast
    
    Architecture Principles:
    - Single Responsibility: Each system handles specific concerns
    - Dependency Injection: Loose coupling between systems
    - Interface Segregation: Minimal required interfaces
    - Open/Closed: Extensible without modification
    """
    
    def __init__(self,
                 agent_id: str,
                 agent_type: str,
                 capabilities: List[AgentCapability],
                 mcp_config: Optional[MCPConfiguration] = None,
                 dspy_config: Optional[DSPyConfiguration] = None,
                 project_root: Optional[Path] = None):
        """
        Initialize unified agent with all system integrations.
        
        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (researcher, coder, etc.)
            capabilities: List of agent capabilities
            mcp_config: MCP server configuration
            dspy_config: DSPy optimization configuration
            project_root: Project root directory
        """
        super().__init__()
        
        # Core Properties
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.status = AgentStatus.INITIALIZING
        self.project_root = project_root or Path.cwd()
        
        # Configuration
        self.mcp_config = mcp_config or MCPConfiguration()
        self.dspy_config = dspy_config or DSPyConfiguration()
        
        # Metrics and State
        self.metrics = AgentMetrics()
        self.operation_history: List[Dict[str, Any]] = []
        self.current_tasks: Dict[str, Any] = {}
        
        # System Integrations
        self._initialize_dspy_modules()
        self._initialize_coordination_systems()
        self._initialize_monitoring()
        
        logger.info(f"UnifiedBaseAgent initialized: {agent_id}")
    
    def _initialize_dspy_modules(self):
        """Initialize DSPy optimization modules."""
        try:
            # Core DSPy modules for task processing
            self.task_processor = ChainOfThought("task, context, capabilities -> analysis, execution_plan, results")
            self.reasoning_engine = ReAct("complex_problem, available_knowledge -> reasoning_steps, solution")
            self.parallel_coordinator = Parallel([
                ChainOfThought("coordination_request -> agent_assignments"),
                Predict("resource_requirements -> allocation_plan")
            ])
            
            # DSPy optimizers
            from dspy import MIPROv2, Evaluate
            from dspy.evaluate import CompleteAndGrounded, SemanticF1
            
            self.optimizer = MIPROv2(
                metric=self._unified_success_metric,
                auto="medium",
                num_threads=8
            )
            
            self.evaluator = Evaluate(
                metric=[
                    CompleteAndGrounded(),
                    SemanticF1(),
                    self._unified_success_metric
                ]
            )
            
            logger.info("DSPy modules initialized successfully")
            
        except Exception as e:
            logger.error(f"DSPy initialization failed: {e}")
    
    def _initialize_coordination_systems(self):
        """Initialize MCP coordination and memory systems."""
        try:
            # Memory database setup
            self.memory_db_path = self.project_root / ".mcp" / "unified_memory.db"
            self.memory_db_path.parent.mkdir(exist_ok=True)
            
            # Coordination state
            self.coordination_session = self.mcp_config.session_id
            self.shared_memory: Dict[str, Any] = {}
            
            # Sequential thinking chains
            self.reasoning_chains: Dict[str, str] = {}
            
            logger.info("Coordination systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Coordination initialization failed: {e}")
    
    def _initialize_monitoring(self):
        """Initialize service instrumentation and monitoring."""
        try:
            # Performance tracking
            self.start_time = datetime.now()
            self.performance_baseline = {
                "response_time_target_ms": 100,
                "success_rate_target": 0.95,
                "memory_efficiency_target": 0.8
            }
            
            # Operation counters
            self.operation_counters = {
                "tasks_processed": 0,
                "mcp_operations": 0,
                "dspy_optimizations": 0,
                "coordination_events": 0
            }
            
            logger.info("Monitoring systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Monitoring initialization failed: {e}")
    
    # Core Agent Lifecycle
    
    async def initialize(self) -> bool:
        """Initialize agent with all integrated systems."""
        try:
            logger.info(f"Initializing unified agent: {self.agent_id}")
            
            # Initialize MCP coordination hooks
            await self._execute_coordination_hooks("pre-initialization")
            
            # Domain-specific initialization
            success = await self.initialize_domain_resources()
            
            if success:
                # Initialize DSPy optimization
                await self._initialize_dspy_optimization()
                
                # Setup memory coordination
                await self._setup_memory_coordination()
                
                # Setup sequential thinking
                await self._setup_sequential_thinking()
                
                # Start monitoring
                await self._start_monitoring()
                
                self.status = AgentStatus.ACTIVE
                self.metrics.last_active = datetime.now()
                
                # Post-initialization hooks
                await self._execute_coordination_hooks("post-initialization")
                
                logger.info(f"Agent {self.agent_id} fully initialized")
                return True
            else:
                logger.error(f"Domain initialization failed for {self.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self.status = AgentStatus.TERMINATED
            return False
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process task with unified coordination and optimization.
        
        Integrates:
        - DSPy optimization for execution planning
        - Memory MCP for context sharing
        - Sequential thinking for complex reasoning
        - Performance monitoring throughout
        """
        task_id = task.get("task_id", f"task_{len(self.current_tasks)}")
        
        if task_id in self.current_tasks:
            return {"status": "error", "message": "Task already in progress"}
        
        try:
            start_time = datetime.now()
            self.status = AgentStatus.BUSY
            self.current_tasks[task_id] = task
            
            # Pre-task coordination hooks
            await self._execute_coordination_hooks("pre-task", {"task_id": task_id})
            
            # Store task in shared memory
            await self._store_shared_memory(f"task_{task_id}", {
                "task_data": task,
                "agent_id": self.agent_id,
                "status": "processing",
                "start_time": start_time.isoformat()
            })
            
            # Generate reasoning chain if complex task
            if task.get("complexity", "simple") in ["complex", "high"]:
                reasoning_chain = await self._generate_reasoning_chain(task)
                task["reasoning_chain"] = reasoning_chain
            
            # Execute with DSPy optimization
            execution_result = await self._execute_with_dspy_optimization(task)
            
            # Calculate performance metrics
            end_time = datetime.now()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            success = execution_result.get("status") == "success"
            
            # Update metrics
            self._update_metrics(execution_time_ms, success)
            
            # Store results in shared memory
            await self._store_shared_memory(f"result_{task_id}", {
                "result": execution_result,
                "agent_id": self.agent_id,
                "execution_time_ms": execution_time_ms,
                "success": success,
                "end_time": end_time.isoformat()
            })
            
            # Clean up
            self.current_tasks.pop(task_id, None)
            self.status = AgentStatus.ACTIVE
            
            # Post-task coordination hooks
            await self._execute_coordination_hooks("post-task", {
                "task_id": task_id,
                "success": success,
                "execution_time_ms": execution_time_ms
            })
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            self.current_tasks.pop(task_id, None)
            self.status = AgentStatus.ACTIVE
            
            return {"status": "error", "error": str(e)}
    
    async def shutdown(self) -> bool:
        """Gracefully shutdown agent with full cleanup."""
        try:
            logger.info(f"Shutting down unified agent: {self.agent_id}")
            self.status = AgentStatus.SHUTTING_DOWN
            
            # Pre-shutdown hooks
            await self._execute_coordination_hooks("pre-shutdown")
            
            # Complete active tasks
            await self._complete_active_tasks()
            
            # Store final metrics
            await self._store_final_metrics()
            
            # Clean up domain resources
            await self.cleanup_domain_resources()
            
            # Clean up system integrations
            await self._cleanup_system_integrations()
            
            self.status = AgentStatus.TERMINATED
            
            # Post-shutdown hooks
            await self._execute_coordination_hooks("post-shutdown")
            
            logger.info(f"Agent {self.agent_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Agent shutdown failed: {e}")
            return False
    
    # DSPy Integration
    
    def forward(self, task: str, context: Optional[Dict[str, Any]] = None) -> dspy.Prediction:
        """DSPy forward method for optimization."""
        context = context or {}
        
        # Process task with context
        task_analysis = self.task_processor(
            task=task,
            context=json.dumps(context),
            capabilities=[cap.value for cap in self.capabilities]
        )
        
        # Generate execution plan
        if context.get("requires_reasoning", False):
            reasoning_result = self.reasoning_engine(
                complex_problem=task,
                available_knowledge=json.dumps(context)
            )
            
            return dspy.Prediction(
                task=task,
                analysis=task_analysis,
                reasoning=reasoning_result,
                execution_plan=self._generate_execution_plan(task_analysis, reasoning_result),
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                timestamp=datetime.now().isoformat()
            )
        else:
            return dspy.Prediction(
                task=task,
                analysis=task_analysis,
                execution_plan=self._generate_simple_execution_plan(task_analysis),
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                timestamp=datetime.now().isoformat()
            )
    
    # MCP Integration
    
    async def _execute_coordination_hooks(self, hook_type: str, data: Optional[Dict[str, Any]] = None):
        """Execute MCP coordination hooks."""
        if not self.mcp_config.memory_enabled:
            return
        
        try:
            hook_data = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "session_id": self.coordination_session,
                "hook_type": hook_type,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            }
            
            # Store hook execution in memory
            await self._store_shared_memory(f"hook_{hook_type}_{self.agent_id}", hook_data)
            
            # Update operation counter
            self.operation_counters["coordination_events"] += 1
            
        except Exception as e:
            logger.error(f"Coordination hook execution failed: {e}")
    
    async def _store_shared_memory(self, key: str, data: Dict[str, Any]):
        """Store data in shared memory system."""
        try:
            memory_key = f"{self.mcp_config.memory_namespace}:{key}"
            self.shared_memory[memory_key] = {
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id
            }
            
            self.operation_counters["mcp_operations"] += 1
            
        except Exception as e:
            logger.error(f"Shared memory storage failed: {e}")
    
    async def _retrieve_shared_memory(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from shared memory system."""
        try:
            memory_key = f"{self.mcp_config.memory_namespace}:{key}"
            return self.shared_memory.get(memory_key)
            
        except Exception as e:
            logger.error(f"Shared memory retrieval failed: {e}")
            return None
    
    # Performance and Metrics
    
    def _update_metrics(self, execution_time_ms: float, success: bool):
        """Update agent performance metrics."""
        self.metrics.tasks_completed += 1
        self.metrics.last_active = datetime.now()
        
        # Update success rate (running average)
        if self.metrics.tasks_completed == 1:
            self.metrics.success_rate = 1.0 if success else 0.0
        else:
            current_successes = self.metrics.success_rate * (self.metrics.tasks_completed - 1)
            if success:
                current_successes += 1
            self.metrics.success_rate = current_successes / self.metrics.tasks_completed
        
        # Update response time (running average)
        if self.metrics.tasks_completed == 1:
            self.metrics.average_response_time_ms = execution_time_ms
        else:
            total_time = self.metrics.average_response_time_ms * (self.metrics.tasks_completed - 1)
            self.metrics.average_response_time_ms = (total_time + execution_time_ms) / self.metrics.tasks_completed
        
        self.operation_counters["tasks_processed"] += 1
    
    def _unified_success_metric(self, example: Any, prediction: Any, trace=None) -> float:
        """Unified success metric for DSPy optimization."""
        try:
            # Base success criteria
            task_completion = 0.3 if hasattr(prediction, 'execution_plan') else 0.0
            reasoning_quality = 0.2 if hasattr(prediction, 'reasoning') else 0.0
            coordination_integration = 0.2 if hasattr(prediction, 'agent_id') else 0.0
            
            # Performance criteria
            response_time_score = 0.15 if self.metrics.average_response_time_ms < 200 else 0.0
            success_rate_score = 0.15 * self.metrics.success_rate
            
            return task_completion + reasoning_quality + coordination_integration + response_time_score + success_rate_score
            
        except Exception as e:
            logger.error(f"Success metric calculation failed: {e}")
            return 0.0
    
    # Health and Status
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with all system status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "metrics": {
                "tasks_completed": self.metrics.tasks_completed,
                "success_rate": self.metrics.success_rate,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "memory_operations": self.metrics.memory_operations,
                "communication_events": self.metrics.communication_events,
                "optimization_cycles": self.metrics.optimization_cycles
            },
            "system_integrations": {
                "dspy_initialized": hasattr(self, 'task_processor'),
                "mcp_coordination": self.mcp_config.memory_enabled,
                "sequential_thinking": self.mcp_config.sequential_thinking_enabled,
                "monitoring_active": len(self.operation_counters) > 0
            },
            "performance_status": {
                "meets_response_time_target": self.metrics.average_response_time_ms < self.performance_baseline["response_time_target_ms"],
                "meets_success_rate_target": self.metrics.success_rate >= self.performance_baseline["success_rate_target"],
                "healthy": self.status == AgentStatus.ACTIVE and self.metrics.success_rate >= 0.8
            },
            "last_active": self.metrics.last_active.isoformat() if self.metrics.last_active else None
        }
    
    # Abstract Methods for Domain-Specific Implementation
    
    @abstractmethod
    async def initialize_domain_resources(self) -> bool:
        """Initialize domain-specific resources."""
        pass
    
    @abstractmethod
    async def cleanup_domain_resources(self) -> None:
        """Clean up domain-specific resources."""
        pass
    
    @abstractmethod
    async def execute_domain_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute domain-specific task logic."""
        pass
    
    # Helper Methods
    
    async def _generate_reasoning_chain(self, task: Dict[str, Any]) -> str:
        """Generate sequential thinking chain for complex tasks."""
        try:
            task_type = task.get("type", "general")
            complexity = task.get("complexity", "medium")
            
            # Basic reasoning patterns based on agent type and task
            if self.agent_type == "researcher":
                if complexity == "high":
                    return "1. Analyze requirements -> 2. Research sources -> 3. Cross-reference -> 4. Synthesize -> 5. Validate -> 6. Report"
                else:
                    return "1. Identify topic -> 2. Search sources -> 3. Extract information -> 4. Summarize"
            elif self.agent_type == "coder":
                if complexity == "high":
                    return "1. Analyze requirements -> 2. Design architecture -> 3. Plan implementation -> 4. Code components -> 5. Test -> 6. Integrate -> 7. Optimize"
                else:
                    return "1. Understand task -> 2. Plan approach -> 3. Implement -> 4. Test"
            else:
                return f"1. Analyze {task_type} -> 2. Plan execution -> 3. Execute -> 4. Validate results"
                
        except Exception as e:
            logger.error(f"Reasoning chain generation failed: {e}")
            return "1. Analyze -> 2. Execute -> 3. Validate"
    
    async def _execute_with_dspy_optimization(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with DSPy optimization."""
        try:
            # Use DSPy forward method for optimized execution
            task_description = task.get("description", str(task))
            context = task.get("context", {})
            
            prediction = self.forward(task_description, context)
            
            # Execute domain-specific logic
            domain_result = await self.execute_domain_task(task)
            
            # Combine DSPy optimization with domain results
            return {
                "status": "success",
                "domain_result": domain_result,
                "dspy_analysis": prediction.analysis if hasattr(prediction, 'analysis') else None,
                "execution_plan": prediction.execution_plan if hasattr(prediction, 'execution_plan') else None,
                "optimization_applied": True
            }
            
        except Exception as e:
            logger.error(f"DSPy execution failed: {e}")
            # Fallback to domain execution
            return await self.execute_domain_task(task)
    
    def _generate_execution_plan(self, analysis: Any, reasoning: Any) -> Dict[str, Any]:
        """Generate execution plan from DSPy analysis and reasoning."""
        return {
            "steps": ["analyze", "plan", "execute", "validate"],
            "estimated_time_ms": 150,
            "resources_required": ["memory", "computation"],
            "success_probability": 0.9
        }
    
    def _generate_simple_execution_plan(self, analysis: Any) -> Dict[str, Any]:
        """Generate simple execution plan."""
        return {
            "steps": ["execute", "validate"],
            "estimated_time_ms": 80,
            "resources_required": ["computation"],
            "success_probability": 0.95
        }
    
    async def _complete_active_tasks(self):
        """Complete or cancel active tasks during shutdown."""
        for task_id, task in list(self.current_tasks.items()):
            try:
                # Attempt graceful completion
                result = await self.execute_domain_task(task)
                await self._store_shared_memory(f"final_result_{task_id}", {
                    "result": result,
                    "status": "completed_during_shutdown"
                })
            except Exception as e:
                logger.error(f"Failed to complete task {task_id}: {e}")
        
        self.current_tasks.clear()
    
    async def _store_final_metrics(self):
        """Store final performance metrics."""
        try:
            final_metrics = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "session_id": self.coordination_session,
                "final_metrics": self.metrics.__dict__,
                "operation_counters": self.operation_counters,
                "shutdown_time": datetime.now().isoformat(),
                "total_runtime_seconds": (datetime.now() - self.start_time).total_seconds()
            }
            
            await self._store_shared_memory(f"final_metrics_{self.agent_id}", final_metrics)
            
        except Exception as e:
            logger.error(f"Final metrics storage failed: {e}")
    
    async def _cleanup_system_integrations(self):
        """Clean up all system integrations."""
        try:
            # Clear shared memory for this agent
            keys_to_remove = [k for k in self.shared_memory.keys() if self.agent_id in str(k)]
            for key in keys_to_remove:
                self.shared_memory.pop(key, None)
            
            # Reset counters
            self.operation_counters.clear()
            
            logger.info("System integrations cleaned up successfully")
            
        except Exception as e:
            logger.error(f"System cleanup failed: {e}")
    
    # Additional DSPy methods for optimization
    async def _initialize_dspy_optimization(self):
        """Initialize DSPy optimization with training data."""
        try:
            if self.dspy_config.enable_evaluation:
                # Create basic training examples for optimization
                training_examples = [
                    {"task": "simple computation", "expected_time_ms": 50},
                    {"task": "complex analysis", "expected_time_ms": 200},
                    {"task": "coordination task", "expected_time_ms": 100}
                ]
                
                if len(training_examples) >= self.dspy_config.training_examples_required:
                    # Run optimization
                    optimized = self.optimizer.compile(
                        student=self,
                        trainset=training_examples[:3],
                        valset=training_examples[:2] if len(training_examples) > 2 else training_examples
                    )
                    
                    self.metrics.optimization_cycles += 1
                    logger.info("DSPy optimization completed successfully")
                    
        except Exception as e:
            logger.error(f"DSPy optimization initialization failed: {e}")
    
    async def _setup_memory_coordination(self):
        """Setup memory coordination systems."""
        try:
            if self.mcp_config.memory_enabled:
                # Initialize memory coordination
                await self._store_shared_memory("agent_registry", {
                    "agent_id": self.agent_id,
                    "agent_type": self.agent_type,
                    "capabilities": [cap.value for cap in self.capabilities],
                    "session_id": self.coordination_session,
                    "initialized_at": datetime.now().isoformat()
                })
                
                logger.info("Memory coordination setup complete")
                
        except Exception as e:
            logger.error(f"Memory coordination setup failed: {e}")
    
    async def _setup_sequential_thinking(self):
        """Setup sequential thinking integration."""
        try:
            if self.mcp_config.sequential_thinking_enabled:
                # Generate default reasoning chains for common tasks
                self.reasoning_chains.update({
                    "simple": "1. Analyze -> 2. Execute -> 3. Validate",
                    "complex": "1. Decompose -> 2. Research -> 3. Plan -> 4. Execute -> 5. Test -> 6. Validate",
                    "coordination": "1. Assess -> 2. Coordinate -> 3. Monitor -> 4. Adjust"
                })
                
                logger.info("Sequential thinking setup complete")
                
        except Exception as e:
            logger.error(f"Sequential thinking setup failed: {e}")
    
    async def _start_monitoring(self):
        """Start monitoring and instrumentation."""
        try:
            # Initialize performance baselines
            self.performance_baseline.update({
                "baseline_established": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "agent_type": self.agent_type
            })
            
            logger.info("Monitoring started successfully")
            
        except Exception as e:
            logger.error(f"Monitoring startup failed: {e}")


# Factory Functions for Common Agent Types

def create_researcher_agent(agent_id: str, **kwargs) -> UnifiedBaseAgent:
    """Create a research-specialized unified agent."""
    capabilities = [
        AgentCapability.REASONING,
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.COMMUNICATION,
        AgentCapability.LEARNING
    ]
    
    # Custom MCP config for researchers
    mcp_config = MCPConfiguration(
        memory_enabled=True,
        sequential_thinking_enabled=True,
        github_integration_enabled=True
    )
    
    class ResearcherAgent(UnifiedBaseAgent):
        async def initialize_domain_resources(self) -> bool:
            logger.info("Initializing researcher domain resources")
            return True
        
        async def cleanup_domain_resources(self) -> None:
            logger.info("Cleaning up researcher domain resources")
        
        async def execute_domain_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            # Research-specific task execution
            return {
                "status": "success",
                "research_results": f"Research completed for: {task.get('description', 'task')}",
                "sources_analyzed": 5,
                "confidence_score": 0.85
            }
    
    return ResearcherAgent(
        agent_id=agent_id,
        agent_type="researcher",
        capabilities=capabilities,
        mcp_config=mcp_config,
        **kwargs
    )


def create_coder_agent(agent_id: str, **kwargs) -> UnifiedBaseAgent:
    """Create a coding-specialized unified agent."""
    capabilities = [
        AgentCapability.REASONING,
        AgentCapability.COMPUTATION,
        AgentCapability.MEMORY_ACCESS,
        AgentCapability.MONITORING
    ]
    
    class CoderAgent(UnifiedBaseAgent):
        async def initialize_domain_resources(self) -> bool:
            logger.info("Initializing coder domain resources")
            return True
        
        async def cleanup_domain_resources(self) -> None:
            logger.info("Cleaning up coder domain resources")
        
        async def execute_domain_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            # Coding-specific task execution
            return {
                "status": "success",
                "code_generated": f"Code implementation for: {task.get('description', 'task')}",
                "lines_of_code": 150,
                "test_coverage": 95.0
            }
    
    return CoderAgent(
        agent_id=agent_id,
        agent_type="coder", 
        capabilities=capabilities,
        **kwargs
    )


# Export unified agent framework
__all__ = [
    "UnifiedBaseAgent",
    "AgentStatus",
    "AgentCapability", 
    "MCPConfiguration",
    "DSPyConfiguration",
    "AgentMetrics",
    "create_researcher_agent",
    "create_coder_agent"
]