"""
Memory Patterns for MCP Server Coordination
Defines standard patterns for memory organization and cross-agent information sharing.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class MemoryScope(Enum):
    """Define memory scopes for different use cases"""
    GLOBAL = "global"           # Long-term, cross-session memory
    PROJECT = "project"         # Medium-term, project-specific
    SESSION = "session"         # Short-term, session-specific
    TASK = "task"              # Ephemeral, task-specific

class MemoryPattern(Enum):
    """Standard memory patterns for different workflows"""
    PROJECT_CONTEXT = "project_context"
    AGENT_COORDINATION = "agent_coordination"
    LEARNING_PATTERNS = "learning_patterns"
    PERFORMANCE_METRICS = "performance_metrics"
    ERROR_PATTERNS = "error_patterns"

@dataclass
class MemoryNamespace:
    """Standard namespace structure for organized memory storage"""
    scope: MemoryScope
    category: str
    subcategory: Optional[str] = None
    
    def to_string(self) -> str:
        """Convert namespace to string format"""
        parts = [self.scope.value, self.category]
        if self.subcategory:
            parts.append(self.subcategory)
        return "/".join(parts)

class MemoryPatterns:
    """Standard memory patterns for MCP server coordination"""
    
    # Project Context Patterns
    PROJECT_CONFIG = "project/{project_name}/config"
    PROJECT_ARCHITECTURE = "project/{project_name}/architecture"
    PROJECT_DEPENDENCIES = "project/{project_name}/dependencies" 
    PROJECT_ISSUES = "project/{project_name}/issues"
    PROJECT_DECISIONS = "project/{project_name}/decisions"
    
    # Agent Coordination Patterns
    AGENT_ASSIGNMENTS = "coordination/{session_id}/assignments"
    AGENT_RESULTS = "coordination/{session_id}/results"
    AGENT_COMMUNICATION = "coordination/{session_id}/communication"
    AGENT_METRICS = "coordination/{session_id}/metrics"
    AGENT_ERRORS = "coordination/{session_id}/errors"
    
    # Learning Patterns
    SUCCESSFUL_STRATEGIES = "patterns/strategies/{category}"
    COMMON_SOLUTIONS = "patterns/solutions/{category}"
    ERROR_PATTERNS = "patterns/errors/{category}"
    OPTIMIZATION_TECHNIQUES = "patterns/optimizations/{category}"
    BEST_PRACTICES = "patterns/practices/{category}"
    
    # Performance Patterns
    PERFORMANCE_BASELINES = "performance/baselines/{component}"
    OPTIMIZATION_HISTORY = "performance/history/{component}"
    BOTTLENECK_PATTERNS = "performance/bottlenecks/{component}"
    IMPROVEMENT_METRICS = "performance/improvements/{component}"
    
    # MCP Server Patterns
    SERVER_CONFIGURATIONS = "mcp/servers/{server_name}/config"
    SERVER_PERFORMANCE = "mcp/servers/{server_name}/performance"
    SERVER_ERRORS = "mcp/servers/{server_name}/errors"
    SERVER_INTEGRATION_PATTERNS = "mcp/integration/{workflow_type}"

class ProjectContextManager:
    """Manages project-specific context and memory patterns"""
    
    def __init__(self, coordinator, project_name: str):
        self.coordinator = coordinator
        self.project_name = project_name
        self.namespace = f"project/{project_name}"
    
    async def store_architecture_decision(self, decision_id: str, decision_data: Dict[str, Any]):
        """Store architectural decision for future reference"""
        key = f"architecture_decision_{decision_id}"
        value = {
            "decision_id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **decision_data
        }
        
        await self.coordinator.store_memory(
            key, value, 
            namespace=f"{self.namespace}/decisions",
            tags=["architecture", "decision"]
        )
    
    async def store_api_contract(self, service_name: str, contract_data: Dict[str, Any]):
        """Store API contract information"""
        key = f"api_contract_{service_name}"
        value = {
            "service": service_name,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **contract_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"{self.namespace}/contracts",
            tags=["api", "contract", service_name]
        )
    
    async def store_configuration(self, config_type: str, config_data: Dict[str, Any]):
        """Store project configuration"""
        key = f"config_{config_type}"
        value = {
            "config_type": config_type,
            "timestamp": datetime.now().isoformat(),
            "project": self.project_name,
            **config_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"{self.namespace}/config",
            tags=["configuration", config_type]
        )
    
    async def get_project_context(self) -> Dict[str, Any]:
        """Retrieve comprehensive project context"""
        context = {
            "project_name": self.project_name,
            "architecture_decisions": [],
            "api_contracts": [],
            "configurations": [],
            "dependencies": [],
            "known_issues": []
        }
        
        # Get architecture decisions
        decisions = await self.coordinator.search_memory(
            "architecture_decision_", 
            namespace=f"{self.namespace}/decisions"
        )
        context["architecture_decisions"] = [d["value"] for d in decisions]
        
        # Get API contracts
        contracts = await self.coordinator.search_memory(
            "api_contract_",
            namespace=f"{self.namespace}/contracts"
        )
        context["api_contracts"] = [c["value"] for c in contracts]
        
        # Get configurations
        configs = await self.coordinator.search_memory(
            "config_",
            namespace=f"{self.namespace}/config"
        )
        context["configurations"] = [c["value"] for c in configs]
        
        return context

class AgentCoordinationManager:
    """Manages agent coordination patterns and memory sharing"""
    
    def __init__(self, coordinator, session_id: str):
        self.coordinator = coordinator
        self.session_id = session_id
        self.namespace = f"coordination/{session_id}"
    
    async def store_task_assignment(self, agent_id: str, task_data: Dict[str, Any]):
        """Store task assignment for an agent"""
        key = f"task_assignment_{agent_id}_{datetime.now().timestamp()}"
        value = {
            "agent_id": agent_id,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            **task_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"{self.namespace}/assignments",
            tags=["task", "assignment", agent_id]
        )
    
    async def store_intermediate_result(self, agent_id: str, phase: str, result_data: Dict[str, Any]):
        """Store intermediate results from agent work"""
        key = f"result_{agent_id}_{phase}_{datetime.now().timestamp()}"
        value = {
            "agent_id": agent_id,
            "phase": phase,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            **result_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"{self.namespace}/results",
            tags=["result", "intermediate", agent_id, phase]
        )
    
    async def store_communication_log(self, from_agent: str, to_agent: str, message: str):
        """Store inter-agent communication"""
        key = f"comm_{from_agent}_{to_agent}_{datetime.now().timestamp()}"
        value = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"{self.namespace}/communication",
            tags=["communication", from_agent, to_agent]
        )
    
    async def get_agent_context(self, agent_id: str) -> Dict[str, Any]:
        """Get all context relevant to a specific agent"""
        context = {
            "agent_id": agent_id,
            "session_id": self.session_id,
            "assignments": [],
            "results": [],
            "communications": []
        }
        
        # Get task assignments
        assignments = await self.coordinator.search_memory(
            f"task_assignment_{agent_id}",
            namespace=f"{self.namespace}/assignments"
        )
        context["assignments"] = [a["value"] for a in assignments]
        
        # Get results
        results = await self.coordinator.search_memory(
            f"result_{agent_id}",
            namespace=f"{self.namespace}/results"
        )
        context["results"] = [r["value"] for r in results]
        
        # Get communications
        comms = await self.coordinator.search_memory(
            agent_id,
            namespace=f"{self.namespace}/communication"
        )
        context["communications"] = [c["value"] for c in comms]
        
        return context

class LearningPatternsManager:
    """Manages learning patterns and knowledge accumulation"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    async def store_successful_strategy(self, category: str, strategy_data: Dict[str, Any]):
        """Store a successful strategy for future use"""
        key = f"strategy_{category}_{datetime.now().timestamp()}"
        value = {
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "success_metrics": strategy_data.get("metrics", {}),
            **strategy_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"patterns/strategies/{category}",
            tags=["strategy", "successful", category]
        )
    
    async def store_common_solution(self, problem_type: str, solution_data: Dict[str, Any]):
        """Store a common solution pattern"""
        key = f"solution_{problem_type}_{datetime.now().timestamp()}"
        value = {
            "problem_type": problem_type,
            "timestamp": datetime.now().isoformat(),
            "effectiveness": solution_data.get("effectiveness", 0),
            **solution_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"patterns/solutions/{problem_type}",
            tags=["solution", "pattern", problem_type]
        )
    
    async def store_error_pattern(self, error_category: str, error_data: Dict[str, Any]):
        """Store error patterns for prevention"""
        key = f"error_{error_category}_{datetime.now().timestamp()}"
        value = {
            "error_category": error_category,
            "timestamp": datetime.now().isoformat(),
            "frequency": error_data.get("frequency", 1),
            "resolution": error_data.get("resolution", ""),
            **error_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"patterns/errors/{error_category}",
            tags=["error", "pattern", error_category]
        )
    
    async def get_learning_insights(self, category: str) -> Dict[str, Any]:
        """Get learning insights for a specific category"""
        insights = {
            "category": category,
            "successful_strategies": [],
            "common_solutions": [],
            "error_patterns": []
        }
        
        # Get successful strategies
        strategies = await self.coordinator.search_memory(
            "strategy_",
            namespace=f"patterns/strategies/{category}"
        )
        insights["successful_strategies"] = [s["value"] for s in strategies]
        
        # Get common solutions  
        solutions = await self.coordinator.search_memory(
            "solution_",
            namespace=f"patterns/solutions/{category}"
        )
        insights["common_solutions"] = [s["value"] for s in solutions]
        
        # Get error patterns
        errors = await self.coordinator.search_memory(
            "error_",
            namespace=f"patterns/errors/{category}"
        )
        insights["error_patterns"] = [e["value"] for e in errors]
        
        return insights

class PerformanceTrackingManager:
    """Manages performance tracking and optimization patterns"""
    
    def __init__(self, coordinator):
        self.coordinator = coordinator
    
    async def store_performance_baseline(self, component: str, metrics: Dict[str, Any]):
        """Store performance baseline for a component"""
        key = f"baseline_{component}"
        value = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "version": metrics.get("version", "unknown")
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"performance/baselines/{component}",
            tags=["performance", "baseline", component]
        )
    
    async def store_optimization_result(self, component: str, optimization_data: Dict[str, Any]):
        """Store optimization results"""
        key = f"optimization_{component}_{datetime.now().timestamp()}"
        value = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "improvement": optimization_data.get("improvement", 0),
            **optimization_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"performance/history/{component}",
            tags=["performance", "optimization", component]
        )
    
    async def store_bottleneck_pattern(self, component: str, bottleneck_data: Dict[str, Any]):
        """Store identified bottleneck patterns"""
        key = f"bottleneck_{component}_{datetime.now().timestamp()}"
        value = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "severity": bottleneck_data.get("severity", "medium"),
            **bottleneck_data
        }
        
        await self.coordinator.store_memory(
            key, value,
            namespace=f"performance/bottlenecks/{component}",
            tags=["performance", "bottleneck", component]
        )
    
    async def get_performance_insights(self, component: str) -> Dict[str, Any]:
        """Get comprehensive performance insights"""
        insights = {
            "component": component,
            "baseline": None,
            "optimization_history": [],
            "bottleneck_patterns": []
        }
        
        # Get baseline
        baseline = await self.coordinator.retrieve_memory(
            f"baseline_{component}",
            namespace=f"performance/baselines/{component}"
        )
        insights["baseline"] = baseline
        
        # Get optimization history
        optimizations = await self.coordinator.search_memory(
            "optimization_",
            namespace=f"performance/history/{component}"
        )
        insights["optimization_history"] = [o["value"] for o in optimizations]
        
        # Get bottleneck patterns
        bottlenecks = await self.coordinator.search_memory(
            "bottleneck_",
            namespace=f"performance/bottlenecks/{component}"
        )
        insights["bottleneck_patterns"] = [b["value"] for b in bottlenecks]
        
        return insights