"""
Enhanced Agent Coordinator - Memory Coordination Specialist Agent
Provides comprehensive MCP server coordination with memory persistence and intelligent agent spawning.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from .mcp_server_coordinator import MCPServerCoordinator
from .memory_patterns import (
    ProjectContextManager, 
    AgentCoordinationManager, 
    LearningPatternsManager,
    PerformanceTrackingManager
)

class EnhancedAgentCoordinator:
    """
    Enhanced Agent Coordinator with comprehensive MCP server integration
    Serves as the Memory Coordination Specialist Agent for the system
    """
    
    def __init__(self, project_name: str = "AIVillage"):
        self.project_name = project_name
        self.coordinator = MCPServerCoordinator()
        
        # Initialize pattern managers
        self.project_context = ProjectContextManager(self.coordinator, project_name)
        self.learning_patterns = LearningPatternsManager(self.coordinator)
        self.performance_tracking = PerformanceTrackingManager(self.coordinator)
        
        self.active_sessions: Dict[str, AgentCoordinationManager] = {}
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up enhanced logging"""
        logger = logging.getLogger("enhanced_agent_coordinator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    async def initialize_session(self, session_id: str, session_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new coordination session with memory persistence"""
        try:
            # Create session coordination manager
            session_manager = AgentCoordinationManager(self.coordinator, session_id)
            self.active_sessions[session_id] = session_manager
            
            # Store session initialization
            session_data = {
                "session_id": session_id,
                "project_name": self.project_name,
                "initialized_at": datetime.now().isoformat(),
                "config": session_config,
                "status": "active"
            }
            
            await self.coordinator.store_memory(
                f"session_{session_id}_init",
                session_data,
                namespace=f"session/{session_id}",
                tags=["session", "initialization"]
            )
            
            # Get project context for agents
            project_context = await self.project_context.get_project_context()
            
            # Recommend MCP servers based on session type
            task_type = session_config.get("task_type", "code_development")
            server_recommendations = await self.coordinator.recommend_mcp_servers(
                task_type, 
                session_config.get("requirements", [])
            )
            
            self.logger.info(f"Initialized session {session_id} with {len(server_recommendations['primary_servers'])} recommended servers")
            
            return {
                "session_id": session_id,
                "status": "initialized",
                "project_context": project_context,
                "recommended_servers": server_recommendations,
                "initialization_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session {session_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_enhanced_agent_prompt(self, agent_type: str, task_description: str, 
                                     context_keys: List[str] = None, 
                                     reasoning_chain: str = None) -> str:
        """Generate enhanced agent prompt with memory, sequential thinking, and DSPy integration"""
        
        base_prompt = f"""
You are a {agent_type} agent with enhanced capabilities through MCP server integration.

TASK: {task_description}

ENHANCED CAPABILITIES:
1. MEMORY INTEGRATION: You have access to persistent memory across sessions
2. SEQUENTIAL THINKING: Use structured reasoning for complex problems  
3. MCP SERVER ACCESS: Leverage specialized servers for enhanced functionality

REQUIRED COORDINATION PROTOCOL:
1. Before starting work, check memory for relevant context using these keys: {context_keys or []}
2. Store all significant findings and decisions in memory for future agents
3. Use sequential thinking for multi-step reasoning: {reasoning_chain or 'Analyze -> Plan -> Execute -> Validate'}
4. Report progress and coordinate with other agents through memory system

MEMORY NAMESPACES TO USE:
- project/{self.project_name}/* - Project-specific context
- coordination/{session_id}/* - Session coordination
- patterns/* - Learning and best practices
- performance/* - Performance tracking

AVAILABLE MCP SERVERS:
- Memory: Persistent storage and retrieval
- Sequential Thinking: Multi-step reasoning
- GitHub: Code repository operations
- HuggingFace: ML model operations
- Firecrawl: Web content extraction
- Context7: Real-time documentation
- And others as needed

YOUR COORDINATION RESPONSIBILITIES:
1. Check memory for prior work and context
2. Store your work products for other agents
3. Use sequential thinking for complex decisions
4. Coordinate through the memory system
5. Learn from patterns and improve over time

Begin your work following these enhanced protocols.
"""
        return base_prompt
    
    async def spawn_coordinated_agents(self, session_id: str, agents_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Spawn multiple agents with full coordination and memory integration"""
        
        if session_id not in self.active_sessions:
            await self.initialize_session(session_id, {"task_type": "multi_agent"})
        
        session_manager = self.active_sessions[session_id]
        spawned_agents = []
        
        try:
            for agent_config in agents_config:
                agent_id = f"{agent_config['type']}_{datetime.now().timestamp()}"
                
                # Generate enhanced prompt with coordination
                enhanced_prompt = self.generate_enhanced_agent_prompt(
                    agent_config['type'],
                    agent_config['task'],
                    agent_config.get('context_keys', []),
                    agent_config.get('reasoning_chain')
                )
                
                # Store agent assignment
                await session_manager.store_task_assignment(agent_id, {
                    "agent_type": agent_config['type'],
                    "task": agent_config['task'],
                    "enhanced_prompt": enhanced_prompt,
                    "required_servers": agent_config.get('servers', []),
                    "priority": agent_config.get('priority', 'medium')
                })
                
                spawned_agents.append({
                    "agent_id": agent_id,
                    "agent_type": agent_config['type'], 
                    "task": agent_config['task'],
                    "enhanced_prompt": enhanced_prompt,
                    "status": "spawned"
                })
                
                self.logger.info(f"Spawned {agent_config['type']} agent {agent_id} for session {session_id}")
            
            # Store coordination metadata
            coordination_data = {
                "session_id": session_id,
                "spawned_agents": len(spawned_agents),
                "agent_types": [a["agent_type"] for a in spawned_agents],
                "spawned_at": datetime.now().isoformat()
            }
            
            await self.coordinator.store_memory(
                f"session_{session_id}_coordination",
                coordination_data,
                namespace=f"coordination/{session_id}",
                tags=["coordination", "agents"]
            )
            
            return {
                "session_id": session_id,
                "spawned_agents": spawned_agents,
                "status": "agents_spawned",
                "coordination_data": coordination_data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to spawn coordinated agents for session {session_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def collect_agent_results(self, session_id: str, agent_id: str, 
                                   results: Dict[str, Any]) -> Dict[str, Any]:
        """Collect and store results from an agent"""
        
        if session_id not in self.active_sessions:
            return {"status": "error", "message": "Session not found"}
        
        session_manager = self.active_sessions[session_id]
        
        try:
            # Store agent results
            await session_manager.store_intermediate_result(
                agent_id, 
                results.get("phase", "completion"),
                results
            )
            
            # Update agent status
            await self.coordinator.update_agent_status(session_id, agent_id, "completed", results)
            
            # Check if this completes the session
            session_status = await self.evaluate_session_completion(session_id)
            
            self.logger.info(f"Collected results from agent {agent_id} in session {session_id}")
            
            return {
                "status": "results_collected",
                "agent_id": agent_id,
                "session_id": session_id,
                "session_status": session_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to collect results from agent {agent_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def evaluate_session_completion(self, session_id: str) -> Dict[str, Any]:
        """Evaluate if a coordination session is complete"""
        
        try:
            # Get all agent contexts for the session
            session_data = await self.coordinator.search_memory(
                "task_assignment_",
                namespace=f"coordination/{session_id}/assignments"
            )
            
            results_data = await self.coordinator.search_memory(
                "result_",
                namespace=f"coordination/{session_id}/results"
            )
            
            total_agents = len(session_data)
            completed_agents = len(set(r["value"]["agent_id"] for r in results_data))
            
            completion_status = {
                "total_agents": total_agents,
                "completed_agents": completed_agents,
                "completion_percentage": (completed_agents / total_agents * 100) if total_agents > 0 else 0,
                "status": "completed" if completed_agents == total_agents else "in_progress"
            }
            
            # If session is complete, store summary
            if completion_status["status"] == "completed":
                await self.finalize_session(session_id, completion_status)
            
            return completion_status
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate session completion for {session_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def finalize_session(self, session_id: str, completion_status: Dict[str, Any]):
        """Finalize a completed coordination session"""
        
        try:
            # Collect all session data
            session_summary = {
                "session_id": session_id,
                "project_name": self.project_name,
                "completion_status": completion_status,
                "finalized_at": datetime.now().isoformat()
            }
            
            # Get session results
            results = await self.coordinator.search_memory(
                "result_",
                namespace=f"coordination/{session_id}/results"
            )
            session_summary["results"] = [r["value"] for r in results]
            
            # Get communications
            communications = await self.coordinator.search_memory(
                "comm_",
                namespace=f"coordination/{session_id}/communication"
            )
            session_summary["communications"] = [c["value"] for c in communications]
            
            # Store session summary
            await self.coordinator.store_memory(
                f"session_{session_id}_summary",
                session_summary,
                namespace=f"completed_sessions",
                tags=["session", "completed", "summary"]
            )
            
            # Extract learning patterns
            await self.extract_learning_patterns(session_id, session_summary)
            
            # Clean up active session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            self.logger.info(f"Finalized session {session_id} with {completion_status['completed_agents']} completed agents")
            
        except Exception as e:
            self.logger.error(f"Failed to finalize session {session_id}: {e}")
    
    async def extract_learning_patterns(self, session_id: str, session_summary: Dict[str, Any]):
        """Extract learning patterns from completed session"""
        
        try:
            # Identify successful strategies
            successful_results = [
                r for r in session_summary["results"] 
                if r.get("success", False) or r.get("quality_score", 0) > 0.7
            ]
            
            if successful_results:
                strategy_data = {
                    "session_id": session_id,
                    "successful_agents": len(successful_results),
                    "success_factors": [r.get("success_factors", []) for r in successful_results],
                    "metrics": {
                        "completion_time": session_summary["completion_status"].get("completion_time", 0),
                        "quality_average": sum(r.get("quality_score", 0) for r in successful_results) / len(successful_results)
                    }
                }
                
                await self.learning_patterns.store_successful_strategy(
                    "multi_agent_coordination",
                    strategy_data
                )
            
            # Identify error patterns
            failed_results = [
                r for r in session_summary["results"]
                if r.get("errors") or r.get("quality_score", 1) < 0.3
            ]
            
            if failed_results:
                error_data = {
                    "session_id": session_id,
                    "failed_agents": len(failed_results),
                    "error_types": [r.get("errors", []) for r in failed_results],
                    "common_issues": self.extract_common_issues(failed_results)
                }
                
                await self.learning_patterns.store_error_pattern(
                    "multi_agent_coordination",
                    error_data
                )
            
        except Exception as e:
            self.logger.error(f"Failed to extract learning patterns from session {session_id}: {e}")
    
    def extract_common_issues(self, failed_results: List[Dict[str, Any]]) -> List[str]:
        """Extract common issues from failed results"""
        all_errors = []
        for result in failed_results:
            errors = result.get("errors", [])
            if isinstance(errors, list):
                all_errors.extend(errors)
            elif isinstance(errors, str):
                all_errors.append(errors)
        
        # Simple frequency analysis
        error_counts = {}
        for error in all_errors:
            error_key = str(error)[:100]  # Truncate for grouping
            error_counts[error_key] = error_counts.get(error_key, 0) + 1
        
        # Return most common issues
        return sorted(error_counts.keys(), key=lambda x: error_counts[x], reverse=True)[:5]
    
    async def get_coordination_analytics(self) -> Dict[str, Any]:
        """Get comprehensive coordination analytics"""
        
        try:
            # Get memory analytics
            memory_analytics = await self.coordinator.get_memory_analytics()
            
            # Get completed sessions
            completed_sessions = await self.coordinator.search_memory(
                "session_",
                namespace="completed_sessions"
            )
            
            # Get learning insights
            learning_insights = await self.learning_patterns.get_learning_insights("multi_agent_coordination")
            
            analytics = {
                "memory_analytics": memory_analytics,
                "completed_sessions": len(completed_sessions),
                "active_sessions": len(self.active_sessions),
                "learning_insights": learning_insights,
                "project_name": self.project_name,
                "generated_at": datetime.now().isoformat()
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get coordination analytics: {e}")
            return {"status": "error", "message": str(e)}

# Usage example
async def main():
    """Example usage of Enhanced Agent Coordinator"""
    
    coordinator = EnhancedAgentCoordinator("AIVillage")
    
    # Initialize a development session
    session_config = {
        "task_type": "code_development",
        "requirements": ["speed", "concurrent", "memory_persistence"],
        "target": "multi_agent_system_enhancement"
    }
    
    session_result = await coordinator.initialize_session("dev_session_001", session_config)
    print(f"Session initialized: {session_result['status']}")
    
    # Spawn coordinated agents
    agents_config = [
        {
            "type": "researcher",
            "task": "Analyze MCP server integration patterns and performance characteristics",
            "context_keys": ["mcp_servers", "integration_patterns"],
            "reasoning_chain": "Research -> Analyze -> Synthesize -> Validate",
            "servers": ["memory", "github", "context7"],
            "priority": "high"
        },
        {
            "type": "system-architect", 
            "task": "Design enhanced coordination framework with memory persistence",
            "context_keys": ["project_architecture", "coordination_patterns"],
            "reasoning_chain": "Analyze Requirements -> Design Architecture -> Validate Design -> Document",
            "servers": ["memory", "sequential_thinking"],
            "priority": "high"
        },
        {
            "type": "coder",
            "task": "Implement memory coordination components and MCP integration",
            "context_keys": ["architecture_decisions", "implementation_patterns"],
            "reasoning_chain": "Plan Implementation -> Code -> Test -> Document",
            "servers": ["memory", "github"],
            "priority": "medium"
        }
    ]
    
    spawn_result = await coordinator.spawn_coordinated_agents("dev_session_001", agents_config)
    print(f"Agents spawned: {spawn_result['status']}")
    print(f"Enhanced prompts generated for {len(spawn_result['spawned_agents'])} agents")
    
    # Get coordination analytics
    analytics = await coordinator.get_coordination_analytics()
    print(f"Coordination analytics: {json.dumps(analytics, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())