"""
Gateway Integration for CognativeNexusController

Provides the interface between the API Gateway and the unified agent system.
Architecture: API Gateway → **Gateway Integration** → CognativeNexusController → Knowledge System → Response
"""

import asyncio
from datetime import datetime
import logging
from typing import Any
from uuid import uuid4

from .cognative_nexus_controller import (
    AgentType,
    CognativeNexusController,
    CognativeTask,
    TaskPriority,
    create_cognative_nexus_controller,
)

logger = logging.getLogger(__name__)


class AgentGatewayInterface:
    """
    Gateway interface for the unified agent orchestration system

    Provides high-level API for external systems to interact with agents
    while maintaining performance targets and error handling.
    """

    def __init__(self, controller: CognativeNexusController):
        self.controller = controller
        self.request_cache: dict[str, Any] = {}
        self.active_sessions: dict[str, dict[str, Any]] = {}

        logger.info("Agent Gateway Interface initialized")

    async def process_request(
        self, request_type: str, content: str, session_id: str | None = None, priority: str = "normal", **kwargs
    ) -> dict[str, Any]:
        """
        Process a request through the agent system

        Args:
            request_type: Type of request (query, analysis, task, etc.)
            content: The content to process
            session_id: Optional session identifier for context
            priority: Request priority (low, normal, high, critical, emergency)
            **kwargs: Additional parameters

        Returns:
            Complete response with results and metadata
        """
        start_time = datetime.now()
        request_id = str(uuid4())

        try:
            logger.info(f"Processing request {request_id}: {request_type}")

            # Map priority string to enum
            priority_map = {
                "low": TaskPriority.LOW,
                "normal": TaskPriority.NORMAL,
                "high": TaskPriority.HIGH,
                "critical": TaskPriority.CRITICAL,
                "emergency": TaskPriority.EMERGENCY,
            }
            task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)

            # Create task based on request type
            if request_type == "query":
                result = await self._process_query(content, task_priority, session_id, **kwargs)
            elif request_type == "analysis":
                result = await self._process_analysis(content, task_priority, session_id, **kwargs)
            elif request_type == "task":
                result = await self._process_task(content, task_priority, session_id, **kwargs)
            elif request_type == "conversation":
                result = await self._process_conversation(content, task_priority, session_id, **kwargs)
            else:
                result = await self._process_generic(content, task_priority, session_id, **kwargs)

            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Build response
            response = {
                "request_id": request_id,
                "session_id": session_id,
                "status": "success",
                "request_type": request_type,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat(),
                "result": result,
            }

            # Update session if provided
            if session_id:
                self._update_session(session_id, request_id, response)

            logger.info(f"Request {request_id} completed in {processing_time:.1f}ms")
            return response

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Request {request_id} failed after {processing_time:.1f}ms: {e}")

            return {
                "request_id": request_id,
                "session_id": session_id,
                "status": "error",
                "error": str(e),
                "request_type": request_type,
                "processing_time_ms": processing_time,
                "timestamp": datetime.now().isoformat(),
            }

    async def _process_query(
        self, content: str, priority: TaskPriority, session_id: str | None, **kwargs
    ) -> dict[str, Any]:
        """Process a knowledge query request"""

        # Select appropriate agent for query processing
        agent_type = self._select_agent_for_query(content)

        # Ensure agent exists
        agents = await self.controller.get_agents_by_type(agent_type)
        if not agents:
            # Create agent if none exists
            agent_id = await self.controller.create_agent(agent_type)
            if not agent_id:
                raise RuntimeError(f"Failed to create {agent_type.value} agent for query processing")

        # Create and process task
        task = CognativeTask(
            task_id=f"query_{uuid4().hex[:8]}",
            description=f"Answer this query: {content}",
            priority=priority,
            requires_reasoning=True,
            reasoning_strategy="probabilistic",
            confidence_threshold=kwargs.get("confidence_threshold", 0.7),
            max_iterations=kwargs.get("max_iterations", 3),
        )

        result = await self.controller.process_task_with_act_halting(task)

        return {
            "query": content,
            "answer": result.get("result", "No answer generated"),
            "confidence": result.get("confidence", 0.0),
            "agent_type": agent_type.value,
            "iterations_used": result.get("iterations_used", 1),
            "reasoning_applied": task.requires_reasoning,
        }

    async def _process_analysis(
        self, content: str, priority: TaskPriority, session_id: str | None, **kwargs
    ) -> dict[str, Any]:
        """Process an analysis request"""

        # Use appropriate analyst agent
        agent_type = AgentType.STRATEGIST  # Default to strategist for analysis

        # Create analysis task with higher iteration limit
        task = CognativeTask(
            task_id=f"analysis_{uuid4().hex[:8]}",
            description=f"Provide comprehensive analysis of: {content}",
            priority=priority,
            requires_reasoning=True,
            reasoning_strategy="deductive",
            confidence_threshold=0.8,
            max_iterations=5,  # More iterations for thorough analysis
            iterative_refinement=True,
        )

        # Ensure analyst agent exists
        agents = await self.controller.get_agents_by_type(agent_type)
        if not agents:
            agent_id = await self.controller.create_agent(agent_type)
            if not agent_id:
                raise RuntimeError(f"Failed to create {agent_type.value} agent for analysis")

        result = await self.controller.process_task_with_act_halting(task)

        return {
            "content_analyzed": content,
            "analysis": result.get("result", "No analysis generated"),
            "confidence": result.get("confidence", 0.0),
            "depth": "comprehensive",
            "agent_type": agent_type.value,
            "iterations_used": result.get("iterations_used", 1),
            "halted_early": result.get("halted_early", False),
        }

    async def _process_task(
        self, content: str, priority: TaskPriority, session_id: str | None, **kwargs
    ) -> dict[str, Any]:
        """Process a general task request"""

        # Infer agent type from task content
        agent_type = self._infer_agent_type_from_task(content)

        # Create task
        task = CognativeTask(
            task_id=f"task_{uuid4().hex[:8]}",
            description=content,
            priority=priority,
            requires_reasoning=kwargs.get("requires_reasoning", True),
            max_iterations=kwargs.get("max_iterations", 3),
        )

        # Ensure suitable agent exists
        agents = await self.controller.get_agents_by_type(agent_type)
        if not agents:
            agent_id = await self.controller.create_agent(agent_type)
            if not agent_id:
                raise RuntimeError(f"Failed to create {agent_type.value} agent for task")

        result = await self.controller.process_task_with_act_halting(task)

        return {
            "task_description": content,
            "task_result": result.get("result", "No result generated"),
            "confidence": result.get("confidence", 0.0),
            "agent_type": agent_type.value,
            "completion_status": "completed" if result.get("status") == "success" else "failed",
        }

    async def _process_conversation(
        self, content: str, priority: TaskPriority, session_id: str | None, **kwargs
    ) -> dict[str, Any]:
        """Process a conversational request with session context"""

        # Get session context
        session_context = self.active_sessions.get(session_id, {})
        previous_exchanges = session_context.get("exchanges", [])

        # Build contextual prompt
        context_prompt = content
        if previous_exchanges:
            recent_context = previous_exchanges[-3:]  # Last 3 exchanges for context
            context_parts = []
            for exchange in recent_context:
                context_parts.append(f"Previous: {exchange.get('query', '')}")
                context_parts.append(f"Response: {exchange.get('response', '')}")

            context_prompt = "\n".join(context_parts) + f"\n\nCurrent: {content}"

        # Use social agent for conversational interactions
        agent_type = AgentType.SOCIAL

        # Create conversational task
        task = CognativeTask(
            task_id=f"conversation_{uuid4().hex[:8]}",
            description=f"Respond conversationally to: {context_prompt}",
            priority=priority,
            requires_reasoning=False,  # Conversational responses don't need heavy reasoning
            max_iterations=2,  # Keep conversations responsive
        )

        # Ensure social agent exists
        agents = await self.controller.get_agents_by_type(agent_type)
        if not agents:
            agent_id = await self.controller.create_agent(agent_type)
            if not agent_id:
                raise RuntimeError("Failed to create social agent for conversation")

        result = await self.controller.process_task_with_act_halting(task)

        return {
            "message": content,
            "response": result.get("result", "I apologize, I could not generate a response."),
            "confidence": result.get("confidence", 0.0),
            "session_id": session_id,
            "has_context": len(previous_exchanges) > 0,
            "agent_type": agent_type.value,
        }

    async def _process_generic(
        self, content: str, priority: TaskPriority, session_id: str | None, **kwargs
    ) -> dict[str, Any]:
        """Process a generic request"""

        # Use sage agent as default for generic requests
        agent_type = AgentType.SAGE

        task = CognativeTask(
            task_id=f"generic_{uuid4().hex[:8]}", description=content, priority=priority, requires_reasoning=True
        )

        # Ensure sage agent exists
        agents = await self.controller.get_agents_by_type(agent_type)
        if not agents:
            agent_id = await self.controller.create_agent(agent_type)
            if not agent_id:
                raise RuntimeError("Failed to create sage agent for generic processing")

        result = await self.controller.process_task_with_act_halting(task)

        return {
            "input": content,
            "output": result.get("result", "No output generated"),
            "confidence": result.get("confidence", 0.0),
            "agent_type": agent_type.value,
            "processing_approach": "generic",
        }

    def _select_agent_for_query(self, query: str) -> AgentType:
        """Select the most appropriate agent type for a query"""

        query_lower = query.lower()

        # Knowledge and research queries
        if any(word in query_lower for word in ["research", "study", "analyze", "investigate"]):
            return AgentType.SAGE

        # Oracle for predictions and insights
        if any(word in query_lower for word in ["predict", "forecast", "future", "trend"]):
            return AgentType.ORACLE

        # Strategic queries
        if any(word in query_lower for word in ["strategy", "plan", "approach", "method"]):
            return AgentType.STRATEGIST

        # Technical queries
        if any(word in query_lower for word in ["code", "program", "technical", "architecture"]):
            return AgentType.ARCHITECT

        # Creative queries
        if any(word in query_lower for word in ["creative", "design", "artistic", "innovative"]):
            return AgentType.CREATIVE

        # Default to sage for general knowledge
        return AgentType.SAGE

    def _infer_agent_type_from_task(self, task_description: str) -> AgentType:
        """Infer the most appropriate agent type from task description"""

        task_lower = task_description.lower()

        # Governance tasks
        if any(word in task_lower for word in ["govern", "manage", "lead", "coordinate"]):
            return AgentType.KING

        # Security tasks
        if any(word in task_lower for word in ["security", "protect", "defend", "secure"]):
            return AgentType.SHIELD

        # Infrastructure tasks
        if any(word in task_lower for word in ["infrastructure", "system", "maintain", "operate"]):
            return AgentType.MAGI

        # Educational tasks
        if any(word in task_lower for word in ["teach", "explain", "educate", "tutorial"]):
            return AgentType.TUTOR

        # Translation tasks
        if any(word in task_lower for word in ["translate", "language", "linguistic"]):
            return AgentType.TRANSLATOR

        # Testing tasks
        if any(word in task_lower for word in ["test", "verify", "validate", "check"]):
            return AgentType.TESTER

        # Financial tasks
        if any(word in task_lower for word in ["financial", "money", "economic", "budget"]):
            return AgentType.FINANCIAL

        # Default to coordinator for general tasks
        return AgentType.COORDINATOR

    def _update_session(self, session_id: str, request_id: str, response: dict[str, Any]) -> None:
        """Update session with request/response information"""

        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now().isoformat(),
                "exchanges": [],
                "total_requests": 0,
            }

        session = self.active_sessions[session_id]
        session["total_requests"] += 1
        session["last_activity"] = datetime.now().isoformat()

        # Store exchange information
        exchange = {
            "request_id": request_id,
            "timestamp": response["timestamp"],
            "query": response.get("result", {}).get("message", ""),
            "response": response.get("result", {}).get("response", ""),
            "processing_time_ms": response["processing_time_ms"],
        }

        session["exchanges"].append(exchange)

        # Keep only last 10 exchanges per session
        if len(session["exchanges"]) > 10:
            session["exchanges"] = session["exchanges"][-10:]

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status including gateway metrics"""

        # Get controller status
        controller_status = await self.controller.get_system_performance_report()

        # Add gateway-specific metrics
        gateway_metrics = {
            "active_sessions": len(self.active_sessions),
            "cache_entries": len(self.request_cache),
            "total_session_requests": sum(
                session.get("total_requests", 0) for session in self.active_sessions.values()
            ),
        }

        return {
            "gateway": gateway_metrics,
            "controller": controller_status,
            "overall_status": "operational" if controller_status["system_status"]["initialized"] else "initializing",
        }

    async def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid4())
        self.active_sessions[session_id] = {
            "created_at": datetime.now().isoformat(),
            "exchanges": [],
            "total_requests": 0,
        }
        return session_id

    async def end_session(self, session_id: str) -> bool:
        """End a session and clean up resources"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False


# Factory function for easy integration
async def create_agent_gateway() -> AgentGatewayInterface:
    """
    Create and initialize the complete agent gateway system

    Returns:
        Fully initialized gateway interface ready for requests
    """
    try:
        # Create and initialize controller
        controller = await create_cognative_nexus_controller(enable_cognitive_nexus=False)

        # Create gateway interface
        gateway = AgentGatewayInterface(controller)

        logger.info("✅ Agent gateway system ready for requests")
        return gateway

    except Exception as e:
        logger.error(f"❌ Failed to create agent gateway: {e}")
        raise


# Example usage and integration points
async def example_usage():
    """Example of how to use the gateway interface"""

    # Initialize gateway
    gateway = await create_agent_gateway()

    # Create session
    session_id = await gateway.create_session()

    # Process different types of requests
    requests = [
        {
            "request_type": "query",
            "content": "What are the latest developments in AI?",
            "session_id": session_id,
            "priority": "high",
        },
        {"request_type": "analysis", "content": "Market trends in renewable energy", "priority": "normal"},
        {"request_type": "conversation", "content": "Hello, how are you today?", "session_id": session_id},
    ]

    for request in requests:
        response = await gateway.process_request(**request)
        print(f"Request: {request['content'][:50]}...")
        print(f"Status: {response['status']}")
        print(f"Processing time: {response['processing_time_ms']:.1f}ms")
        print("---")

    # Get system status
    status = await gateway.get_system_status()
    print(f"System status: {status['overall_status']}")

    # End session
    await gateway.end_session(session_id)


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
