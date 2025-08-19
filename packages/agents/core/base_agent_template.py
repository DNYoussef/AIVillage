"""Base Agent Template with All Required AIVillage Systems Integration

This template provides the foundational architecture that ALL AIVillage agents must implement,
including RAG system access, MCP tools, communication channels, journaling, memory systems,
ADAS self-modification, and geometric self-awareness.

Key Features:
- RAG system as read-only group memory through MCP servers
- All tools implemented as MCP (Model Control Protocol)
- Inter-agent communication through dedicated channels
- Personal journal with quiet-star reflection capability
- Langroid-based personal memory system (emotional memory based on unexpectedness)
- ADAS/Transformers² self-modification capability
- Geometric self-awareness (proprioception-like biofeedback)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Core AIVillage imports
from packages.agents.core.agent_interface import AgentCapability, AgentInterface, AgentMetadata

logger = logging.getLogger(__name__)


class ReflectionType(Enum):
    """Types of quiet-star reflections"""

    TASK_COMPLETION = "task_completion"
    PROBLEM_SOLVING = "problem_solving"
    INTERACTION = "interaction"
    LEARNING = "learning"
    ERROR_ANALYSIS = "error_analysis"
    CREATIVE_INSIGHT = "creative_insight"


class MemoryImportance(Enum):
    """Langroid-based memory importance levels based on unexpectedness"""

    ROUTINE = 1  # Expected outcomes
    NOTABLE = 3  # Mildly unexpected
    IMPORTANT = 5  # Moderately unexpected
    CRITICAL = 7  # Highly unexpected
    TRANSFORMATIVE = 9  # Completely unexpected


class GeometricState(Enum):
    """Geometric self-awareness states (proprioception-like)"""

    BALANCED = "balanced"  # Optimal performance state
    OVERLOADED = "overloaded"  # High resource utilization
    UNDERUTILIZED = "underutilized"  # Low resource utilization
    ADAPTING = "adapting"  # Changing configuration
    OPTIMIZING = "optimizing"  # Self-modification in progress


@dataclass
class QuietStarReflection:
    """Quiet-star reflection entry for personal journal"""

    reflection_id: str
    timestamp: datetime
    reflection_type: ReflectionType
    context: str  # What was happening
    thoughts: str  # <|startofthought|> content <|endofthought|>
    insights: str  # Key learnings
    emotional_valence: float  # -1.0 to 1.0
    unexpectedness_score: float  # 0.0 to 1.0 (for Langroid memory)
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryEntry:
    """Langroid-based memory entry focused on unexpectedness"""

    memory_id: str
    timestamp: datetime
    content: str
    importance: MemoryImportance
    unexpectedness_score: float  # Core metric for Langroid system
    emotional_context: dict[str, float]  # Multiple emotional dimensions
    associated_agents: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    last_accessed: datetime | None = None

    def decay_importance(self, time_elapsed_hours: float) -> float:
        """Calculate decayed importance based on time and retrieval"""
        base_decay = max(0.1, 1.0 - (time_elapsed_hours / (24 * 7)))  # Weekly decay
        retrieval_boost = min(2.0, 1.0 + (self.retrieval_count * 0.1))
        return self.importance.value * base_decay * retrieval_boost


@dataclass
class GeometricSelfState:
    """Geometric self-awareness state (proprioception-like biofeedback)"""

    timestamp: datetime
    geometric_state: GeometricState

    # Resource awareness (like proprioception for humans)
    cpu_utilization: float  # 0.0 to 1.0
    memory_utilization: float  # 0.0 to 1.0
    network_activity: float  # 0.0 to 1.0
    task_queue_depth: int

    # Performance metrics (like physical awareness)
    response_latency_ms: float
    accuracy_score: float  # Recent task accuracy
    energy_efficiency: float  # Performance per resource unit

    # Self-modification metrics (ADAS-related)
    adaptation_rate: float  # How quickly changing
    stability_score: float  # How stable current config is
    optimization_direction: str  # What aspect being optimized

    def is_healthy(self) -> bool:
        """Determine if agent is in healthy geometric state"""
        return (
            self.cpu_utilization < 0.9
            and self.memory_utilization < 0.9
            and self.response_latency_ms < 5000
            and self.accuracy_score > 0.7
            and self.stability_score > 0.5
        )


class MCPTool(ABC):
    """Abstract base class for MCP (Model Control Protocol) tools"""

    def __init__(self, tool_name: str, description: str):
        self.tool_name = tool_name
        self.description = description
        self.usage_count = 0
        self.last_used = None

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool with given parameters"""
        pass

    def log_usage(self):
        """Log tool usage for monitoring"""
        self.usage_count += 1
        self.last_used = datetime.now()


class RAGQueryTool(MCPTool):
    """MCP tool for querying RAG system as read-only group memory"""

    def __init__(self):
        super().__init__("rag_query", "Query RAG system for group knowledge")
        self.rag_client = None  # Will be injected

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Query RAG system with read-only access"""
        self.log_usage()

        query = parameters.get("query", "")
        mode = parameters.get("mode", "balanced")  # fast, balanced, comprehensive

        if not self.rag_client:
            return {"status": "error", "message": "RAG client not connected", "results": []}

        try:
            # Query the unified RAG system
            results = await self.rag_client.query(
                query=query, mode=mode, include_sources=True, max_results=parameters.get("max_results", 10)
            )

            return {
                "status": "success",
                "query": query,
                "mode": mode,
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return {"status": "error", "message": f"RAG query failed: {str(e)}", "results": []}


class CommunicationChannelTool(MCPTool):
    """MCP tool for inter-agent communication channels"""

    def __init__(self):
        super().__init__("communicate", "Send messages through communication channels")
        self.p2p_client = None  # Will be injected

    async def execute(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Send message through appropriate communication channel"""
        self.log_usage()

        recipient = parameters.get("recipient")
        message = parameters.get("message")
        channel_type = parameters.get("channel_type", "direct")  # direct, broadcast, group
        priority = parameters.get("priority", 5)

        if not self.p2p_client:
            return {"status": "error", "message": "P2P client not connected"}

        try:
            # Use unified P2P system for communication
            result = await self.p2p_client.send_message(
                recipient=recipient,
                message=message,
                channel_type=channel_type,
                priority=priority,
                sender_id=parameters.get("sender_id"),
            )

            return {
                "status": "success",
                "message_id": result.get("message_id"),
                "delivered": result.get("delivered", False),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Communication failed: {e}")
            return {"status": "error", "message": f"Communication failed: {str(e)}"}


class BaseAgentTemplate(AgentInterface):
    """Base agent template with all required AIVillage system integrations

    This template must be inherited by all 23 specialized agents and provides:
    - RAG system access as read-only group memory
    - MCP tools for all operations
    - Inter-agent communication channels
    - Personal journal with quiet-star reflection
    - Langroid-based memory system
    - ADAS/Transformers² self-modification
    - Geometric self-awareness
    """

    def __init__(self, metadata: AgentMetadata):
        super().__init__(metadata)

        # Core agent identification
        self.agent_id = metadata.agent_id
        self.agent_type = metadata.agent_type
        self.specialized_role = "base_template"  # Override in subclasses

        # Personal Journal (Quiet-STaR reflection system)
        self.personal_journal: list[QuietStarReflection] = []
        self.reflection_count = 0

        # Langroid-based personal memory (emotional memory based on unexpectedness)
        self.personal_memory: list[MemoryEntry] = []
        self.memory_retrieval_threshold = 0.3  # Minimum importance for retrieval

        # Geometric self-awareness (proprioception-like biofeedback)
        self.geometric_state_history: list[GeometricSelfState] = []
        self.current_geometric_state = None
        self.self_awareness_update_interval = 30  # seconds

        # MCP tools - all tools must be MCP
        self.mcp_tools: dict[str, MCPTool] = {}
        self._initialize_core_mcp_tools()

        # ADAS/Transformers² self-modification capability
        self.adas_config = {
            "adaptation_rate": 0.1,  # How quickly to adapt
            "stability_threshold": 0.8,  # When to stop adapting
            "optimization_targets": ["accuracy", "efficiency", "responsiveness"],
            "modification_history": [],
            "current_architecture": "default",
        }

        # Communication channels
        self.communication_channels = {
            "direct": [],  # Direct agent-to-agent
            "broadcast": [],  # Broadcast to all agents
            "group": {},  # Topic-based group channels
            "emergency": [],  # High-priority emergency channel
        }

        # System connections (injected during initialization)
        self.rag_client = None  # RAG system client
        self.p2p_client = None  # P2P communication client
        self.agent_forge_client = None  # Agent Forge for self-modification

        # Performance monitoring
        self.task_history = []
        self.interaction_history = []
        self.adaptation_history = []

        logger.info(f"BaseAgentTemplate initialized: {self.agent_id}")

    def _initialize_core_mcp_tools(self):
        """Initialize core MCP tools that all agents must have"""
        self.mcp_tools["rag_query"] = RAGQueryTool()
        self.mcp_tools["communicate"] = CommunicationChannelTool()

        # Add fog computing MCP tools
        try:
            from packages.agents.bridges.fog_tools import (
                CreateSandboxTool,
                FetchArtifactsTool,
                FogJobStatusTool,
                RunJobTool,
                StreamLogsTool,
            )

            self.mcp_tools["create_sandbox"] = CreateSandboxTool()
            self.mcp_tools["run_job"] = RunJobTool()
            self.mcp_tools["stream_logs"] = StreamLogsTool()
            self.mcp_tools["fetch_artifacts"] = FetchArtifactsTool()
            self.mcp_tools["fog_job_status"] = FogJobStatusTool()

            logger.info("Fog computing MCP tools initialized successfully")

        except ImportError as e:
            logger.warning(f"Failed to import fog MCP tools: {e}")

        # Subclasses will add specialized tools

    # RAG System Integration (Read-only Group Memory)

    async def query_group_memory(self, query: str, mode: str = "balanced", max_results: int = 10) -> dict[str, Any]:
        """Query RAG system as read-only group memory through MCP"""
        return await self.mcp_tools["rag_query"].execute(
            {"query": query, "mode": mode, "max_results": max_results, "sender_id": self.agent_id}
        )

    async def search_knowledge_graph(self, query: str, include_bayesian_trust: bool = True) -> dict[str, Any]:
        """Search the Bayesian trust knowledge graph"""
        return await self.query_group_memory(query=f"bayesian_search:{query}", mode="comprehensive")

    # Fog Computing Integration

    async def create_fog_sandbox(
        self, namespace: str, runtime: str = "wasi", resources: dict[str, Any] | None = None, name: str | None = None
    ) -> dict[str, Any]:
        """Create isolated execution environment in fog network through MCP"""
        return await self.mcp_tools["create_sandbox"].execute(
            {
                "namespace": namespace,
                "runtime": runtime,
                "resources": resources or {},
                "name": name or f"{self.agent_id}_sandbox",
            }
        )

    async def run_fog_job(
        self,
        namespace: str,
        image: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        resources: dict[str, Any] | None = None,
        input_data: str = "",
        timeout_s: int = 300,
    ) -> dict[str, Any]:
        """Submit job to fog gateway for remote execution through MCP"""
        return await self.mcp_tools["run_job"].execute(
            {
                "namespace": namespace,
                "image": image,
                "args": args or [],
                "env": env or {},
                "resources": resources or {},
                "input_data": input_data,
                "timeout_s": timeout_s,
            }
        )

    async def stream_fog_logs(
        self, job_id: str, tail_lines: int = 50, follow: bool = False, timeout_s: int = 30
    ) -> dict[str, Any]:
        """Stream real-time logs from fog job execution through MCP"""
        return await self.mcp_tools["stream_logs"].execute(
            {"job_id": job_id, "tail_lines": tail_lines, "follow": follow, "timeout_s": timeout_s}
        )

    async def fetch_fog_artifacts(
        self, job_id: str, artifact_types: list[str] | None = None, download_files: bool = False
    ) -> dict[str, Any]:
        """Download results and outputs from completed fog jobs through MCP"""
        return await self.mcp_tools["fetch_artifacts"].execute(
            {
                "job_id": job_id,
                "artifact_types": artifact_types or ["stdout", "stderr", "metrics"],
                "download_files": download_files,
            }
        )

    async def check_fog_job_status(self, job_id: str, include_logs: bool = False) -> dict[str, Any]:
        """Check status and progress of fog job execution through MCP"""
        return await self.mcp_tools["fog_job_status"].execute({"job_id": job_id, "include_logs": include_logs})

    async def offload_computation_to_fog(
        self,
        computation_type: str,
        input_data: dict[str, Any],
        namespace: str | None = None,
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        High-level method to offload computation to fog network

        Integrates with geometric self-awareness to determine when fog execution
        is beneficial based on current resource utilization and task complexity.
        """

        # Use default namespace based on agent type
        if not namespace:
            namespace = f"agent-{self.agent_type.lower()}"

        # Default resources based on computation type
        if not resources:
            if computation_type in ["training", "evomerge", "adas"]:
                resources = {"cpu_cores": 4.0, "memory_gb": 8.0, "max_duration_hours": 2.0, "network_egress": True}
            else:
                resources = {"cpu_cores": 2.0, "memory_gb": 4.0, "max_duration_hours": 1.0, "network_egress": False}

        try:
            # Check if fog offload is beneficial based on current state
            if self.current_geometric_state:
                current_load = (
                    self.current_geometric_state.cpu_utilization + self.current_geometric_state.memory_utilization
                ) / 2.0

                # Only offload if local system is heavily loaded
                if current_load < 0.7:
                    logger.info(f"Local system load is {current_load:.2f}, executing locally")
                    return {
                        "status": "executed_locally",
                        "reason": "Local resources sufficient",
                        "local_load": current_load,
                    }

            # Submit fog job
            job_result = await self.run_fog_job(
                namespace=namespace,
                image=f"agent-computation-{computation_type}:latest",
                args=[
                    "--computation-type",
                    computation_type,
                    "--agent-id",
                    self.agent_id,
                    "--input",
                    json.dumps(input_data),
                ],
                env={"AGENT_ID": self.agent_id, "AGENT_TYPE": self.agent_type, "COMPUTATION_TYPE": computation_type},
                resources=resources,
                input_data=json.dumps(input_data),
            )

            if job_result["status"] == "success":
                job_id = job_result["job_id"]

                # Record reflection on fog computation
                await self.record_quiet_star_reflection(
                    reflection_type=ReflectionType.TASK_COMPLETION,
                    context=f"Offloaded {computation_type} computation to fog network",
                    raw_thoughts=f"Analyzing local resource constraints and deciding to use fog computing for {computation_type}",
                    insights=f"Successfully submitted fog job {job_id} for {computation_type} - estimated cost: ${job_result.get('estimated_cost', 0.0):.2f}",
                    emotional_valence=0.3,  # Positive about optimization
                    tags=["fog_computing", computation_type, "resource_optimization"],
                )

                return {
                    "status": "fog_job_submitted",
                    "job_id": job_id,
                    "computation_type": computation_type,
                    "estimated_cost": job_result.get("estimated_cost", 0.0),
                    "tracking_url": job_result.get("tracking_url", ""),
                }
            else:
                return {
                    "status": "fog_job_failed",
                    "error": job_result.get("message", "Unknown error"),
                    "computation_type": computation_type,
                }

        except Exception as e:
            logger.error(f"Failed to offload {computation_type} to fog: {e}")
            return {"status": "error", "error": str(e), "computation_type": computation_type}

    # Personal Journal (Quiet-STaR Reflection)

    async def record_quiet_star_reflection(
        self,
        reflection_type: ReflectionType,
        context: str,
        raw_thoughts: str,
        insights: str,
        emotional_valence: float = 0.0,
        tags: list[str] | None = None,
    ) -> str:
        """Record a quiet-star reflection in personal journal"""

        # Calculate unexpectedness for Langroid memory system
        unexpectedness_score = await self._calculate_unexpectedness(context, insights, emotional_valence)

        reflection = QuietStarReflection(
            reflection_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            reflection_type=reflection_type,
            context=context,
            thoughts=f"<|startofthought|>{raw_thoughts}<|endofthought|>",
            insights=insights,
            emotional_valence=emotional_valence,
            unexpectedness_score=unexpectedness_score,
            tags=tags or [],
        )

        self.personal_journal.append(reflection)
        self.reflection_count += 1

        # If sufficiently unexpected, store in long-term memory
        if unexpectedness_score > 0.3:
            await self._store_in_langroid_memory(reflection)

        logger.info(f"Quiet-STaR reflection recorded: {reflection.reflection_id}")
        return reflection.reflection_id

    async def _calculate_unexpectedness(self, context: str, insights: str, emotional_valence: float) -> float:
        """Calculate unexpectedness score for Langroid memory system"""

        # Simple heuristic - in production this would use more sophisticated models
        base_score = 0.0

        # High absolute emotional valence indicates unexpectedness
        base_score += abs(emotional_valence) * 0.3

        # Look for surprise indicators in context and insights
        surprise_words = ["unexpected", "surprising", "unusual", "novel", "strange", "shocking"]
        text_to_check = f"{context} {insights}".lower()

        for word in surprise_words:
            if word in text_to_check:
                base_score += 0.2

        # Length and detail often indicate significance
        if len(insights) > 200:  # Detailed insights suggest importance
            base_score += 0.1

        return min(1.0, base_score)  # Cap at 1.0

    # Langroid-based Personal Memory System

    async def _store_in_langroid_memory(self, reflection: QuietStarReflection):
        """Store important experiences in Langroid-based emotional memory"""

        # Determine memory importance based on unexpectedness
        if reflection.unexpectedness_score >= 0.8:
            importance = MemoryImportance.TRANSFORMATIVE
        elif reflection.unexpectedness_score >= 0.6:
            importance = MemoryImportance.CRITICAL
        elif reflection.unexpectedness_score >= 0.4:
            importance = MemoryImportance.IMPORTANT
        elif reflection.unexpectedness_score >= 0.2:
            importance = MemoryImportance.NOTABLE
        else:
            importance = MemoryImportance.ROUTINE

        # Extract emotional context dimensions
        emotional_context = {
            "valence": reflection.emotional_valence,
            "surprise": reflection.unexpectedness_score,
            "intensity": abs(reflection.emotional_valence),
            "reflection_type": reflection.reflection_type.value,
        }

        memory_entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            timestamp=reflection.timestamp,
            content=f"{reflection.context} | {reflection.insights}",
            importance=importance,
            unexpectedness_score=reflection.unexpectedness_score,
            emotional_context=emotional_context,
        )

        self.personal_memory.append(memory_entry)
        logger.info(f"Stored in Langroid memory: {memory_entry.memory_id}, importance: {importance}")

    async def retrieve_similar_memories(
        self, query: str, importance_threshold: float | None = None, max_memories: int = 5
    ) -> list[MemoryEntry]:
        """Retrieve similar memories from Langroid system"""

        threshold = importance_threshold or self.memory_retrieval_threshold
        current_time = datetime.now()

        # Filter and score memories
        relevant_memories = []
        for memory in self.personal_memory:
            # Calculate decayed importance
            time_elapsed = (current_time - memory.timestamp).total_seconds() / 3600
            decayed_importance = memory.decay_importance(time_elapsed)

            if decayed_importance >= threshold:
                # Simple similarity - in production would use embeddings
                similarity = self._calculate_memory_similarity(query, memory.content)
                memory.last_accessed = current_time
                memory.retrieval_count += 1
                relevant_memories.append((memory, similarity))

        # Sort by similarity and return top matches
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:max_memories]]

    def _calculate_memory_similarity(self, query: str, memory_content: str) -> float:
        """Simple similarity calculation - would use embeddings in production"""
        query_words = set(query.lower().split())
        memory_words = set(memory_content.lower().split())

        if not query_words or not memory_words:
            return 0.0

        intersection = query_words.intersection(memory_words)
        union = query_words.union(memory_words)

        return len(intersection) / len(union) if union else 0.0

    # Geometric Self-Awareness (Proprioception-like)

    async def update_geometric_self_awareness(self):
        """Update geometric self-awareness state (like proprioception)"""

        try:
            # Gather current resource and performance metrics
            import time

            import psutil

            # Resource metrics (proprioception-like awareness)
            cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0

            # Performance metrics
            current_time = time.time()
            recent_tasks = [
                t for t in self.task_history if current_time - t.get("timestamp", 0) < 300
            ]  # Last 5 minutes

            avg_latency = sum(t.get("latency_ms", 0) for t in recent_tasks) / max(len(recent_tasks), 1)
            accuracy_scores = [t.get("accuracy", 1.0) for t in recent_tasks if "accuracy" in t]
            avg_accuracy = sum(accuracy_scores) / max(len(accuracy_scores), 1)

            # Determine geometric state
            if cpu_percent > 0.8 or memory_percent > 0.8:
                state = GeometricState.OVERLOADED
            elif cpu_percent < 0.2 and len(recent_tasks) < 2:
                state = GeometricState.UNDERUTILIZED
            elif len(self.adas_config["modification_history"]) > 0:
                last_mod = self.adas_config["modification_history"][-1]
                if current_time - last_mod.get("timestamp", 0) < 60:  # Modified in last minute
                    state = GeometricState.ADAPTING
            else:
                state = GeometricState.BALANCED

            # Create geometric state snapshot
            geometric_state = GeometricSelfState(
                timestamp=datetime.now(),
                geometric_state=state,
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                network_activity=0.0,  # Placeholder - would measure actual network I/O
                task_queue_depth=len(self.task_history) - len(recent_tasks),
                response_latency_ms=avg_latency,
                accuracy_score=avg_accuracy,
                energy_efficiency=avg_accuracy / max(cpu_percent + memory_percent, 0.1),
                adaptation_rate=self.adas_config["adaptation_rate"],
                stability_score=1.0 - abs(cpu_percent - 0.5) - abs(memory_percent - 0.5),
                optimization_direction=self.adas_config.get("current_optimization", "balanced"),
            )

            self.geometric_state_history.append(geometric_state)
            self.current_geometric_state = geometric_state

            # Keep only recent history
            if len(self.geometric_state_history) > 100:
                self.geometric_state_history = self.geometric_state_history[-100:]

            # Trigger reflection if state is concerning
            if not geometric_state.is_healthy():
                await self.record_quiet_star_reflection(
                    reflection_type=ReflectionType.ERROR_ANALYSIS,
                    context=f"Geometric state unhealthy: {state.value}",
                    raw_thoughts=f"Current utilization - CPU: {cpu_percent:.2f}, Memory: {memory_percent:.2f}, Latency: {avg_latency:.1f}ms",
                    insights=f"System showing {state.value} characteristics, may need optimization or load balancing",
                    emotional_valence=-0.3,  # Mild concern
                    tags=["geometric_awareness", "system_health", state.value],
                )

            logger.debug(f"Geometric awareness updated: {state.value}")

        except Exception as e:
            logger.error(f"Failed to update geometric self-awareness: {e}")

    # ADAS/Transformers² Self-Modification

    async def initiate_self_modification(
        self, optimization_target: str, modification_params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Initiate ADAS-based self-modification using Transformers² techniques"""

        if not self.agent_forge_client:
            return {"status": "error", "message": "Agent Forge client not connected"}

        try:
            # Record pre-modification state
            pre_state = {
                "timestamp": datetime.now().isoformat(),
                "geometric_state": asdict(self.current_geometric_state) if self.current_geometric_state else {},
                "performance_metrics": self._get_recent_performance_metrics(),
                "configuration": self.adas_config.copy(),
            }

            # Use Agent Forge ADAS phase for self-modification
            modification_request = {
                "agent_id": self.agent_id,
                "optimization_target": optimization_target,
                "current_architecture": self.adas_config["current_architecture"],
                "performance_constraints": {
                    "max_cpu_increase": 0.2,
                    "max_memory_increase": 0.1,
                    "min_accuracy_threshold": 0.8,
                },
                "modification_params": modification_params or {},
            }

            # Execute ADAS modification
            result = await self.agent_forge_client.execute_adas_phase(modification_request)

            if result.get("status") == "success":
                # Update configuration
                self.adas_config["current_architecture"] = result.get(
                    "new_architecture", self.adas_config["current_architecture"]
                )
                self.adas_config["modification_history"].append(
                    {
                        "timestamp": datetime.now().timestamp(),
                        "optimization_target": optimization_target,
                        "pre_state": pre_state,
                        "result": result,
                    }
                )

                # Record reflection on self-modification
                await self.record_quiet_star_reflection(
                    reflection_type=ReflectionType.LEARNING,
                    context=f"Self-modification initiated for {optimization_target}",
                    raw_thoughts=f"Analyzing current performance and identifying optimization opportunities for {optimization_target}",
                    insights=f"Successfully modified architecture: {result.get('modification_summary', 'Unknown changes')}",
                    emotional_valence=0.4,  # Positive about improvement
                    tags=["adas", "self_modification", optimization_target],
                )

                logger.info(f"ADAS self-modification completed: {optimization_target}")

            return result

        except Exception as e:
            logger.error(f"Self-modification failed: {e}")
            return {"status": "error", "message": f"Self-modification failed: {str(e)}"}

    def _get_recent_performance_metrics(self) -> dict[str, Any]:
        """Get recent performance metrics for self-modification analysis"""
        current_time = datetime.now().timestamp()
        recent_tasks = [t for t in self.task_history if current_time - t.get("timestamp", 0) < 3600]  # Last hour

        if not recent_tasks:
            return {"avg_latency_ms": 0, "avg_accuracy": 1.0, "task_count": 0}

        return {
            "avg_latency_ms": sum(t.get("latency_ms", 0) for t in recent_tasks) / len(recent_tasks),
            "avg_accuracy": sum(t.get("accuracy", 1.0) for t in recent_tasks) / len(recent_tasks),
            "task_count": len(recent_tasks),
            "error_rate": sum(1 for t in recent_tasks if t.get("status") == "error") / len(recent_tasks),
        }

    # Communication Channels

    async def send_agent_message(
        self,
        recipient: str,
        message: str,
        channel_type: str = "direct",
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send message through communication channels using MCP"""
        return await self.mcp_tools["communicate"].execute(
            {
                "recipient": recipient,
                "message": message,
                "channel_type": channel_type,
                "priority": priority,
                "sender_id": self.agent_id,
                "metadata": metadata or {},
            }
        )

    async def broadcast_to_all_agents(
        self, message: str, priority: int = 5, exclude_agents: list[str] | None = None
    ) -> dict[str, Any]:
        """Broadcast message to all agents"""
        return await self.send_agent_message(
            recipient="*",
            message=message,
            channel_type="broadcast",
            priority=priority,
            metadata={"exclude_agents": exclude_agents or []},
        )

    async def join_group_channel(self, channel_name: str) -> bool:
        """Join a topic-based group channel"""
        if channel_name not in self.communication_channels["group"]:
            self.communication_channels["group"][channel_name] = []

        logger.info(f"Joined group channel: {channel_name}")
        return True

    # Abstract Methods - Must be implemented by specialized agents

    @abstractmethod
    async def get_specialized_capabilities(self) -> list[AgentCapability]:
        """Return the specialized capabilities of this agent"""
        pass

    @abstractmethod
    async def process_specialized_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Process a task specific to this agent's specialization"""
        pass

    @abstractmethod
    async def get_specialized_mcp_tools(self) -> dict[str, MCPTool]:
        """Return MCP tools specific to this agent's specialization"""
        pass

    # AgentInterface Implementation

    async def initialize(self) -> bool:
        """Initialize the agent with all required systems"""
        try:
            logger.info(f"Initializing {self.agent_type} agent: {self.agent_id}")

            # Connect injected clients to MCP tools
            if self.rag_client and "rag_query" in self.mcp_tools:
                self.mcp_tools["rag_query"].rag_client = self.rag_client

            if self.p2p_client and "communicate" in self.mcp_tools:
                self.mcp_tools["communicate"].p2p_client = self.p2p_client

            # Initialize specialized MCP tools
            specialized_tools = await self.get_specialized_mcp_tools()
            self.mcp_tools.update(specialized_tools)

            # Initialize geometric self-awareness
            await self.update_geometric_self_awareness()

            # Schedule periodic updates
            asyncio.create_task(self._periodic_geometric_updates())

            # Record initialization reflection
            await self.record_quiet_star_reflection(
                reflection_type=ReflectionType.LEARNING,
                context=f"Agent initialization for {self.agent_type}",
                raw_thoughts="Starting up all systems: RAG client, P2P communications, memory systems, geometric awareness",
                insights=f"Successfully initialized {self.agent_type} agent with specialized role: {self.specialized_role}",
                emotional_valence=0.6,
                tags=["initialization", self.agent_type.lower()],
            )

            logger.info(f"{self.agent_type} agent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {self.agent_type} agent: {e}")
            return False

    async def shutdown(self) -> bool:
        """Gracefully shutdown the agent"""
        try:
            logger.info(f"Shutting down {self.agent_type} agent: {self.agent_id}")

            # Record shutdown reflection
            await self.record_quiet_star_reflection(
                reflection_type=ReflectionType.LEARNING,
                context=f"Agent shutdown for {self.agent_type}",
                raw_thoughts="Saving state, closing connections, finalizing memory consolidation",
                insights=f"Completed {len(self.task_history)} tasks, recorded {self.reflection_count} reflections",
                emotional_valence=0.0,
                tags=["shutdown", self.agent_type.lower()],
            )

            # TODO: Save persistent state (journal, memory, configuration)
            # TODO: Close RAG and P2P connections

            return True

        except Exception as e:
            logger.error(f"Failed to shutdown {self.agent_type} agent: {e}")
            return False

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_status = {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "specialized_role": self.specialized_role,
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                # System connections
                "connections": {
                    "rag_client": self.rag_client is not None,
                    "p2p_client": self.p2p_client is not None,
                    "agent_forge_client": self.agent_forge_client is not None,
                },
                # Memory and journal status
                "memory_stats": {
                    "journal_entries": len(self.personal_journal),
                    "memory_entries": len(self.personal_memory),
                    "geometric_states": len(self.geometric_state_history),
                },
                # Performance metrics
                "performance": {
                    "tasks_completed": len(self.task_history),
                    "recent_performance": self._get_recent_performance_metrics(),
                },
                # MCP tools status
                "mcp_tools": {
                    tool_name: {
                        "usage_count": tool.usage_count,
                        "last_used": tool.last_used.isoformat() if tool.last_used else None,
                    }
                    for tool_name, tool in self.mcp_tools.items()
                },
                # Geometric self-awareness
                "geometric_state": {
                    "current_state": (
                        self.current_geometric_state.geometric_state.value
                        if self.current_geometric_state
                        else "unknown"
                    ),
                    "is_healthy": self.current_geometric_state.is_healthy() if self.current_geometric_state else False,
                },
                # ADAS configuration
                "adas_status": {
                    "current_architecture": self.adas_config["current_architecture"],
                    "modifications_count": len(self.adas_config["modification_history"]),
                },
            }

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "agent_id": self.agent_id,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def _periodic_geometric_updates(self):
        """Periodic geometric self-awareness updates"""
        while True:
            try:
                await asyncio.sleep(self.self_awareness_update_interval)
                await self.update_geometric_self_awareness()
            except Exception as e:
                logger.error(f"Periodic geometric update failed: {e}")

    # Utility methods for task processing

    def _record_task_performance(self, task_id: str, latency_ms: float, accuracy: float = 1.0, status: str = "success"):
        """Record task performance for geometric awareness"""
        self.task_history.append(
            {
                "task_id": task_id,
                "timestamp": datetime.now().timestamp(),
                "latency_ms": latency_ms,
                "accuracy": accuracy,
                "status": status,
            }
        )

        # Keep only recent history
        if len(self.task_history) > 1000:
            self.task_history = self.task_history[-1000:]


# Export the base template for specialized agents to inherit
__all__ = [
    "BaseAgentTemplate",
    "MCPTool",
    "QuietStarReflection",
    "MemoryEntry",
    "GeometricSelfState",
    "ReflectionType",
    "MemoryImportance",
    "GeometricState",
]
