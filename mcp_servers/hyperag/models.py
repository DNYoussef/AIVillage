"""HypeRAG MCP Server Model Injection

Provides the interface for agent-specific reasoning models and manages model registry.
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import time
from typing import Any
import uuid

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query execution modes"""
    NORMAL = "normal"
    CREATIVE = "creative"
    REPAIR = "repair"


@dataclass
class QueryPlan:
    """Query planning result from agent reasoning model"""
    query_id: str
    mode: QueryMode
    max_depth: int = 3
    time_budget_ms: int = 2000
    confidence_threshold: float = 0.7
    include_explanations: bool = True
    search_strategies: list[str] = field(default_factory=lambda: ["vector", "ppr"])
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """Knowledge graph node"""
    id: str
    content: str
    node_type: str
    confidence: float = 1.0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Knowledge graph edge"""
    id: str
    source_id: str
    target_id: str
    relation: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""
    nodes: dict[str, Node]
    edges: dict[str, Edge]
    query_context: str | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph"""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph"""
        self.edges[edge.id] = edge

    def get_neighbors(self, node_id: str) -> list[Node]:
        """Get neighboring nodes"""
        neighbors = []
        for edge in self.edges.values():
            if edge.source_id == node_id and edge.target_id in self.nodes:
                neighbors.append(self.nodes[edge.target_id])
            elif edge.target_id == node_id and edge.source_id in self.nodes:
                neighbors.append(self.nodes[edge.source_id])
        return neighbors


@dataclass
class ReasoningStep:
    """Individual reasoning step"""
    step_id: str
    step_type: str
    description: str
    input_data: Any
    output_data: Any
    confidence: float
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of agent reasoning process"""
    answer: str
    confidence: float
    reasoning_steps: list[ReasoningStep]
    sources: list[Node]
    knowledge_graph: KnowledgeGraph | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentReasoningModel(ABC):
    """Abstract base class for agent-specific reasoning models"""

    def __init__(self, agent_id: str, model_name: str, config: dict[str, Any] | None = None):
        self.agent_id = agent_id
        self.model_name = model_name
        self.config = config or {}
        self.model_id = f"{agent_id}_{model_name}_{hashlib.md5(str(config).encode()).hexdigest()[:8]}"
        self.created_at = time.time()
        self.usage_stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "last_used": None
        }

    @abstractmethod
    async def plan_query(self, query: str, context: dict[str, Any] | None = None) -> QueryPlan:
        """Agent-specific query planning

        Args:
            query: The input query string
            context: Optional context information

        Returns:
            QueryPlan with agent-specific planning decisions
        """

    @abstractmethod
    async def construct_knowledge(
        self,
        retrieved: list[Node],
        plan: QueryPlan,
        context: dict[str, Any] | None = None
    ) -> KnowledgeGraph:
        """Agent-specific knowledge construction

        Args:
            retrieved: List of retrieved nodes
            plan: Query plan from planning phase
            context: Optional context information

        Returns:
            KnowledgeGraph constructed according to agent preferences
        """

    @abstractmethod
    async def reason(
        self,
        knowledge: KnowledgeGraph,
        query: str,
        plan: QueryPlan,
        context: dict[str, Any] | None = None
    ) -> ReasoningResult:
        """Agent-specific reasoning over knowledge graph

        Args:
            knowledge: The knowledge graph to reason over
            query: Original query string
            plan: Query plan from planning phase
            context: Optional context information

        Returns:
            ReasoningResult with answer and reasoning trace
        """

    async def warmup(self) -> None:
        """Warm up the model (optional)"""

    async def cleanup(self) -> None:
        """Clean up model resources (optional)"""

    def update_stats(self, processing_time: float, confidence: float) -> None:
        """Update usage statistics"""
        self.usage_stats["queries_processed"] += 1
        self.usage_stats["total_processing_time"] += processing_time
        self.usage_stats["last_used"] = time.time()

        # Update average confidence using exponential moving average
        if self.usage_stats["average_confidence"] == 0.0:
            self.usage_stats["average_confidence"] = confidence
        else:
            alpha = 0.1  # Smoothing factor
            self.usage_stats["average_confidence"] = (
                alpha * confidence + (1 - alpha) * self.usage_stats["average_confidence"]
            )


class DefaultAgentModel(AgentReasoningModel):
    """Default implementation for agents without custom models"""

    async def plan_query(self, query: str, context: dict[str, Any] | None = None) -> QueryPlan:
        """Simple default planning"""
        return QueryPlan(
            query_id=str(uuid.uuid4()),
            mode=QueryMode.NORMAL,
            max_depth=3,
            time_budget_ms=2000,
            confidence_threshold=0.7,
            include_explanations=True,
            search_strategies=["vector", "ppr"],
            metadata={"model": "default", "agent": self.agent_id}
        )

    async def construct_knowledge(
        self,
        retrieved: list[Node],
        plan: QueryPlan,
        context: dict[str, Any] | None = None
    ) -> KnowledgeGraph:
        """Simple knowledge graph construction"""
        kg = KnowledgeGraph(nodes={}, edges={})

        # Add all retrieved nodes
        for node in retrieved:
            kg.add_node(node)

        # Create simple connections based on similarity
        for i, node1 in enumerate(retrieved):
            for j, node2 in enumerate(retrieved[i+1:], i+1):
                # Simple heuristic: connect nodes with similar content
                if self._calculate_similarity(node1.content, node2.content) > 0.5:
                    edge = Edge(
                        id=f"edge_{node1.id}_{node2.id}",
                        source_id=node1.id,
                        target_id=node2.id,
                        relation="related_to",
                        confidence=0.7
                    )
                    kg.add_edge(edge)

        kg.metadata = {"construction_method": "default", "node_count": len(retrieved)}
        return kg

    async def reason(
        self,
        knowledge: KnowledgeGraph,
        query: str,
        plan: QueryPlan,
        context: dict[str, Any] | None = None
    ) -> ReasoningResult:
        """Simple reasoning implementation"""
        # Simple reasoning: combine top nodes by confidence
        sorted_nodes = sorted(knowledge.nodes.values(), key=lambda n: n.confidence, reverse=True)
        top_nodes = sorted_nodes[:3]

        # Generate simple answer
        if top_nodes:
            answer = f"Based on the knowledge graph, {query} relates to: " + \
                    ", ".join([node.content[:100] for node in top_nodes])
            confidence = sum(node.confidence for node in top_nodes) / len(top_nodes)
        else:
            answer = "No relevant information found in the knowledge graph."
            confidence = 0.1

        # Create reasoning steps
        steps = [
            ReasoningStep(
                step_id="step_1",
                step_type="node_ranking",
                description="Ranked nodes by confidence",
                input_data=len(knowledge.nodes),
                output_data=len(top_nodes),
                confidence=confidence,
                duration_ms=10.0
            ),
            ReasoningStep(
                step_id="step_2",
                step_type="answer_generation",
                description="Generated answer from top nodes",
                input_data=top_nodes,
                output_data=answer,
                confidence=confidence,
                duration_ms=5.0
            )
        ]

        return ReasoningResult(
            answer=answer,
            confidence=confidence,
            reasoning_steps=steps,
            sources=top_nodes,
            knowledge_graph=knowledge,
            metadata={"reasoning_method": "default"}
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0.0


class KingAgentModel(AgentReasoningModel):
    """King agent reasoning model with comprehensive planning"""

    async def plan_query(self, query: str, context: dict[str, Any] | None = None) -> QueryPlan:
        """King-specific comprehensive planning"""
        # King agent prefers comprehensive analysis
        plan = QueryPlan(
            query_id=str(uuid.uuid4()),
            mode=QueryMode.NORMAL,
            max_depth=5,  # Deeper search
            time_budget_ms=5000,  # More time
            confidence_threshold=0.8,  # Higher confidence bar
            include_explanations=True,
            search_strategies=["vector", "ppr", "graph_walk"],
            metadata={"model": "king", "comprehensive": True}
        )

        # Check for creative keywords
        creative_keywords = ["creative", "innovative", "novel", "brainstorm", "imagine"]
        if any(keyword in query.lower() for keyword in creative_keywords):
            plan.mode = QueryMode.CREATIVE
            plan.search_strategies.append("divergent")

        return plan


class SageAgentModel(AgentReasoningModel):
    """Sage agent reasoning model with strategic analysis"""

    async def plan_query(self, query: str, context: dict[str, Any] | None = None) -> QueryPlan:
        """Sage-specific strategic planning"""
        plan = QueryPlan(
            query_id=str(uuid.uuid4()),
            mode=QueryMode.NORMAL,
            max_depth=4,
            time_budget_ms=3000,
            confidence_threshold=0.75,
            include_explanations=True,
            search_strategies=["vector", "ppr", "semantic_expansion"],
            metadata={"model": "sage", "strategic": True}
        )

        # Sage prefers analytical approaches
        if any(word in query.lower() for word in ["analyze", "strategy", "approach", "method"]):
            plan.search_strategies.append("analytical")
            plan.constraints["prefer_analysis"] = True

        return plan


class MagiAgentModel(AgentReasoningModel):
    """Magi agent reasoning model focused on technical/development queries"""

    async def plan_query(self, query: str, context: dict[str, Any] | None = None) -> QueryPlan:
        """Magi-specific technical planning"""
        plan = QueryPlan(
            query_id=str(uuid.uuid4()),
            mode=QueryMode.NORMAL,
            max_depth=3,
            time_budget_ms=2000,
            confidence_threshold=0.7,
            include_explanations=True,
            search_strategies=["vector", "code_search"],
            metadata={"model": "magi", "technical": True}
        )

        # Technical queries get special handling
        technical_keywords = ["code", "implementation", "algorithm", "debug", "api"]
        if any(keyword in query.lower() for keyword in technical_keywords):
            plan.constraints["domain_filter"] = "technical"
            plan.search_strategies.append("technical_docs")

        return plan


class ModelRegistry:
    """Registry for managing agent reasoning models"""

    def __init__(self):
        self.models: dict[str, AgentReasoningModel] = {}
        self.model_classes: dict[str, type[AgentReasoningModel]] = {
            "default": DefaultAgentModel,
            "king": KingAgentModel,
            "sage": SageAgentModel,
            "magi": MagiAgentModel
        }
        self.locks: dict[str, asyncio.Lock] = {}

    async def register_model(
        self,
        agent_id: str,
        model: AgentReasoningModel
    ) -> None:
        """Register a model for an agent"""
        await model.warmup()
        self.models[agent_id] = model
        self.locks[agent_id] = asyncio.Lock()
        logger.info(f"Registered model {model.model_name} for agent {agent_id}")

    async def register_model_class(
        self,
        agent_type: str,
        model_class: type[AgentReasoningModel]
    ) -> None:
        """Register a model class for an agent type"""
        self.model_classes[agent_type] = model_class
        logger.info(f"Registered model class for agent type {agent_type}")

    async def get_model(self, agent_id: str, agent_type: str = "default") -> AgentReasoningModel:
        """Get model for an agent, creating if necessary"""
        if agent_id not in self.models:
            # Create model based on agent type
            model_class = self.model_classes.get(agent_type, DefaultAgentModel)
            model = model_class(agent_id=agent_id, model_name=agent_type)
            await self.register_model(agent_id, model)

        return self.models[agent_id]

    async def remove_model(self, agent_id: str) -> None:
        """Remove model for an agent"""
        if agent_id in self.models:
            model = self.models[agent_id]
            await model.cleanup()
            del self.models[agent_id]
            if agent_id in self.locks:
                del self.locks[agent_id]
            logger.info(f"Removed model for agent {agent_id}")

    async def process_with_model(
        self,
        agent_id: str,
        agent_type: str,
        operation: str,
        *args,
        **kwargs
    ) -> Any:
        """Process operation with agent's model under lock"""
        model = await self.get_model(agent_id, agent_type)

        if agent_id not in self.locks:
            self.locks[agent_id] = asyncio.Lock()

        async with self.locks[agent_id]:
            start_time = time.time()
            try:
                if operation == "plan":
                    result = await model.plan_query(*args, **kwargs)
                elif operation == "construct":
                    result = await model.construct_knowledge(*args, **kwargs)
                elif operation == "reason":
                    result = await model.reason(*args, **kwargs)
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                # Update stats
                processing_time = time.time() - start_time
                confidence = getattr(result, "confidence", 1.0)
                model.update_stats(processing_time, confidence)

                return result

            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"Model operation {operation} failed for {agent_id}: {e!s}")
                raise

    def get_model_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all models"""
        stats = {}
        for agent_id, model in self.models.items():
            stats[agent_id] = {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "created_at": model.created_at,
                "usage_stats": model.usage_stats.copy()
            }
        return stats

    async def cleanup_all(self) -> None:
        """Clean up all models"""
        for agent_id in list(self.models.keys()):
            await self.remove_model(agent_id)
        logger.info("Cleaned up all models")
