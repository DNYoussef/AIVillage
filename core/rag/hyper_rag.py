"""
HyperRAG - Unified RAG System Orchestrator

Coordinates all RAG subsystems:
- HippoRAG (neurobiological episodic memory)
- GraphRAG (Bayesian trust networks)
- VectorRAG (contextual similarity search)
- Cognitive Nexus (analysis and reasoning)
- Database integration (fog computing)
- Edge device integration (mobile optimization)
- P2P network integration (distributed knowledge)

This is the main entry point for the unified RAG system.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Import core components with fallback for missing dependencies
try:
    from ..analysis.graph_fixer import GraphFixer
except ImportError:
    GraphFixer = None

try:
    from ..creativity.insight_engine import CreativityEngine
except ImportError:
    CreativityEngine = None

try:
    from ..graph.bayesian_trust_graph import BayesianTrustGraph
except ImportError:
    BayesianTrustGraph = None

try:
    from ..memory.hippo_index import EpisodicDocument, HippoIndex, create_hippo_node
except ImportError:
    EpisodicDocument = None
    HippoIndex = None
    create_hippo_node = None

try:
    from ..vector.contextual_vector_engine import ContextualVectorEngine
except ImportError:
    ContextualVectorEngine = None

try:
    from .cognitive_nexus import CognitiveNexus, RetrievedInformation, SynthesizedAnswer
except ImportError:
    CognitiveNexus = None

    # Fallback implementations for core data structures
    from dataclasses import dataclass

    @dataclass
    class RetrievedInformation:
        """Fallback RetrievedInformation implementation."""

        id: str
        content: str
        source: str
        relevance_score: float
        retrieval_confidence: float
        graph_connections: list[str] = None
        relationship_types: list[str] = None

    @dataclass
    class SynthesizedAnswer:
        """Fallback SynthesizedAnswer implementation."""

        answer: str
        confidence: float
        supporting_sources: list[str]
        synthesis_method: str


logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query processing modes for different use cases."""

    FAST = "fast"  # Vector-only, fastest response
    BALANCED = "balanced"  # Vector + Graph, good balance
    COMPREHENSIVE = "comprehensive"  # All systems, most thorough
    CREATIVE = "creative"  # Emphasize creativity engine
    ANALYTICAL = "analytical"  # Emphasize cognitive nexus


class MemoryType(Enum):
    """Types of memory for storage routing."""

    EPISODIC = "episodic"  # Recent, temporary (HippoRAG)
    SEMANTIC = "semantic"  # Long-term, structured (GraphRAG)
    VECTOR = "vector"  # Similarity-based (VectorRAG)
    ALL = "all"  # Store in all systems


@dataclass
class RAGConfig:
    """Configuration for HyperRAG system."""

    # Core system settings
    enable_hippo_rag: bool = True
    enable_graph_rag: bool = True
    enable_vector_rag: bool = True
    enable_cognitive_nexus: bool = True
    enable_creativity_engine: bool = True
    enable_graph_fixer: bool = True

    # Database integration
    enable_fog_computing: bool = True
    enable_edge_devices: bool = True
    enable_p2p_network: bool = True

    # Memory management
    hippo_ttl_hours: int = 168  # 7 days
    graph_trust_threshold: float = 0.4
    vector_similarity_threshold: float = 0.7

    # Performance settings
    max_results_per_system: int = 20
    cognitive_analysis_timeout: float = 30.0
    creativity_timeout: float = 15.0

    # Quality thresholds
    min_confidence_threshold: float = 0.3
    min_relevance_threshold: float = 0.5
    synthesis_confidence_threshold: float = 0.6


@dataclass
class QueryResult:
    """Unified query result from HyperRAG system."""

    # Core results
    synthesized_answer: SynthesizedAnswer
    primary_sources: list[RetrievedInformation]

    # System-specific results
    hippo_results: list[Any] = field(default_factory=list)
    graph_results: list[Any] = field(default_factory=list)
    vector_results: list[Any] = field(default_factory=list)

    # Analysis results
    cognitive_analysis: dict[str, Any] | None = None
    creative_insights: dict[str, Any] | None = None
    graph_gaps: list[dict[str, Any]] | None = None

    # Metadata
    total_latency_ms: float = 0.0
    processing_mode: QueryMode = QueryMode.BALANCED
    systems_used: list[str] = field(default_factory=list)
    confidence_score: float = 0.0

    # Edge device context
    edge_device_context: dict[str, Any] | None = None
    mobile_optimizations: dict[str, Any] | None = None


class HyperRAG:
    """
    Unified RAG System - Main Orchestrator

    Integrates hippocampus, graph, vector, and cognitive systems into
    a cohesive knowledge retrieval and reasoning platform with
    fog computing and edge device support.
    """

    def __init__(self, config: RAGConfig = None):
        """Initialize HyperRAG with configuration."""
        self.config = config or RAGConfig()

        # Core subsystems
        self.hippo_index: HippoIndex | None = None
        self.trust_graph: BayesianTrustGraph | None = None
        self.vector_engine: ContextualVectorEngine | None = None
        self.cognitive_nexus: CognitiveNexus | None = None
        self.graph_fixer: GraphFixer | None = None
        self.creativity_engine: CreativityEngine | None = None

        # Integration bridges
        self.edge_device_bridge = None
        self.p2p_network_bridge = None
        self.fog_compute_bridge = None

        # Performance tracking
        self.stats = {
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "system_usage": {"hippo": 0, "graph": 0, "vector": 0, "cognitive": 0, "creativity": 0},
            "edge_queries": 0,
            "fog_compute_tasks": 0,
        }

        self.initialized = False

    async def initialize(self):
        """Initialize all RAG subsystems and integrations."""
        try:
            logger.info("Initializing HyperRAG unified system...")

            # Initialize core memory systems
            if self.config.enable_hippo_rag:
                self.hippo_index = HippoIndex(
                    db_path="./data/hippo_memory.db",
                    redis_url="redis://localhost:6379/1",
                    qdrant_url="http://localhost:6333",
                )
                await self.hippo_index.initialize()
                logger.info("✅ HippoIndex (episodic memory) initialized")

            if self.config.enable_graph_rag:
                self.trust_graph = BayesianTrustGraph(
                    similarity_threshold=self.config.graph_trust_threshold,
                    trust_decay_factor=0.85,
                    max_propagation_hops=3,
                )
                await self.trust_graph.initialize()
                logger.info("✅ BayesianTrustGraph (knowledge graph) initialized")

            if self.config.enable_vector_rag:
                self.vector_engine = ContextualVectorEngine(
                    similarity_threshold=self.config.vector_similarity_threshold,
                    enable_dual_context=True,
                    enable_semantic_chunking=True,
                )
                await self.vector_engine.initialize()
                logger.info("✅ ContextualVectorEngine (vector search) initialized")

            # Initialize analysis systems
            if self.config.enable_cognitive_nexus:
                self.cognitive_nexus = CognitiveNexus()
                await self.cognitive_nexus.initialize()
                logger.info("✅ CognitiveNexus (analysis engine) initialized")

            if self.config.enable_graph_fixer:
                self.graph_fixer = GraphFixer(trust_graph=self.trust_graph, vector_engine=self.vector_engine)
                await self.graph_fixer.initialize()
                logger.info("✅ GraphFixer (gap detection) initialized")

            if self.config.enable_creativity_engine:
                self.creativity_engine = CreativityEngine(
                    trust_graph=self.trust_graph, vector_engine=self.vector_engine, hippo_index=self.hippo_index
                )
                await self.creativity_engine.initialize()
                logger.info("✅ CreativityEngine (insight discovery) initialized")

            # Initialize integration bridges
            if self.config.enable_edge_devices:
                from ..integration.edge_device_bridge import EdgeDeviceRAGBridge

                self.edge_device_bridge = EdgeDeviceRAGBridge(self)
                await self.edge_device_bridge.initialize()
                logger.info("✅ Edge device integration initialized")

            if self.config.enable_p2p_network:
                from ..integration.p2p_network_bridge import P2PNetworkRAGBridge

                self.p2p_network_bridge = P2PNetworkRAGBridge(self)
                await self.p2p_network_bridge.initialize()
                logger.info("✅ P2P network integration initialized")

            if self.config.enable_fog_computing:
                from ..integration.fog_compute_bridge import FogComputeBridge

                self.fog_compute_bridge = FogComputeBridge(self)
                await self.fog_compute_bridge.initialize()
                logger.info("✅ Fog computing integration initialized")

            self.initialized = True
            logger.info("🎉 HyperRAG system fully initialized")

        except Exception as e:
            logger.exception(f"Failed to initialize HyperRAG system: {e}")
            raise

    async def query(
        self,
        query: str,
        mode: QueryMode = QueryMode.BALANCED,
        context: dict[str, Any] | None = None,
        user_id: str | None = None,
        edge_device_id: str | None = None,
    ) -> QueryResult:
        """
        Main query interface for unified RAG system.

        Args:
            query: Natural language query
            mode: Processing mode (fast, balanced, comprehensive, creative, analytical)
            context: Additional context for query processing
            user_id: User identifier for personalization
            edge_device_id: Edge device identifier for mobile optimization

        Returns:
            QueryResult with synthesized answer and metadata
        """
        if not self.initialized:
            raise RuntimeError("HyperRAG not initialized. Call initialize() first.")

        start_time = time.time()
        systems_used = []

        try:
            logger.info(f"Processing query in {mode.value} mode: '{query[:100]}...'")

            # Apply edge device optimizations if available
            edge_context = None
            if edge_device_id and self.edge_device_bridge:
                edge_context = await self.edge_device_bridge.optimize_for_device(edge_device_id, query, mode)

            # Route query based on mode and context
            retrieved_info = await self._route_query(query, mode, context, user_id, edge_context)
            systems_used = list(retrieved_info.keys())

            # Combine results from all systems
            all_results = []
            for system, results in retrieved_info.items():
                all_results.extend(results)
                self.stats["system_usage"][system] += 1

            # Apply cognitive analysis if enabled
            cognitive_analysis = None
            if self.config.enable_cognitive_nexus and mode in [QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL]:
                try:
                    cognitive_results = await asyncio.wait_for(
                        self.cognitive_nexus.analyze_retrieved_information(query, all_results),
                        timeout=self.config.cognitive_analysis_timeout,
                    )
                    cognitive_analysis = {
                        "analysis_results": cognitive_results,
                        "confidence": statistics.mean([r.confidence.value for r in cognitive_results]),
                    }
                except asyncio.TimeoutError:
                    logger.warning(f"Cognitive analysis timed out after {self.config.cognitive_analysis_timeout}s")

            # Generate creative insights if enabled
            creative_insights = None
            if self.config.enable_creativity_engine and mode in [QueryMode.COMPREHENSIVE, QueryMode.CREATIVE]:
                try:
                    creative_insights = await asyncio.wait_for(
                        self.creativity_engine.discover_insights(query, all_results),
                        timeout=self.config.creativity_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Creativity analysis timed out after {self.config.creativity_timeout}s")

            # Detect and propose graph gaps if enabled
            graph_gaps = None
            if self.config.enable_graph_fixer and mode == QueryMode.COMPREHENSIVE:
                graph_gaps = await self.graph_fixer.detect_knowledge_gaps(query, all_results)

            # Synthesize final answer
            synthesized_answer = await self._synthesize_answer(
                query, all_results, cognitive_analysis, creative_insights, edge_context
            )

            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                synthesized_answer, cognitive_analysis, creative_insights
            )

            # Build result
            total_latency = (time.time() - start_time) * 1000

            result = QueryResult(
                synthesized_answer=synthesized_answer,
                primary_sources=all_results[:10],  # Top 10 sources
                hippo_results=retrieved_info.get("hippo", []),
                graph_results=retrieved_info.get("graph", []),
                vector_results=retrieved_info.get("vector", []),
                cognitive_analysis=cognitive_analysis,
                creative_insights=creative_insights,
                graph_gaps=graph_gaps,
                total_latency_ms=total_latency,
                processing_mode=mode,
                systems_used=systems_used,
                confidence_score=confidence_score,
                edge_device_context=edge_context,
            )

            # Update statistics
            self.stats["queries_processed"] += 1
            self.stats["total_processing_time"] += total_latency
            if edge_device_id:
                self.stats["edge_queries"] += 1

            logger.info(f"Query completed in {total_latency:.1f}ms (confidence: {confidence_score:.3f})")
            return result

        except Exception as e:
            logger.exception(f"Query processing failed: {e}")
            # Return minimal fallback result
            fallback_answer = SynthesizedAnswer(
                answer=f"I encountered an error processing your query: {e}",
                confidence=0.1,
                supporting_sources=[],
                synthesis_method="error_fallback",
            )

            return QueryResult(
                synthesized_answer=fallback_answer,
                primary_sources=[],
                total_latency_ms=(time.time() - start_time) * 1000,
                processing_mode=mode,
                systems_used=systems_used,
                confidence_score=0.1,
            )

    async def store_document(
        self,
        content: str,
        title: str = "",
        memory_type: MemoryType = MemoryType.ALL,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> dict[str, bool]:
        """
        Store document in appropriate memory systems.

        Args:
            content: Document content
            title: Document title
            memory_type: Where to store (episodic, semantic, vector, all)
            metadata: Additional metadata
            user_id: User identifier

        Returns:
            Dictionary indicating success for each system
        """
        results = {}

        try:
            # Create base document
            doc_metadata = metadata or {}
            doc_metadata.update(
                {
                    "title": title,
                    "stored_at": datetime.now().isoformat(),
                    "user_id": user_id,
                    "storage_type": memory_type.value,
                }
            )

            # Store in HippoIndex (episodic memory)
            if memory_type in [MemoryType.EPISODIC, MemoryType.ALL] and self.hippo_index:
                episodic_doc = EpisodicDocument(
                    content=content, doc_type="user_document", user_id=user_id, metadata=doc_metadata
                )

                # Create hippo node
                hippo_node = create_hippo_node(content=content, user_id=user_id, ttl_hours=self.config.hippo_ttl_hours)

                results["hippo"] = await self.hippo_index.store_document(episodic_doc)
                await self.hippo_index.store_node(hippo_node)

            # Store in BayesianTrustGraph (semantic memory)
            if memory_type in [MemoryType.SEMANTIC, MemoryType.ALL] and self.trust_graph:
                results["graph"] = await self.trust_graph.add_document_with_relationships(
                    content=content, doc_id=f"doc_{int(time.time()*1000)}", metadata=doc_metadata
                )

            # Store in ContextualVectorEngine (vector memory)
            if memory_type in [MemoryType.VECTOR, MemoryType.ALL] and self.vector_engine:
                results["vector"] = await self.vector_engine.index_document(
                    content=content, doc_id=f"vec_{int(time.time()*1000)}", metadata=doc_metadata
                )

            logger.info(f"Document stored successfully: {results}")
            return results

        except Exception as e:
            logger.exception(f"Failed to store document: {e}")
            return {"error": str(e)}

    async def _route_query(
        self,
        query: str,
        mode: QueryMode,
        context: dict[str, Any] | None,
        user_id: str | None,
        edge_context: dict[str, Any] | None,
    ) -> dict[str, list[RetrievedInformation]]:
        """Route query to appropriate subsystems based on mode."""

        results = {}

        # Vector search (always included for baseline)
        if self.vector_engine:
            vector_results = await self.vector_engine.search(
                query=query, k=self.config.max_results_per_system, user_id=user_id, context=context
            )
            results["vector"] = self._convert_to_retrieved_info(vector_results, "vector")

        # Fast mode: vector only
        if mode == QueryMode.FAST:
            return results

        # Graph search for balanced+ modes
        if mode in [QueryMode.BALANCED, QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL] and self.trust_graph:
            graph_results = await self.trust_graph.retrieve_with_trust_propagation(
                query=query, k=self.config.max_results_per_system, min_trust_score=self.config.graph_trust_threshold
            )
            results["graph"] = self._convert_to_retrieved_info(graph_results, "graph")

        # Hippo search for comprehensive+ modes
        if mode in [QueryMode.COMPREHENSIVE, QueryMode.CREATIVE] and self.hippo_index:
            hippo_results = await self.hippo_index.query_nodes(
                query=query,
                limit=self.config.max_results_per_system,
                user_id=user_id,
                max_age_hours=self.config.hippo_ttl_hours,
            )
            results["hippo"] = self._convert_to_retrieved_info(hippo_results.nodes, "hippo")

        return results

    def _convert_to_retrieved_info(self, raw_results: list[Any], source_system: str) -> list[RetrievedInformation]:
        """Convert system-specific results to unified RetrievedInformation format."""

        converted = []

        for i, result in enumerate(raw_results):
            try:
                # Extract common fields based on result type
                if hasattr(result, "content"):
                    content = result.content
                elif hasattr(result, "text"):
                    content = result.text
                else:
                    content = str(result)

                # Extract confidence/relevance
                if hasattr(result, "confidence"):
                    confidence = result.confidence
                elif hasattr(result, "score"):
                    confidence = result.score
                elif hasattr(result, "similarity"):
                    confidence = result.similarity
                else:
                    confidence = 0.5

                # Extract ID
                result_id = result.id if hasattr(result, "id") else f"{source_system}_{i}"

                # Create unified object
                retrieved_info = RetrievedInformation(
                    id=result_id,
                    content=content,
                    source=source_system,
                    relevance_score=confidence,
                    retrieval_confidence=confidence,
                )

                # Add system-specific metadata
                if source_system == "graph" and hasattr(result, "trust_score"):
                    retrieved_info.graph_connections = getattr(result, "connections", [])
                    retrieved_info.relationship_types = getattr(result, "relationship_types", [])

                converted.append(retrieved_info)

            except Exception as e:
                logger.warning(f"Failed to convert result from {source_system}: {e}")
                continue

        return converted

    async def _synthesize_answer(
        self,
        query: str,
        all_results: list[RetrievedInformation],
        cognitive_analysis: dict[str, Any] | None,
        creative_insights: dict[str, Any] | None,
        edge_context: dict[str, Any] | None,
    ) -> SynthesizedAnswer:
        """Synthesize final answer from all available information."""

        try:
            # Use cognitive nexus if available
            if self.cognitive_nexus and cognitive_analysis:
                analysis_results = cognitive_analysis.get("analysis_results", [])
                synthesized = await self.cognitive_nexus.synthesize_answer(
                    query=query, retrieved_info=all_results, analysis_results=analysis_results
                )

                # Enhance with creative insights
                if creative_insights:
                    creative_text = creative_insights.get("insights_summary", "")
                    if creative_text:
                        synthesized.answer += f"\n\nAdditional insights: {creative_text}"
                        synthesized.confidence = min(1.0, synthesized.confidence + 0.1)

                return synthesized

            # Fallback synthesis
            else:
                # Simple synthesis from top results
                top_results = sorted(all_results, key=lambda x: x.relevance_score, reverse=True)[:5]

                if not top_results:
                    return SynthesizedAnswer(
                        answer="I couldn't find relevant information to answer your query.",
                        confidence=0.1,
                        supporting_sources=[],
                        synthesis_method="fallback_empty",
                    )

                # Create basic synthesis
                answer_parts = []
                supporting_sources = []

                for result in top_results:
                    answer_parts.append(result.content[:200] + "...")
                    supporting_sources.append(result.source)

                answer = f"Based on available information: {' '.join(answer_parts[:2])}"
                confidence = statistics.mean([r.relevance_score for r in top_results])

                return SynthesizedAnswer(
                    answer=answer,
                    confidence=confidence,
                    supporting_sources=supporting_sources,
                    synthesis_method="fallback_basic",
                )

        except Exception as e:
            logger.exception(f"Answer synthesis failed: {e}")
            return SynthesizedAnswer(
                answer=f"I encountered an error synthesizing the answer: {e}",
                confidence=0.1,
                supporting_sources=[],
                synthesis_method="error_fallback",
            )

    def _calculate_overall_confidence(
        self,
        synthesized_answer: SynthesizedAnswer,
        cognitive_analysis: dict[str, Any] | None,
        creative_insights: dict[str, Any] | None,
    ) -> float:
        """Calculate overall confidence score for the response."""

        base_confidence = synthesized_answer.confidence

        # Boost for cognitive analysis
        if cognitive_analysis and cognitive_analysis.get("confidence", 0) > 0.7:
            base_confidence += 0.1

        # Boost for creative insights
        if creative_insights and creative_insights.get("confidence", 0) > 0.6:
            base_confidence += 0.05

        return min(1.0, base_confidence)

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status and statistics."""

        status = {
            "initialized": self.initialized,
            "config": {
                "hippo_enabled": self.config.enable_hippo_rag,
                "graph_enabled": self.config.enable_graph_rag,
                "vector_enabled": self.config.enable_vector_rag,
                "cognitive_enabled": self.config.enable_cognitive_nexus,
                "creativity_enabled": self.config.enable_creativity_engine,
                "edge_enabled": self.config.enable_edge_devices,
                "p2p_enabled": self.config.enable_p2p_network,
                "fog_enabled": self.config.enable_fog_computing,
            },
            "statistics": self.stats.copy(),
            "subsystems": {},
        }

        # Get subsystem health
        if self.hippo_index:
            status["subsystems"]["hippo"] = await self.hippo_index.health_check()

        if self.trust_graph:
            status["subsystems"]["graph"] = await self.trust_graph.get_health_status()

        if self.vector_engine:
            status["subsystems"]["vector"] = await self.vector_engine.get_status()

        if self.cognitive_nexus:
            status["subsystems"]["cognitive"] = await self.cognitive_nexus.get_nexus_stats()

        # Calculate derived metrics
        if self.stats["queries_processed"] > 0:
            status["performance"] = {
                "avg_latency_ms": self.stats["total_processing_time"] / self.stats["queries_processed"],
                "cache_hit_rate": self.stats["cache_hits"] / self.stats["queries_processed"],
                "edge_query_ratio": self.stats["edge_queries"] / self.stats["queries_processed"],
            }

        return status

    async def close(self):
        """Shutdown all subsystems and clean up resources."""
        try:
            logger.info("Shutting down HyperRAG system...")

            if self.hippo_index:
                await self.hippo_index.close()

            if self.trust_graph:
                await self.trust_graph.shutdown()

            if self.vector_engine:
                await self.vector_engine.close()

            if self.cognitive_nexus:
                # CognitiveNexus doesn't have explicit close method
                pass

            if self.edge_device_bridge:
                await self.edge_device_bridge.close()

            if self.p2p_network_bridge:
                await self.p2p_network_bridge.close()

            if self.fog_compute_bridge:
                await self.fog_compute_bridge.close()

            self.initialized = False
            logger.info("HyperRAG system shutdown complete")

        except Exception as e:
            logger.exception(f"Error during HyperRAG shutdown: {e}")


# Convenience function for quick setup
async def create_hyper_rag(
    enable_all: bool = True, fog_computing: bool = False, edge_devices: bool = False, p2p_network: bool = False
) -> HyperRAG:
    """Create and initialize HyperRAG system with common configurations."""

    config = RAGConfig(
        enable_hippo_rag=enable_all,
        enable_graph_rag=enable_all,
        enable_vector_rag=True,  # Always enable vector for baseline
        enable_cognitive_nexus=enable_all,
        enable_creativity_engine=enable_all,
        enable_graph_fixer=enable_all,
        enable_fog_computing=fog_computing,
        enable_edge_devices=edge_devices,
        enable_p2p_network=p2p_network,
    )

    hyper_rag = HyperRAG(config)
    await hyper_rag.initialize()

    return hyper_rag
