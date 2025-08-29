#!/usr/bin/env python3
"""
Unified HyperRAG Implementation - Production Ready

Consolidates the best features from multiple scattered implementations into a single
working system that coordinates all RAG subsystems:
- HippoRAG (neurobiological episodic memory)
- GraphRAG (Bayesian trust networks)
- VectorRAG (contextual similarity search)
- Cognitive Nexus (analysis and reasoning)

This is the main entry point for the consolidated HyperRAG system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import os
import statistics
import sys
import time
from typing import Any

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from cognitive.reasoning_engine import CognitiveReasoningEngine
    from neural_memory.hippo_rag import HippoRAG
    from neural_memory.hippo_rag import MemoryType as HippoMemoryType
    from trust_networks.bayesian_trust import BayesianTrustNetwork, TrustDimension

    ADVANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    ADVANCED_COMPONENTS_AVAILABLE = False
    logger.warning(f"Advanced components not available: {e}")

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Enhanced query processing modes for different use cases."""

    FAST = "fast"  # Vector-only, fastest response
    BALANCED = "balanced"  # Vector + Graph, good balance
    COMPREHENSIVE = "comprehensive"  # All systems, most thorough
    CREATIVE = "creative"  # Emphasize creativity engine
    ANALYTICAL = "analytical"  # Emphasize cognitive nexus
    DISTRIBUTED = "distributed"  # Use P2P network for retrieval
    EDGE_OPTIMIZED = "edge_optimized"  # Mobile/edge device optimization


class MemoryType(Enum):
    """Types of memory for storage routing."""

    EPISODIC = "episodic"  # Recent, temporary (HippoRAG)
    SEMANTIC = "semantic"  # Long-term, structured (GraphRAG)
    VECTOR = "vector"  # Similarity-based (VectorRAG)
    PROCEDURAL = "procedural"  # How-to knowledge
    ALL = "all"  # Store in all systems


@dataclass
class RetrievedInformation:
    """Information retrieved from knowledge base."""

    id: str
    content: str
    source: str
    relevance_score: float
    retrieval_confidence: float
    graph_connections: list[str] = field(default_factory=list)
    relationship_types: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SynthesizedAnswer:
    """Final synthesized answer from multiple sources."""

    answer: str
    confidence: float
    supporting_sources: list[str]
    synthesis_method: str
    retrieval_sources: list[RetrievedInformation] = field(default_factory=list)
    processing_time: float = 0.0
    query_mode: str = "balanced"


@dataclass
class HyperRAGConfig:
    """Configuration for HyperRAG system."""

    max_results: int = 10
    min_confidence: float = 0.1
    vector_dimensions: int = 384
    graph_depth_limit: int = 3
    enable_caching: bool = True
    timeout_seconds: float = 30.0
    fallback_enabled: bool = True

    # Subsystem enablement
    enable_hippo_rag: bool = True
    enable_graph_rag: bool = True
    enable_vector_rag: bool = True
    enable_cognitive_nexus: bool = True
    enable_creativity_engine: bool = True
    enable_graph_fixer: bool = True


class SimpleVectorStore:
    """Simple in-memory vector store for development/testing."""

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.documents: dict[str, str] = {}
        self.vectors: dict[str, list[float]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}

    def add_document(self, doc_id: str, content: str, metadata: dict = None):
        """Add document to vector store."""
        self.documents[doc_id] = content
        # Simple hash-based pseudo-vector (replace with real embeddings in production)
        vector = [float(hash(content + str(i)) % 1000) / 1000.0 for i in range(self.dimensions)]
        self.vectors[doc_id] = vector
        self.metadata[doc_id] = metadata or {}

    def search(self, query: str, top_k: int = 5) -> list[tuple]:
        """Search for similar documents."""
        if not self.documents:
            return []

        # Simple similarity based on content overlap
        query_words = set(query.lower().split())
        results = []

        for doc_id, content in self.documents.items():
            content_words = set(content.lower().split())
            overlap = len(query_words.intersection(content_words))
            similarity = overlap / max(len(query_words), 1)

            if similarity > 0:
                results.append((doc_id, similarity))

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class SimpleGraphStore:
    """Simple in-memory graph store for development/testing."""

    def __init__(self):
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, list[tuple]] = {}

    def add_node(self, node_id: str, properties: dict[str, Any]):
        """Add node to graph."""
        self.nodes[node_id] = properties
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, from_node: str, to_node: str, relationship: str, properties: dict = None):
        """Add edge between nodes."""
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, relationship, properties or {}))

    def get_connected_nodes(self, node_id: str, depth: int = 1) -> list[tuple]:
        """Get nodes connected to given node within depth."""
        if depth <= 0 or node_id not in self.edges:
            return []

        connected = []
        for to_node, relationship, _ in self.edges[node_id]:
            connected.append((to_node, relationship))
            if depth > 1:
                deeper = self.get_connected_nodes(to_node, depth - 1)
                connected.extend(deeper)

        return connected


class HyperRAG:
    """
    Unified HyperRAG Implementation - Production Ready

    Combines vector search, graph reasoning, and synthesis into a single system.
    Coordinates all RAG subsystems with intelligent routing and fallback handling.
    """

    def __init__(self, config: HyperRAGConfig = None):
        """Initialize HyperRAG with configuration."""
        self.config = config or HyperRAGConfig()
        self.logger = logging.getLogger(f"{__name__}.HyperRAG")

        # Enhanced core subsystems with neural-biological components
        if ADVANCED_COMPONENTS_AVAILABLE:
            self.hippo_rag: HippoRAG | None = None
            self.trust_network: BayesianTrustNetwork | None = None
            self.cognitive_engine: CognitiveReasoningEngine | None = None
        else:
            # Fallback to simple systems
            self.vector_store: SimpleVectorStore | None = None
            self.graph_store: SimpleGraphStore | None = None

        self.hippo_index = None  # Backward compatibility
        self.cognitive_nexus = None  # Backward compatibility
        self.creativity_engine = None  # Will be loaded from cognitive module
        self.graph_fixer = None  # Will be loaded from cognitive module

        # Integration bridges (will be loaded)
        self.edge_device_bridge = None
        self.p2p_network_bridge = None
        self.fog_compute_bridge = None

        # Enhanced statistics tracking
        self.stats = {
            "queries_processed": 0,
            "documents_indexed": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "fallback_uses": 0,
            # Neural-biological metrics
            "hippo_retrievals": 0,
            "trust_validations": 0,
            "cognitive_reasoning_sessions": 0,
            "memory_consolidations": 0,
            "trust_conflicts_detected": 0,
        }

        # Enhanced cache for repeated queries
        self.query_cache: dict[str, SynthesizedAnswer] = {}

        self.logger.info(f"HyperRAG initialized with config: {self.config}")
        if ADVANCED_COMPONENTS_AVAILABLE:
            self.logger.info("🧠 Neural-biological components available")
        else:
            self.logger.warning("⚠️  Using fallback systems - advanced components not available")

    async def initialize(self):
        """Initialize async components and load subsystems."""
        try:
            if ADVANCED_COMPONENTS_AVAILABLE:
                # Initialize neural-biological memory system
                if self.config.enable_hippo_rag:
                    self.hippo_rag = HippoRAG(
                        embedding_dim=self.config.vector_dimensions,
                        max_episodic_memories=10000,
                        consolidation_threshold=0.7,
                        forgetting_threshold=0.1,
                    )
                    await self.hippo_rag.initialize()
                    self.logger.info("✅ HippoRAG neural memory system initialized")

                # Initialize Bayesian trust network
                if self.config.enable_graph_rag:
                    self.trust_network = BayesianTrustNetwork(
                        max_propagation_depth=3, trust_threshold=0.6, uncertainty_threshold=0.3
                    )
                    await self.trust_network.initialize()
                    self.logger.info("✅ Bayesian trust network initialized")

                # Initialize cognitive reasoning engine
                if self.config.enable_cognitive_nexus:
                    self.cognitive_engine = CognitiveReasoningEngine(
                        max_reasoning_depth=5,
                        confidence_threshold=0.7,
                        enable_meta_reasoning=True,
                        enable_bias_detection=True,
                    )
                    await self.cognitive_engine.initialize()
                    self.logger.info("✅ Cognitive reasoning engine initialized")
            else:
                # Fallback to simple systems
                if self.config.enable_vector_rag:
                    self.vector_store = SimpleVectorStore(self.config.vector_dimensions)
                    self.logger.info("✅ Simple vector store initialized (fallback)")

                if self.config.enable_graph_rag:
                    self.graph_store = SimpleGraphStore()
                    self.logger.info("✅ Simple graph store initialized (fallback)")

            self.logger.info("🚀 HyperRAG async initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"HyperRAG initialization failed: {e}")
            return False

    async def shutdown(self):
        """Shutdown and cleanup."""
        self.logger.info("HyperRAG shutdown complete")

    async def add_document(
        self, content: str, doc_id: str = None, metadata: dict = None, memory_type: MemoryType = MemoryType.EPISODIC
    ) -> str:
        """Add document to knowledge base with enhanced neural-biological storage."""
        if doc_id is None:
            doc_id = f"doc_{int(time.time() * 1000000)}"

        if ADVANCED_COMPONENTS_AVAILABLE:
            # Store in neural-biological memory system
            if self.hippo_rag and self.config.enable_hippo_rag:
                # Convert to HippoRAG memory type
                hippo_memory_type = (
                    HippoMemoryType.EPISODIC if memory_type == MemoryType.EPISODIC else HippoMemoryType.SEMANTIC
                )

                # Create contextual information
                from neural_memory.hippo_rag import (
                    create_semantic_context,
                    create_spatial_context,
                    create_temporal_context,
                )

                spatial_context = create_spatial_context("knowledge_base", "digital_repository")
                temporal_context = create_temporal_context(event_type="document_ingestion")

                # Extract domain and keywords for semantic context
                domain = metadata.get("domain", "general") if metadata else "general"
                keywords = content.split()[:10]  # Simple keyword extraction
                semantic_context = create_semantic_context(domain, "document_content", keywords)

                memory_id = await self.hippo_rag.encode_memory(
                    content=content,
                    memory_type=hippo_memory_type,
                    spatial_context=spatial_context,
                    temporal_context=temporal_context,
                    semantic_context=semantic_context,
                    metadata=metadata or {},
                )

                self.logger.debug(f"Stored in HippoRAG: {memory_id}")

            # Add to Bayesian trust network
            if self.trust_network and self.config.enable_graph_rag:
                source_id = await self.trust_network.add_source(
                    source_identifier=doc_id,
                    content=content,
                    source_type=metadata.get("source_type", "document") if metadata else "document",
                    domain=metadata.get("domain", "") if metadata else "",
                    keywords=content.split()[:5],  # Simple keyword extraction
                    metadata=metadata or {},
                )

                # Add initial trust evidence based on metadata
                if metadata and "credibility" in metadata:
                    from trust_networks.bayesian_trust import create_accuracy_evidence

                    evidence = create_accuracy_evidence(
                        positive=metadata["credibility"],
                        negative=1.0 - metadata["credibility"],
                        evaluator_id="system",
                        confidence=0.8,
                    )
                    await self.trust_network.add_evidence(
                        source_id,
                        evidence.evidence_type,
                        evidence.dimension,
                        evidence.positive_evidence,
                        evidence.negative_evidence,
                        uncertainty=evidence.uncertainty,
                        evaluator_id=evidence.evaluator_id,
                        confidence=evidence.confidence,
                    )

                self.logger.debug(f"Added to trust network: {source_id}")
        else:
            # Fallback to simple systems
            if self.vector_store:
                self.vector_store.add_document(doc_id, content, metadata)

            # Extract entities and add to graph store
            if self.graph_store:
                words = content.split()
                important_words = [w for w in words if len(w) > 3][:5]  # Simple entity extraction

                for word in important_words:
                    self.graph_store.add_node(f"entity_{word.lower()}", {"type": "entity", "value": word})
                    self.graph_store.add_edge(doc_id, f"entity_{word.lower()}", "contains", {})

        self.stats["documents_indexed"] += 1
        self.logger.info(f"Added document {doc_id} to knowledge base")
        return doc_id

    async def process_query(
        self, query: str, mode: QueryMode = QueryMode.BALANCED, context: dict = None, user_id: str = None
    ) -> SynthesizedAnswer:
        """Process query using advanced neural-biological components."""
        start_time = time.time()

        # Check cache first
        cache_key = f"{query}:{mode.value}:{user_id}"
        if self.config.enable_caching and cache_key in self.query_cache:
            self.stats["cache_hits"] += 1
            return self.query_cache[cache_key]

        try:
            retrieved_info = []
            evidence_sources = []

            if ADVANCED_COMPONENTS_AVAILABLE:
                # Neural-biological memory retrieval
                if self.hippo_rag and self.config.enable_hippo_rag:
                    from neural_memory.hippo_rag import create_semantic_context

                    # Create query context
                    domain = context.get("domain", "general") if context else "general"
                    semantic_context = create_semantic_context(domain, "query_processing", query.split()[:5])

                    hippo_results = await self.hippo_rag.retrieve_memories(
                        query=query,
                        k=self.config.max_results,
                        memory_types=[HippoMemoryType.EPISODIC, HippoMemoryType.SEMANTIC],
                        semantic_context=semantic_context,
                    )

                    for memory_trace in hippo_results.memory_traces:
                        info = RetrievedInformation(
                            id=memory_trace.id,
                            content=memory_trace.content,
                            source="hippo_rag",
                            relevance_score=memory_trace.accessibility,
                            retrieval_confidence=memory_trace.strength,
                        )
                        retrieved_info.append(info)

                        # Prepare for cognitive reasoning
                        evidence_sources.append(
                            {
                                "content": memory_trace.content,
                                "type": "episodic"
                                if memory_trace.memory_type == HippoMemoryType.EPISODIC
                                else "semantic",
                                "confidence": memory_trace.strength,
                                "source": "neural_memory",
                                "metadata": memory_trace.metadata,
                            }
                        )

                    self.stats["hippo_retrievals"] += 1

                # Trust-validated knowledge retrieval
                if self.trust_network and self.config.enable_graph_rag:
                    trust_results = await self.trust_network.retrieve_with_trust_propagation(
                        query=query,
                        k=self.config.max_results,
                        min_trust_score=0.4,
                        trust_dimensions=[TrustDimension.ACCURACY, TrustDimension.AUTHORITY],
                    )

                    for trust_node, combined_score, trust_score in trust_results:
                        info = RetrievedInformation(
                            id=trust_node.id,
                            content=trust_node.content,
                            source="trust_network",
                            relevance_score=combined_score,
                            retrieval_confidence=trust_score.overall_trust,
                        )
                        retrieved_info.append(info)

                        # Prepare for cognitive reasoning
                        evidence_sources.append(
                            {
                                "content": trust_node.content,
                                "type": "validated_knowledge",
                                "confidence": trust_score.overall_trust,
                                "source": "trust_network",
                                "trust_dimensions": trust_score.get_trust_summary(),
                                "metadata": trust_node.metadata,
                            }
                        )

                    self.stats["trust_validations"] += 1

                # Advanced cognitive reasoning
                if self.cognitive_engine and self.config.enable_cognitive_nexus and evidence_sources:
                    cognitive_result = await self.cognitive_engine.reason(
                        query=query,
                        evidence_sources=evidence_sources,
                        context=context,
                        require_multi_perspective=(mode == QueryMode.COMPREHENSIVE),
                    )

                    # Enhanced synthesis using cognitive reasoning
                    answer = SynthesizedAnswer(
                        answer=cognitive_result.synthesized_answer,
                        confidence=self._map_confidence_level_to_float(cognitive_result.confidence_level),
                        supporting_sources=[source["source"] for source in evidence_sources],
                        synthesis_method="cognitive_reasoning",
                        retrieval_sources=retrieved_info,
                        processing_time=time.time() - start_time,
                        query_mode=mode.value,
                    )

                    self.stats["cognitive_reasoning_sessions"] += 1
                else:
                    # Standard synthesis if no cognitive engine
                    answer = self._synthesize_answer(query, retrieved_info, mode)
                    answer.processing_time = time.time() - start_time

            else:
                # Fallback to simple systems
                if self.vector_store:
                    vector_results = self.vector_store.search(query, top_k=self.config.max_results)
                    for doc_id, score in vector_results:
                        if score >= self.config.min_confidence:
                            content = self.vector_store.documents.get(doc_id, "")
                            info = RetrievedInformation(
                                id=doc_id,
                                content=content,
                                source="vector_store",
                                relevance_score=score,
                                retrieval_confidence=score,
                            )

                            # Add graph connections if in comprehensive mode
                            if mode in [QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL] and self.graph_store:
                                connections = self.graph_store.get_connected_nodes(doc_id, depth=2)
                                info.graph_connections = [conn[0] for conn in connections]
                                info.relationship_types = [conn[1] for conn in connections]

                            retrieved_info.append(info)

                # Synthesize answer
                answer = self._synthesize_answer(query, retrieved_info, mode)
                answer.processing_time = time.time() - start_time

            # Cache result
            if self.config.enable_caching:
                self.query_cache[cache_key] = answer

            self.stats["queries_processed"] += 1
            self._update_average_response_time(answer.processing_time)

            return answer

        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            self.stats["fallback_uses"] += 1

            # Fallback response
            return SynthesizedAnswer(
                answer=f"I encountered an error processing your query: {query}. Please try rephrasing your question.",
                confidence=0.1,
                supporting_sources=[],
                synthesis_method="error_fallback",
                processing_time=time.time() - start_time,
                query_mode=mode.value,
            )

    async def process_query_async(
        self, query: str, mode: QueryMode = QueryMode.BALANCED, context: dict = None, user_id: str = None
    ) -> SynthesizedAnswer:
        """Async version of process_query."""
        return await self.process_query(query, mode, context, user_id)

    def _synthesize_answer(
        self, query: str, retrieved_info: list[RetrievedInformation], mode: QueryMode
    ) -> SynthesizedAnswer:
        """Synthesize final answer from retrieved information."""
        if not retrieved_info:
            return SynthesizedAnswer(
                answer="I don't have enough information to answer your query.",
                confidence=0.0,
                supporting_sources=[],
                synthesis_method="no_results",
                retrieval_sources=retrieved_info,
                query_mode=mode.value,
            )

        # Simple synthesis: combine top results
        top_results = retrieved_info[:3]  # Use top 3 results

        if mode == QueryMode.FAST:
            # Fast mode: use best single result
            best_result = top_results[0]
            answer = f"Based on the most relevant information: {best_result.content[:200]}..."
            confidence = best_result.relevance_score
            synthesis_method = "single_source"

        elif mode == QueryMode.CREATIVE:
            # Creative mode: combine and extrapolate
            combined_content = " ".join([info.content for info in top_results])
            answer = f"Synthesizing from multiple sources: {combined_content[:300]}... This suggests innovative approaches to your query about '{query}'."
            confidence = (
                statistics.mean([info.relevance_score for info in top_results]) * 0.9
            )  # Slightly lower confidence for creative responses
            synthesis_method = "creative_synthesis"

        else:
            # Balanced/Comprehensive/Analytical: structured synthesis
            combined_content = "\n".join([f"- {info.content}" for info in top_results])
            answer = f"Based on {len(top_results)} relevant sources:\n{combined_content}"
            confidence = statistics.mean([info.relevance_score for info in top_results])
            synthesis_method = "multi_source_synthesis"

        supporting_sources = [info.id for info in top_results]

        return SynthesizedAnswer(
            answer=answer,
            confidence=confidence,
            supporting_sources=supporting_sources,
            synthesis_method=synthesis_method,
            retrieval_sources=retrieved_info,
            query_mode=mode.value,
        )

    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time."""
        current_avg = self.stats["average_response_time"]
        query_count = self.stats["queries_processed"]

        if query_count == 1:
            self.stats["average_response_time"] = new_time
        else:
            # Rolling average
            self.stats["average_response_time"] = (current_avg * (query_count - 1) + new_time) / query_count

    def _map_confidence_level_to_float(self, confidence_level) -> float:
        """Map ConfidenceLevel enum to float value."""
        if ADVANCED_COMPONENTS_AVAILABLE:
            from cognitive.reasoning_engine import ConfidenceLevel

            mapping = {
                ConfidenceLevel.VERY_HIGH: 0.95,
                ConfidenceLevel.HIGH: 0.8,
                ConfidenceLevel.MEDIUM: 0.6,
                ConfidenceLevel.LOW: 0.4,
                ConfidenceLevel.VERY_LOW: 0.2,
            }
            return mapping.get(confidence_level, 0.5)
        return 0.5

    async def get_stats(self) -> dict[str, Any]:
        """Get enhanced system statistics."""
        base_stats = {
            **self.stats,
            "cache_size": len(self.query_cache),
            "config": {
                "max_results": self.config.max_results,
                "min_confidence": self.config.min_confidence,
                "caching_enabled": self.config.enable_caching,
            },
        }

        if ADVANCED_COMPONENTS_AVAILABLE:
            # Add neural-biological system stats
            if self.hippo_rag:
                hippo_status = await self.hippo_rag.get_status()
                base_stats["hippo_rag"] = {
                    "status": hippo_status["status"],
                    "episodic_memories": hippo_status["memory_statistics"]["episodic_memories"],
                    "semantic_memories": hippo_status["memory_statistics"]["semantic_memories"],
                    "average_accessibility": hippo_status["health_metrics"]["average_accessibility"],
                }

            if self.trust_network:
                trust_status = await self.trust_network.get_network_status()
                base_stats["trust_network"] = {
                    "status": trust_status["status"],
                    "total_nodes": trust_status["network_size"]["total_nodes"],
                    "average_trust": trust_status["trust_metrics"]["average_trust"],
                    "conflicts_detected": trust_status["performance"]["conflicts_detected"],
                }

            if self.cognitive_engine:
                cognitive_status = await self.cognitive_engine.get_system_status()
                base_stats["cognitive_engine"] = {
                    "status": cognitive_status["status"],
                    "queries_processed": cognitive_status["performance_metrics"]["queries_processed"],
                    "average_confidence": cognitive_status["performance_metrics"]["average_confidence"],
                    "bias_detections": cognitive_status["reasoning_analytics"]["bias_detections"],
                }
        else:
            # Add simple system stats
            base_stats["vector_store_docs"] = len(self.vector_store.documents) if self.vector_store else 0
            base_stats["graph_store_nodes"] = len(self.graph_store.nodes) if self.graph_store else 0

        return base_stats

    def clear_cache(self):
        """Clear query cache."""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")

    async def health_check(self) -> dict[str, Any]:
        """Perform comprehensive system health check."""
        if ADVANCED_COMPONENTS_AVAILABLE:
            components = {
                "hippo_rag": "operational" if self.hippo_rag else "disabled",
                "trust_network": "operational" if self.trust_network else "disabled",
                "cognitive_engine": "operational" if self.cognitive_engine else "disabled",
                "cache": "operational",
            }

            # Check individual component health
            issues = []
            if self.hippo_rag:
                hippo_status = await self.hippo_rag.get_status()
                if hippo_status["status"] != "healthy":
                    issues.append("HippoRAG memory system issues")

            if self.trust_network:
                trust_status = await self.trust_network.get_network_status()
                if trust_status["status"] != "healthy":
                    issues.append("Trust network issues")

            if self.cognitive_engine:
                cognitive_status = await self.cognitive_engine.get_system_status()
                if cognitive_status["status"] != "healthy":
                    issues.append("Cognitive reasoning engine issues")

            overall_status = "healthy" if not issues else "degraded"
        else:
            components = {
                "vector_store": "operational" if self.vector_store else "disabled",
                "graph_store": "operational" if self.graph_store else "disabled",
                "cache": "operational",
            }
            issues = ["Advanced components not available - using fallback systems"]
            overall_status = "limited"

        return {
            "status": overall_status,
            "components": components,
            "issues": issues,
            "stats": await self.get_stats(),
            "neural_biological_enabled": ADVANCED_COMPONENTS_AVAILABLE,
        }


# Alias for backward compatibility
HyperRAGSystem = HyperRAG

# Export main classes
__all__ = [
    "HyperRAG",
    "HyperRAGSystem",
    "HyperRAGConfig",
    "QueryMode",
    "MemoryType",
    "RetrievedInformation",
    "SynthesizedAnswer",
]
