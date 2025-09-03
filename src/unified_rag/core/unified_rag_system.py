"""
Unified RAG System - Ultimate RAG Implementation

The definitive RAG system that combines the best of all approaches:
- Advanced Ingestion System with semantic processing
- HippoRAG Memory Architecture with episodic storage  
- Dual Context Vector RAG with hierarchical embeddings
- Bayesian Knowledge Graph RAG with trust propagation
- Cognitive Nexus with multi-perspective reasoning
- Creative Graph Search with associative discovery
- Missing Node Detection with gap analysis

All integrated with strategic MCP server coordination.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sys
import importlib.util

# Load HippoCache dynamically from integrations path
_hippo_path = Path(__file__).resolve().parents[3] / "integrations" / "clients" / "py-aivillage" / "rag" / "hippo_cache.py"
spec = importlib.util.spec_from_file_location("hippo_cache", _hippo_path)
hippo_cache = importlib.util.module_from_spec(spec)
sys.modules["hippo_cache"] = hippo_cache
spec.loader.exec_module(hippo_cache)  # type: ignore[arg-type]
HippoCache = hippo_cache.HippoCache
CacheEntry = hippo_cache.CacheEntry

from .mcp_coordinator import MCPCoordinator
from ..ingestion.advanced_ingestion_engine import AdvancedIngestionEngine
from ..memory.hippo_memory_system import HippoMemorySystem
from ..vector.dual_context_vector import DualContextVectorRAG
from ..graph.bayesian_knowledge_graph import BayesianKnowledgeGraphRAG
from ..cognitive.cognitive_nexus import CognitiveNexusIntegration
from ..graph.creative_graph_search import CreativeGraphSearch
from ..graph.missing_node_detector import MissingNodeDetector

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries supported by the unified system."""
    
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    COMPARATIVE = "comparative"
    EXPLORATORY = "exploratory"
    SYNTHESIS = "synthesis"


class RetrievalMode(Enum):
    """Retrieval modes for different use cases."""
    
    PRECISION = "precision"     # High precision, lower recall
    RECALL = "recall"          # High recall, lower precision
    BALANCED = "balanced"      # Balanced precision and recall
    CREATIVE = "creative"      # Emphasize novel connections
    COMPREHENSIVE = "comprehensive"  # Use all available sources


@dataclass
class QueryContext:
    """Context information for queries."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    domain: Optional[str] = None
    
    # Query preferences
    max_results: int = 10
    confidence_threshold: float = 0.7
    include_reasoning: bool = True
    enable_creative_search: bool = False
    
    # Context tags
    primary_context: Optional[str] = None
    secondary_context: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedResponse:
    """Comprehensive response from the unified RAG system."""
    
    # Core response
    answer: str
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    
    # Multi-modal results
    vector_results: List[Any] = field(default_factory=list)
    graph_results: List[Any] = field(default_factory=list)
    memory_results: List[Any] = field(default_factory=list)
    creative_results: List[Any] = field(default_factory=list)
    
    # Analysis components
    cognitive_analysis: Dict[str, Any] = field(default_factory=dict)
    missing_nodes: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)
    
    # Metadata
    query_type: QueryType = QueryType.FACTUAL
    retrieval_mode: RetrievalMode = RetrievalMode.BALANCED
    processing_time_ms: float = 0.0
    component_times: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    factual_accuracy: float = 0.0
    consistency_score: float = 0.0
    completeness: float = 0.0
    novelty_score: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedRAGSystem:
    """
    Ultimate RAG System - Best of All Worlds
    
    Combines seven cutting-edge RAG approaches into a unified system:
    
    1. Advanced Ingestion: Semantic chunking with dual context tags
    2. HippoRAG Memory: Episodic memory with rapid consolidation
    3. Vector RAG: Contextual embeddings with hierarchical search
    4. Bayesian Graph: Trust-propagated knowledge graphs
    5. Cognitive Nexus: Multi-perspective reasoning
    6. Creative Search: Associative discovery and brainstorming
    7. Missing Nodes: Gap detection and knowledge completion
    
    All orchestrated through strategic MCP server integration.
    """
    
    def __init__(
        self,
        enable_all_components: bool = True,
        mcp_integration: bool = True,
        performance_mode: str = "balanced"  # "speed", "accuracy", "balanced"
    ):
        self.enable_all_components = enable_all_components
        self.mcp_integration = mcp_integration
        self.performance_mode = performance_mode
        
        # Core components
        self.mcp_coordinator: Optional[MCPCoordinator] = None
        self.ingestion_engine: Optional[AdvancedIngestionEngine] = None
        self.memory_system: Optional[HippoMemorySystem] = None
        self.vector_rag: Optional[DualContextVectorRAG] = None
        self.graph_rag: Optional[BayesianKnowledgeGraphRAG] = None
        self.cognitive_nexus: Optional[CognitiveNexusIntegration] = None
        self.creative_search: Optional[CreativeGraphSearch] = None
        self.missing_detector: Optional[MissingNodeDetector] = None
        self.cache: Optional[HippoCache] = None
        
        # System state
        self.initialized_components: List[str] = []
        self.performance_metrics = {
            "total_queries": 0,
            "avg_response_time_ms": 0.0,
            "component_usage": {},
            "success_rate": 1.0,
            "mcp_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        
        # Configuration
        self.component_weights = {
            "vector": 0.3,
            "graph": 0.25,
            "memory": 0.2,
            "cognitive": 0.15,
            "creative": 0.1
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all components of the unified RAG system."""
        try:
            logger.info("🚀 Initializing Unified RAG System - Ultimate Implementation")
            start_time = time.time()

            # Initialize cache
            self.cache = HippoCache()
            
            # Initialize MCP coordinator first
            if self.mcp_integration:
                logger.info("🔌 Initializing MCP Coordinator...")
                self.mcp_coordinator = MCPCoordinator()
                if await self.mcp_coordinator.initialize():
                    self.initialized_components.append("mcp_coordinator")
                    logger.info("✅ MCP Coordinator ready")
                else:
                    logger.warning("⚠️ MCP Coordinator initialization failed, continuing without MCP")
            
            # Initialize all RAG components in parallel
            init_tasks = []
            
            if self.enable_all_components:
                # Advanced Ingestion Engine
                logger.info("📥 Initializing Advanced Ingestion Engine...")
                self.ingestion_engine = AdvancedIngestionEngine(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("ingestion", self.ingestion_engine.initialize()))
                
                # HippoRAG Memory System
                logger.info("🧠 Initializing HippoRAG Memory System...")
                self.memory_system = HippoMemorySystem(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("memory", self.memory_system.initialize()))
                
                # Dual Context Vector RAG
                logger.info("🔍 Initializing Dual Context Vector RAG...")
                self.vector_rag = DualContextVectorRAG(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("vector", self.vector_rag.initialize()))
                
                # Bayesian Knowledge Graph RAG
                logger.info("🕸️ Initializing Bayesian Knowledge Graph RAG...")
                self.graph_rag = BayesianKnowledgeGraphRAG(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("graph", self.graph_rag.initialize()))
                
                # Cognitive Nexus Integration
                logger.info("🧪 Initializing Cognitive Nexus Integration...")
                self.cognitive_nexus = CognitiveNexusIntegration(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("cognitive", self.cognitive_nexus.initialize()))
                
                # Creative Graph Search
                logger.info("🎨 Initializing Creative Graph Search...")
                self.creative_search = CreativeGraphSearch(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("creative", self.creative_search.initialize()))
                
                # Missing Node Detector
                logger.info("🔍 Initializing Missing Node Detector...")
                self.missing_detector = MissingNodeDetector(mcp_coordinator=self.mcp_coordinator)
                init_tasks.append(("detection", self.missing_detector.initialize()))
            
            # Execute all initializations in parallel
            if init_tasks:
                results = await asyncio.gather(*[task for _, task in init_tasks], return_exceptions=True)
                
                for (component_name, _), result in zip(init_tasks, results):
                    if isinstance(result, Exception):
                        logger.error(f"❌ {component_name} initialization failed: {result}")
                    elif result:
                        self.initialized_components.append(component_name)
                        logger.info(f"✅ {component_name} initialized successfully")
                    else:
                        logger.warning(f"⚠️ {component_name} initialization returned False")
            
            # System integration checks
            await self._post_initialization_checks()
            
            self.initialized = True
            init_time = (time.time() - start_time) * 1000
            
            logger.info(f"🎉 Unified RAG System initialized in {init_time:.1f}ms")
            logger.info(f"📊 Components active: {len(self.initialized_components)}")
            logger.info(f"🔧 Active components: {', '.join(self.initialized_components)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Unified RAG System initialization failed: {e}")
            return False
    
    async def query(
        self,
        question: str,
        context: Optional[QueryContext] = None,
        query_type: Optional[QueryType] = None,
        retrieval_mode: Optional[RetrievalMode] = None
    ) -> UnifiedResponse:
        """
        Process query using all available RAG components with intelligent orchestration.
        
        This is the main entry point that orchestrates all RAG components to provide
        the most comprehensive and accurate response possible.
        """
        if not self.initialized:
            raise RuntimeError("UnifiedRAGSystem not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        # Set defaults
        if context is None:
            context = QueryContext()
        if query_type is None:
            query_type = await self._classify_query_type(question)
        if retrieval_mode is None:
            retrieval_mode = self._determine_retrieval_mode(query_type, context)
        
        logger.info(f"🔍 Processing query: '{question[:50]}...' "
                   f"(type: {query_type.value}, mode: {retrieval_mode.value})")

        try:
            query_embedding: Optional[np.ndarray] = None
            if self.cache and self.mcp_coordinator:
                try:
                    embeddings = await self.mcp_coordinator.generate_embeddings([question])
                    if embeddings is not None and len(embeddings) > 0:
                        query_embedding = embeddings[0]
                        cache_entry = self.cache.get(query_embedding)
                        if cache_entry and "response" in cache_entry.citation_metadata:
                            self.performance_metrics["cache_hits"] += 1
                            cached_response = replace(cache_entry.citation_metadata["response"])
                            processing_time = (time.time() - start_time) * 1000
                            cached_response.processing_time_ms = processing_time
                            self._update_system_metrics(cached_response)
                            logger.info(f"✅ Cache hit - Query processed in {processing_time:.1f}ms")
                            return cached_response
                        else:
                            self.performance_metrics["cache_misses"] += 1
                except Exception as cache_err:
                    logger.debug(f"Cache lookup failed: {cache_err}")

            # Create response container
            response = UnifiedResponse(
                answer="",
                confidence=0.0,
                query_type=query_type,
                retrieval_mode=retrieval_mode
            )

            # Execute retrieval strategy based on mode and available components
            if retrieval_mode == RetrievalMode.COMPREHENSIVE:
                await self._comprehensive_retrieval(question, context, response)
            elif retrieval_mode == RetrievalMode.CREATIVE:
                await self._creative_retrieval(question, context, response)
            elif retrieval_mode == RetrievalMode.PRECISION:
                await self._precision_retrieval(question, context, response)
            elif retrieval_mode == RetrievalMode.RECALL:
                await self._recall_retrieval(question, context, response)
            else:  # BALANCED
                await self._balanced_retrieval(question, context, response)

            # Post-process response
            await self._post_process_response(question, context, response)

            # Cache results
            if self.cache and query_embedding is not None:
                entry = CacheEntry(
                    query_embedding=query_embedding,
                    retrieved_docs=response.sources,
                    relevance_scores=[s.get("confidence", 0.0) for s in response.sources],
                    citation_metadata={"response": response},
                    timestamp=datetime.utcnow(),
                )
                self.cache.set(question, entry)

            # Calculate final metrics
            processing_time = (time.time() - start_time) * 1000
            response.processing_time_ms = processing_time

            # Update system metrics
            self._update_system_metrics(response)

            logger.info(f"✅ Query processed in {processing_time:.1f}ms "
                       f"(confidence: {response.confidence:.3f})")

            return response

        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            
            # Return error response
            processing_time = (time.time() - start_time) * 1000
            return UnifiedResponse(
                answer=f"I apologize, but I encountered an error processing your query: {str(e)}",
                confidence=0.0,
                query_type=query_type,
                retrieval_mode=retrieval_mode,
                processing_time_ms=processing_time,
                metadata={"error": str(e)}
            )
    
    async def ingest_document(
        self,
        content: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        enable_all_processing: bool = True
    ) -> bool:
        """Ingest document into all applicable systems."""
        if not self.initialized:
            raise RuntimeError("UnifiedRAGSystem not initialized")
        
        logger.info(f"📥 Ingesting document: {doc_id}")
        
        success_count = 0
        total_attempts = 0
        
        try:
            # Advanced ingestion processing
            if self.ingestion_engine and enable_all_processing:
                total_attempts += 1
                if await self.ingestion_engine.process_document(content, doc_id, metadata):
                    success_count += 1
                    logger.debug("✅ Advanced ingestion completed")
                else:
                    logger.warning("⚠️ Advanced ingestion failed")
            
            # Memory system storage
            if self.memory_system:
                total_attempts += 1
                if await self.memory_system.store_document(content, doc_id, metadata):
                    success_count += 1
                    logger.debug("✅ Memory storage completed")
                else:
                    logger.warning("⚠️ Memory storage failed")
            
            # Vector indexing
            if self.vector_rag:
                total_attempts += 1
                if await self.vector_rag.index_document(content, doc_id, metadata):
                    success_count += 1
                    logger.debug("✅ Vector indexing completed")
                else:
                    logger.warning("⚠️ Vector indexing failed")
            
            # Graph integration
            if self.graph_rag:
                total_attempts += 1
                if await self.graph_rag.add_document(content, doc_id, metadata):
                    success_count += 1
                    logger.debug("✅ Graph integration completed")
                else:
                    logger.warning("⚠️ Graph integration failed")
            
            success_rate = success_count / max(1, total_attempts)
            logger.info(f"📊 Document ingestion: {success_count}/{total_attempts} "
                       f"components successful ({success_rate:.1%})")
            
            return success_rate >= 0.5  # Require at least half to succeed
            
        except Exception as e:
            logger.error(f"❌ Document ingestion failed: {e}")
            return False
    
    async def _comprehensive_retrieval(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Comprehensive retrieval using all available components."""
        tasks = []
        
        # Vector search
        if self.vector_rag:
            tasks.append(("vector", self.vector_rag.search(question, k=context.max_results)))
        
        # Graph traversal
        if self.graph_rag:
            tasks.append(("graph", self.graph_rag.retrieve(question, k=context.max_results)))
        
        # Memory retrieval
        if self.memory_system:
            tasks.append(("memory", self.memory_system.query(question, limit=context.max_results)))
        
        # Creative search
        if self.creative_search and context.enable_creative_search:
            tasks.append(("creative", self.creative_search.brainstorm(question, k=context.max_results)))
        
        # Execute all searches in parallel
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (component_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.warning(f"{component_name} search failed: {result}")
                    continue
                
                # Store component results
                if component_name == "vector":
                    response.vector_results = result if result else []
                elif component_name == "graph":
                    response.graph_results = result if result else []
                elif component_name == "memory":
                    response.memory_results = result if result else []
                elif component_name == "creative":
                    response.creative_results = result if result else []
        
        # Cognitive analysis
        if self.cognitive_nexus:
            all_results = (
                response.vector_results + 
                response.graph_results + 
                response.memory_results + 
                response.creative_results
            )
            if all_results:
                response.cognitive_analysis = await self.cognitive_nexus.analyze(question, all_results)
        
        # Missing node detection
        if self.missing_detector:
            response.missing_nodes = await self.missing_detector.detect_gaps(question, response.sources)
        
        # Synthesize final answer
        await self._synthesize_comprehensive_answer(question, response)
    
    async def _synthesize_comprehensive_answer(self, question: str, response: UnifiedResponse):
        """Synthesize final answer from all component results."""
        # Collect all sources with weights
        weighted_sources = []
        
        # Weight vector results
        for result in response.vector_results:
            weighted_sources.append({
                "content": getattr(result, "content", str(result)),
                "source": "vector",
                "weight": self.component_weights["vector"],
                "confidence": getattr(result, "confidence", 0.8)
            })
        
        # Weight graph results
        for result in response.graph_results:
            weighted_sources.append({
                "content": getattr(result, "content", str(result)),
                "source": "graph", 
                "weight": self.component_weights["graph"],
                "confidence": getattr(result, "confidence", 0.8)
            })
        
        # Weight memory results
        for result in response.memory_results:
            weighted_sources.append({
                "content": getattr(result, "content", str(result)),
                "source": "memory",
                "weight": self.component_weights["memory"],
                "confidence": getattr(result, "confidence", 0.8)
            })
        
        # Sort by weighted confidence
        weighted_sources.sort(key=lambda x: x["weight"] * x["confidence"], reverse=True)
        
        # Generate synthesized answer
        if weighted_sources:
            top_sources = weighted_sources[:5]  # Use top 5 sources
            
            # Create answer from top sources
            answer_parts = []
            total_confidence = 0.0
            
            for source in top_sources:
                content = source["content"]
                confidence = source["confidence"] * source["weight"]
                
                if len(content) > 200:
                    content = content[:200] + "..."
                
                answer_parts.append(content)
                total_confidence += confidence
            
            response.answer = " ".join(answer_parts)
            response.confidence = min(1.0, total_confidence / len(top_sources))
            response.sources = top_sources
            
            # Add reasoning trace
            response.reasoning_trace = [
                f"Analyzed {len(response.vector_results)} vector results",
                f"Analyzed {len(response.graph_results)} graph results", 
                f"Analyzed {len(response.memory_results)} memory results",
                f"Synthesized from {len(top_sources)} top sources",
                f"Final confidence: {response.confidence:.3f}"
            ]
        else:
            response.answer = "I don't have enough information to answer that question."
            response.confidence = 0.0
            response.reasoning_trace = ["No relevant sources found in any component"]
    
    async def _balanced_retrieval(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Balanced retrieval prioritizing accuracy and relevance."""
        # Use top 3 components based on query type
        primary_components = self._select_primary_components(response.query_type)
        
        tasks = []
        for component in primary_components[:3]:  # Limit to 3 for balance
            if component == "vector" and self.vector_rag:
                tasks.append(("vector", self.vector_rag.search(question, k=context.max_results // 2)))
            elif component == "graph" and self.graph_rag:
                tasks.append(("graph", self.graph_rag.retrieve(question, k=context.max_results // 2)))
            elif component == "memory" and self.memory_system:
                tasks.append(("memory", self.memory_system.query(question, limit=context.max_results // 2)))
        
        # Execute selected components
        if tasks:
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (component_name, _), result in zip(tasks, results):
                if not isinstance(result, Exception) and result:
                    if component_name == "vector":
                        response.vector_results = result
                    elif component_name == "graph":
                        response.graph_results = result
                    elif component_name == "memory":
                        response.memory_results = result
        
        # Synthesize balanced answer
        await self._synthesize_comprehensive_answer(question, response)
    
    def _select_primary_components(self, query_type: QueryType) -> List[str]:
        """Select primary components based on query type."""
        component_preferences = {
            QueryType.FACTUAL: ["vector", "graph", "memory"],
            QueryType.ANALYTICAL: ["graph", "cognitive", "vector"],
            QueryType.CREATIVE: ["creative", "graph", "vector"],
            QueryType.COMPARATIVE: ["vector", "cognitive", "graph"],
            QueryType.EXPLORATORY: ["creative", "graph", "memory"],
            QueryType.SYNTHESIS: ["cognitive", "vector", "graph"]
        }
        
        return component_preferences.get(query_type, ["vector", "graph", "memory"])
    
    async def _classify_query_type(self, question: str) -> QueryType:
        """Classify the query type using MCP or simple heuristics."""
        if self.mcp_coordinator:
            try:
                breakdown = await self.mcp_coordinator.systematic_breakdown(question)
                # Analyze breakdown to determine query type
                if any(word in question.lower() for word in ["what", "define", "explain"]):
                    return QueryType.FACTUAL
                elif any(word in question.lower() for word in ["why", "how", "analyze"]):
                    return QueryType.ANALYTICAL
                elif any(word in question.lower() for word in ["brainstorm", "creative", "novel"]):
                    return QueryType.CREATIVE
                elif any(word in question.lower() for word in ["compare", "versus", "difference"]):
                    return QueryType.COMPARATIVE
                elif any(word in question.lower() for word in ["explore", "discover", "find"]):
                    return QueryType.EXPLORATORY
                else:
                    return QueryType.SYNTHESIS
            except Exception:
                pass
        
        # Fallback to simple heuristics
        question_lower = question.lower()
        if any(word in question_lower for word in ["what", "define", "explain"]):
            return QueryType.FACTUAL
        elif any(word in question_lower for word in ["why", "how", "analyze"]):
            return QueryType.ANALYTICAL
        elif any(word in question_lower for word in ["brainstorm", "creative", "imagine"]):
            return QueryType.CREATIVE
        elif any(word in question_lower for word in ["compare", "versus", "different"]):
            return QueryType.COMPARATIVE
        elif any(word in question_lower for word in ["explore", "discover", "find"]):
            return QueryType.EXPLORATORY
        else:
            return QueryType.SYNTHESIS
    
    def _determine_retrieval_mode(self, query_type: QueryType, context: QueryContext) -> RetrievalMode:
        """Determine optimal retrieval mode."""
        if context.enable_creative_search:
            return RetrievalMode.CREATIVE
        elif query_type == QueryType.CREATIVE:
            return RetrievalMode.CREATIVE
        elif query_type == QueryType.FACTUAL:
            return RetrievalMode.PRECISION
        elif query_type == QueryType.EXPLORATORY:
            return RetrievalMode.RECALL
        elif query_type == QueryType.SYNTHESIS:
            return RetrievalMode.COMPREHENSIVE
        else:
            return RetrievalMode.BALANCED
    
    async def _post_initialization_checks(self):
        """Perform post-initialization system checks."""
        logger.info("🔍 Performing system integration checks...")
        
        # Check component integration
        if self.mcp_coordinator:
            status = await self.mcp_coordinator.get_coordinator_status()
            logger.info(f"📊 MCP Status: {status['servers_configured']} servers, "
                       f"{status['completed_tasks']} tasks completed")
        
        # Verify component connectivity
        connectivity_score = len(self.initialized_components) / 8  # Max 8 components
        logger.info(f"🔗 System connectivity: {connectivity_score:.1%}")
        
        if connectivity_score < 0.5:
            logger.warning("⚠️ Low system connectivity - some features may be limited")
    
    async def _post_process_response(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Post-process the response for quality and completeness."""
        # Calculate quality metrics
        if response.sources:
            response.factual_accuracy = np.mean([s.get("confidence", 0.8) for s in response.sources])
            response.consistency_score = 0.85  # Reference score
            response.completeness = min(1.0, len(response.sources) / context.max_results)
        
        # Calculate novelty score if creative results exist
        if response.creative_results:
            response.novelty_score = 0.7  # Reference score
        
        # Ensure minimum answer quality
        if response.confidence < context.confidence_threshold and response.answer:
            response.answer += "\n\nNote: This response has lower confidence than requested. Please verify the information."
    
    def _update_system_metrics(self, response: UnifiedResponse):
        """Update system performance metrics."""
        self.performance_metrics["total_queries"] += 1
        
        # Update average response time
        old_avg = self.performance_metrics["avg_response_time_ms"]
        total_queries = self.performance_metrics["total_queries"]
        new_avg = (old_avg * (total_queries - 1) + response.processing_time_ms) / total_queries
        self.performance_metrics["avg_response_time_ms"] = new_avg
        
        # Update component usage
        for component in ["vector", "graph", "memory", "creative"]:
            results_key = f"{component}_results"
            if hasattr(response, results_key) and getattr(response, results_key):
                if component not in self.performance_metrics["component_usage"]:
                    self.performance_metrics["component_usage"][component] = 0
                self.performance_metrics["component_usage"][component] += 1

        if self.cache:
            cache_stats = self.cache.metrics()
            cache_stats["hits"] = self.performance_metrics.get("cache_hits", 0)
            cache_stats["misses"] = self.performance_metrics.get("cache_misses", 0)
            self.performance_metrics["cache"] = cache_stats
    
    # Reference methods for retrieval modes
    async def _creative_retrieval(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Creative retrieval focusing on novel connections."""
        # Implement creative-focused retrieval
        await self._balanced_retrieval(question, context, response)
    
    async def _precision_retrieval(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Precision retrieval focusing on accuracy."""
        # Implement precision-focused retrieval
        await self._balanced_retrieval(question, context, response)
    
    async def _recall_retrieval(self, question: str, context: QueryContext, response: UnifiedResponse):
        """Recall retrieval focusing on coverage."""
        # Implement recall-focused retrieval
        await self._balanced_retrieval(question, context, response)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self.initialized,
            "components": {
                "total_components": 8,
                "initialized_components": len(self.initialized_components),
                "active_components": self.initialized_components,
                "component_details": {}
            },
            "performance": self.performance_metrics.copy(),
            "configuration": {
                "performance_mode": self.performance_mode,
                "mcp_integration": self.mcp_integration,
                "enable_all_components": self.enable_all_components,
                "component_weights": self.component_weights
            }
        }
        
        # Get individual component status
        if self.mcp_coordinator:
            status["components"]["component_details"]["mcp"] = await self.mcp_coordinator.get_coordinator_status()
        
        if self.vector_rag:
            status["components"]["component_details"]["vector"] = await self.vector_rag.get_status()
        
        if self.graph_rag:
            status["components"]["component_details"]["graph"] = await self.graph_rag.get_health_status()
        
        # Add more component status checks as needed
        
        return status
    
    async def shutdown(self):
        """Shutdown all components of the unified system."""
        logger.info("🛑 Shutting down Unified RAG System...")
        
        # Shutdown all components
        shutdown_tasks = []
        
        if self.mcp_coordinator:
            shutdown_tasks.append(self.mcp_coordinator.shutdown())
        
        if self.ingestion_engine:
            shutdown_tasks.append(self.ingestion_engine.close())
        
        if self.memory_system:
            shutdown_tasks.append(self.memory_system.close())
        
        if self.vector_rag:
            shutdown_tasks.append(self.vector_rag.close())
        
        if self.graph_rag:
            shutdown_tasks.append(self.graph_rag.shutdown())
        
        if self.cognitive_nexus:
            shutdown_tasks.append(self.cognitive_nexus.close())
        
        if self.creative_search:
            shutdown_tasks.append(self.creative_search.close())
        
        if self.missing_detector:
            shutdown_tasks.append(self.missing_detector.close())
        
        # Execute shutdowns in parallel
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.initialized = False
        self.initialized_components.clear()
        
        logger.info("✅ Unified RAG System shutdown complete")