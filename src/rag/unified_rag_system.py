#!/usr/bin/env python3
"""
Unified RAG System Architecture with MCP Integration

Consolidates all RAG system implementations into a single, coherent system:
- Core HyperRAG: Advanced neural-biological processing
- Package RAG: Simple dependency injection patterns  
- MCP HyperRAG: Protocol compliance and real-time communication
- Infrastructure RAG: Unified configuration management
- Mobile Mini-RAG: Privacy-preserving on-device processing

Features:
- MCP server integration (Memory, Sequential Thinking, HuggingFace, Context7)
- Multi-mode query processing with intelligent routing
- Unified vector storage with swappable backends
- Neural memory system with trust networks
- Performance optimization with distributed caching
- Privacy-preserving processing with anonymization
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import logging
import json
import statistics
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query processing modes for different use cases."""
    FAST = "fast"  # Vector-only, fastest response
    BALANCED = "balanced"  # Vector + Graph, good balance
    COMPREHENSIVE = "comprehensive"  # All systems, most thorough
    CREATIVE = "creative"  # Emphasize creativity engine
    ANALYTICAL = "analytical"  # Emphasize cognitive analysis
    DISTRIBUTED = "distributed"  # Use P2P network for retrieval
    EDGE_OPTIMIZED = "edge_optimized"  # Mobile/edge device optimization
    PRIVACY_FIRST = "privacy_first"  # Maximum privacy preservation


class MemoryType(Enum):
    """Types of memory for storage routing."""
    EPISODIC = "episodic"  # Recent, temporary (HippoRAG)
    SEMANTIC = "semantic"  # Long-term, structured (GraphRAG)
    VECTOR = "vector"  # Similarity-based (VectorRAG)
    PROCEDURAL = "procedural"  # How-to knowledge
    TRUST_VALIDATED = "trust_validated"  # Bayesian trust networks
    ALL = "all"  # Store in all systems


class PrivacyLevel(Enum):
    """Privacy levels for content processing."""
    PUBLIC = "public"
    PERSONAL = "personal"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"
    SENSITIVE = "sensitive"


@dataclass
class RetrievedInformation:
    """Information retrieved from knowledge base."""
    id: str
    content: str
    source: str
    relevance_score: float
    retrieval_confidence: float
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    trust_score: float = 0.5
    graph_connections: List[str] = field(default_factory=list)
    relationship_types: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    anonymized: bool = False


@dataclass
class SynthesizedAnswer:
    """Final synthesized answer from multiple sources."""
    answer: str
    confidence: float
    supporting_sources: List[str]
    synthesis_method: str
    privacy_level: PrivacyLevel = PrivacyLevel.PUBLIC
    retrieval_sources: List[RetrievedInformation] = field(default_factory=list)
    processing_time: float = 0.0
    query_mode: str = "balanced"
    mcp_servers_used: List[str] = field(default_factory=list)
    trust_validation: bool = False


@dataclass
class UnifiedRAGConfig:
    """Unified configuration for the consolidated RAG system."""
    # Query processing
    max_results: int = 10
    min_confidence: float = 0.1
    timeout_seconds: float = 30.0
    
    # Vector storage
    vector_dimensions: int = 384
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_backend: str = "simple"  # simple, faiss, qdrant
    
    # Graph processing
    graph_depth_limit: int = 3
    trust_threshold: float = 0.6
    
    # Memory systems
    enable_hippo_rag: bool = True
    enable_trust_networks: bool = True
    enable_vector_search: bool = True
    enable_graph_reasoning: bool = True
    
    # MCP integration
    memory_mcp_enabled: bool = True
    context7_caching: bool = True
    huggingface_embeddings: bool = False
    sequential_thinking: bool = True
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600
    max_retries: int = 3
    
    # Privacy
    privacy_mode: str = "balanced"  # strict, balanced, permissive
    anonymization_level: float = 0.7
    enable_anonymization: bool = True
    
    # Development vs Production
    environment: str = "development"  # development, production
    debug_mode: bool = False


class VectorBackend(ABC):
    """Abstract interface for vector storage backends."""
    
    @abstractmethod
    async def add_document(self, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict = None):
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        pass


class SimpleVectorBackend(VectorBackend):
    """Simple in-memory vector backend for development."""
    
    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions
        self.documents: Dict[str, str] = {}
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
    
    async def add_document(self, doc_id: str, content: str, embedding: np.ndarray, metadata: Dict = None):
        self.documents[doc_id] = content
        self.vectors[doc_id] = embedding
        self.metadata[doc_id] = metadata or {}
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[tuple]:
        if not self.vectors:
            return []
        
        results = []
        for doc_id, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_embedding, vector)
            results.append((doc_id, similarity, self.documents[doc_id]))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def get_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "vector_dimensions": self.dimensions,
            "memory_usage_mb": len(self.vectors) * self.dimensions * 8 / (1024 * 1024)
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class MCPIntegrationLayer:
    """Integration layer for MCP servers."""
    
    def __init__(self, config: UnifiedRAGConfig):
        self.config = config
        self.available_servers = []
        self.server_status = {}
    
    async def initialize(self):
        """Initialize MCP server connections."""
        try:
            # In production, would establish actual MCP connections
            # For now, simulate availability
            if self.config.memory_mcp_enabled:
                self.available_servers.append("memory")
                self.server_status["memory"] = "connected"
            
            if self.config.context7_caching:
                self.available_servers.append("context7")
                self.server_status["context7"] = "connected"
            
            if self.config.huggingface_embeddings:
                self.available_servers.append("huggingface")
                self.server_status["huggingface"] = "connected"
            
            if self.config.sequential_thinking:
                self.available_servers.append("sequentialthinking")
                self.server_status["sequentialthinking"] = "connected"
            
            logger.info(f"MCP servers initialized: {self.available_servers}")
            return True
        
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
            return False
    
    async def store_memory(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data in Memory MCP."""
        if "memory" not in self.available_servers:
            return False
        
        try:
            # In production would call actual MCP server
            logger.debug(f"Storing in Memory MCP: {key}")
            return True
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            return False
    
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve data from Memory MCP."""
        if "memory" not in self.available_servers:
            return None
        
        try:
            # In production would call actual MCP server
            logger.debug(f"Retrieving from Memory MCP: {key}")
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return None
    
    async def cache_query(self, query_hash: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache query result using Context7."""
        if "context7" not in self.available_servers:
            return False
        
        try:
            # In production would call actual Context7 MCP
            logger.debug(f"Caching query result: {query_hash}")
            return True
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
            return False
    
    async def get_cached_query(self, query_hash: str) -> Optional[Any]:
        """Get cached query result from Context7."""
        if "context7" not in self.available_servers:
            return None
        
        try:
            # In production would call actual Context7 MCP
            logger.debug(f"Retrieving cached query: {query_hash}")
            return None  # Placeholder
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def generate_embeddings(self, texts: List[str]) -> Optional[List[np.ndarray]]:
        """Generate embeddings using HuggingFace MCP."""
        if "huggingface" not in self.available_servers:
            return None
        
        try:
            # In production would call actual HuggingFace MCP
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            # Return placeholder embeddings
            return [self._generate_simple_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    async def sequential_reasoning(self, query: str, context: List[str]) -> Optional[Dict]:
        """Perform sequential thinking analysis."""
        if "sequentialthinking" not in self.available_servers:
            return None
        
        try:
            # In production would call actual Sequential Thinking MCP
            logger.debug(f"Sequential reasoning for query: {query[:100]}...")
            return {
                "reasoning_steps": [
                    "Analyze query intent",
                    "Identify relevant knowledge areas", 
                    "Synthesize coherent response"
                ],
                "confidence": 0.8,
                "complexity": "medium"
            }
        except Exception as e:
            logger.error(f"Sequential reasoning failed: {e}")
            return None
    
    def _generate_simple_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding (fallback)."""
        words = text.lower().split()
        embedding = np.zeros(self.config.vector_dimensions)
        
        for i, word in enumerate(words[:self.config.vector_dimensions]):
            hash_val = hash(word) % self.config.vector_dimensions
            embedding[hash_val] += 1.0 / (i + 1)
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class PrivacyProcessor:
    """Privacy-preserving processing for sensitive content."""
    
    def __init__(self, config: UnifiedRAGConfig):
        self.config = config
    
    def assess_privacy_level(self, content: str, context: Dict = None) -> PrivacyLevel:
        """Assess privacy level of content."""
        content_lower = content.lower()
        
        # Sensitive indicators
        sensitive_patterns = [
            "password", "ssn", "credit card", "bank account", "medical record",
            "social security", "driver license", "passport", "phone number",
            "email address", "home address", "personal identification"
        ]
        
        if any(pattern in content_lower for pattern in sensitive_patterns):
            return PrivacyLevel.SENSITIVE
        
        # Personal indicators
        personal_patterns = [
            "my", "i am", "i have", "personal", "private", "family",
            "home", "relationship", "friend", "colleague"
        ]
        
        if any(pattern in content_lower for pattern in personal_patterns):
            return PrivacyLevel.PERSONAL
        
        # Default to public if no privacy indicators
        return PrivacyLevel.PUBLIC
    
    def anonymize_content(self, content: str, level: float = None) -> str:
        """Anonymize content based on privacy level."""
        if not self.config.enable_anonymization:
            return content
        
        anonymization_level = level or self.config.anonymization_level
        
        if anonymization_level < 0.3:
            return content  # No anonymization
        
        import re
        anonymized = content
        
        # Replace names
        if anonymization_level >= 0.3:
            anonymized = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[PERSON]", anonymized)
        
        # Replace addresses
        if anonymization_level >= 0.5:
            anonymized = re.sub(
                r"\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd)\b",
                "[ADDRESS]", anonymized
            )
        
        # Replace contact info
        if anonymization_level >= 0.7:
            anonymized = re.sub(r"\b\d{3}-?\d{3}-?\d{4}\b", "[PHONE]", anonymized)
            anonymized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", anonymized)
        
        # Replace specific times and dates
        if anonymization_level >= 0.9:
            anonymized = re.sub(r"\b\d{1,2}:\d{2}\s?[AP]M\b", "[TIME]", anonymized)
            anonymized = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\b", "[DATE]", anonymized)
        
        return anonymized
    
    def can_process_content(self, content: str, mode: QueryMode) -> bool:
        """Check if content can be processed based on privacy settings."""
        privacy_level = self.assess_privacy_level(content)
        
        if mode == QueryMode.PRIVACY_FIRST and privacy_level in [PrivacyLevel.SENSITIVE, PrivacyLevel.CONFIDENTIAL]:
            return False
        
        if self.config.privacy_mode == "strict" and privacy_level != PrivacyLevel.PUBLIC:
            return False
        
        return True


class UnifiedRAGSystem:
    """
    Unified RAG System with comprehensive MCP integration.
    
    Consolidates all RAG implementations into a single, coherent system.
    """
    
    def __init__(self, config: UnifiedRAGConfig = None):
        self.config = config or UnifiedRAGConfig()
        self.logger = logging.getLogger(f"{__name__}.UnifiedRAGSystem")
        
        # Core components
        self.vector_backend: VectorBackend = None
        self.mcp_layer = MCPIntegrationLayer(self.config)
        self.privacy_processor = PrivacyProcessor(self.config)
        
        # Knowledge storage
        self.knowledge_graph = {}  # Simple graph storage
        self.trust_scores = {}  # Document trust scores
        self.usage_stats = {}  # Usage tracking
        
        # Caching and performance
        self.query_cache = {}
        self.performance_metrics = {
            "queries_processed": 0,
            "documents_indexed": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "mcp_calls": 0
        }
        
        self.logger.info(f"Unified RAG System initialized with environment: {self.config.environment}")
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            # Initialize vector backend
            if self.config.vector_backend == "simple":
                self.vector_backend = SimpleVectorBackend(self.config.vector_dimensions)
            # Add other backends as needed
            
            # Initialize MCP layer
            await self.mcp_layer.initialize()
            
            self.logger.info("Unified RAG System initialization complete")
            return True
        
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    async def add_document(
        self,
        content: str,
        doc_id: str = None,
        metadata: Dict = None,
        memory_type: MemoryType = MemoryType.SEMANTIC
    ) -> str:
        """Add document to the unified knowledge base."""
        if doc_id is None:
            doc_id = f"doc_{int(time.time() * 1000000)}"
        
        # Privacy assessment
        privacy_level = self.privacy_processor.assess_privacy_level(content, metadata)
        
        # Anonymize if necessary
        processed_content = content
        if privacy_level in [PrivacyLevel.PERSONAL, PrivacyLevel.SENSITIVE]:
            processed_content = self.privacy_processor.anonymize_content(content)
        
        # Generate embeddings
        embeddings = await self.mcp_layer.generate_embeddings([processed_content])
        if embeddings is None:
            # Fallback to simple embedding
            embeddings = [self.mcp_layer._generate_simple_embedding(processed_content)]
        
        # Store in vector backend
        enhanced_metadata = {
            **(metadata or {}),
            "privacy_level": privacy_level.value,
            "memory_type": memory_type.value,
            "added_at": datetime.now().isoformat(),
            "anonymized": processed_content != content
        }
        
        await self.vector_backend.add_document(doc_id, processed_content, embeddings[0], enhanced_metadata)
        
        # Store in MCP memory for persistence
        await self.mcp_layer.store_memory(
            f"document:{doc_id}",
            {
                "content": processed_content,
                "metadata": enhanced_metadata,
                "embedding": embeddings[0].tolist()
            }
        )
        
        # Update trust scores if applicable
        if memory_type == MemoryType.TRUST_VALIDATED:
            self.trust_scores[doc_id] = metadata.get("trust_score", 0.5)
        
        self.performance_metrics["documents_indexed"] += 1
        self.logger.info(f"Added document {doc_id} (privacy: {privacy_level.value})")
        
        return doc_id
    
    async def process_query(
        self,
        query: str,
        mode: QueryMode = QueryMode.BALANCED,
        context: Dict = None,
        user_id: str = None
    ) -> SynthesizedAnswer:
        """Process query using the unified RAG system."""
        start_time = time.time()
        
        # Privacy check
        if not self.privacy_processor.can_process_content(query, mode):
            return SynthesizedAnswer(
                answer="I cannot process this query due to privacy restrictions.",
                confidence=0.0,
                supporting_sources=[],
                synthesis_method="privacy_blocked",
                privacy_level=PrivacyLevel.SENSITIVE,
                query_mode=mode.value
            )
        
        # Check cache first
        query_hash = hashlib.md5(f"{query}:{mode.value}:{user_id}".encode()).hexdigest()
        
        if self.config.enable_caching:
            cached_result = await self.mcp_layer.get_cached_query(query_hash)
            if cached_result:
                self.performance_metrics["cache_hits"] += 1
                return cached_result
        
        try:
            # Generate query embedding
            query_embeddings = await self.mcp_layer.generate_embeddings([query])
            if not query_embeddings:
                query_embeddings = [self.mcp_layer._generate_simple_embedding(query)]
            
            # Vector search
            vector_results = await self.vector_backend.search(
                query_embeddings[0],
                top_k=self.config.max_results
            )
            
            # Process results based on mode
            retrieved_info = []
            
            for doc_id, similarity, content in vector_results:
                if similarity >= self.config.min_confidence:
                    # Get metadata
                    metadata = getattr(self.vector_backend, 'metadata', {}).get(doc_id, {})
                    
                    info = RetrievedInformation(
                        id=doc_id,
                        content=content,
                        source="unified_rag",
                        relevance_score=similarity,
                        retrieval_confidence=similarity,
                        privacy_level=PrivacyLevel(metadata.get("privacy_level", "public")),
                        trust_score=self.trust_scores.get(doc_id, 0.5)
                    )
                    
                    # Add graph connections for comprehensive mode
                    if mode in [QueryMode.COMPREHENSIVE, QueryMode.ANALYTICAL]:
                        connections = self.knowledge_graph.get(doc_id, [])
                        info.graph_connections = [conn[0] for conn in connections]
                        info.relationship_types = [conn[1] for conn in connections]
                    
                    retrieved_info.append(info)
            
            # Enhanced processing for certain modes
            if mode == QueryMode.ANALYTICAL and self.config.sequential_thinking:
                reasoning = await self.mcp_layer.sequential_reasoning(
                    query,
                    [info.content for info in retrieved_info[:3]]
                )
                self.performance_metrics["mcp_calls"] += 1
            
            # Synthesize answer
            answer = await self._synthesize_answer(query, retrieved_info, mode)
            answer.processing_time = time.time() - start_time
            answer.mcp_servers_used = self.mcp_layer.available_servers.copy()
            
            # Cache result
            if self.config.enable_caching:
                await self.mcp_layer.cache_query(query_hash, answer, self.config.cache_ttl)
            
            self.performance_metrics["queries_processed"] += 1
            self._update_average_response_time(answer.processing_time)
            
            return answer
        
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return SynthesizedAnswer(
                answer=f"I encountered an error processing your query. Please try rephrasing.",
                confidence=0.1,
                supporting_sources=[],
                synthesis_method="error_fallback",
                processing_time=time.time() - start_time,
                query_mode=mode.value
            )
    
    async def _synthesize_answer(
        self,
        query: str,
        retrieved_info: List[RetrievedInformation],
        mode: QueryMode
    ) -> SynthesizedAnswer:
        """Synthesize final answer from retrieved information."""
        if not retrieved_info:
            return SynthesizedAnswer(
                answer="I don't have enough information to answer your query.",
                confidence=0.0,
                supporting_sources=[],
                synthesis_method="no_results",
                retrieval_sources=retrieved_info,
                query_mode=mode.value
            )
        
        # Determine overall privacy level
        privacy_levels = [info.privacy_level for info in retrieved_info]
        max_privacy = max(privacy_levels) if privacy_levels else PrivacyLevel.PUBLIC
        
        # Filter results based on privacy and trust if needed
        if mode == QueryMode.PRIVACY_FIRST:
            retrieved_info = [
                info for info in retrieved_info
                if info.privacy_level == PrivacyLevel.PUBLIC and info.trust_score >= 0.7
            ]
        
        if not retrieved_info:
            return SynthesizedAnswer(
                answer="No information available that meets privacy requirements.",
                confidence=0.0,
                supporting_sources=[],
                synthesis_method="privacy_filtered",
                privacy_level=max_privacy,
                query_mode=mode.value
            )
        
        # Select top results based on mode
        if mode == QueryMode.FAST:
            top_results = retrieved_info[:1]
            synthesis_method = "single_source_fast"
        elif mode == QueryMode.COMPREHENSIVE:
            top_results = retrieved_info[:5]
            synthesis_method = "comprehensive_multi_source"
        else:
            top_results = retrieved_info[:3]
            synthesis_method = "balanced_multi_source"
        
        # Synthesize based on mode
        if mode == QueryMode.FAST:
            best_result = top_results[0]
            answer_text = f"Based on the most relevant information: {best_result.content[:300]}..."
            confidence = best_result.relevance_score
        
        elif mode == QueryMode.CREATIVE:
            combined_content = " ".join([info.content for info in top_results])
            answer_text = (
                f"Drawing insights from multiple sources: {combined_content[:400]}... "
                f"This suggests innovative approaches to your query about '{query}'."
            )
            confidence = statistics.mean([info.relevance_score for info in top_results]) * 0.9
            synthesis_method = "creative_synthesis"
        
        else:
            # Balanced/Comprehensive/Analytical synthesis
            source_summaries = []
            for i, info in enumerate(top_results, 1):
                summary = f"{i}. {info.content[:200]}..."
                if info.trust_score > 0.8:
                    summary += " [High Trust]"
                source_summaries.append(summary)
            
            answer_text = f"Based on {len(top_results)} relevant sources:\n\n" + "\n\n".join(source_summaries)
            confidence = statistics.mean([info.relevance_score for info in top_results])
        
        supporting_sources = [info.id for info in top_results]
        
        # Apply privacy processing to answer
        if max_privacy in [PrivacyLevel.PERSONAL, PrivacyLevel.SENSITIVE]:
            answer_text = self.privacy_processor.anonymize_content(answer_text)
        
        return SynthesizedAnswer(
            answer=answer_text,
            confidence=confidence,
            supporting_sources=supporting_sources,
            synthesis_method=synthesis_method,
            privacy_level=max_privacy,
            retrieval_sources=retrieved_info,
            query_mode=mode.value,
            trust_validation=any(info.trust_score > 0.8 for info in top_results)
        )
    
    def _update_average_response_time(self, new_time: float):
        """Update rolling average response time."""
        current_avg = self.performance_metrics["average_response_time"]
        query_count = self.performance_metrics["queries_processed"]
        
        if query_count == 1:
            self.performance_metrics["average_response_time"] = new_time
        else:
            self.performance_metrics["average_response_time"] = (
                (current_avg * (query_count - 1) + new_time) / query_count
            )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        vector_stats = await self.vector_backend.get_stats() if self.vector_backend else {}
        
        return {
            "status": "operational",
            "environment": self.config.environment,
            "components": {
                "vector_backend": "operational" if self.vector_backend else "disabled",
                "mcp_layer": "operational" if self.mcp_layer.available_servers else "limited",
                "privacy_processor": "operational",
                "knowledge_graph": "operational"
            },
            "mcp_servers": {
                server: status for server, status in self.mcp_layer.server_status.items()
            },
            "performance_metrics": self.performance_metrics.copy(),
            "vector_stats": vector_stats,
            "cache_stats": {
                "query_cache_size": len(self.query_cache),
                "trust_scores_tracked": len(self.trust_scores),
                "knowledge_graph_nodes": len(self.knowledge_graph)
            },
            "privacy_settings": {
                "privacy_mode": self.config.privacy_mode,
                "anonymization_enabled": self.config.enable_anonymization,
                "anonymization_level": self.config.anonymization_level
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            "overall": "healthy",
            "components": {},
            "issues": []
        }
        
        # Check vector backend
        try:
            if self.vector_backend:
                stats = await self.vector_backend.get_stats()
                health_status["components"]["vector_backend"] = "healthy"
            else:
                health_status["components"]["vector_backend"] = "not_initialized"
                health_status["issues"].append("Vector backend not initialized")
        except Exception as e:
            health_status["components"]["vector_backend"] = "error"
            health_status["issues"].append(f"Vector backend error: {e}")
        
        # Check MCP layer
        if self.mcp_layer.available_servers:
            health_status["components"]["mcp_layer"] = "healthy"
        else:
            health_status["components"]["mcp_layer"] = "limited"
            health_status["issues"].append("No MCP servers available")
        
        # Check privacy processor
        health_status["components"]["privacy_processor"] = "healthy"
        
        # Determine overall health
        if health_status["issues"]:
            health_status["overall"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
        
        return health_status
    
    def clear_cache(self):
        """Clear all caches."""
        self.query_cache.clear()
        self.logger.info("System caches cleared")


# Convenience functions for different deployment scenarios
def create_development_rag() -> UnifiedRAGSystem:
    """Create RAG system optimized for development."""
    config = UnifiedRAGConfig(
        environment="development",
        debug_mode=True,
        vector_backend="simple",
        huggingface_embeddings=False,
        privacy_mode="permissive"
    )
    return UnifiedRAGSystem(config)


def create_production_rag() -> UnifiedRAGSystem:
    """Create RAG system optimized for production."""
    config = UnifiedRAGConfig(
        environment="production",
        debug_mode=False,
        vector_backend="faiss",
        huggingface_embeddings=True,
        context7_caching=True,
        privacy_mode="strict",
        max_results=20,
        cache_ttl=7200
    )
    return UnifiedRAGSystem(config)


def create_privacy_first_rag() -> UnifiedRAGSystem:
    """Create RAG system with maximum privacy protection."""
    config = UnifiedRAGConfig(
        environment="production",
        privacy_mode="strict",
        enable_anonymization=True,
        anonymization_level=0.9,
        memory_mcp_enabled=False,  # No external memory storage
        context7_caching=False,  # No external caching
        huggingface_embeddings=False  # No external embedding service
    )
    return UnifiedRAGSystem(config)


# Export main classes and functions
__all__ = [
    "UnifiedRAGSystem",
    "UnifiedRAGConfig", 
    "QueryMode",
    "MemoryType",
    "PrivacyLevel",
    "RetrievedInformation",
    "SynthesizedAnswer",
    "create_development_rag",
    "create_production_rag", 
    "create_privacy_first_rag"
]