"""
Base RAG System Implementation

Defines the abstract base class that all RAG systems must inherit from.
Provides common functionality and enforces the interface contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..interfaces import (
    KnowledgeRetrievalInterface,
    MemoryInterface, 
    ReasoningInterface,
    SynthesisInterface,
    QueryMode,
    QueryContext,
    RetrievalResult
)


class RAGType(Enum):
    """Types of RAG systems"""
    BASE = "base"
    HYPER = "hyper"
    MINI = "mini"
    CUSTOM = "custom"


class ProcessingMode(Enum):
    """RAG processing modes combining all interface modes"""
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    DISTRIBUTED = "distributed"
    EDGE_OPTIMIZED = "edge_optimized"
    PRIVACY_FIRST = "privacy_first"


@dataclass
class RAGConfiguration:
    """Unified configuration for all RAG systems"""
    
    # System identification
    rag_type: RAGType
    system_id: str
    version: str = "1.0.0"
    
    # Processing configuration
    default_mode: ProcessingMode = ProcessingMode.BALANCED
    max_results: int = 10
    similarity_threshold: float = 0.7
    confidence_threshold: float = 0.7
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_concurrent_queries: int = 10
    
    # Memory settings
    memory_enabled: bool = True
    memory_retention_days: int = 30
    
    # Privacy settings
    privacy_mode: bool = False
    anonymize_queries: bool = False
    
    # Backend configuration
    vector_backend: str = "faiss"  # faiss, simple, huggingface
    storage_backend: str = "sqlite"  # sqlite, memory, external
    
    # Advanced features
    enable_reasoning: bool = True
    enable_synthesis: bool = True
    enable_graph_traversal: bool = False
    
    # Custom configurations per RAG type
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    """Standardized result format for all RAG operations"""
    
    # Core result data
    query: str
    results: List[RetrievalResult]
    total_results: int
    
    # Processing metadata
    processing_mode: ProcessingMode
    processing_time_ms: float
    confidence_score: float
    
    # System metadata
    rag_type: RAGType
    system_id: str
    timestamp: datetime
    
    # Additional data
    reasoning_chain: Optional[List[str]] = None
    synthesized_content: Optional[str] = None
    memory_associations: Optional[List[str]] = None
    
    # Performance metrics
    cache_hit: bool = False
    background_processes_used: List[str] = field(default_factory=list)
    
    # Error handling
    warnings: List[str] = field(default_factory=list)
    partial_failure: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRAGSystem(ABC):
    """
    Abstract base class for all RAG systems
    
    Establishes the common interface and provides shared functionality
    that all RAG implementations inherit. Follows composition pattern
    with pluggable components for different capabilities.
    """
    
    def __init__(self, config: RAGConfiguration):
        """Initialize base RAG system with configuration"""
        self.config = config
        self.system_id = config.system_id
        self.rag_type = config.rag_type
        
        # Component interfaces - to be implemented by subclasses
        self._knowledge_retrieval: Optional[KnowledgeRetrievalInterface] = None
        self._memory: Optional[MemoryInterface] = None
        self._reasoning: Optional[ReasoningInterface] = None
        self._synthesis: Optional[SynthesisInterface] = None
        
        # System state
        self._initialized = False
        self._stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "cache_hits": 0,
            "average_response_time_ms": 0.0,
            "last_query_time": None
        }
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the RAG system and all components"""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Gracefully shutdown the RAG system"""
        pass
    
    @abstractmethod
    async def _setup_knowledge_retrieval(self) -> KnowledgeRetrievalInterface:
        """Setup knowledge retrieval component specific to RAG type"""
        pass
    
    @abstractmethod
    async def _setup_memory(self) -> Optional[MemoryInterface]:
        """Setup memory component (optional for some RAG types)"""
        pass
    
    @abstractmethod
    async def _setup_reasoning(self) -> Optional[ReasoningInterface]:
        """Setup reasoning component (optional for some RAG types)"""  
        pass
    
    @abstractmethod
    async def _setup_synthesis(self) -> Optional[SynthesisInterface]:
        """Setup synthesis component (optional for some RAG types)"""
        pass
    
    # Core RAG operations - with default implementations that can be overridden
    
    async def query(
        self, 
        query: str, 
        mode: Optional[ProcessingMode] = None,
        context: Optional[QueryContext] = None,
        max_results: Optional[int] = None
    ) -> RAGResult:
        """
        Execute RAG query with specified mode and context
        
        This is the main entry point for all RAG operations.
        Subclasses can override for specialized behavior.
        """
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        start_time = datetime.now()
        mode = mode or self.config.default_mode
        max_results = max_results or self.config.max_results
        
        try:
            # Update statistics
            self._stats["total_queries"] += 1
            self._stats["last_query_time"] = start_time
            
            # Convert processing mode to query mode for knowledge retrieval
            query_mode = self._convert_processing_to_query_mode(mode)
            
            # Execute knowledge retrieval
            if not self._knowledge_retrieval:
                raise RuntimeError("Knowledge retrieval component not initialized")
            
            retrieval_results = await self._knowledge_retrieval.query(
                query=query,
                mode=query_mode,
                max_results=max_results,
                context=context
            )
            
            # Optional: Apply reasoning
            reasoning_chain = None
            if self.config.enable_reasoning and self._reasoning:
                reasoning_chain = await self._apply_reasoning(query, retrieval_results)
            
            # Optional: Apply synthesis
            synthesized_content = None
            if self.config.enable_synthesis and self._synthesis:
                synthesized_content = await self._apply_synthesis(query, retrieval_results)
            
            # Optional: Update memory
            memory_associations = None
            if self.config.memory_enabled and self._memory:
                memory_associations = await self._update_memory(query, retrieval_results)
            
            # Calculate metrics
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_milliseconds()
            confidence_score = self._calculate_confidence(retrieval_results)
            
            # Update statistics
            self._stats["successful_queries"] += 1
            self._update_average_response_time(processing_time)
            
            # Create result
            result = RAGResult(
                query=query,
                results=retrieval_results,
                total_results=len(retrieval_results),
                processing_mode=mode,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
                rag_type=self.rag_type,
                system_id=self.system_id,
                timestamp=end_time,
                reasoning_chain=reasoning_chain,
                synthesized_content=synthesized_content,
                memory_associations=memory_associations
            )
            
            return result
            
        except Exception as e:
            # Handle errors gracefully
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_milliseconds()
            
            return RAGResult(
                query=query,
                results=[],
                total_results=0,
                processing_mode=mode,
                processing_time_ms=processing_time,
                confidence_score=0.0,
                rag_type=self.rag_type,
                system_id=self.system_id,
                timestamp=end_time,
                warnings=[f"Query failed: {str(e)}"],
                partial_failure=True
            )
    
    async def store_knowledge(
        self, 
        content: str, 
        title: str, 
        metadata: Dict[str, Any],
        knowledge_type: str = "document"
    ) -> str:
        """Store knowledge in the RAG system"""
        if not self._knowledge_retrieval:
            raise RuntimeError("Knowledge retrieval component not initialized")
        
        return await self._knowledge_retrieval.store_knowledge(
            content=content,
            title=title,
            metadata=metadata,
            knowledge_type=knowledge_type
        )
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and health metrics"""
        base_stats = {
            "system_id": self.system_id,
            "rag_type": self.rag_type.value,
            "initialized": self._initialized,
            "configuration": {
                "default_mode": self.config.default_mode.value,
                "max_results": self.config.max_results,
                "memory_enabled": self.config.memory_enabled,
                "reasoning_enabled": self.config.enable_reasoning,
                "synthesis_enabled": self.config.enable_synthesis
            },
            "performance": self._stats.copy()
        }
        
        # Add component-specific stats
        if self._knowledge_retrieval:
            knowledge_stats = await self._knowledge_retrieval.get_system_stats()
            base_stats["knowledge_retrieval"] = knowledge_stats
        
        if self._memory:
            memory_stats = await self._memory.get_memory_usage()
            base_stats["memory"] = memory_stats
        
        if self._reasoning:
            reasoning_stats = await self._reasoning.get_reasoning_stats()
            base_stats["reasoning"] = reasoning_stats
        
        if self._synthesis:
            synthesis_stats = await self._synthesis.get_synthesis_stats()
            base_stats["synthesis"] = synthesis_stats
        
        return base_stats
    
    # Helper methods
    
    def _convert_processing_to_query_mode(self, mode: ProcessingMode) -> QueryMode:
        """Convert ProcessingMode to QueryMode for knowledge retrieval"""
        mode_mapping = {
            ProcessingMode.FAST: QueryMode.FAST,
            ProcessingMode.BALANCED: QueryMode.BALANCED,
            ProcessingMode.COMPREHENSIVE: QueryMode.COMPREHENSIVE,
            ProcessingMode.CREATIVE: QueryMode.CREATIVE,
            ProcessingMode.ANALYTICAL: QueryMode.ANALYTICAL,
            # Special modes default to balanced
            ProcessingMode.DISTRIBUTED: QueryMode.BALANCED,
            ProcessingMode.EDGE_OPTIMIZED: QueryMode.FAST,
            ProcessingMode.PRIVACY_FIRST: QueryMode.BALANCED
        }
        return mode_mapping.get(mode, QueryMode.BALANCED)
    
    async def _apply_reasoning(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult]
    ) -> Optional[List[str]]:
        """Apply reasoning to retrieval results"""
        if not self._reasoning:
            return None
        
        # Extract premises from retrieval results
        premises = [result.content for result in retrieval_results[:5]]  # Top 5 results
        
        try:
            reasoning_result = await self._reasoning.reason(
                query=query,
                premises=premises,
                mode=ReasoningMode.DEDUCTIVE  # Import from reasoning interface
            )
            
            # Convert inference steps to string list
            return [step.conclusion for step in reasoning_result.inference_chain]
            
        except Exception:
            return None
    
    async def _apply_synthesis(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult]
    ) -> Optional[str]:
        """Apply synthesis to create coherent response"""
        if not self._synthesis:
            return None
        
        # Convert retrieval results to content sources
        from ..interfaces.synthesis_interface import ContentSource  # Import locally to avoid circular imports
        
        sources = [
            ContentSource(
                source_id=result.id,
                content=result.content,
                source_type="knowledge",
                reliability_score=result.confidence_score,
                relevance_score=result.relevance_score,
                metadata=result.metadata
            )
            for result in retrieval_results[:3]  # Top 3 results for synthesis
        ]
        
        try:
            synthesis_result = await self._synthesis.synthesize(
                query=query,
                sources=sources,
                mode=SynthesisMode.HYBRID,  # Import from synthesis interface
                target_format=ContentFormat.MARKDOWN  # Import from synthesis interface
            )
            
            return synthesis_result.content
            
        except Exception:
            return None
    
    async def _update_memory(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult]
    ) -> Optional[List[str]]:
        """Update memory with query and results"""
        if not self._memory:
            return None
        
        try:
            # Store query in memory
            from ..interfaces.memory_interface import MemoryType, MemoryContext  # Import locally
            
            memory_context = MemoryContext(
                session_id=None,  # Could be provided in query context
                domain="rag_query"
            )
            
            query_memory_id = await self._memory.store(
                content=query,
                memory_type=MemoryType.EPISODIC,
                context=memory_context
            )
            
            # Store top results and create associations
            associations = []
            for result in retrieval_results[:3]:  # Top 3 results
                result_memory_id = await self._memory.store(
                    content=result.content,
                    memory_type=MemoryType.SEMANTIC,
                    context=memory_context
                )
                
                # Create association between query and result
                await self._memory.associate(
                    memory_id1=query_memory_id,
                    memory_id2=result_memory_id,
                    association_strength=result.relevance_score,
                    association_type="query_result"
                )
                
                associations.append(result_memory_id)
            
            return associations
            
        except Exception:
            return None
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """Calculate overall confidence score from retrieval results"""
        if not results:
            return 0.0
        
        # Weighted average of confidence scores
        total_score = sum(result.confidence_score * result.relevance_score for result in results)
        total_weight = sum(result.relevance_score for result in results)
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _update_average_response_time(self, response_time_ms: float):
        """Update the running average response time"""
        current_avg = self._stats["average_response_time_ms"]
        successful_queries = self._stats["successful_queries"]
        
        # Calculate running average
        new_avg = ((current_avg * (successful_queries - 1)) + response_time_ms) / successful_queries
        self._stats["average_response_time_ms"] = new_avg
    
    # Properties for component access
    
    @property
    def knowledge_retrieval(self) -> Optional[KnowledgeRetrievalInterface]:
        """Access to knowledge retrieval component"""
        return self._knowledge_retrieval
    
    @property 
    def memory(self) -> Optional[MemoryInterface]:
        """Access to memory component"""
        return self._memory
    
    @property
    def reasoning(self) -> Optional[ReasoningInterface]:
        """Access to reasoning component"""
        return self._reasoning
    
    @property
    def synthesis(self) -> Optional[SynthesisInterface]:
        """Access to synthesis component"""
        return self._synthesis
    
    @property
    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self._initialized