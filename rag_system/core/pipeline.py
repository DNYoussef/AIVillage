"""Enhanced RAG pipeline implementation."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

from .base_component import BaseComponent
from ..core.config import RAGConfig
from ..retrieval.hybrid_retriever import HybridRetriever
from ..processing.reasoning_engine import UncertaintyAwareReasoningEngine
from .latent_space_activation import LatentSpaceActivation
from ..utils.error_handling import log_and_handle_errors, ErrorContext, RAGSystemError

class EnhancedRAGPipeline(BaseComponent):
    """
    Enhanced RAG pipeline that integrates retrieval, reasoning, and knowledge activation.
    Implements the complete RAG workflow with feedback and uncertainty handling.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize pipeline components."""
        self.config = config or RAGConfig()
        self.latent_activator = LatentSpaceActivation()
        self.retriever = None  # Initialize in initialize() method
        self.reasoner = None   # Initialize in initialize() method
        self.agent = None      # Set by test or application code
        self.initialized = False
        self.processing_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_processing_time": 0.0
        }
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if not self.initialized:
            if not self.retriever:
                self.retriever = HybridRetriever(self.config)
            if not self.reasoner:
                self.reasoner = UncertaintyAwareReasoningEngine(self.config)
            
            # Initialize components
            await self.latent_activator.initialize()
            await self.retriever.initialize()
            await self.reasoner.initialize()
            
            # Verify all required components are set
            if not self.agent:
                raise RAGSystemError("Agent not set in pipeline")
            
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        if self.initialized:
            await self.latent_activator.shutdown()
            if self.retriever:
                await self.retriever.shutdown()
            if self.reasoner:
                await self.reasoner.shutdown()
            self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        return {
            "initialized": self.initialized,
            "latent_activator": await self.latent_activator.get_status(),
            "retriever": await self.retriever.get_status() if self.retriever else None,
            "reasoner": await self.reasoner.get_status() if self.reasoner else None,
            "agent": "set" if self.agent else "not set",
            "processing_stats": self.processing_stats
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: RAGConfig) -> None:
        """Update pipeline configuration."""
        self.config = config
        await self.latent_activator.update_config(config)
        if self.retriever:
            await self.retriever.update_config(config)
        if self.reasoner:
            await self.reasoner.update_config(config)
    
    @log_and_handle_errors()
    async def process_query(self,
                          query: str,
                          context: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process query through the enhanced RAG pipeline.
        
        Args:
            query: User's query
            context: Optional context information
            timestamp: Optional timestamp for temporal queries
            
        Returns:
            Dictionary containing processing results
        """
        if not self.initialized:
            await self.initialize()
            
        async with ErrorContext("EnhancedRAGPipeline"):
            start_time = datetime.now()
            self.processing_stats["total_queries"] += 1
            
            try:
                # Get number of documents to retrieve from context or config
                k = context.get("num_results", self.config.num_documents) if context else self.config.num_documents
                
                # Activate relevant knowledge
                activated_knowledge = await self.latent_activator.activate(
                    query,
                    context
                )
                
                # Retrieve relevant information
                retrieval_results = await self.retriever.retrieve(
                    query,
                    k=k,
                    timestamp=timestamp
                )
                
                if not retrieval_results:
                    retrieval_results = []  # Ensure we have a list even if empty
                
                # Generate feedback
                feedback = await self.retriever._generate_feedback(
                    query,
                    retrieval_results
                )
                
                # Perform reasoning
                reasoning_results = await self.reasoner.reason(
                    query,
                    retrieval_results,
                    activated_knowledge
                )
                
                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_stats(processing_time, success=True)
                
                return {
                    "query": query,
                    "activated_knowledge": activated_knowledge,
                    "retrieval_results": retrieval_results,
                    "feedback": feedback,
                    "reasoning_results": reasoning_results,
                    "processing_time": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                self._update_stats(0.0, success=False)
                raise
    
    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics."""
        if success:
            self.processing_stats["successful_queries"] += 1
        else:
            self.processing_stats["failed_queries"] += 1
        
        # Update average processing time
        total_queries = self.processing_stats["successful_queries"]
        current_avg = self.processing_stats["avg_processing_time"]
        
        if total_queries > 1:
            self.processing_stats["avg_processing_time"] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
        else:
            self.processing_stats["avg_processing_time"] = processing_time
