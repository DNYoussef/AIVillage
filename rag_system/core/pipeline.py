"""Enhanced RAG pipeline implementation."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import logging

from .base_component import BaseComponent
from ..core.config import RAGConfig
from ..retrieval.hybrid_retriever import HybridRetriever
from ..processing.reasoning_engine import UncertaintyAwareReasoningEngine
from .latent_space_activation import LatentSpaceActivation
from ..utils.error_handling import log_and_handle_errors, ErrorContext, RAGSystemError

logger = logging.getLogger(__name__)

class EnhancedRAGPipeline(BaseComponent):
    """
    Enhanced RAG pipeline that integrates retrieval, reasoning, and knowledge activation.
    Implements the complete RAG workflow with feedback and uncertainty handling.
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize pipeline components."""
        super().__init__()
        self.config = config or RAGConfig()
        self.latent_activator = LatentSpaceActivation()
        self.retriever = None  # Initialize in initialize() method
        self.reasoner = None   # Initialize in initialize() method
        self.agent = None      # Set by test or application code
        
        # Add pipeline-specific stats
        self.stats.update({
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_processing_time": 0.0,
            "component_status": {
                "latent_activator": False,
                "retriever": False,
                "reasoner": False,
                "agent": False
            }
        })
        
        logger.info("Initialized EnhancedRAGPipeline")
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            await self._pre_initialize()
            
            logger.info("Initializing pipeline components...")
            
            # Initialize retriever if needed
            if not self.retriever:
                self.retriever = HybridRetriever(self.config)
            
            # Initialize reasoner if needed
            if not self.reasoner:
                self.reasoner = UncertaintyAwareReasoningEngine(self.config)
            
            # Initialize all components
            try:
                await self.latent_activator.initialize()
                self.stats["component_status"]["latent_activator"] = True
            except Exception as e:
                logger.error(f"Error initializing latent activator: {str(e)}")
                raise
            
            try:
                await self.retriever.initialize()
                self.stats["component_status"]["retriever"] = True
            except Exception as e:
                logger.error(f"Error initializing retriever: {str(e)}")
                raise
            
            try:
                await self.reasoner.initialize()
                self.stats["component_status"]["reasoner"] = True
            except Exception as e:
                logger.error(f"Error initializing reasoner: {str(e)}")
                raise
            
            # Verify agent is set
            if not self.agent:
                raise RAGSystemError("Agent not set in pipeline")
            self.stats["component_status"]["agent"] = True
            
            await self._post_initialize()
            logger.info("Successfully initialized all pipeline components")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            self.initialized = False
            raise
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        try:
            await self._pre_shutdown()
            
            logger.info("Shutting down pipeline components...")
            
            # Shutdown components in reverse order
            if self.reasoner:
                try:
                    await self.reasoner.shutdown()
                    self.stats["component_status"]["reasoner"] = False
                except Exception as e:
                    logger.error(f"Error shutting down reasoner: {str(e)}")
            
            if self.retriever:
                try:
                    await self.retriever.shutdown()
                    self.stats["component_status"]["retriever"] = False
                except Exception as e:
                    logger.error(f"Error shutting down retriever: {str(e)}")
            
            try:
                await self.latent_activator.shutdown()
                self.stats["component_status"]["latent_activator"] = False
            except Exception as e:
                logger.error(f"Error shutting down latent activator: {str(e)}")
            
            self.stats["component_status"]["agent"] = False
            
            await self._post_shutdown()
            logger.info("Successfully shut down all pipeline components")
            
        except Exception as e:
            logger.error(f"Error shutting down pipeline: {str(e)}")
            raise
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        base_status = await self.get_base_status()
        
        component_status = {
            "latent_activator": await self.latent_activator.get_status(),
            "retriever": await self.retriever.get_status() if self.retriever else None,
            "reasoner": await self.reasoner.get_status() if self.reasoner else None,
            "agent": "set" if self.agent else "not set"
        }
        
        return {
            **base_status,
            "component_status": component_status,
            "config": {
                "num_documents": self.config.num_documents,
                "uncertainty_threshold": self.config.uncertainty_threshold
            }
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: RAGConfig) -> None:
        """Update pipeline configuration."""
        try:
            logger.info("Updating pipeline configuration...")
            
            self.config = config
            
            # Update component configurations
            try:
                await self.latent_activator.update_config(config)
            except Exception as e:
                logger.error(f"Error updating latent activator config: {str(e)}")
                raise
            
            if self.retriever:
                try:
                    await self.retriever.update_config(config)
                except Exception as e:
                    logger.error(f"Error updating retriever config: {str(e)}")
                    raise
            
            if self.reasoner:
                try:
                    await self.reasoner.update_config(config)
                except Exception as e:
                    logger.error(f"Error updating reasoner config: {str(e)}")
                    raise
            
            logger.info("Successfully updated pipeline configuration")
            
        except Exception as e:
            logger.error(f"Error updating pipeline configuration: {str(e)}")
            raise
    
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
            self.stats["total_queries"] += 1
            
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
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
        
        # Update average processing time
        total_queries = self.stats["successful_queries"]
        current_avg = self.stats["avg_processing_time"]
        
        if total_queries > 1:
            self.stats["avg_processing_time"] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
        else:
            self.stats["avg_processing_time"] = processing_time
