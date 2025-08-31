"""
Processing Chain Factory for creating query/task/response processing chains.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.latent_space_activation import LatentSpaceActivation
from rag_system.core.cognitive_nexus import CognitiveNexus
from rag_system.processing.confidence_estimator import ConfidenceEstimator

from ..query_processing import QueryProcessor
from ..task_execution import TaskExecutor
from ..response_generator import ResponseGenerator
from ..user_intent_interpreter import UserIntentInterpreter
from .interfaces import IProcessingService, AbstractServiceBase
from .config import ProcessingServiceConfig

logger = logging.getLogger(__name__)


class ProcessingChainFactory(AbstractServiceBase, IProcessingService):
    """
    Factory for creating and managing processing chains.
    
    Provides unified interface for:
    - QueryProcessor: Query analysis and processing
    - TaskExecutor: Task execution management
    - ResponseGenerator: Response generation
    - UserIntentInterpreter: Intent analysis
    """
    
    def __init__(
        self,
        rag_system: EnhancedRAGPipeline,
        latent_space_activation: LatentSpaceActivation,
        cognitive_nexus: CognitiveNexus,
        confidence_estimator: ConfidenceEstimator,
        sage_agent: Any,  # Forward reference to avoid circular imports
        config: ProcessingServiceConfig
    ):
        super().__init__(config.config_params)
        self.rag_system = rag_system
        self.latent_space_activation = latent_space_activation
        self.cognitive_nexus = cognitive_nexus
        self.confidence_estimator = confidence_estimator
        self.sage_agent = sage_agent
        self.processing_config = config
        
        # Processing components (lazy loaded)
        self._query_processor: Optional[QueryProcessor] = None
        self._task_executor: Optional[TaskExecutor] = None
        self._response_generator: Optional[ResponseGenerator] = None
        self._user_intent_interpreter: Optional[UserIntentInterpreter] = None
        
        # Processing state
        self._processing_queue = asyncio.Queue()
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._processing_stats = {
            "total_queries": 0,
            "total_tasks": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "average_processing_time": 0.0
        }
        
        # Concurrent processing control
        self._processing_semaphore = asyncio.Semaphore(
            config.max_concurrent_tasks if config.parallel_processing else 1
        )
    
    @property
    def query_processor(self) -> QueryProcessor:
        """Get query processor with lazy loading."""
        if self._query_processor is None and self.processing_config.query_processor_enabled:
            self._query_processor = QueryProcessor(
                self.rag_system,
                self.latent_space_activation, 
                self.cognitive_nexus
            )
        return self._query_processor
    
    @property
    def task_executor(self) -> TaskExecutor:
        """Get task executor with lazy loading."""
        if self._task_executor is None and self.processing_config.task_executor_enabled:
            self._task_executor = TaskExecutor(self.sage_agent)
        return self._task_executor
    
    @property
    def response_generator(self) -> ResponseGenerator:
        """Get response generator with lazy loading."""
        if self._response_generator is None and self.processing_config.response_generator_enabled:
            self._response_generator = ResponseGenerator()
        return self._response_generator
    
    @property
    def user_intent_interpreter(self) -> UserIntentInterpreter:
        """Get user intent interpreter with lazy loading."""
        if self._user_intent_interpreter is None:
            self._user_intent_interpreter = UserIntentInterpreter()
        return self._user_intent_interpreter
    
    async def initialize(self) -> None:
        """Initialize processing components."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing processing chain factory...")
            
            # Initialize components based on configuration
            if self.processing_config.query_processor_enabled:
                self._initialize_query_processor()
            
            if self.processing_config.task_executor_enabled:
                self._initialize_task_executor()
            
            if self.processing_config.response_generator_enabled:
                self._initialize_response_generator()
            
            self._initialize_user_intent_interpreter()
            
            self._initialized = True
            logger.info("Processing chain factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processing chain factory: {e}")
            raise
    
    def _initialize_query_processor(self) -> None:
        """Initialize query processor."""
        if self._query_processor is None:
            self._query_processor = QueryProcessor(
                self.rag_system,
                self.latent_space_activation,
                self.cognitive_nexus
            )
            logger.debug("Query processor initialized")
    
    def _initialize_task_executor(self) -> None:
        """Initialize task executor."""
        if self._task_executor is None:
            self._task_executor = TaskExecutor(self.sage_agent)
            logger.debug("Task executor initialized")
    
    def _initialize_response_generator(self) -> None:
        """Initialize response generator."""
        if self._response_generator is None:
            self._response_generator = ResponseGenerator()
            logger.debug("Response generator initialized")
    
    def _initialize_user_intent_interpreter(self) -> None:
        """Initialize user intent interpreter."""
        if self._user_intent_interpreter is None:
            self._user_intent_interpreter = UserIntentInterpreter()
            logger.debug("User intent interpreter initialized")
    
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete processing chain.
        
        Args:
            query: Query string to process
            context: Optional context information
            
        Returns:
            Complete processing result
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._processing_semaphore:
            start_time = datetime.now()
            task_id = f"query_{start_time.timestamp()}"
            
            try:
                logger.debug(f"Processing query: {query[:50]}...")
                
                result = {
                    "task_id": task_id,
                    "query": query,
                    "context": context or {},
                    "processing_steps": {},
                    "timestamp": start_time.isoformat()
                }
                
                # Step 1: Interpret user intent
                if self.user_intent_interpreter:
                    intent = await self.user_intent_interpreter.interpret_intent(query)
                    result["processing_steps"]["intent_interpretation"] = intent
                    result["interpreted_intent"] = intent
                else:
                    result["interpreted_intent"] = {"intent": "general", "confidence": 0.5}
                
                # Step 2: Pre-process query
                processed_query = await self._pre_process_query(query, context)
                result["processing_steps"]["query_preprocessing"] = {
                    "original_query": query,
                    "processed_query": processed_query
                }
                
                # Step 3: RAG processing
                if self.rag_system:
                    rag_result = await self.rag_system.process_query(processed_query)
                    result["processing_steps"]["rag_processing"] = rag_result
                    result["rag_result"] = rag_result
                else:
                    result["rag_result"] = {"error": "RAG system not available"}
                
                # Step 4: Generate response
                if self.response_generator and "rag_result" in result:
                    response = await self.response_generator.generate_response(
                        query, 
                        result["rag_result"], 
                        result["interpreted_intent"]
                    )
                    result["processing_steps"]["response_generation"] = {"response": response}
                    result["response"] = response
                else:
                    result["response"] = "Unable to generate response"
                
                # Step 5: Post-process and add confidence
                final_result = await self._post_process_result(result, query)
                
                # Update statistics
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_processing_stats("query", processing_time, success=True)
                
                self.update_last_used()
                logger.debug(f"Query processed successfully in {processing_time:.3f}s")
                
                return final_result
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_processing_stats("query", processing_time, success=False)
                logger.error(f"Query processing failed: {e}")
                
                return {
                    "task_id": task_id,
                    "query": query,
                    "error": str(e),
                    "processing_time": processing_time,
                    "timestamp": start_time.isoformat()
                }
    
    async def execute_task(self, task: Dict[str, Any]) -> Any:
        """
        Execute a task through the task executor.
        
        Args:
            task: Task dictionary with type, content, priority, etc.
            
        Returns:
            Task execution result
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._processing_semaphore:
            start_time = datetime.now()
            task_id = task.get("id", f"task_{start_time.timestamp()}")
            
            try:
                logger.debug(f"Executing task: {task.get('type', 'unknown')}")
                
                if self.task_executor:
                    # Store active task
                    execution_task = asyncio.create_task(
                        self.task_executor.execute_task(task)
                    )
                    self._active_tasks[task_id] = execution_task
                    
                    try:
                        result = await execution_task
                    finally:
                        # Clean up active task
                        if task_id in self._active_tasks:
                            del self._active_tasks[task_id]
                    
                    # Update statistics
                    processing_time = (datetime.now() - start_time).total_seconds()
                    self._update_processing_stats("task", processing_time, success=True)
                    
                    self.update_last_used()
                    logger.debug(f"Task executed successfully in {processing_time:.3f}s")
                    
                    return result
                else:
                    raise RuntimeError("Task executor not available")
                
            except Exception as e:
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_processing_stats("task", processing_time, success=False)
                logger.error(f"Task execution failed: {e}")
                raise
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the processing service."""
        try:
            # Update processing configuration
            if "max_concurrent_tasks" in config:
                new_limit = config["max_concurrent_tasks"]
                self._processing_semaphore = asyncio.Semaphore(new_limit)
                self.processing_config.max_concurrent_tasks = new_limit
            
            if "parallel_processing" in config:
                self.processing_config.parallel_processing = config["parallel_processing"]
            
            # Configure individual components
            if "query_processor" in config and self._query_processor:
                # Query processor configuration if it supports it
                pass
            
            if "task_executor" in config and self._task_executor:
                # Task executor configuration if it supports it
                pass
            
            if "response_generator" in config and self._response_generator:
                # Response generator configuration if it supports it
                pass
            
            logger.info(f"Processing chain factory reconfigured: {config}")
            
        except Exception as e:
            logger.error(f"Failed to configure processing factory: {e}")
            raise
    
    async def _pre_process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Pre-process query before RAG processing."""
        # Basic preprocessing - can be enhanced
        processed = query.strip()
        
        # Add context-aware preprocessing if context is provided
        if context:
            # Could enhance query with context information
            pass
        
        return processed
    
    async def _post_process_result(
        self,
        result: Dict[str, Any], 
        original_query: str
    ) -> Dict[str, Any]:
        """Post-process the complete result."""
        try:
            # Add confidence estimation if available
            if self.confidence_estimator and "rag_result" in result:
                confidence = await self.confidence_estimator.estimate(
                    original_query, 
                    result["rag_result"]
                )
                result["confidence"] = confidence
            
            # Add processing metadata
            result["processing_metadata"] = {
                "components_used": {
                    "query_processor": self._query_processor is not None,
                    "task_executor": self._task_executor is not None,
                    "response_generator": self._response_generator is not None,
                    "user_intent_interpreter": self._user_intent_interpreter is not None
                },
                "processing_config": {
                    "parallel_processing": self.processing_config.parallel_processing,
                    "max_concurrent_tasks": self.processing_config.max_concurrent_tasks
                }
            }
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            result["post_processing_error"] = str(e)
        
        return result
    
    def _update_processing_stats(
        self, 
        operation_type: str, 
        processing_time: float, 
        success: bool
    ) -> None:
        """Update processing statistics."""
        if operation_type == "query":
            self._processing_stats["total_queries"] += 1
        elif operation_type == "task":
            self._processing_stats["total_tasks"] += 1
        
        if success:
            self._processing_stats["successful_operations"] += 1
        else:
            self._processing_stats["failed_operations"] += 1
        
        # Update rolling average processing time
        total_ops = (self._processing_stats["total_queries"] + 
                    self._processing_stats["total_tasks"])
        current_avg = self._processing_stats["average_processing_time"]
        self._processing_stats["average_processing_time"] = (
            (current_avg * (total_ops - 1) + processing_time) / total_ops
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self._processing_stats,
            "active_tasks": len(self._active_tasks),
            "queue_size": self._processing_queue.qsize(),
            "semaphore_available": self._processing_semaphore._value,
            "config": {
                "max_concurrent_tasks": self.processing_config.max_concurrent_tasks,
                "parallel_processing": self.processing_config.parallel_processing
            }
        }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._active_tasks[task_id]
            logger.info(f"Task {task_id} cancelled")
            return True
        return False
    
    async def cancel_all_tasks(self) -> int:
        """Cancel all active tasks."""
        cancelled_count = 0
        tasks_to_cancel = list(self._active_tasks.keys())
        
        for task_id in tasks_to_cancel:
            if await self.cancel_task(task_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} tasks")
        return cancelled_count
    
    async def shutdown(self) -> None:
        """Shutdown processing chain factory."""
        try:
            logger.info("Shutting down processing chain factory...")
            
            # Cancel all active tasks
            await self.cancel_all_tasks()
            
            # Shutdown individual components if they support it
            for component in [
                self._query_processor,
                self._task_executor,
                self._response_generator,
                self._user_intent_interpreter
            ]:
                if component and hasattr(component, 'shutdown'):
                    await component.shutdown()
            
            # Clear references
            self._query_processor = None
            self._task_executor = None
            self._response_generator = None
            self._user_intent_interpreter = None
            
            self._initialized = False
            logger.info("Processing chain factory shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during processing factory shutdown: {e}")
            raise