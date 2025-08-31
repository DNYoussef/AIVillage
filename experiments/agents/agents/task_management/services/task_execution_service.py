"""
Task Execution Service - Handles task batch processing and execution.
Extracted from UnifiedManagement god class.
"""
import asyncio
import logging
from typing import Any

from core.error_handling import AIVillageException, Message, MessageType, StandardCommunicationProtocol

from ..task import Task
from ..interfaces.task_service_interfaces import ITaskCreationService, ITaskCompletionService

logger = logging.getLogger(__name__)


class TaskExecutionService:
    """Service responsible for task execution and batch processing."""
    
    def __init__(
        self,
        communication_protocol: StandardCommunicationProtocol,
        creation_service: ITaskCreationService,
        completion_service: ITaskCompletionService,
    ) -> None:
        """Initialize with dependencies."""
        self._communication_protocol = communication_protocol
        self._creation_service = creation_service
        self._completion_service = completion_service
        self._batch_size = 5
        self._processing_active = False
        
    async def process_task_batch(self) -> None:
        """Process a batch of tasks concurrently."""
        try:
            pending_tasks = self._creation_service.get_pending_tasks()
            batch = []
            
            # Get batch of tasks to process
            for _ in range(min(self._batch_size, len(pending_tasks))):
                if pending_tasks:
                    task = self._creation_service.remove_pending_task(pending_tasks[0].id)
                    if task:
                        batch.append(task)

            if not batch:
                return

            logger.info("Processing batch of %d tasks", len(batch))
            
            # Process batch concurrently
            results = await asyncio.gather(
                *[self.process_single_task(task) for task in batch],
                return_exceptions=True
            )

            # Handle results
            for task, result in zip(batch, results, strict=False):
                if isinstance(result, Exception):
                    logger.error("Task %s failed: %s", task.id, result)
                    continue
                await self._completion_service.complete_task(task.id, result)
                
        except Exception as e:
            logger.exception("Error processing task batch: %s", e)
            msg = f"Error processing task batch: {e!s}"
            raise AIVillageException(msg) from e

    async def process_single_task(self, task: Task) -> Any:
        """Process a single task by communicating with assigned agent."""
        try:
            if not task.assigned_agents:
                msg = f"Task {task.id} has no assigned agents"
                raise AIVillageException(msg)
                
            agent = task.assigned_agents[0]
            
            message = Message(
                type=MessageType.TASK,
                sender="TaskExecutionService",
                receiver=agent,
                content={
                    "task_id": task.id,
                    "description": task.description,
                    "priority": task.priority,
                }
            )
            
            result = await self._communication_protocol.send_and_wait(message)
            logger.debug("Processed task %s with agent %s", task.id, agent)
            
            return result
        except Exception as e:
            logger.exception("Error processing single task: %s", e)
            msg = f"Error processing single task: {e!s}"
            raise AIVillageException(msg) from e

    async def start_batch_processing(self) -> None:
        """Start continuous batch processing loop."""
        try:
            self._processing_active = True
            logger.info("Started batch processing with batch size %d", self._batch_size)
            
            while self._processing_active:
                await self.process_task_batch()
                await asyncio.sleep(1)  # Configurable delay
                
        except Exception as e:
            logger.exception("Error in batch processing: %s", e)
            msg = f"Error in batch processing: {e!s}"
            raise AIVillageException(msg) from e

    def stop_batch_processing(self) -> None:
        """Stop the batch processing loop."""
        self._processing_active = False
        logger.info("Stopped batch processing")

    def set_batch_size(self, size: int) -> None:
        """Set the batch size for concurrent processing."""
        try:
            if size <= 0:
                msg = "Batch size must be positive"
                raise ValueError(msg)
                
            self._batch_size = size
            logger.info("Set batch size to %d", size)
        except Exception as e:
            logger.exception("Error setting batch size: %s", e)
            msg = f"Error setting batch size: {e!s}"
            raise AIVillageException(msg) from e

    def get_batch_size(self) -> int:
        """Get current batch size."""
        return self._batch_size

    def is_processing_active(self) -> bool:
        """Check if batch processing is active."""
        return self._processing_active