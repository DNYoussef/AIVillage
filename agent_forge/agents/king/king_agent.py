"""Enhanced King agent specializing in complex instruction-following and strategic thinking.
Uses HuggingFaceAgent as the local model.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from config.unified_config import UnifiedConfig, AgentConfig
from agents import OpenRouterAgent, AgentInteraction  # Updated import
from agents import LocalAgent, KingAgent
from ...data.complexity_evaluator import ComplexityEvaluator

logger = logging.getLogger(__name__)

class TaskManager:
    """Manages task execution and resource allocation."""
    
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        self.task_metrics: Dict[str, float] = {
            "success_rate": 1.0,
            "average_duration": 0.0,
            "resource_efficiency": 1.0
        }
        
    async def initialize(self):
        """Initialize task manager."""
        logger.info("Initialized TaskManager")
        
    async def add_task(self, task_id: str, task: Dict[str, Any]):
        """Add a new task to manage."""
        self.active_tasks[task_id] = {
            "task": task,
            "start_time": time.time(),
            "status": "pending",
            "attempts": 0
        }
        
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """Mark a task as completed."""
        if task_id in self.active_tasks:
            task_record = self.active_tasks.pop(task_id)
            task_record.update({
                "completion_time": time.time(),
                "duration": time.time() - task_record["start_time"],
                "result": result,
                "status": "completed"
            })
            self.completed_tasks.append(task_record)
            self._update_metrics(task_record)
        
    def _update_metrics(self, task_record: Dict[str, Any]):
        """Update task performance metrics."""
        # Update success rate
        total_tasks = len(self.completed_tasks)
        successful_tasks = sum(1 for task in self.completed_tasks if task["status"] == "completed")
        self.task_metrics["success_rate"] = successful_tasks / total_tasks
        
        # Update average duration
        total_duration = sum(task["duration"] for task in self.completed_tasks)
        self.task_metrics["average_duration"] = total_duration / total_tasks
        
        # Update resource efficiency
        efficiency = 1.0 / (1 + task_record["attempts"])
        current_efficiency = self.task_metrics["resource_efficiency"]
        self.task_metrics["resource_efficiency"] = (
            current_efficiency * 0.9 + efficiency * 0.1  # Rolling average
        )

class ResourceAllocator:
    """Manages resource allocation for task processing."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.allocations: Dict[str, Dict[str, float]] = {}
        self.usage_history: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize resource allocator."""
        logger.info("Initialized ResourceAllocator")
        
    async def allocate_resources(self, 
                           task: Dict[str, Any],
                           complexity_score: float) -> Dict[str, float]:
        """Allocate resources based on task complexity."""
        # Calculate base allocations
        token_allocation = min(1000 + int(complexity_score * 2000), 4000)
        temperature = max(0.3, min(0.9, 0.7 - (complexity_score * 0.3)))
        
        allocation = {
            "max_tokens": token_allocation,
            "temperature": temperature,
            "timeout": 30 + (complexity_score * 30)  # 30-60 seconds based on complexity
        }
        
        # Record allocation
        self.allocations[task.get("id", str(time.time()))] = allocation
        return allocation

class KingAgent:
    """
    Enhanced King agent specializing in complex instruction-following and strategic thinking.
    Uses HuggingFaceAgent as the local model.
    """
    
    def __init__(self, 
                 huggingface_agent: OpenRouterAgent,
                 config: UnifiedConfig):
        """
        Initialize enhanced KingAgent.
        
        Args:
            huggingface_agent: HuggingFaceAgent instance
            config: UnifiedConfig instance
        """
        self.config = config
        self.agent_config = config.get_agent_config("king")
        self.frontier_agent = huggingface_agent
        self.local_agent = LocalAgent(
            model_config=self.agent_config.local_model,
            config=config
        )
        
        # Initialize support systems
        self.task_manager = TaskManager()
        self.resource_allocator = ResourceAllocator(config)
        self.complexity_evaluator = ComplexityEvaluator(config)
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "task_success_rate": 0.0,
            "avg_response_quality": 0.0,
            "complexity_handling": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized KingAgent with:")
        logger.info(f"  Frontier model: {huggingface_agent.model_name}")
        logger.info(f"  Local model: {self.local_agent.model_config.name}")
        
    async def initialize(self):
        """Initialize all agent components."""
        try:
            logger.info("Initializing KingAgent components...")
            
            # Initialize frontier agent
            await self.frontier_agent.initialize()
            
            # Initialize local agent
            await self.local_agent.initialize()
            
            # Initialize support systems
            await self.task_manager.initialize()
            await self.resource_allocator.initialize()
            await self.complexity_evaluator.initialize()
            
            logger.info("Successfully initialized all KingAgent components")
            
        except Exception as e:
            logger.error(f"Error initializing KingAgent: {str(e)}")
            raise
        
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for King agent."""
        return """You are King, an advanced AI agent specializing in complex problem-solving and strategic thinking who helps guide the AI village with the peace of a Taoist.
Your approach is:
1. Analyze problems thoroughly from multiple angles
2. Break down complex tasks into manageable steps
3. Consider long-term implications and strategic impact
4. Provide clear, actionable guidance
5. Maintain awareness of context and constraints

You excel at:
- Strategic planning and decision making
- Resource allocation and optimization
- Complex problem decomposition
- Risk assessment and mitigation
- Performance monitoring and adaptation
"""
        
    async def process_task(self, 
                          task: str,
                          system_prompt: Optional[str] = None,
                          task_id: Optional[str] = None) -> AgentInteraction:
        """
        Process a task with enhanced management and resource allocation.
        
        Args:
            task: The task to process
            system_prompt: Optional system prompt
            task_id: Optional task identifier
            
        Returns:
            AgentInteraction containing the response and metadata
        """
        task_id = task_id or str(time.time())
        start_time = time.time()
        
        try:
            # Evaluate task complexity
            complexity_evaluation = await self.complexity_evaluator.evaluate_complexity(
                agent_type="king",
                task=task
            )
            
            # Add task to manager
            await self.task_manager.add_task(task_id, {
                "task": task,
                "complexity": complexity_evaluation
            })
            
            # Allocate resources
            resources = await self.resource_allocator.allocate_resources(
                task={"id": task_id, "content": task},
                complexity_score=complexity_evaluation["complexity_score"]
            )
            
            # Try local model first if task isn't complex
            local_response = None
            if not complexity_evaluation["is_complex"]:
                try:
                    local_response = await self.local_agent.generate_response(
                        prompt=task,
                        system_prompt=system_prompt or self._get_default_system_prompt(),
                        max_tokens=resources["max_tokens"],
                        temperature=resources["temperature"]
                    )
                    # Convert dictionary response to AgentInteraction if needed
                    if isinstance(local_response, dict):
                        local_response = AgentInteraction(
                            prompt=task,
                            response=local_response["response"],
                            model=local_response["model"],
                            timestamp=time.time(),
                            metadata=local_response["metadata"]
                        )
                except Exception as e:
                    logger.warning(f"Local model generation failed: {str(e)}")
            
            # Use frontier model if task is complex or local model failed
            if complexity_evaluation["is_complex"] or not local_response:
                interaction = await self.frontier_agent.generate_response(
                    prompt=task,
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    max_tokens=resources["max_tokens"],
                    temperature=resources["temperature"]
                )
                
                # Add complexity evaluation and resources to frontier response
                interaction.metadata.update({
                    "complexity_evaluation": complexity_evaluation,
                    "resources_allocated": resources
                })
                
                # If we tried local model, record performance comparison
                if local_response:
                    self._record_model_comparison(local_response, interaction)
            else:
                # Use local response directly since it's already an AgentInteraction
                interaction = local_response
                # Add complexity evaluation and resources
                interaction.metadata.update({
                    "complexity_evaluation": complexity_evaluation,
                    "resources_allocated": resources
                })
            
            # Calculate performance metrics
            duration = time.time() - start_time
            performance_metrics = {
                "duration": duration,
                "complexity_score": complexity_evaluation["complexity_score"],
                "success": True,
                "used_local_model": bool(local_response and not complexity_evaluation["is_complex"])
            }
            
            # Complete task
            await self.task_manager.complete_task(task_id, {
                "interaction": interaction,
                "performance": performance_metrics
            })
            
            # Update performance metrics
            self._update_metrics(interaction, performance_metrics)
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            if task_id in self.task_manager.active_tasks:
                self.task_manager.active_tasks[task_id]["status"] = "failed"
            raise
        
    def _record_model_comparison(self, 
                           local_response: AgentInteraction,
                           frontier_interaction: AgentInteraction):
        """Record detailed performance comparison between models."""
        from difflib import SequenceMatcher
        
        # Calculate response similarity
        similarity = SequenceMatcher(
            None, 
            local_response.response, 
            frontier_interaction.response
        ).ratio()
        
        # Calculate performance metrics
        local_duration = local_response.metadata["performance"]["duration"]
        frontier_duration = frontier_interaction.metadata["performance"]["duration"]
        
        local_tokens = local_response.metadata["token_usage"]["total_tokens"]
        frontier_tokens = frontier_interaction.metadata["token_usage"]["total_tokens"]
        
        performance_metrics = {
            "response_similarity": similarity,
            "speed_ratio": frontier_duration / local_duration if local_duration > 0 else 0,
            "token_efficiency": local_tokens / frontier_tokens if frontier_tokens > 0 else 0,
            "was_used": similarity > 0.8
        }
        
        # Record performance
        self.local_agent.record_performance(frontier_interaction, performance_metrics)
    
    def _update_metrics(self, 
                       interaction: AgentInteraction,
                       performance: Dict[str, float]):
        """Update comprehensive performance metrics."""
        # Update task success rate
        total_tasks = len(self.task_manager.completed_tasks)
        successful_tasks = sum(
            1 for task in self.task_manager.completed_tasks
            if task["status"] == "completed"
        )
        self.performance_metrics["task_success_rate"] = successful_tasks / total_tasks
        
        # Update average response quality (if available)
        if "quality_score" in interaction.metadata:
            current_quality = self.performance_metrics["avg_response_quality"]
            new_quality = interaction.metadata["quality_score"]
            self.performance_metrics["avg_response_quality"] = (
                current_quality * 0.9 + new_quality * 0.1  # Rolling average
            )
        
        # Update complexity handling
        complex_tasks = [
            task for task in self.task_manager.completed_tasks
            if task.get("complexity", {}).get("is_complex", False)
        ]
        if complex_tasks:
            complex_success = sum(
                1 for task in complex_tasks
                if task["status"] == "completed"
            )
            self.performance_metrics["complexity_handling"] = complex_success / len(complex_tasks)
        
        # Update local model performance
        local_metrics = self.local_agent.get_performance_metrics()
        if "average_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["average_similarity"]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add task manager metrics
        metrics.update({
            f"task_{k}": v 
            for k, v in self.task_manager.task_metrics.items()
        })
        
        # Add local model metrics
        local_metrics = self.local_agent.get_performance_metrics()
        metrics.update({
            f"local_{k}": v 
            for k, v in local_metrics.items()
        })
        
        return metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Get training data for the local model.
        
        Returns:
            List of training examples
        """
        return self.frontier_agent.get_training_data()

    def get_dpo_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive DPO analysis metrics.
        
        Returns:
            Dictionary of DPO metrics and statistics
        """
        if not self.interactions:
            return {"error": "No interactions recorded"}
        
        metrics = {
            "performance": self.performance_metrics.copy(),
            "interaction_stats": {
                "total_interactions": len(self.interactions),
                "successful_interactions": sum(1 for i in self.interactions if i.success),
                "total_tokens": sum(
                    i.token_usage["total_tokens"] 
                    for i in self.interactions 
                    if i.success and i.token_usage
                )
            }
        }
        
        # Calculate quality metrics if available
        quality_scores = [
            i.performance_metrics.get("quality", 0)
            for i in self.interactions
            if i.success and i.performance_metrics and "quality" in i.performance_metrics
        ]
        
        if quality_scores:
            metrics["quality_metrics"] = {
                "average_quality": sum(quality_scores) / len(quality_scores),
                "quality_variance": np.var(quality_scores) if len(quality_scores) > 1 else 0,
                "samples_with_quality": len(quality_scores)
            }
        
        return metrics
