import logging
import time
from typing import Dict, Any, Optional, List
from ..openrouter_agent import OpenRouterAgent, AgentInteraction
from ..local_agent import LocalAgent

logger = logging.getLogger(__name__)

class KingAgent:
    """
    King agent specializing in complex instruction-following and strategic thinking.
    Uses nvidia/llama-3.1-nemotron-70b-instruct as frontier model and
    Qwen/Qwen2.5-3B-Instruct as local model.
    """
    
    def __init__(self, openrouter_agent: OpenRouterAgent):
        """
        Initialize KingAgent.
        
        Args:
            openrouter_agent: OpenRouterAgent instance configured for King's models
        """
        self.frontier_agent = openrouter_agent
        self.local_agent = LocalAgent(openrouter_agent.local_model)
        self.task_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {
            "task_success_rate": 0.0,
            "avg_response_quality": 0.0,
            "complexity_handling": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized KingAgent with:")
        logger.info(f"  Frontier model: {openrouter_agent.model}")
        logger.info(f"  Local model: {openrouter_agent.local_model}")
    
    async def process_task(self, task: str, system_prompt: Optional[str] = None) -> AgentInteraction:
        """
        Process a task using either the frontier or local model based on complexity.
        
        Args:
            task: The task to process
            system_prompt: Optional system prompt for context
            
        Returns:
            AgentInteraction containing the response and metadata
        """
        # Evaluate task complexity
        is_complex = await self._evaluate_complexity(task)
        
        # Try local model first if task isn't complex
        local_response = None
        if not is_complex:
            try:
                local_response = await self.local_agent.generate_response(
                    prompt=task,
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    temperature=0.5  # Lower temperature for local model
                )
            except Exception as e:
                logger.warning(f"Local model generation failed: {str(e)}")
        
        # Use frontier model if task is complex or local model failed
        if is_complex or not local_response:
            interaction = await self.frontier_agent.generate_response(
                prompt=task,
                system_prompt=system_prompt or self._get_default_system_prompt(),
                temperature=0.7
            )
            
            # If we tried local model, record performance comparison
            if local_response:
                self._record_model_comparison(local_response, interaction)
        else:
            # Convert local response to AgentInteraction format
            current_time = time.time()
            interaction = AgentInteraction(
                prompt=task,
                response=local_response["response"],
                model=local_response["model"],
                timestamp=current_time,
                metadata=local_response["metadata"]
            )
        
        # Track task and update metrics
        self._update_task_history(task, interaction, is_complex)
        
        return interaction
    
    async def _evaluate_complexity(self, task: str) -> bool:
        """
        Evaluate if a task is complex enough to require the frontier model.
        
        Args:
            task: The task to evaluate
            
        Returns:
            Boolean indicating if task is complex
        """
        # For now, use a simple heuristic based on task length and keyword presence
        complexity_indicators = [
            "analyze",
            "evaluate",
            "compare",
            "synthesize",
            "design",
            "optimize",
            "recommend",
            "strategic",
            "complex",
            "multi-step"
        ]
        
        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in task.lower())
        
        # Consider task length
        is_long = len(task.split()) > 50
        
        return indicator_count >= 2 or is_long
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for King agent."""
        return """You are King, an advanced AI agent specializing in complex problem-solving and strategic thinking.
        Your approach is:
        1. Analyze problems thoroughly from multiple angles
        2. Break down complex tasks into manageable steps
        3. Consider long-term implications and strategic impact
        4. Provide clear, actionable guidance
        5. Maintain awareness of context and constraints
        """
    
    def _update_task_history(self, task: str, interaction: AgentInteraction, was_complex: bool):
        """Update task history and performance metrics."""
        task_record = {
            "task": task,
            "timestamp": interaction.timestamp,
            "was_complex": was_complex,
            "model_used": interaction.model,
            "tokens_used": interaction.metadata["usage"]["total_tokens"] if "usage" in interaction.metadata else 0
        }
        self.task_history.append(task_record)
        
        # Update rolling performance metrics
        if len(self.task_history) > 100:
            self.task_history = self.task_history[-100:]  # Keep last 100 tasks
    
    def _record_model_comparison(self, local_response: Dict[str, Any], frontier_interaction: AgentInteraction):
        """Record performance comparison between local and frontier models."""
        # Calculate response similarity (you might want to use a more sophisticated metric)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(
            None, 
            local_response["response"], 
            frontier_interaction.response
        ).ratio()
        
        # Record local model performance
        self.local_agent.record_performance({
            "response_similarity": similarity,
            "was_used": similarity > 0.8  # Consider local model successful if very similar
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.task_history:
            return self.performance_metrics
            
        # Calculate metrics from recent history
        recent_tasks = self.task_history[-100:]
        
        # Task success rate (if performance metrics were recorded)
        success_count = sum(
            1 for task in recent_tasks 
            if task.get("performance_metrics", {}).get("success", False)
        )
        self.performance_metrics["task_success_rate"] = success_count / len(recent_tasks)
        
        # Average response quality
        quality_scores = [
            task.get("performance_metrics", {}).get("quality", 0.0)
            for task in recent_tasks
        ]
        if quality_scores:
            self.performance_metrics["avg_response_quality"] = sum(quality_scores) / len(quality_scores)
        
        # Complexity handling (ratio of successful complex tasks)
        complex_tasks = [task for task in recent_tasks if task["was_complex"]]
        if complex_tasks:
            complex_success = sum(
                1 for task in complex_tasks
                if task.get("performance_metrics", {}).get("success", False)
            )
            self.performance_metrics["complexity_handling"] = complex_success / len(complex_tasks)
        
        # Local model performance
        local_metrics = self.local_agent.get_performance_metrics()
        if "response_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["response_similarity"]
        
        return self.performance_metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training data for the local model."""
        return self.frontier_agent.get_training_data()
