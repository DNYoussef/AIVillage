"""
Task router for intelligent model selection based on task characteristics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .model_config import TaskType, MODEL_ROUTING_CONFIG
from .openrouter_client import OpenRouterClient, APIResponse


logger = logging.getLogger(__name__)


@dataclass
class TaskContext:
    """Context for a task to help with routing decisions"""
    difficulty_level: int  # 1-10
    domain: str  # e.g., "python_programming", "mathematical_proofs"
    expected_length: str  # "short", "medium", "long"
    requires_reasoning: bool
    requires_creativity: bool
    cost_sensitive: bool = False
    quality_priority: bool = True


class TaskRouter:
    """Routes tasks to optimal models based on task characteristics and performance."""
    
    def __init__(self, openrouter_client: Optional[OpenRouterClient] = None):
        """Initialize task router.
        
        Args:
            openrouter_client: OpenRouter client instance. Creates new one if not provided.
        """
        self.client = openrouter_client or OpenRouterClient()
        self.performance_history: Dict[Tuple[TaskType, str], List[float]] = {}
    
    def classify_task(
        self,
        prompt: str,
        context: Optional[TaskContext] = None
    ) -> TaskType:
        """Classify a task based on prompt and context.
        
        Args:
            prompt: The prompt to classify
            context: Optional context about the task
            
        Returns:
            TaskType for routing
        """
        prompt_lower = prompt.lower()
        
        # Keywords for classification
        generation_keywords = ['generate', 'create', 'write a problem', 'design a question']
        evaluation_keywords = ['evaluate', 'grade', 'assess', 'check', 'score']
        variation_keywords = ['vary', 'modify', 'alternate', 'rephrase', 'different version']
        research_keywords = ['research', 'analyze', 'investigate', 'document', 'explain']
        code_keywords = ['code', 'program', 'implement', 'function', 'algorithm', 'debug']
        math_keywords = ['prove', 'theorem', 'mathematical', 'equation', 'calculus']
        
        # Check for specific task types
        if any(keyword in prompt_lower for keyword in generation_keywords):
            if context and context.domain in ['python_programming', 'algorithm_design']:
                return TaskType.CODE_GENERATION
            elif context and context.domain == 'mathematical_proofs':
                return TaskType.MATHEMATICAL_REASONING
            return TaskType.PROBLEM_GENERATION
        
        elif any(keyword in prompt_lower for keyword in evaluation_keywords):
            return TaskType.EVALUATION_GRADING
        
        elif any(keyword in prompt_lower for keyword in variation_keywords):
            return TaskType.CONTENT_VARIATION
        
        elif any(keyword in prompt_lower for keyword in research_keywords):
            return TaskType.RESEARCH_DOCUMENTATION
        
        elif any(keyword in prompt_lower for keyword in code_keywords):
            return TaskType.CODE_GENERATION
        
        elif any(keyword in prompt_lower for keyword in math_keywords):
            return TaskType.MATHEMATICAL_REASONING
        
        # Default based on context if available
        if context:
            if context.domain in ['python_programming', 'algorithm_design', 'data_structures']:
                return TaskType.CODE_GENERATION
            elif context.domain in ['mathematical_proofs', 'computational_complexity', 'numerical_analysis']:
                return TaskType.MATHEMATICAL_REASONING
        
        # Default to problem generation
        return TaskType.PROBLEM_GENERATION
    
    def select_model_for_task(
        self,
        task_type: TaskType,
        context: Optional[TaskContext] = None
    ) -> str:
        """Select the best model for a task based on type and context.
        
        Args:
            task_type: Type of task
            context: Optional context for better selection
            
        Returns:
            Model name to use
        """
        config = MODEL_ROUTING_CONFIG[task_type]
        
        # If cost sensitive and not high priority, prefer budget options
        if context and context.cost_sensitive and not context.quality_priority:
            # Try to use a fallback model if it's cheaper
            fallbacks = config.get('fallback', [])
            for model in fallbacks:
                if 'mini' in model or 'haiku' in model or 'flash' in model:
                    return model
        
        # For high difficulty tasks, always use primary model
        if context and context.difficulty_level >= 8:
            return config['primary']
        
        # Check performance history
        primary_model = config['primary']
        if self._should_use_fallback(task_type, primary_model):
            fallbacks = config.get('fallback', [])
            if fallbacks:
                return fallbacks[0]
        
        return primary_model
    
    def _should_use_fallback(self, task_type: TaskType, model: str) -> bool:
        """Determine if we should use a fallback based on performance history."""
        key = (task_type, model)
        
        if key not in self.performance_history:
            return False
        
        history = self.performance_history[key]
        if len(history) < 5:
            return False
        
        # If recent performance is poor (e.g., high error rate), use fallback
        recent_errors = sum(1 for score in history[-5:] if score < 0.5)
        return recent_errors >= 3
    
    async def route_task(
        self,
        prompt: str,
        context: Optional[TaskContext] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> APIResponse:
        """Route a task to the optimal model and execute it.
        
        Args:
            prompt: The task prompt
            context: Optional context about the task
            messages: Optional message history (uses prompt if not provided)
            **kwargs: Additional arguments for the API call
            
        Returns:
            APIResponse with results
        """
        # Classify the task
        task_type = self.classify_task(prompt, context)
        logger.info(f"Classified task as: {task_type.value}")
        
        # Select model based on task and context
        model = self.select_model_for_task(task_type, context)
        logger.info(f"Selected model: {model}")
        
        # Prepare messages if not provided
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Override model if context suggests it
        model_override = None
        if context and context.quality_priority and context.difficulty_level >= 9:
            # For very difficult tasks, always use the best model
            model_override = 'anthropic/claude-3-opus-20240229'
        
        # Execute the request
        response = await self.client.complete(
            task_type=task_type,
            messages=messages,
            model_override=model_override,
            **kwargs
        )
        
        # Update performance tracking (simplified - in production would track actual quality)
        self._update_performance(task_type, response.model_used, 1.0)
        
        return response
    
    def _update_performance(self, task_type: TaskType, model: str, score: float):
        """Update performance history for a model on a task type."""
        key = (task_type, model)
        
        if key not in self.performance_history:
            self.performance_history[key] = []
        
        self.performance_history[key].append(score)
        
        # Keep only recent history
        if len(self.performance_history[key]) > 100:
            self.performance_history[key] = self.performance_history[key][-100:]
    
    async def generate_problem_with_variations(
        self,
        domain: str,
        difficulty: int,
        num_variations: int = 3
    ) -> Dict[str, Any]:
        """Generate a problem and create variations of it.
        
        Args:
            domain: Problem domain
            difficulty: Difficulty level (1-10)
            num_variations: Number of variations to create
            
        Returns:
            Dictionary with original problem and variations
        """
        context = TaskContext(
            difficulty_level=difficulty,
            domain=domain,
            expected_length="medium",
            requires_reasoning=True,
            requires_creativity=True
        )
        
        # Generate the original problem
        problem_prompt = f"""Generate a {domain} problem at difficulty level {difficulty}/10.
The problem should be appropriate for training a specialized AI agent.
Include a clear problem statement and the expected solution.
Format: 
Problem: [problem statement]
Solution: [detailed solution]"""
        
        original_response = await self.route_task(problem_prompt, context)
        original_problem = original_response.content
        
        # Generate variations
        variations = []
        for i in range(num_variations):
            variation_prompt = f"""Create a variation of this problem that tests the same concepts but with different specifics:

Original Problem:
{original_problem}

Create variation #{i+1} that maintains the same difficulty level but changes the context, numbers, or specific details."""
            
            # Use a more cost-effective model for variations
            variation_context = TaskContext(
                difficulty_level=difficulty,
                domain=domain,
                expected_length="medium",
                requires_reasoning=False,
                requires_creativity=True,
                cost_sensitive=True,
                quality_priority=False
            )
            
            variation_response = await self.route_task(variation_prompt, variation_context)
            variations.append(variation_response)
        
        # Calculate total cost and collect models used
        total_variation_cost = sum(v.cost for v in variations)
        variation_models = [v.model_used for v in variations]
        
        return {
            'original': original_problem,
            'variations': [v.content for v in variations],  # Extract content for variations
            'domain': domain,
            'difficulty': difficulty,
            'total_cost': original_response.cost + total_variation_cost,
            'models_used': {
                'original': original_response.model_used,
                'variations': variation_models
            }
        }
    
    async def evaluate_with_explanation(
        self,
        question: str,
        answer: str,
        expected_answer: str
    ) -> Dict[str, Any]:
        """Evaluate an answer with detailed explanation.
        
        Args:
            question: The question that was answered
            answer: The provided answer
            expected_answer: The expected answer
            
        Returns:
            Evaluation results with explanation
        """
        eval_prompt = f"""Evaluate this answer and provide a detailed assessment:

Question: {question}
Expected Answer: {expected_answer}
Provided Answer: {answer}

Please evaluate:
1. Correctness (0-100%)
2. Completeness (0-100%)
3. Clarity (0-100%)
4. Key missing elements (if any)
5. Suggestions for improvement

Format your response as JSON."""
        
        context = TaskContext(
            difficulty_level=5,  # Evaluation is medium difficulty
            domain="evaluation",
            expected_length="medium",
            requires_reasoning=True,
            requires_creativity=False,
            cost_sensitive=True,  # Use efficient model for grading
            quality_priority=False
        )
        
        response = await self.route_task(eval_prompt, context)
        
        # Parse the evaluation (in production, would use proper JSON parsing)
        return {
            'evaluation': response.content,
            'model_used': response.model_used,
            'cost': response.cost,
            'latency': response.latency
        }
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about task routing."""
        stats = {
            'performance_by_task': {},
            'model_usage': self.client.get_metrics_summary()
        }
        
        # Aggregate performance by task type
        for (task_type, model), scores in self.performance_history.items():
            if task_type.value not in stats['performance_by_task']:
                stats['performance_by_task'][task_type.value] = {}
            
            stats['performance_by_task'][task_type.value][model] = {
                'avg_score': sum(scores) / len(scores) if scores else 0,
                'total_requests': len(scores)
            }
        
        return stats