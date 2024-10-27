import logging
import time
from typing import Dict, Any, Optional, List
from ..openrouter_agent import OpenRouterAgent, AgentInteraction
from ..local_agent import LocalAgent

logger = logging.getLogger(__name__)

class MagiAgent:
    """
    Magi agent specializing in code generation and technical problem-solving.
    Uses openai/o1-mini-2024-09-12 as frontier model and
    ibm-granite/granite-3b-code-instruct-128k as local model.
    """
    
    def __init__(self, openrouter_agent: OpenRouterAgent):
        """
        Initialize MagiAgent.
        
        Args:
            openrouter_agent: OpenRouterAgent instance configured for Magi's models
        """
        self.frontier_agent = openrouter_agent
        self.local_agent = LocalAgent(openrouter_agent.local_model)
        self.code_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {
            "code_quality": 0.0,
            "solution_efficiency": 0.0,
            "test_coverage": 0.0,
            "documentation_quality": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized MagiAgent with:")
        logger.info(f"  Frontier model: {openrouter_agent.model}")
        logger.info(f"  Local model: {openrouter_agent.local_model}")
    
    async def generate_code(self, 
                          task: str, 
                          context: Optional[str] = None,
                          language: Optional[str] = None,
                          requirements: Optional[List[str]] = None) -> AgentInteraction:
        """
        Generate code for a given task.
        
        Args:
            task: The coding task description
            context: Optional codebase context
            language: Target programming language
            requirements: Specific requirements or constraints
            
        Returns:
            AgentInteraction containing the generated code
        """
        # Evaluate task complexity
        is_complex = await self._evaluate_code_complexity(task, requirements)
        
        # Prepare coding prompt
        coding_prompt = self._construct_coding_prompt(task, context, language, requirements)
        
        # Try local model first for simpler tasks
        local_response = None
        if not is_complex:
            try:
                local_response = await self.local_agent.generate_response(
                    prompt=coding_prompt,
                    system_prompt=self._get_coding_system_prompt(language),
                    temperature=0.2,  # Lower temperature for more focused code generation
                    max_tokens=1000
                )
            except Exception as e:
                logger.warning(f"Local model code generation failed: {str(e)}")
        
        # Use frontier model if task is complex or local model failed
        if is_complex or not local_response:
            interaction = await self.frontier_agent.generate_response(
                prompt=coding_prompt,
                system_prompt=self._get_coding_system_prompt(language),
                temperature=0.2,
                max_tokens=1500
            )
            
            # If we tried local model, record performance comparison
            if local_response:
                self._record_model_comparison(local_response, interaction)
        else:
            # Convert local response to AgentInteraction format
            current_time = time.time()
            interaction = AgentInteraction(
                prompt=coding_prompt,
                response=local_response["response"],
                model=local_response["model"],
                timestamp=current_time,
                metadata=local_response["metadata"]
            )
        
        # Track code generation and update metrics
        self._update_code_history(task, interaction, is_complex, language)
        
        return interaction
    
    async def _evaluate_code_complexity(self, task: str, requirements: Optional[List[str]]) -> bool:
        """
        Evaluate if a coding task is complex.
        
        Args:
            task: The coding task
            requirements: Optional specific requirements
            
        Returns:
            Boolean indicating if task is complex
        """
        # Complexity indicators for code tasks
        complexity_indicators = [
            "optimize",
            "implement",
            "design",
            "architecture",
            "system",
            "algorithm",
            "performance",
            "scale",
            "concurrent",
            "async"
        ]
        
        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in task.lower())
        
        # Consider requirements complexity
        if requirements:
            req_complexity = sum(1 for req in requirements if any(
                indicator in req.lower() for indicator in complexity_indicators
            ))
            indicator_count += req_complexity
        
        # Consider task sophistication
        requires_advanced = any(phrase in task.lower() for phrase in [
            "distributed",
            "concurrent",
            "real-time",
            "scalable",
            "high-performance",
            "optimization",
            "machine learning",
            "neural network"
        ])
        
        return indicator_count >= 2 or requires_advanced
    
    def _construct_coding_prompt(self, 
                               task: str, 
                               context: Optional[str], 
                               language: Optional[str],
                               requirements: Optional[List[str]]) -> str:
        """Construct a detailed coding prompt."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Code Context:\n{context}\n")
            
        prompt_parts.append(f"Task Description:\n{task}\n")
        
        if language:
            prompt_parts.append(f"Programming Language: {language}\n")
            
        if requirements:
            prompt_parts.append("Requirements:")
            for req in requirements:
                prompt_parts.append(f"- {req}")
        
        prompt_parts.append("""
        Please provide:
        1. Code implementation
        2. Brief explanation of the approach
        3. Any important considerations or assumptions
        4. Example usage (if applicable)
        """)
        
        return "\n\n".join(prompt_parts)
    
    def _get_coding_system_prompt(self, language: Optional[str]) -> str:
        """Get the appropriate system prompt based on programming language."""
        base_prompt = """You are Magi, an expert coding AI specializing in software development and technical problem-solving.
        Your approach:
        1. Write clean, efficient, and maintainable code
        2. Follow best practices and design patterns
        3. Consider edge cases and error handling
        4. Provide clear documentation and explanations
        5. Optimize for performance where appropriate"""
        
        if language:
            language_specific = f"\nYou are writing code in {language}. Follow {language}-specific best practices and conventions."
            return base_prompt + language_specific
            
        return base_prompt
    
    def _update_code_history(self, 
                           task: str, 
                           interaction: AgentInteraction,
                           was_complex: bool,
                           language: Optional[str]):
        """Update code generation history and performance metrics."""
        code_record = {
            "task": task,
            "timestamp": interaction.timestamp,
            "was_complex": was_complex,
            "language": language,
            "model_used": interaction.model,
            "tokens_used": interaction.metadata["usage"]["total_tokens"] if "usage" in interaction.metadata else 0
        }
        self.code_history.append(code_record)
        
        # Keep last 100 coding tasks
        if len(self.code_history) > 100:
            self.code_history = self.code_history[-100:]
    
    def _record_model_comparison(self, local_response: Dict[str, Any], frontier_interaction: AgentInteraction):
        """Record performance comparison between local and frontier models."""
        # Calculate code similarity (you might want to use a more sophisticated metric)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(
            None, 
            local_response["response"], 
            frontier_interaction.response
        ).ratio()
        
        # Record local model performance
        self.local_agent.record_performance({
            "code_similarity": similarity,
            "was_used": similarity > 0.8  # Consider local model successful if very similar
        })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.code_history:
            return self.performance_metrics
            
        # Calculate metrics from recent history
        recent_code = self.code_history[-100:]
        
        # Code quality (if metrics were recorded)
        quality_scores = [
            task.get("performance_metrics", {}).get("code_quality", 0.0)
            for task in recent_code
        ]
        if quality_scores:
            self.performance_metrics["code_quality"] = sum(quality_scores) / len(quality_scores)
        
        # Solution efficiency
        efficiency_scores = [
            task.get("performance_metrics", {}).get("efficiency", 0.0)
            for task in recent_code
        ]
        if efficiency_scores:
            self.performance_metrics["solution_efficiency"] = sum(efficiency_scores) / len(efficiency_scores)
        
        # Test coverage
        coverage_scores = [
            task.get("performance_metrics", {}).get("test_coverage", 0.0)
            for task in recent_code
        ]
        if coverage_scores:
            self.performance_metrics["test_coverage"] = sum(coverage_scores) / len(coverage_scores)
        
        # Documentation quality
        doc_scores = [
            task.get("performance_metrics", {}).get("documentation", 0.0)
            for task in recent_code
        ]
        if doc_scores:
            self.performance_metrics["documentation_quality"] = sum(doc_scores) / len(doc_scores)
        
        # Local model performance
        local_metrics = self.local_agent.get_performance_metrics()
        if "code_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["code_similarity"]
        
        return self.performance_metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training data for the local model."""
        return self.frontier_agent.get_training_data()
