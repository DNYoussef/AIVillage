import logging
import time
from typing import Dict, Any, Optional, List
from ..openrouter_agent import OpenRouterAgent, AgentInteraction
from ..local_agent import LocalAgent

logger = logging.getLogger(__name__)

class SageAgent:
    """
    Sage agent specializing in research, analysis, and knowledge synthesis.
    Uses anthropic/claude-3.5-sonnet as frontier model and
    deepseek-ai/Janus-1.3B as local model for vision and web capabilities.
    """
    
    def __init__(self, openrouter_agent: OpenRouterAgent):
        """
        Initialize SageAgent.
        
        Args:
            openrouter_agent: OpenRouterAgent instance configured for Sage's models
        """
        self.frontier_agent = openrouter_agent
        self.local_agent = LocalAgent(openrouter_agent.local_model)
        self.research_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, float] = {
            "research_quality": 0.0,
            "analysis_depth": 0.0,
            "source_integration": 0.0,
            "insight_generation": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized SageAgent with:")
        logger.info(f"  Frontier model: {openrouter_agent.model}")
        logger.info(f"  Local model: {openrouter_agent.local_model}")
    
    async def conduct_research(self, 
                             query: str, 
                             context: Optional[str] = None,
                             depth: str = "standard") -> AgentInteraction:
        """
        Conduct research and analysis on a given query.
        
        Args:
            query: The research query or topic
            context: Optional background context
            depth: Research depth ("quick", "standard", or "deep")
            
        Returns:
            AgentInteraction containing the research results
        """
        # Determine if query requires deep analysis
        requires_deep_analysis = await self._evaluate_research_complexity(query, depth)
        
        # Prepare research prompt
        research_prompt = self._construct_research_prompt(query, context, depth)
        
        # Try local model first for non-deep research
        local_response = None
        if not requires_deep_analysis and depth != "deep":
            try:
                local_response = await self.local_agent.generate_response(
                    prompt=research_prompt,
                    system_prompt=self._get_research_system_prompt(depth),
                    temperature=0.6,
                    max_tokens=1000
                )
            except Exception as e:
                logger.warning(f"Local model research failed: {str(e)}")
        
        # Use frontier model if deep analysis needed or local model failed
        if requires_deep_analysis or not local_response:
            interaction = await self.frontier_agent.generate_response(
                prompt=research_prompt,
                system_prompt=self._get_research_system_prompt(depth),
                temperature=0.8,
                max_tokens=2000 if depth == "deep" else 1000
            )
            
            # If we tried local model, record performance comparison
            if local_response:
                self._record_model_comparison(local_response, interaction)
        else:
            # Convert local response to AgentInteraction format
            current_time = time.time()
            interaction = AgentInteraction(
                prompt=research_prompt,
                response=local_response["response"],
                model=local_response["model"],
                timestamp=current_time,
                metadata=local_response["metadata"]
            )
        
        # Track research and update metrics
        self._update_research_history(query, interaction, requires_deep_analysis, depth)
        
        return interaction
    
    async def _evaluate_research_complexity(self, query: str, depth: str) -> bool:
        """
        Evaluate if a research query requires deep analysis.
        
        Args:
            query: The research query
            depth: Requested research depth
            
        Returns:
            Boolean indicating if deep analysis is needed
        """
        if depth == "deep":
            return True
            
        # Complexity indicators for research tasks
        complexity_indicators = [
            "analyze",
            "compare",
            "evaluate",
            "synthesize",
            "implications",
            "relationship",
            "impact",
            "trends",
            "patterns",
            "framework"
        ]
        
        # Count complexity indicators
        indicator_count = sum(1 for indicator in complexity_indicators if indicator in query.lower())
        
        # Consider query sophistication
        requires_synthesis = any(phrase in query.lower() for phrase in [
            "how does",
            "why does",
            "what are the implications",
            "how do",
            "what is the relationship",
            "what patterns"
        ])
        
        return indicator_count >= 2 or requires_synthesis
    
    def _construct_research_prompt(self, query: str, context: Optional[str], depth: str) -> str:
        """Construct a detailed research prompt."""
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Context:\n{context}\n")
            
        prompt_parts.append(f"Research Query:\n{query}\n")
        
        depth_instructions = {
            "quick": "Provide a concise overview of the key points.",
            "standard": "Conduct a thorough analysis covering main aspects and supporting evidence.",
            "deep": """Perform an in-depth analysis including:
            1. Comprehensive examination of all relevant aspects
            2. Evaluation of multiple perspectives and approaches
            3. Analysis of relationships and patterns
            4. Synthesis of insights and implications
            5. Identification of gaps and future directions"""
        }
        
        prompt_parts.append(f"Instructions:\n{depth_instructions[depth]}")
        
        return "\n\n".join(prompt_parts)
    
    def _get_research_system_prompt(self, depth: str) -> str:
        """Get the appropriate system prompt based on research depth."""
        base_prompt = """You are Sage, an AI researcher specializing in analysis, research, and knowledge synthesis.
        Your approach:
        1. Thoroughly examine available information
        2. Identify patterns and relationships
        3. Synthesize insights from multiple sources
        4. Generate novel perspectives and implications
        5. Maintain academic rigor and clarity"""
        
        depth_additions = {
            "quick": "\nFocus on providing clear, concise key insights.",
            "standard": "\nProvide comprehensive analysis while maintaining clarity and relevance.",
            "deep": """\nConduct exhaustive analysis including:
            - Multiple theoretical frameworks
            - Cross-domain implications
            - Systematic evaluation of evidence
            - Integration of diverse perspectives
            - Meta-analysis where applicable"""
        }
        
        return base_prompt + depth_additions[depth]
    
    def _update_research_history(self, 
                               query: str, 
                               interaction: AgentInteraction,
                               was_complex: bool,
                               depth: str):
        """Update research history and performance metrics."""
        research_record = {
            "query": query,
            "timestamp": interaction.timestamp,
            "was_complex": was_complex,
            "depth": depth,
            "model_used": interaction.model,
            "tokens_used": interaction.metadata["usage"]["total_tokens"] if "usage" in interaction.metadata else 0
        }
        self.research_history.append(research_record)
        
        # Keep last 100 research tasks
        if len(self.research_history) > 100:
            self.research_history = self.research_history[-100:]
    
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
        if not self.research_history:
            return self.performance_metrics
            
        # Calculate metrics from recent history
        recent_research = self.research_history[-100:]
        
        # Research quality (if metrics were recorded)
        quality_scores = [
            task.get("performance_metrics", {}).get("quality", 0.0)
            for task in recent_research
        ]
        if quality_scores:
            self.performance_metrics["research_quality"] = sum(quality_scores) / len(quality_scores)
        
        # Analysis depth
        deep_research = [task for task in recent_research if task["depth"] == "deep"]
        if deep_research:
            depth_scores = [
                task.get("performance_metrics", {}).get("depth_score", 0.0)
                for task in deep_research
            ]
            if depth_scores:
                self.performance_metrics["analysis_depth"] = sum(depth_scores) / len(depth_scores)
        
        # Source integration
        source_scores = [
            task.get("performance_metrics", {}).get("source_integration", 0.0)
            for task in recent_research
        ]
        if source_scores:
            self.performance_metrics["source_integration"] = sum(source_scores) / len(source_scores)
        
        # Insight generation
        insight_scores = [
            task.get("performance_metrics", {}).get("insight_score", 0.0)
            for task in recent_research
        ]
        if insight_scores:
            self.performance_metrics["insight_generation"] = sum(insight_scores) / len(insight_scores)
        
        # Local model performance
        local_metrics = self.local_agent.get_performance_metrics()
        if "response_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["response_similarity"]
        
        return self.performance_metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training data for the local model."""
        return self.frontier_agent.get_training_data()
