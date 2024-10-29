"""Enhanced Sage agent with improved research and knowledge synthesis capabilities."""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import asyncio

from config.unified_config import UnifiedConfig, AgentConfig
from ..openrouter_agent import OpenRouterAgent, AgentInteraction
from ..local_agent import LocalAgent
from ...data.complexity_evaluator import ComplexityEvaluator
from rag_system.core.exploration_mode import ExplorationMode
from rag_system.core.config import OpenAIGPTConfig
from rag_system.processing.advanced_nlp import AdvancedNLP
from rag_system.retrieval.graph_store import GraphStore

logger = logging.getLogger(__name__)

class ResearchManager:
    """Manages research tasks and knowledge synthesis."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.research_history: List[Dict[str, Any]] = []
        self.knowledge_graph = {}  # Simplified for now, would use proper graph DB
        self.research_metrics = {
            "depth_score": 0.0,
            "breadth_score": 0.0,
            "synthesis_quality": 0.0,
            "verification_rate": 0.0
        }
    
    async def plan_research(self, topic: str) -> Dict[str, Any]:
        """Plan research approach for a topic."""
        research_plan = {
            "topic": topic,
            "timestamp": time.time(),
            "stages": [
                {
                    "name": "initial_exploration",
                    "depth": 2,
                    "focus_areas": self._identify_focus_areas(topic)
                },
                {
                    "name": "deep_analysis",
                    "depth": 4,
                    "focus_areas": []  # To be filled after initial exploration
                },
                {
                    "name": "synthesis",
                    "dependencies": ["initial_exploration", "deep_analysis"]
                }
            ]
        }
        return research_plan
    
    def _identify_focus_areas(self, topic: str) -> List[str]:
        """Identify key areas to focus research on."""
        # This would be more sophisticated in practice
        words = topic.lower().split()
        return [w for w in words if len(w) > 4]  # Simplified example
    
    async def synthesize_knowledge(self, 
                                 research_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize research results into coherent knowledge."""
        # Track synthesis metrics
        start_time = time.time()
        
        # Group findings by theme
        themes = {}
        for result in research_results:
            theme = result.get("theme", "general")
            if theme not in themes:
                themes[theme] = []
            themes[theme].append(result)
        
        # Synthesize each theme
        synthesis = {
            "themes": {},
            "cross_cutting_insights": [],
            "confidence_scores": {}
        }
        
        for theme, findings in themes.items():
            synthesis["themes"][theme] = {
                "key_findings": self._extract_key_findings(findings),
                "implications": self._analyze_implications(findings),
                "confidence": self._calculate_confidence(findings)
            }
        
        # Identify cross-cutting insights
        synthesis["cross_cutting_insights"] = self._find_cross_cutting_insights(themes)
        
        # Update metrics
        duration = time.time() - start_time
        self.research_metrics["synthesis_quality"] = self._evaluate_synthesis_quality(synthesis)
        
        return synthesis
    
    def _extract_key_findings(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from research results."""
        # Would use more sophisticated NLP in practice
        return [f["summary"] for f in findings if f.get("importance_score", 0) > 0.7]
    
    def _analyze_implications(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Analyze implications of research findings."""
        implications = []
        for finding in findings:
            if "implications" in finding:
                implications.extend(finding["implications"])
        return list(set(implications))  # Remove duplicates
    
    def _calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for findings."""
        if not findings:
            return 0.0
        confidence_scores = [f.get("confidence", 0) for f in findings]
        return sum(confidence_scores) / len(confidence_scores)
    
    def _find_cross_cutting_insights(self, themes: Dict[str, List[Dict[str, Any]]]) -> List[str]:
        """Identify insights that span multiple themes."""
        # Would use more sophisticated analysis in practice
        insights = []
        all_implications = []
        
        for theme_findings in themes.values():
            for finding in theme_findings:
                if "implications" in finding:
                    all_implications.extend(finding["implications"])
        
        # Find common implications
        from collections import Counter
        common = Counter(all_implications)
        insights = [impl for impl, count in common.items() if count > 1]
        
        return insights
    
    def _evaluate_synthesis_quality(self, synthesis: Dict[str, Any]) -> float:
        """Evaluate the quality of knowledge synthesis."""
        # Would use more sophisticated evaluation in practice
        scores = []
        
        # Evaluate theme coverage
        scores.append(len(synthesis["themes"]) / 5)  # Normalize to 0-1
        
        # Evaluate insight depth
        insight_count = len(synthesis["cross_cutting_insights"])
        scores.append(min(1.0, insight_count / 10))
        
        # Evaluate confidence
        confidence_scores = synthesis["confidence_scores"].values()
        if confidence_scores:
            scores.append(sum(confidence_scores) / len(confidence_scores))
        
        return sum(scores) / len(scores) if scores else 0.0

class SageAgent:
    """
    Enhanced Sage agent specializing in research and knowledge synthesis.
    Uses anthropic/claude-3.5-sonnet as frontier model and
    deepseek-ai/Janus-1.3B as local model.
    """
    
    def __init__(self, 
                 openrouter_agent: OpenRouterAgent,
                 config: UnifiedConfig):
        """
        Initialize enhanced SageAgent.
        
        Args:
            openrouter_agent: OpenRouterAgent instance
            config: UnifiedConfig instance
        """
        self.config = config
        self.agent_config = config.get_agent_config("sage")
        self.frontier_agent = openrouter_agent
        self.local_agent = LocalAgent(
            model_config=self.agent_config.local_model,
            config=config
        )
        
        # Initialize support systems
        self.research_manager = ResearchManager(config)
        self.complexity_evaluator = ComplexityEvaluator(config)
        
        # Initialize exploration mode with proper configuration
        llm_config = OpenAIGPTConfig(
            api_key=config.get_api_key(),
            model_name=self.agent_config.frontier_model.name,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Initialize components with default settings
        graph_store = GraphStore(config)  # Pass config to GraphStore
        advanced_nlp = AdvancedNLP()
        
        self.exploration_mode = ExplorationMode(
            graph_store=graph_store,
            llm_config=llm_config,
            advanced_nlp=advanced_nlp
        )
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {
            "research_quality": 0.0,
            "synthesis_quality": 0.0,
            "verification_rate": 0.0,
            "local_model_performance": 0.0
        }
        
        logger.info(f"Initialized SageAgent with:")
        logger.info(f"  Frontier model: {openrouter_agent.model}")
        logger.info(f"  Local model: {openrouter_agent.local_model}")
    
    async def conduct_research(self,
                             topic: str,
                             depth: int = 3,
                             system_prompt: Optional[str] = None) -> AgentInteraction:
        """
        Conduct research on a topic with knowledge synthesis.
        
        Args:
            topic: Research topic
            depth: Research depth (1-5)
            system_prompt: Optional system prompt
            
        Returns:
            AgentInteraction containing research results
        """
        start_time = time.time()
        
        try:
            # Evaluate complexity
            complexity_evaluation = await self.complexity_evaluator.evaluate_complexity(
                agent_type="sage",
                task=f"Research: {topic}",
                context={"depth": depth}
            )
            
            # Create research plan
            research_plan = await self.research_manager.plan_research(topic)
            
            # Conduct initial exploration
            exploration_results = await self.exploration_mode.explore_knowledge_graph(
                start_node=topic,
                depth=min(depth, 3)
            )
            
            # Determine if we need frontier model
            use_frontier = complexity_evaluation["is_complex"] or depth > 2
            
            if use_frontier:
                # Use frontier model for deep research
                interaction = await self.frontier_agent.generate_response(
                    prompt=self._create_research_prompt(topic, depth, exploration_results),
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    max_tokens=2000,
                    temperature=0.8
                )
            else:
                # Use local model for basic research
                local_response = await self.local_agent.generate_response(
                    prompt=self._create_research_prompt(topic, depth, exploration_results),
                    system_prompt=system_prompt or self._get_default_system_prompt(),
                    max_tokens=1500,
                    temperature=0.7
                )
                
                # Convert to AgentInteraction format
                interaction = AgentInteraction(
                    prompt=topic,
                    response=local_response["response"],
                    model=local_response["model"],
                    timestamp=time.time(),
                    metadata={
                        **local_response["metadata"],
                        "research_plan": research_plan,
                        "exploration_results": exploration_results
                    }
                )
            
            # Synthesize knowledge
            synthesis = await self.research_manager.synthesize_knowledge([
                {
                    "content": interaction.response,
                    "source": interaction.model,
                    "confidence": complexity_evaluation["confidence"]
                }
            ])
            
            # Add synthesis to interaction metadata
            interaction.metadata["knowledge_synthesis"] = synthesis
            
            # Update performance metrics
            duration = time.time() - start_time
            self._update_metrics(interaction, {
                "duration": duration,
                "depth": depth,
                "complexity_score": complexity_evaluation["complexity_score"],
                "synthesis_quality": synthesis.get("quality_score", 0.0)
            })
            
            return interaction
            
        except Exception as e:
            logger.error(f"Error conducting research: {str(e)}")
            raise
    
    def _create_research_prompt(self,
                              topic: str,
                              depth: int,
                              exploration_results: Dict[str, Any]) -> str:
        """Create detailed research prompt."""
        return f"""Conduct {'deep' if depth > 2 else 'focused'} research on: {topic}

Initial Knowledge Graph Exploration:
{self._format_exploration_results(exploration_results)}

Research Parameters:
- Depth Level: {depth}/5
- Focus Areas: {', '.join(self.research_manager._identify_focus_areas(topic))}
- Required: Evidence-based findings
- Required: Confidence levels for claims
- Required: Implications and connections

Please provide:
1. Key findings with evidence
2. Analysis of implications
3. Identified knowledge gaps
4. Confidence levels for each finding
5. Suggested areas for deeper investigation

Format the response with clear sections and hierarchical organization."""
    
    def _format_exploration_results(self, results: Dict[str, Any]) -> str:
        """Format knowledge graph exploration results."""
        formatted = []
        
        if "explored_nodes" in results:
            formatted.append(f"Related Concepts: {', '.join(results['explored_nodes'][:5])}")
        
        if "exploration_results" in results:
            formatted.append("\nKey Relationships:")
            for result in results["exploration_results"][:3]:
                formatted.append(f"- {result.get('source', '')} → {result.get('target', '')}")
        
        return "\n".join(formatted)
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for Sage agent."""
        return """You are Sage, an AI researcher specializing in comprehensive analysis and knowledge synthesis.
        Your approach is:
        1. Systematic exploration of topics
        2. Evidence-based analysis
        3. Identification of patterns and connections
        4. Clear articulation of findings
        5. Recognition of limitations and uncertainties
        
        You excel at:
        - Deep research and analysis
        - Knowledge synthesis
        - Pattern recognition
        - Implications analysis
        - Evidence evaluation
        """
    
    def _update_metrics(self,
                       interaction: AgentInteraction,
                       performance: Dict[str, Any]):
        """Update comprehensive performance metrics."""
        # Update research quality
        if "synthesis_quality" in performance:
            current_quality = self.performance_metrics["research_quality"]
            new_quality = performance["synthesis_quality"]
            self.performance_metrics["research_quality"] = (
                current_quality * 0.9 + new_quality * 0.1  # Rolling average
            )
        
        # Update synthesis quality
        synthesis_quality = self.research_manager.research_metrics["synthesis_quality"]
        self.performance_metrics["synthesis_quality"] = synthesis_quality
        
        # Update verification rate (if available)
        if "verified_claims" in interaction.metadata:
            verified = interaction.metadata["verified_claims"]
            total = interaction.metadata.get("total_claims", 1)
            self.performance_metrics["verification_rate"] = verified / total
        
        # Update local model performance
        local_metrics = self.local_agent.get_performance_metrics()
        if "average_similarity" in local_metrics:
            self.performance_metrics["local_model_performance"] = local_metrics["average_similarity"]
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add research manager metrics
        metrics.update(self.research_manager.research_metrics)
        
        # Add local model metrics
        local_metrics = self.local_agent.get_performance_metrics()
        metrics.update({
            f"local_{k}": v 
            for k, v in local_metrics.items()
        })
        
        return metrics
    
    def get_training_data(self) -> List[Dict[str, Any]]:
        """Get training data for the local model."""
        return self.frontier_agent.get_training_data()
