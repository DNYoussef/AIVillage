"""Enhanced hybrid retrieval system with dual-level and plan-aware capabilities."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig
from ..retrieval.vector_store import VectorStore
from ..retrieval.graph_store import GraphStore
from ..core.structures import RetrievalResult, RetrievalPlan
from ..utils.graph_utils import distance_sensitive_linearization
from ..utils.error_handling import log_and_handle_errors, ErrorContext

class HybridRetriever(BaseComponent):
    """
    Enhanced hybrid retrieval system combining vector and graph-based approaches.
    Implements dual-level retrieval, active retrieval with feedback, and plan-aware retrieval.
    """
    
    def __init__(self, config: UnifiedConfig):
        """
        Initialize hybrid retriever.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.llm = None  # Will be initialized with language model
        self.agent = None  # Will be initialized with agent
        self.current_results = []  # Store current results for metrics
        self.initialized = False
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize retriever components."""
        if not self.initialized:
            await self.vector_store.initialize()
            await self.graph_store.initialize()
            self.initialized = True
    
    @log_and_handle_errors()
    async def shutdown(self) -> None:
        """Shutdown retriever components."""
        await self.vector_store.shutdown()
        await self.graph_store.shutdown()
        self.initialized = False
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        return {
            "initialized": self.initialized,
            "vector_store": await self.vector_store.get_status(),
            "graph_store": await self.graph_store.get_status(),
            "current_results_count": len(self.current_results)
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Dict[str, Any]) -> None:
        """Update component configuration."""
        self.config = config
        await self.vector_store.update_config(config)
        await self.graph_store.update_config(config)

    @log_and_handle_errors()
    async def retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Retrieve documents using dual-level approach.
        
        Args:
            query: The query string
            k: Number of results to retrieve
            timestamp: Optional timestamp filter
            
        Returns:
            List of retrieval results
        """
        async with ErrorContext("HybridRetriever"):
            return await self.dual_level_retrieve(query, k, timestamp)

    async def dual_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement dual-level retrieval as described in LightRAG."""
        low_level_results = await self.low_level_retrieve(query, k, timestamp)
        high_level_results = await self.high_level_retrieve(query, k, timestamp)
        combined_results = self.merge_results(low_level_results, high_level_results)
        return combined_results[:k]

    async def low_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement specific entity and relation retrieval."""
        return await self.graph_store.retrieve(query, k, timestamp)

    async def high_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement broader topic and theme retrieval."""
        query_vector = await self.agent.get_embedding(query)
        return await self.vector_store.retrieve(query_vector, k, timestamp)

    def _generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate comprehensive feedback for query refinement."""
        relevance_scores = self._calculate_relevance_scores(query, results)
        coverage_analysis = self._analyze_coverage(results)
        semantic_gaps = self._identify_semantic_gaps(query, results)
        suggested_expansions = self._generate_query_expansions(query, results)
        
        return {
            "relevance_scores": relevance_scores,
            "coverage_analysis": coverage_analysis,
            "semantic_gaps": semantic_gaps,
            "suggested_expansions": suggested_expansions,
            "confidence_score": self._calculate_confidence_score(relevance_scores, coverage_analysis),
            "diversity_metric": self._calculate_diversity_metric(results),
            "temporal_distribution": self._analyze_temporal_distribution(results)
        }

    def _calculate_relevance_scores(self, query: str, results: List[RetrievalResult]) -> List[float]:
        """Calculate semantic similarity between query and results."""
        query_embedding = self.agent.get_embedding(query)
        result_embeddings = [self.agent.get_embedding(result.content) for result in results]
        similarities = cosine_similarity([query_embedding], result_embeddings)[0]
        
        # Normalize scores
        min_score = np.min(similarities)
        max_score = np.max(similarities)
        if max_score > min_score:
            normalized_scores = (similarities - min_score) / (max_score - min_score)
        else:
            normalized_scores = similarities
            
        return normalized_scores.tolist()

    def _analyze_coverage(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Analyze how well the results cover different aspects of the topic.

        :param results: List of retrieval results.
        :return: Coverage analysis data.
        """
        # Extract key concepts from results
        concepts = self._extract_key_concepts(results)
        concept_overlap = self._calculate_concept_overlap(concepts)
        coverage_gaps = self._identify_coverage_gaps(concepts)
        
        return {
            "concept_coverage": len(concepts),
            "concept_overlap": concept_overlap,
            "coverage_gaps": coverage_gaps,
            "coverage_score": self._calculate_coverage_score(concepts, coverage_gaps)
        }

    def _identify_semantic_gaps(self, query: str, results: List[RetrievalResult]) -> List[str]:
        """
        Identify semantic concepts from the query that are not well-represented in results.

        :param query: The query string.
        :param results: List of retrieval results.
        :return: List of identified semantic gaps.
        """
        # Extract query concepts
        query_concepts = self._extract_key_concepts([RetrievalResult(content=query, id="query", score=1.0)])
        result_concepts = self._extract_key_concepts(results)
        return [concept for concept in query_concepts if concept not in result_concepts]

    def _generate_query_expansions(self, query: str, results: List[RetrievalResult]) -> List[str]:
        """
        Generate suggested query expansions based on results analysis.

        :param query: The original query string.
        :param results: Current retrieval results.
        :return: List of suggested query expansions.
        """
        # Extract key terms from high-scoring results
        high_scoring_results = [r for r in results if r.score > 0.7]
        key_terms = self._extract_key_terms(high_scoring_results)
        
        expansions = []
        # Synonym-based expansion
        synonyms = self._get_synonyms(query)
        expansions.extend([f"{query} {syn}" for syn in synonyms])
        
        # Context-based expansion
        context_terms = self._extract_context_terms(results)
        expansions.extend([f"{query} {term}" for term in context_terms])
        
        # Concept-based expansion
        concepts = self._extract_key_concepts(results)
        expansions.extend([f"{query} {concept}" for concept in concepts])
        
        return list(set(expansions))

    def _calculate_confidence_score(self, relevance_scores: List[float], coverage_analysis: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the retrieval results.

        :param relevance_scores: List of relevance scores.
        :param coverage_analysis: Coverage analysis data.
        :return: Confidence score between 0 and 1.
        """
        # Weight factors for different components
        relevance_weight = 0.4
        coverage_weight = 0.3
        diversity_weight = 0.3
        
        avg_relevance = np.mean(relevance_scores)
        coverage_score = coverage_analysis['coverage_score']
        diversity_score = self._calculate_diversity_metric(self.current_results)
        
        confidence = (
            relevance_weight * avg_relevance +
            coverage_weight * coverage_score +
            diversity_weight * diversity_score
        )
        
        return min(max(confidence, 0.0), 1.0)

    def _calculate_diversity_metric(self, results: List[RetrievalResult]) -> float:
        """
        Calculate diversity metric for results.

        :param results: List of retrieval results.
        :return: Diversity score between 0 and 1.
        """
        if not results:
            return 0.0
            
        embeddings = [self.agent.get_embedding(r.content) for r in results]
        similarities = cosine_similarity(embeddings)
        avg_similarity = (np.sum(similarities) - len(results)) / (len(results) * (len(results) - 1))
        return 1.0 - avg_similarity

    def _analyze_temporal_distribution(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Analyze temporal distribution of results.

        :param results: List of retrieval results.
        :return: Temporal analysis data.
        """
        timestamps = [r.timestamp for r in results if r.timestamp]
        
        if not timestamps:
            return {"temporal_coverage": 0.0, "time_range": None}
            
        time_range = max(timestamps) - min(timestamps)
        time_buckets = np.histogram(timestamps, bins=10)
        
        return {
            "temporal_coverage": len(timestamps) / len(results),
            "time_range": str(time_range),
            "distribution": {
                "counts": time_buckets[0].tolist(),
                "bin_edges": [str(edge) for edge in time_buckets[1]]
            }
        }

    def merge_results(self, low_level_results: List[RetrievalResult], high_level_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Merge and deduplicate low-level and high-level retrieval results.

        :param low_level_results: Results from low-level retrieval.
        :param high_level_results: Results from high-level retrieval.
        :return: Merged and deduplicated list of results.
        """
        combined_results = low_level_results + high_level_results
        return self._combine_results(combined_results)

    def _combine_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Combine and deduplicate retrieval results.

        :param results: List of retrieval results.
        :return: Deduplicated and sorted list of results.
        """
        unique_results = {}
        for result in results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        return sorted(unique_results.values(), key=lambda x: x.score, reverse=True)

    @log_and_handle_errors()
    async def active_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Active retrieval with feedback iterations.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        initial_results = await self.retrieve(query, k, timestamp)
        feedback_iterations = self.config.FEEDBACK_ITERATIONS
        current_results = initial_results

        for _ in range(feedback_iterations):
            feedback = self._generate_feedback(query, current_results)
            refined_query = self._refine_query(query, feedback)
            new_results = await self.retrieve(refined_query, k, timestamp)
            current_results = self._merge_results(current_results, new_results)

        return current_results

    def _refine_query(self, original_query: str, feedback: Dict[str, Any]) -> str:
        """
        Refine the query based on feedback.

        :param original_query: The original query string.
        :param feedback: Feedback data for refinement.
        :return: Refined query string.
        """
        # Extract relevant components from feedback
        semantic_gaps = feedback.get('semantic_gaps', [])
        suggested_expansions = feedback.get('suggested_expansions', [])
        
        if semantic_gaps:
            gap_terms = ' '.join(semantic_gaps[:2])
            return f"{original_query} {gap_terms}"
        elif suggested_expansions:
            return suggested_expansions[0]
        return original_query

    def _merge_results(self, old_results: List[RetrievalResult], new_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Merge old and new results."""
        return self._combine_results(old_results + new_results)

    @log_and_handle_errors()
    async def plan_aware_retrieve(self, query: str, k: int, plan: RetrievalPlan, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Plan-aware retrieval."""
        initial_results = await self.retrieve(query, k, timestamp)
        return self._apply_plan(initial_results, plan)

    def _apply_plan(self, results: List[RetrievalResult], plan: RetrievalPlan) -> List[RetrievalResult]:
        """Apply retrieval plan to filter and rank results."""
        filtered_results = results

        # Apply filters
        if plan.filters.get("keywords"):
            filtered_results = [r for r in filtered_results if any(kw in r.content for kw in plan.filters["keywords"])]

        if plan.filters.get("date_range"):
            start_date, end_date = plan.filters["date_range"]
            filtered_results = [r for r in filtered_results if start_date <= r.timestamp <= end_date]

        if plan.filters.get("source_types"):
            filtered_results = [r for r in filtered_results if r.source_type in plan.filters["source_types"]]

        # Apply ranking strategy
        if plan.strategy == "recency":
            filtered_results.sort(key=lambda x: x.timestamp, reverse=True)
        elif plan.strategy == "uncertainty":
            filtered_results.sort(key=lambda x: x.uncertainty)

        # Apply linearization
        if plan.use_linearization:
            graph = self.graph_store.get_graph()
            linearized_nodes = distance_sensitive_linearization(graph, plan.query)
            filtered_results.sort(key=lambda x: linearized_nodes.index(x.id) if x.id in linearized_nodes else float('inf'))

        return filtered_results[:self.config.MAX_RESULTS]

    def _extract_key_concepts(self, results: List[RetrievalResult]) -> List[str]:
        """
        Extract key concepts from retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of extracted concepts
        """
        concepts = set()
        for result in results:
            # Extract concepts using NLP (placeholder implementation)
            # In practice, this would use proper NLP techniques
            words = result.content.lower().split()
            concepts.update(words)
        return list(concepts)
    
    def _calculate_concept_overlap(self, concepts: List[str]) -> float:
        """
        Calculate overlap between concepts.
        
        Args:
            concepts: List of concepts
            
        Returns:
            Overlap score between 0 and 1
        """
        if not concepts:
            return 0.0
        
        # Calculate pairwise overlaps (placeholder implementation)
        # In practice, this would use semantic similarity
        total_overlap = 0
        pairs = 0
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                similarity = len(set(c1) & set(c2)) / len(set(c1) | set(c2))
                total_overlap += similarity
                pairs += 1
        
        return total_overlap / pairs if pairs > 0 else 0.0
    
    def _identify_coverage_gaps(self, concepts: List[str]) -> List[str]:
        """
        Identify gaps in concept coverage.
        
        Args:
            concepts: List of concepts
            
        Returns:
            List of identified gaps
        """
        # Placeholder implementation
        # In practice, this would compare against a knowledge base
        expected_concepts = set(["technology", "science", "business", "health"])
        actual_concepts = set(concepts)
        return list(expected_concepts - actual_concepts)
    
    def _calculate_coverage_score(self,
                                concepts: List[str],
                                coverage_gaps: List[str]) -> float:
        """
        Calculate coverage score based on concepts and gaps.
        
        Args:
            concepts: List of concepts
            coverage_gaps: List of coverage gaps
            
        Returns:
            Coverage score between 0 and 1
        """
        if not concepts:
            return 0.0
        
        # Calculate score based on gaps (placeholder implementation)
        # In practice, this would use more sophisticated metrics
        total_concepts = len(concepts) + len(coverage_gaps)
        return len(concepts) / total_concepts if total_concepts > 0 else 0.0
    
    def _extract_key_terms(self, results: List[RetrievalResult]) -> List[str]:
        """
        Extract key terms from results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of key terms
        """
        terms = set()
        for result in results:
            # Extract terms using NLP (placeholder implementation)
            # In practice, this would use proper NLP techniques
            words = result.content.lower().split()
            # Filter common words (simple stopword removal)
            terms.update([w for w in words if len(w) > 3])
        return list(terms)
    
    def _get_synonyms(self, text: str) -> List[str]:
        """
        Get synonyms for terms in text.
        
        Args:
            text: Input text
            
        Returns:
            List of synonyms
        """
        # Placeholder implementation
        # In practice, this would use WordNet or similar
        synonyms = {
            "large": ["big", "huge", "massive"],
            "small": ["tiny", "little", "miniature"],
            "good": ["great", "excellent", "superb"],
            "bad": ["poor", "terrible", "awful"]
        }
        
        words = text.lower().split()
        result = []
        for word in words:
            if word in synonyms:
                result.extend(synonyms[word])
        return result
    
    def _extract_context_terms(self, results: List[RetrievalResult]) -> List[str]:
        """
        Extract contextually relevant terms from results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of context terms
        """
        context_terms = set()
        for result in results:
            # Extract context using NLP (placeholder implementation)
            # In practice, this would use proper NLP techniques
            sentences = result.content.split('.')
            for sentence in sentences:
                words = sentence.lower().split()
                # Simple context extraction
                if len(words) > 2:
                    context_terms.update(words[1:-1])  # Exclude first and last words
        return list(context_terms)
