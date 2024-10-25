"""Hybrid Retriever implementation."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from rag_system.core.config import UnifiedConfig
from rag_system.retrieval.vector_store import VectorStore
from rag_system.retrieval.graph_store import GraphStore
from rag_system.core.structures import RetrievalResult, RetrievalPlan
from rag_system.utils.graph_utils import distance_sensitive_linearization

class HybridRetriever:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.llm = None  # This should be initialized with the appropriate language model
        self.agent = None  # This should be initialized with the appropriate agent
        self.current_results = []  # Store current results for metrics calculation

    async def retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Retrieve documents based on the query using the dual-level retrieval approach.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        return await self.dual_level_retrieve(query, k, timestamp)

    async def dual_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Implement dual-level retrieval as described in LightRAG.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        low_level_results = await self.low_level_retrieve(query, k, timestamp)
        high_level_results = await self.high_level_retrieve(query, k, timestamp)
        
        combined_results = self.merge_results(low_level_results, high_level_results)
        return combined_results[:k]

    async def low_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Implement specific entity and relation retrieval.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        # Use graph_store for entity and relation retrieval
        graph_results = await self.graph_store.retrieve(query, k, timestamp)
        return graph_results

    async def high_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Implement broader topic and theme retrieval.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        # Use vector_store for broader topic and theme retrieval
        query_vector = await self.agent.get_embedding(query)
        vector_results = await self.vector_store.retrieve(query_vector, k, timestamp)
        return vector_results

    def _generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Generate comprehensive feedback for query refinement.

        :param query: The original query string.
        :param results: Current retrieval results.
        :return: Feedback data including relevance scores, coverage analysis, and semantic gaps.
        """
        # Calculate relevance scores
        relevance_scores = self._calculate_relevance_scores(query, results)
        
        # Analyze coverage
        coverage_analysis = self._analyze_coverage(results)
        
        # Identify semantic gaps
        semantic_gaps = self._identify_semantic_gaps(query, results)
        
        # Generate query expansions
        suggested_expansions = self._generate_query_expansions(query, results)
        
        # Combine feedback components
        feedback = {
            "relevance_scores": relevance_scores,
            "coverage_analysis": coverage_analysis,
            "semantic_gaps": semantic_gaps,
            "suggested_expansions": suggested_expansions,
            "confidence_score": self._calculate_confidence_score(relevance_scores, coverage_analysis),
            "diversity_metric": self._calculate_diversity_metric(results),
            "temporal_distribution": self._analyze_temporal_distribution(results)
        }
        
        return feedback

    def _calculate_relevance_scores(self, query: str, results: List[RetrievalResult]) -> List[float]:
        """
        Calculate semantic similarity between query and results.

        :param query: The query string.
        :param results: List of retrieval results.
        :return: List of relevance scores.
        """
        query_embedding = self.agent.get_embedding(query)
        result_embeddings = [self.agent.get_embedding(result.content) for result in results]
        
        # Calculate cosine similarity between query and each result
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
        
        # Calculate concept overlap
        concept_overlap = self._calculate_concept_overlap(concepts)
        
        # Identify coverage gaps
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
        
        # Extract result concepts
        result_concepts = self._extract_key_concepts(results)
        
        # Find concepts in query but not well-represented in results
        semantic_gaps = []
        for concept in query_concepts:
            if concept not in result_concepts:
                semantic_gaps.append(concept)
                
        return semantic_gaps

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
        
        # Generate expansions using different strategies
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
        
        return list(set(expansions))  # Remove duplicates

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
        
        # Calculate weighted average
        avg_relevance = np.mean(relevance_scores)
        coverage_score = coverage_analysis['coverage_score']
        diversity_score = self._calculate_diversity_metric(self.current_results)  # Use stored results
        
        confidence = (
            relevance_weight * avg_relevance +
            coverage_weight * coverage_score +
            diversity_weight * diversity_score
        )
        
        return min(max(confidence, 0.0), 1.0)  # Ensure score is between 0 and 1

    def _calculate_diversity_metric(self, results: List[RetrievalResult]) -> float:
        """
        Calculate diversity metric for results.

        :param results: List of retrieval results.
        :return: Diversity score between 0 and 1.
        """
        if not results:
            return 0.0
            
        # Get embeddings for all results
        embeddings = [self.agent.get_embedding(r.content) for r in results]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Calculate average similarity (lower means more diverse)
        avg_similarity = (np.sum(similarities) - len(results)) / (len(results) * (len(results) - 1))
        
        # Convert to diversity score (higher means more diverse)
        diversity_score = 1.0 - avg_similarity
        
        return diversity_score

    def _analyze_temporal_distribution(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Analyze temporal distribution of results.

        :param results: List of retrieval results.
        :return: Temporal analysis data.
        """
        timestamps = [r.timestamp for r in results if r.timestamp]
        
        if not timestamps:
            return {"temporal_coverage": 0.0, "time_range": None}
            
        # Calculate time range and distribution
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
        sorted_results = sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results

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
        
        # If there are semantic gaps, prioritize addressing them
        if semantic_gaps:
            gap_terms = ' '.join(semantic_gaps[:2])  # Add top 2 missing concepts
            refined_query = f"{original_query} {gap_terms}"
        # Otherwise, use the highest-scored expansion
        elif suggested_expansions:
            refined_query = suggested_expansions[0]
        else:
            refined_query = original_query
            
        return refined_query

    def _merge_results(self, old_results: List[RetrievalResult], new_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Merge old and new retrieval results.

        :param old_results: Previous retrieval results.
        :param new_results: Newly retrieved results.
        :return: Merged list of results.
        """
        combined = old_results + new_results
        deduplicated = self._combine_results(combined)
        return deduplicated

    async def plan_aware_retrieve(self, query: str, k: int, plan: RetrievalPlan, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """
        Retrieve documents using a given retrieval plan.

        :param query: The user's query string.
        :param k: The number of results to retrieve.
        :param plan: A predefined retrieval plan.
        :param timestamp: Optional timestamp to filter results.
        :return: A list of retrieval results.
        """
        initial_results = await self.retrieve(query, k, timestamp)
        filtered_results = self._apply_plan(initial_results, plan)
        return filtered_results

    def _apply_plan(self, results: List[RetrievalResult], plan: RetrievalPlan) -> List[RetrievalResult]:
        """
        Apply the retrieval plan to filter and rank results.

        :param results: List of retrieval results.
        :param plan: The retrieval plan to apply.
        :return: Filtered and ranked list of results.
        """
        filtered_results = results

        # Apply filters from the plan
        if plan.filters.get("keywords"):
            filtered_results = [r for r in filtered_results if any(kw in r.content for kw in plan.filters["keywords"])]

        if plan.filters.get("date_range"):
            start_date, end_date = plan.filters["date_range"]
            filtered_results = [r for r in filtered_results if start_date <= r.timestamp <= end_date]

        if plan.filters.get("source_types"):
            filtered_results = [r for r in filtered_results if r.source_type in plan.filters["source_types"]]

        # Apply custom ranking strategy if specified in the plan
        if plan.strategy == "recency":
            filtered_results.sort(key=lambda x: x.timestamp, reverse=True)
        elif plan.strategy == "uncertainty":
            filtered_results.sort(key=lambda x: x.uncertainty)

        # Apply distance-sensitive linearization if needed
        if plan.use_linearization:
            graph = self.graph_store.get_graph()
            linearized_nodes = distance_sensitive_linearization(graph, plan.query)
            filtered_results.sort(key=lambda x: linearized_nodes.index(x.id) if x.id in linearized_nodes else float('inf'))

        return filtered_results[:self.config.MAX_RESULTS]
