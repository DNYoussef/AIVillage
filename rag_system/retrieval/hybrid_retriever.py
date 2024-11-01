"""Enhanced hybrid retrieval system with dual-level and plan-aware capabilities."""

from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import json
from ..core.base_component import BaseComponent
from ..core.config import UnifiedConfig, RAGConfig
from ..retrieval.vector_store import VectorStore
from ..retrieval.graph_store import GraphStore
from ..core.structures import RetrievalResult, RetrievalPlan
from ..utils.graph_utils import distance_sensitive_linearization
from ..utils.error_handling import log_and_handle_errors, ErrorContext, RAGSystemError

logger = logging.getLogger(__name__)

class MockAgent:
    """Mock agent for testing."""
    def __init__(self):
        self.embeddings = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize mock agent."""
        self.initialized = True
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        # Return mock embedding vector
        return [0.1] * 768

class HybridRetriever(BaseComponent):
    """
    Enhanced hybrid retrieval system combining vector and graph-based approaches.
    Implements dual-level retrieval, active retrieval with feedback, and plan-aware retrieval.
    """
    
    def __init__(self, config: Union[UnifiedConfig, RAGConfig]):
        """
        Initialize hybrid retriever.
        
        Args:
            config: Configuration instance
        """
        super().__init__()  # Call parent's init
        self.config = config
        vector_dimension = (
            config.vector_dimension if isinstance(config, RAGConfig)
            else config.get('vector_dimension', 768)
        )
        self.vector_store = VectorStore(config, vector_dimension)
        self.graph_store = GraphStore(config)
        self.llm = None  # Will be initialized with language model
        self.agent = None  # Will be initialized with agent
        self.current_results = []  # Store current results for metrics
        
        # Add component-specific stats
        self.stats.update({
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_feedbacks": 0,
            "successful_feedbacks": 0,
            "failed_feedbacks": 0,
            "avg_retrieval_time": 0.0,
            "avg_feedback_time": 0.0,
            "avg_result_count": 0.0,
            "last_retrieval": None,
            "last_feedback": None,
            "memory_usage": {
                "vector_store": 0,
                "graph_store": 0,
                "current_results": 0
            }
        })
        
        logger.info("Initialized HybridRetriever")
    
    @log_and_handle_errors()
    async def initialize(self) -> None:
        """Initialize retriever components."""
        try:
            await self._pre_initialize()
            
            logger.info("Initializing HybridRetriever...")
            
            await self.vector_store.initialize()
            await self.graph_store.initialize()
            
            # Initialize mock agent if none provided
            if self.agent is None:
                self.agent = MockAgent()
            if not getattr(self.agent, 'initialized', False):
                await self.agent.initialize()
            
            await self._post_initialize()
            logger.info("Successfully initialized HybridRetriever")
            
        except Exception as e:
            logger.error(f"Error initializing HybridRetriever: {str(e)}")
            self.initialized = False
            raise RAGSystemError(f"Failed to initialize hybrid retriever: {str(e)}") from e
    
    @log_and_handle_errors()
    async def generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate comprehensive feedback for query refinement."""
        return await self._safe_operation("generate_feedback", self._do_generate_feedback(query, results))
    
    async def _do_generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Internal feedback generation implementation."""
        try:
            start_time = datetime.now()
            self.stats["total_feedbacks"] += 1
            
            if not results:
                feedback = {
                    "relevance_scores": [],
                    "coverage_analysis": self._analyze_coverage([]),
                    "semantic_gaps": [],
                    "suggested_expansions": [],
                    "confidence_score": 0.0,
                    "diversity_metric": 0.0,
                    "temporal_distribution": {"temporal_coverage": 0.0, "time_range": None}
                }
            else:
                relevance_scores = await self._calculate_relevance_scores(query, results)
                coverage_analysis = self._analyze_coverage(results)
                semantic_gaps = self._identify_semantic_gaps(query, results)
                suggested_expansions = self._generate_query_expansions(query, results)
                
                feedback = {
                    "relevance_scores": relevance_scores,
                    "coverage_analysis": coverage_analysis,
                    "semantic_gaps": semantic_gaps,
                    "suggested_expansions": suggested_expansions,
                    "confidence_score": self._calculate_confidence_score(relevance_scores, coverage_analysis),
                    "diversity_metric": await self._calculate_diversity_metric(results),
                    "temporal_distribution": self._analyze_temporal_distribution(results)
                }
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_feedbacks"] += 1
            self.stats["avg_feedback_time"] = (
                (self.stats["avg_feedback_time"] * (self.stats["successful_feedbacks"] - 1) + processing_time) /
                self.stats["successful_feedbacks"]
            )
            self.stats["last_feedback"] = datetime.now().isoformat()
            
            # Update memory usage stats
            self.stats["memory_usage"].update({
                "vector_store": await self._get_vector_store_size(),
                "graph_store": await self._get_graph_store_size(),
                "current_results": len(self.current_results)
            })
            
            return feedback
            
        except Exception as e:
            self.stats["failed_feedbacks"] += 1
            raise RAGSystemError(f"Error generating feedback: {str(e)}") from e

    async def _get_vector_store_size(self) -> int:
        """Get vector store memory usage."""
        try:
            status = await self.vector_store.get_status()
            return status.get("memory_usage", {}).get("total", 0)
        except Exception:
            return 0

    async def _get_graph_store_size(self) -> int:
        """Get graph store memory usage."""
        try:
            status = await self.graph_store.get_status()
            return status.get("memory_usage", {}).get("total", 0)
        except Exception:
            return 0

    async def shutdown(self) -> None:
        """Shutdown retriever components."""
        try:
            await self._pre_shutdown()
            
            logger.info("Shutting down HybridRetriever...")
            
            await self.vector_store.shutdown()
            await self.graph_store.shutdown()
            
            self.current_results = []
            
            await self._post_shutdown()
            logger.info("Successfully shut down HybridRetriever")
            
        except Exception as e:
            logger.error(f"Error shutting down HybridRetriever: {str(e)}")
            raise
    
    @log_and_handle_errors()
    async def get_status(self) -> Dict[str, Any]:
        """Get component status."""
        base_status = await self.get_base_status()
        
        component_status = {
            "vector_store": await self.vector_store.get_status(),
            "graph_store": await self.graph_store.get_status(),
            "agent": "initialized" if getattr(self.agent, 'initialized', False) else "not initialized",
            "current_results_count": len(self.current_results)
        }
        
        return {
            **base_status,
            **component_status
        }
    
    @log_and_handle_errors()
    async def update_config(self, config: Union[UnifiedConfig, RAGConfig]) -> None:
        """Update component configuration."""
        try:
            logger.info("Updating HybridRetriever configuration...")
            
            self.config = config
            await self.vector_store.update_config(config)
            await self.graph_store.update_config(config)
            
            logger.info("Successfully updated configuration")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {str(e)}")
            raise RAGSystemError(f"Failed to update configuration: {str(e)}") from e

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
        return await self._safe_operation("retrieve", self._do_retrieve(query, k, timestamp))
    
    async def _do_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Internal retrieve implementation."""
        try:
            if not self.initialized:
                await self.initialize()
            
            if not self.agent:
                raise RAGSystemError("Agent not initialized in HybridRetriever")
            
            start_time = datetime.now()
            self.stats["total_retrievals"] += 1
            
            results = await self.dual_level_retrieve(query, k, timestamp)
            self.current_results = results
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats["successful_retrievals"] += 1
            self.stats["avg_retrieval_time"] = (
                (self.stats["avg_retrieval_time"] * (self.stats["successful_retrievals"] - 1) + processing_time) /
                self.stats["successful_retrievals"]
            )
            self.stats["avg_result_count"] = (
                (self.stats["avg_result_count"] * (self.stats["successful_retrievals"] - 1) + len(results)) /
                self.stats["successful_retrievals"]
            )
            self.stats["last_retrieval"] = datetime.now().isoformat()
            
            return results
            
        except Exception as e:
            self.stats["failed_retrievals"] += 1
            raise RAGSystemError(f"Error in retrieval: {str(e)}") from e

    async def dual_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement dual-level retrieval as described in LightRAG."""
        try:
            low_level_results = await self.low_level_retrieve(query, k, timestamp)
            high_level_results = await self.high_level_retrieve(query, k, timestamp)
            
            # Ensure we have lists even if empty
            low_level_results = low_level_results or []
            high_level_results = high_level_results or []
            
            combined_results = self.merge_results(low_level_results, high_level_results)
            return combined_results[:k]
        except Exception as e:
            raise RAGSystemError(f"Error in dual-level retrieval: {str(e)}") from e

    async def low_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement specific entity and relation retrieval."""
        try:
            results = await self.graph_store.retrieve(query, k, timestamp)
            return results or []  # Return empty list if no results
        except Exception as e:
            raise RAGSystemError(f"Error in low-level retrieval: {str(e)}") from e

    async def high_level_retrieve(self, query: str, k: int, timestamp: Optional[datetime] = None) -> List[RetrievalResult]:
        """Implement broader topic and theme retrieval."""
        try:
            query_vector = await self.agent.get_embedding(query)
            results = await self.vector_store.retrieve(query_vector, k, timestamp)
            return results or []  # Return empty list if no results
        except Exception as e:
            raise RAGSystemError(f"Error in high-level retrieval: {str(e)}") from e

    async def _generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate comprehensive feedback for query refinement."""
        try:
            if not results:
                return {
                    "relevance_scores": [],
                    "coverage_analysis": self._analyze_coverage([]),
                    "semantic_gaps": [],
                    "suggested_expansions": [],
                    "confidence_score": 0.0,
                    "diversity_metric": 0.0,
                    "temporal_distribution": {"temporal_coverage": 0.0, "time_range": None}
                }
                
            relevance_scores = await self._calculate_relevance_scores(query, results)
            coverage_analysis = self._analyze_coverage(results)
            semantic_gaps = self._identify_semantic_gaps(query, results)
            suggested_expansions = self._generate_query_expansions(query, results)
            
            return {
                "relevance_scores": relevance_scores,
                "coverage_analysis": coverage_analysis,
                "semantic_gaps": semantic_gaps,
                "suggested_expansions": suggested_expansions,
                "confidence_score": self._calculate_confidence_score(relevance_scores, coverage_analysis),
                "diversity_metric": await self._calculate_diversity_metric(results),
                "temporal_distribution": self._analyze_temporal_distribution(results)
            }
        except Exception as e:
            raise RAGSystemError(f"Error generating feedback: {str(e)}") from e

    def _identify_semantic_gaps(self, query: str, results: List[RetrievalResult]) -> List[str]:
        """
        Identify semantic concepts from the query that are not well-represented in results.

        :param query: The query string.
        :param results: List of retrieval results.
        :return: List of identified semantic gaps.
        """
        try:
            # Create a RetrievalResult for the query with current timestamp
            current_time = datetime.now()
            query_result = RetrievalResult(
                id="query",
                content=query,
                score=1.0,
                uncertainty=0.0,  # Low uncertainty for query itself
                timestamp=current_time,
                version=1
            )
            
            # Extract concepts
            query_concepts = self._extract_key_concepts([query_result])
            result_concepts = self._extract_key_concepts(results)
            
            # Find gaps
            return [concept for concept in query_concepts if concept not in result_concepts]
        except Exception as e:
            raise RAGSystemError(f"Error identifying semantic gaps: {str(e)}") from e

    def _generate_query_expansions(self, query: str, results: List[RetrievalResult]) -> List[str]:
        """Generate suggested query expansions."""
        try:
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
        except Exception as e:
            raise RAGSystemError(f"Error generating query expansions: {str(e)}") from e

    def _calculate_confidence_score(self, relevance_scores: List[float], coverage_analysis: Dict[str, Any]) -> float:
        """Calculate confidence score."""
        try:
            if not relevance_scores:
                return 0.0
                
            # Weight factors for different components
            relevance_weight = 0.4
            coverage_weight = 0.3
            diversity_weight = 0.3
            
            avg_relevance = np.mean(relevance_scores)
            coverage_score = coverage_analysis.get('coverage_score', 0.0)
            diversity_score = 0.5  # Default middle value
            
            confidence = (
                relevance_weight * avg_relevance +
                coverage_weight * coverage_score +
                diversity_weight * diversity_score
            )
            
            return min(max(confidence, 0.0), 1.0)
        except Exception as e:
            raise RAGSystemError(f"Error calculating confidence score: {str(e)}") from e

    async def _calculate_relevance_scores(self, query: str, results: List[RetrievalResult]) -> List[float]:
        """Calculate semantic similarity between query and results."""
        try:
            if not results:
                return []
                
            query_embedding = await self.agent.get_embedding(query)
            result_embeddings = [await self.agent.get_embedding(result.content) for result in results]
            similarities = cosine_similarity([query_embedding], result_embeddings)[0]
            
            # Normalize scores
            min_score = np.min(similarities)
            max_score = np.max(similarities)
            if max_score > min_score:
                normalized_scores = (similarities - min_score) / (max_score - min_score)
            else:
                normalized_scores = similarities
                
            return normalized_scores.tolist()
        except Exception as e:
            raise RAGSystemError(f"Error calculating relevance scores: {str(e)}") from e

    def merge_results(self, low_level_results: List[RetrievalResult], high_level_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Merge and deduplicate results."""
        try:
            combined_results = (low_level_results or []) + (high_level_results or [])
            return self._combine_results(combined_results)
        except Exception as e:
            raise RAGSystemError(f"Error merging results: {str(e)}") from e

    def _combine_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Combine and deduplicate results."""
        try:
            if not results:
                return []
                
            unique_results = {}
            for result in results:
                if result.id not in unique_results or result.score > unique_results[result.id].score:
                    unique_results[result.id] = result
            return sorted(unique_results.values(), key=lambda x: x.score, reverse=True)
        except Exception as e:
            raise RAGSystemError(f"Error combining results: {str(e)}") from e

    def _analyze_coverage(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Analyze coverage of results."""
        try:
            if not results:
                return {
                    "concept_coverage": 0,
                    "concept_overlap": 0.0,
                    "coverage_gaps": [],
                    "coverage_score": 0.0
                }
                
            concepts = self._extract_key_concepts(results)
            concept_overlap = self._calculate_concept_overlap(concepts)
            coverage_gaps = self._identify_coverage_gaps(concepts)
            
            return {
                "concept_coverage": len(concepts),
                "concept_overlap": concept_overlap,
                "coverage_gaps": coverage_gaps,
                "coverage_score": self._calculate_coverage_score(concepts, coverage_gaps)
            }
        except Exception as e:
            raise RAGSystemError(f"Error analyzing coverage: {str(e)}") from e

    def _extract_key_concepts(self, results: List[RetrievalResult]) -> List[str]:
        """Extract key concepts from results."""
        try:
            concepts = set()
            for result in results:
                if result.content:
                    words = result.content.lower().split()
                    concepts.update(words)
            return list(concepts)
        except Exception as e:
            raise RAGSystemError(f"Error extracting key concepts: {str(e)}") from e

    def _calculate_concept_overlap(self, concepts: List[str]) -> float:
        """Calculate concept overlap."""
        try:
            if not concepts or len(concepts) < 2:
                return 0.0
                
            total_overlap = 0
            pairs = 0
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    if c1 and c2:  # Ensure non-empty strings
                        similarity = len(set(c1) & set(c2)) / len(set(c1) | set(c2))
                        total_overlap += similarity
                        pairs += 1
            
            return total_overlap / pairs if pairs > 0 else 0.0
        except Exception as e:
            raise RAGSystemError(f"Error calculating concept overlap: {str(e)}") from e

    def _identify_coverage_gaps(self, concepts: List[str]) -> List[str]:
        """Identify coverage gaps."""
        try:
            expected_concepts = set(["technology", "science", "business", "health"])
            actual_concepts = set(concepts) if concepts else set()
            return list(expected_concepts - actual_concepts)
        except Exception as e:
            raise RAGSystemError(f"Error identifying coverage gaps: {str(e)}") from e

    def _calculate_coverage_score(self, concepts: List[str], coverage_gaps: List[str]) -> float:
        """Calculate coverage score."""
        try:
            if not concepts and not coverage_gaps:
                return 0.0
                
            total_concepts = len(concepts) + len(coverage_gaps)
            return len(concepts) / total_concepts if total_concepts > 0 else 0.0
        except Exception as e:
            raise RAGSystemError(f"Error calculating coverage score: {str(e)}") from e

    async def _calculate_diversity_metric(self, results: List[RetrievalResult]) -> float:
        """Calculate diversity metric."""
        try:
            if not results:
                return 0.0
                
            embeddings = [await self.agent.get_embedding(r.content) for r in results if r.content]
            if not embeddings:
                return 0.0
                
            similarities = cosine_similarity(embeddings)
            if len(similarities) > 1:
                avg_similarity = (np.sum(similarities) - len(similarities)) / (len(similarities) * (len(similarities) - 1))
                return 1.0 - avg_similarity
            return 0.0
        except Exception as e:
            raise RAGSystemError(f"Error calculating diversity metric: {str(e)}") from e

    def _analyze_temporal_distribution(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Analyze temporal distribution."""
        try:
            if not results:
                return {"temporal_coverage": 0.0, "time_range": None}
                
            timestamps = [r.timestamp for r in results if r.timestamp]
            if not timestamps:
                return {"temporal_coverage": 0.0, "time_range": None}
                
            # Convert timestamps to float values (seconds since epoch)
            timestamp_values = [t.timestamp() for t in timestamps]
            
            # Calculate time range
            time_range = max(timestamps) - min(timestamps)
            
            # Create histogram using float values
            if len(timestamp_values) > 1:
                hist, bin_edges = np.histogram(timestamp_values, bins=min(10, len(timestamp_values)))
                # Convert bin edges back to datetime for display
                bin_edges = [datetime.fromtimestamp(edge) for edge in bin_edges]
            else:
                hist = np.array([1])
                bin_edges = [min(timestamps), min(timestamps) + timedelta(seconds=1)]
            
            return {
                "temporal_coverage": len(timestamps) / len(results),
                "time_range": str(time_range),
                "distribution": {
                    "counts": hist.tolist(),
                    "bin_edges": [edge.isoformat() for edge in bin_edges]
                }
            }
        except Exception as e:
            raise RAGSystemError(f"Error analyzing temporal distribution: {str(e)}") from e

    def _extract_key_terms(self, results: List[RetrievalResult]) -> List[str]:
        """Extract key terms from results."""
        try:
            terms = set()
            for result in results:
                if result.content:
                    words = result.content.lower().split()
                    # Filter common words (simple stopword removal)
                    terms.update([w for w in words if len(w) > 3])
            return list(terms)
        except Exception as e:
            raise RAGSystemError(f"Error extracting key terms: {str(e)}") from e

    def _get_synonyms(self, text: str) -> List[str]:
        """Get synonyms for terms in text."""
        try:
            # Simple synonym dictionary (in practice, use WordNet or similar)
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
        except Exception as e:
            raise RAGSystemError(f"Error getting synonyms: {str(e)}") from e

    def _extract_context_terms(self, results: List[RetrievalResult]) -> List[str]:
        """Extract contextually relevant terms from results."""
        try:
            context_terms = set()
            for result in results:
                if result.content:
                    sentences = result.content.split('.')
                    for sentence in sentences:
                        words = sentence.lower().split()
                        # Simple context extraction
                        if len(words) > 2:
                            context_terms.update(words[1:-1])  # Exclude first and last words
            return list(context_terms)
        except Exception as e:
            raise RAGSystemError(f"Error extracting context terms: {str(e)}") from e


