from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import re
from collections import Counter
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
            refined_query, refined_vector = self._refine_query(query, feedback)
            new_results = await self.retrieve(refined_query, k, timestamp)
            current_results = self._merge_results(current_results, new_results)

        return current_results

    def _generate_feedback(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Generate feedback for query refinement.

        :param query: The original query.
        :param results: Current retrieval results.
        :return: Feedback data.
        """
        # Simple feedback generation based on most common terms in the current
        # retrieval results.  The returned feedback dictionary can then be used
        # for query or plan refinement.

        feedback: Dict[str, Any] = {}
        if not results:
            return feedback

        # Aggregate text from the top results and extract common words.
        text_blob = " ".join(r.content for r in results)
        tokens = re.findall(r"\b\w+\b", text_blob.lower())
        if tokens:
            common = [w for w, _ in Counter(tokens).most_common(5)]
            feedback["keywords"] = common

        # Provide average uncertainty which may inform later decisions.
        avg_uncertainty = sum(r.uncertainty for r in results) / len(results)
        feedback["avg_uncertainty"] = avg_uncertainty

        return feedback

    def _refine_query(self, original_query: str, feedback: Dict[str, Any]) -> Tuple[str, List[float]]:
        """
        Refine the query based on feedback.

        :param original_query: The original query string.
        :param feedback: Feedback data for refinement.
        :return: A tuple of refined query string and its vector embedding.
        """
        # Incorporate feedback keywords that are not already present in the
        # original query.  This simple strategy gradually expands the query with
        # terms extracted from the current retrieval results.

        refined_query = original_query
        keywords = feedback.get("keywords", [])
        if keywords:
            existing_tokens = set(re.findall(r"\b\w+\b", original_query.lower()))
            additions = [kw for kw in keywords if kw not in existing_tokens]
            if additions:
                refined_query = original_query + " " + " ".join(additions)

        # The embedding is computed asynchronously elsewhere when the refined
        # query is used for retrieval, so we simply return an empty vector here.
        refined_vector: List[float] = []
        return refined_query, refined_vector

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

    async def generate_plan(self, query: str) -> RetrievalPlan:
        """
        Generate a retrieval plan for a given query.

        :param query: The user's query string.
        :return: The generated retrieval plan.
        """
        # Basic plan generation that extracts simple keywords from the query and
        # defaults to a recency based ranking strategy.  This serves as a
        # starting plan which can later be refined using ``refine_plan``.

        keywords = re.findall(r"\b\w+\b", query.lower())
        plan = RetrievalPlan(
            query=query,
            strategy="recency",
            filters={"keywords": keywords},
            use_linearization=False,
            timestamp=datetime.now(),
            version=1,
        )
        return plan

    async def refine_plan(self, query: str, current_plan: RetrievalPlan, results: List[RetrievalResult]) -> RetrievalPlan:
        """
        Refine the retrieval plan based on results.

        :param query: The user's query string.
        :param current_plan: The current retrieval plan.
        :param results: Retrieval results to base refinements on.
        :return: The refined retrieval plan.
        """
        # Refine the existing plan using feedback generated from the latest
        # retrieval results.  Additional keywords are appended and the strategy
        # may switch to uncertainty-based ranking if the average uncertainty is
        # high.

        feedback = self._generate_feedback(query, results)
        keywords = list(current_plan.filters.get("keywords", []))
        for kw in feedback.get("keywords", []):
            if kw not in keywords:
                keywords.append(kw)

        strategy = current_plan.strategy
        if feedback.get("avg_uncertainty", 0) > 0.5:
            strategy = "uncertainty"

        refined_plan = RetrievalPlan(
            query=query,
            strategy=strategy,
            filters={**current_plan.filters, "keywords": keywords},
            use_linearization=current_plan.use_linearization,
            timestamp=datetime.now(),
            version=current_plan.version + 1,
        )
        return refined_plan

    def _apply_upo(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Apply uncertainty-aware probabilistic ordering (UPO) to results.

        :param results: List of retrieval results.
        :param query: The original query.
        :return: Results after applying UPO.
        """
        for result in results:
            # Adjust score based on uncertainty
            certainty = 1 - result.uncertainty
            result.score *= certainty

            # Apply time decay
            time_diff = (datetime.now() - result.timestamp).total_seconds()
            decay_factor = 1 / (1 + time_diff / self.config.TEMPORAL_GRANULARITY.total_seconds())
            result.score *= decay_factor

        # Re-sort results based on the new scores
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _causal_retrieval(self, query: str, initial_results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply causal retrieval adjustments to results.

        :param query: The original query.
        :param initial_results: Initial retrieval results.
        :return: Results after applying causal adjustments.
        """
        causal_scores = {}
        for result in initial_results:
            causal_score = 0
            for edge in self.graph_store.causal_edges.values():
                if edge.source == result.id:
                    causal_score += edge.strength
            causal_scores[result.id] = causal_score

        # Combine original scores with causal scores
        for result in initial_results:
            result.score = 0.7 * result.score + 0.3 * causal_scores.get(result.id, 0)

        return sorted(initial_results, key=lambda x: x.score, reverse=True)
