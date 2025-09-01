#!/usr/bin/env python3
"""HypeRAG Personalisation Benchmark.

Compares retrieval performance across different personalization approaches:
A) Base PPR (Personalized PageRank)
B) PPR + Rel-GAT α (Relational Graph Attention with alpha rescoring)
C) PPR + α + ICL (In-Context Learning with single triple)

Evaluation Metrics:
- MAP (Mean Average Precision)
- NDCG@10 (Normalized Discounted Cumulative Gain at 10)
- Recall@20 (Recall at 20)
- Token Cost Delta (computational cost comparison)

Datasets:
- MovieLens: Movie recommendation with user preferences
- Domain Document Clicks: Academic paper relevance based on click patterns
"""

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import statistics

# Import HypeRAG components
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger(__name__)


@dataclass
class PersonalizationMetrics:
    """Personalization benchmark results."""

    approach_name: str
    map_score: float
    ndcg_at_10: float
    recall_at_20: float
    token_cost_delta: float
    total_queries: int
    avg_retrieval_time: float
    dataset_name: str
    timestamp: str


@dataclass
class UserQuery:
    """User query with ground truth relevance."""

    query_id: str
    user_id: str
    query_text: str
    relevant_items: list[str]  # Ground truth relevant item IDs
    relevance_scores: dict[str, float]  # Item ID -> relevance score
    domain: str


@dataclass
class PersonalizationContext:
    """User personalization context."""

    user_id: str
    preferences: dict[str, float]  # Feature -> preference weight
    interaction_history: list[str]  # Previously interacted item IDs
    domain_expertise: float  # 0.0 to 1.0
    temporal_context: dict[str, Any]  # Time-based context


class PersonalizationDatasetGenerator:
    """Generates personalization evaluation datasets."""

    def __init__(self) -> None:
        self.movielens_queries = self._create_movielens_queries()
        self.doc_click_queries = self._create_doc_click_queries()

    def _create_movielens_queries(self) -> list[UserQuery]:
        """Create MovieLens-style queries."""
        queries = []

        # User 1: Action movie enthusiast
        queries.append(
            UserQuery(
                query_id="ml_001",
                user_id="user_001",
                query_text="exciting action movies with great special effects",
                relevant_items=["movie_001", "movie_005", "movie_012", "movie_018"],
                relevance_scores={
                    "movie_001": 0.95,  # Top Gun: Maverick
                    "movie_005": 0.88,  # John Wick 4
                    "movie_012": 0.82,  # Fast X
                    "movie_018": 0.79,  # Mission Impossible
                    "movie_023": 0.65,  # Marvel movie
                },
                domain="movies",
            )
        )

        # User 2: Drama lover
        queries.append(
            UserQuery(
                query_id="ml_002",
                user_id="user_002",
                query_text="emotional character-driven stories",
                relevant_items=["movie_003", "movie_008", "movie_015", "movie_021"],
                relevance_scores={
                    "movie_003": 0.92,  # The Pianist
                    "movie_008": 0.89,  # Manchester by the Sea
                    "movie_015": 0.85,  # Moonlight
                    "movie_021": 0.78,  # Lady Bird
                    "movie_024": 0.68,  # A Star is Born
                },
                domain="movies",
            )
        )

        # User 3: Sci-fi enthusiast
        queries.append(
            UserQuery(
                query_id="ml_003",
                user_id="user_003",
                query_text="mind-bending science fiction films",
                relevant_items=["movie_002", "movie_007", "movie_011", "movie_019"],
                relevance_scores={
                    "movie_002": 0.97,  # Inception
                    "movie_007": 0.93,  # Interstellar
                    "movie_011": 0.87,  # Arrival
                    "movie_019": 0.84,  # Ex Machina
                    "movie_025": 0.73,  # Blade Runner 2049
                },
                domain="movies",
            )
        )

        # User 4: Comedy fan
        queries.append(
            UserQuery(
                query_id="ml_004",
                user_id="user_004",
                query_text="funny movies that make you laugh out loud",
                relevant_items=["movie_004", "movie_009", "movie_014", "movie_020"],
                relevance_scores={
                    "movie_004": 0.91,  # Superbad
                    "movie_009": 0.86,  # The Grand Budapest Hotel
                    "movie_014": 0.83,  # Knives Out
                    "movie_020": 0.77,  # Game Night
                    "movie_026": 0.71,  # Thor: Ragnarok
                },
                domain="movies",
            )
        )

        # User 5: Horror fan
        queries.append(
            UserQuery(
                query_id="ml_005",
                user_id="user_005",
                query_text="scary psychological thrillers",
                relevant_items=["movie_006", "movie_010", "movie_016", "movie_022"],
                relevance_scores={
                    "movie_006": 0.94,  # Hereditary
                    "movie_010": 0.90,  # Get Out
                    "movie_016": 0.85,  # The Babadook
                    "movie_022": 0.81,  # Midsommar
                    "movie_027": 0.74,  # A Quiet Place
                },
                domain="movies",
            )
        )

        return queries

    def _create_doc_click_queries(self) -> list[UserQuery]:
        """Create document click-based queries."""
        queries = []

        # User 1: Machine Learning researcher
        queries.append(
            UserQuery(
                query_id="dc_001",
                user_id="researcher_001",
                query_text="transformer attention mechanisms deep learning",
                relevant_items=["paper_001", "paper_008", "paper_015", "paper_023"],
                relevance_scores={
                    "paper_001": 0.96,  # Attention Is All You Need
                    "paper_008": 0.91,  # BERT
                    "paper_015": 0.87,  # GPT-3
                    "paper_023": 0.83,  # Vision Transformer
                    "paper_031": 0.76,  # RoBERTa
                },
                domain="computer_science",
            )
        )

        # User 2: Computer Vision specialist
        queries.append(
            UserQuery(
                query_id="dc_002",
                user_id="researcher_002",
                query_text="object detection convolutional neural networks",
                relevant_items=["paper_002", "paper_009", "paper_016", "paper_024"],
                relevance_scores={
                    "paper_002": 0.94,  # YOLO
                    "paper_009": 0.89,  # R-CNN
                    "paper_016": 0.84,  # RetinaNet
                    "paper_024": 0.80,  # EfficientDet
                    "paper_032": 0.72,  # Mask R-CNN
                },
                domain="computer_science",
            )
        )

        # User 3: NLP researcher
        queries.append(
            UserQuery(
                query_id="dc_003",
                user_id="researcher_003",
                query_text="natural language processing sentiment analysis",
                relevant_items=["paper_003", "paper_010", "paper_017", "paper_025"],
                relevance_scores={
                    "paper_003": 0.93,  # Word2Vec
                    "paper_010": 0.88,  # GloVe
                    "paper_017": 0.85,  # ULMFiT
                    "paper_025": 0.79,  # FastText
                    "paper_033": 0.74,  # ELMo
                },
                domain="computer_science",
            )
        )

        # User 4: Reinforcement Learning expert
        queries.append(
            UserQuery(
                query_id="dc_004",
                user_id="researcher_004",
                query_text="reinforcement learning policy gradient methods",
                relevant_items=["paper_004", "paper_011", "paper_018", "paper_026"],
                relevance_scores={
                    "paper_004": 0.95,  # DQN
                    "paper_011": 0.90,  # AlphaGo
                    "paper_018": 0.86,  # PPO
                    "paper_026": 0.82,  # A3C
                    "paper_034": 0.77,  # DDPG
                },
                domain="computer_science",
            )
        )

        # User 5: Theoretical CS researcher
        queries.append(
            UserQuery(
                query_id="dc_005",
                user_id="researcher_005",
                query_text="algorithmic complexity optimization theory",
                relevant_items=["paper_005", "paper_012", "paper_019", "paper_027"],
                relevance_scores={
                    "paper_005": 0.92,  # Complexity Theory
                    "paper_012": 0.87,  # Approximation Algorithms
                    "paper_019": 0.83,  # Linear Programming
                    "paper_027": 0.78,  # Graph Algorithms
                    "paper_035": 0.71,  # Randomized Algorithms
                },
                domain="computer_science",
            )
        )

        return queries

    def get_personalization_contexts(self) -> dict[str, PersonalizationContext]:
        """Get user personalization contexts."""
        contexts = {}

        # MovieLens user contexts
        contexts["user_001"] = PersonalizationContext(
            user_id="user_001",
            preferences={
                "action": 0.9,
                "adventure": 0.8,
                "sci_fi": 0.6,
                "drama": 0.3,
                "comedy": 0.5,
            },
            interaction_history=["movie_001", "movie_028", "movie_035", "movie_042"],
            domain_expertise=0.7,
            temporal_context={"time_of_day": "evening", "weekend": True},
        )

        contexts["user_002"] = PersonalizationContext(
            user_id="user_002",
            preferences={
                "drama": 0.95,
                "romance": 0.8,
                "independent": 0.85,
                "action": 0.2,
                "comedy": 0.4,
            },
            interaction_history=["movie_003", "movie_029", "movie_036", "movie_043"],
            domain_expertise=0.8,
            temporal_context={"time_of_day": "afternoon", "weekend": False},
        )

        contexts["user_003"] = PersonalizationContext(
            user_id="user_003",
            preferences={
                "sci_fi": 0.98,
                "thriller": 0.75,
                "mystery": 0.7,
                "action": 0.6,
                "horror": 0.3,
            },
            interaction_history=["movie_002", "movie_030", "movie_037", "movie_044"],
            domain_expertise=0.9,
            temporal_context={"time_of_day": "night", "weekend": True},
        )

        # Research user contexts
        contexts["researcher_001"] = PersonalizationContext(
            user_id="researcher_001",
            preferences={
                "deep_learning": 0.95,
                "attention_mechanisms": 0.9,
                "nlp": 0.8,
                "computer_vision": 0.6,
                "theory": 0.4,
            },
            interaction_history=["paper_001", "paper_045", "paper_052", "paper_061"],
            domain_expertise=0.95,
            temporal_context={
                "research_phase": "implementation",
                "deadline_pressure": 0.3,
            },
        )

        contexts["researcher_002"] = PersonalizationContext(
            user_id="researcher_002",
            preferences={
                "computer_vision": 0.92,
                "object_detection": 0.88,
                "image_segmentation": 0.83,
                "deep_learning": 0.75,
                "robotics": 0.6,
            },
            interaction_history=["paper_002", "paper_046", "paper_053", "paper_062"],
            domain_expertise=0.88,
            temporal_context={
                "research_phase": "literature_review",
                "deadline_pressure": 0.7,
            },
        )

        return contexts


class PersonalizationApproach:
    """Base class for personalization approaches."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.token_costs = []
        self.retrieval_times = []

    async def retrieve(
        self, query: UserQuery, context: PersonalizationContext, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Retrieve personalized results for a query."""
        # Base implementation using simple semantic similarity
        import time

        start_time = time.time()

        # Mock retrieval based on user preferences and query
        results = []

        # Simulate personalized scoring based on user preferences
        for i in range(min(top_k, 100)):  # Mock document collection
            doc_id = f"doc_{i}"

            # Base relevance score (0.1 to 0.9)
            base_score = 0.9 - (i * 0.8 / 100)

            # Apply personalization boost based on user preferences
            personalization_boost = 0.0
            if context.user_preferences:
                # Boost for matching preferences
                if "technical" in context.user_preferences and i % 3 == 0:
                    personalization_boost += 0.1
                if "recent" in context.user_preferences and i < 20:
                    personalization_boost += 0.05

            # Apply history boost
            if context.interaction_history:
                # Boost for documents similar to previously interacted content
                if i in [h % 100 for h in range(len(context.interaction_history))]:
                    personalization_boost += 0.15

            final_score = min(1.0, base_score + personalization_boost)
            results.append((doc_id, final_score))

        # Sort by score and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        # Record retrieval time
        retrieval_time = time.time() - start_time
        self.retrieval_times.append(retrieval_time)

        return results

    def calculate_token_cost(self, query_length: int, context_size: int, retrieved_items: int) -> int:
        """Calculate token cost for the approach."""
        # Base token cost calculation
        # Cost includes: query processing + context encoding + result processing

        # Query processing cost (base cost per token)
        query_cost = query_length * 2  # 2 tokens per input token for processing

        # Context encoding cost (depends on personalization complexity)
        # More complex personalization requires more context processing
        context_cost = context_size * 1.5  # 1.5 tokens per context token

        # Result processing cost (ranking, reranking, personalization)
        result_cost = retrieved_items * 10  # 10 tokens per retrieved item for scoring

        # Additional overhead for personalization algorithms
        personalization_overhead = 50  # Base overhead for personalization processing

        total_cost = int(query_cost + context_cost + result_cost + personalization_overhead)

        # Record cost for metrics
        self.token_costs.append(total_cost)

        return total_cost


class BasePPRApproach(PersonalizationApproach):
    """Base Personalized PageRank approach."""

    def __init__(self) -> None:
        super().__init__("Base PPR")
        self.ppr_retriever = None  # Would be initialized with actual retriever

    async def retrieve(
        self, query: UserQuery, context: PersonalizationContext, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Retrieve using base PPR only."""
        start_time = datetime.now(UTC)

        # Mock PPR retrieval - in reality would use actual PPR algorithm
        all_items = list(query.relevance_scores.keys())

        # Simulate PPR scoring based on query text similarity
        scored_items = []
        for item_id in all_items:
            # Base PPR score (mock)
            base_score = np.random.uniform(0.3, 0.8)
            scored_items.append((item_id, base_score))

        # Add some random items to simulate full retrieval
        for i in range(15):  # Add 15 more items
            item_id = f"item_{i:03d}"
            score = np.random.uniform(0.1, 0.6)
            scored_items.append((item_id, score))

        # Sort by score and take top-k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        results = scored_items[:top_k]

        # Track performance
        retrieval_time = (datetime.now(UTC) - start_time).total_seconds()
        self.retrieval_times.append(retrieval_time)

        # Calculate token cost
        token_cost = self.calculate_token_cost(len(query.query_text.split()), 0, len(results))
        self.token_costs.append(token_cost)

        return results

    def calculate_token_cost(self, query_length: int, context_size: int, retrieved_items: int) -> int:
        """Base PPR has minimal token cost."""
        return query_length + retrieved_items * 2  # Query + item representations


class AlphaRescoredPPRApproach(PersonalizationApproach):
    """PPR + Rel-GAT α rescoring approach."""

    def __init__(self) -> None:
        super().__init__("PPR + α Rescoring")
        self.alpha_weight = 0.3  # Weight for α rescoring

    async def retrieve(
        self, query: UserQuery, context: PersonalizationContext, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Retrieve using PPR + α rescoring."""
        start_time = datetime.now(UTC)

        # Step 1: Get base PPR results
        base_ppr_approach = BasePPRApproach()
        base_results = await base_ppr_approach.retrieve(query, context, top_k * 2)  # Get more for rescoring

        # Step 2: Apply α rescoring based on user preferences
        rescored_items = []
        for item_id, base_score in base_results:
            # Calculate α rescoring factor based on user preferences
            alpha_factor = self._calculate_alpha_factor(item_id, context)

            # Combine base PPR score with α rescoring
            final_score = base_score * (1 - self.alpha_weight) + alpha_factor * self.alpha_weight
            rescored_items.append((item_id, final_score))

        # Sort by final score and take top-k
        rescored_items.sort(key=lambda x: x[1], reverse=True)
        results = rescored_items[:top_k]

        # Track performance
        retrieval_time = (datetime.now(UTC) - start_time).total_seconds()
        self.retrieval_times.append(retrieval_time)

        # Calculate token cost (higher due to α rescoring)
        token_cost = self.calculate_token_cost(len(query.query_text.split()), len(context.preferences), len(results))
        self.token_costs.append(token_cost)

        return results

    def _calculate_alpha_factor(self, item_id: str, context: PersonalizationContext) -> float:
        """Calculate α rescoring factor based on user preferences."""
        # Mock α calculation based on user preferences
        alpha_score = 0.5

        # Boost score based on user preferences (mock implementation)
        if "movie" in item_id:
            # Movie domain logic
            if "action" in context.preferences and np.random.random() > 0.5:
                alpha_score += context.preferences["action"] * 0.3
            if "drama" in context.preferences and np.random.random() > 0.3:
                alpha_score += context.preferences["drama"] * 0.4

        elif "paper" in item_id:
            # Academic paper domain logic
            if "deep_learning" in context.preferences and np.random.random() > 0.4:
                alpha_score += context.preferences["deep_learning"] * 0.3
            if "computer_vision" in context.preferences and np.random.random() > 0.6:
                alpha_score += context.preferences["computer_vision"] * 0.2

        # Factor in interaction history
        if item_id in context.interaction_history:
            alpha_score += 0.2

        # Factor in domain expertise
        alpha_score += context.domain_expertise * 0.1

        return min(max(alpha_score, 0.0), 1.0)

    def calculate_token_cost(self, query_length: int, context_size: int, retrieved_items: int) -> int:
        """α rescoring adds computational overhead."""
        base_cost = query_length + retrieved_items * 2
        alpha_cost = context_size * 5 + retrieved_items * 3  # Preference processing + rescoring
        return base_cost + alpha_cost


class ICLEnhancedApproach(PersonalizationApproach):
    """PPR + α + ICL (In-Context Learning) approach."""

    def __init__(self) -> None:
        super().__init__("PPR + α + ICL")
        self.alpha_weight = 0.3
        self.icl_weight = 0.2

    async def retrieve(
        self, query: UserQuery, context: PersonalizationContext, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Retrieve using PPR + α + ICL."""
        start_time = datetime.now(UTC)

        # Step 1: Get α rescored results
        alpha_approach = AlphaRescoredPPRApproach()
        alpha_results = await alpha_approach.retrieve(query, context, top_k * 2)

        # Step 2: Apply ICL enhancement with single triple context
        icl_enhanced_items = []
        for item_id, alpha_score in alpha_results:
            # Calculate ICL enhancement factor
            icl_factor = self._calculate_icl_factor(query, item_id, context)

            # Combine α score with ICL enhancement
            final_score = alpha_score * (1 - self.icl_weight) + icl_factor * self.icl_weight
            icl_enhanced_items.append((item_id, final_score))

        # Sort by final score and take top-k
        icl_enhanced_items.sort(key=lambda x: x[1], reverse=True)
        results = icl_enhanced_items[:top_k]

        # Track performance
        retrieval_time = (datetime.now(UTC) - start_time).total_seconds()
        self.retrieval_times.append(retrieval_time)

        # Calculate token cost (highest due to ICL processing)
        token_cost = self.calculate_token_cost(len(query.query_text.split()), len(context.preferences), len(results))
        self.token_costs.append(token_cost)

        return results

    def _calculate_icl_factor(self, query: UserQuery, item_id: str, context: PersonalizationContext) -> float:
        """Calculate ICL enhancement factor using single triple context."""
        # Mock ICL calculation - in reality would use language models
        icl_score = 0.5

        # Create single triple context from user history
        if context.interaction_history:
            recent_item = context.interaction_history[-1]  # Most recent interaction

            # Generate contextual triple: (user, liked, recent_item)

            # Calculate similarity between current item and triple context
            # Mock similarity calculation
            if "movie" in item_id and "movie" in recent_item:
                # Same domain bonus
                icl_score += 0.2

                # Extract number from item_id for mock similarity
                try:
                    current_num = int(item_id.split("_")[1])
                    recent_num = int(recent_item.split("_")[1])
                    similarity = 1.0 / (1.0 + abs(current_num - recent_num) * 0.1)
                    icl_score += similarity * 0.3
                except (ValueError, IndexError):
                    # Skip if item_id format is unexpected
                    pass

            elif "paper" in item_id and "paper" in recent_item:
                # Academic domain logic
                icl_score += 0.15

                # Mock topic similarity
                if np.random.random() > 0.4:  # 60% chance of topic similarity
                    icl_score += 0.25

        # Factor in query-item semantic similarity (mock)
        query_terms = set(query.query_text.lower().split())
        if "action" in query_terms and "movie" in item_id:
            icl_score += 0.1
        elif "learning" in query_terms and "paper" in item_id:
            icl_score += 0.15

        return min(max(icl_score, 0.0), 1.0)

    def calculate_token_cost(self, query_length: int, context_size: int, retrieved_items: int) -> int:
        """ICL adds significant token cost for context processing."""
        base_cost = query_length + retrieved_items * 2
        alpha_cost = context_size * 5 + retrieved_items * 3
        icl_cost = 50 + retrieved_items * 8  # Single triple context + LM processing
        return base_cost + alpha_cost + icl_cost


class PersonalizationBenchmark:
    """Main personalization benchmark evaluation system."""

    def __init__(self, output_dir: Path = Path("./personalization_results")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_generator = PersonalizationDatasetGenerator()
        self.approaches = [
            BasePPRApproach(),
            AlphaRescoredPPRApproach(),
            ICLEnhancedApproach(),
        ]

        self.evaluation_results = {}

    async def run_full_benchmark(self) -> dict[str, PersonalizationMetrics]:
        """Run the complete personalization benchmark."""
        logger.info("Starting HypeRAG Personalization Benchmark...")

        # Get datasets and contexts
        movielens_queries = self.dataset_generator.movielens_queries
        doc_click_queries = self.dataset_generator.doc_click_queries
        contexts = self.dataset_generator.get_personalization_contexts()

        all_queries = movielens_queries + doc_click_queries

        results = {}

        # Evaluate each approach
        for approach in self.approaches:
            logger.info(f"Evaluating {approach.name}...")

            approach_results = await self._evaluate_approach(approach, all_queries, contexts)
            results[approach.name] = approach_results

            logger.info(
                f"Completed {approach.name}: MAP={approach_results.map_score:.3f}, "
                f"NDCG@10={approach_results.ndcg_at_10:.3f}"
            )

        # Save results
        await self._save_benchmark_results(results)

        logger.info("Personalization benchmark completed successfully")
        return results

    async def _evaluate_approach(
        self,
        approach: PersonalizationApproach,
        queries: list[UserQuery],
        contexts: dict[str, PersonalizationContext],
    ) -> PersonalizationMetrics:
        """Evaluate a single personalization approach."""
        all_map_scores = []
        all_ndcg_scores = []
        all_recall_scores = []

        for query in queries:
            context = contexts.get(query.user_id)
            if not context:
                logger.warning(f"No context found for user {query.user_id}")
                continue

            # Retrieve results
            retrieved_items = await approach.retrieve(query, context, top_k=20)

            # Calculate metrics for this query
            map_score = self._calculate_map(query, retrieved_items)
            ndcg_score = self._calculate_ndcg_at_k(query, retrieved_items, k=10)
            recall_score = self._calculate_recall_at_k(query, retrieved_items, k=20)

            all_map_scores.append(map_score)
            all_ndcg_scores.append(ndcg_score)
            all_recall_scores.append(recall_score)

        # Calculate token cost delta (compared to base approach)
        avg_token_cost = statistics.mean(approach.token_costs) if approach.token_costs else 0
        base_token_cost = 50  # Baseline token cost
        token_cost_delta = (avg_token_cost - base_token_cost) / base_token_cost

        # Calculate average retrieval time
        avg_retrieval_time = statistics.mean(approach.retrieval_times) if approach.retrieval_times else 0

        return PersonalizationMetrics(
            approach_name=approach.name,
            map_score=statistics.mean(all_map_scores) if all_map_scores else 0.0,
            ndcg_at_10=statistics.mean(all_ndcg_scores) if all_ndcg_scores else 0.0,
            recall_at_20=(statistics.mean(all_recall_scores) if all_recall_scores else 0.0),
            token_cost_delta=token_cost_delta,
            total_queries=len(queries),
            avg_retrieval_time=avg_retrieval_time,
            dataset_name="movielens_and_doc_clicks",
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _calculate_map(self, query: UserQuery, retrieved_items: list[tuple[str, float]]) -> float:
        """Calculate Mean Average Precision."""
        relevant_items = set(query.relevant_items)

        if not relevant_items:
            return 0.0

        precision_sum = 0.0
        relevant_retrieved = 0

        for i, (item_id, _score) in enumerate(retrieved_items):
            if item_id in relevant_items:
                relevant_retrieved += 1
                precision_at_i = relevant_retrieved / (i + 1)
                precision_sum += precision_at_i

        if relevant_retrieved == 0:
            return 0.0

        return precision_sum / len(relevant_items)

    def _calculate_ndcg_at_k(self, query: UserQuery, retrieved_items: list[tuple[str, float]], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""

        def dcg_at_k(relevance_scores: list[float], k: int) -> float:
            """Calculate DCG at k."""
            dcg = 0.0
            for i in range(min(k, len(relevance_scores))):
                dcg += (2 ** relevance_scores[i] - 1) / math.log2(i + 2)
            return dcg

        # Get relevance scores for retrieved items
        retrieved_relevance = []
        for item_id, _score in retrieved_items[:k]:
            relevance = query.relevance_scores.get(item_id, 0.0)
            retrieved_relevance.append(relevance)

        # Calculate DCG for retrieved items
        dcg = dcg_at_k(retrieved_relevance, k)

        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(query.relevance_scores.values(), reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _calculate_recall_at_k(self, query: UserQuery, retrieved_items: list[tuple[str, float]], k: int) -> float:
        """Calculate Recall at k."""
        relevant_items = set(query.relevant_items)
        retrieved_relevant = set()

        for item_id, _score in retrieved_items[:k]:
            if item_id in relevant_items:
                retrieved_relevant.add(item_id)

        if not relevant_items:
            return 0.0

        return len(retrieved_relevant) / len(relevant_items)

    async def _save_benchmark_results(self, results: dict[str, PersonalizationMetrics]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"personalization_results_{timestamp}.json"
        detailed_results = {
            "metadata": {
                "benchmark_version": "1.0",
                "timestamp": datetime.now(UTC).isoformat(),
                "approaches_evaluated": list(results.keys()),
            },
            "results": {name: asdict(metrics) for name, metrics in results.items()},
            "comparison": self._generate_comparison_analysis(results),
        }

        with open(results_file, "w") as f:
            json.dump(detailed_results, f, indent=2)

        # Save metrics summary
        metrics_file = self.output_dir / f"personalization_metrics_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {name: asdict(metrics) for name, metrics in results.items()},
                f,
                indent=2,
            )

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")

    def _generate_comparison_analysis(self, results: dict[str, PersonalizationMetrics]) -> dict[str, Any]:
        """Generate comparative analysis of approaches."""
        base_result = results.get("Base PPR")
        if not base_result:
            return {}

        analysis = {
            "map_improvements": {},
            "ndcg_improvements": {},
            "recall_improvements": {},
            "cost_analysis": {},
            "recommendations": [],
        }

        for name, metrics in results.items():
            if name == "Base PPR":
                continue

            # Calculate improvements over base
            map_improvement = (metrics.map_score - base_result.map_score) / base_result.map_score
            ndcg_improvement = (metrics.ndcg_at_10 - base_result.ndcg_at_10) / base_result.ndcg_at_10
            recall_improvement = (metrics.recall_at_20 - base_result.recall_at_20) / base_result.recall_at_20

            analysis["map_improvements"][name] = map_improvement
            analysis["ndcg_improvements"][name] = ndcg_improvement
            analysis["recall_improvements"][name] = recall_improvement
            analysis["cost_analysis"][name] = metrics.token_cost_delta

            # Generate recommendations
            if map_improvement > 0.1 and metrics.token_cost_delta < 2.0:
                analysis["recommendations"].append(f"{name}: Good balance of performance and cost")
            elif map_improvement > 0.2:
                analysis["recommendations"].append(
                    f"{name}: High performance improvement, consider for high-value queries"
                )
            elif metrics.token_cost_delta < 0.5:
                analysis["recommendations"].append(f"{name}: Cost-effective option")

        return analysis


async def main() -> None:
    parser = argparse.ArgumentParser(description="HypeRAG Personalization Benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./personalization_results"),
        help="Output directory for results",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run benchmark
    benchmark = PersonalizationBenchmark(output_dir=args.output_dir)

    try:
        results = await benchmark.run_full_benchmark()

        # Print comparison results
        print("\n" + "=" * 70)
        print("HYPERAG PERSONALIZATION BENCHMARK RESULTS")
        print("=" * 70)

        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  MAP Score:          {metrics.map_score:.3f}")
            print(f"  NDCG@10:           {metrics.ndcg_at_10:.3f}")
            print(f"  Recall@20:         {metrics.recall_at_20:.3f}")
            print(f"  Token Cost Delta:  {metrics.token_cost_delta:+.1%}")
            print(f"  Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")

        # Show improvements over base
        base_result = results.get("Base PPR")
        if base_result:
            print("\n" + "=" * 40)
            print("IMPROVEMENTS OVER BASE PPR")
            print("=" * 40)

            for name, metrics in results.items():
                if name == "Base PPR":
                    continue

                map_imp = (metrics.map_score - base_result.map_score) / base_result.map_score
                ndcg_imp = (metrics.ndcg_at_10 - base_result.ndcg_at_10) / base_result.ndcg_at_10
                recall_imp = (metrics.recall_at_20 - base_result.recall_at_20) / base_result.recall_at_20

                print(f"\n{name}:")
                print(f"  MAP Improvement:    {map_imp:+.1%}")
                print(f"  NDCG@10 Improvement: {ndcg_imp:+.1%}")
                print(f"  Recall@20 Improvement: {recall_imp:+.1%}")

        print("=" * 70)

    except Exception as e:
        logger.exception(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
