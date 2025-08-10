"""Hybrid Retriever Orchestration.

Orchestrates the complete retrieval pipeline:
- Vector similarity search
- Personalized PageRank with α-fusion
- Creative mode routing
- Result fusion and ranking
"""

from dataclasses import dataclass, field
import logging
import time
from typing import Any

from AIVillage.src.mcp_servers.hyperag.memory.hippo_index import HippoIndex
from AIVillage.src.mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG
from AIVillage.src.mcp_servers.hyperag.models import QueryPlan

from .ppr_retriever import AlphaProfileStore, PersonalizedPageRank, PPRResults

logger = logging.getLogger(__name__)


@dataclass
class HybridResults:
    """Combined results from hybrid retrieval."""

    vector_results: list[dict[str, Any]]
    ppr_results: PPRResults
    fused_results: list[dict[str, Any]]
    total_time_ms: float
    reasoning_trace: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """Orchestrates vector + PPR + α-fusion retrieval pipeline.

    Pipeline:
    1. Vector similarity search on both memory systems
    2. PersonalizedPageRank with recency boost and α-weights
    3. Result fusion with configurable weights
    4. Creative mode handling
    """

    def __init__(
        self,
        hippo_index: HippoIndex,
        hypergraph_kg: HypergraphKG,
        ppr_retriever: PersonalizedPageRank,
        alpha_store: AlphaProfileStore | None = None,
    ) -> None:
        self.hippo_index = hippo_index
        self.hypergraph_kg = hypergraph_kg
        self.ppr_retriever = ppr_retriever
        self.alpha_store = alpha_store

        # Fusion weights (could be configurable)
        self.vector_weight = 0.4
        self.ppr_weight = 0.6
        self.creative_threshold = 0.5  # Confidence threshold for auto-creative mode

        logger.info("HybridRetriever initialized")

    async def retrieve(
        self, query: str, user_id: str | None, plan: QueryPlan, limit: int = 50
    ) -> HybridResults:
        """Main retrieval method orchestrating the complete pipeline.

        Args:
            query: Natural language query
            user_id: User ID for personalization
            plan: Query plan with mode and parameters
            limit: Maximum results to return

        Returns:
            HybridResults with vector, PPR, and fused results
        """
        start_time = time.time()
        reasoning_trace = [f"Starting hybrid retrieval for query: '{query[:50]}...'"]

        try:
            # Determine if creative mode should be used
            creative_flag = self._should_use_creative_mode(plan)
            reasoning_trace.append(f"Creative mode: {creative_flag}")

            # Phase 1: Vector similarity search
            reasoning_trace.append("Phase 1: Vector similarity search")
            vector_results = await self._vector_search(query, user_id, limit)
            reasoning_trace.append(
                f"Vector search returned {len(vector_results)} results"
            )

            # Phase 2: Extract seeds for PPR
            query_seeds = self._extract_seeds(vector_results, plan)
            reasoning_trace.append(f"Extracted {len(query_seeds)} seeds for PPR")

            # Phase 3: PersonalizedPageRank retrieval
            reasoning_trace.append("Phase 3: Personalized PageRank")
            ppr_results = await self.ppr_retriever.retrieve(
                query_seeds=query_seeds,
                user_id=user_id,
                plan=plan,
                creative_mode=creative_flag,
            )
            reasoning_trace.extend(ppr_results.reasoning_trace)

            # Phase 4: Result fusion
            reasoning_trace.append("Phase 4: Result fusion")
            fused_results = self._fuse_results(vector_results, ppr_results, limit)
            reasoning_trace.append(
                f"Fusion produced {len(fused_results)} final results"
            )

            total_time = (time.time() - start_time) * 1000
            reasoning_trace.append(f"Hybrid retrieval completed in {total_time:.2f}ms")

            return HybridResults(
                vector_results=vector_results,
                ppr_results=ppr_results,
                fused_results=fused_results,
                total_time_ms=total_time,
                reasoning_trace=reasoning_trace,
                metadata={
                    "query": query,
                    "user_id": user_id,
                    "creative_mode": creative_flag,
                    "vector_weight": self.vector_weight,
                    "ppr_weight": self.ppr_weight,
                    "seeds_extracted": len(query_seeds),
                },
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            error_msg = f"Hybrid retrieval failed: {e!s}"
            logger.exception(error_msg)
            reasoning_trace.append(error_msg)

            return HybridResults(
                vector_results=[],
                ppr_results=PPRResults([], [], {}, [], 0.0),
                fused_results=[],
                total_time_ms=total_time,
                reasoning_trace=reasoning_trace,
                metadata={"error": str(e)},
            )

    def _should_use_creative_mode(self, plan: QueryPlan) -> bool:
        """Determine if creative mode should be enabled."""
        mode = getattr(plan, "mode", "NORMAL")

        if mode == "CREATIVE":
            return True
        if mode == "AUTO_CREATIVE_IF_LOW_CONF":
            # Check if previous results had low confidence
            # For now, use a simple heuristic
            return getattr(plan, "confidence_hint", 1.0) < self.creative_threshold
        return False

    async def _vector_search(
        self, query: str, user_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search on both memory systems."""
        try:
            vector_results = []

            # Search episodic memory (HippoIndex)
            if hasattr(self.hippo_index, "vector_similarity_search"):
                hippo_results = await self.hippo_index.vector_similarity_search(
                    query_text=query,
                    limit=limit // 2,  # Split between systems
                    score_threshold=0.7,
                )

                for node, score in hippo_results:
                    vector_results.append(
                        {
                            "id": node.id,
                            "content": node.content,
                            "score": score,
                            "source": "episodic",
                            "memory_type": "episodic",
                            "confidence": node.confidence,
                            "created_at": (
                                node.created_at.isoformat() if node.created_at else None
                            ),
                        }
                    )

            # Search semantic memory (HypergraphKG)
            if hasattr(self.hypergraph_kg, "semantic_similarity_search"):
                semantic_results = await self.hypergraph_kg.semantic_similarity_search(
                    query_text=query,
                    limit=limit // 2,
                    score_threshold=0.7,
                    community_filter=None,  # Could use user's community preference
                )

                for node, score in semantic_results:
                    vector_results.append(
                        {
                            "id": node.id,
                            "content": node.content,
                            "score": score,
                            "source": "semantic",
                            "memory_type": "semantic",
                            "confidence": node.confidence,
                            "community_id": getattr(node, "community_id", None),
                            "pagerank_score": getattr(node, "pagerank_score", 0.0),
                        }
                    )

            # Sort by similarity score
            vector_results.sort(key=lambda x: x["score"], reverse=True)

            return vector_results[:limit]

        except Exception as e:
            logger.exception(f"Vector search failed: {e!s}")
            return []

    def _extract_seeds(
        self, vector_results: list[dict[str, Any]], plan: QueryPlan
    ) -> list[str]:
        """Extract seed node IDs for PPR from vector results."""
        try:
            # Use top vector results as seeds
            max_seeds = getattr(plan, "max_seeds", 10)
            min_score = getattr(plan, "seed_score_threshold", 0.5)

            seeds = []
            for result in vector_results:
                if result["score"] >= min_score and len(seeds) < max_seeds:
                    seeds.append(result["id"])

            # Ensure we have at least some seeds
            if not seeds and vector_results:
                # Take top result even if below threshold
                seeds.append(vector_results[0]["id"])

            return seeds

        except Exception as e:
            logger.exception(f"Seed extraction failed: {e!s}")
            return []

    def _fuse_results(
        self, vector_results: list[dict[str, Any]], ppr_results: PPRResults, limit: int
    ) -> list[dict[str, Any]]:
        """Fuse vector and PPR results with weighted scoring."""
        try:
            # Create lookup for vector scores
            {result["id"]: result["score"] for result in vector_results}

            # Create lookup for PPR scores

            # Combine all unique items
            all_items = {}

            # Add vector results
            for result in vector_results:
                item_id = result["id"]
                all_items[item_id] = result.copy()
                all_items[item_id]["vector_score"] = result["score"]
                all_items[item_id]["ppr_score"] = 0.0

            # Add PPR results (nodes and edges)
            for node in ppr_results.nodes:
                item_id = node["id"]
                if item_id not in all_items:
                    all_items[item_id] = {
                        "id": item_id,
                        "content": f"Node {item_id}",  # Would get from actual node
                        "source": "ppr",
                        "vector_score": 0.0,
                    }
                all_items[item_id]["ppr_score"] = node["score"]
                all_items[item_id]["uncertainty"] = node.get("uncertainty", 0.0)
                all_items[item_id]["confidence"] = node.get("confidence", 1.0)

            for edge in ppr_results.edges:
                item_id = edge["id"]
                if item_id not in all_items:
                    all_items[item_id] = {
                        "id": item_id,
                        "content": f"Edge {item_id}",  # Would get from actual edge
                        "source": "ppr",
                        "vector_score": 0.0,
                        "type": "edge",
                    }
                all_items[item_id]["ppr_score"] = edge["score"]
                all_items[item_id]["uncertainty"] = edge.get("uncertainty", 0.0)
                all_items[item_id]["confidence"] = edge.get("confidence", 1.0)

            # Calculate fused scores
            fused_items = []
            for item_id, item in all_items.items():
                vector_score = item.get("vector_score", 0.0)
                ppr_score = item.get("ppr_score", 0.0)

                # Weighted fusion
                fused_score = (
                    self.vector_weight * vector_score + self.ppr_weight * ppr_score
                )

                # Apply confidence boost
                confidence = item.get("confidence", 1.0)
                fused_score *= confidence

                # Apply uncertainty penalty
                uncertainty = item.get("uncertainty", 0.0)
                fused_score *= 1.0 - uncertainty * 0.5

                item["fused_score"] = fused_score
                fused_items.append(item)

            # Sort by fused score and limit
            fused_items.sort(key=lambda x: x["fused_score"], reverse=True)

            return fused_items[:limit]

        except Exception as e:
            logger.exception(f"Result fusion failed: {e!s}")
            # Return vector results as fallback
            return vector_results[:limit]

    def get_stats(self) -> dict[str, Any]:
        """Get retrieval statistics."""
        ppr_stats = self.ppr_retriever.get_performance_stats()

        return {
            "fusion_weights": {
                "vector_weight": self.vector_weight,
                "ppr_weight": self.ppr_weight,
            },
            "creative_threshold": self.creative_threshold,
            "ppr_performance": ppr_stats,
            "alpha_store_enabled": self.alpha_store is not None,
        }

    def update_fusion_weights(self, vector_weight: float, ppr_weight: float) -> None:
        """Update fusion weights (could be based on user feedback)."""
        total = vector_weight + ppr_weight
        if total > 0:
            self.vector_weight = vector_weight / total
            self.ppr_weight = ppr_weight / total
            logger.info(
                f"Updated fusion weights: vector={self.vector_weight:.3f}, ppr={self.ppr_weight:.3f}"
            )


# Factory function


def create_hybrid_retriever(
    hippo_index: HippoIndex,
    hypergraph_kg: HypergraphKG,
    ppr_retriever: PersonalizedPageRank,
    alpha_store: AlphaProfileStore | None = None,
) -> HybridRetriever:
    """Create a HybridRetriever with the given components."""
    return HybridRetriever(
        hippo_index=hippo_index,
        hypergraph_kg=hypergraph_kg,
        ppr_retriever=ppr_retriever,
        alpha_store=alpha_store,
    )
