"""Personalized PageRank Retriever with Rel-GAT α-weight fusion.

Core retrieval engine that combines:
- Standard PPR over hypergraph knowledge
- Recency boost from episodic memory
- α-weight personalization (Rel-GAT profiles)
- Creative mode routing to divergent retrieval
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any

import yaml

from AIVillage.src.mcp_servers.hyperag.memory.hippo_index import HippoIndex
from AIVillage.src.mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG
from AIVillage.src.mcp_servers.hyperag.models import QueryPlan

logger = logging.getLogger(__name__)


@dataclass
class PPRResults:
    """Results from Personalized PageRank retrieval."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    scores: dict[str, float]
    reasoning_trace: list[str]
    query_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_results(self) -> int:
        return len(self.nodes) + len(self.edges)


@dataclass
class AlphaProfile:
    """User's α-weight profile for relation personalization."""

    user_id: str
    relation_weights: dict[str, float]  # relation -> α-weight
    last_updated: datetime
    interaction_count: int = 0
    confidence: float = 1.0

    def get_weight(self, relation: str, default: float = 1.0) -> float:
        """Get α-weight for a relation with fallback to default."""
        return self.relation_weights.get(relation, default)


class AlphaProfileStore:
    """Storage and management for user α-profiles."""

    def __init__(self, redis_client=None) -> None:
        self.redis_client = redis_client
        self.profiles_cache: dict[str, AlphaProfile] = {}
        self.cache_ttl = 3600  # 1 hour

    async def get_profile(self, user_id: str) -> AlphaProfile | None:
        """Get user's α-profile."""
        # Check local cache first
        if user_id in self.profiles_cache:
            return self.profiles_cache[user_id]

        # Try Redis if available
        if self.redis_client:
            try:
                profile_data = await self.redis_client.get(f"hyperag:alpha:{user_id}")
                if profile_data:
                    data = json.loads(profile_data)
                    profile = AlphaProfile(
                        user_id=data["user_id"],
                        relation_weights=data["relation_weights"],
                        last_updated=datetime.fromisoformat(data["last_updated"]),
                        interaction_count=data.get("interaction_count", 0),
                        confidence=data.get("confidence", 1.0),
                    )
                    self.profiles_cache[user_id] = profile
                    return profile
            except Exception as e:
                logger.warning(f"Failed to load α-profile for {user_id}: {e!s}")

        return None

    async def get_top_alpha(self, user_id: str, node_ids: list[str]) -> dict[str, float]:
        """Get α-weights for specific nodes based on their relations."""
        profile = await self.get_profile(user_id)
        if not profile:
            return {}

        # This would typically query the graph to get relations for these nodes
        # For now, return sample weights
        alpha_scores = {}
        for node_id in node_ids:
            # In practice, would look up node's primary relations
            # and apply user's α-weights
            alpha_scores[node_id] = 1.0  # Placeholder

        return alpha_scores

    async def update_profile(self, profile: AlphaProfile) -> bool:
        """Update user's α-profile."""
        try:
            self.profiles_cache[profile.user_id] = profile

            if self.redis_client:
                profile_data = {
                    "user_id": profile.user_id,
                    "relation_weights": profile.relation_weights,
                    "last_updated": profile.last_updated.isoformat(),
                    "interaction_count": profile.interaction_count,
                    "confidence": profile.confidence,
                }

                await self.redis_client.setex(
                    f"hyperag:alpha:{profile.user_id}",
                    self.cache_ttl,
                    json.dumps(profile_data),
                )

            return True

        except Exception as e:
            logger.exception(f"Failed to update α-profile for {profile.user_id}: {e!s}")
            return False


class PersonalizedPageRank:
    """Single-pass, uncertainty-aware multi-hop retrieval.

    Pipeline:
    1) Seed collection from query planner
    2) Merge Hippo-Index recency seeds
    3) Personalised PageRank over Hypergraph-KG
    4) α-weight fusion (Rel-GAT personalisation)
    5) Return scored Node / Hyperedge list
    """

    def __init__(
        self,
        hippo_index: HippoIndex,
        hypergraph: HypergraphKG,
        alpha_store: AlphaProfileStore | None = None,
        damping: float = 0.85,
        config_path: str | None = None,
    ) -> None:
        self.hippo_index = hippo_index
        self.hypergraph = hypergraph
        self.alpha_store = alpha_store
        self.damping = damping

        # Load configuration
        self.config = self._load_config(config_path)

        # Update damping from config
        self.damping = self.config.get("pagerank", {}).get("damping_factor", damping)

        # Performance tracking
        self.query_count = 0
        self.total_time_ms = 0.0

        logger.info(f"PersonalizedPageRank initialized with damping={self.damping}")

    async def retrieve(
        self,
        query_seeds: list[str],
        user_id: str | None,
        plan: QueryPlan,
        *,
        creative_mode: bool = False,
    ) -> PPRResults:
        """Parameters
        ----------
        query_seeds : initial entity ids
        user_id     : Digital Twin id (for α-weights)
        plan        : contains temporal filters, hop limits
        creative_mode : if True → delegate entirely to DivergentRetriever

        Returns:
        -------
        PPRResults(nodes, edges, reasoning_trace)
        """
        start_time = time.time()
        reasoning_trace = [f"Starting PPR retrieval with {len(query_seeds)} seeds"]

        try:
            # Handle creative mode delegation
            if creative_mode:
                reasoning_trace.append("Routing to DivergentRetriever for creative mode")
                return await self._route_to_creative(query_seeds, user_id, plan, reasoning_trace)

            # 1. Standard dense k-NN on Hippo-Index for recency boost
            reasoning_trace.append("Fetching recency nodes from HippoIndex")
            recency_nodes = await self._knn_hippo(query_seeds, plan)
            reasoning_trace.append(f"Found {len(recency_nodes)} recent nodes")

            # 2. Run personalised PageRank on Hypergraph-KG
            all_seeds = query_seeds + recency_nodes
            reasoning_trace.append(f"Running PPR with {len(all_seeds)} total seeds")
            base_scores = await self._pagerank(all_seeds, plan)
            reasoning_trace.append(f"PPR completed with {len(base_scores)} scored nodes")

            # 3. α-weight fusion if profile exists
            if user_id and self.alpha_store:
                reasoning_trace.append(f"Applying α-weight fusion for user {user_id}")
                alpha_scores = await self.alpha_store.get_top_alpha(user_id, list(base_scores.keys()))
                fusion_scores = self._fuse_with_alpha(base_scores, alpha_scores)
                reasoning_trace.append(f"α-fusion applied to {len(alpha_scores)} nodes")
            else:
                fusion_scores = base_scores
                reasoning_trace.append("No α-weight fusion (no user profile)")

            # 4. Uncertainty weighting & pruning
            reasoning_trace.append("Applying uncertainty weighting and pruning")
            results = await self._apply_uncertainty(fusion_scores, plan)

            query_time = (time.time() - start_time) * 1000
            self.query_count += 1
            self.total_time_ms += query_time

            reasoning_trace.append(f"Retrieval completed in {query_time:.2f}ms")

            return PPRResults(
                nodes=results["nodes"],
                edges=results["edges"],
                scores=results["scores"],
                reasoning_trace=reasoning_trace,
                query_time_ms=query_time,
                metadata={
                    "query_seeds": query_seeds,
                    "user_id": user_id,
                    "recency_boost": len(recency_nodes),
                    "alpha_fusion": user_id is not None and self.alpha_store is not None,
                    "total_scored_items": len(fusion_scores),
                },
            )

        except Exception as e:
            query_time = (time.time() - start_time) * 1000
            error_msg = f"PPR retrieval failed: {e!s}"
            logger.exception(error_msg)
            reasoning_trace.append(error_msg)

            return PPRResults(
                nodes=[],
                edges=[],
                scores={},
                reasoning_trace=reasoning_trace,
                query_time_ms=query_time,
                metadata={"error": str(e)},
            )

    async def _knn_hippo(self, query_seeds: list[str], plan: QueryPlan) -> list[str]:
        """Recency-biased vector search on episodic memory."""
        try:
            recency_config = self.config.get("recency_boost", {})

            if not recency_config.get("enabled", True):
                return []

            knn_limit = recency_config.get("knn_limit", 20)
            max_age_hours = recency_config.get("max_age_hours", 24)

            # Get recent nodes from HippoIndex
            recent_nodes = await self.hippo_index.get_recent_nodes(
                hours=max_age_hours,
                user_id=getattr(plan, "user_id", None),
                limit=knn_limit,
            )

            # Extract node IDs
            recency_node_ids = [node.id for node in recent_nodes[:knn_limit]]

            logger.debug(f"Retrieved {len(recency_node_ids)} recent nodes from HippoIndex")
            return recency_node_ids

        except Exception as e:
            logger.warning(f"Failed to get recency nodes: {e!s}")
            return []

    async def _pagerank(self, seed_nodes: list[str], plan: QueryPlan) -> dict[str, float]:
        """Run Personalized PageRank on the hypergraph."""
        try:
            ppr_config = self.config.get("pagerank", {})
            max_iterations = ppr_config.get("max_iterations", 50)
            convergence_tolerance = ppr_config.get("convergence_tolerance", 1e-6)
            min_score_threshold = ppr_config.get("min_score_threshold", 0.001)

            # Use the hypergraph's built-in PPR if available
            if hasattr(self.hypergraph, "personalized_pagerank"):
                scores = await self.hypergraph.personalized_pagerank(
                    start_nodes=seed_nodes,
                    user_id=getattr(plan, "user_id", None),
                    alpha=1 - self.damping,  # Convert to restart probability
                    max_iterations=max_iterations,
                    tolerance=convergence_tolerance,
                )
            else:
                # Fallback implementation
                scores = await self._pagerank_iteration(seed_nodes, max_iterations, convergence_tolerance)

            # Filter by minimum score threshold
            filtered_scores = {node_id: score for node_id, score in scores.items() if score >= min_score_threshold}

            logger.debug(f"PPR computed {len(filtered_scores)} scores above threshold")
            return filtered_scores

        except Exception as e:
            logger.exception(f"PageRank computation failed: {e!s}")
            # Return uniform scores for seeds as fallback
            return {node_id: 1.0 / len(seed_nodes) for node_id in seed_nodes}

    async def _pagerank_iteration(
        self, seed_nodes: list[str], max_iterations: int, tolerance: float
    ) -> dict[str, float]:
        """Fallback PPR implementation with uncertainty decay."""
        # Initialize scores
        scores = {node_id: 1.0 / len(seed_nodes) for node_id in seed_nodes}
        personalization = scores.copy()

        uncertainty_config = self.config.get("uncertainty", {})
        decay_per_hop = uncertainty_config.get("decay_per_hop", 0.1)

        for iteration in range(max_iterations):
            new_scores = {}

            # Get neighbors for each node (simplified)
            for node_id in scores:
                # Restart probability component
                new_score = (1 - self.damping) * personalization.get(node_id, 0.0)

                # Random walk component (would need actual graph structure)
                # For now, apply simple decay
                current_score = scores[node_id]
                uncertainty_factor = 1.0 - (decay_per_hop * iteration)
                new_score += self.damping * current_score * max(0.1, uncertainty_factor)

                new_scores[node_id] = new_score

            # Check convergence
            max_diff = max(abs(new_scores[node_id] - scores[node_id]) for node_id in scores)

            scores = new_scores

            if max_diff < tolerance:
                logger.debug(f"PPR converged after {iteration + 1} iterations")
                break

        return scores

    def _fuse_with_alpha(self, base_scores: dict[str, float], alpha_scores: dict[str, float]) -> dict[str, float]:
        """Fuse base PPR scores with α-weights: score = base + λ₁·α - λ₂·popularity_rank."""
        fusion_config = self.config.get("alpha_fusion", {})
        base_weight = fusion_config.get("base_weight", 1.0)
        alpha_weight = fusion_config.get("alpha_weight", 0.3)
        popularity_penalty = fusion_config.get("popularity_penalty", 0.1)
        min_alpha_threshold = fusion_config.get("min_alpha_threshold", 0.1)

        fusion_scores = {}

        for node_id, base_score in base_scores.items():
            # Start with base score
            fused_score = base_weight * base_score

            # Add α-weight boost if available and above threshold
            alpha_score = alpha_scores.get(node_id, 0.0)
            if alpha_score >= min_alpha_threshold:
                fused_score += alpha_weight * alpha_score

            # Apply popularity penalty (would need actual popularity ranks)
            # For now, apply small random penalty to simulate
            popularity_rank = hash(node_id) % 100  # Simulate popularity rank
            popularity_factor = popularity_penalty * (popularity_rank / 100.0)
            fused_score -= popularity_factor

            fusion_scores[node_id] = max(0.0, fused_score)  # Ensure non-negative

        logger.debug(f"α-fusion applied to {len(fusion_scores)} nodes")
        return fusion_scores

    async def _apply_uncertainty(self, scores: dict[str, float], plan: QueryPlan) -> dict[str, Any]:
        """Apply uncertainty weighting and convert to result format."""
        uncertainty_config = self.config.get("uncertainty", {})
        max_uncertainty = uncertainty_config.get("max_uncertainty", 0.8)
        confidence_weight = uncertainty_config.get("confidence_weight", 0.5)

        performance_config = self.config.get("performance", {})
        max_nodes = performance_config.get("max_nodes_per_query", 1000)

        # Filter and sort scores
        filtered_items = []

        for node_id, score in scores.items():
            # Simulate uncertainty (would come from actual nodes)
            uncertainty = min(0.5, abs(hash(node_id)) % 100 / 200.0)  # 0-0.5 range

            if uncertainty <= max_uncertainty:
                # Apply confidence weighting
                confidence = 1.0 - uncertainty
                adjusted_score = score * (1.0 + confidence_weight * confidence)

                filtered_items.append(
                    {
                        "id": node_id,
                        "score": adjusted_score,
                        "uncertainty": uncertainty,
                        "confidence": confidence,
                    }
                )

        # Sort by score and limit results
        filtered_items.sort(key=lambda x: x["score"], reverse=True)
        top_items = filtered_items[:max_nodes]

        # Split into nodes and edges (simplified - would need actual type info)
        nodes = []
        edges = []
        final_scores = {}

        for item in top_items:
            final_scores[item["id"]] = item["score"]

            # Simulate node/edge classification
            if hash(item["id"]) % 2 == 0:
                nodes.append(
                    {
                        "id": item["id"],
                        "score": item["score"],
                        "uncertainty": item["uncertainty"],
                        "confidence": item["confidence"],
                        "type": "node",
                    }
                )
            else:
                edges.append(
                    {
                        "id": item["id"],
                        "score": item["score"],
                        "uncertainty": item["uncertainty"],
                        "confidence": item["confidence"],
                        "type": "edge",
                    }
                )

        return {"nodes": nodes, "edges": edges, "scores": final_scores}

    async def _route_to_creative(
        self,
        query_seeds: list[str],
        user_id: str | None,
        plan: QueryPlan,
        reasoning_trace: list[str],
    ) -> PPRResults:
        """Route to DivergentRetriever for creative mode."""
        try:
            # Dynamic import to avoid circular dependencies
            from .divergent_retriever import DivergentRetriever

            # Create divergent retriever instance
            divergent = DivergentRetriever(hypergraph=self.hypergraph, config=self.config.get("creative_mode", {}))

            # Delegate to creative retrieval
            creative_results = await divergent.retrieve_creative(query_seeds=query_seeds, user_id=user_id, plan=plan)

            reasoning_trace.extend(creative_results.reasoning_trace)
            return creative_results

        except ImportError:
            # DivergentRetriever not yet implemented
            reasoning_trace.append("DivergentRetriever not available, falling back to standard PPR")

            # Fall back to standard retrieval
            temp_plan = plan
            return await self.retrieve(
                query_seeds=query_seeds,
                user_id=user_id,
                plan=temp_plan,
                creative_mode=False,
            )
        except Exception as e:
            error_msg = f"Creative mode routing failed: {e!s}"
            logger.exception(error_msg)
            reasoning_trace.append(error_msg)

            # Return empty results
            return PPRResults(
                nodes=[],
                edges=[],
                scores={},
                reasoning_trace=reasoning_trace,
                query_time_ms=0.0,
                metadata={"creative_mode_error": str(e)},
            )

    def _load_config(self, config_path: str | None = None) -> dict[str, Any]:
        """Load retrieval configuration."""
        if config_path is None:
            # Default config path
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent
            config_path = project_root / "config" / "retrieval.yaml"

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded retrieval config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e!s}")
            # Return default configuration
            return {
                "pagerank": {"damping_factor": 0.85, "max_iterations": 50},
                "alpha_fusion": {"base_weight": 1.0, "alpha_weight": 0.3},
                "recency_boost": {"enabled": True, "knn_limit": 20},
                "uncertainty": {"max_uncertainty": 0.8, "confidence_weight": 0.5},
                "performance": {"max_nodes_per_query": 1000},
            }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.total_time_ms / self.query_count if self.query_count > 0 else 0.0

        return {
            "query_count": self.query_count,
            "total_time_ms": self.total_time_ms,
            "average_time_ms": avg_time,
            "damping_factor": self.damping,
            "alpha_store_enabled": self.alpha_store is not None,
        }


# Factory functions


def create_ppr_retriever(
    hippo_index: HippoIndex,
    hypergraph_kg: HypergraphKG,
    alpha_store: AlphaProfileStore | None = None,
    config_path: str | None = None,
) -> PersonalizedPageRank:
    """Create a PersonalizedPageRank retriever with the given backends."""
    return PersonalizedPageRank(
        hippo_index=hippo_index,
        hypergraph=hypergraph_kg,
        alpha_store=alpha_store,
        config_path=config_path,
    )


def create_alpha_profile_store(redis_client=None) -> AlphaProfileStore:
    """Create an AlphaProfileStore for managing user personalization."""
    return AlphaProfileStore(redis_client=redis_client)
