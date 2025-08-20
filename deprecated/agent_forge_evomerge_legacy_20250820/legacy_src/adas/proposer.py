"""
ADAS Proposer for generating expert configurations.
Uses meta-search to propose diverse ExpertSpec + DispatchSpec pairs.
"""

import json
import logging
import random
from typing import Any

import numpy as np

from .archive import ADASArchive, ExperimentResult

logger = logging.getLogger(__name__)


class ADASProposer:
    """
    Generates diverse expert configurations using evolutionary and rule-based approaches.
    Can also interface with LLM planners for more sophisticated proposal generation.
    """

    def __init__(
        self,
        archive: ADASArchive | None = None,
        use_llm: bool = False,
        model_name: str = "gpt-4",
    ):
        self.archive = archive
        self.use_llm = use_llm
        self.model_name = model_name

        # Configuration space definitions
        self.layer_options = [
            ["attn_qkv"],
            ["mlp"],
            ["block_0"],
            ["block_6"],
            ["block_12"],
            ["attn_qkv", "mlp"],
            ["attn_qkv", "block_12"],
            ["mlp", "block_6"],
            ["block_0", "block_6", "block_12"],
        ]

        self.rank_options = [1, 2, 4, 8, 16]
        self.svd_scope_options = ["per-matrix", "per-block"]
        self.init_options = ["random", "pca_activations", "fisher"]
        self.activation_rule_options = ["always", "gated"]

        self.feature_combinations = [
            ["prompt_stats"],
            ["logits_entropy"],
            ["activation_sketch"],
            ["prompt_stats", "logits_entropy"],
            ["prompt_stats", "activation_sketch"],
            ["logits_entropy", "activation_sketch"],
            ["prompt_stats", "logits_entropy", "activation_sketch"],
        ]

        self.mix_fn_options = ["linear", "softmax", "energy"]
        self.granularity_options = ["sequence", "segment", "token"]

    def propose(
        self,
        n_proposals: int = 8,
        target_latency_ms: int = 100,
        previous_results: list[ExperimentResult] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate n_proposals diverse expert configurations.

        Args:
            n_proposals: Number of proposals to generate
            target_latency_ms: Latency budget constraint
            previous_results: Previous experimental results for informed search

        Returns:
            List of proposal dictionaries with expert, dispatch, and motivation
        """
        if self.use_llm and previous_results:
            return self._propose_with_llm(n_proposals, target_latency_ms, previous_results)
        else:
            return self._propose_heuristic(n_proposals, target_latency_ms, previous_results)

    def _propose_heuristic(
        self,
        n_proposals: int,
        target_latency_ms: int,
        previous_results: list[ExperimentResult] | None,
    ) -> list[dict[str, Any]]:
        """Generate proposals using heuristic rules and evolutionary mutations."""
        proposals = []

        # Strategy 1: Random baseline proposals (25%)
        n_random = max(1, n_proposals // 4)
        for _ in range(n_random):
            proposals.append(self._generate_random_proposal(target_latency_ms))

        # Strategy 2: Template-based proposals (25%)
        n_template = max(1, n_proposals // 4)
        for _ in range(n_template):
            proposals.append(self._generate_template_proposal(target_latency_ms))

        # Strategy 3: Archive-guided mutations (25%)
        n_archive = max(1, n_proposals // 4)
        if previous_results:
            for _ in range(n_archive):
                proposals.append(self._mutate_from_archive(previous_results, target_latency_ms))
        else:
            # Fallback to template if no archive
            for _ in range(n_archive):
                proposals.append(self._generate_template_proposal(target_latency_ms))

        # Strategy 4: Fill remaining with diverse sampling
        remaining = n_proposals - len(proposals)
        for _ in range(remaining):
            if random.random() < 0.5:
                proposals.append(self._generate_diverse_proposal(target_latency_ms, proposals))
            else:
                proposals.append(self._generate_random_proposal(target_latency_ms))

        return proposals[:n_proposals]

    def _generate_random_proposal(self, target_latency_ms: int) -> dict[str, Any]:
        """Generate a completely random proposal."""
        # Adjust rank based on latency budget
        max_rank = 16 if target_latency_ms > 200 else 8 if target_latency_ms > 100 else 4

        expert = {
            "layers": random.choice(self.layer_options),
            "rank": random.choice([r for r in self.rank_options if r <= max_rank]),
            "svd_scope": random.choice(self.svd_scope_options),
            "init": random.choice(self.init_options),
            "activation_rule": random.choice(self.activation_rule_options),
            "budget": {
                "max_active": random.randint(1, min(4, len(self.layer_options))),
                "max_latency_ms": target_latency_ms,
            },
        }

        dispatch = {
            "features": random.choice(self.feature_combinations),
            "mix_fn": random.choice(self.mix_fn_options),
            "granularity": random.choice(self.granularity_options),
        }

        return {
            "expert": expert,
            "dispatch": dispatch,
            "motivation": "Random baseline configuration",
        }

    def _generate_template_proposal(self, target_latency_ms: int) -> dict[str, Any]:
        """Generate proposal from proven templates."""
        templates = [
            # Template 1: Lightweight attention specialization
            {
                "expert": {
                    "layers": ["attn_qkv"],
                    "rank": 2,
                    "svd_scope": "per-matrix",
                    "init": "pca_activations",
                    "activation_rule": "gated",
                    "budget": {
                        "max_active": 2,
                        "max_latency_ms": min(target_latency_ms, 50),
                    },
                },
                "dispatch": {
                    "features": ["prompt_stats", "activation_sketch"],
                    "mix_fn": "softmax",
                    "granularity": "sequence",
                },
                "motivation": "Lightweight attention specialization with gating",
            },
            # Template 2: Deep MLP specialization
            {
                "expert": {
                    "layers": ["mlp", "block_12"],
                    "rank": 4,
                    "svd_scope": "per-block",
                    "init": "fisher",
                    "activation_rule": "always",
                    "budget": {
                        "max_active": 3,
                        "max_latency_ms": min(target_latency_ms, 80),
                    },
                },
                "dispatch": {
                    "features": ["logits_entropy", "activation_sketch"],
                    "mix_fn": "energy",
                    "granularity": "token",
                },
                "motivation": "Deep MLP specialization with Fisher initialization",
            },
            # Template 3: Multi-layer low-rank
            {
                "expert": {
                    "layers": ["attn_qkv", "mlp"],
                    "rank": 1,
                    "svd_scope": "per-matrix",
                    "init": "random",
                    "activation_rule": "gated",
                    "budget": {
                        "max_active": 4,
                        "max_latency_ms": min(target_latency_ms, 30),
                    },
                },
                "dispatch": {
                    "features": ["prompt_stats"],
                    "mix_fn": "linear",
                    "granularity": "sequence",
                },
                "motivation": "Ultra-lightweight multi-layer adaptation",
            },
        ]

        # Select template and add small variations
        template = random.choice(templates).copy()

        # Add small mutations
        if random.random() < 0.3:  # 30% chance of rank mutation
            current_rank = template["expert"]["rank"]
            new_rank = random.choice([max(1, current_rank - 1), current_rank, min(8, current_rank + 1)])
            template["expert"]["rank"] = new_rank

        if random.random() < 0.2:  # 20% chance of feature mutation
            template["dispatch"]["features"] = random.choice(self.feature_combinations)

        return template

    def _mutate_from_archive(self, previous_results: list[ExperimentResult], target_latency_ms: int) -> dict[str, Any]:
        """Generate proposal by mutating successful configurations from archive."""
        # Select parent configuration (prefer successful, high-scoring ones)
        successful = [r for r in previous_results if r.success and r.score > 0]
        if not successful:
            return self._generate_random_proposal(target_latency_ms)

        # Weight selection by score
        weights = np.array([r.score for r in successful])
        weights = weights / weights.sum()
        parent = np.random.choice(successful, p=weights)

        # Start with parent configuration
        expert = parent.expert_spec.copy()
        dispatch = parent.dispatch_spec.copy()

        # Apply mutations
        mutations_applied = []

        # Rank mutation (40% chance)
        if random.random() < 0.4:
            old_rank = expert.get("rank", 2)
            # Smart rank adjustment based on performance
            if parent.latency_ms > target_latency_ms * 0.8:  # High latency -> reduce rank
                new_rank = max(1, old_rank - 1)
            else:  # Low latency -> can increase rank
                new_rank = min(16, old_rank + random.choice([1, 2]))
            expert["rank"] = new_rank
            mutations_applied.append(f"rank: {old_rank}->{new_rank}")

        # Layer mutation (30% chance)
        if random.random() < 0.3:
            expert["layers"] = random.choice(self.layer_options)
            mutations_applied.append(f"layers: {expert['layers']}")

        # Initialization mutation (20% chance)
        if random.random() < 0.2:
            expert["init"] = random.choice(self.init_options)
            mutations_applied.append(f"init: {expert['init']}")

        # Dispatch feature mutation (25% chance)
        if random.random() < 0.25:
            dispatch["features"] = random.choice(self.feature_combinations)
            mutations_applied.append(f"features: {dispatch['features']}")

        # Update budget
        expert["budget"] = {
            "max_active": random.randint(1, 4),
            "max_latency_ms": target_latency_ms,
        }

        motivation = f"Mutation from score={parent.score:.3f} config: {', '.join(mutations_applied)}"

        return {"expert": expert, "dispatch": dispatch, "motivation": motivation}

    def _generate_diverse_proposal(
        self, target_latency_ms: int, existing_proposals: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Generate proposal that maximizes diversity from existing ones."""
        # Extract features from existing proposals for diversity computation
        existing_features = []
        for prop in existing_proposals:
            features = (
                tuple(sorted(prop["expert"]["layers"])),
                prop["expert"]["rank"],
                prop["expert"]["init"],
                tuple(sorted(prop["dispatch"]["features"])),
                prop["dispatch"]["granularity"],
            )
            existing_features.append(features)

        # Generate candidates and pick most diverse
        candidates = []
        for _ in range(20):  # Generate 20 candidates
            candidate = self._generate_random_proposal(target_latency_ms)
            candidate_features = (
                tuple(sorted(candidate["expert"]["layers"])),
                candidate["expert"]["rank"],
                candidate["expert"]["init"],
                tuple(sorted(candidate["dispatch"]["features"])),
                candidate["dispatch"]["granularity"],
            )

            # Compute diversity score (minimum distance to existing)
            if existing_features:
                diversity = min(self._feature_distance(candidate_features, existing) for existing in existing_features)
            else:
                diversity = 1.0

            candidates.append((candidate, diversity))

        # Select most diverse candidate
        best_candidate = max(candidates, key=lambda x: x[1])[0]
        best_candidate["motivation"] = "Diversity-maximizing configuration"

        return best_candidate

    def _feature_distance(self, features_a: tuple, features_b: tuple) -> float:
        """Compute distance between two feature tuples."""
        distance = 0.0

        # Layer overlap
        layers_a, layers_b = set(features_a[0]), set(features_b[0])
        layer_jaccard = len(layers_a & layers_b) / len(layers_a | layers_b) if layers_a | layers_b else 0
        distance += 1 - layer_jaccard

        # Rank difference
        rank_diff = abs(features_a[1] - features_b[1]) / 16  # Normalize by max rank
        distance += rank_diff

        # Categorical features (init, granularity)
        distance += 0 if features_a[2] == features_b[2] else 1  # init
        distance += 0 if features_a[4] == features_b[4] else 1  # granularity

        # Feature overlap
        feat_a, feat_b = set(features_a[3]), set(features_b[3])
        feat_jaccard = len(feat_a & feat_b) / len(feat_a | feat_b) if feat_a | feat_b else 0
        distance += 1 - feat_jaccard

        return distance / 5  # Normalize by number of components

    def _propose_with_llm(
        self,
        n_proposals: int,
        target_latency_ms: int,
        previous_results: list[ExperimentResult],
    ) -> list[dict[str, Any]]:
        """Generate proposals using LLM planner (requires API access)."""
        # This would integrate with an LLM API to generate proposals
        # For now, fall back to heuristic approach
        logger.warning("LLM proposal generation not implemented, using heuristic approach")
        return self._propose_heuristic(n_proposals, target_latency_ms, previous_results)

    def get_search_statistics(self) -> dict[str, Any]:
        """Get statistics about the search space and proposal generation."""
        return {
            "layer_combinations": len(self.layer_options),
            "rank_options": len(self.rank_options),
            "init_methods": len(self.init_options),
            "feature_combinations": len(self.feature_combinations),
            "total_expert_configs": (
                len(self.layer_options)
                * len(self.rank_options)
                * len(self.svd_scope_options)
                * len(self.init_options)
                * len(self.activation_rule_options)
            ),
            "total_dispatch_configs": (
                len(self.feature_combinations) * len(self.mix_fn_options) * len(self.granularity_options)
            ),
        }


def create_llm_proposal_prompt(
    target_latency_ms: int,
    previous_results: list[ExperimentResult],
    n_proposals: int = 8,
) -> str:
    """Create prompt for LLM-based proposal generation."""

    # Summarize previous results
    if previous_results:
        successful = [r for r in previous_results if r.success]
        if successful:
            best_score = max(r.score for r in successful)
            avg_latency = sum(r.latency_ms for r in successful) / len(successful)
            results_summary = f"Best score: {best_score:.3f}, Avg latency: {avg_latency:.1f}ms, Success rate: {len(successful)}/{len(previous_results)}"
        else:
            results_summary = "No successful results yet - all proposals failed"
    else:
        results_summary = "No previous results available"

    # Get schemas
    from .archive import DISPATCH_SPEC_SCHEMA, EXPERT_SPEC_SCHEMA

    schema_str = json.dumps(
        {"ExpertSpec": EXPERT_SPEC_SCHEMA, "DispatchSpec": DISPATCH_SPEC_SCHEMA},
        indent=2,
    )

    prompt = f"""Given RESULTS_SUMMARY and the SCHEMA below, propose N={n_proposals} diverse ExpertSpec + DispatchSpec pairs that:
- stay within latency budget (≤ {target_latency_ms}ms),
- diversify layer scopes and ranks,
- include at least 2 token-granularity dispatchers,
- include at least 2 PCA-initialized singular directions.

Return JSON only, list[{{"expert": {{...}}, "dispatch": {{...}}, "motivation": "..."}}.
Prefer small max_active experts (≤ 4). Use fisher init for deep blocks when latency is tight.

SCHEMA = {schema_str}

TARGET_LATENCY_MS = {target_latency_ms}
RESULTS_SUMMARY = {results_summary}
"""

    return prompt
