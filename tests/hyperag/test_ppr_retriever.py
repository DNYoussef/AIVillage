"""
Unit Tests for PersonalizedPageRank Retriever

Test cases:
1. Baseline: PPR vs iterative recall parity (small KG)
2. α-fusion boost MAP ≥ 5%
3. Creative mode routes to DivergentRetriever
4. Performance targets: ≤ 150ms latency, < 10MB memory overhead
"""

import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp_servers.hyperag.memory.hippo_index import HippoIndex, HippoNode
from mcp_servers.hyperag.memory.hypergraph_kg import HypergraphKG, SemanticNode
from mcp_servers.hyperag.models import QueryPlan
from mcp_servers.hyperag.retrieval.ppr_retriever import (
    AlphaProfile,
    AlphaProfileStore,
    PersonalizedPageRank,
    PPRResults,
)


class TestPersonalizedPageRank:
    """Test suite for PersonalizedPageRank retriever"""

    @pytest.fixture
    async def mock_hippo_index(self):
        """Create mock HippoIndex for testing"""
        hippo = AsyncMock(spec=HippoIndex)

        # Mock recent nodes
        mock_nodes = [HippoNode(content=f"Recent node {i}", user_id="test_user") for i in range(5)]
        hippo.get_recent_nodes.return_value = mock_nodes

        # Mock vector similarity search
        mock_results = [(mock_nodes[0], 0.9), (mock_nodes[1], 0.8)]
        hippo.vector_similarity_search.return_value = mock_results

        return hippo

    @pytest.fixture
    async def mock_hypergraph_kg(self):
        """Create mock HypergraphKG for testing"""
        hypergraph = AsyncMock(spec=HypergraphKG)

        # Mock PageRank computation
        mock_ppr_scores = {f"node_{i}": 0.1 + (i * 0.05) for i in range(10)}
        hypergraph.personalized_pagerank.return_value = mock_ppr_scores

        # Mock semantic similarity search
        mock_semantic_nodes = [SemanticNode(content=f"Semantic node {i}", confidence=0.8) for i in range(3)]
        mock_semantic_results = [
            (mock_semantic_nodes[0], 0.85),
            (mock_semantic_nodes[1], 0.75),
        ]
        hypergraph.semantic_similarity_search.return_value = mock_semantic_results

        return hypergraph

    @pytest.fixture
    async def mock_alpha_store(self):
        """Create mock AlphaProfileStore for testing"""
        alpha_store = AsyncMock(spec=AlphaProfileStore)

        # Mock alpha profile
        mock_profile = AlphaProfile(
            user_id="test_user",
            relation_weights={"related_to": 1.5, "influences": 1.2, "causes": 0.8},
            last_updated=datetime.now(),
            confidence=0.9,
        )
        alpha_store.get_profile.return_value = mock_profile

        # Mock alpha scores
        mock_alpha_scores = {"node_0": 1.2, "node_1": 1.5, "node_2": 0.8}
        alpha_store.get_top_alpha.return_value = mock_alpha_scores

        return alpha_store

    @pytest.fixture
    async def ppr_retriever(self, mock_hippo_index, mock_hypergraph_kg, mock_alpha_store):
        """Create PersonalizedPageRank retriever for testing"""
        return PersonalizedPageRank(
            hippo_index=mock_hippo_index,
            hypergraph=mock_hypergraph_kg,
            alpha_store=mock_alpha_store,
            damping=0.85,
        )

    @pytest.fixture
    def sample_query_plan(self):
        """Create sample QueryPlan for testing"""
        plan = MagicMock(spec=QueryPlan)
        plan.mode = "NORMAL"
        plan.user_id = "test_user"
        plan.max_seeds = 10
        plan.seed_score_threshold = 0.5
        plan.confidence_hint = 0.8
        return plan

    async def test_basic_retrieval_pipeline(self, ppr_retriever, sample_query_plan):
        """Test basic retrieval pipeline without creative mode"""
        query_seeds = ["node_0", "node_1", "node_2"]
        user_id = "test_user"

        result = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id=user_id,
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Verify result structure
        assert isinstance(result, PPRResults)
        assert isinstance(result.nodes, list)
        assert isinstance(result.edges, list)
        assert isinstance(result.scores, dict)
        assert isinstance(result.reasoning_trace, list)
        assert result.query_time_ms > 0

        # Verify reasoning trace
        assert len(result.reasoning_trace) > 0
        assert "Starting PPR retrieval" in result.reasoning_trace[0]

        # Verify metadata
        assert result.metadata["query_seeds"] == query_seeds
        assert result.metadata["user_id"] == user_id
        assert "alpha_fusion" in result.metadata

    async def test_ppr_vs_iterative_recall_parity(self, ppr_retriever, sample_query_plan):
        """Test baseline: PPR vs iterative recall parity on small KG"""
        query_seeds = ["node_0", "node_1"]

        # Run PPR retrieval
        ppr_result = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Simulate iterative retrieval (simplified)
        iterative_scores = {}
        for seed in query_seeds:
            iterative_scores[seed] = 1.0
            # Add neighbors with decay
            for i in range(3):
                neighbor_id = f"{seed}_neighbor_{i}"
                iterative_scores[neighbor_id] = 1.0 / (i + 2)

        # Compare coverage (should have similar nodes)
        ppr_nodes = set(ppr_result.scores.keys())
        iterative_nodes = set(iterative_scores.keys())

        # Calculate recall (intersection over iterative set)
        recall = len(ppr_nodes.intersection(iterative_nodes)) / len(iterative_nodes)

        # Should have reasonable recall (>= 0.5 for this test)
        assert recall >= 0.5, f"PPR recall {recall:.3f} below threshold"

    async def test_alpha_fusion_map_boost(self, ppr_retriever, sample_query_plan):
        """Test α-fusion boost MAP ≥ 5%"""
        query_seeds = ["node_0", "node_1", "node_2"]

        # Test with α-fusion
        result_with_alpha = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",  # Triggers α-fusion
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Test without α-fusion
        result_without_alpha = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id=None,  # No α-fusion
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Compare scores (simplified MAP calculation)
        def calculate_map(scores, relevant_nodes):
            """Simplified MAP calculation"""
            if not scores or not relevant_nodes:
                return 0.0

            sorted_nodes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            relevant_set = set(relevant_nodes)

            precision_sum = 0.0
            relevant_count = 0

            for i, node in enumerate(sorted_nodes):
                if node in relevant_set:
                    relevant_count += 1
                    precision_sum += relevant_count / (i + 1)

            return precision_sum / len(relevant_set) if relevant_set else 0.0

        # Define relevant nodes (nodes that should be boosted by α-fusion)
        relevant_nodes = ["node_0", "node_1", "node_2"]  # Seeds should be boosted

        map_with_alpha = calculate_map(result_with_alpha.scores, relevant_nodes)
        map_without_alpha = calculate_map(result_without_alpha.scores, relevant_nodes)

        # Calculate improvement
        if map_without_alpha > 0:
            improvement = (map_with_alpha - map_without_alpha) / map_without_alpha
            assert improvement >= 0.05, f"α-fusion MAP improvement {improvement:.3f} below 5% threshold"
        else:
            # If baseline MAP is 0, just check that α-fusion gives positive MAP
            assert map_with_alpha > 0, "α-fusion should produce positive MAP when baseline is 0"

    async def test_creative_mode_routing(self, ppr_retriever, sample_query_plan):
        """Test creative mode routes to DivergentRetriever"""
        query_seeds = ["node_0", "node_1"]

        # Mock DivergentRetriever
        with patch("mcp_servers.hyperag.retrieval.ppr_retriever.DivergentRetriever") as mock_divergent:
            mock_instance = AsyncMock()
            mock_creative_result = PPRResults(
                nodes=[{"id": "creative_node", "score": 0.9}],
                edges=[],
                scores={"creative_node": 0.9},
                reasoning_trace=["Creative retrieval executed"],
                query_time_ms=50.0,
                metadata={"creative_mode": True},
            )
            mock_instance.retrieve_creative.return_value = mock_creative_result
            mock_divergent.return_value = mock_instance

            # Test creative mode
            result = await ppr_retriever.retrieve(
                query_seeds=query_seeds,
                user_id="test_user",
                plan=sample_query_plan,
                creative_mode=True,
            )

            # Verify creative mode was triggered
            assert "DivergentRetriever" in str(result.reasoning_trace)
            mock_divergent.assert_called_once()
            mock_instance.retrieve_creative.assert_called_once()

    async def test_creative_mode_fallback(self, ppr_retriever, sample_query_plan):
        """Test creative mode fallback when DivergentRetriever not available"""
        query_seeds = ["node_0", "node_1"]

        # Test creative mode with ImportError (DivergentRetriever not available)
        result = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=True,
        )

        # Should fall back to standard retrieval
        assert "not available" in str(result.reasoning_trace) or "falling back" in str(result.reasoning_trace)
        assert len(result.scores) > 0  # Should still return results

    async def test_performance_latency_target(self, ppr_retriever, sample_query_plan):
        """Test performance target: ≤ 150ms latency"""
        query_seeds = ["node_0", "node_1", "node_2"]

        # Warm up
        await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Measure latency
        start_time = time.time()
        result = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )
        actual_latency = (time.time() - start_time) * 1000

        # Check both actual and reported latency
        assert actual_latency <= 150, f"Actual latency {actual_latency:.2f}ms exceeds 150ms target"
        assert result.query_time_ms <= 150, f"Reported latency {result.query_time_ms:.2f}ms exceeds 150ms target"

    async def test_memory_overhead_limit(self, ppr_retriever, sample_query_plan):
        """Test performance target: < 10MB memory overhead"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        query_seeds = [f"node_{i}" for i in range(100)]  # Larger query

        # Run retrieval
        await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )

        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_overhead = current_memory - baseline_memory

        assert memory_overhead < 10, f"Memory overhead {memory_overhead:.2f}MB exceeds 10MB limit"

    async def test_uncertainty_weighting(self, ppr_retriever, sample_query_plan):
        """Test uncertainty weighting and pruning"""
        query_seeds = ["node_0", "node_1"]

        result = await ppr_retriever.retrieve(
            query_seeds=query_seeds,
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Verify uncertainty is applied
        for node in result.nodes:
            assert "uncertainty" in node
            assert "confidence" in node
            assert 0.0 <= node["uncertainty"] <= 1.0
            assert 0.0 <= node["confidence"] <= 1.0

        for edge in result.edges:
            assert "uncertainty" in edge
            assert "confidence" in edge

    async def test_empty_seeds_handling(self, ppr_retriever, sample_query_plan):
        """Test handling of empty query seeds"""
        result = await ppr_retriever.retrieve(
            query_seeds=[],
            user_id="test_user",
            plan=sample_query_plan,
            creative_mode=False,
        )

        # Should handle gracefully
        assert isinstance(result, PPRResults)
        assert "error" not in result.metadata or result.total_results == 0

    async def test_config_loading(self):
        """Test configuration loading and defaults"""
        # Test with missing config file
        ppr = PersonalizedPageRank(
            hippo_index=AsyncMock(),
            hypergraph=AsyncMock(),
            config_path="/nonexistent/path.yaml",
        )

        # Should fall back to defaults
        assert ppr.damping == 0.85  # Default damping
        assert ppr.config is not None

    async def test_performance_stats(self, ppr_retriever, sample_query_plan):
        """Test performance statistics tracking"""
        initial_stats = ppr_retriever.get_performance_stats()
        assert initial_stats["query_count"] == 0

        # Run some queries
        for i in range(3):
            await ppr_retriever.retrieve(
                query_seeds=[f"node_{i}"],
                user_id="test_user",
                plan=sample_query_plan,
                creative_mode=False,
            )

        final_stats = ppr_retriever.get_performance_stats()
        assert final_stats["query_count"] == 3
        assert final_stats["total_time_ms"] > 0
        assert final_stats["average_time_ms"] > 0


class TestAlphaProfileStore:
    """Test suite for AlphaProfileStore"""

    @pytest.fixture
    async def alpha_store(self):
        """Create AlphaProfileStore for testing"""
        mock_redis = AsyncMock()
        return AlphaProfileStore(redis_client=mock_redis)

    async def test_profile_creation_and_retrieval(self, alpha_store):
        """Test creating and retrieving alpha profiles"""
        profile = AlphaProfile(
            user_id="test_user",
            relation_weights={"test_relation": 1.5},
            last_updated=datetime.now(),
        )

        # Store profile
        success = await alpha_store.update_profile(profile)
        assert success

        # Retrieve profile
        retrieved = await alpha_store.get_profile("test_user")
        assert retrieved is not None
        assert retrieved.user_id == "test_user"
        assert retrieved.get_weight("test_relation") == 1.5
        assert retrieved.get_weight("unknown_relation", 2.0) == 2.0  # Default

    async def test_profile_caching(self, alpha_store):
        """Test profile caching behavior"""
        profile = AlphaProfile(
            user_id="cached_user",
            relation_weights={"cached_relation": 0.8},
            last_updated=datetime.now(),
        )

        # Store in cache
        alpha_store.profiles_cache["cached_user"] = profile

        # Should retrieve from cache
        retrieved = await alpha_store.get_profile("cached_user")
        assert retrieved.user_id == "cached_user"
        assert retrieved.get_weight("cached_relation") == 0.8


# Performance benchmarks


@pytest.mark.benchmark
class TestPPRPerformanceBenchmarks:
    """Performance benchmarks for PPR retriever"""

    async def test_large_graph_performance(self):
        """Benchmark performance on larger graph"""
        # This would test with a larger mock graph
        # For now, just verify the test structure
        assert True

    async def test_concurrent_queries(self):
        """Benchmark concurrent query handling"""
        # This would test multiple concurrent queries
        # For now, just verify the test structure
        assert True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
