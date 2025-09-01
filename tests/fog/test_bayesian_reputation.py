"""
Comprehensive Tests for Bayesian Reputation System

Tests the Beta distribution-based reputation management with:
- Uncertainty quantification
- Time-based decay
- Trust composition
- Integration with fog scheduling and pricing
"""

import pytest
import time

from infrastructure.fog.reputation import (
    BayesianReputationEngine,
    ReputationScore,
    ReputationEvent,
    ReputationConfig,
    TrustComposition,
    EventType,
    ReputationTier,
    integrate_with_scheduler,
    integrate_with_pricing,
    create_reputation_metrics,
)


class TestReputationScore:
    """Test ReputationScore calculations and properties"""

    def test_mean_score_calculation(self):
        """Test expected value calculation"""
        score = ReputationScore(node_id="test_node", alpha=3.0, beta=2.0, last_updated=time.time())

        # Expected value of Beta(3,2) = 3/(3+2) = 0.6
        assert abs(score.mean_score - 0.6) < 1e-6

    def test_uncertainty_calculation(self):
        """Test uncertainty/variance calculation"""
        score = ReputationScore(node_id="test_node", alpha=10.0, beta=5.0, last_updated=time.time())

        # Variance = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        expected_uncertainty = (10.0 * 5.0) / (15.0 * 15.0 * 16.0)
        assert abs(score.uncertainty - expected_uncertainty) < 1e-6

    def test_tier_assignment(self):
        """Test reputation tier assignment based on score and confidence"""

        # High score, high confidence
        high_score = ReputationScore(
            node_id="high_node", alpha=100.0, beta=5.0, last_updated=time.time()  # 95% success rate
        )
        tier = high_score.update_tier()
        assert tier in [ReputationTier.GOLD, ReputationTier.PLATINUM, ReputationTier.DIAMOND]

        # Low score
        low_score = ReputationScore(
            node_id="low_node", alpha=2.0, beta=3.0, last_updated=time.time()  # 40% success rate
        )
        tier = low_score.update_tier()
        assert tier in [ReputationTier.UNTRUSTED, ReputationTier.BRONZE]

        # High score but low sample size
        new_node = ReputationScore(
            node_id="new_node", alpha=3.0, beta=1.0, last_updated=time.time()  # 75% success rate but only 2 samples
        )
        tier = new_node.update_tier()
        assert tier in [ReputationTier.UNTRUSTED, ReputationTier.BRONZE]


class TestReputationEvent:
    """Test reputation event processing"""

    def test_positive_event_detection(self):
        """Test identification of positive events"""
        positive_event = ReputationEvent(node_id="test_node", event_type=EventType.TASK_SUCCESS, timestamp=time.time())
        assert positive_event.is_positive() is True

        negative_event = ReputationEvent(node_id="test_node", event_type=EventType.TASK_FAILURE, timestamp=time.time())
        assert negative_event.is_positive() is False


class TestBayesianReputationEngine:
    """Test main reputation engine functionality"""

    def setup_method(self):
        """Set up test environment"""
        self.engine = BayesianReputationEngine()
        self.node_id = "test_node_1"

    def test_initial_node_creation(self):
        """Test initial reputation score creation"""
        event = ReputationEvent(node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=time.time())

        self.engine.record_event(event)

        score = self.engine.get_reputation_score(self.node_id)
        assert score is not None
        assert score.node_id == self.node_id
        assert score.alpha > 1.0  # Should increase from prior

    def test_positive_event_recording(self):
        """Test recording positive events"""
        initial_time = time.time()

        # Record multiple positive events
        for i in range(5):
            event = ReputationEvent(
                node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=initial_time + i, quality_score=0.8
            )
            self.engine.record_event(event)

        score = self.engine.get_reputation_score(self.node_id)
        assert score.alpha > score.beta  # More successes than failures
        assert score.mean_score > 0.7  # High success rate

    def test_negative_event_recording(self):
        """Test recording negative events"""
        initial_time = time.time()

        # Record multiple negative events
        for i in range(5):
            event = ReputationEvent(node_id=self.node_id, event_type=EventType.TASK_FAILURE, timestamp=initial_time + i)
            self.engine.record_event(event)

        score = self.engine.get_reputation_score(self.node_id)
        assert score.beta > score.alpha  # More failures than successes
        assert score.mean_score < 0.4  # Low success rate

    def test_quality_score_weighting(self):
        """Test quality score influence on event weights"""
        initial_time = time.time()

        # High quality success
        high_quality_event = ReputationEvent(
            node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=initial_time, quality_score=0.9
        )
        self.engine.record_event(high_quality_event)

        score_after_high_quality = self.engine.get_reputation_score(self.node_id)
        initial_alpha = score_after_high_quality.alpha

        # Reset and test low quality
        self.engine = BayesianReputationEngine()

        low_quality_event = ReputationEvent(
            node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=initial_time, quality_score=0.1
        )
        self.engine.record_event(low_quality_event)

        score_after_low_quality = self.engine.get_reputation_score(self.node_id)

        # High quality should contribute more to alpha
        assert initial_alpha > score_after_low_quality.alpha

    def test_time_decay(self):
        """Test time-based reputation decay"""
        config = ReputationConfig(decay_rate=0.1)  # High decay rate for testing
        engine = BayesianReputationEngine(config)

        past_time = time.time() - (10 * 24 * 3600)  # 10 days ago
        current_time = time.time()

        # Record old positive event
        old_event = ReputationEvent(node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=past_time)
        engine.record_event(old_event)

        score_old = engine.get_reputation_score(self.node_id)
        initial_alpha = score_old.alpha

        # Check score after time decay
        score_decayed = engine.get_reputation_score(self.node_id, current_time)

        # Alpha should decay towards prior
        assert score_decayed.alpha < initial_alpha

    def test_trust_score_calculation(self):
        """Test trust score with uncertainty penalty"""
        # Create node with high reputation but high uncertainty
        for i in range(3):  # Few events = high uncertainty
            event = ReputationEvent(node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=time.time() + i)
            self.engine.record_event(event)

        score = self.engine.get_reputation_score(self.node_id)
        trust_score = self.engine.get_trust_score(self.node_id)

        # Trust score should be lower than raw mean score due to uncertainty
        assert trust_score < score.mean_score

    def test_node_ranking(self):
        """Test node ranking functionality"""
        nodes = ["node_a", "node_b", "node_c"]
        success_rates = [0.9, 0.7, 0.5]

        # Create nodes with different success rates
        for node_id, success_rate in zip(nodes, success_rates):
            for i in range(10):
                if i < success_rate * 10:
                    event_type = EventType.TASK_SUCCESS
                else:
                    event_type = EventType.TASK_FAILURE

                event = ReputationEvent(node_id=node_id, event_type=event_type, timestamp=time.time() + i)
                self.engine.record_event(event)

        ranking = self.engine.get_node_ranking()

        # Should be sorted by trust score (descending)
        assert len(ranking) == 3
        assert ranking[0][1] >= ranking[1][1] >= ranking[2][1]  # Trust scores descending

    def test_batch_metrics_update(self):
        """Test batch update from metrics data"""
        metrics_data = [
            {
                "node_id": "batch_node_1",
                "tasks_completed": 8,
                "tasks_failed": 2,
                "avg_quality_score": 0.85,
                "uptime_ratio": 0.95,
            },
            {"node_id": "batch_node_2", "tasks_completed": 3, "tasks_failed": 7, "uptime_ratio": 0.6},
        ]

        self.engine.batch_update_from_metrics(metrics_data)

        # Check first node (good performance)
        score1 = self.engine.get_reputation_score("batch_node_1")
        assert score1 is not None
        assert score1.mean_score > 0.7

        # Check second node (poor performance)
        score2 = self.engine.get_reputation_score("batch_node_2")
        assert score2 is not None
        assert score2.mean_score < 0.5

    def test_reputation_insights(self):
        """Test detailed reputation insights generation"""
        # Create some history
        events = [
            (EventType.TASK_SUCCESS, 0.8),
            (EventType.TASK_SUCCESS, 0.9),
            (EventType.TASK_FAILURE, None),
            (EventType.QUALITY_HIGH, 0.95),
        ]

        for i, (event_type, quality) in enumerate(events):
            event = ReputationEvent(
                node_id=self.node_id, event_type=event_type, timestamp=time.time() + i, quality_score=quality
            )
            self.engine.record_event(event)

        insights = self.engine.get_reputation_insights(self.node_id)

        assert insights["node_id"] == self.node_id
        assert "reputation_score" in insights
        assert "uncertainty" in insights
        assert "confidence_interval" in insights
        assert "event_distribution" in insights
        assert insights["total_events"] == len(events)

    def test_recommend_nodes_for_task(self):
        """Test node recommendation for tasks"""
        # Create nodes with different trust levels
        nodes = ["high_trust", "medium_trust", "low_trust"]

        for node_id in nodes:
            success_count = {"high_trust": 9, "medium_trust": 7, "low_trust": 3}[node_id]
            failure_count = 10 - success_count

            for i in range(success_count):
                event = ReputationEvent(node_id=node_id, event_type=EventType.TASK_SUCCESS, timestamp=time.time() + i)
                self.engine.record_event(event)

            for i in range(failure_count):
                event = ReputationEvent(
                    node_id=node_id, event_type=EventType.TASK_FAILURE, timestamp=time.time() + success_count + i
                )
                self.engine.record_event(event)

        # Recommend nodes with minimum trust threshold
        recommendations = self.engine.recommend_nodes_for_task(task_requirements={}, min_trust=0.6, max_nodes=2)

        assert len(recommendations) <= 2
        assert "high_trust" in recommendations
        assert "low_trust" not in recommendations

    def test_state_export_import(self):
        """Test reputation system state persistence"""
        # Create some reputation data
        event = ReputationEvent(node_id=self.node_id, event_type=EventType.TASK_SUCCESS, timestamp=time.time())
        self.engine.record_event(event)

        # Export state
        state = self.engine.export_state()

        assert "reputation_scores" in state
        assert "config" in state
        assert self.node_id in state["reputation_scores"]

        # Import into new engine
        new_engine = BayesianReputationEngine()
        new_engine.import_state(state)

        # Verify data was imported correctly
        imported_score = new_engine.get_reputation_score(self.node_id)
        original_score = self.engine.get_reputation_score(self.node_id)

        assert imported_score is not None
        assert abs(imported_score.alpha - original_score.alpha) < 1e-6
        assert abs(imported_score.beta - original_score.beta) < 1e-6


class TestTrustComposition:
    """Test trust composition across network tiers"""

    def setup_method(self):
        """Set up test environment"""
        self.config = ReputationConfig()
        self.trust_composition = TrustComposition(self.config)

    def test_direct_trust(self):
        """Test direct trust calculation"""
        trust_graph = {"node_a": {"node_b": 0.8, "node_c": 0.6}, "node_b": {"node_c": 0.9}}

        # Direct trust should return exact value
        trust = self.trust_composition.compute_transitive_trust("node_a", "node_b", trust_graph)
        assert trust == 0.8

    def test_transitive_trust(self):
        """Test transitive trust calculation"""
        trust_graph = {"node_a": {"node_b": 0.8}, "node_b": {"node_c": 0.9}, "node_c": {"node_d": 0.7}}

        # Transitive trust with decay
        trust = self.trust_composition.compute_transitive_trust("node_a", "node_c", trust_graph, max_hops=3)

        # Should be less than direct product due to decay
        assert trust > 0
        assert trust < 0.8 * 0.9

    def test_self_trust(self):
        """Test self-trust (should be 1.0)"""
        trust_graph = {"node_a": {"node_b": 0.5}}

        trust = self.trust_composition.compute_transitive_trust("node_a", "node_a", trust_graph)
        assert trust == 1.0

    def test_tier_trust_aggregation(self):
        """Test trust aggregation by tier"""
        node_scores = {
            "gold_1": ReputationScore("gold_1", 90, 10, time.time(), ReputationTier.GOLD),
            "gold_2": ReputationScore("gold_2", 85, 15, time.time(), ReputationTier.GOLD),
            "silver_1": ReputationScore("silver_1", 70, 30, time.time(), ReputationTier.SILVER),
        }

        tier_structure = {
            ReputationTier.GOLD: ["gold_1", "gold_2"],
            ReputationTier.SILVER: ["silver_1"],
            ReputationTier.BRONZE: [],
        }

        tier_trust = self.trust_composition.aggregate_tier_trust(node_scores, tier_structure)

        assert ReputationTier.GOLD in tier_trust
        assert ReputationTier.SILVER in tier_trust
        assert tier_trust[ReputationTier.GOLD] > tier_trust[ReputationTier.SILVER]
        assert tier_trust[ReputationTier.BRONZE] == 0.0


class TestIntegrationFunctions:
    """Test integration with scheduler and pricing"""

    def setup_method(self):
        """Set up test environment"""
        self.engine = BayesianReputationEngine()

        # Create some nodes with different reputations
        nodes = ["high_rep", "med_rep", "low_rep"]
        success_rates = [0.9, 0.7, 0.3]

        for node_id, success_rate in zip(nodes, success_rates):
            for i in range(10):
                event_type = EventType.TASK_SUCCESS if i < success_rate * 10 else EventType.TASK_FAILURE
                event = ReputationEvent(node_id=node_id, event_type=event_type, timestamp=time.time() + i)
                self.engine.record_event(event)

    def test_scheduler_integration(self):
        """Test integration with fog scheduler"""
        scheduler_config = {"available_nodes": ["high_rep", "med_rep", "low_rep"]}

        trust_scores = integrate_with_scheduler(self.engine, scheduler_config)

        assert len(trust_scores) == 3
        assert trust_scores["high_rep"] > trust_scores["med_rep"]
        assert trust_scores["med_rep"] > trust_scores["low_rep"]

    def test_pricing_integration(self):
        """Test integration with pricing system"""
        base_prices = {"high_rep": 1.0, "med_rep": 1.0, "low_rep": 1.0}

        adjusted_prices = integrate_with_pricing(self.engine, base_prices)

        assert len(adjusted_prices) == 3
        # High reputation should get higher prices (premium)
        assert adjusted_prices["high_rep"] > base_prices["high_rep"]
        # Low reputation should get lower prices
        assert adjusted_prices["low_rep"] < base_prices["low_rep"]

    def test_reputation_metrics(self):
        """Test reputation system metrics generation"""
        metrics = create_reputation_metrics(self.engine)

        assert "total_nodes" in metrics
        assert "tier_distribution" in metrics
        assert "high_trust_ratio" in metrics
        assert "system_trust_health" in metrics

        assert metrics["total_nodes"] > 0
        assert metrics["high_trust_ratio"] >= 0


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_engine(self):
        """Test operations on empty reputation engine"""
        engine = BayesianReputationEngine()

        # Non-existent node
        score = engine.get_reputation_score("nonexistent")
        assert score is None

        # Trust score for non-existent node
        trust = engine.get_trust_score("nonexistent")
        assert trust == 0.0

        # Ranking with no nodes
        ranking = engine.get_node_ranking()
        assert len(ranking) == 0

    def test_invalid_quality_scores(self):
        """Test handling of invalid quality scores"""
        engine = BayesianReputationEngine()

        # Quality score outside valid range
        event = ReputationEvent(
            node_id="test_node",
            event_type=EventType.TASK_SUCCESS,
            timestamp=time.time(),
            quality_score=1.5,  # Invalid: > 1.0
        )

        # Should handle gracefully
        engine.record_event(event)
        score = engine.get_reputation_score("test_node")
        assert score is not None

    def test_extreme_decay_rates(self):
        """Test extreme decay rates"""
        # Very high decay rate
        config = ReputationConfig(decay_rate=1.0)
        engine = BayesianReputationEngine(config)

        old_time = time.time() - (365 * 24 * 3600)  # 1 year ago
        event = ReputationEvent(node_id="old_node", event_type=EventType.TASK_SUCCESS, timestamp=old_time)

        engine.record_event(event)
        score = engine.get_reputation_score("old_node", time.time())

        # Should decay close to prior
        assert abs(score.alpha - config.prior_alpha) < 0.1
        assert abs(score.beta - config.prior_beta) < 0.1

    def test_fraud_detection_events(self):
        """Test handling of fraud detection events"""
        engine = BayesianReputationEngine()

        # Record fraud event (should heavily penalize)
        fraud_event = ReputationEvent(
            node_id="fraud_node",
            event_type=EventType.FRAUD_DETECTED,
            timestamp=time.time(),
            weight=3.0,  # High weight penalty
        )

        engine.record_event(fraud_event)
        score = engine.get_reputation_score("fraud_node")

        # Should have very low reputation
        assert score.mean_score < 0.3
        assert score.tier == ReputationTier.UNTRUSTED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
