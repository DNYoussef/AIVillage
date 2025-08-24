"""
Unit tests for Query Classification System
"""

from pathlib import Path
import sys

from mcp_servers.hyperag.planning.plan_structures import QueryType, ReasoningStrategy
from mcp_servers.hyperag.planning.query_classifier import QueryClassifier
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


class TestQueryClassifier:
    """Test suite for QueryClassifier"""

    @pytest.fixture
    def classifier(self):
        """Create QueryClassifier instance"""
        return QueryClassifier()

    def test_classifier_initialization(self, classifier):
        """Test classifier initialization"""
        assert classifier is not None
        assert len(classifier.temporal_patterns) > 0
        assert len(classifier.causal_patterns) > 0
        assert len(classifier.comparative_patterns) > 0
        assert len(classifier.meta_patterns) > 0
        assert len(classifier.aggregation_patterns) > 0

    def test_simple_fact_classification(self, classifier):
        """Test classification of simple fact queries"""
        simple_queries = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "Define machine learning",
            "Paris population",
        ]

        for query in simple_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            # Should classify as simple fact or have low complexity
            assert query_type in [QueryType.SIMPLE_FACT, QueryType.META_KNOWLEDGE]
            assert confidence > 0.5
            assert analysis["complexity_score"] < 0.7

    def test_temporal_classification(self, classifier):
        """Test classification of temporal queries"""
        temporal_queries = [
            "What happened before the Civil War?",
            "When was the first computer invented?",
            "Show me the timeline of World War II",
            "What events occurred after 1950?",
            "During the Renaissance, what art movements emerged?",
        ]

        for query in temporal_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.TEMPORAL_ANALYSIS
            assert confidence > 0.6
            assert "temporal" in analysis["pattern_matches"]
            assert analysis["pattern_matches"]["temporal"] > 0

    def test_causal_classification(self, classifier):
        """Test classification of causal queries"""
        causal_queries = [
            "Why did the stock market crash?",
            "What causes global warming?",
            "How does smoking lead to cancer?",
            "What are the reasons for inflation?",
            "Explain the causes of the French Revolution",
        ]

        for query in causal_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.CAUSAL_CHAIN
            assert confidence > 0.6
            assert "causal" in analysis["pattern_matches"]
            assert analysis["pattern_matches"]["causal"] > 0

    def test_comparative_classification(self, classifier):
        """Test classification of comparative queries"""
        comparative_queries = [
            "Compare Python and Java programming languages",
            "What's the difference between cats and dogs?",
            "Which is better: Mac or PC?",
            "How do electric cars compare to gas cars?",
            "Contrast democracy versus authoritarianism",
        ]

        for query in comparative_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.COMPARATIVE
            assert confidence > 0.6
            assert "comparative" in analysis["pattern_matches"]
            assert analysis["pattern_matches"]["comparative"] > 0

    def test_meta_knowledge_classification(self, classifier):
        """Test classification of meta-knowledge queries"""
        meta_queries = [
            "What do you know about artificial intelligence?",
            "Tell me about quantum physics",
            "How confident are you about this information?",
            "What are your sources for this data?",
            "Summarize the key facts about climate change",
        ]

        for query in meta_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.META_KNOWLEDGE
            assert confidence > 0.6
            assert "meta" in analysis["pattern_matches"]

    def test_aggregation_classification(self, classifier):
        """Test classification of aggregation queries"""
        aggregation_queries = [
            "How many countries are in Europe?",
            "List all Nobel Prize winners in Physics",
            "What's the average temperature in July?",
            "Count the number of programming languages",
            "Show me all movies from 2020",
        ]

        for query in aggregation_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.AGGREGATION
            assert confidence > 0.6
            assert "aggregation" in analysis["pattern_matches"]

    def test_hypothetical_classification(self, classifier):
        """Test classification of hypothetical queries"""
        hypothetical_queries = [
            "What if the internet never existed?",
            "Suppose we could travel faster than light",
            "Imagine if AI becomes sentient",
            "What would happen if the polar ice caps melted?",
            "Predict the future of renewable energy",
        ]

        for query in hypothetical_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            assert query_type == QueryType.HYPOTHETICAL
            assert confidence > 0.6
            assert "hypothetical" in analysis["pattern_matches"]

    def test_multi_hop_classification(self, classifier):
        """Test classification of multi-hop queries"""
        multihop_queries = [
            "How are climate change and economic policy related through environmental regulations?",
            "What's the connection between social media and political polarization via echo chambers?",
            "Trace the relationship between artificial intelligence and job displacement through automation",
            "How does education quality affect economic growth through innovation and productivity?",
        ]

        for query in multihop_queries:
            query_type, confidence, analysis = classifier.classify_query(query)

            # Should be classified as multi-hop or have high complexity
            assert query_type in [
                QueryType.MULTI_HOP,
                QueryType.CAUSAL_CHAIN,
                QueryType.COMPARATIVE,
            ]
            assert analysis["complexity_score"] > 0.6

    def test_complexity_calculation(self, classifier):
        """Test complexity score calculation"""

        # Simple query
        simple_query = "What is 2+2?"
        _, _, analysis = classifier.classify_query(simple_query)
        simple_complexity = analysis["complexity_score"]

        # Complex query
        complex_query = (
            "How do socioeconomic factors, environmental conditions, and government policies "
            "interact to influence public health outcomes, and what are the long-term implications "
            "for healthcare systems and social inequality?"
        )
        _, _, analysis = classifier.classify_query(complex_query)
        complex_complexity = analysis["complexity_score"]

        assert complex_complexity > simple_complexity
        assert simple_complexity < 0.5
        assert complex_complexity > 0.7

    def test_strategy_suggestion(self, classifier):
        """Test strategy suggestion based on query type"""

        test_cases = [
            (QueryType.SIMPLE_FACT, 0.2, ReasoningStrategy.DIRECT_RETRIEVAL),
            (QueryType.TEMPORAL_ANALYSIS, 0.5, ReasoningStrategy.TEMPORAL_REASONING),
            (QueryType.CAUSAL_CHAIN, 0.6, ReasoningStrategy.CAUSAL_REASONING),
            (QueryType.COMPARATIVE, 0.5, ReasoningStrategy.COMPARATIVE_ANALYSIS),
            (QueryType.META_KNOWLEDGE, 0.4, ReasoningStrategy.META_REASONING),
            (QueryType.MULTI_HOP, 0.8, ReasoningStrategy.STEP_BY_STEP),
            (QueryType.AGGREGATION, 0.6, ReasoningStrategy.GRAPH_TRAVERSAL),
        ]

        for query_type, complexity, expected_strategy in test_cases:
            strategy = classifier.suggest_strategy(query_type, complexity)

            if complexity > 0.7:
                # High complexity should suggest hybrid
                assert strategy == ReasoningStrategy.HYBRID
            else:
                assert strategy == expected_strategy

    def test_pattern_scoring(self, classifier):
        """Test pattern scoring functionality"""

        # Test temporal patterns
        temporal_text = "when did this happen before the war during 1945"
        temporal_score = classifier._check_temporal_patterns(temporal_text)
        assert temporal_score > 0

        # Test causal patterns
        causal_text = "because of the cause this leads to the effect due to reasons"
        causal_score = classifier._check_causal_patterns(causal_text)
        assert causal_score > 0

        # Test no patterns
        neutral_text = "hello world testing"
        temporal_score_neutral = classifier._check_temporal_patterns(neutral_text)
        assert temporal_score_neutral == 0

    def test_reasoning_hints(self, classifier):
        """Test generation of reasoning hints"""

        # Test temporal hints
        hints = classifier.get_reasoning_hints("When did this happen?", QueryType.TEMPORAL_ANALYSIS)
        assert len(hints) > 0
        assert any("temporal" in hint.lower() for hint in hints)

        # Test causal hints
        hints = classifier.get_reasoning_hints("Why did this happen?", QueryType.CAUSAL_CHAIN)
        assert len(hints) > 0
        assert any("causal" in hint.lower() for hint in hints)

        # Test comparative hints
        hints = classifier.get_reasoning_hints("Compare A and B", QueryType.COMPARATIVE)
        assert len(hints) > 0
        assert any("compar" in hint.lower() for hint in hints)

    def test_edge_cases(self, classifier):
        """Test edge cases and error handling"""

        # Empty query
        query_type, confidence, analysis = classifier.classify_query("")
        assert query_type == QueryType.SIMPLE_FACT
        assert confidence > 0

        # Very short query
        query_type, confidence, analysis = classifier.classify_query("Hi")
        assert query_type == QueryType.SIMPLE_FACT

        # Very long query
        long_query = "word " * 100
        query_type, confidence, analysis = classifier.classify_query(long_query)
        assert analysis["complexity_score"] > 0.5

        # Special characters
        special_query = "What is @#$%^&*()?"
        query_type, confidence, analysis = classifier.classify_query(special_query)
        assert query_type in QueryType
        assert confidence > 0

    def test_classification_consistency(self, classifier):
        """Test that classification is consistent for same queries"""

        query = "What causes climate change and how does it affect weather patterns?"

        # Run classification multiple times
        results = []
        for _ in range(5):
            query_type, confidence, analysis = classifier.classify_query(query)
            results.append((query_type, confidence))

        # All results should be the same
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0]  # Same query type
            assert abs(result[1] - first_result[1]) < 0.01  # Same confidence (allowing for floating point)

    def test_mixed_query_types(self, classifier):
        """Test queries that could match multiple types"""

        # Query with both temporal and causal elements
        mixed_query = "Why did the stock market crash happen in 1929 and what were the causes?"
        query_type, confidence, analysis = classifier.classify_query(mixed_query)

        # Should identify multiple patterns
        pattern_matches = analysis["pattern_matches"]
        assert len(pattern_matches) >= 2
        assert "temporal" in pattern_matches or "causal" in pattern_matches

        # Should have reasonable confidence
        assert confidence > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
