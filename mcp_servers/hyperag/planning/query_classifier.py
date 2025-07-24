"""
Query Classification System

Analyzes queries to determine reasoning requirements and appropriate strategies.
Uses pattern matching, NLP analysis, and machine learning for classification.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .plan_structures import QueryType, ReasoningStrategy

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies queries by analyzing structure, intent, and reasoning requirements.
    Supports temporal, causal, comparative, and meta-reasoning detection.
    """

    def __init__(self):
        self.temporal_patterns = self._build_temporal_patterns()
        self.causal_patterns = self._build_causal_patterns()
        self.comparative_patterns = self._build_comparative_patterns()
        self.meta_patterns = self._build_meta_patterns()
        self.aggregation_patterns = self._build_aggregation_patterns()
        self.complexity_indicators = self._build_complexity_indicators()

        # Classification weights (could be learned)
        self.pattern_weights = {
            QueryType.TEMPORAL_ANALYSIS: 0.8,
            QueryType.CAUSAL_CHAIN: 0.9,
            QueryType.COMPARATIVE: 0.7,
            QueryType.META_KNOWLEDGE: 0.85,
            QueryType.AGGREGATION: 0.75,
            QueryType.MULTI_HOP: 0.6,
            QueryType.HYPOTHETICAL: 0.8
        }

    def classify_query(self, query: str) -> Tuple[QueryType, float, Dict[str, Any]]:
        """
        Classify query type and return confidence and analysis details

        Args:
            query: Input query string

        Returns:
            Tuple of (query_type, confidence, analysis_details)
        """
        query_lower = query.lower().strip()
        analysis = {
            "original_query": query,
            "normalized_query": query_lower,
            "pattern_matches": {},
            "complexity_indicators": [],
            "reasoning_hints": []
        }

        # Score each query type
        type_scores = {}

        # Check temporal patterns
        temporal_score = self._check_temporal_patterns(query_lower)
        if temporal_score > 0:
            type_scores[QueryType.TEMPORAL_ANALYSIS] = temporal_score
            analysis["pattern_matches"]["temporal"] = temporal_score

        # Check causal patterns
        causal_score = self._check_causal_patterns(query_lower)
        if causal_score > 0:
            type_scores[QueryType.CAUSAL_CHAIN] = causal_score
            analysis["pattern_matches"]["causal"] = causal_score

        # Check comparative patterns
        comparative_score = self._check_comparative_patterns(query_lower)
        if comparative_score > 0:
            type_scores[QueryType.COMPARATIVE] = comparative_score
            analysis["pattern_matches"]["comparative"] = comparative_score

        # Check meta-knowledge patterns
        meta_score = self._check_meta_patterns(query_lower)
        if meta_score > 0:
            type_scores[QueryType.META_KNOWLEDGE] = meta_score
            analysis["pattern_matches"]["meta"] = meta_score

        # Check aggregation patterns
        agg_score = self._check_aggregation_patterns(query_lower)
        if agg_score > 0:
            type_scores[QueryType.AGGREGATION] = agg_score
            analysis["pattern_matches"]["aggregation"] = agg_score

        # Check hypothetical patterns
        hyp_score = self._check_hypothetical_patterns(query_lower)
        if hyp_score > 0:
            type_scores[QueryType.HYPOTHETICAL] = hyp_score
            analysis["pattern_matches"]["hypothetical"] = hyp_score

        # Check multi-hop indicators
        multihop_score = self._check_multihop_indicators(query_lower)
        if multihop_score > 0:
            type_scores[QueryType.MULTI_HOP] = multihop_score
            analysis["pattern_matches"]["multihop"] = multihop_score

        # Determine complexity
        complexity_score = self._calculate_complexity(query_lower, type_scores)
        analysis["complexity_score"] = complexity_score

        # Select best type
        if type_scores:
            # Weight scores by pattern reliability
            weighted_scores = {
                qtype: score * self.pattern_weights.get(qtype, 0.5)
                for qtype, score in type_scores.items()
            }

            best_type = max(weighted_scores.items(), key=lambda x: x[1])
            query_type, confidence = best_type

            # Adjust confidence based on complexity
            confidence = min(confidence, 1.0)

        else:
            # Default to simple fact retrieval
            query_type = QueryType.SIMPLE_FACT
            confidence = 0.9 if len(query.split()) <= 5 else 0.7

        analysis["final_type"] = query_type.value
        analysis["final_confidence"] = confidence

        logger.debug(f"Classified query '{query[:50]}...' as {query_type.value} "
                    f"with confidence {confidence:.3f}")

        return query_type, confidence, analysis

    def suggest_strategy(self, query_type: QueryType,
                        complexity_score: float) -> ReasoningStrategy:
        """Suggest reasoning strategy based on query type and complexity"""

        strategy_map = {
            QueryType.SIMPLE_FACT: ReasoningStrategy.DIRECT_RETRIEVAL,
            QueryType.TEMPORAL_ANALYSIS: ReasoningStrategy.TEMPORAL_REASONING,
            QueryType.CAUSAL_CHAIN: ReasoningStrategy.CAUSAL_REASONING,
            QueryType.COMPARATIVE: ReasoningStrategy.COMPARATIVE_ANALYSIS,
            QueryType.META_KNOWLEDGE: ReasoningStrategy.META_REASONING,
            QueryType.AGGREGATION: ReasoningStrategy.GRAPH_TRAVERSAL,
            QueryType.HYPOTHETICAL: ReasoningStrategy.STEP_BY_STEP,
            QueryType.MULTI_HOP: ReasoningStrategy.STEP_BY_STEP
        }

        base_strategy = strategy_map.get(query_type, ReasoningStrategy.DIRECT_RETRIEVAL)

        # Use hybrid strategy for complex queries
        if complexity_score > 0.7:
            return ReasoningStrategy.HYBRID

        return base_strategy

    def _build_temporal_patterns(self) -> List[str]:
        """Build patterns for temporal reasoning detection"""
        return [
            r'\b(when|before|after|during|since|until|while)\b',
            r'\b(first|last|previous|next|earlier|later)\b',
            r'\b(timeline|chronology|sequence|order|history)\b',
            r'\b(\d{4}|\d{1,2}/\d{1,2}/\d{4}|january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(yesterday|today|tomorrow|recently|currently)\b',
            r'\b(then|now|eventually|finally|initially)\b'
        ]

    def _build_causal_patterns(self) -> List[str]:
        """Build patterns for causal reasoning detection"""
        return [
            r'\b(because|cause|caused|causes|reason|why|how)\b',
            r'\b(result|consequence|effect|affect|impact|influence)\b',
            r'\b(lead to|leads to|led to|due to|thanks to)\b',
            r'\b(therefore|thus|hence|consequently|as a result)\b',
            r'\b(trigger|triggered|triggers|enable|enables|prevent|prevents)\b',
            r'\b(explain|explanation|responsible for|blame)\b'
        ]

    def _build_comparative_patterns(self) -> List[str]:
        """Build patterns for comparative analysis detection"""
        return [
            r'\b(compare|comparison|versus|vs|against|between)\b',
            r'\b(different|difference|similar|similarity|alike|unlike)\b',
            r'\b(better|worse|best|worst|more|less|most|least)\b',
            r'\b(higher|lower|greater|smaller|larger)\b',
            r'\b(advantage|disadvantage|pros|cons|benefits|drawbacks)\b',
            r'\b(alternative|alternatives|option|options|choice|choices)\b'
        ]

    def _build_meta_patterns(self) -> List[str]:
        """Build patterns for meta-knowledge detection"""
        return [
            r'\b(what do you know about|tell me about|information about)\b',
            r'\b(how much|how many|how often|how long)\b',
            r'\b(source|sources|reference|references|citation|evidence)\b',
            r'\b(confidence|certainty|reliability|credibility|trust)\b',
            r'\b(knowledge|data|information|facts|details)\b',
            r'\b(summarize|summary|overview|explain|describe)\b'
        ]

    def _build_aggregation_patterns(self) -> List[str]:
        """Build patterns for aggregation queries"""
        return [
            r'\b(total|sum|count|average|mean|median|maximum|minimum)\b',
            r'\b(all|every|each|any|none|some|many|few)\b',
            r'\b(list|listing|enumerate|show me all|find all)\b',
            r'\b(statistics|stats|numbers|data|metrics)\b',
            r'\b(how many|how much|percentage|proportion|ratio)\b'
        ]

    def _build_complexity_indicators(self) -> List[str]:
        """Build patterns that indicate query complexity"""
        return [
            r'\b(and|or|but|however|although|while|whereas)\b',
            r'\b(if|unless|provided|assuming|given that)\b',
            r'\b(multiple|several|various|different|numerous)\b',
            r'\?.*\?',  # Multiple questions
            r'\b(step by step|process|procedure|method|approach)\b',
            r'\b(analyze|analysis|evaluate|assessment|review)\b'
        ]

    def _check_temporal_patterns(self, query: str) -> float:
        """Check for temporal reasoning patterns"""
        return self._pattern_score(query, self.temporal_patterns)

    def _check_causal_patterns(self, query: str) -> float:
        """Check for causal reasoning patterns"""
        return self._pattern_score(query, self.causal_patterns)

    def _check_comparative_patterns(self, query: str) -> float:
        """Check for comparative analysis patterns"""
        return self._pattern_score(query, self.comparative_patterns)

    def _check_meta_patterns(self, query: str) -> float:
        """Check for meta-knowledge patterns"""
        return self._pattern_score(query, self.meta_patterns)

    def _check_aggregation_patterns(self, query: str) -> float:
        """Check for aggregation patterns"""
        return self._pattern_score(query, self.aggregation_patterns)

    def _check_hypothetical_patterns(self, query: str) -> float:
        """Check for hypothetical reasoning patterns"""
        hypothetical_patterns = [
            r'\b(what if|suppose|assuming|imagine|hypothetically)\b',
            r'\b(would|could|might|may|should)\b',
            r'\b(scenario|situation|case|example|instance)\b',
            r'\b(predict|prediction|forecast|estimate)\b'
        ]
        return self._pattern_score(query, hypothetical_patterns)

    def _check_multihop_indicators(self, query: str) -> float:
        """Check for multi-hop reasoning indicators"""
        multihop_patterns = [
            r'\b(related to|connected to|associated with|linked to)\b',
            r'\b(through|via|by way of|using|utilizing)\b',
            r'\b(network|graph|relationship|connection|link)\b',
            r'\b(indirect|intermediate|pathway|route|chain)\b'
        ]

        # Also check for complex sentence structure
        complex_score = 0.0
        if len(query.split(',')) > 2:  # Multiple clauses
            complex_score += 0.3
        if len(query.split()) > 15:    # Long query
            complex_score += 0.2

        pattern_score = self._pattern_score(query, multihop_patterns)
        return min(pattern_score + complex_score, 1.0)

    def _pattern_score(self, query: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matches"""
        total_matches = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, query, re.IGNORECASE))
            total_matches += matches

        # Normalize by query length and number of patterns
        query_words = len(query.split())
        max_possible = len(patterns)

        if max_possible == 0:
            return 0.0

        # Score based on match density
        density = total_matches / max(query_words, 1)
        coverage = min(total_matches / max_possible, 1.0)

        return min(density + coverage, 1.0)

    def _calculate_complexity(self, query: str, type_scores: Dict[QueryType, float]) -> float:
        """Calculate overall query complexity score"""
        base_complexity = 0.1

        # Length-based complexity
        word_count = len(query.split())
        if word_count > 20:
            base_complexity += 0.3
        elif word_count > 10:
            base_complexity += 0.2
        elif word_count > 5:
            base_complexity += 0.1

        # Multiple question marks
        question_count = query.count('?')
        if question_count > 1:
            base_complexity += 0.2

        # Type-based complexity
        type_complexity = {
            QueryType.SIMPLE_FACT: 0.1,
            QueryType.TEMPORAL_ANALYSIS: 0.4,
            QueryType.CAUSAL_CHAIN: 0.6,
            QueryType.COMPARATIVE: 0.5,
            QueryType.META_KNOWLEDGE: 0.3,
            QueryType.AGGREGATION: 0.4,
            QueryType.MULTI_HOP: 0.8,
            QueryType.HYPOTHETICAL: 0.7
        }

        if type_scores:
            max_type_complexity = max(
                type_complexity.get(qtype, 0.3) * score
                for qtype, score in type_scores.items()
            )
            base_complexity += max_type_complexity

        # Complexity indicators
        complexity_score = self._pattern_score(query, self.complexity_indicators)
        base_complexity += complexity_score * 0.3

        return min(base_complexity, 1.0)

    def get_reasoning_hints(self, query: str, query_type: QueryType) -> List[str]:
        """Generate reasoning hints for query execution"""
        hints = []

        query_lower = query.lower()

        if query_type == QueryType.TEMPORAL_ANALYSIS:
            hints.extend([
                "Consider temporal ordering of events",
                "Look for time-based relationships",
                "Check for chronological dependencies"
            ])

        elif query_type == QueryType.CAUSAL_CHAIN:
            hints.extend([
                "Identify cause-effect relationships",
                "Look for causal mechanisms",
                "Consider multiple causal factors"
            ])

        elif query_type == QueryType.COMPARATIVE:
            hints.extend([
                "Gather comparable entities",
                "Identify comparison dimensions",
                "Look for contrasting features"
            ])

        elif query_type == QueryType.META_KNOWLEDGE:
            hints.extend([
                "Consider knowledge sources",
                "Evaluate information confidence",
                "Provide context and explanations"
            ])

        elif query_type == QueryType.MULTI_HOP:
            hints.extend([
                "Plan multi-step reasoning path",
                "Consider intermediate results",
                "Build evidence chain"
            ])

        # Add specific hints based on query content
        if 'uncertainty' in query_lower or 'confidence' in query_lower:
            hints.append("Include uncertainty estimates")

        if 'recent' in query_lower or 'latest' in query_lower:
            hints.append("Prioritize recent information")

        if 'evidence' in query_lower or 'source' in query_lower:
            hints.append("Provide supporting evidence")

        return hints
