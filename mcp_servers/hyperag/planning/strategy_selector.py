"""
Strategy Selection System

Selects and configures reasoning strategies based on query classification and requirements.
Manages strategy registry and provides adaptive strategy selection.
"""

import logging
from typing import Dict, List, Optional, Type, Union

from .plan_structures import (
    QueryType, ReasoningStrategy, QueryPlan, ExecutionStep,
    RetrievalConstraints
)

logger = logging.getLogger(__name__)


class StrategySelector:
    """
    Selects appropriate reasoning strategies based on query analysis.
    Maintains registry of available strategies and provides adaptive selection.
    """

    def __init__(self):
        # Import strategies here to avoid circular imports
        from .strategies import (
            SimpleFactStrategy, TemporalStrategy, CausalStrategy,
            ComparativeStrategy, MetaQueryStrategy, MultiHopStrategy,
            AggregationStrategy, HypotheticalStrategy, HybridStrategy
        )

        # Strategy registry
        self.strategies = {
            ReasoningStrategy.DIRECT_RETRIEVAL: SimpleFactStrategy,
            ReasoningStrategy.TEMPORAL_REASONING: TemporalStrategy,
            ReasoningStrategy.CAUSAL_REASONING: CausalStrategy,
            ReasoningStrategy.COMPARATIVE_ANALYSIS: ComparativeStrategy,
            ReasoningStrategy.META_REASONING: MetaQueryStrategy,
            ReasoningStrategy.STEP_BY_STEP: MultiHopStrategy,
            ReasoningStrategy.GRAPH_TRAVERSAL: AggregationStrategy,
            ReasoningStrategy.HYBRID: HybridStrategy
        }

        # Strategy preferences by query type
        self.type_strategy_map = {
            QueryType.SIMPLE_FACT: [
                ReasoningStrategy.DIRECT_RETRIEVAL
            ],
            QueryType.TEMPORAL_ANALYSIS: [
                ReasoningStrategy.TEMPORAL_REASONING,
                ReasoningStrategy.STEP_BY_STEP
            ],
            QueryType.CAUSAL_CHAIN: [
                ReasoningStrategy.CAUSAL_REASONING,
                ReasoningStrategy.GRAPH_TRAVERSAL
            ],
            QueryType.COMPARATIVE: [
                ReasoningStrategy.COMPARATIVE_ANALYSIS,
                ReasoningStrategy.GRAPH_TRAVERSAL
            ],
            QueryType.META_KNOWLEDGE: [
                ReasoningStrategy.META_REASONING,
                ReasoningStrategy.DIRECT_RETRIEVAL
            ],
            QueryType.MULTI_HOP: [
                ReasoningStrategy.STEP_BY_STEP,
                ReasoningStrategy.GRAPH_TRAVERSAL
            ],
            QueryType.AGGREGATION: [
                ReasoningStrategy.GRAPH_TRAVERSAL,
                ReasoningStrategy.STEP_BY_STEP
            ],
            QueryType.HYPOTHETICAL: [
                ReasoningStrategy.STEP_BY_STEP,
                ReasoningStrategy.HYBRID
            ]
        }

        # Strategy performance tracking (would be learned over time)
        self.strategy_performance = {
            strategy: {"success_rate": 0.8, "avg_confidence": 0.7, "avg_time_ms": 2000}
            for strategy in ReasoningStrategy
        }

    def select_strategy(self,
                       query_type: QueryType,
                       complexity_score: float,
                       constraints: Optional[RetrievalConstraints] = None,
                       context: Optional[Dict] = None) -> ReasoningStrategy:
        """
        Select best reasoning strategy for the given query characteristics

        Args:
            query_type: Classified query type
            complexity_score: Query complexity [0,1]
            constraints: Execution constraints
            context: Additional context for selection

        Returns:
            Selected reasoning strategy
        """
        context = context or {}
        constraints = constraints or RetrievalConstraints()

        # Get candidate strategies for this query type
        candidates = self.type_strategy_map.get(query_type, [ReasoningStrategy.DIRECT_RETRIEVAL])

        # Apply complexity-based filtering
        if complexity_score > 0.8:
            # Very complex queries benefit from hybrid or step-by-step
            if ReasoningStrategy.HYBRID in self.strategies:
                return ReasoningStrategy.HYBRID
            elif ReasoningStrategy.STEP_BY_STEP in candidates:
                return ReasoningStrategy.STEP_BY_STEP

        elif complexity_score < 0.3:
            # Simple queries can use direct retrieval
            if ReasoningStrategy.DIRECT_RETRIEVAL in candidates:
                return ReasoningStrategy.DIRECT_RETRIEVAL

        # Apply constraint-based filtering
        if constraints.time_budget_ms < 1000:
            # Time-constrained queries prefer faster strategies
            fast_strategies = [ReasoningStrategy.DIRECT_RETRIEVAL, ReasoningStrategy.META_REASONING]
            candidates = [s for s in candidates if s in fast_strategies] or candidates

        # Select based on performance metrics
        best_strategy = self._select_by_performance(candidates, context)

        logger.debug(f"Selected {best_strategy.value} for {query_type.value} "
                    f"(complexity: {complexity_score:.2f})")

        return best_strategy

    def create_strategy_instance(self, strategy: ReasoningStrategy, **kwargs):
        """Create instance of selected strategy"""
        strategy_class = self.strategies.get(strategy)

        if not strategy_class:
            raise ValueError(f"Unknown strategy: {strategy}")

        return strategy_class(**kwargs)

    def _select_by_performance(self,
                              candidates: List[ReasoningStrategy],
                              context: Dict) -> ReasoningStrategy:
        """Select strategy based on performance metrics"""
        if not candidates:
            return ReasoningStrategy.DIRECT_RETRIEVAL

        if len(candidates) == 1:
            return candidates[0]

        # Score candidates based on performance metrics
        scores = {}
        for strategy in candidates:
            perf = self.strategy_performance.get(strategy, {})

            # Base score from historical performance
            score = (
                perf.get("success_rate", 0.5) * 0.4 +
                perf.get("avg_confidence", 0.5) * 0.3 +
                (1.0 - min(perf.get("avg_time_ms", 5000) / 10000, 1.0)) * 0.3
            )

            # Adjust for context preferences
            if context.get("prefer_fast", False):
                score += 0.1 if perf.get("avg_time_ms", 5000) < 2000 else -0.1

            if context.get("prefer_accurate", False):
                score += 0.1 if perf.get("avg_confidence", 0.5) > 0.8 else -0.1

            scores[strategy] = score

        # Return strategy with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def update_strategy_performance(self,
                                   strategy: ReasoningStrategy,
                                   success: bool,
                                   confidence: float,
                                   execution_time_ms: float) -> None:
        """Update performance metrics for a strategy"""
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {
                "success_rate": 0.8,
                "avg_confidence": 0.7,
                "avg_time_ms": 2000,
                "sample_count": 0
            }

        perf = self.strategy_performance[strategy]
        count = perf.get("sample_count", 0)
        alpha = 0.1  # Learning rate

        # Update success rate
        current_success = 1.0 if success else 0.0
        perf["success_rate"] = (1 - alpha) * perf["success_rate"] + alpha * current_success

        # Update average confidence
        if confidence > 0:
            perf["avg_confidence"] = (1 - alpha) * perf["avg_confidence"] + alpha * confidence

        # Update average time
        if execution_time_ms > 0:
            perf["avg_time_ms"] = (1 - alpha) * perf["avg_time_ms"] + alpha * execution_time_ms

        perf["sample_count"] = count + 1

        logger.debug(f"Updated {strategy.value} performance: "
                    f"success={perf['success_rate']:.3f}, "
                    f"confidence={perf['avg_confidence']:.3f}, "
                    f"time={perf['avg_time_ms']:.1f}ms")

    def get_strategy_info(self, strategy: ReasoningStrategy) -> Dict:
        """Get information about a strategy"""
        strategy_class = self.strategies.get(strategy)
        perf = self.strategy_performance.get(strategy, {})

        return {
            "strategy": strategy.value,
            "class_name": strategy_class.__name__ if strategy_class else None,
            "description": strategy_class.description if strategy_class and hasattr(strategy_class, 'description') else "",
            "performance": perf,
            "suitable_for": [qt.value for qt, strategies in self.type_strategy_map.items()
                           if strategy in strategies]
        }

    def list_available_strategies(self) -> List[Dict]:
        """List all available strategies with their information"""
        return [self.get_strategy_info(strategy) for strategy in ReasoningStrategy]

    def recommend_fallback_strategy(self,
                                   failed_strategy: ReasoningStrategy,
                                   query_type: QueryType) -> Optional[ReasoningStrategy]:
        """Recommend fallback strategy when primary strategy fails"""
        candidates = self.type_strategy_map.get(query_type, [])

        # Remove failed strategy from candidates
        fallback_candidates = [s for s in candidates if s != failed_strategy]

        if not fallback_candidates:
            # Use simple direct retrieval as last resort
            return ReasoningStrategy.DIRECT_RETRIEVAL

        # Select fallback based on performance
        return self._select_by_performance(fallback_candidates, {"prefer_reliable": True})

    def can_handle_complexity(self,
                             strategy: ReasoningStrategy,
                             complexity_score: float) -> bool:
        """Check if strategy can handle given complexity level"""
        complexity_thresholds = {
            ReasoningStrategy.DIRECT_RETRIEVAL: 0.5,
            ReasoningStrategy.META_REASONING: 0.6,
            ReasoningStrategy.TEMPORAL_REASONING: 0.7,
            ReasoningStrategy.COMPARATIVE_ANALYSIS: 0.7,
            ReasoningStrategy.CAUSAL_REASONING: 0.8,
            ReasoningStrategy.GRAPH_TRAVERSAL: 0.8,
            ReasoningStrategy.STEP_BY_STEP: 0.9,
            ReasoningStrategy.HYBRID: 1.0
        }

        threshold = complexity_thresholds.get(strategy, 0.5)
        return complexity_score <= threshold

    def get_strategy_requirements(self, strategy: ReasoningStrategy) -> Dict:
        """Get resource requirements for a strategy"""
        requirements = {
            ReasoningStrategy.DIRECT_RETRIEVAL: {
                "min_memory_mb": 50,
                "estimated_time_ms": 500,
                "graph_traversal_depth": 1,
                "requires_reasoning": False
            },
            ReasoningStrategy.META_REASONING: {
                "min_memory_mb": 100,
                "estimated_time_ms": 1000,
                "graph_traversal_depth": 2,
                "requires_reasoning": True
            },
            ReasoningStrategy.TEMPORAL_REASONING: {
                "min_memory_mb": 150,
                "estimated_time_ms": 2000,
                "graph_traversal_depth": 3,
                "requires_reasoning": True
            },
            ReasoningStrategy.COMPARATIVE_ANALYSIS: {
                "min_memory_mb": 200,
                "estimated_time_ms": 2500,
                "graph_traversal_depth": 3,
                "requires_reasoning": True
            },
            ReasoningStrategy.CAUSAL_REASONING: {
                "min_memory_mb": 250,
                "estimated_time_ms": 3000,
                "graph_traversal_depth": 4,
                "requires_reasoning": True
            },
            ReasoningStrategy.GRAPH_TRAVERSAL: {
                "min_memory_mb": 300,
                "estimated_time_ms": 3500,
                "graph_traversal_depth": 5,
                "requires_reasoning": True
            },
            ReasoningStrategy.STEP_BY_STEP: {
                "min_memory_mb": 400,
                "estimated_time_ms": 5000,
                "graph_traversal_depth": 6,
                "requires_reasoning": True
            },
            ReasoningStrategy.HYBRID: {
                "min_memory_mb": 500,
                "estimated_time_ms": 7000,
                "graph_traversal_depth": 8,
                "requires_reasoning": True
            }
        }

        return requirements.get(strategy, requirements[ReasoningStrategy.DIRECT_RETRIEVAL])
