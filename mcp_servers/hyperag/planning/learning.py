"""Planning Learning System

Learns from plan execution outcomes to improve strategy selection and planning.
Tracks plan effectiveness, updates strategy weights, and learns from agent feedback.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
from typing import Any

import numpy as np

from .plan_structures import QueryPlan, QueryType, ReasoningStrategy

logger = logging.getLogger(__name__)


@dataclass
class StrategyFeedback:
    """Feedback about strategy execution performance"""

    strategy: ReasoningStrategy
    query_type: QueryType
    complexity_score: float

    # Execution outcomes
    success: bool
    confidence_achieved: float
    execution_time_ms: float
    steps_completed: int
    steps_failed: int

    # Quality metrics
    result_quality: float | None = None      # Agent assessment of result quality
    user_satisfaction: float | None = None   # User feedback
    correctness: bool | None = None          # Ground truth correctness

    # Context
    agent_model: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Aggregated learning metrics for strategies"""

    strategy: ReasoningStrategy
    query_type: QueryType

    # Performance statistics
    total_executions: int = 0
    successful_executions: int = 0
    avg_confidence: float = 0.0
    avg_execution_time_ms: float = 0.0
    avg_result_quality: float = 0.0

    # Trend tracking
    recent_success_rate: float = 0.0
    success_trend: float = 0.0  # +/- indicating improvement/degradation
    confidence_trend: float = 0.0

    # Learning parameters
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    learning_weight: float = 1.0  # How much to trust this strategy


class PlanLearner:
    """Learning system for query planning optimization.

    Tracks plan effectiveness, updates strategy weights, and learns from feedback
    to improve future planning decisions.
    """

    def __init__(self, learning_rate: float = 0.1, min_samples: int = 5):
        self.learning_rate = learning_rate
        self.min_samples = min_samples

        # Strategy performance tracking
        self.strategy_metrics: dict[tuple[ReasoningStrategy, QueryType], LearningMetrics] = {}
        self.global_strategy_weights: dict[ReasoningStrategy, float] = dict.fromkeys(ReasoningStrategy, 1.0)

        # Feedback history
        self.feedback_history: list[StrategyFeedback] = []
        self.max_history_size = 10000

        # Learning statistics
        self.learning_stats = {
            "total_feedback_received": 0,
            "successful_adaptations": 0,
            "strategy_weight_updates": 0,
            "last_learning_update": datetime.now(timezone.utc)
        }

        # Pattern recognition
        self.query_patterns: dict[str, list[tuple[ReasoningStrategy, float]]] = {}

    def record_execution_feedback(self,
                                 plan: QueryPlan,
                                 feedback: StrategyFeedback) -> None:
        """Record feedback from plan execution"""
        # Store feedback
        self.feedback_history.append(feedback)
        if len(self.feedback_history) > self.max_history_size:
            self.feedback_history = self.feedback_history[-self.max_history_size:]

        # Update strategy metrics
        self._update_strategy_metrics(feedback)

        # Update global strategy weights
        self._update_strategy_weights(feedback)

        # Learn query patterns
        self._learn_query_patterns(plan, feedback)

        # Update learning statistics
        self.learning_stats["total_feedback_received"] += 1
        self.learning_stats["last_learning_update"] = datetime.now(timezone.utc)

        if feedback.success:
            self.learning_stats["successful_adaptations"] += 1

        logger.debug(f"Recorded feedback for {feedback.strategy.value} on "
                    f"{feedback.query_type.value} (success: {feedback.success})")

    def get_strategy_recommendation(self,
                                   query_type: QueryType,
                                   complexity_score: float,
                                   context: dict[str, Any] | None = None) -> tuple[ReasoningStrategy, float]:
        """Get strategy recommendation based on learned performance"""
        context = context or {}

        # Get candidate strategies for this query type
        candidates = self._get_candidate_strategies(query_type)

        # Score candidates based on learned performance
        strategy_scores = {}

        for strategy in candidates:
            score = self._calculate_strategy_score(
                strategy, query_type, complexity_score, context
            )
            strategy_scores[strategy] = score

        # Select best strategy
        if strategy_scores:
            best_strategy = max(strategy_scores.items(), key=lambda x: x[1])
            return best_strategy
        # Fallback to direct retrieval
        return ReasoningStrategy.DIRECT_RETRIEVAL, 0.5

    def analyze_strategy_performance(self,
                                   strategy: ReasoningStrategy | None = None,
                                   query_type: QueryType | None = None,
                                   time_window_hours: int = 24) -> dict[str, Any]:
        """Analyze strategy performance over time window"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)
        recent_feedback = [
            f for f in self.feedback_history
            if f.timestamp >= cutoff_time
        ]

        if strategy:
            recent_feedback = [f for f in recent_feedback if f.strategy == strategy]
        if query_type:
            recent_feedback = [f for f in recent_feedback if f.query_type == query_type]

        if not recent_feedback:
            return {"message": "No recent feedback data"}

        # Calculate performance metrics
        total_executions = len(recent_feedback)
        successful_executions = sum(1 for f in recent_feedback if f.success)
        success_rate = successful_executions / total_executions

        avg_confidence = np.mean([f.confidence_achieved for f in recent_feedback])
        avg_time = np.mean([f.execution_time_ms for f in recent_feedback])

        quality_scores = [f.result_quality for f in recent_feedback if f.result_quality is not None]
        avg_quality = np.mean(quality_scores) if quality_scores else None

        # Analyze trends
        if len(recent_feedback) >= 10:
            # Split into first and second half to detect trends
            mid_point = len(recent_feedback) // 2
            first_half = recent_feedback[:mid_point]
            second_half = recent_feedback[mid_point:]

            first_success_rate = sum(1 for f in first_half if f.success) / len(first_half)
            second_success_rate = sum(1 for f in second_half if f.success) / len(second_half)

            success_trend = second_success_rate - first_success_rate
        else:
            success_trend = 0.0

        analysis = {
            "time_window_hours": time_window_hours,
            "total_executions": total_executions,
            "success_rate": success_rate,
            "avg_confidence": avg_confidence,
            "avg_execution_time_ms": avg_time,
            "avg_result_quality": avg_quality,
            "success_trend": success_trend,
            "strategy_filter": strategy.value if strategy else "all",
            "query_type_filter": query_type.value if query_type else "all"
        }

        return analysis

    def get_learning_insights(self) -> dict[str, Any]:
        """Get insights from learning system"""
        insights = {
            "learning_stats": self.learning_stats.copy(),
            "top_performing_strategies": self._get_top_strategies(),
            "problematic_combinations": self._get_problematic_combinations(),
            "query_pattern_insights": self._get_pattern_insights(),
            "recommendations": self._generate_recommendations()
        }

        return insights

    def _update_strategy_metrics(self, feedback: StrategyFeedback) -> None:
        """Update performance metrics for strategy-query type combination"""
        key = (feedback.strategy, feedback.query_type)

        if key not in self.strategy_metrics:
            self.strategy_metrics[key] = LearningMetrics(
                strategy=feedback.strategy,
                query_type=feedback.query_type
            )

        metrics = self.strategy_metrics[key]

        # Update counts
        metrics.total_executions += 1
        if feedback.success:
            metrics.successful_executions += 1

        # Update averages using exponential moving average
        alpha = self.learning_rate

        if feedback.confidence_achieved > 0:
            metrics.avg_confidence = (
                (1 - alpha) * metrics.avg_confidence +
                alpha * feedback.confidence_achieved
            )

        if feedback.execution_time_ms > 0:
            metrics.avg_execution_time_ms = (
                (1 - alpha) * metrics.avg_execution_time_ms +
                alpha * feedback.execution_time_ms
            )

        if feedback.result_quality is not None:
            metrics.avg_result_quality = (
                (1 - alpha) * metrics.avg_result_quality +
                alpha * feedback.result_quality
            )

        # Update recent success rate (last 20 executions)
        recent_feedback = [
            f for f in self.feedback_history[-20:]
            if f.strategy == feedback.strategy and f.query_type == feedback.query_type
        ]

        if len(recent_feedback) >= self.min_samples:
            recent_successes = sum(1 for f in recent_feedback if f.success)
            metrics.recent_success_rate = recent_successes / len(recent_feedback)

            # Calculate trend
            if len(recent_feedback) >= 10:
                mid = len(recent_feedback) // 2
                first_half_success = sum(1 for f in recent_feedback[:mid] if f.success) / mid
                second_half_success = sum(1 for f in recent_feedback[mid:] if f.success) / (len(recent_feedback) - mid)
                metrics.success_trend = second_half_success - first_half_success

        metrics.last_updated = datetime.now(timezone.utc)

    def _update_strategy_weights(self, feedback: StrategyFeedback) -> None:
        """Update global strategy weights based on feedback"""
        strategy = feedback.strategy
        current_weight = self.global_strategy_weights[strategy]

        # Adjust weight based on success and quality
        adjustment = 0.0

        if feedback.success:
            adjustment += 0.1
            if feedback.confidence_achieved > 0.8:
                adjustment += 0.05
            if feedback.result_quality and feedback.result_quality > 0.8:
                adjustment += 0.05
        else:
            adjustment -= 0.1
            if feedback.confidence_achieved < 0.3:
                adjustment -= 0.05

        # Apply learning rate
        adjustment *= self.learning_rate

        # Update weight with bounds
        new_weight = np.clip(current_weight + adjustment, 0.1, 2.0)
        self.global_strategy_weights[strategy] = new_weight

        if abs(adjustment) > 0.01:
            self.learning_stats["strategy_weight_updates"] += 1
            logger.debug(f"Updated {strategy.value} weight: {current_weight:.3f} -> {new_weight:.3f}")

    def _learn_query_patterns(self, plan: QueryPlan, feedback: StrategyFeedback) -> None:
        """Learn patterns from successful query-strategy combinations"""
        # Extract query features for pattern recognition
        query_features = self._extract_query_features(plan.original_query)

        # Store successful patterns
        if feedback.success and feedback.confidence_achieved > 0.7:
            for feature in query_features:
                if feature not in self.query_patterns:
                    self.query_patterns[feature] = []

                # Add or update strategy-score pair
                strategy_scores = self.query_patterns[feature]

                # Find existing entry
                existing_idx = None
                for i, (strategy, score) in enumerate(strategy_scores):
                    if strategy == feedback.strategy:
                        existing_idx = i
                        break

                if existing_idx is not None:
                    # Update existing score
                    old_score = strategy_scores[existing_idx][1]
                    new_score = (1 - self.learning_rate) * old_score + self.learning_rate * feedback.confidence_achieved
                    strategy_scores[existing_idx] = (feedback.strategy, new_score)
                else:
                    # Add new entry
                    strategy_scores.append((feedback.strategy, feedback.confidence_achieved))

                # Keep only top strategies per feature
                strategy_scores.sort(key=lambda x: x[1], reverse=True)
                self.query_patterns[feature] = strategy_scores[:5]

    def _extract_query_features(self, query: str) -> list[str]:
        """Extract features from query for pattern recognition"""
        query_lower = query.lower()
        features = []

        # Length-based features
        word_count = len(query.split())
        if word_count <= 5:
            features.append("short_query")
        elif word_count <= 15:
            features.append("medium_query")
        else:
            features.append("long_query")

        # Keyword-based features
        temporal_keywords = ["when", "before", "after", "since", "until", "during"]
        if any(kw in query_lower for kw in temporal_keywords):
            features.append("temporal_query")

        causal_keywords = ["why", "because", "cause", "reason", "due to"]
        if any(kw in query_lower for kw in causal_keywords):
            features.append("causal_query")

        comparative_keywords = ["compare", "versus", "difference", "similar"]
        if any(kw in query_lower for kw in comparative_keywords):
            features.append("comparative_query")

        question_keywords = ["what", "how", "where", "who", "which"]
        if any(kw in query_lower for kw in question_keywords):
            features.append("question_query")

        # Complexity indicators
        if "?" in query:
            question_count = query.count("?")
            if question_count > 1:
                features.append("multi_question")

        if " and " in query_lower or " or " in query_lower:
            features.append("compound_query")

        return features

    def _get_candidate_strategies(self, query_type: QueryType) -> list[ReasoningStrategy]:
        """Get candidate strategies for query type"""
        type_strategy_map = {
            QueryType.SIMPLE_FACT: [ReasoningStrategy.DIRECT_RETRIEVAL, ReasoningStrategy.META_REASONING],
            QueryType.TEMPORAL_ANALYSIS: [ReasoningStrategy.TEMPORAL_REASONING, ReasoningStrategy.STEP_BY_STEP],
            QueryType.CAUSAL_CHAIN: [ReasoningStrategy.CAUSAL_REASONING, ReasoningStrategy.GRAPH_TRAVERSAL],
            QueryType.COMPARATIVE: [ReasoningStrategy.COMPARATIVE_ANALYSIS, ReasoningStrategy.GRAPH_TRAVERSAL],
            QueryType.META_KNOWLEDGE: [ReasoningStrategy.META_REASONING, ReasoningStrategy.DIRECT_RETRIEVAL],
            QueryType.MULTI_HOP: [ReasoningStrategy.STEP_BY_STEP, ReasoningStrategy.GRAPH_TRAVERSAL],
            QueryType.AGGREGATION: [ReasoningStrategy.GRAPH_TRAVERSAL, ReasoningStrategy.STEP_BY_STEP],
            QueryType.HYPOTHETICAL: [ReasoningStrategy.STEP_BY_STEP, ReasoningStrategy.HYBRID]
        }

        return type_strategy_map.get(query_type, [ReasoningStrategy.DIRECT_RETRIEVAL])

    def _calculate_strategy_score(self,
                                 strategy: ReasoningStrategy,
                                 query_type: QueryType,
                                 complexity_score: float,
                                 context: dict[str, Any]) -> float:
        """Calculate score for strategy based on learned performance"""
        # Base score from global weight
        base_score = self.global_strategy_weights.get(strategy, 1.0)

        # Strategy-specific metrics
        key = (strategy, query_type)
        if key in self.strategy_metrics:
            metrics = self.strategy_metrics[key]

            if metrics.total_executions >= self.min_samples:
                # Use learned performance
                success_score = metrics.recent_success_rate * 0.4
                confidence_score = metrics.avg_confidence * 0.3
                quality_score = metrics.avg_result_quality * 0.2 if metrics.avg_result_quality > 0 else 0.1
                trend_score = max(metrics.success_trend, 0) * 0.1

                learned_score = success_score + confidence_score + quality_score + trend_score
                base_score = (base_score + learned_score) / 2

        # Adjust for complexity
        complexity_adjustment = 1.0
        if complexity_score > 0.8 and strategy in [ReasoningStrategy.DIRECT_RETRIEVAL]:
            complexity_adjustment = 0.7  # Penalize simple strategies for complex queries
        elif complexity_score < 0.3 and strategy in [ReasoningStrategy.HYBRID, ReasoningStrategy.STEP_BY_STEP]:
            complexity_adjustment = 0.8  # Penalize complex strategies for simple queries

        # Context adjustments
        if context.get("prefer_fast", False):
            if strategy in [ReasoningStrategy.DIRECT_RETRIEVAL, ReasoningStrategy.META_REASONING]:
                complexity_adjustment *= 1.2
            elif strategy in [ReasoningStrategy.HYBRID, ReasoningStrategy.STEP_BY_STEP]:
                complexity_adjustment *= 0.8

        return base_score * complexity_adjustment

    def _get_top_strategies(self) -> list[dict[str, Any]]:
        """Get top performing strategies"""
        strategy_performance = []

        for strategy in ReasoningStrategy:
            # Calculate overall performance
            relevant_feedback = [f for f in self.feedback_history if f.strategy == strategy]

            if len(relevant_feedback) >= self.min_samples:
                success_rate = sum(1 for f in relevant_feedback if f.success) / len(relevant_feedback)
                avg_confidence = np.mean([f.confidence_achieved for f in relevant_feedback])

                strategy_performance.append({
                    "strategy": strategy.value,
                    "success_rate": success_rate,
                    "avg_confidence": avg_confidence,
                    "execution_count": len(relevant_feedback),
                    "global_weight": self.global_strategy_weights[strategy]
                })

        # Sort by composite score
        strategy_performance.sort(
            key=lambda x: x["success_rate"] * 0.6 + x["avg_confidence"] * 0.4,
            reverse=True
        )

        return strategy_performance[:5]

    def _get_problematic_combinations(self) -> list[dict[str, Any]]:
        """Identify problematic strategy-query type combinations"""
        problematic = []

        for key, metrics in self.strategy_metrics.items():
            if metrics.total_executions >= self.min_samples:
                success_rate = metrics.successful_executions / metrics.total_executions

                if success_rate < 0.5 or metrics.avg_confidence < 0.6:
                    problematic.append({
                        "strategy": key[0].value,
                        "query_type": key[1].value,
                        "success_rate": success_rate,
                        "avg_confidence": metrics.avg_confidence,
                        "execution_count": metrics.total_executions
                    })

        problematic.sort(key=lambda x: x["success_rate"])
        return problematic[:5]

    def _get_pattern_insights(self) -> dict[str, Any]:
        """Get insights from learned query patterns"""
        insights = {
            "total_patterns": len(self.query_patterns),
            "most_reliable_patterns": [],
            "emerging_patterns": []
        }

        # Find most reliable patterns
        for feature, strategies in self.query_patterns.items():
            if strategies:
                best_strategy, best_score = strategies[0]
                if best_score > 0.8:
                    insights["most_reliable_patterns"].append({
                        "pattern": feature,
                        "best_strategy": best_strategy.value,
                        "confidence": best_score
                    })

        insights["most_reliable_patterns"].sort(key=lambda x: x["confidence"], reverse=True)
        insights["most_reliable_patterns"] = insights["most_reliable_patterns"][:5]

        return insights

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on learning"""
        recommendations = []

        # Check for consistently failing strategies
        for key, metrics in self.strategy_metrics.items():
            if (metrics.total_executions >= self.min_samples and
                metrics.recent_success_rate < 0.4):
                recommendations.append(
                    f"Consider avoiding {key[0].value} for {key[1].value} queries "
                    f"(recent success rate: {metrics.recent_success_rate:.2f})"
                )

        # Check for underutilized high-performing strategies
        recent_feedback = self.feedback_history[-100:] if len(self.feedback_history) >= 100 else self.feedback_history
        strategy_usage = {}
        for feedback in recent_feedback:
            strategy_usage[feedback.strategy] = strategy_usage.get(feedback.strategy, 0) + 1

        for strategy, weight in self.global_strategy_weights.items():
            usage = strategy_usage.get(strategy, 0)
            if weight > 1.5 and usage < 5:
                recommendations.append(
                    f"Consider using {strategy.value} more often "
                    f"(high performance weight: {weight:.2f}, low recent usage: {usage})"
                )

        return recommendations[:5]
