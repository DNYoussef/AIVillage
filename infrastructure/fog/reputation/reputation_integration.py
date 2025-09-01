"""
Reputation System Integration Module

Provides unified integration points for the Bayesian reputation system
with other fog computing components including:
- Task scheduling and placement
- Market pricing and bidding
- Quality assurance systems
- Node performance monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .bayesian_reputation import BayesianReputationEngine, ReputationEvent, EventType, ReputationConfig, ReputationTier

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Task execution result for reputation tracking"""

    task_id: str
    node_id: str
    success: bool
    completion_time: float
    quality_score: Optional[float] = None
    error_type: Optional[str] = None
    resource_usage: Dict[str, float] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class NodeMetrics:
    """Node performance metrics for reputation updates"""

    node_id: str
    uptime_ratio: float
    avg_response_time: float
    success_rate: float
    tasks_completed: int
    tasks_failed: int
    quality_scores: List[float]
    availability_score: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


class ReputationIntegrationManager:
    """
    Central manager for reputation system integration

    Coordinates reputation updates from various fog computing subsystems
    and provides unified reputation-based decision making.
    """

    def __init__(
        self, reputation_engine: Optional[BayesianReputationEngine] = None, config: Optional[ReputationConfig] = None
    ):
        self.reputation_engine = reputation_engine or BayesianReputationEngine(config)
        self.integration_config = {
            "task_result_timeout": 300,  # 5 minutes
            "metrics_update_interval": 60,  # 1 minute
            "batch_update_size": 100,
            "quality_weight": 1.5,
            "performance_weight": 1.0,
            "availability_weight": 0.8,
        }

        # Event batching for performance
        self._pending_events: List[ReputationEvent] = []
        self._last_batch_update = datetime.now()

        # Performance tracking
        self.integration_stats = {
            "events_processed": 0,
            "batch_updates": 0,
            "scheduler_queries": 0,
            "pricing_queries": 0,
            "last_reset": datetime.now(),
        }

        # Background tasks
        self._batch_update_task = None
        self._metrics_collection_task = None

        logger.info("Reputation integration manager initialized")

    async def start(self):
        """Start integration manager background tasks"""
        self._batch_update_task = asyncio.create_task(self._batch_update_loop())
        self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        logger.info("Reputation integration manager started")

    async def stop(self):
        """Stop integration manager background tasks"""
        if self._batch_update_task:
            self._batch_update_task.cancel()
        if self._metrics_collection_task:
            self._metrics_collection_task.cancel()

        # Process any pending events
        await self._flush_pending_events()
        logger.info("Reputation integration manager stopped")

    # Task Execution Integration

    async def record_task_result(self, task_result: TaskResult):
        """Record task execution result for reputation update"""

        # Determine event type based on result
        if task_result.success:
            event_type = (
                EventType.QUALITY_HIGH
                if task_result.quality_score and task_result.quality_score > 0.8
                else EventType.TASK_SUCCESS
            )
        else:
            event_type = EventType.TASK_FAILURE

        # Calculate event weight based on task characteristics
        base_weight = 1.0

        # Adjust weight based on quality score
        quality_weight = 1.0
        if task_result.quality_score is not None:
            quality_weight = self.integration_config["quality_weight"] * task_result.quality_score

        # Adjust weight based on completion time (faster = better)
        performance_weight = self.integration_config["performance_weight"]
        if task_result.completion_time > 0:
            # Normalize performance (assume 60s is baseline)
            performance_factor = min(2.0, 60.0 / max(1.0, task_result.completion_time))
            performance_weight *= performance_factor

        final_weight = base_weight * quality_weight * performance_weight

        # Create reputation event
        event = ReputationEvent(
            node_id=task_result.node_id,
            event_type=event_type,
            timestamp=task_result.timestamp,
            weight=final_weight,
            quality_score=task_result.quality_score,
            context={
                "task_id": task_result.task_id,
                "completion_time": task_result.completion_time,
                "error_type": task_result.error_type,
                "resource_usage": task_result.resource_usage or {},
            },
        )

        # Add to batch queue
        self._pending_events.append(event)
        self.integration_stats["events_processed"] += 1

        # Flush if batch is full
        if len(self._pending_events) >= self.integration_config["batch_update_size"]:
            await self._flush_pending_events()

        logger.debug(f"Recorded task result for {task_result.node_id}: {task_result.success}")

    async def record_node_metrics(self, metrics: NodeMetrics):
        """Record node performance metrics for reputation update"""

        events = []

        # Availability events
        if metrics.uptime_ratio >= 0.95:
            events.append(
                ReputationEvent(
                    node_id=metrics.node_id,
                    event_type=EventType.AVAILABILITY,
                    timestamp=metrics.timestamp,
                    weight=metrics.uptime_ratio * self.integration_config["availability_weight"],
                )
            )
        elif metrics.uptime_ratio < 0.8:
            events.append(
                ReputationEvent(
                    node_id=metrics.node_id,
                    event_type=EventType.UNAVAILABILITY,
                    timestamp=metrics.timestamp,
                    weight=(1.0 - metrics.uptime_ratio) * self.integration_config["availability_weight"],
                )
            )

        # Task completion events
        if metrics.tasks_completed > 0:
            avg_quality = sum(metrics.quality_scores) / len(metrics.quality_scores) if metrics.quality_scores else None

            for _ in range(metrics.tasks_completed):
                events.append(
                    ReputationEvent(
                        node_id=metrics.node_id,
                        event_type=EventType.TASK_SUCCESS,
                        timestamp=metrics.timestamp,
                        quality_score=avg_quality,
                        weight=1.0,
                    )
                )

        if metrics.tasks_failed > 0:
            for _ in range(metrics.tasks_failed):
                events.append(
                    ReputationEvent(
                        node_id=metrics.node_id,
                        event_type=EventType.TASK_FAILURE,
                        timestamp=metrics.timestamp,
                        weight=1.0,
                    )
                )

        # Quality events based on average quality
        if metrics.quality_scores:
            avg_quality = sum(metrics.quality_scores) / len(metrics.quality_scores)
            if avg_quality >= 0.9:
                events.append(
                    ReputationEvent(
                        node_id=metrics.node_id,
                        event_type=EventType.QUALITY_HIGH,
                        timestamp=metrics.timestamp,
                        quality_score=avg_quality,
                        weight=1.5,
                    )
                )
            elif avg_quality <= 0.3:
                events.append(
                    ReputationEvent(
                        node_id=metrics.node_id,
                        event_type=EventType.QUALITY_LOW,
                        timestamp=metrics.timestamp,
                        quality_score=avg_quality,
                        weight=1.5,
                    )
                )

        # Add events to batch queue
        self._pending_events.extend(events)
        self.integration_stats["events_processed"] += len(events)

        logger.debug(f"Recorded metrics for {metrics.node_id}: {len(events)} events")

    # Scheduler Integration

    async def get_node_trust_scores(self, node_ids: List[str]) -> Dict[str, float]:
        """Get trust scores for scheduler node selection"""
        trust_scores = {}

        for node_id in node_ids:
            trust_score = self.reputation_engine.get_trust_score(node_id)
            trust_scores[node_id] = trust_score

        self.integration_stats["scheduler_queries"] += 1
        return trust_scores

    async def recommend_trusted_nodes(
        self, task_requirements: Dict[str, Any], available_nodes: List[str], min_trust: float = 0.6, max_nodes: int = 5
    ) -> List[str]:
        """Recommend most trusted nodes for task scheduling"""

        # Get trust scores for available nodes
        trust_scores = await self.get_node_trust_scores(available_nodes)

        # Filter by minimum trust threshold
        trusted_nodes = [node_id for node_id, trust_score in trust_scores.items() if trust_score >= min_trust]

        # Sort by trust score (descending)
        trusted_nodes.sort(key=lambda node_id: trust_scores[node_id], reverse=True)

        # Return top N nodes
        recommendations = trusted_nodes[:max_nodes]

        logger.info(
            f"Recommended {len(recommendations)} trusted nodes from {len(available_nodes)} available "
            f"(min_trust={min_trust})"
        )

        return recommendations

    async def get_tier_based_scheduling_weights(self, node_ids: List[str]) -> Dict[str, float]:
        """Get scheduling weights based on reputation tiers"""
        weights = {}

        tier_weights = {
            ReputationTier.DIAMOND: 2.0,
            ReputationTier.PLATINUM: 1.8,
            ReputationTier.GOLD: 1.5,
            ReputationTier.SILVER: 1.2,
            ReputationTier.BRONZE: 1.0,
            ReputationTier.UNTRUSTED: 0.5,
        }

        for node_id in node_ids:
            score = self.reputation_engine.get_reputation_score(node_id)
            if score:
                weights[node_id] = tier_weights.get(score.tier, 1.0)
            else:
                weights[node_id] = tier_weights[ReputationTier.UNTRUSTED]

        return weights

    # Pricing Integration

    async def get_reputation_pricing_multipliers(self, node_ids: List[str]) -> Dict[str, float]:
        """Get pricing multipliers based on node reputation"""
        multipliers = {}

        for node_id in node_ids:
            trust_score = self.reputation_engine.get_trust_score(node_id)

            # Higher trust nodes can charge premium (up to 50% bonus)
            # Lower trust nodes must offer discount (up to 30% discount)
            base_multiplier = 1.0
            trust_adjustment = (trust_score - 0.5) * 1.0  # -0.5 to +0.5 range
            multiplier = base_multiplier + trust_adjustment

            # Clamp to reasonable bounds
            multipliers[node_id] = max(0.7, min(1.5, multiplier))

        self.integration_stats["pricing_queries"] += 1
        return multipliers

    async def calculate_trust_based_reserve_prices(
        self, base_reserve_prices: Dict[str, float], node_ids: List[str]
    ) -> Dict[str, float]:
        """Calculate reserve prices adjusted for node reputation"""

        multipliers = await self.get_reputation_pricing_multipliers(node_ids)
        adjusted_prices = {}

        for node_id in node_ids:
            base_price = base_reserve_prices.get(node_id, 1.0)
            multiplier = multipliers.get(node_id, 1.0)
            adjusted_prices[node_id] = base_price * multiplier

        return adjusted_prices

    # Quality Assurance Integration

    async def detect_quality_anomalies(
        self, recent_hours: int = 24, quality_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect nodes with anomalous quality patterns"""

        anomalies = []

        # Get all nodes with recent activity
        self.reputation_engine.get_tier_distribution()

        for node_id in self.reputation_engine.reputation_scores.keys():
            insights = self.reputation_engine.get_reputation_insights(node_id)

            if insights.get("recent_events_30d", 0) > 5:  # Sufficient recent activity
                reputation_score = insights.get("reputation_score", 0)
                uncertainty = insights.get("uncertainty", 0)

                # Detect anomalies
                if reputation_score < quality_threshold and uncertainty > 0.2:
                    anomalies.append(
                        {
                            "node_id": node_id,
                            "anomaly_type": "low_quality_high_uncertainty",
                            "reputation_score": reputation_score,
                            "uncertainty": uncertainty,
                            "tier": insights.get("tier", "UNKNOWN"),
                            "recent_events": insights.get("recent_events_30d", 0),
                            "recommendation": "Monitor closely or reduce task allocation",
                        }
                    )
                elif reputation_score < 0.3:
                    anomalies.append(
                        {
                            "node_id": node_id,
                            "anomaly_type": "very_low_reputation",
                            "reputation_score": reputation_score,
                            "uncertainty": uncertainty,
                            "tier": insights.get("tier", "UNKNOWN"),
                            "recent_events": insights.get("recent_events_30d", 0),
                            "recommendation": "Consider removing from available pool",
                        }
                    )

        logger.info(f"Detected {len(anomalies)} quality anomalies")
        return anomalies

    async def record_fraud_detection(self, node_id: str, fraud_type: str, evidence: Dict[str, Any]):
        """Record fraud detection for immediate reputation penalty"""

        fraud_event = ReputationEvent(
            node_id=node_id,
            event_type=EventType.FRAUD_DETECTED,
            timestamp=datetime.now().timestamp(),
            weight=5.0,  # Heavy penalty
            context={"fraud_type": fraud_type, "evidence": evidence, "detection_timestamp": datetime.now().isoformat()},
        )

        # Process immediately (don't batch)
        self.reputation_engine.record_event(fraud_event)

        logger.critical(f"Recorded fraud detection for {node_id}: {fraud_type}")

    # Analytics and Monitoring

    async def get_system_reputation_health(self) -> Dict[str, Any]:
        """Get comprehensive system reputation health metrics"""

        # Get basic metrics
        self.reputation_engine.export_state()
        tier_distribution = self.reputation_engine.get_tier_distribution()

        total_nodes = sum(tier_distribution.values())
        high_trust_nodes = (
            tier_distribution.get(ReputationTier.GOLD, 0)
            + tier_distribution.get(ReputationTier.PLATINUM, 0)
            + tier_distribution.get(ReputationTier.DIAMOND, 0)
        )

        untrusted_nodes = tier_distribution.get(ReputationTier.UNTRUSTED, 0)

        # Calculate health scores
        trust_health = high_trust_nodes / max(1, total_nodes)
        untrusted_ratio = untrusted_nodes / max(1, total_nodes)

        # System health classification
        if trust_health > 0.6 and untrusted_ratio < 0.1:
            system_health = "excellent"
        elif trust_health > 0.4 and untrusted_ratio < 0.2:
            system_health = "good"
        elif trust_health > 0.2 and untrusted_ratio < 0.4:
            system_health = "fair"
        else:
            system_health = "poor"

        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": system_health,
            "total_nodes": total_nodes,
            "tier_distribution": {tier.name: count for tier, count in tier_distribution.items()},
            "trust_metrics": {
                "high_trust_ratio": trust_health,
                "untrusted_ratio": untrusted_ratio,
                "average_trust_score": await self._calculate_average_trust_score(),
            },
            "integration_stats": self.integration_stats.copy(),
            "recommendations": await self._generate_system_recommendations(trust_health, untrusted_ratio),
        }

    async def get_node_reputation_report(self, node_id: str) -> Dict[str, Any]:
        """Get comprehensive reputation report for a specific node"""

        insights = self.reputation_engine.get_reputation_insights(node_id)
        if "error" in insights:
            return insights

        # Add integration-specific data
        trust_score = self.reputation_engine.get_trust_score(node_id)
        pricing_multiplier = (await self.get_reputation_pricing_multipliers([node_id])).get(node_id, 1.0)
        scheduling_weight = (await self.get_tier_based_scheduling_weights([node_id])).get(node_id, 1.0)

        # Performance trends
        score = self.reputation_engine.get_reputation_score(node_id)
        trend_analysis = "stable"
        if score and len(score.alpha) > 5:  # Rough trend analysis
            trend_analysis = "improving" if score.alpha > score.beta else "declining"

        return {
            **insights,
            "trust_score": trust_score,
            "pricing_multiplier": pricing_multiplier,
            "scheduling_weight": scheduling_weight,
            "trend_analysis": trend_analysis,
            "recommendations": self._generate_node_recommendations(insights),
            "risk_assessment": self._assess_node_risk(insights),
        }

    # Private methods

    async def _batch_update_loop(self):
        """Background task for batched event processing"""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                time_since_last = datetime.now() - self._last_batch_update
                if (
                    time_since_last.total_seconds() >= 60
                    or len(self._pending_events) >= self.integration_config["batch_update_size"]
                ):
                    await self._flush_pending_events()

            except Exception as e:
                logger.error(f"Error in batch update loop: {e}")
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Background task for collecting system metrics"""

        while True:
            try:
                await asyncio.sleep(self.integration_config["metrics_update_interval"])

                # Reset stats periodically
                if (datetime.now() - self.integration_stats["last_reset"]).total_seconds() > 3600:
                    self.integration_stats = {
                        "events_processed": 0,
                        "batch_updates": 0,
                        "scheduler_queries": 0,
                        "pricing_queries": 0,
                        "last_reset": datetime.now(),
                    }

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(120)

    async def _flush_pending_events(self):
        """Flush pending events to reputation engine"""

        if not self._pending_events:
            return

        events_to_process = self._pending_events.copy()
        self._pending_events.clear()

        for event in events_to_process:
            self.reputation_engine.record_event(event)

        self.integration_stats["batch_updates"] += 1
        self._last_batch_update = datetime.now()

        logger.debug(f"Processed batch of {len(events_to_process)} reputation events")

    async def _calculate_average_trust_score(self) -> float:
        """Calculate average trust score across all nodes"""

        total_trust = 0.0
        node_count = 0

        for node_id in self.reputation_engine.reputation_scores.keys():
            trust_score = self.reputation_engine.get_trust_score(node_id)
            total_trust += trust_score
            node_count += 1

        return total_trust / max(1, node_count)

    async def _generate_system_recommendations(self, trust_health: float, untrusted_ratio: float) -> List[str]:
        """Generate system-level recommendations"""

        recommendations = []

        if trust_health < 0.3:
            recommendations.append("System has low overall trust - investigate node quality issues")

        if untrusted_ratio > 0.3:
            recommendations.append("High percentage of untrusted nodes - consider stricter onboarding")

        if trust_health > 0.7:
            recommendations.append("System trust health is excellent - consider incentivizing high-reputation nodes")

        if not recommendations:
            recommendations.append("System reputation health is stable")

        return recommendations

    def _generate_node_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate node-specific recommendations"""

        recommendations = []
        reputation_score = insights.get("reputation_score", 0)
        uncertainty = insights.get("uncertainty", 0)
        sample_size = insights.get("sample_size", 0)

        if reputation_score < 0.3:
            recommendations.append("Low reputation - consider removing from active pool")
        elif reputation_score < 0.6:
            recommendations.append("Moderate reputation - monitor performance closely")

        if uncertainty > 0.3:
            recommendations.append("High uncertainty - needs more observation data")

        if sample_size < 10:
            recommendations.append("Insufficient data - new node requiring extended evaluation")

        if not recommendations:
            recommendations.append("Node performance within acceptable parameters")

        return recommendations

    def _assess_node_risk(self, insights: Dict[str, Any]) -> str:
        """Assess risk level for a node"""

        reputation_score = insights.get("reputation_score", 0)
        uncertainty = insights.get("uncertainty", 0)

        if reputation_score < 0.3 or uncertainty > 0.5:
            return "high"
        elif reputation_score < 0.6 or uncertainty > 0.3:
            return "medium"
        else:
            return "low"


# Global integration manager instance
_integration_manager: Optional[ReputationIntegrationManager] = None


async def get_integration_manager() -> ReputationIntegrationManager:
    """Get global reputation integration manager instance"""
    global _integration_manager

    if _integration_manager is None:
        _integration_manager = ReputationIntegrationManager()
        await _integration_manager.start()

    return _integration_manager


# Convenience functions for common operations


async def record_task_completion(
    task_id: str, node_id: str, success: bool, completion_time: float, quality_score: Optional[float] = None
):
    """Record task completion for reputation tracking"""

    manager = await get_integration_manager()
    task_result = TaskResult(
        task_id=task_id, node_id=node_id, success=success, completion_time=completion_time, quality_score=quality_score
    )

    await manager.record_task_result(task_result)


async def get_trusted_nodes_for_scheduling(
    available_nodes: List[str], min_trust: float = 0.6, max_nodes: int = 5
) -> List[str]:
    """Get most trusted nodes for task scheduling"""

    manager = await get_integration_manager()
    return await manager.recommend_trusted_nodes({}, available_nodes, min_trust, max_nodes)


async def get_reputation_based_pricing(node_ids: List[str]) -> Dict[str, float]:
    """Get reputation-based pricing multipliers"""

    manager = await get_integration_manager()
    return await manager.get_reputation_pricing_multipliers(node_ids)
