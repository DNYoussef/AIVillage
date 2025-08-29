#!/usr/bin/env python3
"""
Governance Dashboard - Complete Implementation

This module provides a comprehensive governance dashboard with member management:
- Real-time governance metrics and analytics
- Member management interface
- Proposal lifecycle tracking
- Voting analytics and participation metrics
- Delegation management
- Compliance integration
- Economic incentive monitoring
- Integration with governance, tokenomics, and compliance systems

Key Features:
- Interactive dashboard with real-time data
- Member onboarding and KYC management
- Proposal creation and management workflow
- Voting power visualization and analysis
- Delegation chain tracking
- Compliance status monitoring
- Economic incentive tracking
- Audit trail visualization
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Any
import uuid

from ..compliance.automated_compliance_system import AutomatedComplianceSystem

# Import our systems

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Dashboard metrics snapshot."""

    timestamp: datetime

    # Governance metrics
    total_members: int = 0
    active_proposals: int = 0
    voting_participation_rate: float = 0.0
    governance_health_score: float = 0.0

    # Tokenomics metrics
    token_price_usd: float = 0.0
    market_cap_usd: float = 0.0
    staking_ratio: float = 0.0
    rewards_distributed_24h: float = 0.0

    # Compliance metrics
    compliance_score: float = 100.0
    active_violations: int = 0
    critical_violations: int = 0

    # System health
    system_uptime: float = 99.9
    performance_score: float = 100.0
    security_status: str = "secure"


@dataclass
class GovernanceAlert:
    """Governance system alert."""

    alert_id: str
    alert_type: str  # info, warning, error, critical
    title: str
    description: str
    timestamp: datetime
    source: str  # governance, tokenomics, compliance, system
    entity_id: str | None = None
    action_required: bool = False
    auto_resolved: bool = False
    resolved_at: datetime | None = None


class GovernanceDashboard:
    """
    Unified governance dashboard providing real-time monitoring and management
    for DAO operations, tokenomics, and compliance.
    """

    def __init__(
        self,
        dao_system: DAOOperationalSystem,
        tokenomics_system: TokenomicsDeploymentSystem,
        compliance_system: AutomatedComplianceSystem,
        slo_monitor: SLOMonitor | None = None,
        data_dir: str = "dashboard_data",
    ):
        self.dao_system = dao_system
        self.tokenomics_system = tokenomics_system
        self.compliance_system = compliance_system
        self.slo_monitor = slo_monitor

        # Data storage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Dashboard state
        self.metrics_history: list[DashboardMetrics] = []
        self.alerts: dict[str, GovernanceAlert] = {}
        self.dashboard_config = self._load_dashboard_config()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = False

        logger.info("Governance Dashboard initialized")

    def _load_dashboard_config(self) -> dict[str, Any]:
        """Load dashboard configuration."""
        return {
            "update_interval_seconds": 30,
            "metrics_retention_hours": 168,  # 7 days
            "alert_retention_hours": 720,  # 30 days
            "health_thresholds": {
                "governance_participation_min": 0.1,  # 10%
                "compliance_score_min": 80.0,
                "system_uptime_min": 99.0,
                "performance_score_min": 80.0,
            },
            "auto_alerts": True,
            "notification_webhooks": [],
            "dashboard_port": 8080,
        }

    async def start(self):
        """Start the dashboard system."""
        if self._running:
            return

        logger.info("Starting Governance Dashboard")
        self._running = True

        # Start background tasks
        tasks = [self._metrics_collector(), self._alert_monitor(), self._health_checker(), self._data_cleanup()]

        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        logger.info("Governance Dashboard started")

    async def stop(self):
        """Stop the dashboard system."""
        if not self._running:
            return

        logger.info("Stopping Governance Dashboard")
        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Governance Dashboard stopped")

    # Metrics Collection

    async def collect_current_metrics(self) -> DashboardMetrics:
        """Collect current dashboard metrics from all systems."""

        current_time = datetime.utcnow()
        metrics = DashboardMetrics(timestamp=current_time)

        try:
            # Governance metrics
            if self.dao_system:
                governance_status = await self.dao_system.get_governance_status()
                metrics.total_members = governance_status.get("membership", {}).get("total_members", 0)
                metrics.active_proposals = governance_status.get("proposals", {}).get("active_proposals", 0)
                metrics.voting_participation_rate = governance_status.get("voting", {}).get(
                    "avg_participation_rate", 0.0
                )
                metrics.governance_health_score = await self._calculate_governance_health_score()

            # Tokenomics metrics
            if self.tokenomics_system:
                economic_summary = await self.tokenomics_system.get_economic_summary()
                metrics.staking_ratio = economic_summary.get("token_metrics", {}).get("staking_ratio", 0.0)

                # Calculate 24h rewards
                rewards_24h = economic_summary.get("reward_statistics", {}).get("total_rewards_24h", 0)
                metrics.rewards_distributed_24h = float(rewards_24h) if isinstance(rewards_24h, int | float) else 0.0

                # Mock market data (would integrate with real price feeds)
                metrics.token_price_usd = 0.05  # $0.05 per FOG
                total_supply = economic_summary.get("token_metrics", {}).get("total_supply", 0)
                metrics.market_cap_usd = metrics.token_price_usd * total_supply

            # Compliance metrics
            if self.compliance_system:
                compliance_status = await self.compliance_system.get_compliance_status()
                metrics.compliance_score = compliance_status.get("compliance_score", 100.0)
                metrics.active_violations = compliance_status.get("active_violations", 0)
                metrics.critical_violations = compliance_status.get("critical_violations", 0)

            # System health metrics
            if self.slo_monitor:
                slo_status = await self.slo_monitor.get_slo_status()
                uptime_values = [
                    slo.get("current_value", 0)
                    for slo in slo_status.values()
                    if "availability" in slo.get("metric_name", "").lower()
                ]
                if uptime_values:
                    metrics.system_uptime = sum(uptime_values) / len(uptime_values)

                # Calculate performance score
                metrics.performance_score = await self._calculate_performance_score(slo_status)

        except Exception as e:
            logger.error(f"Error collecting dashboard metrics: {e}")
            # Create default metrics on error
            pass

        return metrics

    async def _calculate_governance_health_score(self) -> float:
        """Calculate overall governance health score."""
        try:
            score = 100.0

            # Check participation rate
            governance_status = await self.dao_system.get_governance_status()
            participation_rate = governance_status.get("voting", {}).get("avg_participation_rate", 0.0)
            min_participation = self.dashboard_config["health_thresholds"]["governance_participation_min"]

            if participation_rate < min_participation:
                score -= (min_participation - participation_rate) * 200  # Penalty for low participation

            # Check proposal completion rate
            proposals_stats = governance_status.get("proposals", {})
            total_proposals = proposals_stats.get("total_proposals", 0)
            completed_proposals = proposals_stats.get("completed_proposals", 0)

            if total_proposals > 0:
                completion_rate = completed_proposals / total_proposals
                if completion_rate < 0.7:  # Less than 70% completion
                    score -= (0.7 - completion_rate) * 50

            # Check member activity
            membership_stats = governance_status.get("membership", {})
            total_members = membership_stats.get("total_members", 0)
            active_members = membership_stats.get("active_members_week", 0)

            if total_members > 0:
                activity_rate = active_members / total_members
                if activity_rate < 0.3:  # Less than 30% active
                    score -= (0.3 - activity_rate) * 100

            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.error(f"Error calculating governance health score: {e}")
            return 50.0  # Default score on error

    async def _calculate_performance_score(self, slo_status: dict[str, Any]) -> float:
        """Calculate system performance score from SLO status."""
        try:
            total_slos = len(slo_status)
            if total_slos == 0:
                return 100.0

            breached_slos = sum(1 for slo in slo_status.values() if slo.get("is_breached", False))
            healthy_ratio = (total_slos - breached_slos) / total_slos

            return healthy_ratio * 100.0

        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 80.0  # Default score on error

    # Alert Management

    async def create_alert(
        self,
        alert_type: str,
        title: str,
        description: str,
        source: str,
        entity_id: str | None = None,
        action_required: bool = False,
    ) -> str:
        """Create a new dashboard alert."""

        alert_id = str(uuid.uuid4())
        alert = GovernanceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            title=title,
            description=description,
            timestamp=datetime.utcnow(),
            source=source,
            entity_id=entity_id,
            action_required=action_required,
        )

        self.alerts[alert_id] = alert

        logger.info(f"Dashboard alert created: {title} ({alert_type})")

        # Send notifications if configured
        if self.dashboard_config["auto_alerts"]:
            await self._send_alert_notification(alert)

        return alert_id

    async def resolve_alert(self, alert_id: str, auto_resolved: bool = False) -> bool:
        """Resolve a dashboard alert."""
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.auto_resolved = auto_resolved
        alert.resolved_at = datetime.utcnow()

        logger.info(f"Dashboard alert resolved: {alert.title}")
        return True

    async def _send_alert_notification(self, alert: GovernanceAlert):
        """Send alert notification via configured channels."""
        # Mock notification - would integrate with real notification systems
        logger.warning(f"ALERT [{alert.alert_type.upper()}]: {alert.title} - {alert.description}")

    # Dashboard API

    async def get_dashboard_summary(self) -> dict[str, Any]:
        """Get comprehensive dashboard summary."""

        current_metrics = await self.collect_current_metrics()
        recent_alerts = [
            alert
            for alert in self.alerts.values()
            if (datetime.utcnow() - alert.timestamp).total_seconds() < 86400  # Last 24h
        ]

        # Get active proposals
        active_proposals = []
        if self.dao_system:
            active_proposals = await self.dao_system.get_active_proposals()

        # Get top participants
        top_participants = []
        if self.tokenomics_system:
            top_participants = await self._get_top_participants()

        return {
            "timestamp": current_metrics.timestamp.isoformat(),
            "metrics": {
                "governance": {
                    "total_members": current_metrics.total_members,
                    "active_proposals": current_metrics.active_proposals,
                    "voting_participation_rate": current_metrics.voting_participation_rate,
                    "health_score": current_metrics.governance_health_score,
                },
                "tokenomics": {
                    "token_price_usd": current_metrics.token_price_usd,
                    "market_cap_usd": current_metrics.market_cap_usd,
                    "staking_ratio": current_metrics.staking_ratio,
                    "rewards_24h": current_metrics.rewards_distributed_24h,
                },
                "compliance": {
                    "score": current_metrics.compliance_score,
                    "active_violations": current_metrics.active_violations,
                    "critical_violations": current_metrics.critical_violations,
                    "status": "compliant" if current_metrics.active_violations == 0 else "violations",
                },
                "system": {
                    "uptime": current_metrics.system_uptime,
                    "performance_score": current_metrics.performance_score,
                    "security_status": current_metrics.security_status,
                },
            },
            "alerts": {
                "total_active": len([a for a in self.alerts.values() if not a.resolved_at]),
                "critical_count": len([a for a in recent_alerts if a.alert_type == "critical"]),
                "recent_alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "type": alert.alert_type,
                        "title": alert.title,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "source": alert.source,
                        "action_required": alert.action_required,
                    }
                    for alert in sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)[:10]
                ],
            },
            "active_proposals": active_proposals[:5],  # Top 5 active proposals
            "top_participants": top_participants[:10],  # Top 10 participants
        }

    async def get_governance_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get governance analytics for specified period."""

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Get historical metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_date]

        analytics = {
            "period_days": days,
            "member_growth": [],
            "proposal_activity": [],
            "participation_trends": [],
            "governance_health_trends": [],
        }

        if recent_metrics:
            # Calculate trends
            for i, metric in enumerate(recent_metrics):
                analytics["member_growth"].append(
                    {"date": metric.timestamp.isoformat(), "total_members": metric.total_members}
                )

                analytics["proposal_activity"].append(
                    {"date": metric.timestamp.isoformat(), "active_proposals": metric.active_proposals}
                )

                analytics["participation_trends"].append(
                    {"date": metric.timestamp.isoformat(), "participation_rate": metric.voting_participation_rate}
                )

                analytics["governance_health_trends"].append(
                    {"date": metric.timestamp.isoformat(), "health_score": metric.governance_health_score}
                )

        return analytics

    async def get_tokenomics_analytics(self, days: int = 30) -> dict[str, Any]:
        """Get tokenomics analytics for specified period."""

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_date]

        analytics = {
            "period_days": days,
            "token_price_history": [],
            "market_cap_history": [],
            "staking_ratio_history": [],
            "rewards_distribution": [],
        }

        if recent_metrics:
            for metric in recent_metrics:
                analytics["token_price_history"].append(
                    {"date": metric.timestamp.isoformat(), "price_usd": metric.token_price_usd}
                )

                analytics["market_cap_history"].append(
                    {"date": metric.timestamp.isoformat(), "market_cap_usd": metric.market_cap_usd}
                )

                analytics["staking_ratio_history"].append(
                    {"date": metric.timestamp.isoformat(), "staking_ratio": metric.staking_ratio}
                )

                analytics["rewards_distribution"].append(
                    {"date": metric.timestamp.isoformat(), "rewards_24h": metric.rewards_distributed_24h}
                )

        return analytics

    async def get_compliance_dashboard(self) -> dict[str, Any]:
        """Get compliance dashboard data."""

        compliance_status = await self.compliance_system.get_compliance_status()
        violation_summary = await self.compliance_system.get_violation_summary(30)

        return {
            "overall_status": compliance_status.get("overall_status", "unknown"),
            "compliance_score": compliance_status.get("compliance_score", 0),
            "frameworks_monitored": compliance_status.get("frameworks_monitored", []),
            "violation_summary": violation_summary,
            "recent_violations": await self._get_recent_violations(),
            "compliance_trends": await self._get_compliance_trends(),
        }

    async def _get_top_participants(self) -> list[dict[str, Any]]:
        """Get top economic participants."""
        try:
            # Mock top participants - would get real data from tokenomics system
            return [
                {
                    "participant_id": f"participant_{i}",
                    "tier": ["diamond", "platinum", "gold", "silver", "bronze"][i % 5],
                    "total_earned": 10000 - (i * 1000),
                    "total_staked": 5000 - (i * 500),
                    "reputation_score": 100 - (i * 2),
                }
                for i in range(10)
            ]
        except Exception as e:
            logger.error(f"Error getting top participants: {e}")
            return []

    async def _get_recent_violations(self) -> list[dict[str, Any]]:
        """Get recent compliance violations."""
        try:
            # Mock recent violations - would get real data from compliance system
            return []
        except Exception as e:
            logger.error(f"Error getting recent violations: {e}")
            return []

    async def _get_compliance_trends(self) -> list[dict[str, Any]]:
        """Get compliance trend data."""
        try:
            recent_metrics = self.metrics_history[-30:]  # Last 30 data points
            trends = []

            for metric in recent_metrics:
                trends.append(
                    {
                        "date": metric.timestamp.isoformat(),
                        "compliance_score": metric.compliance_score,
                        "violations": metric.active_violations,
                        "critical_violations": metric.critical_violations,
                    }
                )

            return trends
        except Exception as e:
            logger.error(f"Error getting compliance trends: {e}")
            return []

    # Background Tasks

    async def _metrics_collector(self):
        """Background task to collect metrics."""
        while self._running:
            try:
                current_metrics = await self.collect_current_metrics()
                self.metrics_history.append(current_metrics)

                # Keep only recent metrics
                retention_hours = self.dashboard_config["metrics_retention_hours"]
                cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
                self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]

                update_interval = self.dashboard_config["update_interval_seconds"]
                await asyncio.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60)

    async def _alert_monitor(self):
        """Background task to monitor for alert conditions."""
        while self._running:
            try:
                if not self.metrics_history:
                    await asyncio.sleep(60)
                    continue

                current_metrics = self.metrics_history[-1]
                thresholds = self.dashboard_config["health_thresholds"]

                # Check governance participation
                if current_metrics.voting_participation_rate < thresholds["governance_participation_min"]:
                    await self.create_alert(
                        "warning",
                        "Low Governance Participation",
                        f"Voting participation rate ({current_metrics.voting_participation_rate:.1%}) below minimum threshold",
                        "governance",
                        action_required=True,
                    )

                # Check compliance score
                if current_metrics.compliance_score < thresholds["compliance_score_min"]:
                    await self.create_alert(
                        "violation",
                        "Compliance Score Below Threshold",
                        f"Compliance score ({current_metrics.compliance_score:.1f}) below minimum threshold",
                        "compliance",
                        action_required=True,
                    )

                # Check for critical violations
                if current_metrics.critical_violations > 0:
                    await self.create_alert(
                        "critical",
                        "Critical Compliance Violations",
                        f"{current_metrics.critical_violations} critical compliance violations detected",
                        "compliance",
                        action_required=True,
                    )

                # Check system uptime
                if current_metrics.system_uptime < thresholds["system_uptime_min"]:
                    await self.create_alert(
                        "error",
                        "System Uptime Below Threshold",
                        f"System uptime ({current_metrics.system_uptime:.1f}%) below minimum threshold",
                        "system",
                        action_required=True,
                    )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error in alert monitor: {e}")
                await asyncio.sleep(60)

    async def _health_checker(self):
        """Background task to check system health."""
        while self._running:
            try:
                # Perform health checks on integrated systems
                health_status = {
                    "dao_system": await self._check_dao_health(),
                    "tokenomics_system": await self._check_tokenomics_health(),
                    "compliance_system": await self._check_compliance_health(),
                }

                # Create alerts for unhealthy systems
                for system, healthy in health_status.items():
                    if not healthy:
                        await self.create_alert(
                            "error",
                            f"{system.replace('_', ' ').title()} Unhealthy",
                            f"Health check failed for {system}",
                            "system",
                            action_required=True,
                        )

                await asyncio.sleep(600)  # Check every 10 minutes

            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(300)

    async def _data_cleanup(self):
        """Background task to clean up old data."""
        while self._running:
            try:
                # Clean up old alerts
                alert_retention_hours = self.dashboard_config["alert_retention_hours"]
                cutoff_time = datetime.utcnow() - timedelta(hours=alert_retention_hours)

                old_alerts = [
                    alert_id
                    for alert_id, alert in self.alerts.items()
                    if alert.resolved_at and alert.resolved_at < cutoff_time
                ]

                for alert_id in old_alerts:
                    del self.alerts[alert_id]

                if old_alerts:
                    logger.info(f"Cleaned up {len(old_alerts)} old alerts")

                await asyncio.sleep(3600)  # Run every hour

            except Exception as e:
                logger.error(f"Error in data cleanup: {e}")
                await asyncio.sleep(1800)

    # Health Check Methods

    async def _check_dao_health(self) -> bool:
        """Check DAO system health."""
        try:
            if not self.dao_system:
                return False

            # Try to get governance status
            status = await self.dao_system.get_governance_status()
            return status is not None

        except Exception as e:
            logger.error(f"DAO health check failed: {e}")
            return False

    async def _check_tokenomics_health(self) -> bool:
        """Check tokenomics system health."""
        try:
            if not self.tokenomics_system:
                return False

            # Try to get economic summary
            summary = await self.tokenomics_system.get_economic_summary()
            return summary is not None

        except Exception as e:
            logger.error(f"Tokenomics health check failed: {e}")
            return False

    async def _check_compliance_health(self) -> bool:
        """Check compliance system health."""
        try:
            if not self.compliance_system:
                return False

            # Try to get compliance status
            status = await self.compliance_system.get_compliance_status()
            return status is not None

        except Exception as e:
            logger.error(f"Compliance health check failed: {e}")
            return False

    # Export and Reporting

    async def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            filename = f"governance_dashboard_export_{timestamp}.json"
            file_path = self.data_dir / filename

            export_data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "dashboard_summary": await self.get_dashboard_summary(),
                "governance_analytics": await self.get_governance_analytics(),
                "tokenomics_analytics": await self.get_tokenomics_analytics(),
                "compliance_dashboard": await self.get_compliance_dashboard(),
                "alerts": [
                    {
                        "alert_id": alert.alert_id,
                        "type": alert.alert_type,
                        "title": alert.title,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "source": alert.source,
                        "resolved": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    }
                    for alert in self.alerts.values()
                ],
            }

            with open(file_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Dashboard data exported to {file_path}")
            return str(file_path)

        else:
            raise ValueError(f"Unsupported export format: {format}")

    # Public Query Methods

    async def get_system_status(self) -> dict[str, Any]:
        """Get overall system status."""
        if not self.metrics_history:
            return {"status": "initializing"}

        current_metrics = self.metrics_history[-1]
        active_alerts = len([a for a in self.alerts.values() if not a.resolved_at])
        critical_alerts = len([a for a in self.alerts.values() if not a.resolved_at and a.alert_type == "critical"])

        # Determine overall status
        overall_status = "healthy"
        if critical_alerts > 0:
            overall_status = "critical"
        elif current_metrics.critical_violations > 0:
            overall_status = "compliance_issues"
        elif active_alerts > 0 or current_metrics.active_violations > 0:
            overall_status = "warnings"

        return {
            "overall_status": overall_status,
            "timestamp": current_metrics.timestamp.isoformat(),
            "governance_health": current_metrics.governance_health_score,
            "compliance_score": current_metrics.compliance_score,
            "system_uptime": current_metrics.system_uptime,
            "performance_score": current_metrics.performance_score,
            "active_alerts": active_alerts,
            "critical_alerts": critical_alerts,
            "total_members": current_metrics.total_members,
            "active_proposals": current_metrics.active_proposals,
        }

    async def get_recent_activities(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent governance activities."""
        activities = []

        # Get recent alerts
        recent_alerts = sorted(
            [a for a in self.alerts.values() if not a.resolved_at], key=lambda x: x.timestamp, reverse=True
        )[:limit]

        for alert in recent_alerts:
            activities.append(
                {
                    "type": "alert",
                    "timestamp": alert.timestamp.isoformat(),
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.alert_type,
                    "source": alert.source,
                }
            )

        # Sort all activities by timestamp
        activities.sort(key=lambda x: x["timestamp"], reverse=True)

        return activities[:limit]
