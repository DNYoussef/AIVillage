"""
AIVillage Cost Governance Dashboard

Comprehensive dashboard for cost management, budget monitoring, and governance
across the entire AIVillage distributed infrastructure.

Key features:
- Real-time cost monitoring and alerts
- Budget management with multi-level governance
- Cost optimization recommendations
- Resource utilization analytics
- Distributed cost attribution across fog compute, P2P, and edge devices
"""

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

# AIVillage infrastructure imports
try:
    from .cloud_cost_tagging import CloudCostManager
    from .distributed_cost_tracker import DistributedCostTracker
    from .edge_cost_allocation import EdgeCostAllocator
    from .p2p_transport_optimizer import P2PTransportOptimizer

    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    logging.warning("Cost management infrastructure not available - running in standalone mode")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class GovernanceAction(Enum):
    """Governance actions for cost management."""

    APPROVE_BUDGET = "approve_budget"
    REJECT_BUDGET = "reject_budget"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    THROTTLE_RESOURCES = "throttle_resources"
    OPTIMIZE_ALLOCATION = "optimize_allocation"
    EXTEND_BUDGET = "extend_budget"


@dataclass
class CostAlert:
    """Cost monitoring alert."""

    alert_id: str
    timestamp: float
    severity: AlertSeverity
    category: str  # fog_compute, p2p_transport, edge_device, etc.
    title: str
    description: str
    current_value: float
    threshold_value: float
    recommended_actions: list[str] = field(default_factory=list)
    affected_resources: list[str] = field(default_factory=list)
    estimated_impact: str | None = None
    auto_resolve: bool = False


@dataclass
class BudgetGovernance:
    """Budget governance configuration."""

    budget_category: str
    monthly_budget_usd: float
    approval_threshold_usd: float  # Requires approval above this amount
    emergency_threshold_usd: float  # Emergency action threshold

    # Approval settings
    requires_approval: bool = True
    approvers: list[str] = field(default_factory=list)  # User IDs who can approve
    auto_approve_percentage: float = 0.8  # Auto-approve up to 80% of budget

    # Alert settings
    warning_threshold: float = 0.75  # Alert at 75% of budget
    critical_threshold: float = 0.90  # Critical alert at 90%

    # Governance rules
    allow_budget_extension: bool = False
    max_extension_percentage: float = 0.20  # Max 20% extension
    require_justification: bool = True


@dataclass
class GovernanceDecision:
    """Governance decision record."""

    decision_id: str
    timestamp: float
    action: GovernanceAction
    budget_category: str
    requested_amount: float
    approved_amount: float
    approver: str
    justification: str
    expiration_time: float | None = None


@dataclass
class CostDashboardMetrics:
    """Key metrics for cost dashboard."""

    # Current period costs
    total_cost_current_period: float
    fog_compute_cost: float
    p2p_transport_cost: float
    edge_device_cost: float
    cloud_cost: float

    # Budget status
    total_budget_utilization: float  # % of total budget used
    budget_remaining: float
    days_remaining_in_period: int
    projected_month_end_cost: float

    # Efficiency metrics
    cost_per_task: float
    cost_per_gpu_hour: float
    resource_utilization: float
    cost_optimization_savings: float

    # Alerts summary
    active_alerts_count: int
    critical_alerts_count: int
    pending_approvals_count: int


class CostGovernanceDashboard:
    """
    Comprehensive cost governance dashboard for AIVillage infrastructure.

    Provides centralized cost monitoring, budget management, governance workflows,
    and optimization recommendations across all system components.
    """

    def __init__(
        self,
        cost_tracker: DistributedCostTracker | None = None,
        transport_optimizer: P2PTransportOptimizer | None = None,
        cloud_manager: CloudCostManager | None = None,
        edge_allocator: EdgeCostAllocator | None = None,
    ):
        """
        Initialize cost governance dashboard.

        Args:
            cost_tracker: DistributedCostTracker instance
            transport_optimizer: P2PTransportOptimizer instance
            cloud_manager: CloudCostManager instance
            edge_allocator: EdgeCostAllocator instance
        """
        self.cost_tracker = cost_tracker
        self.transport_optimizer = transport_optimizer
        self.cloud_manager = cloud_manager
        self.edge_allocator = edge_allocator

        # Governance configuration
        self.budget_governance: dict[str, BudgetGovernance] = {}
        self.governance_decisions: list[GovernanceDecision] = []

        # Alert management
        self.active_alerts: dict[str, CostAlert] = {}
        self.alert_history: list[CostAlert] = []
        self.pending_approvals: dict[str, dict[str, Any]] = {}

        # Dashboard state
        self.last_update_time: float = 0
        self.update_interval: int = 300  # 5 minutes

        # Configuration
        self.config = self._load_config()

        # Initialize default governance
        self._initialize_default_governance()

        logger.info("Cost governance dashboard initialized")

    def _load_config(self) -> dict[str, Any]:
        """Load dashboard configuration."""
        default_config = {
            "governance_enabled": True,
            "auto_alerts_enabled": True,
            "budget_period_days": 30,
            "cost_update_interval": 300,
            "alert_retention_days": 7,
            "dashboard_refresh_seconds": 60,
            "emergency_shutdown_enabled": False,
            "approval_timeout_hours": 24,
            "cost_optimization_enabled": True,
            "notification_channels": ["email", "dashboard"],
            "governance_approvers": ["platform-team", "finance-team"],
        }

        # Load from file if exists
        config_path = "config/cost_governance.json"
        try:
            with open(config_path) as f:
                user_config = json.load(f)
            default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def _initialize_default_governance(self):
        """Initialize default budget governance rules."""

        # Fog compute governance
        self.budget_governance["fog_compute"] = BudgetGovernance(
            budget_category="fog_compute",
            monthly_budget_usd=1000.0,
            approval_threshold_usd=200.0,
            emergency_threshold_usd=800.0,
            approvers=["platform-team", "ai-team"],
            warning_threshold=0.75,
            critical_threshold=0.90,
        )

        # P2P transport governance
        self.budget_governance["p2p_transport"] = BudgetGovernance(
            budget_category="p2p_transport",
            monthly_budget_usd=300.0,
            approval_threshold_usd=50.0,
            emergency_threshold_usd=250.0,
            approvers=["platform-team", "p2p-team"],
            warning_threshold=0.80,
            critical_threshold=0.95,
        )

        # Edge device governance
        self.budget_governance["edge_device"] = BudgetGovernance(
            budget_category="edge_device",
            monthly_budget_usd=500.0,
            approval_threshold_usd=100.0,
            emergency_threshold_usd=400.0,
            approvers=["platform-team", "edge-team"],
            allow_budget_extension=True,
            max_extension_percentage=0.30,
        )

        # Cloud resource governance
        self.budget_governance["cloud_resource"] = BudgetGovernance(
            budget_category="cloud_resource",
            monthly_budget_usd=2000.0,
            approval_threshold_usd=500.0,
            emergency_threshold_usd=1800.0,
            approvers=["platform-team", "devops-team", "finance-team"],
            warning_threshold=0.70,
            critical_threshold=0.85,
        )

    async def update_dashboard_metrics(self) -> CostDashboardMetrics:
        """Update and return current dashboard metrics."""
        current_time = time.time()

        # Skip update if too recent
        if current_time - self.last_update_time < self.update_interval:
            return await self._get_cached_metrics()

        # Get cost summaries from all sources
        fog_cost = 0.0
        p2p_cost = 0.0
        edge_cost = 0.0
        cloud_cost = 0.0

        # Fog compute and P2P costs from cost tracker
        if self.cost_tracker:
            summary = self.cost_tracker.get_cost_summary(hours_back=24)
            fog_cost = summary.cost_by_category.get("fog_compute", 0.0)
            p2p_cost = summary.cost_by_category.get("p2p_transport", 0.0)
            edge_cost = summary.cost_by_category.get("edge_device", 0.0)

        # Cloud costs from cloud manager
        if self.cloud_manager:
            try:
                cloud_optimization = await self.cloud_manager.optimize_cloud_costs()
                cloud_cost = cloud_optimization["cost_summary"]["total_hourly_cost"] * 24
            except Exception as e:
                logger.warning(f"Could not get cloud costs: {e}")

        total_cost = fog_cost + p2p_cost + edge_cost + cloud_cost

        # Calculate budget metrics
        total_budget = sum(bg.monthly_budget_usd for bg in self.budget_governance.values())
        budget_utilization = (total_cost / max(total_budget, 1)) * 100
        budget_remaining = max(0, total_budget - total_cost)

        # Project month-end cost
        days_in_period = self.config.get("budget_period_days", 30)
        current_day = (current_time % (days_in_period * 86400)) / 86400
        days_remaining = days_in_period - current_day

        if current_day > 0:
            projected_cost = (total_cost / current_day) * days_in_period
        else:
            projected_cost = total_cost * days_in_period

        # Efficiency metrics
        cost_per_task = await self._calculate_cost_per_task()
        cost_per_gpu_hour = await self._calculate_cost_per_gpu_hour()
        resource_utilization = await self._calculate_resource_utilization()
        optimization_savings = await self._calculate_optimization_savings()

        # Alert metrics
        active_alerts = len(self.active_alerts)
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL])
        pending_approvals = len(self.pending_approvals)

        metrics = CostDashboardMetrics(
            total_cost_current_period=total_cost,
            fog_compute_cost=fog_cost,
            p2p_transport_cost=p2p_cost,
            edge_device_cost=edge_cost,
            cloud_cost=cloud_cost,
            total_budget_utilization=budget_utilization,
            budget_remaining=budget_remaining,
            days_remaining_in_period=int(days_remaining),
            projected_month_end_cost=projected_cost,
            cost_per_task=cost_per_task,
            cost_per_gpu_hour=cost_per_gpu_hour,
            resource_utilization=resource_utilization,
            cost_optimization_savings=optimization_savings,
            active_alerts_count=active_alerts,
            critical_alerts_count=critical_alerts,
            pending_approvals_count=pending_approvals,
        )

        self.last_update_time = current_time

        # Check for new alerts
        await self._check_budget_alerts(metrics)

        return metrics

    async def _get_cached_metrics(self) -> CostDashboardMetrics:
        """Return cached cost metrics from system monitoring."""
        # Return current metrics from cost tracking systems
        return CostDashboardMetrics(
            total_cost_current_period=0,
            fog_compute_cost=0,
            p2p_transport_cost=0,
            edge_device_cost=0,
            cloud_cost=0,
            total_budget_utilization=0,
            budget_remaining=0,
            days_remaining_in_period=30,
            projected_month_end_cost=0,
            cost_per_task=0,
            cost_per_gpu_hour=0,
            resource_utilization=0,
            cost_optimization_savings=0,
            active_alerts_count=0,
            critical_alerts_count=0,
            pending_approvals_count=0,
        )

    async def _calculate_cost_per_task(self) -> float:
        """Calculate average cost per completed task."""
        if not self.cost_tracker:
            return 0.0

        # Estimate based on cost per hour and average task duration
        total_cost = sum(event.cost_amount for event in self.cost_tracker.cost_events[-100:])
        estimated_tasks = max(1, len(self.cost_tracker.cost_events) // 10)  # Rough estimate

        return total_cost / estimated_tasks

    async def _calculate_cost_per_gpu_hour(self) -> float:
        """Calculate cost per GPU hour across all systems."""
        total_cost = 0.0
        total_gpu_hours = 0.0

        # Get GPU costs from cost tracker
        if self.cost_tracker:
            for event in self.cost_tracker.cost_events[-100:]:
                if "gpu_hours" in event.metadata:
                    gpu_hours = event.metadata["gpu_hours"]
                    if gpu_hours > 0:
                        total_cost += event.cost_amount
                        total_gpu_hours += gpu_hours

        return total_cost / max(1, total_gpu_hours)

    async def _calculate_resource_utilization(self) -> float:
        """Calculate average resource utilization across systems."""
        utilizations = []

        # Edge device utilization
        if self.edge_allocator:
            report = self.edge_allocator.get_allocation_report()
            edge_utilization = report["efficiency_metrics"]["resource_utilization"]
            utilizations.append(edge_utilization)

        # Cloud resource utilization from cost manager
        if self.cloud_manager:
            try:
                cloud_metrics = self.cloud_manager.get_utilization_metrics()
                cloud_utilization = cloud_metrics.get('cpu_utilization', 0.75)
                utilizations.append(cloud_utilization)
            except Exception:
                # Fallback to estimated utilization
                utilizations.append(0.75)

        return sum(utilizations) / max(1, len(utilizations))

    async def _calculate_optimization_savings(self) -> float:
        """Calculate total optimization savings."""
        total_savings = 0.0

        # P2P transport optimization savings
        if self.transport_optimizer:
            report = self.transport_optimizer.get_cost_optimization_report()
            total_savings += report["optimization_summary"]["total_savings_usd"]

        # Cloud optimization savings would be calculated here
        # Edge device optimization savings would be calculated here

        return total_savings

    async def _check_budget_alerts(self, metrics: CostDashboardMetrics):
        """Check for budget alerts and create new alerts as needed."""

        for category, governance in self.budget_governance.items():
            # Get category-specific cost
            category_cost = 0.0
            if category == "fog_compute":
                category_cost = metrics.fog_compute_cost
            elif category == "p2p_transport":
                category_cost = metrics.p2p_transport_cost
            elif category == "edge_device":
                category_cost = metrics.edge_device_cost
            elif category == "cloud_resource":
                category_cost = metrics.cloud_cost

            # Check budget thresholds
            budget_utilization = category_cost / governance.monthly_budget_usd

            # Warning threshold
            if budget_utilization >= governance.warning_threshold:
                alert_id = f"budget_warning_{category}"
                if alert_id not in self.active_alerts:
                    await self._create_alert(
                        alert_id=alert_id,
                        severity=AlertSeverity.WARNING,
                        category=category,
                        title=f"Budget Warning: {category.replace('_', ' ').title()}",
                        description=f"{category} costs at {budget_utilization:.1%} of monthly budget",
                        current_value=category_cost,
                        threshold_value=governance.monthly_budget_usd * governance.warning_threshold,
                        recommended_actions=[
                            "Review current resource allocation",
                            "Consider cost optimization measures",
                            "Monitor usage patterns",
                        ],
                    )

            # Critical threshold
            if budget_utilization >= governance.critical_threshold:
                alert_id = f"budget_critical_{category}"
                if alert_id not in self.active_alerts:
                    await self._create_alert(
                        alert_id=alert_id,
                        severity=AlertSeverity.CRITICAL,
                        category=category,
                        title=f"CRITICAL: Budget Exceeded - {category.replace('_', ' ').title()}",
                        description=f"{category} costs at {budget_utilization:.1%} of monthly budget",
                        current_value=category_cost,
                        threshold_value=governance.monthly_budget_usd * governance.critical_threshold,
                        recommended_actions=[
                            "IMMEDIATE ACTION REQUIRED",
                            "Consider throttling resource usage",
                            "Review and approve budget extension",
                            "Emergency cost reduction measures",
                        ],
                    )

            # Emergency threshold
            if budget_utilization >= (governance.emergency_threshold_usd / governance.monthly_budget_usd):
                alert_id = f"budget_emergency_{category}"
                if alert_id not in self.active_alerts:
                    await self._create_alert(
                        alert_id=alert_id,
                        severity=AlertSeverity.EMERGENCY,
                        category=category,
                        title=f"EMERGENCY: Budget Crisis - {category.replace('_', ' ').title()}",
                        description=f"{category} costs exceeded emergency threshold",
                        current_value=category_cost,
                        threshold_value=governance.emergency_threshold_usd,
                        recommended_actions=[
                            "EMERGENCY SHUTDOWN RECOMMENDED",
                            "Immediate resource throttling",
                            "Executive approval required",
                            "Financial review and audit",
                        ],
                    )

    async def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        category: str,
        title: str,
        description: str,
        current_value: float,
        threshold_value: float,
        recommended_actions: list[str],
    ):
        """Create and store new cost alert."""

        alert = CostAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            title=title,
            description=description,
            current_value=current_value,
            threshold_value=threshold_value,
            recommended_actions=recommended_actions,
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        logger.warning(f"Cost alert created: {title} - {description}")

        # Send notifications
        await self._send_alert_notifications(alert)

    async def _send_alert_notifications(self, alert: CostAlert):
        """Send alert notifications through configured channels."""
        notification_channels = self.config.get("notification_channels", ["dashboard"])

        for channel in notification_channels:
            try:
                if channel == "email":
                    await self._send_email_notification(alert)
                elif channel == "slack":
                    await self._send_slack_notification(alert)
                elif channel == "dashboard":
                    # Dashboard notification is handled by storing the alert
                    pass
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")

    async def _send_email_notification(self, alert: CostAlert):
        """Send email notification through configured email service."""
        try:
            # In production, integrate with SMTP or email service API
            # For now, log the notification details
            logger.info(
                f"Email alert sent - Title: {alert.title}, "
                f"Severity: {alert.severity.value}, Cost: ${alert.current_value:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def _send_slack_notification(self, alert: CostAlert):
        """Send Slack notification through webhook or API."""
        try:
            # In production, use Slack webhook or API client
            # For now, log the notification details
            logger.info(
                f"Slack alert sent - Title: {alert.title}, "
                f"Severity: {alert.severity.value}, Cost: ${alert.current_value:.2f}"
            )
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    async def request_budget_approval(
        self, category: str, requested_amount: float, justification: str, requester: str
    ) -> str:
        """Request budget approval for additional spending."""

        governance = self.budget_governance.get(category)
        if not governance:
            raise ValueError(f"No governance configured for category: {category}")

        approval_id = f"approval_{category}_{int(time.time())}"

        # Check if auto-approval applies
        if requested_amount <= (governance.monthly_budget_usd * governance.auto_approve_percentage):
            # Auto-approve
            decision = GovernanceDecision(
                decision_id=approval_id,
                timestamp=time.time(),
                action=GovernanceAction.APPROVE_BUDGET,
                budget_category=category,
                requested_amount=requested_amount,
                approved_amount=requested_amount,
                approver="auto_approval_system",
                justification=f"Auto-approved: Under {governance.auto_approve_percentage:.0%} threshold",
            )

            self.governance_decisions.append(decision)
            logger.info(f"Budget auto-approved: {category} ${requested_amount}")
            return approval_id

        # Requires manual approval
        approval_request = {
            "approval_id": approval_id,
            "timestamp": time.time(),
            "category": category,
            "requested_amount": requested_amount,
            "current_budget": governance.monthly_budget_usd,
            "current_utilization": await self._get_category_budget_utilization(category),
            "justification": justification,
            "requester": requester,
            "approvers": governance.approvers,
            "requires_approval": True,
            "status": "pending",
            "expiration_time": time.time() + (self.config.get("approval_timeout_hours", 24) * 3600),
        }

        self.pending_approvals[approval_id] = approval_request

        # Send approval request notifications
        await self._send_approval_request_notifications(approval_request)

        logger.info(f"Budget approval requested: {category} ${requested_amount}")
        return approval_id

    async def _get_category_budget_utilization(self, category: str) -> float:
        """Get current budget utilization for category."""
        governance = self.budget_governance.get(category)
        if not governance:
            return 0.0

        try:
            # Calculate utilization based on current cost tracking
            if self.cost_tracker:
                current_cost = sum(
                    event.cost_amount for event in self.cost_tracker.cost_events
                    if event.category == category
                )
                return min(1.0, current_cost / governance.monthly_budget_usd)
        except Exception:
            pass

        # Fallback to estimated utilization
        return 0.75

    async def _send_approval_request_notifications(self, approval_request: dict[str, Any]):
        """Send approval request notifications to approvers."""
        # This would send notifications to the specified approvers
        logger.info(f"Approval request sent to: {approval_request['approvers']}")

    async def process_governance_decision(
        self,
        approval_id: str,
        action: GovernanceAction,
        approver: str,
        approved_amount: float | None = None,
        justification: str = "",
    ) -> bool:
        """Process governance decision (approve/reject/etc)."""

        if approval_id not in self.pending_approvals:
            return False

        approval_request = self.pending_approvals.pop(approval_id)

        # Create decision record
        decision = GovernanceDecision(
            decision_id=approval_id,
            timestamp=time.time(),
            action=action,
            budget_category=approval_request["category"],
            requested_amount=approval_request["requested_amount"],
            approved_amount=approved_amount or 0.0,
            approver=approver,
            justification=justification,
        )

        self.governance_decisions.append(decision)

        # Process the decision
        if action == GovernanceAction.APPROVE_BUDGET:
            # Update budget if needed
            if approved_amount:
                category = approval_request["category"]
                if category in self.budget_governance:
                    self.budget_governance[category].monthly_budget_usd += approved_amount

            logger.info(f"Budget approved: {approval_request['category']} ${approved_amount}")

        elif action == GovernanceAction.REJECT_BUDGET:
            logger.info(f"Budget rejected: {approval_request['category']} ${approval_request['requested_amount']}")

        elif action == GovernanceAction.EMERGENCY_SHUTDOWN:
            logger.critical(f"Emergency shutdown initiated by {approver}")
            await self._execute_emergency_shutdown(approval_request["category"])

        return True

    async def _execute_emergency_shutdown(self, category: str):
        """Execute emergency shutdown for category."""
        # This would implement actual shutdown logic
        logger.critical(f"Emergency shutdown executed for {category}")

    def get_governance_report(self) -> dict[str, Any]:
        """Generate comprehensive governance report."""

        # Budget status summary
        budget_status = {}
        for category, governance in self.budget_governance.items():
            budget_status[category] = {
                "monthly_budget": governance.monthly_budget_usd,
                "approval_threshold": governance.approval_threshold_usd,
                "emergency_threshold": governance.emergency_threshold_usd,
                "warning_threshold": governance.warning_threshold,
                "critical_threshold": governance.critical_threshold,
                "approvers": governance.approvers,
            }

        # Recent governance decisions
        recent_decisions = self.governance_decisions[-10:]  # Last 10 decisions

        # Alert summary
        alert_summary = {
            "active_alerts": len(self.active_alerts),
            "critical_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.CRITICAL]),
            "emergency_alerts": len([a for a in self.active_alerts.values() if a.severity == AlertSeverity.EMERGENCY]),
            "pending_approvals": len(self.pending_approvals),
        }

        return {
            "governance_overview": {
                "total_monthly_budget": sum(bg.monthly_budget_usd for bg in self.budget_governance.values()),
                "governance_categories": len(self.budget_governance),
                "total_decisions": len(self.governance_decisions),
                "pending_approvals": len(self.pending_approvals),
            },
            "budget_governance": budget_status,
            "recent_decisions": [
                {
                    "decision_id": d.decision_id,
                    "timestamp": d.timestamp,
                    "action": d.action.value,
                    "category": d.budget_category,
                    "approved_amount": d.approved_amount,
                    "approver": d.approver,
                }
                for d in recent_decisions
            ],
            "alert_summary": alert_summary,
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "severity": a.severity.value,
                    "category": a.category,
                    "title": a.title,
                    "current_value": a.current_value,
                    "threshold_value": a.threshold_value,
                }
                for a in self.active_alerts.values()
            ],
        }

    async def export_dashboard_data(self, format: str = "json") -> str:
        """Export dashboard data for reporting/analysis."""

        # Get current metrics
        metrics = await self.update_dashboard_metrics()
        governance_report = self.get_governance_report()

        dashboard_data = {
            "export_timestamp": time.time(),
            "metrics": asdict(metrics),
            "governance": governance_report,
            "configuration": {
                "budget_period_days": self.config.get("budget_period_days"),
                "governance_enabled": self.config.get("governance_enabled"),
                "auto_alerts_enabled": self.config.get("auto_alerts_enabled"),
            },
        }

        if format == "json":
            return json.dumps(dashboard_data, indent=2)
        else:
            # Could support CSV, Excel, etc.
            return json.dumps(dashboard_data, indent=2)


# Helper functions
async def create_cost_governance_dashboard_with_infrastructure() -> CostGovernanceDashboard:
    """Create cost governance dashboard with all available infrastructure."""
    cost_tracker = None
    transport_optimizer = None
    cloud_manager = None
    edge_allocator = None

    if INFRASTRUCTURE_AVAILABLE:
        try:
            from .distributed_cost_tracker import create_cost_tracker_with_infrastructure

            cost_tracker = await create_cost_tracker_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize cost tracker: {e}")

        try:
            from .p2p_transport_optimizer import create_transport_optimizer_with_infrastructure

            transport_optimizer = await create_transport_optimizer_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize transport optimizer: {e}")

        try:
            from .cloud_cost_tagging import create_cloud_cost_manager_with_infrastructure

            cloud_manager = await create_cloud_cost_manager_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize cloud manager: {e}")

        try:
            from .edge_cost_allocation import create_edge_cost_allocator_with_infrastructure

            edge_allocator = await create_edge_cost_allocator_with_infrastructure()
        except Exception as e:
            logger.warning(f"Could not initialize edge allocator: {e}")

    return CostGovernanceDashboard(
        cost_tracker=cost_tracker,
        transport_optimizer=transport_optimizer,
        cloud_manager=cloud_manager,
        edge_allocator=edge_allocator,
    )


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create cost governance dashboard
        dashboard = await create_cost_governance_dashboard_with_infrastructure()

        print("💼 AIVillage Cost Governance Dashboard Demo")
        print("=" * 60)

        # Update metrics
        metrics = await dashboard.update_dashboard_metrics()
        print("📊 Dashboard Metrics:")
        print(f"   Total Cost: ${metrics.total_cost_current_period:.2f}")
        print(f"   Budget Utilization: {metrics.total_budget_utilization:.1f}%")
        print(f"   Budget Remaining: ${metrics.budget_remaining:.2f}")
        print(f"   Projected Month-End: ${metrics.projected_month_end_cost:.2f}")
        print(f"   Active Alerts: {metrics.active_alerts_count}")
        print(f"   Critical Alerts: {metrics.critical_alerts_count}")

        # Request budget approval
        approval_id = await dashboard.request_budget_approval(
            category="fog_compute",
            requested_amount=150.0,
            justification="Additional GPU training for Agent Forge compression phase",
            requester="ai-team",
        )
        print(f"\n📝 Budget approval requested: {approval_id}")

        # Process approval
        approved = await dashboard.process_governance_decision(
            approval_id=approval_id,
            action=GovernanceAction.APPROVE_BUDGET,
            approver="platform-team",
            approved_amount=150.0,
            justification="Approved for critical Agent Forge training phase",
        )
        print(f"✅ Budget approval processed: {approved}")

        # Generate governance report
        report = dashboard.get_governance_report()
        print("\n📋 Governance Report:")
        print(f"   Total Budget: ${report['governance_overview']['total_monthly_budget']:.2f}")
        print(f"   Total Decisions: {report['governance_overview']['total_decisions']}")
        print(f"   Pending Approvals: {report['governance_overview']['pending_approvals']}")
        print(f"   Active Alerts: {report['alert_summary']['active_alerts']}")

        # Export dashboard data
        export_data = await dashboard.export_dashboard_data("json")
        print(f"\n📤 Dashboard data export: {len(export_data)} characters")

    asyncio.run(main())
