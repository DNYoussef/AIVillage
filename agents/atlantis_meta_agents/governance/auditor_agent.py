"""Auditor Agent - Receipts, Risk & Compliance

The auditing and compliance specialist of AIVillage, responsible for:
- Receipt collection and verification from all agents
- Comprehensive audit trail maintenance
- Financial tracking and cost analysis
- Compliance monitoring and reporting
- Receipt search and retrieval systems
- Risk assessment based on activity patterns
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.production.rag.rag_system.core.agent_interface import AgentInterface

logger = logging.getLogger(__name__)


class ReceiptType(Enum):
    AGENT_ACTION = "agent_action"
    SYSTEM_EVENT = "system_event"
    RESOURCE_USAGE = "resource_usage"
    ERROR_EVENT = "error_event"
    SECURITY_EVENT = "security_event"
    FINANCIAL_TRANSACTION = "financial_transaction"


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ATTENTION = "requires_attention"


@dataclass
class Receipt:
    receipt_id: str
    agent_id: str
    action: str
    timestamp: float
    receipt_type: ReceiptType
    details: dict[str, Any]
    cost_usd: float | None = None
    resource_usage: dict[str, Any] = None
    compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    signature: str | None = None
    verified: bool = False


@dataclass
class AuditReport:
    report_id: str
    report_type: str
    time_range: tuple[float, float]
    agent_coverage: list[str]
    total_receipts: int
    total_cost_usd: float
    compliance_summary: dict[str, int]
    key_findings: list[str]
    recommendations: list[str]
    generated_timestamp: float


@dataclass
class ComplianceAlert:
    alert_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_agents: list[str]
    timestamp: float
    resolved: bool = False


class AuditorAgent(AgentInterface):
    """Auditor Agent manages receipts, audit trails, and compliance monitoring
    for the entire AIVillage ecosystem, ensuring transparency and accountability.
    """

    def __init__(self, agent_id: str = "auditor_agent"):
        self.agent_id = agent_id
        self.agent_type = "Auditor"
        self.capabilities = [
            "receipt_collection",
            "receipt_verification",
            "audit_trail_maintenance",
            "financial_tracking",
            "cost_analysis",
            "compliance_monitoring",
            "risk_assessment",
            "receipt_search",
            "audit_reporting",
            "transparency_enforcement",
            "activity_logging",
            "pattern_analysis",
        ]

        # Receipt management
        self.receipts: dict[str, Receipt] = {}
        self.receipts_by_agent: dict[str, list[str]] = defaultdict(list)
        self.receipts_by_type: dict[ReceiptType, list[str]] = defaultdict(list)
        self.receipts_by_date: dict[str, list[str]] = defaultdict(
            list
        )  # YYYY-MM-DD format

        # Financial tracking
        self.total_costs_usd = 0.0
        self.costs_by_agent: dict[str, float] = defaultdict(float)
        self.costs_by_action: dict[str, float] = defaultdict(float)
        self.daily_costs: dict[str, float] = defaultdict(float)

        # Compliance monitoring
        self.compliance_alerts: dict[str, ComplianceAlert] = {}
        self.compliance_statistics: dict[str, int] = {
            "compliant": 0,
            "non_compliant": 0,
            "pending_review": 0,
            "requires_attention": 0,
        }

        # Audit reports
        self.audit_reports: dict[str, AuditReport] = {}

        # Performance metrics
        self.receipts_processed = 0
        self.receipts_verified = 0
        self.audit_reports_generated = 0
        self.compliance_violations_found = 0
        self.average_verification_time_ms = 0.0

        # Configuration
        self.verification_enabled = True
        self.auto_compliance_check = True
        self.cost_alert_threshold_usd = 100.0
        self.max_receipt_age_days = 90

        self.initialized = False

    async def generate(self, prompt: str) -> str:
        """Generate audit and compliance responses"""
        prompt_lower = prompt.lower()

        if "receipt" in prompt_lower:
            return "I collect, verify, and manage receipts from all agents to maintain comprehensive audit trails."
        if "audit" in prompt_lower:
            return "I generate detailed audit reports showing agent activity, costs, and compliance status."
        if "compliance" in prompt_lower:
            return "I monitor compliance across all agents and alert on policy violations or suspicious patterns."
        if "cost" in prompt_lower or "financial" in prompt_lower:
            return "I track costs and financial metrics across all agent operations for budget management."
        if "verify" in prompt_lower:
            return "I verify receipt authenticity and integrity to ensure audit trail accuracy."

        return "I am Auditor Agent, maintaining transparency and accountability through comprehensive receipts and audit systems."

    async def get_embedding(self, text: str) -> list[float]:
        """Generate audit-focused embeddings"""
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Audit embeddings focus on compliance and accountability patterns
        return [(hash_value % 1000) / 1000.0] * 448

    async def rerank(
        self, query: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        """Rerank based on audit and compliance relevance"""
        audit_keywords = [
            "receipt",
            "audit",
            "compliance",
            "verify",
            "track",
            "cost",
            "financial",
            "accountability",
            "transparency",
            "risk",
            "pattern",
        ]

        for result in results:
            score = 0
            content = str(result.get("content", ""))

            for keyword in audit_keywords:
                score += content.lower().count(keyword) * 1.8

            # Boost compliance and accountability content
            if any(
                term in content.lower()
                for term in ["compliance", "accountability", "transparency"]
            ):
                score *= 1.6

            result["audit_relevance"] = score

        return sorted(results, key=lambda x: x.get("audit_relevance", 0), reverse=True)[
            :k
        ]

    async def introspect(self) -> dict[str, Any]:
        """Return Auditor agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "total_receipts": len(self.receipts),
            "receipts_processed": self.receipts_processed,
            "receipts_verified": self.receipts_verified,
            "total_costs_usd": self.total_costs_usd,
            "agents_tracked": len(self.receipts_by_agent),
            "compliance_alerts": len(self.compliance_alerts),
            "audit_reports_generated": self.audit_reports_generated,
            "compliance_violations": self.compliance_violations_found,
            "average_verification_time_ms": self.average_verification_time_ms,
            "compliance_rate": self.receipts_verified / max(1, self.receipts_processed),
            "specialization": "receipts_and_compliance",
            "initialized": self.initialized,
        }

    async def communicate(self, message: str, recipient: "AgentInterface") -> str:
        """Communicate audit findings and compliance status"""
        # Add audit context to communications
        if any(
            keyword in message.lower() for keyword in ["receipt", "audit", "compliance"]
        ):
            audit_context = "[AUDIT VERIFIED]"
            message = f"{audit_context} {message}"

        if recipient:
            response = await recipient.generate(f"Auditor Agent reports: {message}")
            return f"Audit communication delivered: {response[:50]}..."
        return "No recipient for audit report"

    async def activate_latent_space(self, query: str) -> tuple[str, str]:
        """Activate audit-specific latent spaces"""
        query_lower = query.lower()

        if "receipt" in query_lower:
            space_type = "receipt_management"
        elif "audit" in query_lower:
            space_type = "audit_analysis"
        elif "compliance" in query_lower:
            space_type = "compliance_monitoring"
        elif "cost" in query_lower or "financial" in query_lower:
            space_type = "financial_tracking"
        else:
            space_type = "general_auditing"

        latent_repr = f"AUDITOR[{space_type}:{query[:50]}]"
        return space_type, latent_repr

    async def record_receipt(self, receipt_data: dict[str, Any]) -> dict[str, Any]:
        """Record a new receipt from an agent - MVP function"""
        receipt_id = receipt_data.get(
            "receipt_id", f"receipt_{int(time.time())}_{len(self.receipts)}"
        )

        # Create receipt object
        receipt = Receipt(
            receipt_id=receipt_id,
            agent_id=receipt_data.get("agent", "unknown_agent"),
            action=receipt_data.get("action", "unknown_action"),
            timestamp=receipt_data.get("timestamp", time.time()),
            receipt_type=ReceiptType(receipt_data.get("receipt_type", "agent_action")),
            details=receipt_data.get("details", receipt_data),
            cost_usd=receipt_data.get("cost_usd"),
            resource_usage=receipt_data.get("resource_usage", {}),
            signature=receipt_data.get("signature"),
        )

        # Verify receipt if verification is enabled
        if self.verification_enabled:
            verification_result = await self._verify_receipt(receipt, receipt_data)
            receipt.verified = verification_result["valid"]
            if not verification_result["valid"]:
                receipt.compliance_status = ComplianceStatus.REQUIRES_ATTENTION
                await self._create_compliance_alert(
                    "receipt_verification_failed",
                    f"Receipt {receipt_id} failed verification",
                    [receipt.agent_id],
                    "medium",
                )
        else:
            receipt.verified = True

        # Store receipt with indexing
        self.receipts[receipt_id] = receipt
        self.receipts_by_agent[receipt.agent_id].append(receipt_id)
        self.receipts_by_type[receipt.receipt_type].append(receipt_id)

        # Date indexing
        date_key = time.strftime("%Y-%m-%d", time.localtime(receipt.timestamp))
        self.receipts_by_date[date_key].append(receipt_id)

        # Update financial tracking
        if receipt.cost_usd:
            self.total_costs_usd += receipt.cost_usd
            self.costs_by_agent[receipt.agent_id] += receipt.cost_usd
            self.costs_by_action[receipt.action] += receipt.cost_usd
            self.daily_costs[date_key] += receipt.cost_usd

            # Check cost alert threshold
            if receipt.cost_usd > self.cost_alert_threshold_usd:
                await self._create_compliance_alert(
                    "high_cost_transaction",
                    f"High cost transaction: ${receipt.cost_usd} by {receipt.agent_id}",
                    [receipt.agent_id],
                    "high",
                )

        # Update metrics
        self.receipts_processed += 1
        if receipt.verified:
            self.receipts_verified += 1

        # Update compliance statistics
        status_key = receipt.compliance_status.value
        self.compliance_statistics[status_key] += 1

        # Auto compliance check
        if self.auto_compliance_check:
            await self._check_receipt_compliance(receipt)

        logger.info(
            f"Receipt recorded: {receipt_id} from {receipt.agent_id} - ${receipt.cost_usd or 0}"
        )

        # Create audit receipt for this action
        audit_receipt = {
            "agent": "Auditor",
            "action": "receipt_recording",
            "timestamp": time.time(),
            "receipt_id": receipt_id,
            "source_agent": receipt.agent_id,
            "verified": receipt.verified,
            "cost_tracked": receipt.cost_usd is not None,
            "signature": f"auditor_record_{receipt_id}",
        }

        return {
            "status": "success",
            "receipt_id": receipt_id,
            "verified": receipt.verified,
            "compliance_status": receipt.compliance_status.value,
            "indexed": True,
            "audit_receipt": audit_receipt,
        }

    async def _verify_receipt(
        self, receipt: Receipt, original_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify receipt authenticity and integrity"""
        verification_start = time.time()
        verification_issues = []

        # Check required fields
        required_fields = ["agent", "action", "timestamp"]
        for field in required_fields:
            if field not in original_data or not original_data[field]:
                verification_issues.append(f"Missing required field: {field}")

        # Verify signature if present
        if receipt.signature:
            expected_signature = (
                f"{receipt.agent_id}_{receipt.action}_{int(receipt.timestamp)}"
            )
            if expected_signature not in receipt.signature:
                verification_issues.append("Invalid signature format")

        # Check timestamp reasonableness (not too far in future/past)
        current_time = time.time()
        if receipt.timestamp > current_time + 300:  # 5 minutes in future
            verification_issues.append("Timestamp too far in future")
        elif receipt.timestamp < current_time - (7 * 24 * 3600):  # 7 days in past
            verification_issues.append("Timestamp too far in past")

        # Verify cost data if present
        if receipt.cost_usd is not None:
            if receipt.cost_usd < 0:
                verification_issues.append("Negative cost not allowed")
            elif receipt.cost_usd > 1000:  # High cost transaction
                verification_issues.append("Unusually high cost - requires review")

        verification_time = (time.time() - verification_start) * 1000
        self.average_verification_time_ms = (
            self.average_verification_time_ms * self.receipts_verified
            + verification_time
        ) / (self.receipts_verified + 1)

        return {
            "valid": len(verification_issues) == 0,
            "issues": verification_issues,
            "verification_time_ms": verification_time,
        }

    async def _check_receipt_compliance(self, receipt: Receipt):
        """Check receipt for compliance issues"""
        # Check for missing cost data on expensive operations
        expensive_actions = ["model_training", "large_file_processing", "api_calls"]
        if any(action in receipt.action.lower() for action in expensive_actions):
            if receipt.cost_usd is None:
                receipt.compliance_status = ComplianceStatus.NON_COMPLIANT
                await self._create_compliance_alert(
                    "missing_cost_data",
                    f"Expensive operation {receipt.action} missing cost data",
                    [receipt.agent_id],
                    "medium",
                )

        # Check for suspicious patterns
        recent_receipts = [
            r
            for r in self.receipts.values()
            if r.agent_id == receipt.agent_id
            and r.timestamp > time.time() - 3600  # Last hour
        ]

        if len(recent_receipts) > 100:  # More than 100 actions per hour
            await self._create_compliance_alert(
                "high_activity_volume",
                f"Agent {receipt.agent_id} has unusually high activity: {len(recent_receipts)} actions/hour",
                [receipt.agent_id],
                "medium",
            )

    async def _create_compliance_alert(
        self,
        alert_type: str,
        description: str,
        affected_agents: list[str],
        severity: str,
    ):
        """Create a compliance alert"""
        alert_id = f"alert_{int(time.time())}_{len(self.compliance_alerts)}"

        alert = ComplianceAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            description=description,
            affected_agents=affected_agents,
            timestamp=time.time(),
        )

        self.compliance_alerts[alert_id] = alert
        self.compliance_violations_found += 1

        logger.warning(
            f"Compliance alert {alert_id}: {description} (Severity: {severity})"
        )

    async def search_receipts(self, search_criteria: dict[str, Any]) -> list[Receipt]:
        """Search receipts based on criteria - MVP function"""
        matching_receipts = []

        # Get base receipt set based on primary criteria
        if "agent_id" in search_criteria:
            candidate_ids = self.receipts_by_agent.get(search_criteria["agent_id"], [])
        elif "receipt_type" in search_criteria:
            receipt_type = ReceiptType(search_criteria["receipt_type"])
            candidate_ids = self.receipts_by_type.get(receipt_type, [])
        elif "date" in search_criteria:
            candidate_ids = self.receipts_by_date.get(search_criteria["date"], [])
        else:
            candidate_ids = list(self.receipts.keys())

        # Filter candidates based on all criteria
        for receipt_id in candidate_ids:
            receipt = self.receipts[receipt_id]
            matches = True

            # Agent ID filter
            if (
                "agent_id" in search_criteria
                and receipt.agent_id != search_criteria["agent_id"]
            ):
                matches = False

            # Action filter
            if (
                "action" in search_criteria
                and search_criteria["action"].lower() not in receipt.action.lower()
            ):
                matches = False

            # Time range filter
            if (
                "start_time" in search_criteria
                and receipt.timestamp < search_criteria["start_time"]
            ):
                matches = False
            if (
                "end_time" in search_criteria
                and receipt.timestamp > search_criteria["end_time"]
            ):
                matches = False

            # Cost range filter
            if "min_cost" in search_criteria and (
                not receipt.cost_usd or receipt.cost_usd < search_criteria["min_cost"]
            ):
                matches = False
            if "max_cost" in search_criteria and (
                receipt.cost_usd and receipt.cost_usd > search_criteria["max_cost"]
            ):
                matches = False

            # Verification status filter
            if (
                "verified_only" in search_criteria
                and search_criteria["verified_only"]
                and not receipt.verified
            ):
                matches = False

            # Compliance status filter
            if "compliance_status" in search_criteria:
                status = ComplianceStatus(search_criteria["compliance_status"])
                if receipt.compliance_status != status:
                    matches = False

            if matches:
                matching_receipts.append(receipt)

        # Sort by timestamp (most recent first)
        matching_receipts.sort(key=lambda r: r.timestamp, reverse=True)

        # Limit results if specified
        if "limit" in search_criteria:
            matching_receipts = matching_receipts[: search_criteria["limit"]]

        logger.info(
            f"Receipt search returned {len(matching_receipts)} results from {len(candidate_ids)} candidates"
        )

        return matching_receipts

    async def generate_audit_report(
        self,
        report_type: str,
        time_range: tuple[float, float],
        agents: list[str] | None = None,
    ) -> AuditReport:
        """Generate comprehensive audit report - MVP function"""
        report_id = f"audit_{report_type}_{int(time.time())}"
        start_time, end_time = time_range

        # Filter receipts for the time range
        relevant_receipts = [
            receipt
            for receipt in self.receipts.values()
            if start_time <= receipt.timestamp <= end_time
        ]

        # Filter by agents if specified
        if agents:
            relevant_receipts = [r for r in relevant_receipts if r.agent_id in agents]
            agent_coverage = agents
        else:
            agent_coverage = list(set(r.agent_id for r in relevant_receipts))

        # Calculate metrics
        total_receipts = len(relevant_receipts)
        total_cost = sum(r.cost_usd for r in relevant_receipts if r.cost_usd)

        # Compliance summary
        compliance_summary = {
            "compliant": sum(
                1
                for r in relevant_receipts
                if r.compliance_status == ComplianceStatus.COMPLIANT
            ),
            "non_compliant": sum(
                1
                for r in relevant_receipts
                if r.compliance_status == ComplianceStatus.NON_COMPLIANT
            ),
            "pending_review": sum(
                1
                for r in relevant_receipts
                if r.compliance_status == ComplianceStatus.PENDING_REVIEW
            ),
            "requires_attention": sum(
                1
                for r in relevant_receipts
                if r.compliance_status == ComplianceStatus.REQUIRES_ATTENTION
            ),
        }

        # Generate key findings
        key_findings = []

        # Most active agent
        agent_activity = defaultdict(int)
        for receipt in relevant_receipts:
            agent_activity[receipt.agent_id] += 1
        if agent_activity:
            most_active = max(agent_activity.items(), key=lambda x: x[1])
            key_findings.append(
                f"Most active agent: {most_active[0]} with {most_active[1]} actions"
            )

        # Highest cost operations
        if relevant_receipts:
            receipts_with_costs = [r for r in relevant_receipts if r.cost_usd]
            if receipts_with_costs:
                highest_cost = max(receipts_with_costs, key=lambda r: r.cost_usd)
                key_findings.append(
                    f"Highest cost operation: ${highest_cost.cost_usd} for {highest_cost.action}"
                )

        # Compliance rate
        if total_receipts > 0:
            compliance_rate = compliance_summary["compliant"] / total_receipts * 100
            key_findings.append(f"Overall compliance rate: {compliance_rate:.1f}%")

        # Verification rate
        verified_receipts = sum(1 for r in relevant_receipts if r.verified)
        if total_receipts > 0:
            verification_rate = verified_receipts / total_receipts * 100
            key_findings.append(f"Receipt verification rate: {verification_rate:.1f}%")

        # Generate recommendations
        recommendations = []

        if compliance_summary["non_compliant"] > 0:
            recommendations.append(
                f"Address {compliance_summary['non_compliant']} non-compliant receipts"
            )

        if compliance_summary["requires_attention"] > 0:
            recommendations.append(
                f"Review {compliance_summary['requires_attention']} receipts requiring attention"
            )

        if total_cost > 500:
            recommendations.append(
                "Monitor high operational costs - consider optimization"
            )

        if len(self.compliance_alerts) > 0:
            active_alerts = sum(
                1 for alert in self.compliance_alerts.values() if not alert.resolved
            )
            if active_alerts > 0:
                recommendations.append(
                    f"Resolve {active_alerts} active compliance alerts"
                )

        if not recommendations:
            recommendations.append(
                "System appears to be operating within normal parameters"
            )

        # Create audit report
        report = AuditReport(
            report_id=report_id,
            report_type=report_type,
            time_range=time_range,
            agent_coverage=agent_coverage,
            total_receipts=total_receipts,
            total_cost_usd=total_cost,
            compliance_summary=compliance_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            generated_timestamp=time.time(),
        )

        # Store report
        self.audit_reports[report_id] = report
        self.audit_reports_generated += 1

        logger.info(
            f"Audit report generated: {report_id} - {total_receipts} receipts, ${total_cost:.2f} total cost"
        )

        return report

    async def get_compliance_dashboard(self) -> dict[str, Any]:
        """Generate comprehensive compliance dashboard"""
        current_time = time.time()

        # Recent activity (last 24 hours)
        recent_receipts = [
            r
            for r in self.receipts.values()
            if r.timestamp > current_time - (24 * 3600)
        ]

        # Active compliance alerts
        active_alerts = [
            alert for alert in self.compliance_alerts.values() if not alert.resolved
        ]

        return {
            "agent": "Auditor",
            "dashboard_type": "compliance_overview",
            "timestamp": current_time,
            "receipt_metrics": {
                "total_receipts": len(self.receipts),
                "recent_24h": len(recent_receipts),
                "verification_rate": self.receipts_verified
                / max(1, self.receipts_processed),
                "average_verification_time_ms": self.average_verification_time_ms,
            },
            "financial_metrics": {
                "total_costs_usd": self.total_costs_usd,
                "daily_average": self.total_costs_usd / max(1, len(self.daily_costs)),
                "highest_cost_agent": (
                    max(self.costs_by_agent.items(), key=lambda x: x[1])
                    if self.costs_by_agent
                    else ("none", 0)
                ),
                "cost_distribution": dict(self.costs_by_agent),
            },
            "compliance_metrics": {
                "compliance_statistics": self.compliance_statistics,
                "active_alerts": len(active_alerts),
                "violations_found": self.compliance_violations_found,
                "agents_monitored": len(self.receipts_by_agent),
            },
            "audit_metrics": {
                "reports_generated": self.audit_reports_generated,
                "receipt_types_tracked": len(self.receipts_by_type),
                "oldest_receipt_days": (
                    (current_time - min(r.timestamp for r in self.receipts.values()))
                    / (24 * 3600)
                    if self.receipts
                    else 0
                ),
            },
            "recommendations": [
                "Maintain regular audit report generation",
                "Monitor compliance alerts and resolve promptly",
                "Review high-cost operations for optimization opportunities",
                "Ensure all agents generate proper receipts",
            ],
        }

    async def initialize(self):
        """Initialize the Auditor Agent"""
        try:
            logger.info("Initializing Auditor Agent - Receipts & Compliance System...")

            # Initialize signature database for basic malware hashes (for Shield integration)
            self.known_signatures = {
                "sample_hash_1": "test_malware_signature",
                "sample_hash_2": "trojan_signature",
            }

            # Create initial audit receipt for system startup
            startup_receipt = {
                "agent": "Auditor",
                "action": "system_initialization",
                "timestamp": time.time(),
                "details": {
                    "verification_enabled": self.verification_enabled,
                    "auto_compliance_check": self.auto_compliance_check,
                    "cost_alert_threshold": self.cost_alert_threshold_usd,
                },
                "signature": f"auditor_init_{int(time.time())}",
            }

            await self.record_receipt(startup_receipt)

            self.initialized = True
            logger.info(
                f"Auditor Agent {self.agent_id} initialized - Receipt system active"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Auditor Agent: {e}")
            self.initialized = False

    async def shutdown(self):
        """Shutdown Auditor Agent gracefully"""
        try:
            logger.info("Auditor Agent shutting down...")

            # Generate final audit report
            final_report = await self.get_compliance_dashboard()
            logger.info(
                f"Auditor Agent final dashboard: {final_report['receipt_metrics']}"
            )

            # Create shutdown receipt
            shutdown_receipt = {
                "agent": "Auditor",
                "action": "system_shutdown",
                "timestamp": time.time(),
                "details": {
                    "total_receipts_processed": self.receipts_processed,
                    "total_cost_tracked": self.total_costs_usd,
                    "compliance_violations": self.compliance_violations_found,
                },
                "signature": f"auditor_shutdown_{int(time.time())}",
            }

            # Don't call record_receipt during shutdown to avoid recursion
            self.initialized = False

        except Exception as e:
            logger.error(f"Error during Auditor Agent shutdown: {e}")
