"""
Infrastructure Diversity Monitoring Dashboard

Real-time monitoring and visualization of infrastructure diversity metrics
for heterogeneous quorum requirements and Gold tier SLA compliance.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging
from typing import Any

from ..quorum.infrastructure_classifier import InfrastructureProfile
from ..quorum.quorum_manager import QuorumManager
from ..scheduler.enhanced_sla_tiers import EnhancedSLATierManager


@dataclass
class DiversityAlert:
    """Diversity violation alert"""

    alert_id: str
    service_id: str
    tier: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    violation_type: str
    description: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class DiversityMetrics:
    """Comprehensive diversity metrics"""

    timestamp: datetime
    total_devices: int
    unique_asns: int
    unique_tee_vendors: int
    unique_power_regions: int
    unique_countries: int
    diversity_score: float
    asn_distribution: dict[str, int]
    tee_vendor_distribution: dict[str, int]
    power_region_distribution: dict[str, int]
    country_distribution: dict[str, int]


@dataclass
class ServiceHealthStatus:
    """Service health and diversity status"""

    service_id: str
    tier: str
    status: str  # 'healthy', 'degraded', 'critical'
    diversity_score: float
    device_count: int
    sla_compliance: bool
    last_validation: datetime
    violations: list[str]
    recommendations: list[str]


class DiversityDashboard:
    """Real-time infrastructure diversity monitoring dashboard"""

    def __init__(
        self, quorum_manager: QuorumManager, sla_tier_manager: EnhancedSLATierManager, history_size: int = 1000
    ):
        self.quorum_manager = quorum_manager
        self.sla_tier_manager = sla_tier_manager
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self.is_monitoring = False
        self.history_size = history_size

        # Historical data storage
        self.diversity_history: deque = deque(maxlen=history_size)
        self.service_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.alert_history: deque = deque(maxlen=history_size)

        # Current state
        self.current_metrics: DiversityMetrics | None = None
        self.active_alerts: dict[str, DiversityAlert] = {}
        self.service_statuses: dict[str, ServiceHealthStatus] = {}

        # Thresholds
        self.alert_thresholds = {
            "diversity_score": {"critical": 0.2, "high": 0.4, "medium": 0.6, "low": 0.8},
            "asn_diversity": {"critical": 0.1, "high": 0.3, "medium": 0.5, "low": 0.7},
            "tee_vendor_diversity": {"critical": 0.2, "high": 0.4, "medium": 0.6, "low": 0.8},
        }

    async def start_monitoring(self, interval_seconds: int = 60):
        """Start real-time diversity monitoring"""
        self.is_monitoring = True
        self.logger.info(f"Starting diversity monitoring (interval: {interval_seconds}s)")

        # Start monitoring tasks
        tasks = [
            self._diversity_monitoring_task(interval_seconds),
            self._service_monitoring_task(interval_seconds * 2),
            self._alert_processing_task(30),
            self._cleanup_task(3600),  # Cleanup every hour
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_monitoring(self):
        """Stop diversity monitoring"""
        self.is_monitoring = False
        self.logger.info("Stopped diversity monitoring")

    async def _diversity_monitoring_task(self, interval_seconds: int):
        """Monitor overall infrastructure diversity"""
        while self.is_monitoring:
            try:
                # Get all active service profiles
                all_profiles = []
                for service_id, service_status in self.service_statuses.items():
                    # In production, would get actual profiles from service registry
                    # For now, create mock profiles based on service data
                    pass

                # Calculate current diversity metrics
                if all_profiles:
                    diversity_metrics = self.quorum_manager.classifier.get_diversity_metrics(all_profiles)

                    current_metrics = DiversityMetrics(
                        timestamp=datetime.utcnow(),
                        total_devices=len(all_profiles),
                        unique_asns=diversity_metrics["unique_asns"],
                        unique_tee_vendors=diversity_metrics["unique_tee_vendors"],
                        unique_power_regions=diversity_metrics["unique_power_regions"],
                        unique_countries=diversity_metrics["unique_countries"],
                        diversity_score=diversity_metrics["total_diversity_score"],
                        asn_distribution=self._get_asn_distribution(all_profiles),
                        tee_vendor_distribution=self._get_tee_distribution(all_profiles),
                        power_region_distribution=self._get_power_distribution(all_profiles),
                        country_distribution=self._get_country_distribution(all_profiles),
                    )

                    self.current_metrics = current_metrics
                    self.diversity_history.append(current_metrics)

                    # Check for diversity alerts
                    await self._check_diversity_alerts(current_metrics)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Diversity monitoring error: {e}")
                await asyncio.sleep(60)

    async def _service_monitoring_task(self, interval_seconds: int):
        """Monitor individual service diversity and health"""
        while self.is_monitoring:
            try:
                # Get all services from SLA tier manager
                all_services = self.sla_tier_manager.get_all_services_status()

                for tier_name, services in all_services.get("services_by_tier", {}).items():
                    for service in services:
                        service_id = service["service_id"]

                        # Create service health status
                        status = ServiceHealthStatus(
                            service_id=service_id,
                            tier=service["tier"],
                            status=self._determine_service_health(service),
                            diversity_score=service.get("diversity_score", 0.0),
                            device_count=service.get("device_count", 0),
                            sla_compliance=service.get("validation_status") == "valid",
                            last_validation=datetime.fromisoformat(
                                service.get("last_validation", datetime.utcnow().isoformat())
                            ),
                            violations=[],  # Would be populated from actual service data
                            recommendations=[],  # Would be generated based on violations
                        )

                        self.service_statuses[service_id] = status
                        self.service_history[service_id].append(status)

                        # Check for service-specific alerts
                        await self._check_service_alerts(status)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(60)

    async def _alert_processing_task(self, interval_seconds: int):
        """Process and manage diversity alerts"""
        while self.is_monitoring:
            try:
                # Auto-acknowledge low severity alerts after 1 hour
                cutoff_time = datetime.utcnow() - timedelta(hours=1)

                for alert_id, alert in list(self.active_alerts.items()):
                    if alert.severity == "low" and alert.timestamp < cutoff_time and not alert.acknowledged:

                        alert.acknowledged = True
                        self.logger.info(f"Auto-acknowledged low severity alert: {alert_id}")

                # Auto-resolve acknowledged alerts after 24 hours
                resolve_cutoff = datetime.utcnow() - timedelta(hours=24)

                for alert_id, alert in list(self.active_alerts.items()):
                    if alert.acknowledged and alert.timestamp < resolve_cutoff:

                        alert.resolved = True
                        del self.active_alerts[alert_id]
                        self.alert_history.append(alert)
                        self.logger.info(f"Auto-resolved alert: {alert_id}")

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)

    async def _cleanup_task(self, interval_seconds: int):
        """Cleanup old data and maintain system health"""
        while self.is_monitoring:
            try:
                # Remove old service history entries
                cutoff_time = datetime.utcnow() - timedelta(days=7)

                for service_id, history in self.service_history.items():
                    while history and history[0].last_validation < cutoff_time:
                        history.popleft()

                # Clean up resolved alerts older than 30 days
                alert_cutoff = datetime.utcnow() - timedelta(days=30)
                while self.alert_history and self.alert_history[0].timestamp < alert_cutoff:
                    self.alert_history.popleft()

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(3600)

    async def _check_diversity_alerts(self, metrics: DiversityMetrics):
        """Check for diversity-related alerts"""
        alerts_to_create = []

        # Overall diversity score alert
        if metrics.diversity_score < self.alert_thresholds["diversity_score"]["critical"]:
            alerts_to_create.append(
                {
                    "type": "diversity_score_critical",
                    "severity": "critical",
                    "description": f"Critical diversity score: {metrics.diversity_score:.2f}",
                }
            )
        elif metrics.diversity_score < self.alert_thresholds["diversity_score"]["high"]:
            alerts_to_create.append(
                {
                    "type": "diversity_score_high",
                    "severity": "high",
                    "description": f"Low diversity score: {metrics.diversity_score:.2f}",
                }
            )

        # ASN diversity alert
        asn_diversity = metrics.unique_asns / max(metrics.total_devices, 1)
        if asn_diversity < self.alert_thresholds["asn_diversity"]["critical"]:
            alerts_to_create.append(
                {
                    "type": "asn_diversity_critical",
                    "severity": "critical",
                    "description": f"Critical ASN diversity: {asn_diversity:.2f} ({metrics.unique_asns}/{metrics.total_devices})",
                }
            )

        # TEE vendor diversity alert
        tee_diversity = metrics.unique_tee_vendors / max(metrics.total_devices, 1)
        if tee_diversity < self.alert_thresholds["tee_vendor_diversity"]["critical"]:
            alerts_to_create.append(
                {
                    "type": "tee_diversity_critical",
                    "severity": "critical",
                    "description": f"Critical TEE vendor diversity: {tee_diversity:.2f} ({metrics.unique_tee_vendors}/{metrics.total_devices})",
                }
            )

        # Create alerts
        for alert_data in alerts_to_create:
            alert_id = f"{alert_data['type']}_{int(datetime.utcnow().timestamp())}"
            alert = DiversityAlert(
                alert_id=alert_id,
                service_id="global",
                tier="system",
                severity=alert_data["severity"],
                violation_type=alert_data["type"],
                description=alert_data["description"],
                timestamp=datetime.utcnow(),
            )

            # Only create if similar alert doesn't exist
            similar_exists = any(
                a.violation_type == alert.violation_type and not a.acknowledged for a in self.active_alerts.values()
            )

            if not similar_exists:
                self.active_alerts[alert_id] = alert
                self.logger.warning(f"Created diversity alert: {alert.description}")

    async def _check_service_alerts(self, status: ServiceHealthStatus):
        """Check for service-specific alerts"""
        if status.status == "critical":
            alert_id = f"service_critical_{status.service_id}_{int(datetime.utcnow().timestamp())}"
            alert = DiversityAlert(
                alert_id=alert_id,
                service_id=status.service_id,
                tier=status.tier,
                severity="critical",
                violation_type="service_critical",
                description=f"Service {status.service_id} in critical state",
                timestamp=datetime.utcnow(),
            )

            if alert_id not in self.active_alerts:
                self.active_alerts[alert_id] = alert
                self.logger.critical(f"Critical service alert: {alert.description}")

    def _determine_service_health(self, service: dict) -> str:
        """Determine service health status"""
        validation_status = service.get("validation_status", "unknown")
        diversity_score = service.get("diversity_score", 0.0)

        if validation_status == "invalid" or diversity_score < 0.3:
            return "critical"
        elif validation_status == "valid" and diversity_score > 0.7:
            return "healthy"
        else:
            return "degraded"

    def _get_asn_distribution(self, profiles: list[InfrastructureProfile]) -> dict[str, int]:
        """Get ASN distribution"""
        distribution = defaultdict(int)
        for profile in profiles:
            key = f"AS{profile.asn}" if profile.asn else "Unknown"
            distribution[key] += 1
        return dict(distribution)

    def _get_tee_distribution(self, profiles: list[InfrastructureProfile]) -> dict[str, int]:
        """Get TEE vendor distribution"""
        distribution = defaultdict(int)
        for profile in profiles:
            distribution[profile.tee_vendor.value] += 1
        return dict(distribution)

    def _get_power_distribution(self, profiles: list[InfrastructureProfile]) -> dict[str, int]:
        """Get power region distribution"""
        distribution = defaultdict(int)
        for profile in profiles:
            distribution[profile.power_region.value] += 1
        return dict(distribution)

    def _get_country_distribution(self, profiles: list[InfrastructureProfile]) -> dict[str, int]:
        """Get country distribution"""
        distribution = defaultdict(int)
        for profile in profiles:
            distribution[profile.country_code or "Unknown"] += 1
        return dict(distribution)

    # Public API methods

    def get_current_dashboard(self) -> dict[str, Any]:
        """Get current dashboard state"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "diversity_metrics": asdict(self.current_metrics) if self.current_metrics else None,
            "active_alerts": {alert_id: asdict(alert) for alert_id, alert in self.active_alerts.items()},
            "service_statuses": {service_id: asdict(status) for service_id, status in self.service_statuses.items()},
            "alert_summary": {
                "total_active": len(self.active_alerts),
                "critical": len([a for a in self.active_alerts.values() if a.severity == "critical"]),
                "high": len([a for a in self.active_alerts.values() if a.severity == "high"]),
                "medium": len([a for a in self.active_alerts.values() if a.severity == "medium"]),
                "low": len([a for a in self.active_alerts.values() if a.severity == "low"]),
            },
            "system_health": self._get_system_health_summary(),
        }

    def get_historical_metrics(self, hours_back: int = 24, metric_type: str = "diversity") -> list[dict]:
        """Get historical metrics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        if metric_type == "diversity":
            return [asdict(metric) for metric in self.diversity_history if metric.timestamp > cutoff_time]

        return []

    def get_service_trends(self, service_id: str, hours_back: int = 24) -> list[dict]:
        """Get service trend data"""
        if service_id not in self.service_history:
            return []

        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        return [asdict(status) for status in self.service_history[service_id] if status.last_validation > cutoff_time]

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self.logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False

    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            self.alert_history.append(alert)
            self.logger.info(f"Alert {alert_id} resolved by {user}")
            return True
        return False

    def _get_system_health_summary(self) -> dict[str, Any]:
        """Get overall system health summary"""
        if not self.current_metrics:
            return {"status": "unknown", "details": "No metrics available"}

        # Determine overall health
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == "critical"])
        high_alerts = len([a for a in self.active_alerts.values() if a.severity == "high"])

        if critical_alerts > 0:
            status = "critical"
        elif high_alerts > 0:
            status = "degraded"
        elif self.current_metrics.diversity_score > 0.8:
            status = "excellent"
        elif self.current_metrics.diversity_score > 0.6:
            status = "good"
        else:
            status = "fair"

        return {
            "status": status,
            "diversity_score": self.current_metrics.diversity_score,
            "total_devices": self.current_metrics.total_devices,
            "critical_alerts": critical_alerts,
            "high_alerts": high_alerts,
            "last_update": self.current_metrics.timestamp.isoformat(),
        }
