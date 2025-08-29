"""
Enhanced SLA Tiers with Infrastructure Diversity Requirements

Implements Bronze, Silver, and Gold tier SLA guarantees with heterogeneous
quorum requirements for Gold tier services.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..quorum.infrastructure_classifier import InfrastructureProfile
from ..quorum.quorum_manager import QuorumManager, QuorumRequirement


class SLATier(Enum):
    """Service Level Agreement tiers"""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


@dataclass
class SLAMetrics:
    """SLA performance metrics"""
    p95_latency_ms: float
    uptime_percentage: float
    error_rate_percentage: float
    throughput_ops_per_second: float


@dataclass
class SLARequirements:
    """SLA tier requirements"""
    tier: SLATier
    max_p95_latency_ms: float
    min_uptime_percentage: float
    max_error_rate_percentage: float
    min_replication_factor: int
    quorum_requirement: QuorumRequirement
    pricing_multiplier: float
    description: str


@dataclass
class ServiceInstance:
    """Service instance configuration"""
    service_id: str
    tier: SLATier
    allocated_devices: list[str]
    device_profiles: list[InfrastructureProfile]
    creation_time: datetime
    last_validation: datetime
    validation_status: str
    diversity_score: float
    metadata: dict


class EnhancedSLATierManager:
    """Manages enhanced SLA tiers with infrastructure diversity"""

    def __init__(self, quorum_manager: QuorumManager | None = None):
        self.quorum_manager = quorum_manager or QuorumManager()
        self.logger = logging.getLogger(__name__)

        # Define SLA tier requirements
        self.sla_requirements = {
            SLATier.BRONZE: SLARequirements(
                tier=SLATier.BRONZE,
                max_p95_latency_ms=2500.0,
                min_uptime_percentage=97.0,
                max_error_rate_percentage=3.0,
                min_replication_factor=1,
                quorum_requirement=QuorumRequirement.NONE,
                pricing_multiplier=1.0,
                description="Basic service with best-effort availability"
            ),

            SLATier.SILVER: SLARequirements(
                tier=SLATier.SILVER,
                max_p95_latency_ms=1200.0,
                min_uptime_percentage=99.0,
                max_error_rate_percentage=1.0,
                min_replication_factor=2,  # 1 primary + 1 canary
                quorum_requirement=QuorumRequirement.ENHANCED,
                pricing_multiplier=2.5,
                description="Balanced performance with geographic diversity"
            ),

            SLATier.GOLD: SLARequirements(
                tier=SLATier.GOLD,
                max_p95_latency_ms=400.0,
                min_uptime_percentage=99.9,
                max_error_rate_percentage=0.1,
                min_replication_factor=3,
                quorum_requirement=QuorumRequirement.GOLD,
                pricing_multiplier=5.0,
                description="Critical service with disjoint infrastructure diversity"
            )
        }

        # Active service instances
        self.service_instances: dict[str, ServiceInstance] = {}

        # Performance tracking
        self.performance_history: dict[str, list[tuple[datetime, SLAMetrics]]] = {}

    async def provision_service(self,
                              service_id: str,
                              tier: SLATier,
                              available_devices: list[dict],
                              service_config: dict | None = None) -> dict:
        """
        Provision service with appropriate SLA tier requirements

        Args:
            service_id: Unique service identifier
            tier: Requested SLA tier
            available_devices: Pool of available devices
            service_config: Additional service configuration

        Returns:
            Provisioning result with allocated devices and validation
        """
        requirements = self.sla_requirements[tier]

        self.logger.info(f"Provisioning {tier.value} tier service {service_id}")

        # Validate quorum requirements
        quorum_result = await self.quorum_manager.validate_quorum(
            device_candidates=available_devices,
            requirement=requirements.quorum_requirement,
            target_size=requirements.min_replication_factor
        )

        if not quorum_result.is_valid:
            return {
                'success': False,
                'error': 'Failed to meet infrastructure diversity requirements',
                'violations': quorum_result.violations,
                'recommendations': quorum_result.recommendations,
                'tier': tier.value
            }

        # Create service instance
        service_instance = ServiceInstance(
            service_id=service_id,
            tier=tier,
            allocated_devices=[p.device_id for p in quorum_result.profiles_used],
            device_profiles=quorum_result.profiles_used,
            creation_time=datetime.utcnow(),
            last_validation=datetime.utcnow(),
            validation_status='valid',
            diversity_score=quorum_result.diversity_score,
            metadata={
                'service_config': service_config or {},
                'diversity_metrics': quorum_result.metadata.get('diversity_metrics', {}),
                'pricing_multiplier': requirements.pricing_multiplier,
                'requirements': asdict(requirements)
            }
        )

        self.service_instances[service_id] = service_instance

        # Initialize performance tracking
        self.performance_history[service_id] = []

        return {
            'success': True,
            'service_id': service_id,
            'tier': tier.value,
            'allocated_devices': service_instance.allocated_devices,
            'diversity_score': quorum_result.diversity_score,
            'pricing_multiplier': requirements.pricing_multiplier,
            'sla_guarantees': {
                'max_p95_latency_ms': requirements.max_p95_latency_ms,
                'min_uptime_percentage': requirements.min_uptime_percentage,
                'max_error_rate': requirements.max_error_rate_percentage
            },
            'infrastructure_diversity': quorum_result.metadata.get('diversity_metrics', {}),
            'created_at': service_instance.creation_time.isoformat()
        }

    async def validate_sla_compliance(self,
                                    service_id: str,
                                    current_metrics: SLAMetrics) -> dict:
        """
        Validate service compliance with SLA requirements

        Args:
            service_id: Service to validate
            current_metrics: Current performance metrics

        Returns:
            Compliance validation result
        """
        if service_id not in self.service_instances:
            return {
                'compliant': False,
                'error': f'Service {service_id} not found'
            }

        service = self.service_instances[service_id]
        requirements = self.sla_requirements[service.tier]

        # Track metrics
        self.performance_history[service_id].append((datetime.utcnow(), current_metrics))

        # Keep only recent history (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.performance_history[service_id] = [
            (timestamp, metrics) for timestamp, metrics in self.performance_history[service_id]
            if timestamp > cutoff
        ]

        violations = []

        # Latency compliance
        if current_metrics.p95_latency_ms > requirements.max_p95_latency_ms:
            violations.append({
                'metric': 'p95_latency_ms',
                'current': current_metrics.p95_latency_ms,
                'requirement': requirements.max_p95_latency_ms,
                'severity': 'high' if current_metrics.p95_latency_ms > requirements.max_p95_latency_ms * 1.5 else 'medium'
            })

        # Uptime compliance
        if current_metrics.uptime_percentage < requirements.min_uptime_percentage:
            violations.append({
                'metric': 'uptime_percentage',
                'current': current_metrics.uptime_percentage,
                'requirement': requirements.min_uptime_percentage,
                'severity': 'high'
            })

        # Error rate compliance
        if current_metrics.error_rate_percentage > requirements.max_error_rate_percentage:
            violations.append({
                'metric': 'error_rate_percentage',
                'current': current_metrics.error_rate_percentage,
                'requirement': requirements.max_error_rate_percentage,
                'severity': 'high' if current_metrics.error_rate_percentage > requirements.max_error_rate_percentage * 2 else 'medium'
            })

        # Infrastructure diversity revalidation for Gold tier
        diversity_valid = True
        if service.tier == SLATier.GOLD:
            # Revalidate quorum every hour
            if datetime.utcnow() - service.last_validation > timedelta(hours=1):
                device_data = [{'id': device_id, 'ip_address': '127.0.0.1'} for device_id in service.allocated_devices]
                quorum_result = await self.quorum_manager.validate_quorum(
                    device_candidates=device_data,
                    requirement=requirements.quorum_requirement,
                    target_size=len(service.allocated_devices)
                )

                if not quorum_result.is_valid:
                    diversity_valid = False
                    violations.append({
                        'metric': 'infrastructure_diversity',
                        'current': 'failed',
                        'requirement': 'disjoint_infrastructure',
                        'severity': 'critical',
                        'details': quorum_result.violations
                    })

                service.last_validation = datetime.utcnow()
                service.validation_status = 'valid' if diversity_valid else 'invalid'

        compliant = len(violations) == 0

        return {
            'compliant': compliant,
            'service_id': service_id,
            'tier': service.tier.value,
            'violations': violations,
            'current_metrics': asdict(current_metrics),
            'requirements': asdict(requirements),
            'diversity_score': service.diversity_score,
            'diversity_valid': diversity_valid,
            'validation_time': datetime.utcnow().isoformat()
        }

    async def rebalance_service(self,
                              service_id: str,
                              available_devices: list[dict],
                              force_rebalance: bool = False) -> dict:
        """
        Rebalance service allocation to maintain SLA compliance

        Args:
            service_id: Service to rebalance
            available_devices: Pool of available devices
            force_rebalance: Force rebalancing even if current allocation is valid

        Returns:
            Rebalancing result
        """
        if service_id not in self.service_instances:
            return {
                'success': False,
                'error': f'Service {service_id} not found'
            }

        service = self.service_instances[service_id]
        requirements = self.sla_requirements[service.tier]

        # Check if rebalancing is needed
        if not force_rebalance and service.validation_status == 'valid':
            return {
                'success': True,
                'action': 'no_rebalance_needed',
                'service_id': service_id,
                'current_allocation': service.allocated_devices
            }

        self.logger.info(f"Rebalancing service {service_id} ({service.tier.value})")

        # Find new optimal allocation
        quorum_result = await self.quorum_manager.validate_quorum(
            device_candidates=available_devices,
            requirement=requirements.quorum_requirement,
            target_size=max(requirements.min_replication_factor, len(service.allocated_devices))
        )

        if not quorum_result.is_valid:
            return {
                'success': False,
                'error': 'Failed to find valid rebalancing allocation',
                'violations': quorum_result.violations,
                'recommendations': quorum_result.recommendations
            }

        # Update service allocation
        old_devices = service.allocated_devices.copy()
        service.allocated_devices = [p.device_id for p in quorum_result.profiles_used]
        service.device_profiles = quorum_result.profiles_used
        service.last_validation = datetime.utcnow()
        service.validation_status = 'valid'
        service.diversity_score = quorum_result.diversity_score

        # Update metadata
        service.metadata['diversity_metrics'] = quorum_result.metadata.get('diversity_metrics', {})
        service.metadata['last_rebalance'] = datetime.utcnow().isoformat()
        service.metadata['rebalance_reason'] = 'forced' if force_rebalance else 'compliance'

        return {
            'success': True,
            'action': 'rebalanced',
            'service_id': service_id,
            'old_allocation': old_devices,
            'new_allocation': service.allocated_devices,
            'diversity_improvement': quorum_result.diversity_score - service.metadata.get('previous_diversity_score', 0),
            'diversity_score': quorum_result.diversity_score,
            'rebalance_time': datetime.utcnow().isoformat()
        }

    def get_tier_pricing(self, tier: SLATier, base_cost: float = 1.0) -> float:
        """Get pricing for SLA tier"""
        requirements = self.sla_requirements[tier]
        return base_cost * requirements.pricing_multiplier

    def get_service_status(self, service_id: str) -> dict | None:
        """Get comprehensive service status"""
        if service_id not in self.service_instances:
            return None

        service = self.service_instances[service_id]
        requirements = self.sla_requirements[service.tier]

        # Recent performance metrics
        recent_metrics = None
        if service_id in self.performance_history and self.performance_history[service_id]:
            _, recent_metrics = self.performance_history[service_id][-1]

        # Quorum status
        quorum_status = self.quorum_manager.get_quorum_status_summary(service.device_profiles)

        return {
            'service_id': service_id,
            'tier': service.tier.value,
            'status': 'active',
            'allocated_devices': service.allocated_devices,
            'device_count': len(service.allocated_devices),
            'diversity_score': service.diversity_score,
            'validation_status': service.validation_status,
            'last_validation': service.last_validation.isoformat(),
            'created_at': service.creation_time.isoformat(),
            'sla_requirements': asdict(requirements),
            'recent_metrics': asdict(recent_metrics) if recent_metrics else None,
            'quorum_status': quorum_status,
            'pricing_multiplier': requirements.pricing_multiplier,
            'infrastructure_diversity': service.metadata.get('diversity_metrics', {})
        }

    def get_all_services_status(self) -> dict:
        """Get status of all managed services"""
        services_by_tier = {tier: [] for tier in SLATier}

        for service_id, service in self.service_instances.items():
            status = self.get_service_status(service_id)
            if status:
                services_by_tier[service.tier].append(status)

        # Calculate aggregate metrics
        total_services = len(self.service_instances)
        tier_counts = {tier.value: len(services) for tier, services in services_by_tier.items()}

        # Active violations
        active_violations = 0
        for service in self.service_instances.values():
            if service.validation_status != 'valid':
                active_violations += 1

        return {
            'total_services': total_services,
            'tier_distribution': tier_counts,
            'active_violations': active_violations,
            'services_by_tier': {
                tier.value: services for tier, services in services_by_tier.items()
            },
            'sla_requirements': {
                tier.value: asdict(req) for tier, req in self.sla_requirements.items()
            },
            'timestamp': datetime.utcnow().isoformat()
        }

    async def cleanup_expired_services(self,
                                     max_age_hours: int = 24,
                                     inactive_threshold_hours: int = 2) -> dict:
        """Clean up expired or inactive services"""
        now = datetime.utcnow()
        expired_services = []
        inactive_services = []

        for service_id, service in list(self.service_instances.items()):
            age = now - service.creation_time
            time_since_validation = now - service.last_validation

            # Check for expired services
            if age.total_seconds() / 3600 > max_age_hours:
                expired_services.append(service_id)
                del self.service_instances[service_id]
                if service_id in self.performance_history:
                    del self.performance_history[service_id]

            # Check for inactive services
            elif time_since_validation.total_seconds() / 3600 > inactive_threshold_hours:
                inactive_services.append(service_id)

        return {
            'cleaned_up': len(expired_services),
            'expired_services': expired_services,
            'inactive_services': inactive_services,
            'total_remaining': len(self.service_instances),
            'cleanup_time': now.isoformat()
        }
