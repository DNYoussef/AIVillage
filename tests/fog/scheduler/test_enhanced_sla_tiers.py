"""
Tests for enhanced SLA tiers with infrastructure diversity requirements
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from infrastructure.fog.quorum.infrastructure_classifier import InfrastructureProfile, PowerRegion, TEEVendor
from infrastructure.fog.quorum.quorum_manager import QuorumManager, QuorumRequirement, QuorumValidationResult
from infrastructure.fog.scheduler.enhanced_sla_tiers import EnhancedSLATierManager, SLAMetrics, SLATier


@pytest.fixture
def mock_quorum_manager():
    """Mock quorum manager"""
    quorum_manager = Mock(spec=QuorumManager)
    quorum_manager.validate_quorum = AsyncMock()
    quorum_manager.get_quorum_status_summary = Mock()
    return quorum_manager


@pytest.fixture
def sla_tier_manager(mock_quorum_manager):
    """EnhancedSLATierManager instance with mock quorum manager"""
    return EnhancedSLATierManager(quorum_manager=mock_quorum_manager)


@pytest.fixture
def sample_devices():
    """Sample device candidates"""
    return [
        {
            "id": "device1",
            "ip_address": "10.1.1.1",
            "attestation_data": {"platform": "amd", "sev_snp": True},
            "network_info": {"datacenter": True},
        },
        {
            "id": "device2",
            "ip_address": "10.2.2.2",
            "attestation_data": {"platform": "intel", "tdx": True},
            "network_info": {"public": True},
        },
        {
            "id": "device3",
            "ip_address": "10.3.3.3",
            "attestation_data": {"platform": "amd", "sev_snp": True},
            "network_info": {"residential": True},
        },
    ]


@pytest.fixture
def sample_profiles():
    """Sample infrastructure profiles"""
    return [
        InfrastructureProfile(
            device_id="device1",
            asn=12345,
            asn_name="Provider A",
            tee_vendor=TEEVendor.AMD_SEV_SNP,
            tee_version="1.0",
            power_region=PowerRegion.NERC_RFC,
            country_code="US",
            region="Virginia",
            city="Reston",
            network_topology="datacenter",
            attestation_hash="abc123",
            classification_time=datetime.utcnow(),
            confidence_score=0.9,
        ),
        InfrastructureProfile(
            device_id="device2",
            asn=67890,
            asn_name="Provider B",
            tee_vendor=TEEVendor.INTEL_TDX,
            tee_version="2.0",
            power_region=PowerRegion.NERC_WECC,
            country_code="US",
            region="California",
            city="San Francisco",
            network_topology="public",
            attestation_hash="def456",
            classification_time=datetime.utcnow(),
            confidence_score=0.95,
        ),
        InfrastructureProfile(
            device_id="device3",
            asn=11111,
            asn_name="Provider C",
            tee_vendor=TEEVendor.AMD_SEV_SNP,
            tee_version="1.1",
            power_region=PowerRegion.NERC_SERC,
            country_code="US",
            region="Georgia",
            city="Atlanta",
            network_topology="residential",
            attestation_hash="ghi789",
            classification_time=datetime.utcnow(),
            confidence_score=0.8,
        ),
    ]


class TestSLATierRequirements:
    """Test SLA tier requirement definitions"""

    def test_bronze_tier_requirements(self, sla_tier_manager):
        """Test Bronze tier requirements"""
        requirements = sla_tier_manager.sla_requirements[SLATier.BRONZE]

        assert requirements.tier == SLATier.BRONZE
        assert requirements.max_p95_latency_ms == 2500.0
        assert requirements.min_uptime_percentage == 97.0
        assert requirements.min_replication_factor == 1
        assert requirements.quorum_requirement == QuorumRequirement.NONE
        assert requirements.pricing_multiplier == 1.0

    def test_silver_tier_requirements(self, sla_tier_manager):
        """Test Silver tier requirements"""
        requirements = sla_tier_manager.sla_requirements[SLATier.SILVER]

        assert requirements.tier == SLATier.SILVER
        assert requirements.max_p95_latency_ms == 1200.0
        assert requirements.min_uptime_percentage == 99.0
        assert requirements.min_replication_factor == 2
        assert requirements.quorum_requirement == QuorumRequirement.ENHANCED
        assert requirements.pricing_multiplier == 2.5

    def test_gold_tier_requirements(self, sla_tier_manager):
        """Test Gold tier requirements"""
        requirements = sla_tier_manager.sla_requirements[SLATier.GOLD]

        assert requirements.tier == SLATier.GOLD
        assert requirements.max_p95_latency_ms == 400.0
        assert requirements.min_uptime_percentage == 99.9
        assert requirements.min_replication_factor == 3
        assert requirements.quorum_requirement == QuorumRequirement.GOLD
        assert requirements.pricing_multiplier == 5.0


class TestServiceProvisioning:
    """Test service provisioning with different SLA tiers"""

    @pytest.mark.asyncio
    async def test_provision_bronze_service(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test Bronze tier service provisioning"""
        # Setup mock for successful quorum validation
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.5,  # Bronze doesn't require high diversity
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:1],  # Bronze only needs 1 device
            metadata={
                "diversity_metrics": {
                    "unique_asns": 1,
                    "unique_tee_vendors": 1,
                    "unique_power_regions": 1,
                    "unique_countries": 1,
                }
            },
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        result = await sla_tier_manager.provision_service(
            service_id="test-service-bronze", tier=SLATier.BRONZE, available_devices=sample_devices
        )

        assert result["success"] is True
        assert result["tier"] == "bronze"
        assert result["pricing_multiplier"] == 1.0
        assert len(result["allocated_devices"]) == 1
        assert "sla_guarantees" in result

    @pytest.mark.asyncio
    async def test_provision_gold_service_success(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test successful Gold tier service provisioning"""
        # Setup mock for successful Gold tier quorum validation
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.9,  # High diversity for Gold
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles,  # All 3 devices
            metadata={
                "diversity_metrics": {
                    "unique_asns": 3,
                    "unique_tee_vendors": 2,
                    "unique_power_regions": 3,
                    "unique_countries": 1,
                }
            },
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        result = await sla_tier_manager.provision_service(
            service_id="test-service-gold", tier=SLATier.GOLD, available_devices=sample_devices
        )

        assert result["success"] is True
        assert result["tier"] == "gold"
        assert result["pricing_multiplier"] == 5.0
        assert len(result["allocated_devices"]) == 3
        assert result["diversity_score"] == 0.9
        assert "infrastructure_diversity" in result

    @pytest.mark.asyncio
    async def test_provision_gold_service_failure(self, sla_tier_manager, mock_quorum_manager, sample_devices):
        """Test failed Gold tier service provisioning due to insufficient diversity"""
        # Setup mock for failed quorum validation
        mock_quorum_result = QuorumValidationResult(
            is_valid=False,
            diversity_score=0.3,  # Low diversity
            violations=["Insufficient ASN diversity: 1 < 3", "Insufficient TEE vendor diversity: 1 < 2"],
            recommendations=["Add 2 more devices from different ASNs", "Add Intel TDX capable device"],
            profiles_used=[],
            metadata={},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        result = await sla_tier_manager.provision_service(
            service_id="test-service-gold-fail", tier=SLATier.GOLD, available_devices=sample_devices
        )

        assert result["success"] is False
        assert result["error"] == "Failed to meet infrastructure diversity requirements"
        assert len(result["violations"]) == 2
        assert len(result["recommendations"]) == 2

    @pytest.mark.asyncio
    async def test_provision_service_with_config(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test service provisioning with custom configuration"""
        # Setup mock
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        service_config = {"cpu_cores": 4, "memory_gb": 8, "storage_gb": 100, "network_bandwidth_mbps": 1000}

        result = await sla_tier_manager.provision_service(
            service_id="test-service-config",
            tier=SLATier.SILVER,
            available_devices=sample_devices,
            service_config=service_config,
        )

        assert result["success"] is True
        assert service_config in sla_tier_manager.service_instances["test-service-config"].metadata.values()


class TestSLACompliance:
    """Test SLA compliance validation"""

    @pytest.mark.asyncio
    async def test_validate_sla_compliance_success(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test successful SLA compliance validation"""
        # First provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="compliance-test", tier=SLATier.SILVER, available_devices=sample_devices
        )

        # Test compliance with good metrics
        good_metrics = SLAMetrics(
            p95_latency_ms=800.0,  # Under Silver limit of 1200ms
            uptime_percentage=99.5,  # Over Silver requirement of 99%
            error_rate_percentage=0.5,  # Under Silver limit of 1%
            throughput_ops_per_second=1000.0,
        )

        result = await sla_tier_manager.validate_sla_compliance("compliance-test", good_metrics)

        assert result["compliant"] is True
        assert result["service_id"] == "compliance-test"
        assert result["tier"] == "silver"
        assert len(result["violations"]) == 0

    @pytest.mark.asyncio
    async def test_validate_sla_compliance_violations(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test SLA compliance validation with violations"""
        # First provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="violation-test", tier=SLATier.SILVER, available_devices=sample_devices
        )

        # Test compliance with bad metrics
        bad_metrics = SLAMetrics(
            p95_latency_ms=2000.0,  # Over Silver limit of 1200ms
            uptime_percentage=98.0,  # Under Silver requirement of 99%
            error_rate_percentage=2.0,  # Over Silver limit of 1%
            throughput_ops_per_second=500.0,
        )

        result = await sla_tier_manager.validate_sla_compliance("violation-test", bad_metrics)

        assert result["compliant"] is False
        assert len(result["violations"]) == 3  # Latency, uptime, error rate

        # Check violation details
        violation_metrics = [v["metric"] for v in result["violations"]]
        assert "p95_latency_ms" in violation_metrics
        assert "uptime_percentage" in violation_metrics
        assert "error_rate_percentage" in violation_metrics

    @pytest.mark.asyncio
    async def test_validate_gold_tier_diversity_revalidation(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test Gold tier diversity revalidation"""
        # Provision Gold tier service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.9,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles,
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="gold-diversity-test", tier=SLATier.GOLD, available_devices=sample_devices
        )

        # Set last validation to over 1 hour ago to trigger revalidation
        service = sla_tier_manager.service_instances["gold-diversity-test"]
        service.last_validation = datetime.utcnow() - timedelta(hours=2)

        # Mock failed diversity revalidation
        mock_quorum_manager.validate_quorum.return_value = QuorumValidationResult(
            is_valid=False,
            diversity_score=0.3,
            violations=["Infrastructure diversity compromised"],
            recommendations=[],
            profiles_used=[],
            metadata={},
        )

        # Test compliance - should fail due to diversity
        good_metrics = SLAMetrics(
            p95_latency_ms=300.0, uptime_percentage=99.95, error_rate_percentage=0.05, throughput_ops_per_second=2000.0
        )

        result = await sla_tier_manager.validate_sla_compliance("gold-diversity-test", good_metrics)

        assert result["compliant"] is False
        assert result["diversity_valid"] is False
        assert any(v["metric"] == "infrastructure_diversity" for v in result["violations"])


class TestServiceRebalancing:
    """Test service rebalancing functionality"""

    @pytest.mark.asyncio
    async def test_rebalance_service_success(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test successful service rebalancing"""
        # First provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.7,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="rebalance-test", tier=SLATier.SILVER, available_devices=sample_devices
        )

        # Mark service as invalid to trigger rebalancing
        service = sla_tier_manager.service_instances["rebalance-test"]
        service.validation_status = "invalid"

        # Mock better allocation for rebalancing
        better_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.9,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles,
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = better_quorum_result

        result = await sla_tier_manager.rebalance_service(service_id="rebalance-test", available_devices=sample_devices)

        assert result["success"] is True
        assert result["action"] == "rebalanced"
        assert result["diversity_score"] == 0.9
        assert len(result["new_allocation"]) == 3
        assert result["diversity_improvement"] > 0

    @pytest.mark.asyncio
    async def test_rebalance_service_no_rebalance_needed(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test rebalancing when no rebalance is needed"""
        # Provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="no-rebalance-test", tier=SLATier.SILVER, available_devices=sample_devices
        )

        # Service is valid, no rebalancing needed
        result = await sla_tier_manager.rebalance_service(
            service_id="no-rebalance-test", available_devices=sample_devices
        )

        assert result["success"] is True
        assert result["action"] == "no_rebalance_needed"

    @pytest.mark.asyncio
    async def test_rebalance_service_failure(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test failed service rebalancing"""
        # Provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.7,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="rebalance-fail-test", tier=SLATier.GOLD, available_devices=sample_devices, force_rebalance=True
        )

        # Mock failed rebalancing attempt
        failed_quorum_result = QuorumValidationResult(
            is_valid=False,
            diversity_score=0.3,
            violations=["Cannot find valid allocation"],
            recommendations=[],
            profiles_used=[],
            metadata={},
        )
        mock_quorum_manager.validate_quorum.return_value = failed_quorum_result

        result = await sla_tier_manager.rebalance_service(
            service_id="rebalance-fail-test", available_devices=sample_devices, force_rebalance=True
        )

        assert result["success"] is False
        assert "Failed to find valid rebalancing allocation" in result["error"]


class TestServiceManagement:
    """Test service management functions"""

    def test_get_tier_pricing(self, sla_tier_manager):
        """Test tier pricing calculation"""
        assert sla_tier_manager.get_tier_pricing(SLATier.BRONZE, 100.0) == 100.0
        assert sla_tier_manager.get_tier_pricing(SLATier.SILVER, 100.0) == 250.0
        assert sla_tier_manager.get_tier_pricing(SLATier.GOLD, 100.0) == 500.0

    @pytest.mark.asyncio
    async def test_get_service_status(self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles):
        """Test getting service status"""
        # Provision a service first
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:2],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result
        mock_quorum_manager.get_quorum_status_summary.return_value = {"status": "active"}

        await sla_tier_manager.provision_service(
            service_id="status-test", tier=SLATier.SILVER, available_devices=sample_devices
        )

        status = sla_tier_manager.get_service_status("status-test")

        assert status is not None
        assert status["service_id"] == "status-test"
        assert status["tier"] == "silver"
        assert status["status"] == "active"
        assert status["device_count"] == 2
        assert status["diversity_score"] == 0.8
        assert "sla_requirements" in status
        assert "quorum_status" in status

    def test_get_service_status_nonexistent(self, sla_tier_manager):
        """Test getting status of nonexistent service"""
        status = sla_tier_manager.get_service_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_get_all_services_status(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test getting status of all services"""
        # Provision services in different tiers
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles,
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result
        mock_quorum_manager.get_quorum_status_summary.return_value = {"status": "active"}

        await sla_tier_manager.provision_service(
            service_id="bronze-service", tier=SLATier.BRONZE, available_devices=sample_devices
        )
        await sla_tier_manager.provision_service(
            service_id="gold-service", tier=SLATier.GOLD, available_devices=sample_devices
        )

        all_status = sla_tier_manager.get_all_services_status()

        assert all_status["total_services"] == 2
        assert all_status["tier_distribution"]["bronze"] == 1
        assert all_status["tier_distribution"]["gold"] == 1
        assert "services_by_tier" in all_status
        assert "sla_requirements" in all_status

    @pytest.mark.asyncio
    async def test_cleanup_expired_services(
        self, sla_tier_manager, mock_quorum_manager, sample_devices, sample_profiles
    ):
        """Test cleanup of expired services"""
        # Provision a service
        mock_quorum_result = QuorumValidationResult(
            is_valid=True,
            diversity_score=0.8,
            violations=[],
            recommendations=[],
            profiles_used=sample_profiles[:1],
            metadata={"diversity_metrics": {}},
        )
        mock_quorum_manager.validate_quorum.return_value = mock_quorum_result

        await sla_tier_manager.provision_service(
            service_id="cleanup-test", tier=SLATier.BRONZE, available_devices=sample_devices
        )

        # Make service old
        service = sla_tier_manager.service_instances["cleanup-test"]
        service.creation_time = datetime.utcnow() - timedelta(hours=25)  # Older than 24 hours

        cleanup_result = await sla_tier_manager.cleanup_expired_services(max_age_hours=24)

        assert cleanup_result["cleaned_up"] == 1
        assert "cleanup-test" in cleanup_result["expired_services"]
        assert len(sla_tier_manager.service_instances) == 0


if __name__ == "__main__":
    pytest.main([__file__])
