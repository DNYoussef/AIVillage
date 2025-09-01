"""
Integration tests for heterogeneous quorum requirements system

Tests the complete end-to-end flow from device classification through
SLA tier provisioning and monitoring.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from infrastructure.fog.quorum.quorum_manager import QuorumManager, QuorumRequirement
from infrastructure.fog.quorum.infrastructure_classifier import InfrastructureClassifier, TEEVendor, PowerRegion
from infrastructure.fog.scheduler.enhanced_sla_tiers import EnhancedSLATierManager, SLATier, SLAMetrics
from infrastructure.fog.monitoring.diversity_dashboard import DiversityDashboard


@pytest.fixture
def realistic_device_pool():
    """Realistic device pool with diverse infrastructure"""
    return [
        # US East Coast - AMD SEV-SNP datacenter
        {
            "id": "dc-east-amd-1",
            "ip_address": "10.1.1.1",
            "attestation_data": {
                "platform": "amd",
                "sev_snp": True,
                "snp_version": "1.51",
                "measurement": "amd_measurement_123",
            },
            "network_info": {"datacenter": True, "provider": "AWS"},
        },
        # US West Coast - Intel TDX datacenter
        {
            "id": "dc-west-intel-1",
            "ip_address": "10.2.2.2",
            "attestation_data": {
                "platform": "intel",
                "tdx": True,
                "tdx_version": "2.0",
                "measurement": "intel_measurement_456",
            },
            "network_info": {"datacenter": True, "provider": "GCP"},
        },
        # US South - AMD SEV-SNP edge
        {
            "id": "edge-south-amd-1",
            "ip_address": "10.3.3.3",
            "attestation_data": {
                "platform": "amd",
                "sev_snp": True,
                "snp_version": "1.51",
                "measurement": "amd_measurement_789",
            },
            "network_info": {"datacenter": False, "edge": True},
        },
        # Canada - Intel TDX mobile
        {
            "id": "mobile-ca-intel-1",
            "ip_address": "10.4.4.4",
            "attestation_data": {
                "platform": "intel",
                "tdx": True,
                "tdx_version": "2.1",
                "measurement": "intel_measurement_abc",
            },
            "network_info": {"mobile": True, "country": "CA"},
        },
        # US Midwest - ARM TrustZone IoT
        {
            "id": "iot-midwest-arm-1",
            "ip_address": "10.5.5.5",
            "attestation_data": {
                "platform": "arm",
                "trustzone": True,
                "tz_version": "1.0",
                "measurement": "arm_measurement_def",
            },
            "network_info": {"iot": True, "residential": True},
        },
        # Europe - Intel TDX datacenter
        {
            "id": "dc-eu-intel-1",
            "ip_address": "10.6.6.6",
            "attestation_data": {
                "platform": "intel",
                "tdx": True,
                "tdx_version": "2.0",
                "measurement": "intel_measurement_ghi",
            },
            "network_info": {"datacenter": True, "provider": "Azure"},
        },
    ]


@pytest.fixture
async def integrated_system():
    """Fully integrated fog computing system for testing"""
    # Create real components (not mocked) for integration testing
    classifier = InfrastructureClassifier()
    quorum_manager = QuorumManager(classifier)
    sla_tier_manager = EnhancedSLATierManager(quorum_manager)
    dashboard = DiversityDashboard(quorum_manager, sla_tier_manager)

    return {
        "classifier": classifier,
        "quorum_manager": quorum_manager,
        "sla_tier_manager": sla_tier_manager,
        "dashboard": dashboard,
    }


class TestInfrastructureClassification:
    """Test infrastructure classification accuracy"""

    @pytest.mark.asyncio
    async def test_classify_diverse_devices(self, integrated_system, realistic_device_pool):
        """Test classification of diverse device pool"""
        classifier = integrated_system["classifier"]

        # Mock external services (GeoIP, ASN lookups)
        with patch.object(classifier, "_classify_asn") as mock_asn, patch.object(
            classifier, "_classify_geo_location"
        ) as mock_geo:

            # Setup realistic mock responses
            mock_asn.side_effect = [
                {"asn": 16509, "name": "Amazon.com Inc."},
                {"asn": 15169, "name": "Google LLC"},
                {"asn": 7922, "name": "Comcast Cable"},
                {"asn": 577, "name": "Bell Canada"},
                {"asn": 7018, "name": "AT&T Services Inc."},
                {"asn": 8075, "name": "Microsoft Corporation"},
            ]

            mock_geo.side_effect = [
                {"country_code": "US", "region": "Virginia", "city": "Ashburn"},
                {"country_code": "US", "region": "California", "city": "Mountain View"},
                {"country_code": "US", "region": "Georgia", "city": "Atlanta"},
                {"country_code": "CA", "region": "Ontario", "city": "Toronto"},
                {"country_code": "US", "region": "Iowa", "city": "Des Moines"},
                {"country_code": "IE", "region": "Dublin", "city": "Dublin"},
            ]

            # Classify all devices
            profiles = []
            for device in realistic_device_pool:
                profile = await classifier.classify_device(
                    device_id=device["id"],
                    ip_address=device["ip_address"],
                    attestation_data=device["attestation_data"],
                    network_info=device["network_info"],
                )
                profiles.append(profile)

        # Verify classification results
        assert len(profiles) == 6

        # Check TEE vendor diversity
        tee_vendors = set(p.tee_vendor for p in profiles)
        assert TEEVendor.AMD_SEV_SNP in tee_vendors
        assert TEEVendor.INTEL_TDX in tee_vendors
        assert TEEVendor.ARM_TRUSTZONE in tee_vendors

        # Check ASN diversity
        asns = set(p.asn for p in profiles if p.asn)
        assert len(asns) >= 5  # Should have diverse ASNs

        # Check power region diversity
        power_regions = set(p.power_region for p in profiles)
        assert PowerRegion.NERC_RFC in power_regions  # Virginia
        assert PowerRegion.NERC_WECC in power_regions  # California
        assert PowerRegion.NERC_SERC in power_regions  # Georgia
        assert PowerRegion.INTERNATIONAL in power_regions  # Ireland

        # Check confidence scores
        avg_confidence = sum(p.confidence_score for p in profiles) / len(profiles)
        assert avg_confidence > 0.8  # Should have high confidence


class TestQuorumValidation:
    """Test quorum validation with realistic scenarios"""

    @pytest.mark.asyncio
    async def test_gold_tier_quorum_validation_success(self, integrated_system, realistic_device_pool):
        """Test successful Gold tier quorum validation"""
        quorum_manager = integrated_system["quorum_manager"]

        # Mock the classifier methods for realistic results
        with patch.object(quorum_manager, "_classify_devices") as mock_classify:
            # Create diverse profiles that should pass Gold tier
            mock_profiles = [
                # Device 1: US East, AMD, Power Region 1, ASN 1
                Mock(
                    device_id="dc-east-amd-1",
                    asn=16509,
                    tee_vendor=TEEVendor.AMD_SEV_SNP,
                    power_region=PowerRegion.NERC_RFC,
                    country_code="US",
                    confidence_score=0.95,
                ),
                # Device 2: US West, Intel, Power Region 2, ASN 2
                Mock(
                    device_id="dc-west-intel-1",
                    asn=15169,
                    tee_vendor=TEEVendor.INTEL_TDX,
                    power_region=PowerRegion.NERC_WECC,
                    country_code="US",
                    confidence_score=0.9,
                ),
                # Device 3: US South, AMD, Power Region 3, ASN 3
                Mock(
                    device_id="edge-south-amd-1",
                    asn=7922,
                    tee_vendor=TEEVendor.AMD_SEV_SNP,
                    power_region=PowerRegion.NERC_SERC,
                    country_code="US",
                    confidence_score=0.85,
                ),
            ]
            mock_classify.return_value = mock_profiles

            # Mock diversity metrics
            quorum_manager.classifier.get_diversity_metrics.return_value = {
                "unique_asns": 3,
                "unique_tee_vendors": 2,
                "unique_power_regions": 3,
                "unique_countries": 1,
                "total_diversity_score": 0.9,
            }

            result = await quorum_manager.validate_quorum(
                device_candidates=realistic_device_pool[:3], requirement=QuorumRequirement.GOLD, target_size=3
            )

            assert result.is_valid is True
            assert result.diversity_score == 0.9
            assert len(result.violations) == 0
            assert len(result.profiles_used) == 3

    @pytest.mark.asyncio
    async def test_gold_tier_quorum_validation_failure(self, integrated_system, realistic_device_pool):
        """Test Gold tier quorum validation failure scenarios"""
        quorum_manager = integrated_system["quorum_manager"]

        # Test scenario: All devices from same ASN and power region
        with patch.object(quorum_manager, "_classify_devices") as mock_classify:
            # Create profiles with insufficient diversity
            insufficient_profiles = [
                Mock(
                    device_id=f"device{i}",
                    asn=16509,
                    tee_vendor=TEEVendor.AMD_SEV_SNP,
                    power_region=PowerRegion.NERC_RFC,
                    country_code="US",
                    confidence_score=0.9,
                )
                for i in range(3)
            ]
            mock_classify.return_value = insufficient_profiles

            quorum_manager.classifier.get_diversity_metrics.return_value = {
                "unique_asns": 1,  # Fails requirement
                "unique_tee_vendors": 1,  # Fails requirement
                "unique_power_regions": 1,  # Fails requirement
                "unique_countries": 1,
                "total_diversity_score": 0.25,
            }

            result = await quorum_manager.validate_quorum(
                device_candidates=realistic_device_pool[:3], requirement=QuorumRequirement.GOLD, target_size=3
            )

            assert result.is_valid is False
            assert len(result.violations) >= 3  # ASN, TEE, power region violations
            assert any("ASN diversity" in v for v in result.violations)
            assert any("TEE vendor diversity" in v for v in result.violations)
            assert any("power region diversity" in v for v in result.violations)


class TestSLATierProvisioning:
    """Test SLA tier provisioning with realistic scenarios"""

    @pytest.mark.asyncio
    async def test_provision_gold_tier_service_end_to_end(self, integrated_system, realistic_device_pool):
        """Test complete Gold tier service provisioning"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Mock successful quorum validation
        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.92,
                violations=[],
                recommendations=[],
                profiles_used=[Mock(device_id=f"device{i}") for i in range(3)],
                metadata={
                    "diversity_metrics": {
                        "unique_asns": 3,
                        "unique_tee_vendors": 2,
                        "unique_power_regions": 3,
                        "unique_countries": 2,
                    }
                },
            )

            # Provision Gold tier service
            result = await sla_tier_manager.provision_service(
                service_id="gold-critical-service",
                tier=SLATier.GOLD,
                available_devices=realistic_device_pool,
                service_config={
                    "workload_type": "financial_trading",
                    "data_sensitivity": "highly_confidential",
                    "compliance_requirements": ["SOC2", "PCI-DSS"],
                },
            )

            assert result["success"] is True
            assert result["tier"] == "gold"
            assert result["diversity_score"] == 0.92
            assert result["pricing_multiplier"] == 5.0
            assert "infrastructure_diversity" in result

            # Verify service is tracked
            service_status = sla_tier_manager.get_service_status("gold-critical-service")
            assert service_status is not None
            assert service_status["tier"] == "gold"
            assert service_status["diversity_score"] == 0.92

    @pytest.mark.asyncio
    async def test_provision_multiple_tier_services(self, integrated_system, realistic_device_pool):
        """Test provisioning services across different tiers"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Mock quorum validation for different tiers
        def mock_validate_side_effect(device_candidates, requirement, target_size, **kwargs):
            if requirement == QuorumRequirement.NONE:  # Bronze
                return Mock(
                    is_valid=True,
                    diversity_score=0.5,
                    violations=[],
                    recommendations=[],
                    profiles_used=[Mock(device_id="bronze-device")],
                    metadata={"diversity_metrics": {}},
                )
            elif requirement == QuorumRequirement.ENHANCED:  # Silver
                return Mock(
                    is_valid=True,
                    diversity_score=0.75,
                    violations=[],
                    recommendations=[],
                    profiles_used=[Mock(device_id=f"silver-device{i}") for i in range(2)],
                    metadata={"diversity_metrics": {}},
                )
            elif requirement == QuorumRequirement.GOLD:  # Gold
                return Mock(
                    is_valid=True,
                    diversity_score=0.9,
                    violations=[],
                    recommendations=[],
                    profiles_used=[Mock(device_id=f"gold-device{i}") for i in range(3)],
                    metadata={"diversity_metrics": {}},
                )

        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum", side_effect=mock_validate_side_effect):
            # Provision Bronze service
            bronze_result = await sla_tier_manager.provision_service(
                service_id="bronze-basic-service", tier=SLATier.BRONZE, available_devices=realistic_device_pool
            )

            # Provision Silver service
            silver_result = await sla_tier_manager.provision_service(
                service_id="silver-balanced-service", tier=SLATier.SILVER, available_devices=realistic_device_pool
            )

            # Provision Gold service
            gold_result = await sla_tier_manager.provision_service(
                service_id="gold-premium-service", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

            # Verify all services provisioned successfully
            assert bronze_result["success"] is True
            assert silver_result["success"] is True
            assert gold_result["success"] is True

            # Verify pricing differences
            assert bronze_result["pricing_multiplier"] == 1.0
            assert silver_result["pricing_multiplier"] == 2.5
            assert gold_result["pricing_multiplier"] == 5.0

            # Verify diversity score progression
            assert bronze_result["diversity_score"] == 0.5
            assert silver_result["diversity_score"] == 0.75
            assert gold_result["diversity_score"] == 0.9

            # Check overall system status
            all_services = sla_tier_manager.get_all_services_status()
            assert all_services["total_services"] == 3
            assert all_services["tier_distribution"]["bronze"] == 1
            assert all_services["tier_distribution"]["silver"] == 1
            assert all_services["tier_distribution"]["gold"] == 1


class TestSLAComplianceMonitoring:
    """Test SLA compliance monitoring and alerts"""

    @pytest.mark.asyncio
    async def test_sla_compliance_monitoring_cycle(self, integrated_system, realistic_device_pool):
        """Test complete SLA compliance monitoring cycle"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Provision a Gold service first
        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.9,
                violations=[],
                recommendations=[],
                profiles_used=[Mock(device_id=f"device{i}") for i in range(3)],
                metadata={"diversity_metrics": {}},
            )

            await sla_tier_manager.provision_service(
                service_id="monitoring-test-service", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

        # Test compliance with good metrics
        good_metrics = SLAMetrics(
            p95_latency_ms=350.0,  # Under Gold limit of 400ms
            uptime_percentage=99.95,  # Over Gold requirement of 99.9%
            error_rate_percentage=0.05,  # Under Gold limit of 0.1%
            throughput_ops_per_second=5000.0,
        )

        # Mock diversity revalidation as successful
        mock_validate.return_value = Mock(is_valid=True, violations=[])

        compliance_result = await sla_tier_manager.validate_sla_compliance("monitoring-test-service", good_metrics)

        assert compliance_result["compliant"] is True
        assert len(compliance_result["violations"]) == 0

        # Test compliance with bad metrics (SLA breach)
        bad_metrics = SLAMetrics(
            p95_latency_ms=600.0,  # Over Gold limit of 400ms
            uptime_percentage=99.5,  # Under Gold requirement of 99.9%
            error_rate_percentage=0.2,  # Over Gold limit of 0.1%
            throughput_ops_per_second=500.0,
        )

        breach_result = await sla_tier_manager.validate_sla_compliance("monitoring-test-service", bad_metrics)

        assert breach_result["compliant"] is False
        assert len(breach_result["violations"]) == 3

        # Verify violation details
        violation_types = [v["metric"] for v in breach_result["violations"]]
        assert "p95_latency_ms" in violation_types
        assert "uptime_percentage" in violation_types
        assert "error_rate_percentage" in violation_types

    @pytest.mark.asyncio
    async def test_diversity_breach_detection(self, integrated_system, realistic_device_pool):
        """Test detection of infrastructure diversity breaches"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Provision Gold service
        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            # Initial provisioning succeeds
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.9,
                violations=[],
                recommendations=[],
                profiles_used=[Mock(device_id=f"device{i}") for i in range(3)],
                metadata={"diversity_metrics": {}},
            )

            await sla_tier_manager.provision_service(
                service_id="diversity-breach-test", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

            # Force service to need revalidation
            service = sla_tier_manager.service_instances["diversity-breach-test"]
            service.last_validation = datetime.utcnow() - timedelta(hours=2)

            # Mock diversity breach during revalidation
            mock_validate.return_value = Mock(
                is_valid=False,
                violations=[
                    "Infrastructure diversity compromised - ASN consolidation detected",
                    "Power region diversity below Gold tier requirements",
                ],
            )

            # Test SLA compliance with good performance metrics but bad diversity
            good_metrics = SLAMetrics(
                p95_latency_ms=300.0,
                uptime_percentage=99.95,
                error_rate_percentage=0.03,
                throughput_ops_per_second=8000.0,
            )

            result = await sla_tier_manager.validate_sla_compliance("diversity-breach-test", good_metrics)

            assert result["compliant"] is False
            assert result["diversity_valid"] is False

            # Should have infrastructure diversity violation
            diversity_violations = [v for v in result["violations"] if v["metric"] == "infrastructure_diversity"]
            assert len(diversity_violations) == 1
            assert diversity_violations[0]["severity"] == "critical"


class TestDiversityDashboard:
    """Test diversity monitoring dashboard"""

    @pytest.mark.asyncio
    async def test_dashboard_monitoring_integration(self, integrated_system, realistic_device_pool):
        """Test integration with diversity monitoring dashboard"""
        dashboard = integrated_system["dashboard"]
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Provision some services first
        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.85,
                violations=[],
                recommendations=[],
                profiles_used=[Mock(device_id=f"device{i}") for i in range(3)],
                metadata={"diversity_metrics": {"unique_asns": 3, "unique_tee_vendors": 2}},
            )

            await sla_tier_manager.provision_service(
                service_id="dashboard-test-service-1", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

            await sla_tier_manager.provision_service(
                service_id="dashboard-test-service-2", tier=SLATier.SILVER, available_devices=realistic_device_pool
            )

        # Get current dashboard state
        dashboard_state = dashboard.get_current_dashboard()

        assert dashboard_state["monitoring_status"] == "inactive"  # Not started monitoring yet
        assert len(dashboard_state["service_statuses"]) == 0  # No monitoring data yet

        # Test dashboard with mock monitoring data
        dashboard.service_statuses = {
            "dashboard-test-service-1": Mock(
                service_id="dashboard-test-service-1",
                tier="gold",
                status="healthy",
                diversity_score=0.85,
                sla_compliance=True,
            ),
            "dashboard-test-service-2": Mock(
                service_id="dashboard-test-service-2",
                tier="silver",
                status="healthy",
                diversity_score=0.75,
                sla_compliance=True,
            ),
        }

        # Create some mock alerts
        dashboard.active_alerts = {
            "test-alert-1": Mock(
                alert_id="test-alert-1",
                severity="medium",
                violation_type="asn_diversity_medium",
                description="ASN diversity below optimal level",
            )
        }

        dashboard_state = dashboard.get_current_dashboard()

        assert len(dashboard_state["service_statuses"]) == 2
        assert dashboard_state["alert_summary"]["total_active"] == 1
        assert dashboard_state["alert_summary"]["medium"] == 1


class TestFogCoordinatorIntegration:
    """Test integration with fog coordinator"""

    @pytest.mark.asyncio
    async def test_fog_coordinator_sla_service_requests(self, realistic_device_pool):
        """Test SLA service requests through fog coordinator"""
        # Create fog coordinator with mocked dependencies
        coordinator = Mock()
        coordinator.sla_tier_manager = Mock()
        coordinator.quorum_manager = Mock()

        # Mock successful Gold tier provisioning
        coordinator.sla_tier_manager.provision_service = AsyncMock(
            return_value={
                "success": True,
                "service_id": "coord-test-service",
                "tier": "gold",
                "diversity_score": 0.92,
                "pricing_multiplier": 5.0,
                "allocated_devices": ["device1", "device2", "device3"],
            }
        )

        # Test provisioning request
        request_data = {
            "service_id": "coord-test-service",
            "tier": "GOLD",
            "available_devices": realistic_device_pool,
            "service_config": {
                "workload_type": "high_frequency_trading",
                "latency_requirements": "ultra_low",
                "compliance": ["SOX", "MiFID"],
            },
        }

        # Simulate the coordinator's process_fog_request method
        result = await coordinator.sla_tier_manager.provision_service(
            service_id=request_data["service_id"],
            tier=SLATier.GOLD,
            available_devices=request_data["available_devices"],
            service_config=request_data["service_config"],
        )

        assert result["success"] is True
        assert result["tier"] == "gold"
        assert result["diversity_score"] > 0.9
        assert len(result["allocated_devices"]) == 3


class TestResilienceAndFailover:
    """Test system resilience and failover scenarios"""

    @pytest.mark.asyncio
    async def test_device_failure_and_rebalancing(self, integrated_system, realistic_device_pool):
        """Test automatic rebalancing when devices fail"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Provision Gold service with 3 devices
        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            # Initial provisioning succeeds
            initial_profiles = [Mock(device_id=f"device{i}") for i in range(3)]
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.9,
                violations=[],
                recommendations=[],
                profiles_used=initial_profiles,
                metadata={"diversity_metrics": {}},
            )

            await sla_tier_manager.provision_service(
                service_id="resilience-test-service", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

            # Simulate device failure requiring rebalancing
            # Mock rebalancing with different devices
            rebalanced_profiles = [Mock(device_id=f"new-device{i}") for i in range(3)]
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.92,
                violations=[],
                recommendations=[],
                profiles_used=rebalanced_profiles,
                metadata={"diversity_metrics": {}},
            )

            rebalance_result = await sla_tier_manager.rebalance_service(
                service_id="resilience-test-service", available_devices=realistic_device_pool, force_rebalance=True
            )

            assert rebalance_result["success"] is True
            assert rebalance_result["action"] == "rebalanced"
            assert rebalance_result["diversity_score"] == 0.92
            assert rebalance_result["diversity_improvement"] > 0

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, integrated_system, realistic_device_pool):
        """Test prevention of cascading failures in Gold tier services"""
        sla_tier_manager = integrated_system["sla_tier_manager"]

        # Provision multiple Gold services
        services_to_provision = ["critical-service-1", "critical-service-2", "critical-service-3"]

        with patch.object(sla_tier_manager.quorum_manager, "validate_quorum") as mock_validate:
            mock_validate.return_value = Mock(
                is_valid=True,
                diversity_score=0.88,
                violations=[],
                recommendations=[],
                profiles_used=[Mock(device_id=f"device{i}") for i in range(3)],
                metadata={"diversity_metrics": {}},
            )

            # Provision multiple services
            for service_id in services_to_provision:
                result = await sla_tier_manager.provision_service(
                    service_id=service_id, tier=SLATier.GOLD, available_devices=realistic_device_pool
                )
                assert result["success"] is True

            # Simulate infrastructure stress (low diversity available)
            mock_validate.return_value = Mock(
                is_valid=False,
                violations=["Insufficient infrastructure diversity across multiple services"],
                recommendations=["Add more diverse infrastructure to prevent cascading failures"],
            )

            # Attempt to provision another service - should fail gracefully
            stressed_result = await sla_tier_manager.provision_service(
                service_id="stressed-service", tier=SLATier.GOLD, available_devices=realistic_device_pool
            )

            assert stressed_result["success"] is False
            assert "infrastructure diversity" in stressed_result["error"]

            # Existing services should remain unaffected
            all_services = sla_tier_manager.get_all_services_status()
            assert all_services["total_services"] == 3  # Original services still active


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
