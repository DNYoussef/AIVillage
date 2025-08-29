"""
Tests for heterogeneous quorum manager and infrastructure diversity validation
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from infrastructure.fog.quorum.infrastructure_classifier import (
    InfrastructureClassifier,
    InfrastructureProfile,
    PowerRegion,
    TEEVendor,
)
from infrastructure.fog.quorum.quorum_manager import (
    DiversityConstraints,
    QuorumManager,
    QuorumRequirement,
    QuorumValidationResult,
)


@pytest.fixture
def mock_classifier():
    """Mock infrastructure classifier"""
    classifier = Mock(spec=InfrastructureClassifier)
    classifier.classify_device = AsyncMock()
    classifier.get_diversity_metrics = Mock()
    return classifier


@pytest.fixture
def quorum_manager(mock_classifier):
    """QuorumManager instance with mock classifier"""
    return QuorumManager(classifier=mock_classifier)


@pytest.fixture
def sample_profiles():
    """Sample infrastructure profiles for testing"""
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
            confidence_score=0.9
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
            confidence_score=0.95
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
            confidence_score=0.8
        )
    ]


@pytest.fixture
def device_candidates():
    """Sample device candidates"""
    return [
        {
            'id': 'device1',
            'ip_address': '10.1.1.1',
            'attestation_data': {'platform': 'amd', 'sev_snp': True},
            'network_info': {'datacenter': True}
        },
        {
            'id': 'device2',
            'ip_address': '10.2.2.2',
            'attestation_data': {'platform': 'intel', 'tdx': True},
            'network_info': {'public': True}
        },
        {
            'id': 'device3',
            'ip_address': '10.3.3.3',
            'attestation_data': {'platform': 'amd', 'sev_snp': True},
            'network_info': {'residential': True}
        }
    ]


class TestQuorumManager:
    """Test QuorumManager functionality"""

    @pytest.mark.asyncio
    async def test_validate_quorum_bronze_tier(self, quorum_manager, mock_classifier, device_candidates, sample_profiles):
        """Test Bronze tier quorum validation (no diversity requirements)"""
        # Setup mock
        mock_classifier.classify_device.side_effect = sample_profiles
        mock_classifier.get_diversity_metrics.return_value = {
            'unique_asns': 3,
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1,
            'total_diversity_score': 0.7
        }

        # Test Bronze tier (no requirements)
        result = await quorum_manager.validate_quorum(
            device_candidates=device_candidates,
            requirement=QuorumRequirement.NONE,
            target_size=3
        )

        assert result.is_valid is True
        assert len(result.profiles_used) == 3
        assert result.diversity_score == 0.7
        assert len(result.violations) == 0

    @pytest.mark.asyncio
    async def test_validate_quorum_gold_tier_success(self, quorum_manager, mock_classifier, device_candidates, sample_profiles):
        """Test Gold tier quorum validation with valid diversity"""
        # Setup mock for Gold tier requirements
        mock_classifier.classify_device.side_effect = sample_profiles
        mock_classifier.get_diversity_metrics.return_value = {
            'unique_asns': 3,
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1,
            'total_diversity_score': 0.85
        }

        # Test Gold tier with sufficient diversity
        result = await quorum_manager.validate_quorum(
            device_candidates=device_candidates,
            requirement=QuorumRequirement.GOLD,
            target_size=3
        )

        assert result.is_valid is True
        assert len(result.profiles_used) == 3
        assert result.diversity_score == 0.85
        assert result.metadata['constraint_level'] == 'gold'

    @pytest.mark.asyncio
    async def test_validate_quorum_gold_tier_failure(self, quorum_manager, mock_classifier, device_candidates):
        """Test Gold tier quorum validation with insufficient diversity"""
        # Create profiles with insufficient diversity (all same ASN)
        insufficient_profiles = [
            InfrastructureProfile(
                device_id=f"device{i}",
                asn=12345,  # Same ASN
                asn_name="Provider A",
                tee_vendor=TEEVendor.AMD_SEV_SNP,  # Same TEE vendor
                tee_version="1.0",
                power_region=PowerRegion.NERC_RFC,  # Same power region
                country_code="US",
                region="Virginia",
                city="Reston",
                network_topology="datacenter",
                attestation_hash=f"hash{i}",
                classification_time=datetime.utcnow(),
                confidence_score=0.9
            ) for i in range(3)
        ]

        # Setup mock
        mock_classifier.classify_device.side_effect = insufficient_profiles
        mock_classifier.get_diversity_metrics.return_value = {
            'unique_asns': 1,  # Insufficient
            'unique_tee_vendors': 1,  # Insufficient
            'unique_power_regions': 1,  # Insufficient
            'unique_countries': 1,
            'total_diversity_score': 0.25
        }

        # Test Gold tier with insufficient diversity
        result = await quorum_manager.validate_quorum(
            device_candidates=device_candidates,
            requirement=QuorumRequirement.GOLD,
            target_size=3
        )

        assert result.is_valid is False
        assert len(result.violations) > 0
        assert any('ASN diversity' in v for v in result.violations)
        assert any('TEE vendor diversity' in v for v in result.violations)
        assert any('power region diversity' in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_validate_quorum_insufficient_devices(self, quorum_manager, mock_classifier, device_candidates):
        """Test quorum validation with insufficient devices"""
        # Only return 1 profile but need 3
        single_profile = [
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
                confidence_score=0.9
            )
        ]

        mock_classifier.classify_device.side_effect = single_profile

        result = await quorum_manager.validate_quorum(
            device_candidates=[device_candidates[0]],  # Only one device
            requirement=QuorumRequirement.GOLD,
            target_size=3
        )

        assert result.is_valid is False
        assert any('Insufficient qualified devices' in v for v in result.violations)

    @pytest.mark.asyncio
    async def test_custom_constraints(self, quorum_manager, mock_classifier, device_candidates, sample_profiles):
        """Test quorum validation with custom constraints"""
        # Custom constraints requiring 4 unique ASNs
        custom_constraints = DiversityConstraints(
            min_asn_diversity=4,  # More than we have
            min_tee_vendor_diversity=1,
            min_power_region_diversity=1,
            min_geographic_diversity=1,
            min_confidence_score=0.8
        )

        mock_classifier.classify_device.side_effect = sample_profiles
        mock_classifier.get_diversity_metrics.return_value = {
            'unique_asns': 3,  # Less than required 4
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1,
            'total_diversity_score': 0.7
        }

        result = await quorum_manager.validate_quorum(
            device_candidates=device_candidates,
            requirement=QuorumRequirement.NONE,
            target_size=3,
            custom_constraints=custom_constraints
        )

        assert result.is_valid is False
        assert any('ASN diversity' in v for v in result.violations)

    def test_get_quorum_status_summary(self, quorum_manager, sample_profiles):
        """Test quorum status summary generation"""
        status = quorum_manager.get_quorum_status_summary(sample_profiles)

        assert status['status'] == 'active'
        assert status['devices'] == 3
        assert 'diversity_score' in status
        assert 'distributions' in status
        assert 'asn' in status['distributions']
        assert 'tee_vendor' in status['distributions']
        assert 'power_region' in status['distributions']

    def test_empty_quorum_status_summary(self, quorum_manager):
        """Test quorum status summary with empty profiles"""
        status = quorum_manager.get_quorum_status_summary([])

        assert status['status'] == 'empty'
        assert status['devices'] == 0

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, quorum_manager, mock_classifier, device_candidates, sample_profiles):
        """Test continuous quorum monitoring"""
        mock_classifier.classify_device.side_effect = sample_profiles
        mock_classifier.get_diversity_metrics.return_value = {
            'unique_asns': 3,
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1,
            'total_diversity_score': 0.85
        }

        # Track callback calls
        callback_results = []

        async def test_callback(result):
            callback_results.append(result)

        # Start monitoring with short interval for testing
        monitoring_task = asyncio.create_task(
            quorum_manager.continuously_monitor_quorum(
                device_candidates=device_candidates,
                requirement=QuorumRequirement.GOLD,
                callback=test_callback,
                interval_seconds=0.1
            )
        )

        # Let it run for a short time
        await asyncio.sleep(0.3)

        # Cancel monitoring
        monitoring_task.cancel()

        # Check that callback was called
        assert len(callback_results) > 0
        assert all(isinstance(result, QuorumValidationResult) for result in callback_results)


class TestConstraintValidation:
    """Test diversity constraint validation"""

    def test_validate_constraints_success(self, quorum_manager, sample_profiles):
        """Test successful constraint validation"""
        constraints = DiversityConstraints(
            min_asn_diversity=2,
            min_tee_vendor_diversity=2,
            min_power_region_diversity=2,
            min_geographic_diversity=1,
            max_devices_per_asn=1,
            max_devices_per_power_region=1
        )

        violations = quorum_manager._validate_constraints(sample_profiles, constraints)
        assert len(violations) == 0

    def test_validate_constraints_asn_violations(self, quorum_manager, sample_profiles):
        """Test ASN constraint violations"""
        # Modify profiles to have same ASN
        for profile in sample_profiles:
            profile.asn = 12345

        constraints = DiversityConstraints(
            min_asn_diversity=3,
            max_devices_per_asn=1
        )

        violations = quorum_manager._validate_constraints(sample_profiles, constraints)
        assert len(violations) >= 2  # Both min diversity and max per ASN
        assert any('ASN diversity' in v for v in violations)
        assert any('Too many devices in ASN' in v for v in violations)

    def test_validate_constraints_tee_requirement(self, quorum_manager, sample_profiles):
        """Test TEE diversity requirement"""
        # Set all to same TEE vendor
        for profile in sample_profiles:
            profile.tee_vendor = TEEVendor.AMD_SEV_SNP

        constraints = DiversityConstraints(
            require_tee_diversity=True
        )

        violations = quorum_manager._validate_constraints(sample_profiles, constraints)
        assert any('multiple known TEE vendors' in v for v in violations)

    def test_validate_constraints_confidence_threshold(self, quorum_manager, sample_profiles):
        """Test confidence score threshold"""
        # Set low confidence scores
        for profile in sample_profiles:
            profile.confidence_score = 0.4

        constraints = DiversityConstraints(
            min_confidence_score=0.8
        )

        violations = quorum_manager._validate_constraints(sample_profiles, constraints)
        assert any('below confidence threshold' in v for v in violations)


class TestRecommendations:
    """Test recommendation generation"""

    def test_generate_recommendations_all_satisfied(self, quorum_manager, sample_profiles):
        """Test recommendations when all constraints are satisfied"""
        constraints = DiversityConstraints()
        violations = []
        metrics = {
            'unique_asns': 3,
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1
        }

        recommendations = quorum_manager._generate_recommendations(
            sample_profiles, constraints, violations, metrics
        )

        assert len(recommendations) == 1
        assert 'meets all diversity requirements' in recommendations[0]

    def test_generate_recommendations_asn_diversity(self, quorum_manager, sample_profiles):
        """Test recommendations for ASN diversity"""
        constraints = DiversityConstraints(min_asn_diversity=5)
        violations = ['Insufficient ASN diversity']
        metrics = {
            'unique_asns': 3,
            'unique_tee_vendors': 2,
            'unique_power_regions': 3,
            'unique_countries': 1
        }

        recommendations = quorum_manager._generate_recommendations(
            sample_profiles, constraints, violations, metrics
        )

        assert any('different ASNs' in r for r in recommendations)

    def test_generate_recommendations_tee_diversity(self, quorum_manager, sample_profiles):
        """Test recommendations for TEE diversity"""
        # Remove Intel TDX
        for profile in sample_profiles:
            profile.tee_vendor = TEEVendor.AMD_SEV_SNP

        constraints = DiversityConstraints(min_tee_vendor_diversity=2)
        violations = ['Insufficient TEE vendor diversity']
        metrics = {
            'unique_asns': 3,
            'unique_tee_vendors': 1,
            'unique_power_regions': 3,
            'unique_countries': 1
        }

        recommendations = quorum_manager._generate_recommendations(
            sample_profiles, constraints, violations, metrics
        )

        assert any('Intel TDX' in r for r in recommendations)


class TestSelectionAlgorithm:
    """Test optimal quorum selection algorithm"""

    @pytest.mark.asyncio
    async def test_select_optimal_quorum_diversity_prioritization(self, quorum_manager, mock_classifier):
        """Test that selection algorithm prioritizes diversity"""
        # Create profiles with varying diversity potential
        profiles = [
            InfrastructureProfile(
                device_id="device1", asn=1, tee_vendor=TEEVendor.AMD_SEV_SNP,
                power_region=PowerRegion.NERC_RFC, country_code="US",
                region="VA", city="Reston", network_topology="dc",
                attestation_hash="1", classification_time=datetime.utcnow(),
                confidence_score=0.9
            ),
            InfrastructureProfile(
                device_id="device2", asn=2, tee_vendor=TEEVendor.INTEL_TDX,
                power_region=PowerRegion.NERC_WECC, country_code="US",
                region="CA", city="SF", network_topology="public",
                attestation_hash="2", classification_time=datetime.utcnow(),
                confidence_score=0.9
            ),
            InfrastructureProfile(
                device_id="device3", asn=3, tee_vendor=TEEVendor.ARM_TRUSTZONE,
                power_region=PowerRegion.NERC_SERC, country_code="CA",
                region="ON", city="Toronto", network_topology="mobile",
                attestation_hash="3", classification_time=datetime.utcnow(),
                confidence_score=0.9
            ),
            InfrastructureProfile(
                device_id="device4", asn=1, tee_vendor=TEEVendor.AMD_SEV_SNP,  # Same ASN as device1
                power_region=PowerRegion.NERC_RFC, country_code="US",  # Same region
                region="VA", city="Norfolk", network_topology="residential",
                attestation_hash="4", classification_time=datetime.utcnow(),
                confidence_score=0.9
            )
        ]

        constraints = DiversityConstraints(
            min_asn_diversity=3,
            min_tee_vendor_diversity=2,
            min_power_region_diversity=2
        )

        selected, score = await quorum_manager._select_optimal_quorum(profiles, constraints, 3)

        # Should select first 3 devices (most diverse)
        assert len(selected) == 3
        selected_ids = [p.device_id for p in selected]
        assert "device1" in selected_ids
        assert "device2" in selected_ids
        assert "device3" in selected_ids
        assert "device4" not in selected_ids  # Less diverse option


if __name__ == "__main__":
    pytest.main([__file__])
