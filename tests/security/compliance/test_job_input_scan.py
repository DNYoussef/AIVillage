"""
Tests for job input PII/PHI scanning and compliance validation

Tests the fog computing security policy that validates job inputs
for PII/PHI violations according to Task 3 acceptance criteria:
- PII violation → 422 with audit log
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from packages.core.compliance.pii_phi_manager import PIIPHIManager


@pytest.fixture
async def pii_phi_manager():
    """Create PII/PHI manager for testing"""
    manager = PIIPHIManager()
    await manager.initialize()
    return manager


@pytest.fixture
def sample_job_inputs():
    """Sample job inputs for testing"""
    return {
        "clean_job": {
            "payload": {"message": "Hello world", "count": 42},
            "env_vars": {"LOG_LEVEL": "INFO"},
            "args": ["--verbose"],
            "input_data": b"Simple data",
            "metadata": {"type": "greeting"},
        },
        "pii_in_payload": {
            "payload": {"email": "john.doe@example.com", "message": "Hello"},
            "env_vars": {},
            "args": [],
            "input_data": b"",
            "metadata": {},
        },
        "ssn_in_env": {
            "payload": {"message": "Hello"},
            "env_vars": {"USER_SSN": "123-45-6789"},
            "args": [],
            "input_data": b"",
            "metadata": {},
        },
        "phi_in_data": {
            "payload": {"message": "Hello"},
            "env_vars": {},
            "args": [],
            "input_data": b"Patient ID: 12345, Blood pressure: 120/80",
            "metadata": {},
        },
        "credit_card_in_args": {
            "payload": {"message": "Hello"},
            "env_vars": {},
            "args": ["--card", "4532-1234-5678-9012"],
            "input_data": b"",
            "metadata": {},
        },
        "phone_in_metadata": {
            "payload": {"message": "Hello"},
            "env_vars": {},
            "args": [],
            "input_data": b"",
            "metadata": {"contact": "555-123-4567"},
        },
    }


class TestJobInputScanning:
    """Test job input scanning for PII/PHI violations"""

    @pytest.mark.asyncio
    async def test_clean_job_passes_validation(self, pii_phi_manager, sample_job_inputs):
        """Test that clean job inputs pass validation"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_001", job_inputs=sample_job_inputs["clean_job"], namespace="test_ns", user_id="user_001"
        )

        assert result["violations_found"] is False
        assert len(result["violations"]) == 0
        assert result["compliance_status"] == "PASS"
        assert result["scan_time_ms"] > 0

        # Validation should allow the job
        validation = await pii_phi_manager.validate_job_compliance(
            job_id="job_001", job_inputs=sample_job_inputs["clean_job"], namespace="test_ns", user_id="user_001"
        )

        assert validation["allowed"] is True
        assert len(validation["violations"]) == 0

    @pytest.mark.asyncio
    async def test_email_in_payload_detected(self, pii_phi_manager, sample_job_inputs):
        """Test that email addresses in payload are detected as PII"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_002", job_inputs=sample_job_inputs["pii_in_payload"], namespace="test_ns"
        )

        assert result["violations_found"] is True
        assert len(result["violations"]) >= 1

        # Check for email violation
        email_violations = [v for v in result["violations"] if v["type"] == "EMAIL"]
        assert len(email_violations) == 1

        violation = email_violations[0]
        assert violation["location"] == "payload"
        assert violation["field"] == "email"
        assert violation["severity"] == "MEDIUM"
        assert "john.doe@example.com" in violation["matched_value"]

    @pytest.mark.asyncio
    async def test_ssn_in_env_vars_detected(self, pii_phi_manager, sample_job_inputs):
        """Test that SSN in environment variables is detected"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_003", job_inputs=sample_job_inputs["ssn_in_env"], namespace="test_ns"
        )

        assert result["violations_found"] is True

        # Check for SSN violation
        ssn_violations = [v for v in result["violations"] if v["type"] == "SSN"]
        assert len(ssn_violations) == 1

        violation = ssn_violations[0]
        assert violation["location"] == "env_vars"
        assert violation["field"] == "USER_SSN"
        assert violation["severity"] == "HIGH"

    @pytest.mark.asyncio
    async def test_phi_in_input_data_detected(self, pii_phi_manager, sample_job_inputs):
        """Test that PHI in input data is detected"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_004", job_inputs=sample_job_inputs["phi_in_data"], namespace="test_ns"
        )

        assert result["violations_found"] is True

        # Check for medical data violation
        medical_violations = [v for v in result["violations"] if v["type"] == "MEDICAL_DATA"]
        assert len(medical_violations) >= 1

        violation = medical_violations[0]
        assert violation["location"] == "input_data"
        assert violation["severity"] in ["HIGH", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_credit_card_in_args_detected(self, pii_phi_manager, sample_job_inputs):
        """Test that credit card numbers in args are detected"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_005", job_inputs=sample_job_inputs["credit_card_in_args"], namespace="test_ns"
        )

        assert result["violations_found"] is True

        # Check for credit card violation
        cc_violations = [v for v in result["violations"] if v["type"] == "CREDIT_CARD"]
        assert len(cc_violations) == 1

        violation = cc_violations[0]
        assert violation["location"] == "args"
        assert violation["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_phone_in_metadata_detected(self, pii_phi_manager, sample_job_inputs):
        """Test that phone numbers in metadata are detected"""

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_006", job_inputs=sample_job_inputs["phone_in_metadata"], namespace="test_ns"
        )

        assert result["violations_found"] is True

        # Check for phone violation
        phone_violations = [v for v in result["violations"] if v["type"] == "PHONE"]
        assert len(phone_violations) == 1

        violation = phone_violations[0]
        assert violation["location"] == "metadata"
        assert violation["field"] == "contact"
        assert violation["severity"] == "LOW"

    @pytest.mark.asyncio
    async def test_strict_mode_blocks_violations(self, pii_phi_manager, sample_job_inputs):
        """Test that strict mode blocks jobs with PII/PHI violations"""

        validation = await pii_phi_manager.validate_job_compliance(
            job_id="job_007",
            job_inputs=sample_job_inputs["pii_in_payload"],
            namespace="test_ns",
            user_id="user_001",
            strict_mode=True,
        )

        assert validation["allowed"] is False
        assert len(validation["violations"]) >= 1
        assert validation["reason"] == "PII/PHI violations detected in job inputs"
        assert validation["http_status"] == 422  # Task 3 acceptance criteria

    @pytest.mark.asyncio
    async def test_lenient_mode_allows_with_warnings(self, pii_phi_manager, sample_job_inputs):
        """Test that lenient mode allows jobs with warnings"""

        validation = await pii_phi_manager.validate_job_compliance(
            job_id="job_008",
            job_inputs=sample_job_inputs["phone_in_metadata"],  # Low severity
            namespace="test_ns",
            user_id="user_001",
            strict_mode=False,
        )

        # Low severity violations might be allowed in lenient mode
        # This depends on the specific implementation
        assert "violations" in validation
        assert "warnings" in validation

    @pytest.mark.asyncio
    async def test_audit_logging_for_violations(self, pii_phi_manager, sample_job_inputs):
        """Test that violations are properly logged for audit"""

        # Mock the audit logging
        with patch.object(pii_phi_manager, "_log_compliance_event") as mock_log:
            await pii_phi_manager.validate_job_compliance(
                job_id="job_009",
                job_inputs=sample_job_inputs["ssn_in_env"],
                namespace="test_ns",
                user_id="user_001",
                strict_mode=True,
            )

            # Should have logged the violation
            mock_log.assert_called()

            # Check the logged event
            call_args = mock_log.call_args[1]
            assert call_args["event_type"] == "job_validation_failed"
            assert call_args["job_id"] == "job_009"
            assert call_args["user_id"] == "user_001"
            assert call_args["namespace"] == "test_ns"
            assert "violations" in call_args["details"]

    @pytest.mark.asyncio
    async def test_multiple_violations_detected(self, pii_phi_manager):
        """Test that multiple violations in the same job are all detected"""

        complex_job = {
            "payload": {"email": "test@example.com", "ssn": "123-45-6789"},
            "env_vars": {"CC_NUMBER": "4532-1234-5678-9012"},
            "args": ["--phone", "555-123-4567"],
            "input_data": b"Patient Name: John Doe, DOB: 1980-01-01",
            "metadata": {"sensitive": "data"},
        }

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="job_010", job_inputs=complex_job, namespace="test_ns"
        )

        assert result["violations_found"] is True
        assert len(result["violations"]) >= 4  # Email, SSN, Credit Card, Phone

        # Check that different violation types are present
        violation_types = {v["type"] for v in result["violations"]}
        expected_types = {"EMAIL", "SSN", "CREDIT_CARD", "PHONE"}
        assert expected_types.issubset(violation_types)

    @pytest.mark.asyncio
    async def test_compliance_metrics_tracking(self, pii_phi_manager, sample_job_inputs):
        """Test that compliance metrics are properly tracked"""

        # Scan several jobs
        jobs = [
            ("clean", sample_job_inputs["clean_job"]),
            ("pii", sample_job_inputs["pii_in_payload"]),
            ("phi", sample_job_inputs["phi_in_data"]),
        ]

        for job_name, job_inputs in jobs:
            await pii_phi_manager.scan_job_inputs_for_compliance(
                job_id=f"metrics_test_{job_name}", job_inputs=job_inputs, namespace="metrics_ns"
            )

        # Get compliance metrics
        metrics = pii_phi_manager.get_compliance_metrics("metrics_ns")

        assert metrics["total_scans"] >= 3
        assert metrics["violations_detected"] >= 2
        assert metrics["clean_scans"] >= 1
        assert "violation_types" in metrics
        assert "scan_performance" in metrics

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, pii_phi_manager, sample_job_inputs):
        """Test that compliance tracking is isolated by namespace"""

        # Scan in different namespaces
        await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="iso_test_1", job_inputs=sample_job_inputs["pii_in_payload"], namespace="ns1"
        )

        await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="iso_test_2", job_inputs=sample_job_inputs["clean_job"], namespace="ns2"
        )

        # Check metrics are isolated
        ns1_metrics = pii_phi_manager.get_compliance_metrics("ns1")
        ns2_metrics = pii_phi_manager.get_compliance_metrics("ns2")

        assert ns1_metrics["violations_detected"] >= 1
        assert ns2_metrics["violations_detected"] == 0
        assert ns1_metrics["total_scans"] >= 1
        assert ns2_metrics["total_scans"] >= 1

    @pytest.mark.asyncio
    async def test_performance_within_limits(self, pii_phi_manager, sample_job_inputs):
        """Test that scanning performance is within acceptable limits"""

        import time

        # Large job input for performance testing
        large_job = {
            "payload": {f"field_{i}": f"value_{i}" for i in range(100)},
            "env_vars": {f"ENV_{i}": f"value_{i}" for i in range(50)},
            "args": [f"arg_{i}" for i in range(20)],
            "input_data": b"Large data block " * 1000,
            "metadata": {f"meta_{i}": f"value_{i}" for i in range(30)},
        }

        start_time = time.time()

        result = await pii_phi_manager.scan_job_inputs_for_compliance(
            job_id="perf_test", job_inputs=large_job, namespace="perf_ns"
        )

        scan_time = time.time() - start_time

        # Should complete within reasonable time (adjust threshold as needed)
        assert scan_time < 5.0  # 5 seconds max
        assert result["scan_time_ms"] < 5000
        assert "violations_found" in result


class TestComplianceIntegration:
    """Integration tests for compliance with fog gateway"""

    @pytest.mark.asyncio
    async def test_fog_gateway_integration(self, pii_phi_manager, sample_job_inputs):
        """Test integration with fog gateway security policy"""

        # This would test the actual integration with the fog gateway
        # For now, simulate the expected behavior

        validation = await pii_phi_manager.validate_job_compliance(
            job_id="gateway_test",
            job_inputs=sample_job_inputs["ssn_in_env"],
            namespace="production",
            user_id="user_123",
            strict_mode=True,
        )

        # Should return 422 status as per acceptance criteria
        assert validation["allowed"] is False
        assert validation["http_status"] == 422
        assert len(validation["violations"]) >= 1

        # Should have audit trail
        assert "audit_logged" in validation
        assert validation["audit_logged"] is True

    @pytest.mark.asyncio
    async def test_compliance_reporting(self, pii_phi_manager):
        """Test compliance reporting functionality"""

        # Generate some test data
        test_jobs = [
            ("clean_1", {"payload": {"msg": "hello"}}, False),
            ("pii_1", {"payload": {"email": "test@example.com"}}, True),
            ("clean_2", {"payload": {"count": 42}}, False),
            ("phi_1", {"input_data": b"Patient ID: 12345"}, True),
        ]

        for job_id, job_inputs, has_violations in test_jobs:
            await pii_phi_manager.scan_job_inputs_for_compliance(
                job_id=job_id, job_inputs=job_inputs, namespace="reporting_test"
            )

        # Generate compliance report
        report = await pii_phi_manager.generate_compliance_report(
            namespace="reporting_test", start_date=datetime.now() - timedelta(hours=1), end_date=datetime.now()
        )

        assert report["total_jobs_scanned"] >= 4
        assert report["violations_detected"] >= 2
        assert report["compliance_rate"] <= 0.5  # 50% or less due to violations
        assert "violation_breakdown" in report
        assert "recommendations" in report


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
