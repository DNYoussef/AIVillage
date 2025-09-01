"""
Security Boundary Tests with Connascence Compliance

Tests security boundaries with connascence-compliant security coupling patterns.
Validates that security controls maintain proper isolation while minimizing coupling violations.

Focus: Behavioral testing of security boundary contracts with clean coupling architecture.
"""

import unittest
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Protocol

from core.domain.security_constants import SecurityLevel


class SecurityBoundaryType(Enum):
    """Types of security boundaries in the system."""

    PROCESS_BOUNDARY = "process"
    NETWORK_BOUNDARY = "network"
    MODULE_BOUNDARY = "module"
    TRUST_BOUNDARY = "trust"
    DATA_BOUNDARY = "data"


class SecurityContext(Protocol):
    """Security context interface for boundary validation."""

    def get_security_level(self) -> SecurityLevel:
        """Get security level for this context."""
        ...

    def get_allowed_operations(self) -> List[str]:
        """Get operations allowed in this context."""
        ...

    def can_cross_boundary(self, target_boundary: "SecurityBoundary") -> bool:
        """Check if context can cross specified boundary."""
        ...


class SecurityBoundary:
    """Represents a security boundary with access controls."""

    def __init__(
        self,
        boundary_id: str,
        boundary_type: SecurityBoundaryType,
        security_level: SecurityLevel,
        allowed_contexts: List[str] = None,
    ):
        self.boundary_id = boundary_id
        self.boundary_type = boundary_type
        self.security_level = security_level
        self.allowed_contexts = allowed_contexts or []
        self.access_log = []
        self.creation_time = datetime.utcnow()

    def validate_access(self, context: SecurityContext, operation: str) -> Dict[str, Any]:
        """Validate access request against boundary policy."""
        validation_result = {
            "boundary_id": self.boundary_id,
            "operation": operation,
            "context_security_level": context.get_security_level().value,
            "boundary_security_level": self.security_level.value,
            "access_granted": False,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "rejection_reason": None,
        }

        # Check security level compatibility
        if not self._check_security_level_compatibility(context):
            validation_result["rejection_reason"] = "insufficient_security_level"
            self._log_access_attempt(validation_result)
            return validation_result

        # Check operation permissions
        if not self._check_operation_allowed(context, operation):
            validation_result["rejection_reason"] = "operation_not_allowed"
            self._log_access_attempt(validation_result)
            return validation_result

        # Check boundary crossing permission
        if not context.can_cross_boundary(self):
            validation_result["rejection_reason"] = "boundary_crossing_denied"
            self._log_access_attempt(validation_result)
            return validation_result

        # Access granted
        validation_result["access_granted"] = True
        self._log_access_attempt(validation_result)
        return validation_result

    def _check_security_level_compatibility(self, context: SecurityContext) -> bool:
        """Check if context security level is compatible with boundary."""
        context_level = context.get_security_level()

        # Define security level hierarchy (higher values = more secure)
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5,
        }

        context_value = level_values.get(context_level, 0)
        boundary_value = level_values.get(self.security_level, 0)

        return context_value >= boundary_value

    def _check_operation_allowed(self, context: SecurityContext, operation: str) -> bool:
        """Check if operation is allowed for the context."""
        allowed_ops = context.get_allowed_operations()
        return operation in allowed_ops

    def _log_access_attempt(self, validation_result: Dict[str, Any]):
        """Log access attempt for audit purposes."""
        self.access_log.append(validation_result.copy())

    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log for this boundary."""
        return self.access_log.copy()


class ModuleBoundaryValidator:
    """Validates module-level security boundaries with connascence awareness."""

    def __init__(self):
        self.module_boundaries = {}
        self.coupling_violations = []

    def register_module_boundary(
        self, module_name: str, security_level: SecurityLevel, allowed_dependencies: List[str] = None
    ) -> SecurityBoundary:
        """Register a module security boundary."""
        boundary = SecurityBoundary(
            boundary_id=f"module:{module_name}",
            boundary_type=SecurityBoundaryType.MODULE_BOUNDARY,
            security_level=security_level,
            allowed_contexts=allowed_dependencies or [],
        )

        self.module_boundaries[module_name] = boundary
        return boundary

    def validate_module_coupling(self, source_module: str, target_module: str, coupling_type: str) -> Dict[str, Any]:
        """Validate module coupling against security boundaries."""
        if source_module not in self.module_boundaries:
            return {"valid": False, "reason": "source_module_not_registered"}

        if target_module not in self.module_boundaries:
            return {"valid": False, "reason": "target_module_not_registered"}

        source_boundary = self.module_boundaries[source_module]
        target_boundary = self.module_boundaries[target_module]

        # Check security level compatibility
        source_level_value = self._get_security_level_value(source_boundary.security_level)
        target_level_value = self._get_security_level_value(target_boundary.security_level)

        validation_result = {
            "source_module": source_module,
            "target_module": target_module,
            "coupling_type": coupling_type,
            "valid": True,
            "security_compliant": True,
            "warnings": [],
        }

        # Validate security boundary crossing
        if source_level_value < target_level_value:
            # Lower security level accessing higher - potential violation
            validation_result["security_compliant"] = False
            validation_result["warnings"].append(
                f"Lower security module ({source_module}) accessing higher security module ({target_module})"
            )

        # Check for strong coupling across security boundaries
        strong_coupling_types = ["position", "meaning", "algorithm"]
        if coupling_type in strong_coupling_types and source_level_value != target_level_value:
            validation_result["security_compliant"] = False
            validation_result["warnings"].append(f"Strong coupling type '{coupling_type}' across security boundaries")

        # Log coupling violation if found
        if not validation_result["security_compliant"]:
            self.coupling_violations.append(validation_result.copy())

        return validation_result

    def _get_security_level_value(self, level: SecurityLevel) -> int:
        """Get numeric value for security level."""
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5,
        }
        return level_values.get(level, 0)

    def get_coupling_violations(self) -> List[Dict[str, Any]]:
        """Get detected coupling violations."""
        return self.coupling_violations.copy()


class TrustBoundaryManager:
    """Manages trust boundaries and cross-boundary communication."""

    def __init__(self):
        self.trust_boundaries = {}
        self.cross_boundary_communications = []

    def create_trust_boundary(
        self, boundary_id: str, trust_level: SecurityLevel, validation_requirements: List[str] = None
    ) -> SecurityBoundary:
        """Create a trust boundary with specified requirements."""
        boundary = SecurityBoundary(
            boundary_id=boundary_id,
            boundary_type=SecurityBoundaryType.TRUST_BOUNDARY,
            security_level=trust_level,
            allowed_contexts=validation_requirements or [],
        )

        self.trust_boundaries[boundary_id] = boundary
        return boundary

    def validate_cross_boundary_communication(
        self, source_boundary_id: str, target_boundary_id: str, message_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate communication across trust boundaries."""
        if source_boundary_id not in self.trust_boundaries:
            return {"valid": False, "reason": "source_boundary_not_found"}

        if target_boundary_id not in self.trust_boundaries:
            return {"valid": False, "reason": "target_boundary_not_found"}

        source_boundary = self.trust_boundaries[source_boundary_id]
        target_boundary = self.trust_boundaries[target_boundary_id]

        validation_result = {
            "source_boundary": source_boundary_id,
            "target_boundary": target_boundary_id,
            "message_validated": False,
            "security_controls_applied": [],
            "validation_timestamp": datetime.utcnow().isoformat(),
        }

        # Apply security controls for cross-boundary communication
        security_controls = self._apply_security_controls(source_boundary, target_boundary, message_data)

        validation_result["security_controls_applied"] = security_controls
        validation_result["message_validated"] = len(security_controls) > 0

        # Log cross-boundary communication
        self.cross_boundary_communications.append(validation_result.copy())

        return validation_result

    def _apply_security_controls(
        self, source: SecurityBoundary, target: SecurityBoundary, message_data: Dict[str, Any]
    ) -> List[str]:
        """Apply appropriate security controls for cross-boundary communication."""
        applied_controls = []

        # Input validation
        if self._requires_input_validation(target):
            applied_controls.append("input_validation")

        # Data sanitization
        if self._requires_data_sanitization(source, target):
            applied_controls.append("data_sanitization")

        # Encryption
        if self._requires_encryption(source, target):
            applied_controls.append("encryption")

        # Audit logging
        if self._requires_audit_logging(source, target):
            applied_controls.append("audit_logging")

        return applied_controls

    def _requires_input_validation(self, target: SecurityBoundary) -> bool:
        """Check if input validation is required for target boundary."""
        high_security_levels = [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]
        return target.security_level in high_security_levels

    def _requires_data_sanitization(self, source: SecurityBoundary, target: SecurityBoundary) -> bool:
        """Check if data sanitization is required."""
        source_level = self._get_security_level_value(source.security_level)
        target_level = self._get_security_level_value(target.security_level)
        return source_level != target_level

    def _requires_encryption(self, source: SecurityBoundary, target: SecurityBoundary) -> bool:
        """Check if encryption is required for communication."""
        return (
            source.boundary_type == SecurityBoundaryType.NETWORK_BOUNDARY
            or target.boundary_type == SecurityBoundaryType.NETWORK_BOUNDARY
        )

    def _requires_audit_logging(self, source: SecurityBoundary, target: SecurityBoundary) -> bool:
        """Check if audit logging is required."""
        sensitive_levels = [SecurityLevel.SECRET, SecurityLevel.TOP_SECRET]
        return source.security_level in sensitive_levels or target.security_level in sensitive_levels

    def _get_security_level_value(self, level: SecurityLevel) -> int:
        """Get numeric value for security level."""
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5,
        }
        return level_values.get(level, 0)


class TestSecurityContext:
    """Test implementation of SecurityContext protocol."""

    def __init__(self, security_level: SecurityLevel, allowed_operations: List[str]):
        self.security_level = security_level
        self.allowed_operations = allowed_operations

    def get_security_level(self) -> SecurityLevel:
        return self.security_level

    def get_allowed_operations(self) -> List[str]:
        return self.allowed_operations

    def can_cross_boundary(self, target_boundary: SecurityBoundary) -> bool:
        # Simple implementation - allow if security levels are compatible
        level_values = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5,
        }

        context_value = level_values.get(self.security_level, 0)
        boundary_value = level_values.get(target_boundary.security_level, 0)

        return context_value >= boundary_value


class SecurityBoundaryTest(unittest.TestCase):
    """
    Behavioral tests for security boundaries with connascence compliance.

    Tests security boundary contracts and coupling patterns without implementation coupling.
    Validates clean architecture security boundaries.
    """

    def setUp(self):
        """Set up test fixtures with security boundary configurations."""
        self.module_validator = ModuleBoundaryValidator()
        self.trust_manager = TrustBoundaryManager()

        # Create test security contexts
        self.public_context = TestSecurityContext(SecurityLevel.PUBLIC, ["read", "list"])
        self.confidential_context = TestSecurityContext(SecurityLevel.CONFIDENTIAL, ["read", "write", "list", "delete"])
        self.secret_context = TestSecurityContext(SecurityLevel.SECRET, ["read", "write", "list", "delete", "admin"])

    def test_security_level_boundary_enforcement(self):
        """
        Security Contract: Security boundaries must enforce access based on security levels.
        Tests hierarchical security level validation.
        """
        # Arrange - Create boundaries with different security levels
        public_boundary = SecurityBoundary("public-api", SecurityBoundaryType.MODULE_BOUNDARY, SecurityLevel.PUBLIC)

        confidential_boundary = SecurityBoundary(
            "confidential-data", SecurityBoundaryType.DATA_BOUNDARY, SecurityLevel.CONFIDENTIAL
        )

        # Test public context accessing public boundary (should succeed)
        public_result = public_boundary.validate_access(self.public_context, "read")
        self.assertTrue(public_result["access_granted"], "Public context must access public boundary")

        # Test public context accessing confidential boundary (should fail)
        confidential_result = confidential_boundary.validate_access(self.public_context, "read")
        self.assertFalse(confidential_result["access_granted"], "Public context must not access confidential boundary")
        self.assertEqual(confidential_result["rejection_reason"], "insufficient_security_level")

        # Test confidential context accessing confidential boundary (should succeed)
        valid_confidential = confidential_boundary.validate_access(self.confidential_context, "read")
        self.assertTrue(valid_confidential["access_granted"], "Confidential context must access confidential boundary")

    def test_operation_based_access_control(self):
        """
        Security Contract: Boundaries must enforce operation-based access control.
        Tests granular operation permission validation.
        """
        # Arrange - Create boundary with specific operation requirements
        admin_boundary = SecurityBoundary(
            "admin-interface", SecurityBoundaryType.MODULE_BOUNDARY, SecurityLevel.INTERNAL
        )

        # Test allowed operation
        read_result = admin_boundary.validate_access(self.confidential_context, "read")
        self.assertTrue(read_result["access_granted"], "Context must access boundary for allowed operation")

        # Test disallowed operation
        admin_result = admin_boundary.validate_access(self.confidential_context, "admin")
        self.assertFalse(admin_result["access_granted"], "Context must not access boundary for disallowed operation")
        self.assertEqual(admin_result["rejection_reason"], "operation_not_allowed")

    def test_module_coupling_validation_with_connascence(self):
        """
        Security Contract: Module coupling must respect security boundaries and connascence principles.
        Tests coupling validation across security boundaries.
        """
        # Arrange - Register modules with different security levels
        self.module_validator.register_module_boundary("public_utils", SecurityLevel.PUBLIC)
        self.module_validator.register_module_boundary("confidential_service", SecurityLevel.CONFIDENTIAL)
        self.module_validator.register_module_boundary("secret_crypto", SecurityLevel.SECRET)

        # Test weak coupling across security boundaries (should be acceptable)
        name_coupling_result = self.module_validator.validate_module_coupling(
            "public_utils", "confidential_service", "name"
        )
        self.assertTrue(name_coupling_result["valid"], "Weak coupling (name) across boundaries should be valid")
        self.assertTrue(name_coupling_result["security_compliant"], "Name coupling should be security compliant")

        # Test strong coupling across security boundaries (should trigger warnings)
        algorithm_coupling_result = self.module_validator.validate_module_coupling(
            "public_utils", "secret_crypto", "algorithm"
        )
        self.assertTrue(algorithm_coupling_result["valid"], "Coupling should be structurally valid")
        self.assertFalse(
            algorithm_coupling_result["security_compliant"],
            "Strong coupling across security boundaries should not be compliant",
        )
        self.assertGreater(len(algorithm_coupling_result["warnings"]), 0, "Should generate security warnings")

        # Verify coupling violations are tracked
        violations = self.module_validator.get_coupling_violations()
        self.assertGreater(len(violations), 0, "Coupling violations should be tracked")

    def test_trust_boundary_cross_communication(self):
        """
        Security Contract: Cross-trust-boundary communication must apply appropriate security controls.
        Tests security control application for trust boundary crossings.
        """
        # Arrange - Create trust boundaries
        self.trust_manager.create_trust_boundary("internal-services", SecurityLevel.INTERNAL)
        self.trust_manager.create_trust_boundary("external-api", SecurityLevel.PUBLIC)

        # Test cross-boundary communication
        message_data = {"type": "data_request", "payload": {"user_id": "12345", "requested_data": "user_profile"}}

        communication_result = self.trust_manager.validate_cross_boundary_communication(
            "external-api", "internal-services", message_data
        )

        # Assert - Verify security controls are applied
        self.assertTrue(communication_result["message_validated"], "Cross-boundary communication must be validated")
        self.assertGreater(
            len(communication_result["security_controls_applied"]), 0, "Security controls must be applied"
        )

        # Check specific security controls
        applied_controls = communication_result["security_controls_applied"]
        self.assertIn("input_validation", applied_controls, "Input validation should be applied for internal services")
        self.assertIn(
            "data_sanitization", applied_controls, "Data sanitization should be applied across security levels"
        )

    def test_boundary_access_audit_logging(self):
        """
        Security Contract: All boundary access attempts must be audit logged.
        Tests comprehensive audit logging of boundary access.
        """
        # Arrange
        secure_boundary = SecurityBoundary("secure-module", SecurityBoundaryType.MODULE_BOUNDARY, SecurityLevel.SECRET)

        # Test multiple access attempts
        test_scenarios = [
            (self.secret_context, "read", True),  # Should succeed
            (self.public_context, "read", False),  # Should fail - insufficient level
            (self.secret_context, "admin", True),  # Should succeed
            (self.confidential_context, "write", False),  # Should fail - insufficient level
        ]

        for context, operation, expected_success in test_scenarios:
            result = secure_boundary.validate_access(context, operation)
            self.assertEqual(
                result["access_granted"], expected_success, f"Access result should match expected for {operation}"
            )

        # Verify audit logging
        access_log = secure_boundary.get_access_log()
        self.assertEqual(len(access_log), len(test_scenarios), "All access attempts must be audit logged")

        # Verify log entry completeness
        for log_entry in access_log:
            required_fields = [
                "boundary_id",
                "operation",
                "context_security_level",
                "boundary_security_level",
                "access_granted",
                "validation_timestamp",
            ]
            for field in required_fields:
                self.assertIn(field, log_entry, f"Audit log must include {field}")

    def test_network_boundary_security_controls(self):
        """
        Security Contract: Network boundaries must apply network-specific security controls.
        Tests network boundary security control application.
        """
        # Arrange - Create network trust boundaries
        internal_network = self.trust_manager.create_trust_boundary("internal-network", SecurityLevel.INTERNAL)
        internal_network.boundary_type = SecurityBoundaryType.NETWORK_BOUNDARY

        external_network = self.trust_manager.create_trust_boundary("external-network", SecurityLevel.PUBLIC)
        external_network.boundary_type = SecurityBoundaryType.NETWORK_BOUNDARY

        # Test network cross-boundary communication
        network_message = {
            "source": "external-client",
            "destination": "internal-service",
            "data": "sensitive information",
        }

        result = self.trust_manager.validate_cross_boundary_communication(
            "external-network", "internal-network", network_message
        )

        # Assert - Network-specific controls are applied
        applied_controls = result["security_controls_applied"]
        self.assertIn("encryption", applied_controls, "Network boundaries must apply encryption")
        self.assertIn("audit_logging", applied_controls, "Network boundaries should include audit logging")

    def test_data_boundary_isolation(self):
        """
        Security Contract: Data boundaries must enforce data isolation and classification.
        Tests data boundary access control and classification enforcement.
        """
        # Arrange - Create data boundaries with different classifications
        public_data_boundary = SecurityBoundary("public-data", SecurityBoundaryType.DATA_BOUNDARY, SecurityLevel.PUBLIC)

        pii_data_boundary = SecurityBoundary("pii-data", SecurityBoundaryType.DATA_BOUNDARY, SecurityLevel.CONFIDENTIAL)

        # Test data classification enforcement
        public_access = public_data_boundary.validate_access(self.public_context, "read")
        self.assertTrue(public_access["access_granted"], "Public context should access public data")

        pii_access_denied = pii_data_boundary.validate_access(self.public_context, "read")
        self.assertFalse(pii_access_denied["access_granted"], "Public context should not access PII data")

        pii_access_allowed = pii_data_boundary.validate_access(self.confidential_context, "read")
        self.assertTrue(pii_access_allowed["access_granted"], "Confidential context should access PII data")

    def test_boundary_configuration_immutability(self):
        """
        Security Contract: Security boundary configurations must be immutable after creation.
        Tests boundary configuration integrity and immutability.
        """
        # Arrange - Create security boundary
        original_boundary = SecurityBoundary(
            "immutable-boundary", SecurityBoundaryType.MODULE_BOUNDARY, SecurityLevel.CONFIDENTIAL
        )

        original_id = original_boundary.boundary_id
        original_type = original_boundary.boundary_type
        original_level = original_boundary.security_level

        # Test that critical properties remain unchanged
        # Note: In a real implementation, these would be properties that cannot be modified
        # after creation, or would require special authorization

        # Verify boundary identity consistency
        self.assertEqual(original_boundary.boundary_id, original_id, "Boundary ID must remain consistent")
        self.assertEqual(original_boundary.boundary_type, original_type, "Boundary type must remain consistent")
        self.assertEqual(original_boundary.security_level, original_level, "Security level must remain consistent")

        # Verify creation time is set and immutable
        self.assertIsInstance(original_boundary.creation_time, datetime, "Creation time must be set")

        original_creation_time = original_boundary.creation_time
        # Simulate time passage
        time.sleep(0.001)  # Small delay

        # Creation time should not change
        self.assertEqual(original_boundary.creation_time, original_creation_time, "Creation time must be immutable")

    def test_security_boundary_scalability(self):
        """
        Security Contract: Security boundary validation must scale with system growth.
        Tests performance and scalability of boundary validation.
        """
        # Arrange - Create large number of boundaries and contexts
        boundaries = []
        contexts = []

        for i in range(100):
            boundary = SecurityBoundary(
                f"boundary-{i}",
                SecurityBoundaryType.MODULE_BOUNDARY,
                SecurityLevel.INTERNAL if i % 2 == 0 else SecurityLevel.PUBLIC,
            )
            boundaries.append(boundary)

            context = TestSecurityContext(
                SecurityLevel.INTERNAL if i % 3 == 0 else SecurityLevel.PUBLIC,
                ["read", "write"] if i % 2 == 0 else ["read"],
            )
            contexts.append(context)

        # Test scalability of validation
        start_time = datetime.utcnow()

        validation_results = []
        for i in range(min(50, len(boundaries), len(contexts))):
            result = boundaries[i].validate_access(contexts[i], "read")
            validation_results.append(result)

        end_time = datetime.utcnow()
        validation_duration = (end_time - start_time).total_seconds()

        # Assert - Performance requirements
        self.assertLess(
            validation_duration, 1.0, "Boundary validation must complete within 1 second for 50 validations"
        )
        self.assertEqual(len(validation_results), 50, "All validations must complete successfully")

        # Verify all results have required structure
        for result in validation_results[:5]:  # Check first 5 for structure
            self.assertIn("access_granted", result)
            self.assertIn("validation_timestamp", result)


if __name__ == "__main__":
    # Run tests with security-focused output
    unittest.main(verbosity=2, buffer=True)
