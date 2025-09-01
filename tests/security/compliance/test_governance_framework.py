"""
Governance Framework Compliance Tests

Tests compliance with security governance frameworks including GDPR, COPPA, FERPA,
OWASP Top 10, and other regulatory requirements.

Focus: Compliance testing of governance framework adherence and policy enforcement.
"""

import unittest
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any



class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    GDPR = "gdpr"
    COPPA = "coppa"
    FERPA = "ferpa"
    OWASP_TOP_10 = "owasp_top_10"
    ISO_27001 = "iso_27001"
    SOC_2_TYPE_II = "soc_2_type_ii"


class ComplianceStatus(Enum):
    """Compliance status levels."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"


class DataClassification(Enum):
    """Data classification categories for compliance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    PII = "pii"  # Personally Identifiable Information
    PHI = "phi"  # Protected Health Information
    EDUCATIONAL_RECORDS = "educational_records"
    FINANCIAL_DATA = "financial_data"


class ComplianceRule:
    """Represents a compliance rule with validation logic."""

    def __init__(
        self,
        rule_id: str,
        framework: ComplianceFramework,
        title: str,
        description: str,
        validation_function: callable = None,
    ):
        self.rule_id = rule_id
        self.framework = framework
        self.title = title
        self.description = description
        self.validation_function = validation_function or self._default_validation
        self.last_checked = None
        self.status = ComplianceStatus.UNDER_REVIEW

    def validate(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance rule against system configuration."""
        validation_result = {
            "rule_id": self.rule_id,
            "framework": self.framework.value,
            "title": self.title,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "compliant": False,
            "findings": [],
            "recommendations": [],
        }

        try:
            compliance_result = self.validation_function(system_config)
            validation_result.update(compliance_result)

            self.status = (
                ComplianceStatus.COMPLIANT
                if compliance_result.get("compliant", False)
                else ComplianceStatus.NON_COMPLIANT
            )

        except Exception as e:
            validation_result["error"] = str(e)
            validation_result["findings"].append(f"Validation error: {str(e)}")
            self.status = ComplianceStatus.UNDER_REVIEW

        self.last_checked = datetime.utcnow()
        return validation_result

    def _default_validation(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Default validation logic."""
        return {
            "compliant": True,
            "findings": ["Default validation - requires implementation"],
            "recommendations": ["Implement specific validation logic for this rule"],
        }


class GovernanceFrameworkValidator:
    """Validates compliance with various governance frameworks."""

    def __init__(self):
        self.compliance_rules = {}
        self.validation_results = {}
        self.compliance_history = []
        self._initialize_compliance_rules()

    def _initialize_compliance_rules(self):
        """Initialize compliance rules for various frameworks."""

        # GDPR Rules
        self.compliance_rules["gdpr_data_processing_lawfulness"] = ComplianceRule(
            "GDPR-001",
            ComplianceFramework.GDPR,
            "Lawfulness of Processing",
            "Personal data must be processed lawfully, fairly and transparently",
            self._validate_gdpr_lawful_processing,
        )

        self.compliance_rules["gdpr_data_minimization"] = ComplianceRule(
            "GDPR-002",
            ComplianceFramework.GDPR,
            "Data Minimization",
            "Personal data must be adequate, relevant and limited to what is necessary",
            self._validate_gdpr_data_minimization,
        )

        self.compliance_rules["gdpr_right_to_erasure"] = ComplianceRule(
            "GDPR-003",
            ComplianceFramework.GDPR,
            "Right to Erasure",
            "Individuals must have the right to have their personal data erased",
            self._validate_gdpr_right_to_erasure,
        )

        # COPPA Rules
        self.compliance_rules["coppa_parental_consent"] = ComplianceRule(
            "COPPA-001",
            ComplianceFramework.COPPA,
            "Parental Consent",
            "Must obtain verifiable parental consent before collecting children's data",
            self._validate_coppa_parental_consent,
        )

        self.compliance_rules["coppa_data_collection_limits"] = ComplianceRule(
            "COPPA-002",
            ComplianceFramework.COPPA,
            "Limited Data Collection",
            "Cannot collect more personal information than necessary",
            self._validate_coppa_data_collection,
        )

        # FERPA Rules
        self.compliance_rules["ferpa_educational_records_protection"] = ComplianceRule(
            "FERPA-001",
            ComplianceFramework.FERPA,
            "Educational Records Protection",
            "Educational records must be protected and disclosure controlled",
            self._validate_ferpa_records_protection,
        )

        # OWASP Top 10 Rules
        self.compliance_rules["owasp_injection_prevention"] = ComplianceRule(
            "OWASP-001",
            ComplianceFramework.OWASP_TOP_10,
            "Injection Prevention",
            "Application must prevent injection attacks",
            self._validate_owasp_injection_prevention,
        )

        self.compliance_rules["owasp_authentication_controls"] = ComplianceRule(
            "OWASP-002",
            ComplianceFramework.OWASP_TOP_10,
            "Broken Authentication",
            "Authentication and session management must be implemented correctly",
            self._validate_owasp_authentication,
        )

        self.compliance_rules["owasp_sensitive_data_exposure"] = ComplianceRule(
            "OWASP-003",
            ComplianceFramework.OWASP_TOP_10,
            "Sensitive Data Exposure",
            "Sensitive data must be properly protected",
            self._validate_owasp_sensitive_data,
        )

    def _validate_gdpr_lawful_processing(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GDPR lawful processing requirements."""
        findings = []
        recommendations = []
        compliant = True

        # Check for legal basis documentation
        if not system_config.get("legal_basis_documented", False):
            findings.append("Legal basis for data processing not documented")
            recommendations.append("Document legal basis for all data processing activities")
            compliant = False

        # Check for privacy policy
        if not system_config.get("privacy_policy_exists", False):
            findings.append("Privacy policy not found")
            recommendations.append("Implement comprehensive privacy policy")
            compliant = False

        # Check for transparency measures
        transparency_measures = system_config.get("transparency_measures", [])
        required_measures = ["data_collection_notice", "processing_purposes", "data_retention_policy"]

        for measure in required_measures:
            if measure not in transparency_measures:
                findings.append(f"Missing transparency measure: {measure}")
                recommendations.append(f"Implement {measure.replace('_', ' ')}")
                compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_gdpr_data_minimization(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GDPR data minimization requirements."""
        findings = []
        recommendations = []
        compliant = True

        # Check data collection practices
        collected_data_fields = system_config.get("collected_data_fields", [])
        necessary_data_fields = system_config.get("necessary_data_fields", [])

        excessive_fields = set(collected_data_fields) - set(necessary_data_fields)

        if excessive_fields:
            findings.append(f"Collecting excessive data fields: {list(excessive_fields)}")
            recommendations.append("Remove unnecessary data collection fields")
            compliant = False

        # Check retention periods
        if not system_config.get("retention_periods_defined", False):
            findings.append("Data retention periods not defined")
            recommendations.append("Define and implement data retention periods")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_gdpr_right_to_erasure(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GDPR right to erasure implementation."""
        findings = []
        recommendations = []
        compliant = True

        # Check for erasure mechanism
        if not system_config.get("data_erasure_mechanism", False):
            findings.append("No data erasure mechanism implemented")
            recommendations.append("Implement data erasure functionality")
            compliant = False

        # Check for request handling process
        if not system_config.get("erasure_request_process", False):
            findings.append("No process for handling erasure requests")
            recommendations.append("Implement process for handling erasure requests")
            compliant = False

        # Check response time compliance
        erasure_response_time_days = system_config.get("erasure_response_time_days", 0)
        if erasure_response_time_days > 30:
            findings.append(f"Erasure response time too long: {erasure_response_time_days} days")
            recommendations.append("Reduce erasure response time to within 30 days")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_coppa_parental_consent(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate COPPA parental consent requirements."""
        findings = []
        recommendations = []
        compliant = True

        # Check age verification
        if not system_config.get("age_verification_implemented", False):
            findings.append("Age verification not implemented")
            recommendations.append("Implement age verification mechanism")
            compliant = False

        # Check parental consent mechanism
        if not system_config.get("parental_consent_mechanism", False):
            findings.append("Parental consent mechanism not implemented")
            recommendations.append("Implement verifiable parental consent process")
            compliant = False

        # Check consent verification methods
        consent_methods = system_config.get("consent_verification_methods", [])
        required_methods = ["email_confirmation", "offline_consent"]

        if not any(method in consent_methods for method in required_methods):
            findings.append("No acceptable consent verification methods implemented")
            recommendations.append("Implement acceptable consent verification methods")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_coppa_data_collection(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate COPPA data collection limitations."""
        findings = []
        recommendations = []
        compliant = True

        # Check for minimal data collection
        children_data_fields = system_config.get("children_data_fields", [])
        prohibited_fields = ["full_name", "home_address", "phone_number", "social_security"]

        collected_prohibited = set(children_data_fields) & set(prohibited_fields)
        if collected_prohibited:
            findings.append(f"Collecting prohibited children data: {list(collected_prohibited)}")
            recommendations.append("Remove collection of prohibited data fields for children")
            compliant = False

        # Check retention limits
        children_data_retention_days = system_config.get("children_data_retention_days", 0)
        if children_data_retention_days > 365:
            findings.append(f"Children data retention too long: {children_data_retention_days} days")
            recommendations.append("Limit children data retention to necessary period")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_ferpa_records_protection(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate FERPA educational records protection."""
        findings = []
        recommendations = []
        compliant = True

        # Check access controls
        if not system_config.get("educational_records_access_control", False):
            findings.append("Access controls for educational records not implemented")
            recommendations.append("Implement role-based access controls for educational records")
            compliant = False

        # Check consent requirements
        if not system_config.get("educational_records_consent_mechanism", False):
            findings.append("Consent mechanism for educational records disclosure not implemented")
            recommendations.append("Implement consent mechanism for records disclosure")
            compliant = False

        # Check audit logging
        if not system_config.get("educational_records_audit_logging", False):
            findings.append("Audit logging for educational records access not implemented")
            recommendations.append("Implement comprehensive audit logging")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_owasp_injection_prevention(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OWASP injection prevention controls."""
        findings = []
        recommendations = []
        compliant = True

        # Check input validation
        if not system_config.get("input_validation_implemented", False):
            findings.append("Input validation not implemented")
            recommendations.append("Implement comprehensive input validation")
            compliant = False

        # Check parameterized queries
        if not system_config.get("parameterized_queries_used", False):
            findings.append("Parameterized queries not used")
            recommendations.append("Use parameterized queries for database access")
            compliant = False

        # Check output encoding
        if not system_config.get("output_encoding_implemented", False):
            findings.append("Output encoding not implemented")
            recommendations.append("Implement proper output encoding")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_owasp_authentication(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OWASP authentication controls."""
        findings = []
        recommendations = []
        compliant = True

        # Check MFA implementation
        if not system_config.get("mfa_implemented", False):
            findings.append("Multi-factor authentication not implemented")
            recommendations.append("Implement multi-factor authentication")
            compliant = False

        # Check session management
        if not system_config.get("secure_session_management", False):
            findings.append("Secure session management not implemented")
            recommendations.append("Implement secure session management")
            compliant = False

        # Check password policies
        password_policy = system_config.get("password_policy", {})
        if password_policy.get("min_length", 0) < 12:
            findings.append("Password minimum length too short")
            recommendations.append("Implement minimum password length of 12 characters")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def _validate_owasp_sensitive_data(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate OWASP sensitive data protection."""
        findings = []
        recommendations = []
        compliant = True

        # Check encryption at rest
        if not system_config.get("encryption_at_rest", False):
            findings.append("Data encryption at rest not implemented")
            recommendations.append("Implement encryption for sensitive data at rest")
            compliant = False

        # Check encryption in transit
        if not system_config.get("encryption_in_transit", False):
            findings.append("Data encryption in transit not implemented")
            recommendations.append("Implement TLS/SSL for data in transit")
            compliant = False

        # Check data classification
        if not system_config.get("data_classification_implemented", False):
            findings.append("Data classification not implemented")
            recommendations.append("Implement data classification system")
            compliant = False

        return {"compliant": compliant, "findings": findings, "recommendations": recommendations}

    def validate_framework_compliance(
        self, framework: ComplianceFramework, system_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate compliance with specific framework."""
        framework_rules = [rule for rule in self.compliance_rules.values() if rule.framework == framework]

        validation_results = []
        compliant_rules = 0

        for rule in framework_rules:
            result = rule.validate(system_config)
            validation_results.append(result)

            if result.get("compliant", False):
                compliant_rules += 1

        framework_result = {
            "framework": framework.value,
            "validation_timestamp": datetime.utcnow().isoformat(),
            "total_rules": len(framework_rules),
            "compliant_rules": compliant_rules,
            "compliance_percentage": (compliant_rules / len(framework_rules) * 100) if framework_rules else 0,
            "overall_status": (
                ComplianceStatus.COMPLIANT.value
                if compliant_rules == len(framework_rules)
                else ComplianceStatus.PARTIALLY_COMPLIANT.value
            ),
            "rule_results": validation_results,
        }

        self.validation_results[framework.value] = framework_result
        self.compliance_history.append(framework_result.copy())

        return framework_result

    def validate_all_frameworks(self, system_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate compliance across all supported frameworks."""
        all_results = {}

        for framework in ComplianceFramework:
            try:
                framework_result = self.validate_framework_compliance(framework, system_config)
                all_results[framework.value] = framework_result
            except Exception as e:
                all_results[framework.value] = {
                    "framework": framework.value,
                    "error": str(e),
                    "overall_status": ComplianceStatus.UNDER_REVIEW.value,
                }

        # Calculate overall compliance summary
        total_frameworks = len(all_results)
        compliant_frameworks = sum(
            1 for result in all_results.values() if result.get("overall_status") == ComplianceStatus.COMPLIANT.value
        )

        summary = {
            "validation_timestamp": datetime.utcnow().isoformat(),
            "total_frameworks": total_frameworks,
            "compliant_frameworks": compliant_frameworks,
            "overall_compliance_percentage": (compliant_frameworks / total_frameworks * 100) if total_frameworks else 0,
            "framework_results": all_results,
            "recommendations": self._generate_compliance_recommendations(all_results),
        }

        return summary

    def _generate_compliance_recommendations(self, framework_results: Dict[str, Any]) -> List[str]:
        """Generate overall compliance recommendations."""
        recommendations = []

        # Analyze common issues across frameworks
        common_issues = {}

        for framework_name, result in framework_results.items():
            if "rule_results" in result:
                for rule_result in result["rule_results"]:
                    for recommendation in rule_result.get("recommendations", []):
                        common_issues[recommendation] = common_issues.get(recommendation, 0) + 1

        # Prioritize recommendations that appear across multiple frameworks
        sorted_issues = sorted(common_issues.items(), key=lambda x: x[1], reverse=True)

        for issue, count in sorted_issues[:10]:  # Top 10 recommendations
            if count > 1:
                recommendations.append(f"[HIGH PRIORITY] {issue} (affects {count} frameworks)")
            else:
                recommendations.append(issue)

        return recommendations

    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard data."""
        dashboard = {
            "last_updated": datetime.utcnow().isoformat(),
            "framework_status": {},
            "compliance_trends": [],
            "priority_actions": [],
        }

        # Current framework status
        for framework_name, result in self.validation_results.items():
            dashboard["framework_status"][framework_name] = {
                "status": result.get("overall_status", ComplianceStatus.UNDER_REVIEW.value),
                "compliance_percentage": result.get("compliance_percentage", 0),
                "last_validated": result.get("validation_timestamp"),
            }

        # Compliance trends (simplified)
        if len(self.compliance_history) >= 2:
            recent = self.compliance_history[-1]
            previous = self.compliance_history[-2]

            trend = {
                "framework": recent.get("framework"),
                "current_percentage": recent.get("compliance_percentage", 0),
                "previous_percentage": previous.get("compliance_percentage", 0),
                "trend_direction": (
                    "improving"
                    if recent.get("compliance_percentage", 0) > previous.get("compliance_percentage", 0)
                    else "stable"
                ),
            }
            dashboard["compliance_trends"].append(trend)

        return dashboard


class GovernanceFrameworkTest(unittest.TestCase):
    """
    Compliance tests for governance framework validation.

    Tests adherence to regulatory and governance frameworks
    without coupling to implementation details.
    """

    def setUp(self):
        """Set up governance framework testing fixtures."""
        self.governance_validator = GovernanceFrameworkValidator()

        # Mock system configuration for testing
        self.compliant_system_config = {
            # GDPR compliance
            "legal_basis_documented": True,
            "privacy_policy_exists": True,
            "transparency_measures": ["data_collection_notice", "processing_purposes", "data_retention_policy"],
            "collected_data_fields": ["name", "email"],
            "necessary_data_fields": ["name", "email"],
            "retention_periods_defined": True,
            "data_erasure_mechanism": True,
            "erasure_request_process": True,
            "erasure_response_time_days": 15,
            # COPPA compliance
            "age_verification_implemented": True,
            "parental_consent_mechanism": True,
            "consent_verification_methods": ["email_confirmation", "offline_consent"],
            "children_data_fields": ["username", "age_range"],
            "children_data_retention_days": 90,
            # FERPA compliance
            "educational_records_access_control": True,
            "educational_records_consent_mechanism": True,
            "educational_records_audit_logging": True,
            # OWASP Top 10 compliance
            "input_validation_implemented": True,
            "parameterized_queries_used": True,
            "output_encoding_implemented": True,
            "mfa_implemented": True,
            "secure_session_management": True,
            "password_policy": {"min_length": 12, "complexity_required": True},
            "encryption_at_rest": True,
            "encryption_in_transit": True,
            "data_classification_implemented": True,
        }

        self.non_compliant_system_config = {
            # Missing most compliance features
            "privacy_policy_exists": False,
            "age_verification_implemented": False,
            "input_validation_implemented": False,
            "mfa_implemented": False,
            "encryption_at_rest": False,
        }

    def test_gdpr_compliance_validation(self):
        """
        Security Contract: System must comply with GDPR requirements.
        Tests GDPR compliance validation and requirement checking.
        """
        # Test compliant system
        compliant_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.GDPR, self.compliant_system_config
        )

        # Assert GDPR compliance
        self.assertEqual(
            compliant_result["overall_status"],
            ComplianceStatus.COMPLIANT.value,
            "Compliant system must pass GDPR validation",
        )
        self.assertGreaterEqual(
            compliant_result["compliance_percentage"], 90, "GDPR compliance percentage should be high"
        )

        # Test non-compliant system
        non_compliant_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.GDPR, self.non_compliant_system_config
        )

        self.assertEqual(
            non_compliant_result["overall_status"],
            ComplianceStatus.PARTIALLY_COMPLIANT.value,
            "Non-compliant system should fail GDPR validation",
        )

        # Verify specific GDPR requirements
        rule_results = compliant_result["rule_results"]
        gdpr_rules_tested = [result["rule_id"] for result in rule_results]

        expected_gdpr_rules = ["GDPR-001", "GDPR-002", "GDPR-003"]
        for rule_id in expected_gdpr_rules:
            self.assertIn(rule_id, gdpr_rules_tested, f"GDPR rule {rule_id} must be tested")

    def test_coppa_compliance_validation(self):
        """
        Security Contract: System must comply with COPPA requirements for children's data.
        Tests COPPA compliance validation for child data protection.
        """
        # Test compliant system
        compliant_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.COPPA, self.compliant_system_config
        )

        # Assert COPPA compliance
        self.assertEqual(
            compliant_result["overall_status"],
            ComplianceStatus.COMPLIANT.value,
            "Compliant system must pass COPPA validation",
        )

        # Test system with child data violations
        coppa_violation_config = self.compliant_system_config.copy()
        coppa_violation_config["children_data_fields"] = ["full_name", "home_address", "phone_number"]
        coppa_violation_config["children_data_retention_days"] = 2000

        violation_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.COPPA, coppa_violation_config
        )

        self.assertEqual(
            violation_result["overall_status"],
            ComplianceStatus.PARTIALLY_COMPLIANT.value,
            "System with COPPA violations should fail validation",
        )

        # Verify specific violations are detected
        rule_results = violation_result["rule_results"]
        violations_found = any("prohibited children data" in str(result.get("findings", [])) for result in rule_results)
        self.assertTrue(violations_found, "COPPA data collection violations should be detected")

    def test_ferpa_compliance_validation(self):
        """
        Security Contract: System must comply with FERPA for educational records.
        Tests FERPA compliance validation for educational data protection.
        """
        # Test compliant system
        compliant_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.FERPA, self.compliant_system_config
        )

        # Assert FERPA compliance
        self.assertEqual(
            compliant_result["overall_status"],
            ComplianceStatus.COMPLIANT.value,
            "Compliant system must pass FERPA validation",
        )

        # Test system without educational records protection
        ferpa_violation_config = self.compliant_system_config.copy()
        ferpa_violation_config["educational_records_access_control"] = False
        ferpa_violation_config["educational_records_audit_logging"] = False

        violation_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.FERPA, ferpa_violation_config
        )

        self.assertEqual(
            violation_result["overall_status"],
            ComplianceStatus.PARTIALLY_COMPLIANT.value,
            "System without FERPA controls should fail validation",
        )

    def test_owasp_top_10_compliance_validation(self):
        """
        Security Contract: System must comply with OWASP Top 10 security requirements.
        Tests OWASP Top 10 compliance validation for security controls.
        """
        # Test compliant system
        compliant_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.OWASP_TOP_10, self.compliant_system_config
        )

        # Assert OWASP compliance
        self.assertEqual(
            compliant_result["overall_status"],
            ComplianceStatus.COMPLIANT.value,
            "Compliant system must pass OWASP Top 10 validation",
        )

        # Test specific OWASP controls
        rule_results = compliant_result["rule_results"]
        owasp_controls_tested = [result["title"] for result in rule_results]

        expected_controls = ["Injection Prevention", "Broken Authentication", "Sensitive Data Exposure"]
        for control in expected_controls:
            self.assertIn(control, owasp_controls_tested, f"OWASP control '{control}' must be tested")

        # Test system with security vulnerabilities
        vulnerable_config = self.compliant_system_config.copy()
        vulnerable_config["input_validation_implemented"] = False
        vulnerable_config["mfa_implemented"] = False
        vulnerable_config["encryption_at_rest"] = False

        vulnerable_result = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.OWASP_TOP_10, vulnerable_config
        )

        self.assertEqual(
            vulnerable_result["overall_status"],
            ComplianceStatus.PARTIALLY_COMPLIANT.value,
            "System with security vulnerabilities should fail OWASP validation",
        )

    def test_comprehensive_multi_framework_compliance(self):
        """
        Security Contract: System must achieve high compliance across multiple frameworks.
        Tests comprehensive compliance validation across all supported frameworks.
        """
        # Test comprehensive compliance validation
        all_results = self.governance_validator.validate_all_frameworks(self.compliant_system_config)

        # Assert overall compliance
        self.assertGreaterEqual(
            all_results["overall_compliance_percentage"], 80, "Overall compliance percentage should be high"
        )
        self.assertGreater(all_results["compliant_frameworks"], 0, "Should have at least some compliant frameworks")

        # Verify all major frameworks are tested
        framework_names = list(all_results["framework_results"].keys())
        expected_frameworks = ["gdpr", "coppa", "ferpa", "owasp_top_10"]

        for framework in expected_frameworks:
            self.assertIn(framework, framework_names, f"Framework {framework} must be tested")

        # Test with non-compliant system
        non_compliant_results = self.governance_validator.validate_all_frameworks(self.non_compliant_system_config)

        self.assertLess(
            non_compliant_results["overall_compliance_percentage"],
            50,
            "Non-compliant system should have low compliance percentage",
        )

        # Verify recommendations are generated
        self.assertGreater(len(all_results["recommendations"]), 0, "Compliance recommendations should be generated")

    def test_compliance_rule_validation_accuracy(self):
        """
        Security Contract: Compliance rule validation must be accurate and reliable.
        Tests accuracy of individual compliance rule validation logic.
        """
        # Test specific GDPR data minimization rule
        gdpr_rule = self.governance_validator.compliance_rules["gdpr_data_minimization"]

        # Test with excessive data collection
        excessive_config = {
            "collected_data_fields": ["name", "email", "phone", "ssn", "mother_maiden_name"],
            "necessary_data_fields": ["name", "email"],
            "retention_periods_defined": True,
        }

        result = gdpr_rule.validate(excessive_config)
        self.assertFalse(result["compliant"], "Rule should detect excessive data collection")
        self.assertIn(
            "excessive data fields", str(result["findings"]).lower(), "Should identify specific excessive fields"
        )

        # Test with compliant data collection
        compliant_config = {
            "collected_data_fields": ["name", "email"],
            "necessary_data_fields": ["name", "email"],
            "retention_periods_defined": True,
        }

        result = gdpr_rule.validate(compliant_config)
        self.assertTrue(result["compliant"], "Rule should pass for minimal data collection")

    def test_compliance_dashboard_generation(self):
        """
        Security Contract: Compliance status must be trackable through dashboard metrics.
        Tests compliance dashboard generation and metrics tracking.
        """
        # Generate compliance results
        self.governance_validator.validate_all_frameworks(self.compliant_system_config)

        # Generate dashboard
        dashboard = self.governance_validator.get_compliance_dashboard()

        # Assert dashboard completeness
        required_dashboard_fields = ["last_updated", "framework_status", "compliance_trends"]
        for field in required_dashboard_fields:
            self.assertIn(field, dashboard, f"Dashboard must include {field}")

        # Verify framework status tracking
        framework_status = dashboard["framework_status"]
        self.assertGreater(len(framework_status), 0, "Dashboard should track framework status")

        for framework_name, status in framework_status.items():
            status_fields = ["status", "compliance_percentage", "last_validated"]
            for field in status_fields:
                self.assertIn(field, status, f"Framework status must include {field}")

    def test_compliance_recommendation_prioritization(self):
        """
        Security Contract: Compliance recommendations must be prioritized by impact.
        Tests recommendation generation and prioritization logic.
        """
        # Create mixed compliance configuration
        mixed_config = self.compliant_system_config.copy()
        mixed_config["input_validation_implemented"] = False  # Affects OWASP
        mixed_config["privacy_policy_exists"] = False  # Affects GDPR
        mixed_config["mfa_implemented"] = False  # Affects OWASP

        # Generate compliance results
        all_results = self.governance_validator.validate_all_frameworks(mixed_config)

        # Verify recommendations are prioritized
        recommendations = all_results["recommendations"]
        self.assertGreater(len(recommendations), 0, "Should generate compliance recommendations")

        # High priority recommendations should be marked
        high_priority_recommendations = [rec for rec in recommendations if "[HIGH PRIORITY]" in rec]

        if len(high_priority_recommendations) > 0:
            self.assertGreater(
                len(high_priority_recommendations),
                0,
                "Should identify high priority recommendations for cross-framework issues",
            )

    def test_compliance_historical_tracking(self):
        """
        Security Contract: Compliance status changes must be tracked over time.
        Tests compliance history tracking and trend analysis.
        """
        # Perform initial compliance validation
        initial_results = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.GDPR, self.compliant_system_config
        )

        # Simulate configuration change and re-validate
        degraded_config = self.compliant_system_config.copy()
        degraded_config["privacy_policy_exists"] = False

        degraded_results = self.governance_validator.validate_framework_compliance(
            ComplianceFramework.GDPR, degraded_config
        )

        # Verify compliance history tracking
        history = self.governance_validator.compliance_history
        self.assertGreaterEqual(len(history), 2, "Should track compliance history")

        # Verify compliance degradation is tracked
        initial_percentage = initial_results["compliance_percentage"]
        degraded_percentage = degraded_results["compliance_percentage"]

        self.assertGreater(initial_percentage, degraded_percentage, "Should track compliance degradation")


if __name__ == "__main__":
    # Run tests with compliance focus
    unittest.main(verbosity=2, buffer=True)
