"""
Automotive Cybersecurity Compliance Validation System
Implements UN R155, ISO/SAE 21434, and automotive cybersecurity standards
Provides comprehensive compliance assessment and documentation
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import re

class ComplianceStandard(Enum):
    """Supported compliance standards"""
    UN_R155 = "UN_R155"
    ISO_SAE_21434 = "ISO_SAE_21434"
    ISO_26262 = "ISO_26262"
    NIST_CSF = "NIST_CSF"
    SAE_J3061 = "SAE_J3061"

class ComplianceLevel(Enum):
    """Compliance assessment levels"""
    NOT_APPLICABLE = "not_applicable"
    NOT_COMPLIANT = "not_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    FULLY_COMPLIANT = "fully_compliant"

class SeverityLevel(Enum):
    """Severity levels for compliance findings"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatCategory(Enum):
    """UNECE WP.29 threat categories"""
    BACKEND_SERVER = "backend_server"
    COMMUNICATION_CHANNELS = "communication_channels"
    ECU_APPLICATION_SOFTWARE = "ecu_application_software"
    ECU_FIRMWARE = "ecu_firmware"
    ECU_HARDWARE = "ecu_hardware"
    EXTERNAL_CONNECTIVITY = "external_connectivity"
    VEHICLE_COMMUNICATION_BUS = "vehicle_communication_bus"
    WIRELESS_COMMUNICATION = "wireless_communication"

@dataclass
class ComplianceRequirement:
    """Individual compliance requirement definition"""
    requirement_id: str
    standard: ComplianceStandard
    title: str
    description: str
    category: str
    mandatory: bool
    verification_method: str
    evidence_required: List[str]
    threat_categories: List[ThreatCategory]

@dataclass
class ComplianceFinding:
    """Compliance assessment finding"""
    finding_id: str
    requirement_id: str
    standard: ComplianceStandard
    compliance_level: ComplianceLevel
    severity: SeverityLevel
    description: str
    evidence: List[str]
    gaps: List[str]
    recommendations: List[str]
    remediation_effort: str
    timestamp: float

@dataclass
class VulnerabilityAssessment:
    """Security vulnerability assessment result"""
    vulnerability_id: str
    title: str
    description: str
    cvss_score: float
    cvss_vector: str
    threat_categories: List[ThreatCategory]
    affected_components: List[str]
    risk_level: SeverityLevel
    mitigation_status: str
    remediation_plan: str
    discovery_date: float
    remediation_date: Optional[float]

@dataclass
class SecurityControl:
    """Security control implementation"""
    control_id: str
    name: str
    description: str
    implementation_status: str
    effectiveness: str
    test_results: List[str]
    last_assessment: float
    responsible_party: str
    related_requirements: List[str]

class UN_R155_Validator:
    """UN R155 cybersecurity regulation validator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = self._initialize_un_r155_requirements()

    def _initialize_un_r155_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Initialize UN R155 requirements"""
        requirements = {}

        # Cybersecurity Management System (CSMS)
        requirements["R155_5.1"] = ComplianceRequirement(
            requirement_id="R155_5.1",
            standard=ComplianceStandard.UN_R155,
            title="Cybersecurity Management System",
            description="Manufacturer shall establish and maintain a CSMS",
            category="Management System",
            mandatory=True,
            verification_method="documentation_review",
            evidence_required=["csms_documentation", "policies", "procedures"],
            threat_categories=list(ThreatCategory)
        )

        requirements["R155_5.2"] = ComplianceRequirement(
            requirement_id="R155_5.2",
            standard=ComplianceStandard.UN_R155,
            title="Risk Assessment Process",
            description="CSMS shall include risk assessment processes",
            category="Risk Management",
            mandatory=True,
            verification_method="process_audit",
            evidence_required=["risk_assessments", "threat_analysis", "impact_analysis"],
            threat_categories=list(ThreatCategory)
        )

        requirements["R155_5.3"] = ComplianceRequirement(
            requirement_id="R155_5.3",
            standard=ComplianceStandard.UN_R155,
            title="Risk Treatment",
            description="CSMS shall include risk treatment activities",
            category="Risk Management",
            mandatory=True,
            verification_method="implementation_review",
            evidence_required=["mitigation_controls", "risk_register", "treatment_plans"],
            threat_categories=list(ThreatCategory)
        )

        requirements["R155_6.1"] = ComplianceRequirement(
            requirement_id="R155_6.1",
            standard=ComplianceStandard.UN_R155,
            title="Vehicle Type Approval",
            description="Cybersecurity measures for vehicle type approval",
            category="Type Approval",
            mandatory=True,
            verification_method="technical_assessment",
            evidence_required=["security_architecture", "penetration_tests", "vulnerability_scans"],
            threat_categories=[
                ThreatCategory.ECU_APPLICATION_SOFTWARE,
                ThreatCategory.ECU_FIRMWARE,
                ThreatCategory.VEHICLE_COMMUNICATION_BUS
            ]
        )

        requirements["R155_7.1"] = ComplianceRequirement(
            requirement_id="R155_7.1",
            standard=ComplianceStandard.UN_R155,
            title="Monitoring and Response",
            description="Post-production monitoring and incident response",
            category="Operations",
            mandatory=True,
            verification_method="process_review",
            evidence_required=["monitoring_system", "incident_response_plan", "vulnerability_disclosure"],
            threat_categories=list(ThreatCategory)
        )

        # Communication Channel Security
        requirements["R155_A1"] = ComplianceRequirement(
            requirement_id="R155_A1",
            standard=ComplianceStandard.UN_R155,
            title="Secure Communication Channels",
            description="Communication channels shall be secured against unauthorized access",
            category="Communication Security",
            mandatory=True,
            verification_method="technical_testing",
            evidence_required=["encryption_implementation", "authentication_mechanisms", "key_management"],
            threat_categories=[ThreatCategory.COMMUNICATION_CHANNELS, ThreatCategory.WIRELESS_COMMUNICATION]
        )

        # Vehicle Bus Security
        requirements["R155_A2"] = ComplianceRequirement(
            requirement_id="R155_A2",
            standard=ComplianceStandard.UN_R155,
            title="Vehicle Communication Bus Protection",
            description="Vehicle communication buses shall be protected against unauthorized access",
            category="Bus Security",
            mandatory=True,
            verification_method="technical_testing",
            evidence_required=["can_security_measures", "bus_monitoring", "intrusion_detection"],
            threat_categories=[ThreatCategory.VEHICLE_COMMUNICATION_BUS]
        )

        return requirements

    def validate_csms(self, csms_documentation: Dict[str, Any]) -> List[ComplianceFinding]:
        """Validate Cybersecurity Management System"""
        findings = []

        # Check CSMS documentation completeness
        required_elements = [
            'governance_structure',
            'roles_responsibilities',
            'risk_management_process',
            'incident_response_process',
            'supplier_management',
            'training_awareness',
            'monitoring_process'
        ]

        missing_elements = []
        for element in required_elements:
            if element not in csms_documentation:
                missing_elements.append(element)

        if missing_elements:
            findings.append(ComplianceFinding(
                finding_id=f"CSMS_INCOMPLETE_{int(time.time())}",
                requirement_id="R155_5.1",
                standard=ComplianceStandard.UN_R155,
                compliance_level=ComplianceLevel.PARTIALLY_COMPLIANT,
                severity=SeverityLevel.HIGH,
                description="CSMS documentation is incomplete",
                evidence=list(csms_documentation.keys()),
                gaps=missing_elements,
                recommendations=[f"Implement missing CSMS element: {elem}" for elem in missing_elements],
                remediation_effort="3-6 months",
                timestamp=time.time()
            ))
        else:
            findings.append(ComplianceFinding(
                finding_id=f"CSMS_COMPLETE_{int(time.time())}",
                requirement_id="R155_5.1",
                standard=ComplianceStandard.UN_R155,
                compliance_level=ComplianceLevel.FULLY_COMPLIANT,
                severity=SeverityLevel.INFO,
                description="CSMS documentation is complete",
                evidence=list(csms_documentation.keys()),
                gaps=[],
                recommendations=[],
                remediation_effort="None",
                timestamp=time.time()
            ))

        return findings

    def validate_risk_assessment(self, risk_assessments: List[Dict[str, Any]]) -> List[ComplianceFinding]:
        """Validate risk assessment process"""
        findings = []

        if not risk_assessments:
            findings.append(ComplianceFinding(
                finding_id=f"RISK_ASSESS_MISSING_{int(time.time())}",
                requirement_id="R155_5.2",
                standard=ComplianceStandard.UN_R155,
                compliance_level=ComplianceLevel.NOT_COMPLIANT,
                severity=SeverityLevel.CRITICAL,
                description="No risk assessments found",
                evidence=[],
                gaps=["risk_assessment_process", "threat_identification", "vulnerability_analysis"],
                recommendations=["Establish comprehensive risk assessment process"],
                remediation_effort="6-12 months",
                timestamp=time.time()
            ))
            return findings

        # Check risk assessment quality
        required_elements = ['threat_identification', 'vulnerability_analysis', 'impact_assessment', 'risk_calculation']

        complete_assessments = 0
        for assessment in risk_assessments:
            element_count = sum(1 for elem in required_elements if elem in assessment)
            if element_count == len(required_elements):
                complete_assessments += 1

        completeness_ratio = complete_assessments / len(risk_assessments)

        if completeness_ratio >= 0.9:
            compliance_level = ComplianceLevel.FULLY_COMPLIANT
            severity = SeverityLevel.INFO
        elif completeness_ratio >= 0.7:
            compliance_level = ComplianceLevel.SUBSTANTIALLY_COMPLIANT
            severity = SeverityLevel.LOW
        elif completeness_ratio >= 0.5:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            severity = SeverityLevel.MEDIUM
        else:
            compliance_level = ComplianceLevel.NOT_COMPLIANT
            severity = SeverityLevel.HIGH

        findings.append(ComplianceFinding(
            finding_id=f"RISK_ASSESS_QUALITY_{int(time.time())}",
            requirement_id="R155_5.2",
            standard=ComplianceStandard.UN_R155,
            compliance_level=compliance_level,
            severity=severity,
            description=f"Risk assessment completeness: {completeness_ratio:.1%}",
            evidence=[f"{len(risk_assessments)} risk assessments reviewed"],
            gaps=[] if completeness_ratio >= 0.9 else ["Incomplete risk assessments found"],
            recommendations=[] if completeness_ratio >= 0.9 else ["Complete all risk assessment elements"],
            remediation_effort="1-3 months" if completeness_ratio < 0.9 else "None",
            timestamp=time.time()
        ))

        return findings

class ISO_21434_Validator:
    """ISO/SAE 21434 cybersecurity engineering validator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.requirements = self._initialize_iso_21434_requirements()

    def _initialize_iso_21434_requirements(self) -> Dict[str, ComplianceRequirement]:
        """Initialize ISO/SAE 21434 requirements"""
        requirements = {}

        requirements["ISO21434_5"] = ComplianceRequirement(
            requirement_id="ISO21434_5",
            standard=ComplianceStandard.ISO_SAE_21434,
            title="Cybersecurity Governance",
            description="Organizational cybersecurity governance framework",
            category="Governance",
            mandatory=True,
            verification_method="governance_audit",
            evidence_required=["governance_framework", "policy_documents", "role_definitions"],
            threat_categories=list(ThreatCategory)
        )

        requirements["ISO21434_8"] = ComplianceRequirement(
            requirement_id="ISO21434_8",
            standard=ComplianceStandard.ISO_SAE_21434,
            title="Concept Phase Security",
            description="Cybersecurity activities during concept phase",
            category="Concept",
            mandatory=True,
            verification_method="design_review",
            evidence_required=["security_concept", "threat_model", "security_goals"],
            threat_categories=list(ThreatCategory)
        )

        requirements["ISO21434_9"] = ComplianceRequirement(
            requirement_id="ISO21434_9",
            standard=ComplianceStandard.ISO_SAE_21434,
            title="Product Development Security",
            description="Cybersecurity during product development",
            category="Development",
            mandatory=True,
            verification_method="development_review",
            evidence_required=["security_specification", "security_architecture", "security_testing"],
            threat_categories=list(ThreatCategory)
        )

        requirements["ISO21434_10"] = ComplianceRequirement(
            requirement_id="ISO21434_10",
            standard=ComplianceStandard.ISO_SAE_21434,
            title="Cybersecurity Validation",
            description="Cybersecurity validation and verification",
            category="Validation",
            mandatory=True,
            verification_method="validation_testing",
            evidence_required=["validation_plan", "test_results", "penetration_testing"],
            threat_categories=list(ThreatCategory)
        )

        requirements["ISO21434_12"] = ComplianceRequirement(
            requirement_id="ISO21434_12",
            standard=ComplianceStandard.ISO_SAE_21434,
            title="Incident Response",
            description="Cybersecurity incident monitoring and response",
            category="Operations",
            mandatory=True,
            verification_method="process_audit",
            evidence_required=["monitoring_system", "response_procedures", "incident_records"],
            threat_categories=list(ThreatCategory)
        )

        return requirements

    def validate_threat_analysis(self, threat_analysis: Dict[str, Any]) -> List[ComplianceFinding]:
        """Validate TARA (Threat Analysis and Risk Assessment)"""
        findings = []

        required_tara_elements = [
            'item_definition',
            'threat_scenarios',
            'impact_rating',
            'attack_path_analysis',
            'attack_feasibility',
            'risk_determination',
            'risk_treatment_decision'
        ]

        missing_elements = []
        for element in required_tara_elements:
            if element not in threat_analysis:
                missing_elements.append(element)

        # Check threat scenario coverage
        if 'threat_scenarios' in threat_analysis:
            scenarios = threat_analysis['threat_scenarios']
            covered_categories = set()

            for scenario in scenarios:
                if 'threat_category' in scenario:
                    covered_categories.add(scenario['threat_category'])

            # Should cover at least 80% of relevant threat categories
            coverage_ratio = len(covered_categories) / len(ThreatCategory)

            if coverage_ratio < 0.8:
                missing_elements.append(f"threat_coverage_{coverage_ratio:.1%}")

        if missing_elements:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            severity = SeverityLevel.MEDIUM if len(missing_elements) <= 2 else SeverityLevel.HIGH
        else:
            compliance_level = ComplianceLevel.FULLY_COMPLIANT
            severity = SeverityLevel.INFO

        findings.append(ComplianceFinding(
            finding_id=f"TARA_COMPLETENESS_{int(time.time())}",
            requirement_id="ISO21434_8",
            standard=ComplianceStandard.ISO_SAE_21434,
            compliance_level=compliance_level,
            severity=severity,
            description="TARA completeness assessment",
            evidence=list(threat_analysis.keys()),
            gaps=missing_elements,
            recommendations=[f"Complete missing TARA element: {elem}" for elem in missing_elements],
            remediation_effort="2-4 months" if missing_elements else "None",
            timestamp=time.time()
        ))

        return findings

    def validate_security_testing(self, test_results: List[Dict[str, Any]]) -> List[ComplianceFinding]:
        """Validate cybersecurity testing activities"""
        findings = []

        if not test_results:
            findings.append(ComplianceFinding(
                finding_id=f"SEC_TEST_MISSING_{int(time.time())}",
                requirement_id="ISO21434_10",
                standard=ComplianceStandard.ISO_SAE_21434,
                compliance_level=ComplianceLevel.NOT_COMPLIANT,
                severity=SeverityLevel.CRITICAL,
                description="No security testing evidence found",
                evidence=[],
                gaps=["penetration_testing", "vulnerability_scanning", "security_unit_tests"],
                recommendations=["Implement comprehensive security testing program"],
                remediation_effort="6-12 months",
                timestamp=time.time()
            ))
            return findings

        # Categorize test types
        test_types = set()
        for test in test_results:
            if 'test_type' in test:
                test_types.add(test['test_type'])

        required_test_types = {
            'penetration_testing',
            'vulnerability_scanning',
            'fuzz_testing',
            'security_unit_tests',
            'integration_security_tests'
        }

        missing_test_types = required_test_types - test_types
        test_coverage = len(test_types) / len(required_test_types)

        if test_coverage >= 0.9:
            compliance_level = ComplianceLevel.FULLY_COMPLIANT
            severity = SeverityLevel.INFO
        elif test_coverage >= 0.7:
            compliance_level = ComplianceLevel.SUBSTANTIALLY_COMPLIANT
            severity = SeverityLevel.LOW
        else:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
            severity = SeverityLevel.MEDIUM

        findings.append(ComplianceFinding(
            finding_id=f"SEC_TEST_COVERAGE_{int(time.time())}",
            requirement_id="ISO21434_10",
            standard=ComplianceStandard.ISO_SAE_21434,
            compliance_level=compliance_level,
            severity=severity,
            description=f"Security testing coverage: {test_coverage:.1%}",
            evidence=list(test_types),
            gaps=list(missing_test_types),
            recommendations=[f"Implement missing test type: {test_type}" for test_type in missing_test_types],
            remediation_effort="3-6 months" if missing_test_types else "None",
            timestamp=time.time()
        ))

        return findings

class VulnerabilityManager:
    """Vulnerability management and assessment"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vulnerabilities = {}
        self.vulnerability_database = self._load_vulnerability_patterns()

    def _load_vulnerability_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known vulnerability patterns"""
        return {
            "CAN_BUS_NO_AUTH": {
                "title": "CAN Bus Lacks Authentication",
                "description": "CAN bus messages are not authenticated",
                "cvss_base_score": 7.5,
                "threat_categories": [ThreatCategory.VEHICLE_COMMUNICATION_BUS],
                "mitigation": "Implement CAN message authentication"
            },
            "WIRELESS_WEAK_ENCRYPTION": {
                "title": "Weak Wireless Encryption",
                "description": "Wireless communications use weak encryption",
                "cvss_base_score": 8.2,
                "threat_categories": [ThreatCategory.WIRELESS_COMMUNICATION],
                "mitigation": "Upgrade to strong encryption algorithms"
            },
            "ECU_NO_SECURE_BOOT": {
                "title": "ECU Lacks Secure Boot",
                "description": "ECU does not implement secure boot process",
                "cvss_base_score": 6.8,
                "threat_categories": [ThreatCategory.ECU_FIRMWARE],
                "mitigation": "Implement secure boot with code signing"
            },
            "BACKEND_SQL_INJECTION": {
                "title": "SQL Injection Vulnerability",
                "description": "Backend services vulnerable to SQL injection",
                "cvss_base_score": 9.1,
                "threat_categories": [ThreatCategory.BACKEND_SERVER],
                "mitigation": "Use parameterized queries and input validation"
            }
        }

    def assess_vulnerabilities(self, security_assessment: Dict[str, Any]) -> List[VulnerabilityAssessment]:
        """Assess system for vulnerabilities"""
        vulnerabilities = []

        # Check for CAN bus authentication
        if not security_assessment.get('can_authentication', False):
            vuln = self._create_vulnerability("CAN_BUS_NO_AUTH", security_assessment)
            vulnerabilities.append(vuln)

        # Check wireless encryption strength
        wireless_config = security_assessment.get('wireless_configuration', {})
        if wireless_config.get('encryption_strength', '') in ['WEP', 'WPA', 'none']:
            vuln = self._create_vulnerability("WIRELESS_WEAK_ENCRYPTION", security_assessment)
            vulnerabilities.append(vuln)

        # Check secure boot implementation
        if not security_assessment.get('secure_boot_enabled', False):
            vuln = self._create_vulnerability("ECU_NO_SECURE_BOOT", security_assessment)
            vulnerabilities.append(vuln)

        # Check backend security
        backend_security = security_assessment.get('backend_security', {})
        if not backend_security.get('sql_injection_protection', False):
            vuln = self._create_vulnerability("BACKEND_SQL_INJECTION", security_assessment)
            vulnerabilities.append(vuln)

        return vulnerabilities

    def _create_vulnerability(self, pattern_id: str, assessment: Dict[str, Any]) -> VulnerabilityAssessment:
        """Create vulnerability assessment from pattern"""
        pattern = self.vulnerability_database[pattern_id]

        vulnerability_id = f"{pattern_id}_{int(time.time())}"

        return VulnerabilityAssessment(
            vulnerability_id=vulnerability_id,
            title=pattern["title"],
            description=pattern["description"],
            cvss_score=pattern["cvss_base_score"],
            cvss_vector=f"CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",  # Simplified
            threat_categories=pattern["threat_categories"],
            affected_components=assessment.get('affected_components', ['Unknown']),
            risk_level=self._cvss_to_severity(pattern["cvss_base_score"]),
            mitigation_status="Open",
            remediation_plan=pattern["mitigation"],
            discovery_date=time.time(),
            remediation_date=None
        )

    def _cvss_to_severity(self, cvss_score: float) -> SeverityLevel:
        """Convert CVSS score to severity level"""
        if cvss_score >= 9.0:
            return SeverityLevel.CRITICAL
        elif cvss_score >= 7.0:
            return SeverityLevel.HIGH
        elif cvss_score >= 4.0:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

class SecurityControlValidator:
    """Security control implementation validator"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.control_frameworks = {
            ComplianceStandard.NIST_CSF: self._initialize_nist_controls(),
            ComplianceStandard.ISO_26262: self._initialize_iso26262_controls()
        }

    def _initialize_nist_controls(self) -> Dict[str, SecurityControl]:
        """Initialize NIST Cybersecurity Framework controls"""
        controls = {}

        controls["ID.AM-1"] = SecurityControl(
            control_id="ID.AM-1",
            name="Asset Management",
            description="Physical devices and systems are inventoried",
            implementation_status="Not Implemented",
            effectiveness="Not Assessed",
            test_results=[],
            last_assessment=0.0,
            responsible_party="IT Security Team",
            related_requirements=["R155_5.1"]
        )

        controls["PR.AC-1"] = SecurityControl(
            control_id="PR.AC-1",
            name="Access Control",
            description="Identities and credentials are issued and managed",
            implementation_status="Not Implemented",
            effectiveness="Not Assessed",
            test_results=[],
            last_assessment=0.0,
            responsible_party="IT Security Team",
            related_requirements=["R155_5.3"]
        )

        controls["DE.AE-1"] = SecurityControl(
            control_id="DE.AE-1",
            name="Anomaly Detection",
            description="Baseline network operations are established",
            implementation_status="Not Implemented",
            effectiveness="Not Assessed",
            test_results=[],
            last_assessment=0.0,
            responsible_party="SOC Team",
            related_requirements=["R155_7.1"]
        )

        return controls

    def _initialize_iso26262_controls(self) -> Dict[str, SecurityControl]:
        """Initialize ISO 26262 functional safety controls"""
        controls = {}

        controls["ISO26262-6-7"] = SecurityControl(
            control_id="ISO26262-6-7",
            name="Software Safety Requirements",
            description="Software safety requirements specification",
            implementation_status="Not Implemented",
            effectiveness="Not Assessed",
            test_results=[],
            last_assessment=0.0,
            responsible_party="Software Team",
            related_requirements=["ISO21434_9"]
        )

        return controls

    def validate_controls(self, implemented_controls: Dict[str, Dict[str, Any]]) -> List[ComplianceFinding]:
        """Validate security control implementation"""
        findings = []

        for standard, controls in self.control_frameworks.items():
            for control_id, control in controls.items():
                if control_id in implemented_controls:
                    # Control is implemented, assess effectiveness
                    impl_data = implemented_controls[control_id]

                    effectiveness = impl_data.get('effectiveness', 'Not Assessed')
                    test_results = impl_data.get('test_results', [])

                    if effectiveness == 'Effective' and test_results:
                        compliance_level = ComplianceLevel.FULLY_COMPLIANT
                        severity = SeverityLevel.INFO
                    elif effectiveness == 'Partially Effective':
                        compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
                        severity = SeverityLevel.MEDIUM
                    else:
                        compliance_level = ComplianceLevel.NOT_COMPLIANT
                        severity = SeverityLevel.HIGH

                    findings.append(ComplianceFinding(
                        finding_id=f"CONTROL_{control_id}_{int(time.time())}",
                        requirement_id=control_id,
                        standard=standard,
                        compliance_level=compliance_level,
                        severity=severity,
                        description=f"Control {control_id} implementation assessment",
                        evidence=[f"Implementation status: {effectiveness}"] + test_results,
                        gaps=[] if compliance_level == ComplianceLevel.FULLY_COMPLIANT else ["Control effectiveness issues"],
                        recommendations=[] if compliance_level == ComplianceLevel.FULLY_COMPLIANT else ["Improve control effectiveness"],
                        remediation_effort="1-3 months" if compliance_level != ComplianceLevel.FULLY_COMPLIANT else "None",
                        timestamp=time.time()
                    ))
                else:
                    # Control not implemented
                    findings.append(ComplianceFinding(
                        finding_id=f"CONTROL_MISSING_{control_id}_{int(time.time())}",
                        requirement_id=control_id,
                        standard=standard,
                        compliance_level=ComplianceLevel.NOT_COMPLIANT,
                        severity=SeverityLevel.HIGH,
                        description=f"Security control {control_id} not implemented",
                        evidence=[],
                        gaps=[f"Missing control: {control.name}"],
                        recommendations=[f"Implement security control: {control.name}"],
                        remediation_effort="3-6 months",
                        timestamp=time.time()
                    ))

        return findings

class ComplianceReportGenerator:
    """Generate comprehensive compliance reports"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_compliance_report(self,
                                 findings: List[ComplianceFinding],
                                 vulnerabilities: List[VulnerabilityAssessment],
                                 vehicle_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""

        # Aggregate findings by standard
        findings_by_standard = {}
        for finding in findings:
            if finding.standard not in findings_by_standard:
                findings_by_standard[finding.standard] = []
            findings_by_standard[finding.standard].append(finding)

        # Calculate compliance scores
        compliance_scores = {}
        for standard, standard_findings in findings_by_standard.items():
            scores = [self._compliance_to_score(f.compliance_level) for f in standard_findings]
            compliance_scores[standard.value] = {
                'overall_score': sum(scores) / len(scores) if scores else 0,
                'total_requirements': len(standard_findings),
                'compliant_requirements': len([f for f in standard_findings if f.compliance_level == ComplianceLevel.FULLY_COMPLIANT]),
                'non_compliant_requirements': len([f for f in standard_findings if f.compliance_level == ComplianceLevel.NOT_COMPLIANT])
            }

        # Vulnerability summary
        vuln_by_severity = {}
        for vuln in vulnerabilities:
            if vuln.risk_level not in vuln_by_severity:
                vuln_by_severity[vuln.risk_level] = []
            vuln_by_severity[vuln.risk_level].append(vuln)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(findings, vulnerabilities, compliance_scores)

        report = {
            'report_metadata': {
                'generated_timestamp': time.time(),
                'generated_date': datetime.now().isoformat(),
                'vehicle_info': vehicle_info,
                'report_version': '1.0',
                'assessment_scope': list(findings_by_standard.keys())
            },
            'executive_summary': executive_summary,
            'compliance_scores': compliance_scores,
            'findings_summary': {
                'total_findings': len(findings),
                'critical_findings': len([f for f in findings if f.severity == SeverityLevel.CRITICAL]),
                'high_findings': len([f for f in findings if f.severity == SeverityLevel.HIGH]),
                'medium_findings': len([f for f in findings if f.severity == SeverityLevel.MEDIUM]),
                'low_findings': len([f for f in findings if f.severity == SeverityLevel.LOW])
            },
            'vulnerability_summary': {
                'total_vulnerabilities': len(vulnerabilities),
                'critical_vulnerabilities': len(vuln_by_severity.get(SeverityLevel.CRITICAL, [])),
                'high_vulnerabilities': len(vuln_by_severity.get(SeverityLevel.HIGH, [])),
                'medium_vulnerabilities': len(vuln_by_severity.get(SeverityLevel.MEDIUM, [])),
                'low_vulnerabilities': len(vuln_by_severity.get(SeverityLevel.LOW, []))
            },
            'detailed_findings': [asdict(finding) for finding in findings],
            'vulnerability_assessments': [asdict(vuln) for vuln in vulnerabilities],
            'recommendations': self._generate_recommendations(findings, vulnerabilities),
            'remediation_roadmap': self._generate_remediation_roadmap(findings, vulnerabilities)
        }

        return report

    def _compliance_to_score(self, compliance_level: ComplianceLevel) -> float:
        """Convert compliance level to numeric score"""
        scores = {
            ComplianceLevel.FULLY_COMPLIANT: 1.0,
            ComplianceLevel.SUBSTANTIALLY_COMPLIANT: 0.8,
            ComplianceLevel.PARTIALLY_COMPLIANT: 0.5,
            ComplianceLevel.NOT_COMPLIANT: 0.0,
            ComplianceLevel.NOT_APPLICABLE: 1.0  # Don't penalize N/A items
        }
        return scores.get(compliance_level, 0.0)

    def _generate_executive_summary(self, findings: List[ComplianceFinding],
                                   vulnerabilities: List[VulnerabilityAssessment],
                                   compliance_scores: Dict[str, Any]) -> str:
        """Generate executive summary"""

        critical_issues = len([f for f in findings if f.severity == SeverityLevel.CRITICAL])
        critical_vulns = len([v for v in vulnerabilities if v.risk_level == SeverityLevel.CRITICAL])

        overall_scores = [score['overall_score'] for score in compliance_scores.values()]
        avg_compliance = sum(overall_scores) / len(overall_scores) if overall_scores else 0

        summary = f"""
AUTOMOTIVE CYBERSECURITY COMPLIANCE ASSESSMENT

Overall Compliance Score: {avg_compliance:.1%}

Critical Issues Identified: {critical_issues + critical_vulns}
- {critical_issues} critical compliance findings
- {critical_vulns} critical vulnerabilities

Key Findings:
- Total compliance requirements assessed: {len(findings)}
- Total security vulnerabilities identified: {len(vulnerabilities)}

Standards Assessed: {', '.join(compliance_scores.keys())}

Immediate Actions Required:
- Address all critical severity findings
- Implement missing security controls
- Establish vulnerability management process

The assessment indicates {'URGENT REMEDIATION REQUIRED' if critical_issues + critical_vulns > 0 else 'GOOD SECURITY POSTURE'}.
        """

        return summary.strip()

    def _generate_recommendations(self, findings: List[ComplianceFinding],
                                vulnerabilities: List[VulnerabilityAssessment]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []

        # High-priority recommendations from critical findings
        critical_findings = [f for f in findings if f.severity == SeverityLevel.CRITICAL]
        for finding in critical_findings:
            recommendations.extend(finding.recommendations)

        # Add vulnerability mitigation recommendations
        critical_vulns = [v for v in vulnerabilities if v.risk_level == SeverityLevel.CRITICAL]
        for vuln in critical_vulns:
            recommendations.append(f"Mitigate {vuln.title}: {vuln.remediation_plan}")

        # General recommendations
        recommendations.extend([
            "Establish continuous security monitoring",
            "Implement regular penetration testing",
            "Develop incident response capabilities",
            "Provide cybersecurity training to development teams",
            "Establish secure development lifecycle processes"
        ])

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        return unique_recommendations[:20]  # Top 20 recommendations

    def _generate_remediation_roadmap(self, findings: List[ComplianceFinding],
                                    vulnerabilities: List[VulnerabilityAssessment]) -> List[Dict[str, Any]]:
        """Generate remediation roadmap with priorities and timelines"""
        roadmap = []

        # Critical items (0-3 months)
        critical_items = [f for f in findings if f.severity == SeverityLevel.CRITICAL]
        critical_vulns = [v for v in vulnerabilities if v.risk_level == SeverityLevel.CRITICAL]

        if critical_items or critical_vulns:
            roadmap.append({
                'phase': 'Immediate (0-3 months)',
                'priority': 'Critical',
                'items': len(critical_items) + len(critical_vulns),
                'focus': 'Address critical security gaps and vulnerabilities',
                'deliverables': ['Security patches', 'Critical control implementations', 'Vulnerability remediation']
            })

        # High priority items (3-6 months)
        high_items = [f for f in findings if f.severity == SeverityLevel.HIGH]
        high_vulns = [v for v in vulnerabilities if v.risk_level == SeverityLevel.HIGH]

        if high_items or high_vulns:
            roadmap.append({
                'phase': 'Short-term (3-6 months)',
                'priority': 'High',
                'items': len(high_items) + len(high_vulns),
                'focus': 'Implement core security controls and address high-risk issues',
                'deliverables': ['Security architecture improvements', 'Control implementations', 'Process enhancements']
            })

        # Medium priority items (6-12 months)
        medium_items = [f for f in findings if f.severity == SeverityLevel.MEDIUM]
        medium_vulns = [v for v in vulnerabilities if v.risk_level == SeverityLevel.MEDIUM]

        if medium_items or medium_vulns:
            roadmap.append({
                'phase': 'Medium-term (6-12 months)',
                'priority': 'Medium',
                'items': len(medium_items) + len(medium_vulns),
                'focus': 'Comprehensive compliance achievement and security optimization',
                'deliverables': ['Full compliance documentation', 'Security testing program', 'Continuous monitoring']
            })

        return roadmap

class AutomotiveComplianceValidator:
    """Main automotive compliance validation system"""

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.logger = logging.getLogger(__name__)

        # Initialize validators
        self.un_r155_validator = UN_R155_Validator()
        self.iso_21434_validator = ISO_21434_Validator()
        self.vulnerability_manager = VulnerabilityManager()
        self.control_validator = SecurityControlValidator()
        self.report_generator = ComplianceReportGenerator()

    def perform_compliance_assessment(self, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive compliance assessment"""
        self.logger.info(f"Starting compliance assessment for vehicle {self.vehicle_id}")

        all_findings = []

        # UN R155 Validation
        if 'csms_documentation' in assessment_data:
            csms_findings = self.un_r155_validator.validate_csms(assessment_data['csms_documentation'])
            all_findings.extend(csms_findings)

        if 'risk_assessments' in assessment_data:
            risk_findings = self.un_r155_validator.validate_risk_assessment(assessment_data['risk_assessments'])
            all_findings.extend(risk_findings)

        # ISO/SAE 21434 Validation
        if 'threat_analysis' in assessment_data:
            tara_findings = self.iso_21434_validator.validate_threat_analysis(assessment_data['threat_analysis'])
            all_findings.extend(tara_findings)

        if 'security_test_results' in assessment_data:
            test_findings = self.iso_21434_validator.validate_security_testing(assessment_data['security_test_results'])
            all_findings.extend(test_findings)

        # Security Control Validation
        if 'implemented_controls' in assessment_data:
            control_findings = self.control_validator.validate_controls(assessment_data['implemented_controls'])
            all_findings.extend(control_findings)

        # Vulnerability Assessment
        vulnerabilities = []
        if 'security_assessment' in assessment_data:
            vulnerabilities = self.vulnerability_manager.assess_vulnerabilities(assessment_data['security_assessment'])

        # Generate comprehensive report
        vehicle_info = {
            'vehicle_id': self.vehicle_id,
            'assessment_date': datetime.now().isoformat(),
            'assessment_scope': list(assessment_data.keys())
        }

        compliance_report = self.report_generator.generate_compliance_report(
            all_findings, vulnerabilities, vehicle_info
        )

        self.logger.info(f"Compliance assessment completed. {len(all_findings)} findings, {len(vulnerabilities)} vulnerabilities")

        return compliance_report

    def generate_certification_package(self, compliance_report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate certification documentation package"""

        certification_package = {
            'certification_metadata': {
                'vehicle_id': self.vehicle_id,
                'certification_date': datetime.now().isoformat(),
                'certifying_organization': 'Internal Assessment',
                'certification_scope': 'Automotive Cybersecurity'
            },
            'compliance_attestation': {
                'overall_compliance_score': compliance_report.get('compliance_scores', {}),
                'critical_issues_resolved': len([f for f in compliance_report.get('detailed_findings', [])
                                               if f['severity'] == 'critical']) == 0,
                'certification_recommendation': self._determine_certification_recommendation(compliance_report)
            },
            'evidence_package': {
                'security_architecture_documentation': 'Required',
                'penetration_test_reports': 'Required',
                'vulnerability_assessment_reports': 'Provided',
                'security_control_implementations': 'Required',
                'incident_response_procedures': 'Required'
            },
            'compliance_matrices': self._generate_compliance_matrices(compliance_report),
            'remediation_status': compliance_report.get('remediation_roadmap', [])
        }

        return certification_package

    def _determine_certification_recommendation(self, compliance_report: Dict[str, Any]) -> str:
        """Determine certification recommendation based on assessment results"""

        critical_findings = compliance_report.get('findings_summary', {}).get('critical_findings', 0)
        high_findings = compliance_report.get('findings_summary', {}).get('high_findings', 0)
        critical_vulns = compliance_report.get('vulnerability_summary', {}).get('critical_vulnerabilities', 0)

        compliance_scores = compliance_report.get('compliance_scores', {})
        avg_compliance = sum(score.get('overall_score', 0) for score in compliance_scores.values()) / len(compliance_scores) if compliance_scores else 0

        if critical_findings > 0 or critical_vulns > 0:
            return "CERTIFICATION DENIED - Critical issues must be resolved"
        elif high_findings > 5 or avg_compliance < 0.7:
            return "CONDITIONAL APPROVAL - High-priority issues must be addressed within 90 days"
        elif avg_compliance >= 0.9:
            return "FULL CERTIFICATION RECOMMENDED - Excellent security posture"
        else:
            return "CERTIFICATION APPROVED - Good security posture with minor improvements needed"

    def _generate_compliance_matrices(self, compliance_report: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate compliance requirement matrices"""

        matrices = {}

        findings = compliance_report.get('detailed_findings', [])
        findings_by_standard = {}

        for finding in findings:
            standard = finding['standard']
            if standard not in findings_by_standard:
                findings_by_standard[standard] = []
            findings_by_standard[standard].append(finding)

        for standard, standard_findings in findings_by_standard.items():
            matrices[standard] = []
            for finding in standard_findings:
                matrices[standard].append({
                    'requirement_id': finding['requirement_id'],
                    'compliance_level': finding['compliance_level'],
                    'evidence': finding['evidence'],
                    'gaps': finding['gaps'],
                    'status': 'Complete' if finding['compliance_level'] == 'fully_compliant' else 'Needs Work'
                })

        return matrices

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize compliance validator
    validator = AutomotiveComplianceValidator("VEHICLE_001")

    # Sample assessment data
    assessment_data = {
        'csms_documentation': {
            'governance_structure': True,
            'roles_responsibilities': True,
            'risk_management_process': True,
            'incident_response_process': False,  # Missing
            'supplier_management': True,
            'training_awareness': False,  # Missing
            'monitoring_process': True
        },
        'risk_assessments': [
            {
                'threat_identification': True,
                'vulnerability_analysis': True,
                'impact_assessment': True,
                'risk_calculation': True
            },
            {
                'threat_identification': True,
                'vulnerability_analysis': False,  # Incomplete
                'impact_assessment': True,
                'risk_calculation': False  # Incomplete
            }
        ],
        'threat_analysis': {
            'item_definition': True,
            'threat_scenarios': [
                {'threat_category': 'vehicle_communication_bus'},
                {'threat_category': 'wireless_communication'}
            ],
            'impact_rating': True,
            'attack_path_analysis': True,
            'attack_feasibility': True,
            'risk_determination': True,
            'risk_treatment_decision': False  # Missing
        },
        'security_test_results': [
            {'test_type': 'penetration_testing', 'result': 'passed'},
            {'test_type': 'vulnerability_scanning', 'result': 'failed'},
            {'test_type': 'fuzz_testing', 'result': 'passed'}
        ],
        'security_assessment': {
            'can_authentication': False,
            'secure_boot_enabled': True,
            'wireless_configuration': {
                'encryption_strength': 'WPA2'
            },
            'backend_security': {
                'sql_injection_protection': False
            },
            'affected_components': ['CAN Controller', 'Wireless Module', 'Backend API']
        },
        'implemented_controls': {
            'ID.AM-1': {
                'effectiveness': 'Effective',
                'test_results': ['Asset inventory complete', 'Regular updates maintained']
            },
            'PR.AC-1': {
                'effectiveness': 'Partially Effective',
                'test_results': ['Some access controls implemented']
            }
        }
    }

    # Perform compliance assessment
    compliance_report = validator.perform_compliance_assessment(assessment_data)

    # Print summary results
    print(f"Compliance Assessment Results:")
    print(f"Total Findings: {compliance_report['findings_summary']['total_findings']}")
    print(f"Critical Findings: {compliance_report['findings_summary']['critical_findings']}")
    print(f"Total Vulnerabilities: {compliance_report['vulnerability_summary']['total_vulnerabilities']}")
    print(f"Critical Vulnerabilities: {compliance_report['vulnerability_summary']['critical_vulnerabilities']}")

    # Generate certification package
    cert_package = validator.generate_certification_package(compliance_report)
    print(f"\nCertification Recommendation: {cert_package['compliance_attestation']['certification_recommendation']}")

    # Save reports to files
    with open(f"compliance_report_{validator.vehicle_id}.json", 'w') as f:
        json.dump(compliance_report, f, indent=2, default=str)

    with open(f"certification_package_{validator.vehicle_id}.json", 'w') as f:
        json.dump(cert_package, f, indent=2, default=str)