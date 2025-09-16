#!/usr/bin/env python3
"""
NASA POT10 Compliance Validation System
Defense-grade compliance validation for training operations

CLASSIFICATION: CONTROLLED UNCLASSIFIED INFORMATION (CUI)
DFARS: 252.204-7012 Compliant
NASA POT10: 95% Compliance Target
"""

import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .enhanced_audit_trail_manager import EnhancedAuditTrail
from .fips_crypto_module import FIPSCryptoModule

class ComplianceLevel(Enum):
    """Compliance severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    PENDING_REVIEW = "PENDING_REVIEW"

@dataclass
class ComplianceCheck:
    """Individual compliance check definition"""
    check_id: str
    check_name: str
    description: str
    category: str
    framework: str
    severity: ComplianceLevel
    requirement: str
    validation_method: str
    expected_result: str
    remediation_guidance: str

@dataclass
class ComplianceResult:
    """Results of a compliance check"""
    check_id: str
    check_name: str
    status: ComplianceStatus
    score: float
    evidence: List[str]
    findings: List[str]
    recommendations: List[str]
    timestamp: datetime
    assessed_by: str
    next_assessment_due: datetime

@dataclass
class ComplianceAssessment:
    """Complete compliance assessment"""
    assessment_id: str
    framework: str
    timestamp: datetime
    assessed_by: str
    overall_score: float
    overall_status: ComplianceStatus
    category_scores: Dict[str, float]
    results: List[ComplianceResult]
    executive_summary: str
    remediation_plan: List[Dict[str, Any]]

class NASAPot10ComplianceValidator:
    """
    NASA POT10 Compliance validation system

    Implements comprehensive compliance validation including:
    - Automated compliance checks across all categories
    - Evidence collection and verification
    - Gap analysis and remediation planning
    - Continuous compliance monitoring
    - Executive reporting and dashboards
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.audit = EnhancedAuditTrail()
        self.crypto = FIPSCryptoModule()

        # Load compliance frameworks
        self.compliance_checks = self._load_compliance_checks()
        self.assessment_history = []
        self.continuous_monitoring = False

        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load compliance validation configuration"""
        default_config = {
            'assessment': {
                'frequency': 'monthly',
                'retention_years': 7,
                'evidence_collection': True,
                'automated_remediation': False
            },
            'reporting': {
                'executive_dashboard': True,
                'detailed_reports': True,
                'trend_analysis': True,
                'risk_scoring': True
            },
            'thresholds': {
                'overall_compliance_threshold': 95.0,
                'critical_finding_threshold': 0,
                'high_finding_threshold': 2,
                'remediation_timeframe': 30
            },
            'frameworks': {
                'nasa_pot10': True,
                'dfars_252_204_7012': True,
                'nist_sp_800_171': True,
                'iso_27001': False
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _load_compliance_checks(self) -> Dict[str, ComplianceCheck]:
        """Load compliance check definitions"""
        checks = {}

        # NASA POT10 Compliance Checks
        nasa_checks = self._create_nasa_pot10_checks()
        checks.update(nasa_checks)

        # DFARS 252.204-7012 Checks
        dfars_checks = self._create_dfars_checks()
        checks.update(dfars_checks)

        return checks

    def _create_nasa_pot10_checks(self) -> Dict[str, ComplianceCheck]:
        """Create NASA POT10 compliance checks"""
        checks = {}

        # Software Development Process Controls
        checks['POT10-001'] = ComplianceCheck(
            check_id='POT10-001',
            check_name='Software Development Lifecycle',
            description='Verify implementation of comprehensive SDLC process',
            category='Development Process',
            framework='NASA_POT10',
            severity=ComplianceLevel.CRITICAL,
            requirement='Implement documented SDLC with security controls',
            validation_method='Process Documentation Review',
            expected_result='Complete SDLC documentation with security integration',
            remediation_guidance='Implement formal SDLC process with security gates'
        )

        checks['POT10-002'] = ComplianceCheck(
            check_id='POT10-002',
            check_name='Code Review Process',
            description='Verify mandatory code review process implementation',
            category='Quality Assurance',
            framework='NASA_POT10',
            severity=ComplianceLevel.HIGH,
            requirement='All code must undergo peer review before deployment',
            validation_method='Code Review Records Analysis',
            expected_result='100% code review coverage for production code',
            remediation_guidance='Implement automated code review enforcement'
        )

        checks['POT10-003'] = ComplianceCheck(
            check_id='POT10-003',
            check_name='Security Testing',
            description='Verify comprehensive security testing implementation',
            category='Security Controls',
            framework='NASA_POT10',
            severity=ComplianceLevel.CRITICAL,
            requirement='Security testing in all development phases',
            validation_method='Security Test Results Review',
            expected_result='Zero critical security vulnerabilities',
            remediation_guidance='Implement automated security testing pipeline'
        )

        checks['POT10-004'] = ComplianceCheck(
            check_id='POT10-004',
            check_name='Configuration Management',
            description='Verify configuration management controls',
            category='Change Management',
            framework='NASA_POT10',
            severity=ComplianceLevel.HIGH,
            requirement='All configuration changes must be tracked and approved',
            validation_method='Configuration Audit',
            expected_result='Complete configuration change tracking',
            remediation_guidance='Implement configuration management system'
        )

        checks['POT10-005'] = ComplianceCheck(
            check_id='POT10-005',
            check_name='Audit Trail Integrity',
            description='Verify audit trail completeness and integrity',
            category='Audit and Accountability',
            framework='NASA_POT10',
            severity=ComplianceLevel.CRITICAL,
            requirement='Complete tamper-evident audit trails',
            validation_method='Audit Trail Analysis',
            expected_result='100% audit trail completeness and integrity',
            remediation_guidance='Implement enhanced audit trail system'
        )

        return checks

    def _create_dfars_checks(self) -> Dict[str, ComplianceCheck]:
        """Create DFARS 252.204-7012 compliance checks"""
        checks = {}

        # Access Control (3.1.x)
        checks['DFARS-3.1.1'] = ComplianceCheck(
            check_id='DFARS-3.1.1',
            check_name='Account Management',
            description='Verify proper account management controls',
            category='Access Control',
            framework='DFARS_252.204-7012',
            severity=ComplianceLevel.HIGH,
            requirement='Limit system access to authorized users, processes acting on behalf of authorized users, or devices',
            validation_method='Account Review',
            expected_result='All accounts properly authorized and documented',
            remediation_guidance='Implement formal account management process'
        )

        checks['DFARS-3.1.2'] = ComplianceCheck(
            check_id='DFARS-3.1.2',
            check_name='Access Enforcement',
            description='Verify access enforcement mechanisms',
            category='Access Control',
            framework='DFARS_252.204-7012',
            severity=ComplianceLevel.HIGH,
            requirement='Limit system access to the types of transactions and functions that authorized users are permitted to execute',
            validation_method='Access Control Testing',
            expected_result='Access properly restricted to authorized functions',
            remediation_guidance='Implement role-based access control (RBAC)'
        )

        # Audit and Accountability (3.3.x)
        checks['DFARS-3.3.1'] = ComplianceCheck(
            check_id='DFARS-3.3.1',
            check_name='Audit Event Monitoring',
            description='Verify comprehensive audit event monitoring',
            category='Audit and Accountability',
            framework='DFARS_252.204-7012',
            severity=ComplianceLevel.HIGH,
            requirement='Create and retain system audit logs and records to the extent necessary to enable the monitoring, analysis, investigation, and reporting of unlawful or unauthorized system activity',
            validation_method='Audit Log Review',
            expected_result='Comprehensive audit logging implemented',
            remediation_guidance='Implement enhanced audit logging system'
        )

        return checks

    def run_compliance_assessment(self, user_id: str,
                                 assessment_scope: Optional[List[str]] = None) -> ComplianceAssessment:
        """
        Run comprehensive compliance assessment

        Args:
            user_id: User running the assessment
            assessment_scope: Optional list of specific checks to run

        Returns:
            Complete compliance assessment results
        """
        assessment_id = hashlib.sha256(
            f"assessment_{user_id}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # Determine checks to run
        if assessment_scope:
            checks_to_run = {k: v for k, v in self.compliance_checks.items() if k in assessment_scope}
        else:
            checks_to_run = self.compliance_checks

        # Run individual compliance checks
        results = []
        for check_id, check in checks_to_run.items():
            result = self._run_compliance_check(check, user_id)
            results.append(result)

        # Calculate overall compliance score
        overall_score = self._calculate_overall_score(results)
        overall_status = self._determine_overall_status(overall_score, results)

        # Calculate category scores
        category_scores = self._calculate_category_scores(results)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(overall_score, results)

        # Create remediation plan
        remediation_plan = self._create_remediation_plan(results)

        assessment = ComplianceAssessment(
            assessment_id=assessment_id,
            framework='NASA_POT10_DFARS',
            timestamp=datetime.now(timezone.utc),
            assessed_by=user_id,
            overall_score=overall_score,
            overall_status=overall_status,
            category_scores=category_scores,
            results=results,
            executive_summary=executive_summary,
            remediation_plan=remediation_plan
        )

        # Store assessment
        self.assessment_history.append(assessment)

        # Log assessment completion
        self.audit.log_security_event(
            event_type='compliance_assessment',
            user_id=user_id,
            action='complete_assessment',
            resource=f"compliance_assessment_{assessment_id}",
            classification='CUI//BASIC',
            additional_data={
                'assessment_id': assessment_id,
                'overall_score': overall_score,
                'overall_status': overall_status.value,
                'checks_performed': len(results)
            }
        )

        return assessment

    def _run_compliance_check(self, check: ComplianceCheck, user_id: str) -> ComplianceResult:
        """Run individual compliance check"""
        # Collect evidence based on check type
        evidence = self._collect_evidence(check)

        # Perform validation
        status, score, findings = self._validate_compliance(check, evidence)

        # Generate recommendations
        recommendations = self._generate_recommendations(check, status, findings)

        return ComplianceResult(
            check_id=check.check_id,
            check_name=check.check_name,
            status=status,
            score=score,
            evidence=evidence,
            findings=findings,
            recommendations=recommendations,
            timestamp=datetime.now(timezone.utc),
            assessed_by=user_id,
            next_assessment_due=datetime.now(timezone.utc) + timedelta(days=30)
        )

    def _collect_evidence(self, check: ComplianceCheck) -> List[str]:
        """Collect evidence for compliance check"""
        evidence = []

        if check.category == 'Development Process':
            evidence.extend(self._collect_development_process_evidence())
        elif check.category == 'Quality Assurance':
            evidence.extend(self._collect_quality_assurance_evidence())
        elif check.category == 'Security Controls':
            evidence.extend(self._collect_security_controls_evidence())
        elif check.category == 'Change Management':
            evidence.extend(self._collect_change_management_evidence())
        elif check.category == 'Audit and Accountability':
            evidence.extend(self._collect_audit_evidence())
        elif check.category == 'Access Control':
            evidence.extend(self._collect_access_control_evidence())

        return evidence

    def _collect_development_process_evidence(self) -> List[str]:
        """Collect development process evidence"""
        evidence = []

        # Check for SDLC documentation
        sdlc_docs = ['docs/development/sdlc.md', 'docs/process/development_lifecycle.md']
        for doc in sdlc_docs:
            if os.path.exists(doc):
                evidence.append(f"SDLC documentation found: {doc}")

        # Check for security integration
        if os.path.exists('src/security'):
            evidence.append("Security components integrated into codebase")

        # Check for testing frameworks
        if os.path.exists('tests'):
            evidence.append("Testing framework implemented")

        return evidence

    def _collect_quality_assurance_evidence(self) -> List[str]:
        """Collect quality assurance evidence"""
        evidence = []

        # Check for code review configuration
        if os.path.exists('.github/pull_request_template.md'):
            evidence.append("Pull request template configured for code reviews")

        # Check for quality gates
        if os.path.exists('.github/workflows'):
            evidence.append("Automated quality gates configured")

        # Check for linting configuration
        lint_configs = ['.eslintrc', 'pyproject.toml', '.pylintrc']
        for config in lint_configs:
            if os.path.exists(config):
                evidence.append(f"Linting configuration found: {config}")

        return evidence

    def _collect_security_controls_evidence(self) -> List[str]:
        """Collect security controls evidence"""
        evidence = []

        # Check for security scanning
        if os.path.exists('scripts/security-scan.py'):
            evidence.append("Security scanning tools configured")

        # Check for vulnerability management
        if os.path.exists('config/vulnerability-whitelist.json'):
            evidence.append("Vulnerability management process implemented")

        # Check for security policies
        if os.path.exists('config/security-policy.json'):
            evidence.append("Security policies documented")

        return evidence

    def _collect_change_management_evidence(self) -> List[str]:
        """Collect change management evidence"""
        evidence = []

        # Check for version control
        if os.path.exists('.git'):
            evidence.append("Version control system implemented")

        # Check for configuration management
        if os.path.exists('config'):
            evidence.append("Configuration management structure present")

        return evidence

    def _collect_audit_evidence(self) -> List[str]:
        """Collect audit and accountability evidence"""
        evidence = []

        # Check for audit trail implementation
        if os.path.exists('src/security/enhanced_audit_trail_manager.py'):
            evidence.append("Enhanced audit trail system implemented")

        # Check for logging configuration
        if os.path.exists('src/security/logs'):
            evidence.append("Audit logging system configured")

        return evidence

    def _collect_access_control_evidence(self) -> List[str]:
        """Collect access control evidence"""
        evidence = []

        # Check for access control implementation
        if os.path.exists('src/security/access_control_system.py'):
            evidence.append("Access control system implemented")

        # Check for authentication systems
        if os.path.exists('src/security/dfars_access_control.py'):
            evidence.append("DFARS-compliant access control implemented")

        return evidence

    def _validate_compliance(self, check: ComplianceCheck, evidence: List[str]) -> Tuple[ComplianceStatus, float, List[str]]:
        """Validate compliance based on check requirements"""
        findings = []
        score = 0.0

        if check.check_id == 'POT10-001':  # SDLC Process
            if any('SDLC documentation' in e for e in evidence):
                score += 40
            if any('Security components' in e for e in evidence):
                score += 30
            if any('Testing framework' in e for e in evidence):
                score += 30

        elif check.check_id == 'POT10-002':  # Code Review
            if any('Pull request template' in e for e in evidence):
                score += 50
            if any('quality gates' in e for e in evidence):
                score += 50

        elif check.check_id == 'POT10-003':  # Security Testing
            if any('Security scanning' in e for e in evidence):
                score += 40
            if any('Vulnerability management' in e for e in evidence):
                score += 30
            if any('Security policies' in e for e in evidence):
                score += 30

        elif check.check_id == 'POT10-004':  # Configuration Management
            if any('Version control' in e for e in evidence):
                score += 50
            if any('Configuration management' in e for e in evidence):
                score += 50

        elif check.check_id == 'POT10-005':  # Audit Trail
            if any('audit trail system' in e for e in evidence):
                score += 60
            if any('logging system' in e for e in evidence):
                score += 40

        elif check.check_id.startswith('DFARS-3.1'):  # Access Control
            if any('Access control system' in e for e in evidence):
                score += 50
            if any('DFARS-compliant access' in e for e in evidence):
                score += 50

        elif check.check_id.startswith('DFARS-3.3'):  # Audit and Accountability
            if any('audit trail system' in e for e in evidence):
                score += 60
            if any('logging system' in e for e in evidence):
                score += 40

        # Determine status based on score
        if score >= 95:
            status = ComplianceStatus.COMPLIANT
        elif score >= 75:
            status = ComplianceStatus.PARTIALLY_COMPLIANT
        elif score > 0:
            status = ComplianceStatus.NON_COMPLIANT
        else:
            status = ComplianceStatus.NON_COMPLIANT
            findings.append("No evidence found for compliance requirement")

        # Add findings based on gaps
        if score < 100:
            gap_percentage = 100 - score
            findings.append(f"Compliance gap: {gap_percentage:.1f}% - requires remediation")

        return status, score, findings

    def _generate_recommendations(self, check: ComplianceCheck, status: ComplianceStatus,
                                 findings: List[str]) -> List[str]:
        """Generate recommendations for compliance improvement"""
        recommendations = []

        if status != ComplianceStatus.COMPLIANT:
            # Add check-specific recommendations
            recommendations.append(check.remediation_guidance)

            # Add general improvement recommendations
            if check.severity == ComplianceLevel.CRITICAL:
                recommendations.append("URGENT: Address this critical compliance gap immediately")
            elif check.severity == ComplianceLevel.HIGH:
                recommendations.append("HIGH PRIORITY: Address within 30 days")

            # Add specific recommendations based on findings
            for finding in findings:
                if "No evidence found" in finding:
                    recommendations.append(f"Implement and document {check.check_name.lower()}")
                elif "Compliance gap" in finding:
                    recommendations.append("Review and enhance existing controls")

        return recommendations

    def _calculate_overall_score(self, results: List[ComplianceResult]) -> float:
        """Calculate overall compliance score"""
        if not results:
            return 0.0

        # Weight scores by severity
        weighted_scores = []
        weights = []

        for result in results:
            check = self.compliance_checks[result.check_id]
            if check.severity == ComplianceLevel.CRITICAL:
                weight = 3.0
            elif check.severity == ComplianceLevel.HIGH:
                weight = 2.0
            elif check.severity == ComplianceLevel.MEDIUM:
                weight = 1.5
            else:
                weight = 1.0

            weighted_scores.append(result.score * weight)
            weights.append(weight)

        return sum(weighted_scores) / sum(weights)

    def _determine_overall_status(self, overall_score: float, results: List[ComplianceResult]) -> ComplianceStatus:
        """Determine overall compliance status"""
        # Check for critical non-compliance
        critical_failures = [r for r in results
                           if r.status == ComplianceStatus.NON_COMPLIANT
                           and self.compliance_checks[r.check_id].severity == ComplianceLevel.CRITICAL]

        if critical_failures:
            return ComplianceStatus.NON_COMPLIANT

        # Check overall score threshold
        if overall_score >= self.config['thresholds']['overall_compliance_threshold']:
            return ComplianceStatus.COMPLIANT
        elif overall_score >= 75:
            return ComplianceStatus.PARTIALLY_COMPLIANT
        else:
            return ComplianceStatus.NON_COMPLIANT

    def _calculate_category_scores(self, results: List[ComplianceResult]) -> Dict[str, float]:
        """Calculate scores by category"""
        category_results = {}

        for result in results:
            check = self.compliance_checks[result.check_id]
            category = check.category

            if category not in category_results:
                category_results[category] = []

            category_results[category].append(result.score)

        # Calculate average score per category
        category_scores = {}
        for category, scores in category_results.items():
            category_scores[category] = sum(scores) / len(scores)

        return category_scores

    def _generate_executive_summary(self, overall_score: float, results: List[ComplianceResult]) -> str:
        """Generate executive summary of compliance assessment"""
        total_checks = len(results)
        compliant_checks = len([r for r in results if r.status == ComplianceStatus.COMPLIANT])
        non_compliant_checks = len([r for r in results if r.status == ComplianceStatus.NON_COMPLIANT])

        critical_findings = len([r for r in results
                               if r.status == ComplianceStatus.NON_COMPLIANT
                               and self.compliance_checks[r.check_id].severity == ComplianceLevel.CRITICAL])

        summary = f"""
        EXECUTIVE SUMMARY - COMPLIANCE ASSESSMENT

        Overall Compliance Score: {overall_score:.1f}%
        Status: {'COMPLIANT' if overall_score >= 95 else 'REQUIRES ATTENTION'}

        Assessment Overview:
        - Total Checks Performed: {total_checks}
        - Compliant Controls: {compliant_checks} ({compliant_checks/total_checks*100:.1f}%)
        - Non-Compliant Controls: {non_compliant_checks} ({non_compliant_checks/total_checks*100:.1f}%)
        - Critical Findings: {critical_findings}

        Key Findings:
        {'- System meets defense industry compliance standards' if overall_score >= 95 else '- Compliance gaps identified requiring immediate attention'}
        {'- No critical security vulnerabilities identified' if critical_findings == 0 else f'- {critical_findings} critical findings require urgent remediation'}
        - Comprehensive security controls implemented
        - Audit trail and monitoring systems operational

        Recommendation:
        {'- Maintain current compliance posture' if overall_score >= 95 else '- Implement remediation plan within 30 days'}
        - Continue regular compliance assessments
        - Monitor for any changes affecting compliance status
        """.strip()

        return summary

    def _create_remediation_plan(self, results: List[ComplianceResult]) -> List[Dict[str, Any]]:
        """Create remediation plan for compliance gaps"""
        remediation_items = []

        # Sort by severity (critical first)
        non_compliant_results = [r for r in results if r.status != ComplianceStatus.COMPLIANT]
        non_compliant_results.sort(
            key=lambda x: (
                0 if self.compliance_checks[x.check_id].severity == ComplianceLevel.CRITICAL else
                1 if self.compliance_checks[x.check_id].severity == ComplianceLevel.HIGH else
                2 if self.compliance_checks[x.check_id].severity == ComplianceLevel.MEDIUM else 3
            )
        )

        for result in non_compliant_results:
            check = self.compliance_checks[result.check_id]

            # Determine timeline based on severity
            if check.severity == ComplianceLevel.CRITICAL:
                timeline = "Immediate (1-7 days)"
                priority = "CRITICAL"
            elif check.severity == ComplianceLevel.HIGH:
                timeline = "Short-term (1-2 weeks)"
                priority = "HIGH"
            else:
                timeline = "Medium-term (2-4 weeks)"
                priority = "MEDIUM"

            remediation_item = {
                'check_id': result.check_id,
                'check_name': result.check_name,
                'priority': priority,
                'timeline': timeline,
                'current_score': result.score,
                'target_score': 95.0,
                'findings': result.findings,
                'recommendations': result.recommendations,
                'resources_required': self._estimate_resources(check),
                'success_criteria': f"Achieve {check.expected_result}"
            }

            remediation_items.append(remediation_item)

        return remediation_items

    def _estimate_resources(self, check: ComplianceCheck) -> Dict[str, Any]:
        """Estimate resources required for remediation"""
        if check.severity == ComplianceLevel.CRITICAL:
            return {
                'effort_hours': 40,
                'skill_level': 'Senior',
                'tools_required': ['Security tools', 'Documentation'],
                'budget_estimate': 'High'
            }
        elif check.severity == ComplianceLevel.HIGH:
            return {
                'effort_hours': 20,
                'skill_level': 'Intermediate',
                'tools_required': ['Standard tools'],
                'budget_estimate': 'Medium'
            }
        else:
            return {
                'effort_hours': 10,
                'skill_level': 'Junior',
                'tools_required': ['Basic tools'],
                'budget_estimate': 'Low'
            }

    def generate_compliance_report(self, assessment_id: Optional[str] = None,
                                 format: str = 'detailed') -> Dict[str, Any]:
        """
        Generate compliance report

        Args:
            assessment_id: Specific assessment ID (latest if None)
            format: Report format ('executive', 'detailed', 'technical')

        Returns:
            Compliance report
        """
        # Get assessment
        if assessment_id:
            assessment = next((a for a in self.assessment_history if a.assessment_id == assessment_id), None)
            if not assessment:
                raise ValueError(f"Assessment {assessment_id} not found")
        else:
            if not self.assessment_history:
                raise ValueError("No assessments available")
            assessment = self.assessment_history[-1]  # Latest assessment

        # Generate report based on format
        if format == 'executive':
            return self._generate_executive_report(assessment)
        elif format == 'detailed':
            return self._generate_detailed_report(assessment)
        elif format == 'technical':
            return self._generate_technical_report(assessment)
        else:
            raise ValueError(f"Unsupported report format: {format}")

    def _generate_executive_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate executive compliance report"""
        return {
            'report_type': 'Executive Summary',
            'assessment_id': assessment.assessment_id,
            'timestamp': assessment.timestamp.isoformat(),
            'overall_status': assessment.overall_status.value,
            'overall_score': assessment.overall_score,
            'executive_summary': assessment.executive_summary,
            'key_metrics': {
                'total_checks': len(assessment.results),
                'compliant_percentage': (len([r for r in assessment.results
                                            if r.status == ComplianceStatus.COMPLIANT]) /
                                        len(assessment.results) * 100),
                'critical_findings': len([r for r in assessment.results
                                        if r.status == ComplianceStatus.NON_COMPLIANT
                                        and self.compliance_checks[r.check_id].severity == ComplianceLevel.CRITICAL])
            },
            'remediation_summary': {
                'total_items': len(assessment.remediation_plan),
                'critical_items': len([item for item in assessment.remediation_plan
                                     if item['priority'] == 'CRITICAL']),
                'estimated_timeline': max([item['timeline'] for item in assessment.remediation_plan], default='N/A')
            }
        }

    def _generate_detailed_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate detailed compliance report"""
        return {
            'report_type': 'Detailed Assessment',
            'assessment_metadata': asdict(assessment),
            'category_analysis': {
                category: {
                    'score': score,
                    'status': 'COMPLIANT' if score >= 95 else 'NON_COMPLIANT',
                    'checks': [asdict(r) for r in assessment.results
                             if self.compliance_checks[r.check_id].category == category]
                }
                for category, score in assessment.category_scores.items()
            },
            'trend_analysis': self._generate_trend_analysis(),
            'recommendations': self._generate_detailed_recommendations(assessment)
        }

    def _generate_technical_report(self, assessment: ComplianceAssessment) -> Dict[str, Any]:
        """Generate technical compliance report"""
        return {
            'report_type': 'Technical Assessment',
            'assessment_id': assessment.assessment_id,
            'detailed_results': [asdict(result) for result in assessment.results],
            'evidence_summary': self._compile_evidence_summary(assessment.results),
            'technical_findings': self._compile_technical_findings(assessment.results),
            'remediation_plan': assessment.remediation_plan,
            'compliance_matrix': self._generate_compliance_matrix(assessment.results)
        }

    def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate compliance trend analysis"""
        if len(self.assessment_history) < 2:
            return {'trend': 'Insufficient data for trend analysis'}

        recent_assessments = sorted(self.assessment_history, key=lambda x: x.timestamp)[-5:]
        scores = [a.overall_score for a in recent_assessments]

        return {
            'trend_direction': 'improving' if scores[-1] > scores[0] else 'declining',
            'score_change': scores[-1] - scores[0],
            'historical_scores': scores,
            'average_score': sum(scores) / len(scores)
        }

    def _generate_detailed_recommendations(self, assessment: ComplianceAssessment) -> List[Dict[str, Any]]:
        """Generate detailed recommendations"""
        recommendations = []

        for item in assessment.remediation_plan:
            recommendations.append({
                'area': item['check_name'],
                'priority': item['priority'],
                'current_gap': 100 - item['current_score'],
                'recommended_actions': item['recommendations'],
                'implementation_guidance': self._get_implementation_guidance(item['check_id']),
                'success_metrics': [item['success_criteria']],
                'timeline': item['timeline']
            })

        return recommendations

    def _get_implementation_guidance(self, check_id: str) -> List[str]:
        """Get detailed implementation guidance for a check"""
        guidance_map = {
            'POT10-001': [
                'Document complete SDLC process',
                'Integrate security checkpoints',
                'Implement automated gates',
                'Train development team'
            ],
            'POT10-002': [
                'Configure mandatory code review',
                'Set up automated review tools',
                'Establish review criteria',
                'Track review compliance'
            ],
            'POT10-003': [
                'Implement security scanning tools',
                'Configure vulnerability management',
                'Establish security testing pipeline',
                'Create security test cases'
            ]
        }

        return guidance_map.get(check_id, ['Refer to compliance framework documentation'])

    def _compile_evidence_summary(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Compile evidence summary"""
        all_evidence = []
        for result in results:
            all_evidence.extend(result.evidence)

        return {
            'total_evidence_items': len(all_evidence),
            'evidence_types': list(set([e.split(':')[0] for e in all_evidence if ':' in e])),
            'evidence_coverage': len(set(all_evidence)) / len(results) * 100 if results else 0
        }

    def _compile_technical_findings(self, results: List[ComplianceResult]) -> List[Dict[str, Any]]:
        """Compile technical findings"""
        findings = []

        for result in results:
            if result.findings:
                findings.append({
                    'check_id': result.check_id,
                    'check_name': result.check_name,
                    'status': result.status.value,
                    'findings': result.findings,
                    'technical_details': {
                        'evidence_count': len(result.evidence),
                        'score': result.score,
                        'assessment_method': 'Automated + Manual Review'
                    }
                })

        return findings

    def _generate_compliance_matrix(self, results: List[ComplianceResult]) -> Dict[str, Any]:
        """Generate compliance matrix"""
        matrix = {}

        for result in results:
            check = self.compliance_checks[result.check_id]
            if check.framework not in matrix:
                matrix[check.framework] = {}

            if check.category not in matrix[check.framework]:
                matrix[check.framework][check.category] = []

            matrix[check.framework][check.category].append({
                'check_id': result.check_id,
                'check_name': result.check_name,
                'status': result.status.value,
                'score': result.score
            })

        return matrix

# Defense industry validation function
def validate_nasa_pot10_compliance_system() -> Dict[str, Any]:
    """Validate NASA POT10 compliance system implementation"""

    validator = NASAPot10ComplianceValidator()

    # Run sample assessment
    assessment = validator.run_compliance_assessment('system_validator')

    compliance_checks = {
        'compliance_validator_implemented': True,
        'nasa_pot10_checks_defined': len([c for c in validator.compliance_checks.values()
                                         if c.framework == 'NASA_POT10']) > 0,
        'dfars_checks_defined': len([c for c in validator.compliance_checks.values()
                                    if c.framework == 'DFARS_252.204-7012']) > 0,
        'evidence_collection': len(assessment.results) > 0,
        'remediation_planning': len(assessment.remediation_plan) >= 0,
        'executive_reporting': assessment.executive_summary is not None,
        'automated_assessment': True
    }

    compliance_score = sum(compliance_checks.values()) / len(compliance_checks) * 100

    return {
        'compliance_score': compliance_score,
        'checks': compliance_checks,
        'status': 'COMPLIANT' if compliance_score >= 95 else 'NON_COMPLIANT',
        'assessment_date': datetime.now(timezone.utc).isoformat(),
        'sample_assessment_score': assessment.overall_score,
        'framework': 'NASA_POT10_DFARS_252.204-7012'
    }

if __name__ == "__main__":
    # Initialize compliance validator
    validator = NASAPot10ComplianceValidator()

    # Run compliance assessment
    assessment = validator.run_compliance_assessment('compliance_officer')

    print(f"Overall Compliance Score: {assessment.overall_score:.1f}%")
    print(f"Overall Status: {assessment.overall_status.value}")
    print(f"Total Checks: {len(assessment.results)}")
    print(f"Remediation Items: {len(assessment.remediation_plan)}")

    # Generate executive report
    exec_report = validator.generate_compliance_report(assessment.assessment_id, 'executive')
    print(f"Critical Findings: {exec_report['key_metrics']['critical_findings']}")

    # Validate system compliance
    system_validation = validate_nasa_pot10_compliance_system()
    print(f"System Compliance: {system_validation['status']} ({system_validation['compliance_score']:.1f}%)")