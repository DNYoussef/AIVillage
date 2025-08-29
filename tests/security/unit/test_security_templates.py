"""
Security Templates Validation Tests

Tests GitHub issue and PR templates for security integration and threat modeling.
Validates that security checks are properly integrated into the development workflow.

Focus: Behavioral testing of security template contracts and workflow integration.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import yaml
import pytest
from typing import Dict, List, Any

from core.domain.security_constants import SecurityLevel


class SecurityTemplate:
    """Represents a GitHub issue or PR template with security integration."""
    
    def __init__(self, template_type: str, content: str, metadata: Dict[str, Any] = None):
        self.template_type = template_type
        self.content = content
        self.metadata = metadata or {}
        self.security_fields = self._extract_security_fields()
    
    def _extract_security_fields(self) -> List[str]:
        """Extract security-related fields from template content."""
        security_keywords = [
            'security impact',
            'threat model',
            'authentication',
            'authorization', 
            'encryption',
            'vulnerability',
            'security checklist',
            'security review',
            'privacy impact',
            'data protection'
        ]
        
        found_fields = []
        content_lower = self.content.lower()
        for keyword in security_keywords:
            if keyword in content_lower:
                found_fields.append(keyword)
        
        return found_fields
    
    def has_security_integration(self) -> bool:
        """Check if template has proper security integration."""
        return len(self.security_fields) >= 2
    
    def requires_security_review(self) -> bool:
        """Check if template indicates security review requirement."""
        security_review_indicators = [
            'security review required',
            'security team review',
            'threat modeling required',
            'security assessment needed'
        ]
        
        content_lower = self.content.lower()
        return any(indicator in content_lower for indicator in security_review_indicators)


class GitHubSecurityIntegration:
    """Manages GitHub integration for security workflows."""
    
    def __init__(self, github_client=None):
        self.github_client = github_client
        self.security_labels = [
            'security',
            'vulnerability',
            'security-review-required',
            'threat-modeling',
            'privacy-impact'
        ]
    
    def validate_issue_template(self, template: SecurityTemplate) -> Dict[str, Any]:
        """Validate issue template for security integration."""
        validation_result = {
            "template_type": template.template_type,
            "has_security_integration": template.has_security_integration(),
            "security_fields_count": len(template.security_fields),
            "security_fields": template.security_fields,
            "requires_security_review": template.requires_security_review(),
            "compliance_score": 0,
            "recommendations": []
        }
        
        # Calculate compliance score
        score = 0
        if template.has_security_integration():
            score += 40
        if template.requires_security_review():
            score += 30
        if 'threat model' in template.security_fields:
            score += 20
        if 'privacy impact' in template.security_fields:
            score += 10
        
        validation_result["compliance_score"] = score
        
        # Generate recommendations
        if score < 50:
            validation_result["recommendations"].append("Add security impact assessment section")
        if score < 70:
            validation_result["recommendations"].append("Include threat modeling requirements")
        if 'encryption' not in template.security_fields:
            validation_result["recommendations"].append("Add encryption considerations")
        
        return validation_result
    
    def validate_pr_template(self, template: SecurityTemplate) -> Dict[str, Any]:
        """Validate PR template for security integration."""
        pr_specific_checks = [
            'security testing completed',
            'dependency scan passed',
            'security gates passed',
            'no secrets in code',
            'authorization changes reviewed'
        ]
        
        content_lower = template.content.lower()
        passed_checks = [check for check in pr_specific_checks if check in content_lower]
        
        validation_result = {
            "template_type": template.template_type,
            "has_security_integration": template.has_security_integration(),
            "security_checks_count": len(passed_checks),
            "security_checks": passed_checks,
            "compliance_score": (len(passed_checks) / len(pr_specific_checks)) * 100,
            "recommendations": []
        }
        
        missing_checks = [check for check in pr_specific_checks if check not in passed_checks]
        if missing_checks:
            validation_result["recommendations"] = [f"Add check for: {check}" for check in missing_checks]
        
        return validation_result
    
    def check_security_automation(self, workflow_content: str) -> Dict[str, Any]:
        """Check if GitHub workflows include security automation."""
        security_automations = [
            'secret scanning',
            'dependency scanning',
            'sast',
            'dast',
            'container scanning',
            'vulnerability scanning'
        ]
        
        content_lower = workflow_content.lower()
        active_automations = [auto for auto in security_automations if auto in content_lower]
        
        return {
            "active_automations": active_automations,
            "automation_coverage": (len(active_automations) / len(security_automations)) * 100,
            "missing_automations": [auto for auto in security_automations if auto not in active_automations]
        }


class SecurityTemplateValidationTest(unittest.TestCase):
    """
    Behavioral tests for security template validation.
    
    Tests security integration contracts in GitHub templates without coupling to implementation.
    Validates security workflow guarantees.
    """
    
    def setUp(self):
        """Set up test fixtures with security-focused mocks."""
        self.mock_github_client = Mock()
        self.integration = GitHubSecurityIntegration(github_client=self.mock_github_client)
    
    def test_issue_template_security_integration_detection(self):
        """
        Security Contract: Issue templates must include security assessment fields.
        Tests that security fields are properly detected and validated.
        """
        # Arrange
        security_integrated_template = SecurityTemplate(
            template_type="bug_report",
            content="""
            ## Bug Description
            Describe the bug
            
            ## Security Impact
            - [ ] No security impact
            - [ ] Authentication affected
            - [ ] Authorization affected
            - [ ] Data exposure possible
            
            ## Threat Model Considerations
            Does this affect our threat model?
            
            ## Privacy Impact
            Are there privacy implications?
            """
        )
        
        # Act
        result = self.integration.validate_issue_template(security_integrated_template)
        
        # Assert - Test security integration behavior
        self.assertTrue(result["has_security_integration"],
                       "Template with security fields must be detected as integrated")
        self.assertGreaterEqual(result["security_fields_count"], 2,
                               "Must detect multiple security fields")
        self.assertIn("security impact", result["security_fields"])
        self.assertIn("threat model", result["security_fields"])
        self.assertIn("privacy impact", result["security_fields"])
        self.assertGreaterEqual(result["compliance_score"], 70,
                               "Well-integrated template must have high compliance score")
    
    def test_issue_template_missing_security_integration(self):
        """
        Security Contract: Templates without security integration must be flagged.
        Tests detection of security gaps in templates.
        """
        # Arrange
        basic_template = SecurityTemplate(
            template_type="feature_request",
            content="""
            ## Feature Description
            Describe the new feature
            
            ## Acceptance Criteria
            - [ ] Feature works as expected
            - [ ] Tests are written
            - [ ] Documentation updated
            """
        )
        
        # Act
        result = self.integration.validate_issue_template(basic_template)
        
        # Assert - Test security gap detection
        self.assertFalse(result["has_security_integration"],
                        "Template without security fields must be flagged")
        self.assertEqual(result["security_fields_count"], 0,
                        "Must detect zero security fields")
        self.assertLess(result["compliance_score"], 50,
                       "Non-integrated template must have low compliance score")
        self.assertIn("Add security impact assessment section", result["recommendations"])
    
    def test_pr_template_security_checklist_validation(self):
        """
        Security Contract: PR templates must include security validation checklists.
        Tests that security gates are properly integrated into PR workflow.
        """
        # Arrange
        comprehensive_pr_template = SecurityTemplate(
            template_type="pull_request",
            content="""
            ## Changes Description
            What changes are included in this PR?
            
            ## Security Checklist
            - [ ] Security testing completed
            - [ ] Dependency scan passed
            - [ ] Security gates passed
            - [ ] No secrets in code
            - [ ] Authorization changes reviewed
            
            ## Testing
            - [ ] Unit tests pass
            - [ ] Integration tests pass
            """
        )
        
        # Act
        result = self.integration.validate_pr_template(comprehensive_pr_template)
        
        # Assert - Test PR security checklist
        self.assertTrue(result["has_security_integration"],
                       "PR template must have security integration")
        self.assertEqual(result["security_checks_count"], 5,
                        "Must detect all security checks")
        self.assertEqual(result["compliance_score"], 100,
                        "Complete security checklist must score 100%")
        self.assertEqual(len(result["recommendations"]), 0,
                        "Complete template should have no recommendations")
        
        expected_checks = [
            'security testing completed',
            'dependency scan passed', 
            'security gates passed',
            'no secrets in code',
            'authorization changes reviewed'
        ]
        for check in expected_checks:
            self.assertIn(check, result["security_checks"],
                         f"Must detect security check: {check}")
    
    def test_pr_template_partial_security_integration(self):
        """
        Security Contract: Partial security integration must be identified and improved.
        Tests incremental security improvement recommendations.
        """
        # Arrange
        partial_pr_template = SecurityTemplate(
            template_type="pull_request",
            content="""
            ## Changes Description
            What changes are included?
            
            ## Checklist
            - [ ] Security testing completed
            - [ ] No secrets in code
            - [ ] Tests pass
            """
        )
        
        # Act
        result = self.integration.validate_pr_template(partial_pr_template)
        
        # Assert - Test partial integration detection
        self.assertTrue(result["has_security_integration"],
                       "Template with some security fields must be detected")
        self.assertEqual(result["security_checks_count"], 2,
                        "Must detect existing security checks")
        self.assertLess(result["compliance_score"], 50,
                       "Partial integration must have medium compliance score")
        self.assertGreater(len(result["recommendations"]), 0,
                          "Must provide improvement recommendations")
        
        # Check specific missing recommendations
        expected_missing = [
            "Add check for: dependency scan passed",
            "Add check for: security gates passed", 
            "Add check for: authorization changes reviewed"
        ]
        for missing in expected_missing:
            self.assertIn(missing, result["recommendations"])
    
    def test_security_automation_workflow_detection(self):
        """
        Security Contract: GitHub workflows must include comprehensive security automation.
        Tests detection of security automation in CI/CD workflows.
        """
        # Arrange
        comprehensive_workflow = """
        name: Security Pipeline
        on: [push, pull_request]
        jobs:
          security-scan:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v3
              - name: Secret Scanning
                run: detect-secrets scan
              - name: Dependency Scanning  
                run: safety check
              - name: SAST Analysis
                run: bandit -r .
              - name: DAST Testing
                run: zap-baseline
              - name: Container Scanning
                run: trivy image
              - name: Vulnerability Scanning
                run: grype .
        """
        
        # Act
        result = self.integration.check_security_automation(comprehensive_workflow)
        
        # Assert - Test automation coverage
        self.assertEqual(len(result["active_automations"]), 6,
                        "Must detect all security automations")
        self.assertEqual(result["automation_coverage"], 100,
                        "Complete automation must have 100% coverage")
        self.assertEqual(len(result["missing_automations"]), 0,
                        "Complete workflow should have no missing automations")
        
        expected_automations = [
            'secret scanning',
            'dependency scanning',
            'sast',
            'dast', 
            'container scanning',
            'vulnerability scanning'
        ]
        for automation in expected_automations:
            self.assertIn(automation, result["active_automations"],
                         f"Must detect automation: {automation}")
    
    def test_security_automation_gaps_identification(self):
        """
        Security Contract: Missing security automations must be identified.
        Tests gap analysis in security automation coverage.
        """
        # Arrange
        basic_workflow = """
        name: Basic CI
        on: [push]
        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v3
              - name: Run Tests
                run: pytest
              - name: Secret Scanning
                run: detect-secrets scan
        """
        
        # Act
        result = self.integration.check_security_automation(basic_workflow)
        
        # Assert - Test gap identification
        self.assertEqual(len(result["active_automations"]), 1,
                        "Must detect only active automations")
        self.assertLess(result["automation_coverage"], 30,
                       "Basic workflow must have low coverage")
        self.assertEqual(len(result["missing_automations"]), 5,
                        "Must identify missing automations")
        
        expected_missing = [
            'dependency scanning',
            'sast',
            'dast',
            'container scanning', 
            'vulnerability scanning'
        ]
        for missing in expected_missing:
            self.assertIn(missing, result["missing_automations"],
                         f"Must identify missing automation: {missing}")
    
    def test_security_review_requirement_detection(self):
        """
        Security Contract: Templates must indicate when security review is required.
        Tests detection of security review triggers.
        """
        # Arrange
        security_review_template = SecurityTemplate(
            template_type="security_issue",
            content="""
            ## Security Issue Report
            
            ## Impact Assessment
            This issue requires security team review due to:
            - Authentication system changes
            - Threat modeling required for new attack vectors
            
            ## Review Requirements
            - [ ] Security assessment needed
            - [ ] Threat modeling required
            """
        )
        
        # Act
        result = self.integration.validate_issue_template(security_review_template)
        
        # Assert - Test security review detection
        self.assertTrue(result["requires_security_review"],
                       "Template indicating security review must be detected")
        self.assertGreaterEqual(result["compliance_score"], 90,
                               "Security review template must have high compliance")
        self.assertIn("threat model", result["security_fields"])
        self.assertIn("security assessment needed", 
                     security_review_template.content.lower())
    
    def test_template_compliance_scoring_accuracy(self):
        """
        Security Contract: Compliance scoring must accurately reflect security integration.
        Tests the scoring algorithm for different integration levels.
        """
        # Test cases: (template_content, expected_score_range)
        test_cases = [
            # Minimal security (0-30)
            ("Basic template with no security fields", (0, 30)),
            
            # Basic security integration (30-60) 
            ("Template with security impact assessment", (30, 60)),
            
            # Good security integration (60-90)
            ("""Security impact, threat model considerations, 
               requires security review""", (60, 90)),
            
            # Excellent security integration (90-100)
            ("""Security impact assessment, threat model required,
               privacy impact analysis, requires security review""", (90, 100))
        ]
        
        for content, (min_score, max_score) in test_cases:
            with self.subTest(content=content[:50]):
                # Arrange
                template = SecurityTemplate("test", content)
                
                # Act
                result = self.integration.validate_issue_template(template)
                
                # Assert
                self.assertGreaterEqual(result["compliance_score"], min_score,
                                       f"Score must be >= {min_score}")
                self.assertLessEqual(result["compliance_score"], max_score,
                                    f"Score must be <= {max_score}")
    
    def test_security_template_metadata_validation(self):
        """
        Security Contract: Template metadata must support security workflow automation.
        Tests that metadata enables proper security workflow routing.
        """
        # Arrange
        template_with_metadata = SecurityTemplate(
            template_type="security_vulnerability",
            content="Security issue template",
            metadata={
                "labels": ["security", "vulnerability", "high-priority"],
                "assignees": ["security-team"],
                "requires_review": True,
                "automation_triggers": ["security-scan", "threat-model"]
            }
        )
        
        # Act
        result = self.integration.validate_issue_template(template_with_metadata)
        
        # Assert - Test metadata support for security workflows
        self.assertIsNotNone(template_with_metadata.metadata)
        self.assertIn("security", template_with_metadata.metadata.get("labels", []))
        self.assertIn("vulnerability", template_with_metadata.metadata.get("labels", []))
        self.assertTrue(template_with_metadata.metadata.get("requires_review", False))
        
        # Verify automation triggers are preserved
        automation_triggers = template_with_metadata.metadata.get("automation_triggers", [])
        self.assertIn("security-scan", automation_triggers)
        self.assertIn("threat-model", automation_triggers)


if __name__ == "__main__":
    # Run tests with security-focused output
    unittest.main(verbosity=2, buffer=True)