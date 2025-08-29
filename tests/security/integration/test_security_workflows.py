"""
Security Workflow Integration Tests

Tests end-to-end security workflows including vulnerability reporting to resolution,
SBOM generation to deployment, and admin security processes.

Focus: Integration testing of security workflow contracts and process validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio

# Import security components for integration testing
from tests.security.unit.test_vulnerability_reporting import (
    VulnerabilityReportingWorkflow, VulnerabilityReport, VulnerabilitySeverity
)
from tests.security.unit.test_dependency_auditing import (
    DependencyScanner, DependencyEcosystem, VulnerabilityResult as DepVulnerabilityResult
)
from tests.security.unit.test_sbom_generation import (
    SBOMGenerator, SBOMFormat, CryptographicSigner, SBOMComponent, ComponentType
)
from tests.security.unit.test_admin_security import (
    AdminInterfaceServer, AdminAccessLevel
)

from core.domain.security_constants import SecurityLevel


class SecurityWorkflowOrchestrator:
    """Orchestrates end-to-end security workflows and processes."""
    
    def __init__(self):
        self.vulnerability_workflow = VulnerabilityReportingWorkflow()
        self.dependency_scanner = DependencyScanner()
        self.sbom_generator = SBOMGenerator()
        self.admin_server = AdminInterfaceServer(require_mfa=True)
        self.workflow_state = {}
        self.integration_audit_log = []
    
    def execute_vulnerability_to_resolution_workflow(self, 
                                                   vulnerability_report: VulnerabilityReport) -> Dict[str, Any]:
        """Execute complete vulnerability workflow from report to resolution."""
        workflow_id = f"vuln-workflow-{int(time.time())}"
        
        workflow_result = {
            "workflow_id": workflow_id,
            "workflow_type": "vulnerability_resolution",
            "start_time": datetime.utcnow().isoformat(),
            "stages_completed": [],
            "current_stage": "initial_processing",
            "overall_status": "in_progress"
        }
        
        try:
            # Stage 1: Process vulnerability report
            report_result = self.vulnerability_workflow.process_report(vulnerability_report)
            workflow_result["stages_completed"].append({
                "stage": "report_processing",
                "status": "completed",
                "result": report_result
            })
            
            # Stage 2: Trigger dependency scanning if applicable
            if self._should_trigger_dependency_scan(vulnerability_report):
                workflow_result["current_stage"] = "dependency_scanning"
                
                ecosystem = self._determine_ecosystem(vulnerability_report)
                scan_result = self.dependency_scanner.scan_ecosystem(ecosystem, "requirements.txt")
                
                workflow_result["stages_completed"].append({
                    "stage": "dependency_scanning", 
                    "status": "completed",
                    "result": {
                        "vulnerabilities_found": scan_result["vulnerabilities_found"],
                        "risk_level": scan_result["risk_summary"]["risk_level"]
                    }
                })
            
            # Stage 3: Generate updated SBOM
            workflow_result["current_stage"] = "sbom_generation"
            
            # Add affected components to SBOM
            self._add_vulnerability_components_to_sbom(vulnerability_report)
            sbom_content = self.sbom_generator.generate_sbom({
                "name": f"Security-Update-{workflow_id}",
                "vulnerability_context": True
            })
            
            signed_sbom = self.sbom_generator.sign_sbom(sbom_content)
            
            workflow_result["stages_completed"].append({
                "stage": "sbom_generation",
                "status": "completed", 
                "result": {
                    "sbom_signed": signed_sbom["signed"],
                    "component_count": sbom_content["component_count"]
                }
            })
            
            # Stage 4: Security review and approval
            workflow_result["current_stage"] = "security_review"
            
            if vulnerability_report.severity == VulnerabilitySeverity.CRITICAL:
                # Critical vulnerabilities require admin approval
                review_result = self._require_admin_approval(workflow_id, vulnerability_report)
                workflow_result["stages_completed"].append({
                    "stage": "security_review",
                    "status": "completed",
                    "result": review_result
                })
            
            # Stage 5: Resolution and cleanup
            workflow_result["current_stage"] = "resolution"
            workflow_result["overall_status"] = "completed"
            workflow_result["end_time"] = datetime.utcnow().isoformat()
            
            # Log successful workflow completion
            self._audit_log("workflow_completed", {
                "workflow_id": workflow_id,
                "vulnerability_id": vulnerability_report.report_id,
                "severity": vulnerability_report.severity.value,
                "total_stages": len(workflow_result["stages_completed"])
            })
            
        except Exception as e:
            workflow_result["overall_status"] = "failed"
            workflow_result["error"] = str(e)
            workflow_result["end_time"] = datetime.utcnow().isoformat()
            
            self._audit_log("workflow_failed", {
                "workflow_id": workflow_id,
                "error": str(e),
                "stage": workflow_result["current_stage"]
            })
        
        self.workflow_state[workflow_id] = workflow_result
        return workflow_result
    
    def _should_trigger_dependency_scan(self, vulnerability_report: VulnerabilityReport) -> bool:
        """Determine if vulnerability report should trigger dependency scanning."""
        dependency_triggers = [
            "dependency" in vulnerability_report.description.lower(),
            "library" in vulnerability_report.description.lower(),
            "package" in vulnerability_report.description.lower(),
            any(comp in ["Core System", "P2P Network"] for comp in vulnerability_report.affected_components)
        ]
        return any(dependency_triggers)
    
    def _determine_ecosystem(self, vulnerability_report: VulnerabilityReport) -> DependencyEcosystem:
        """Determine dependency ecosystem from vulnerability report."""
        description_lower = vulnerability_report.description.lower()
        
        if any(keyword in description_lower for keyword in ["python", "pip", "pypi"]):
            return DependencyEcosystem.PYTHON
        elif any(keyword in description_lower for keyword in ["javascript", "npm", "node"]):
            return DependencyEcosystem.JAVASCRIPT
        elif any(keyword in description_lower for keyword in ["rust", "cargo"]):
            return DependencyEcosystem.RUST
        else:
            return DependencyEcosystem.PYTHON  # Default
    
    def _add_vulnerability_components_to_sbom(self, vulnerability_report: VulnerabilityReport):
        """Add vulnerability-related components to SBOM."""
        for component_name in vulnerability_report.affected_components:
            # Create security-focused component entry
            component = SBOMComponent(
                name=component_name.lower().replace(" ", "-"),
                version="security-review",
                component_type=ComponentType.LIBRARY,
                supplier="Internal-Security-Review"
            )
            self.sbom_generator.add_component(component)
    
    def _require_admin_approval(self, workflow_id: str, 
                               vulnerability_report: VulnerabilityReport) -> Dict[str, Any]:
        """Require admin approval for critical vulnerabilities."""
        # Start admin server if not running
        if not self.admin_server.is_running:
            self.admin_server.start_server()
        
        # Mock admin approval process
        approval_request = {
            "workflow_id": workflow_id,
            "vulnerability_id": vulnerability_report.report_id,
            "severity": vulnerability_report.severity.value,
            "approval_required": True,
            "approval_timestamp": datetime.utcnow().isoformat(),
            "approved_by": "security_admin"  # Mock approval
        }
        
        return approval_request
    
    def execute_dependency_to_sbom_workflow(self, ecosystems: List[DependencyEcosystem]) -> Dict[str, Any]:
        """Execute workflow from dependency scanning to SBOM generation."""
        workflow_id = f"dep-sbom-{int(time.time())}"
        
        workflow_result = {
            "workflow_id": workflow_id,
            "workflow_type": "dependency_to_sbom",
            "start_time": datetime.utcnow().isoformat(),
            "ecosystems_processed": [],
            "total_dependencies": 0,
            "total_vulnerabilities": 0,
            "sbom_generated": False
        }
        
        try:
            # Stage 1: Scan all ecosystems
            all_components = []
            
            for ecosystem in ecosystems:
                scan_result = self.dependency_scanner.scan_ecosystem(
                    ecosystem, f"{ecosystem.value}_dependencies"
                )
                
                workflow_result["ecosystems_processed"].append({
                    "ecosystem": ecosystem.value,
                    "dependencies": scan_result["dependencies_scanned"],
                    "vulnerabilities": scan_result["vulnerabilities_found"]
                })
                
                workflow_result["total_dependencies"] += scan_result["dependencies_scanned"]
                workflow_result["total_vulnerabilities"] += scan_result["vulnerabilities_found"]
                
                # Convert scan results to SBOM components
                ecosystem_components = self._convert_scan_to_components(scan_result, ecosystem)
                all_components.extend(ecosystem_components)
            
            # Stage 2: Generate comprehensive SBOM
            for component in all_components:
                self.sbom_generator.add_component(component)
            
            sbom_content = self.sbom_generator.generate_sbom({
                "name": f"Dependency-Scan-SBOM-{workflow_id}",
                "scan_timestamp": datetime.utcnow().isoformat(),
                "ecosystems": [eco.value for eco in ecosystems]
            })
            
            # Stage 3: Sign SBOM
            signed_sbom = self.sbom_generator.sign_sbom(sbom_content)
            workflow_result["sbom_generated"] = signed_sbom["signed"]
            workflow_result["sbom_components"] = len(all_components)
            
            workflow_result["overall_status"] = "completed"
            workflow_result["end_time"] = datetime.utcnow().isoformat()
            
            self._audit_log("dependency_sbom_workflow_completed", {
                "workflow_id": workflow_id,
                "ecosystems": len(ecosystems),
                "total_components": len(all_components),
                "vulnerabilities_found": workflow_result["total_vulnerabilities"]
            })
            
        except Exception as e:
            workflow_result["overall_status"] = "failed"
            workflow_result["error"] = str(e)
        
        return workflow_result
    
    def _convert_scan_to_components(self, scan_result: Dict[str, Any], 
                                  ecosystem: DependencyEcosystem) -> List[SBOMComponent]:
        """Convert dependency scan results to SBOM components."""
        components = []
        
        # Mock conversion - in real implementation would extract from scan_result
        mock_dependencies = {
            DependencyEcosystem.PYTHON: [
                ("requests", "2.28.1", "Apache-2.0"),
                ("flask", "2.3.0", "BSD-3-Clause"),
                ("cryptography", "38.0.1", "Apache-2.0")
            ],
            DependencyEcosystem.JAVASCRIPT: [
                ("express", "4.18.1", "MIT"),
                ("lodash", "4.17.21", "MIT")
            ]
        }
        
        for name, version, license_info in mock_dependencies.get(ecosystem, []):
            component = SBOMComponent(
                name=name,
                version=version,
                component_type=ComponentType.LIBRARY,
                license_info=license_info,
                package_url=f"pkg:{ecosystem.value}/{name}@{version}"
            )
            components.append(component)
        
        return components
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow."""
        if workflow_id not in self.workflow_state:
            return {"error": "workflow_not_found"}
        
        return self.workflow_state[workflow_id].copy()
    
    def _audit_log(self, event_type: str, event_data: Dict[str, Any]):
        """Log workflow integration events."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": event_data
        }
        self.integration_audit_log.append(log_entry)


class SecurityWorkflowIntegrationTest(unittest.TestCase):
    """
    Integration tests for security workflows.
    
    Tests end-to-end security workflow integration and cross-component behavior
    without coupling to implementation details.
    """
    
    def setUp(self):
        """Set up test fixtures for workflow integration testing."""
        self.workflow_orchestrator = SecurityWorkflowOrchestrator()
        
        # Create test vulnerability reports
        self.critical_vulnerability = VulnerabilityReport(
            severity=VulnerabilitySeverity.CRITICAL,
            classification="Injection",
            affected_components=["Core System", "API Gateway"],
            description="SQL injection vulnerability in user authentication Python library allows privilege escalation"
        )
        
        self.dependency_vulnerability = VulnerabilityReport(
            severity=VulnerabilitySeverity.HIGH,
            classification="Cryptographic Issues", 
            affected_components=["P2P Network"],
            description="Weak key generation in JavaScript cryptography package affects network security"
        )
    
    def test_vulnerability_to_resolution_workflow_integration(self):
        """
        Security Contract: Vulnerability reports must flow through complete resolution workflow.
        Tests integration of vulnerability reporting, scanning, SBOM generation, and approval.
        """
        # Act
        workflow_result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
            self.critical_vulnerability
        )
        
        # Assert - Workflow completion
        self.assertEqual(workflow_result["overall_status"], "completed",
                        "Critical vulnerability workflow must complete successfully")
        self.assertIn("workflow_id", workflow_result,
                     "Workflow must have unique identifier")
        
        # Verify workflow stages
        stages_completed = workflow_result["stages_completed"]
        expected_stages = ["report_processing", "dependency_scanning", "sbom_generation", "security_review"]
        
        completed_stage_names = [stage["stage"] for stage in stages_completed]
        for expected_stage in expected_stages:
            self.assertIn(expected_stage, completed_stage_names,
                         f"Workflow must complete {expected_stage} stage")
        
        # Verify critical vulnerability requires security review
        security_review_stage = next(
            stage for stage in stages_completed if stage["stage"] == "security_review"
        )
        self.assertEqual(security_review_stage["status"], "completed",
                        "Security review must be completed for critical vulnerabilities")
        
        # Verify SBOM generation integration
        sbom_stage = next(
            stage for stage in stages_completed if stage["stage"] == "sbom_generation"
        )
        self.assertTrue(sbom_stage["result"]["sbom_signed"],
                       "SBOM must be cryptographically signed")
    
    def test_dependency_scanning_triggers_sbom_update(self):
        """
        Security Contract: Dependency vulnerability findings must trigger SBOM updates.
        Tests integration between dependency scanning and SBOM generation.
        """
        # Act
        workflow_result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
            self.dependency_vulnerability
        )
        
        # Assert - Dependency scanning integration
        self.assertEqual(workflow_result["overall_status"], "completed",
                        "Dependency vulnerability workflow must complete")
        
        stages = workflow_result["stages_completed"]
        stage_names = [stage["stage"] for stage in stages]
        
        self.assertIn("dependency_scanning", stage_names,
                     "Dependency vulnerability must trigger scanning")
        self.assertIn("sbom_generation", stage_names,
                     "Dependency scanning must trigger SBOM generation")
        
        # Verify dependency scan results are integrated
        dep_scan_stage = next(
            stage for stage in stages if stage["stage"] == "dependency_scanning"
        )
        self.assertIn("vulnerabilities_found", dep_scan_stage["result"],
                     "Dependency scan results must be captured")
        self.assertIn("risk_level", dep_scan_stage["result"],
                     "Risk assessment must be included")
    
    def test_multi_ecosystem_dependency_to_sbom_workflow(self):
        """
        Security Contract: Multi-ecosystem scanning must produce comprehensive SBOM.
        Tests integration across multiple dependency ecosystems.
        """
        # Arrange
        ecosystems = [
            DependencyEcosystem.PYTHON,
            DependencyEcosystem.JAVASCRIPT,
            DependencyEcosystem.RUST
        ]
        
        # Act
        workflow_result = self.workflow_orchestrator.execute_dependency_to_sbom_workflow(ecosystems)
        
        # Assert - Multi-ecosystem integration
        self.assertEqual(workflow_result["overall_status"], "completed",
                        "Multi-ecosystem workflow must complete successfully")
        self.assertTrue(workflow_result["sbom_generated"],
                       "Comprehensive SBOM must be generated")
        
        # Verify all ecosystems processed
        ecosystems_processed = workflow_result["ecosystems_processed"]
        self.assertEqual(len(ecosystems_processed), len(ecosystems),
                        "All ecosystems must be processed")
        
        processed_ecosystem_names = [eco["ecosystem"] for eco in ecosystems_processed]
        for expected_ecosystem in ecosystems:
            self.assertIn(expected_ecosystem.value, processed_ecosystem_names,
                         f"Ecosystem {expected_ecosystem.value} must be processed")
        
        # Verify aggregated results
        self.assertGreater(workflow_result["total_dependencies"], 0,
                          "Must aggregate dependencies across ecosystems")
        self.assertGreater(workflow_result["sbom_components"], 0,
                          "SBOM must contain components from all ecosystems")
    
    def test_critical_vulnerability_admin_approval_integration(self):
        """
        Security Contract: Critical vulnerabilities must integrate with admin approval workflow.
        Tests integration between vulnerability processing and admin security.
        """
        # Act
        workflow_result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
            self.critical_vulnerability
        )
        
        # Assert - Admin approval integration
        self.assertEqual(workflow_result["overall_status"], "completed",
                        "Critical vulnerability with admin approval must complete")
        
        # Find security review stage
        security_review_stage = None
        for stage in workflow_result["stages_completed"]:
            if stage["stage"] == "security_review":
                security_review_stage = stage
                break
        
        self.assertIsNotNone(security_review_stage,
                           "Critical vulnerability must have security review stage")
        
        # Verify admin approval details
        approval_result = security_review_stage["result"]
        self.assertTrue(approval_result["approval_required"],
                       "Critical vulnerability must require approval")
        self.assertIn("approved_by", approval_result,
                     "Approval must include approver identity")
        self.assertIn("approval_timestamp", approval_result,
                     "Approval must be timestamped")
    
    def test_workflow_audit_logging_integration(self):
        """
        Security Contract: All workflow operations must be comprehensively audit logged.
        Tests audit logging integration across workflow components.
        """
        # Act - Execute multiple workflows
        workflow1 = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
            self.critical_vulnerability
        )
        
        workflow2 = self.workflow_orchestrator.execute_dependency_to_sbom_workflow([
            DependencyEcosystem.PYTHON
        ])
        
        # Assert - Audit logging integration
        audit_log = self.workflow_orchestrator.integration_audit_log
        self.assertGreater(len(audit_log), 0,
                          "Workflow operations must be audit logged")
        
        # Verify log entries for both workflows
        workflow_events = [log for log in audit_log if "workflow" in log["event_type"]]
        self.assertGreaterEqual(len(workflow_events), 2,
                               "Both workflows must generate audit events")
        
        # Check audit log structure
        for log_entry in workflow_events:
            required_fields = ["timestamp", "event_type", "data"]
            for field in required_fields:
                self.assertIn(field, log_entry,
                             f"Audit log entry must include {field}")
    
    def test_workflow_state_tracking_and_retrieval(self):
        """
        Security Contract: Workflow states must be trackable and retrievable.
        Tests workflow state management and status tracking.
        """
        # Act
        workflow_result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
            self.dependency_vulnerability
        )
        
        workflow_id = workflow_result["workflow_id"]
        
        # Test workflow status retrieval
        status = self.workflow_orchestrator.get_workflow_status(workflow_id)
        
        # Assert - State tracking
        self.assertNotEqual(status.get("error"), "workflow_not_found",
                          "Executed workflow must be trackable")
        self.assertEqual(status["workflow_id"], workflow_id,
                        "Status must match workflow ID")
        self.assertIn("overall_status", status,
                     "Status must include overall status")
        
        # Test non-existent workflow
        invalid_status = self.workflow_orchestrator.get_workflow_status("non-existent-id")
        self.assertEqual(invalid_status["error"], "workflow_not_found",
                        "Non-existent workflow must return error")
    
    def test_workflow_error_handling_and_recovery(self):
        """
        Security Contract: Workflow failures must be handled gracefully with audit trails.
        Tests error handling and recovery mechanisms in workflows.
        """
        # Arrange - Create invalid vulnerability report to trigger errors
        invalid_vulnerability = VulnerabilityReport(
            severity=VulnerabilitySeverity.HIGH,
            classification=None,  # Missing classification
            affected_components=[],  # Empty components
            description=""  # Empty description
        )
        
        # Mock dependency scanner to raise exception
        with patch.object(self.workflow_orchestrator.dependency_scanner, 'scan_ecosystem') as mock_scan:
            mock_scan.side_effect = Exception("Mock dependency scan failure")
            
            # Act
            workflow_result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(
                invalid_vulnerability
            )
            
            # Assert - Error handling
            self.assertEqual(workflow_result["overall_status"], "failed",
                            "Workflow with errors must have failed status")
            self.assertIn("error", workflow_result,
                         "Failed workflow must include error information")
            self.assertIn("end_time", workflow_result,
                         "Failed workflow must record completion time")
            
            # Verify error is audit logged
            audit_log = self.workflow_orchestrator.integration_audit_log
            error_events = [log for log in audit_log if "failed" in log["event_type"]]
            self.assertGreater(len(error_events), 0,
                              "Workflow failures must be audit logged")
    
    def test_workflow_performance_and_scalability(self):
        """
        Security Contract: Workflows must complete within reasonable time limits.
        Tests workflow performance and scalability characteristics.
        """
        # Arrange - Create multiple vulnerability reports
        test_vulnerabilities = []
        for i in range(5):
            vulnerability = VulnerabilityReport(
                severity=VulnerabilitySeverity.MEDIUM,
                classification="Test Classification",
                affected_components=[f"Component {i}"],
                description=f"Test vulnerability {i} for performance testing"
            )
            test_vulnerabilities.append(vulnerability)
        
        # Act - Execute multiple workflows and measure performance
        start_time = datetime.utcnow()
        
        workflow_results = []
        for vulnerability in test_vulnerabilities:
            result = self.workflow_orchestrator.execute_vulnerability_to_resolution_workflow(vulnerability)
            workflow_results.append(result)
        
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        
        # Assert - Performance requirements
        self.assertLess(total_duration, 30.0,
                       "5 workflow executions must complete within 30 seconds")
        
        # Verify all workflows completed successfully
        successful_workflows = [r for r in workflow_results if r["overall_status"] == "completed"]
        self.assertEqual(len(successful_workflows), len(test_vulnerabilities),
                        "All test workflows must complete successfully")
        
        # Verify average workflow time
        avg_time_per_workflow = total_duration / len(test_vulnerabilities)
        self.assertLess(avg_time_per_workflow, 10.0,
                       "Average workflow execution time must be under 10 seconds")


if __name__ == "__main__":
    # Run tests with integration-focused output
    unittest.main(verbosity=2, buffer=True)