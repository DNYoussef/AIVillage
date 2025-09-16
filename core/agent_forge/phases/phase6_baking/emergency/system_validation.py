#!/usr/bin/env python3
"""
EMERGENCY PHASE 6 SYSTEM VALIDATION AND WORKING VERIFICATION
===========================================================

Complete system validation to verify Phase 6 is working:
- End-to-end system functionality verification
- All critical components working validation
- Performance targets achievement verification
- Quality gates passing confirmation
- Production readiness assessment

This provides final verification that system is 95%+ complete and working
"""

import asyncio
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import traceback
import subprocess
import psutil

@dataclass
class ValidationResult:
    """Result of a system validation check"""
    check_name: str
    category: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    error_message: Optional[str] = None

@dataclass
class SystemValidationReport:
    """Complete system validation report"""
    validation_id: str
    timestamp: str
    overall_passed: bool
    overall_score: float
    category_scores: Dict[str, float]
    validation_results: List[ValidationResult]
    system_working: bool
    production_ready: bool
    performance_targets_met: bool
    quality_gates_passed: bool
    critical_issues: List[str]
    recommendations: List[str]

class SystemValidator:
    """Complete system validation framework"""

    def __init__(self):
        self.logger = logging.getLogger("SystemValidator")
        self.validation_results: List[ValidationResult] = []

    async def run_complete_validation(self) -> SystemValidationReport:
        """Run complete system validation"""
        self.logger.info("Starting complete Phase 6 system validation...")

        validation_id = f"SYSTEM_VALIDATION_{int(time.time())}"
        start_time = time.time()

        # Clear previous results
        self.validation_results = []

        try:
            # 1. Core Infrastructure Validation
            await self._validate_core_infrastructure()

            # 2. Performance System Validation
            await self._validate_performance_system()

            # 3. Quality Gates Validation
            await self._validate_quality_gates()

            # 4. Integration Validation
            await self._validate_integration()

            # 5. Compliance Validation
            await self._validate_compliance()

            # 6. End-to-End Workflow Validation
            await self._validate_end_to_end_workflow()

            # 7. Production Readiness Validation
            await self._validate_production_readiness()

            # Calculate overall metrics
            overall_score, category_scores = self._calculate_scores()
            overall_passed = overall_score >= 0.95  # 95% threshold

            # Assess system status
            system_working = self._assess_system_working()
            production_ready = self._assess_production_readiness()
            performance_targets_met = self._assess_performance_targets()
            quality_gates_passed = self._assess_quality_gates()

            # Identify critical issues
            critical_issues = self._identify_critical_issues()

            # Generate recommendations
            recommendations = self._generate_recommendations()

            total_time = time.time() - start_time
            self.logger.info(f"System validation completed in {total_time:.2f}s")

            return SystemValidationReport(
                validation_id=validation_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                overall_passed=overall_passed,
                overall_score=overall_score,
                category_scores=category_scores,
                validation_results=self.validation_results,
                system_working=system_working,
                production_ready=production_ready,
                performance_targets_met=performance_targets_met,
                quality_gates_passed=quality_gates_passed,
                critical_issues=critical_issues,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"System validation failed: {e}")
            return SystemValidationReport(
                validation_id=validation_id,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                overall_passed=False,
                overall_score=0.0,
                category_scores={},
                validation_results=self.validation_results,
                system_working=False,
                production_ready=False,
                performance_targets_met=False,
                quality_gates_passed=False,
                critical_issues=[f"Validation framework failure: {str(e)}"],
                recommendations=["Fix validation framework before proceeding"]
            )

    async def _validate_core_infrastructure(self):
        """Validate core infrastructure components"""
        self.logger.info("Validating core infrastructure...")

        start_time = time.time()

        try:
            # Test core infrastructure availability
            sys_path_backup = sys.path.copy()
            import sys
            sys.path.append(str(Path(__file__).parent))

            from core_infrastructure import BakingSystemInfrastructure

            # Test infrastructure startup
            infrastructure = BakingSystemInfrastructure()
            infrastructure.start_system()

            # Test system status
            status = infrastructure.get_system_status()
            system_started = status["system_started"]
            agent_count = status.get("total_agents", 0)

            # Test diagnostics
            diagnostics = infrastructure.run_system_diagnostics()
            diagnostics_passed = diagnostics["infrastructure_check"] in ["PASS", "WARNING"]

            # Test agent communication
            agents_responsive = self._test_agent_responsiveness(infrastructure)

            infrastructure.stop_system()

            # Calculate score
            checks = {
                "system_started": system_started,
                "adequate_agents": agent_count >= 5,  # Minimum agent count
                "diagnostics_passed": diagnostics_passed,
                "agents_responsive": agents_responsive
            }

            score = sum(checks.values()) / len(checks)
            passed = all(checks.values())

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="core_infrastructure",
                category="INFRASTRUCTURE",
                passed=passed,
                score=score,
                details={
                    "system_started": system_started,
                    "agent_count": agent_count,
                    "diagnostics": diagnostics,
                    "agents_responsive": agents_responsive,
                    "checks": checks
                },
                execution_time=execution_time
            ))

            sys.path = sys_path_backup

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Core infrastructure validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="core_infrastructure",
                category="INFRASTRUCTURE",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    def _test_agent_responsiveness(self, infrastructure) -> bool:
        """Test if agents are responsive"""
        try:
            # Check if agents are in good state
            status = infrastructure.get_system_status()
            agent_states = status.get("agent_states", {})

            if not agent_states:
                return False

            # Check for failed agents
            failed_agents = 0
            for agent_id, agent_state in agent_states.items():
                if isinstance(agent_state, dict):
                    agent_status = agent_state.get("status", "unknown")
                    if agent_status == "failed":
                        failed_agents += 1

            # Consider responsive if less than 20% of agents are failed
            total_agents = len(agent_states)
            failure_rate = failed_agents / total_agents if total_agents > 0 else 1.0

            return failure_rate < 0.2

        except Exception as e:
            self.logger.warning(f"Agent responsiveness test failed: {e}")
            return False

    async def _validate_performance_system(self):
        """Validate performance optimization system"""
        self.logger.info("Validating performance system...")

        start_time = time.time()

        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from performance_fixes import AdvancedModelOptimizer, PerformanceTargets

            # Create test model
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(100, 256)
                    self.fc2 = nn.Linear(256, 128)
                    self.fc3 = nn.Linear(128, 10)
                    self.relu = nn.ReLU()

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)

            model = TestModel()
            sample_inputs = torch.randn(8, 100)

            # Setup performance targets
            targets = PerformanceTargets(
                max_inference_latency_ms=50.0,
                min_compression_ratio=0.75,
                min_accuracy_retention=0.995
            )

            # Test optimization
            optimizer = AdvancedModelOptimizer(targets)
            result = optimizer.optimize_model(
                model, sample_inputs,
                techniques=["dynamic_quantization", "pruning"]
            )

            optimization_success = result["success"]
            targets_met = result.get("targets_met", {})
            targets_met_count = sum(targets_met.values()) if targets_met else 0
            total_targets = len(targets_met) if targets_met else 0

            score = targets_met_count / total_targets if total_targets > 0 else 0.0
            passed = optimization_success and score >= 0.8

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="performance_system",
                category="PERFORMANCE",
                passed=passed,
                score=score,
                details={
                    "optimization_success": optimization_success,
                    "targets_met": targets_met,
                    "targets_met_count": targets_met_count,
                    "total_targets": total_targets,
                    "final_metrics": asdict(result.get("final_metrics")) if result.get("final_metrics") else {}
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Performance system validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="performance_system",
                category="PERFORMANCE",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    async def _validate_quality_gates(self):
        """Validate quality gates system"""
        self.logger.info("Validating quality gates...")

        start_time = time.time()

        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from quality_gates import QualityGateManager

            # Initialize quality gate manager
            gate_manager = QualityGateManager()

            # Run quality gates
            report = gate_manager.run_all_gates()

            overall_passed = report.overall_passed
            overall_score = report.overall_score
            gates_passed = sum(1 for result in report.gate_results if result.passed)
            total_gates = len(report.gate_results)

            score = overall_score
            passed = overall_passed and score >= 0.8

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="quality_gates",
                category="QUALITY",
                passed=passed,
                score=score,
                details={
                    "overall_passed": overall_passed,
                    "gates_passed": gates_passed,
                    "total_gates": total_gates,
                    "coverage_percentage": report.coverage_percentage,
                    "security_issues": report.security_issues,
                    "gate_results": [asdict(result) for result in report.gate_results]
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gates validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="quality_gates",
                category="QUALITY",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    async def _validate_integration(self):
        """Validate system integration"""
        self.logger.info("Validating system integration...")

        start_time = time.time()

        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from integration_fixes import IntegrationFixManager

            # Initialize integration manager
            integration_manager = IntegrationFixManager()

            # Test integration fixes
            deployment_result = await integration_manager.deploy_integration_fixes()

            integration_success = deployment_result["success"]
            successful_fixes = deployment_result.get("successful_fixes", 0)
            total_fixes = deployment_result.get("total_fixes", 1)

            # Get integration status
            status = integration_manager.get_integration_status()
            integration_percentage = status["integration_percentage"]
            fully_integrated = status["fully_integrated"]

            score = integration_percentage / 100.0
            passed = integration_success and fully_integrated

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="system_integration",
                category="INTEGRATION",
                passed=passed,
                score=score,
                details={
                    "integration_success": integration_success,
                    "successful_fixes": successful_fixes,
                    "total_fixes": total_fixes,
                    "integration_percentage": integration_percentage,
                    "fully_integrated": fully_integrated,
                    "fix_results": deployment_result.get("fix_results", {})
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Integration validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="system_integration",
                category="INTEGRATION",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    async def _validate_compliance(self):
        """Validate NASA POT10 compliance"""
        self.logger.info("Validating NASA POT10 compliance...")

        start_time = time.time()

        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from compliance_remediation import NASA_POT10_ComplianceFramework

            # Initialize compliance framework
            compliance_framework = NASA_POT10_ComplianceFramework()

            # Run emergency compliance fixes
            fixes_result = compliance_framework.implement_emergency_compliance_fixes()

            compliance_success = fixes_result["success"]
            final_compliance = fixes_result.get("final_compliance_percentage", 0)
            certification_ready = fixes_result.get("certification_ready", False)

            score = final_compliance / 100.0
            passed = compliance_success and final_compliance >= 95.0

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="nasa_pot10_compliance",
                category="COMPLIANCE",
                passed=passed,
                score=score,
                details={
                    "compliance_success": compliance_success,
                    "final_compliance_percentage": final_compliance,
                    "certification_ready": certification_ready,
                    "critical_violations_remaining": fixes_result.get("critical_violations_remaining", 0),
                    "fixes_implemented": fixes_result.get("fixes_implemented", {})
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Compliance validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="nasa_pot10_compliance",
                category="COMPLIANCE",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    async def _validate_end_to_end_workflow(self):
        """Validate complete end-to-end workflow"""
        self.logger.info("Validating end-to-end workflow...")

        start_time = time.time()

        try:
            import sys
            sys.path.append(str(Path(__file__).parent))
            from integration_fixes import StateManager

            # Create state manager
            state_manager = StateManager()

            # Create test model and data
            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(50, 100),
                        nn.ReLU(),
                        nn.Linear(100, 50),
                        nn.ReLU(),
                        nn.Linear(50, 10)
                    )

                def forward(self, x):
                    return self.fc(x)

            model = TestModel()
            sample_inputs = torch.randn(4, 50)

            # Create pipeline
            pipeline_id = state_manager.create_pipeline()

            # Prepare input data
            input_data = {
                "model": model,
                "sample_inputs": sample_inputs
            }

            config = {
                "optimization_techniques": ["dynamic_quantization"],
                "output_dir": "emergency/e2e_test_output"
            }

            # Execute end-to-end workflow
            result = await state_manager.execute_pipeline(pipeline_id, input_data, config)

            workflow_success = result["success"]
            stages_completed = len(result.get("stage_results", []))
            expected_stages = 4  # preparation, optimization, validation, packaging

            # Check final output
            final_output = result.get("final_output")
            has_packaged_model = final_output is not None and "packaged_model" in final_output

            score = stages_completed / expected_stages if expected_stages > 0 else 0.0
            passed = workflow_success and has_packaged_model and score == 1.0

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="end_to_end_workflow",
                category="WORKFLOW",
                passed=passed,
                score=score,
                details={
                    "workflow_success": workflow_success,
                    "stages_completed": stages_completed,
                    "expected_stages": expected_stages,
                    "has_packaged_model": has_packaged_model,
                    "pipeline_id": pipeline_id,
                    "execution_time": result.get("total_execution_time", 0)
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"End-to-end workflow validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="end_to_end_workflow",
                category="WORKFLOW",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    async def _validate_production_readiness(self):
        """Validate production readiness"""
        self.logger.info("Validating production readiness...")

        start_time = time.time()

        try:
            # Check system resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('.')

            # Performance checks
            cpu_ok = cpu_percent < 80.0  # Less than 80% CPU usage
            memory_ok = memory_info.percent < 80.0  # Less than 80% memory usage
            disk_ok = disk_usage.percent < 90.0  # Less than 90% disk usage

            # Check emergency directory structure
            emergency_dir = Path(__file__).parent
            required_files = [
                "core_infrastructure.py",
                "performance_fixes.py",
                "quality_gates.py",
                "compliance_remediation.py",
                "integration_fixes.py"
            ]

            files_present = all((emergency_dir / file).exists() for file in required_files)

            # Check if all validation results so far are mostly passing
            current_results = [r for r in self.validation_results if r.category != "PRODUCTION"]
            passed_results = sum(1 for r in current_results if r.passed)
            total_results = len(current_results)
            validation_pass_rate = passed_results / total_results if total_results > 0 else 0.0

            production_checks = {
                "cpu_usage_ok": cpu_ok,
                "memory_usage_ok": memory_ok,
                "disk_usage_ok": disk_ok,
                "files_present": files_present,
                "validation_pass_rate_ok": validation_pass_rate >= 0.8
            }

            score = sum(production_checks.values()) / len(production_checks)
            passed = all(production_checks.values())

            execution_time = time.time() - start_time

            self.validation_results.append(ValidationResult(
                check_name="production_readiness",
                category="PRODUCTION",
                passed=passed,
                score=score,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "disk_percent": disk_usage.percent,
                    "files_present": files_present,
                    "validation_pass_rate": validation_pass_rate,
                    "production_checks": production_checks
                },
                execution_time=execution_time
            ))

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Production readiness validation failed: {e}")

            self.validation_results.append(ValidationResult(
                check_name="production_readiness",
                category="PRODUCTION",
                passed=False,
                score=0.0,
                details={"error_details": traceback.format_exc()},
                execution_time=execution_time,
                error_message=str(e)
            ))

    def _calculate_scores(self) -> Tuple[float, Dict[str, float]]:
        """Calculate overall and category scores"""
        if not self.validation_results:
            return 0.0, {}

        # Calculate category scores
        categories = set(result.category for result in self.validation_results)
        category_scores = {}

        for category in categories:
            category_results = [r for r in self.validation_results if r.category == category]
            category_score = sum(r.score for r in category_results) / len(category_results)
            category_scores[category] = category_score

        # Calculate overall score (weighted average)
        category_weights = {
            "INFRASTRUCTURE": 0.25,
            "PERFORMANCE": 0.20,
            "QUALITY": 0.20,
            "INTEGRATION": 0.15,
            "COMPLIANCE": 0.10,
            "WORKFLOW": 0.05,
            "PRODUCTION": 0.05
        }

        overall_score = 0.0
        total_weight = 0.0

        for category, score in category_scores.items():
            weight = category_weights.get(category, 0.1)
            overall_score += score * weight
            total_weight += weight

        if total_weight > 0:
            overall_score /= total_weight

        return overall_score, category_scores

    def _assess_system_working(self) -> bool:
        """Assess if system is working"""
        critical_checks = ["core_infrastructure", "system_integration", "end_to_end_workflow"]
        critical_results = [r for r in self.validation_results if r.check_name in critical_checks]

        return all(result.passed for result in critical_results)

    def _assess_production_readiness(self) -> bool:
        """Assess production readiness"""
        production_result = next((r for r in self.validation_results if r.category == "PRODUCTION"), None)
        if not production_result:
            return False

        return production_result.passed and production_result.score >= 0.9

    def _assess_performance_targets(self) -> bool:
        """Assess if performance targets are met"""
        performance_result = next((r for r in self.validation_results if r.category == "PERFORMANCE"), None)
        if not performance_result:
            return False

        return performance_result.passed and performance_result.score >= 0.8

    def _assess_quality_gates(self) -> bool:
        """Assess if quality gates are passing"""
        quality_result = next((r for r in self.validation_results if r.category == "QUALITY"), None)
        if not quality_result:
            return False

        return quality_result.passed and quality_result.score >= 0.8

    def _identify_critical_issues(self) -> List[str]:
        """Identify critical issues"""
        critical_issues = []

        for result in self.validation_results:
            if not result.passed:
                if result.category in ["INFRASTRUCTURE", "INTEGRATION", "WORKFLOW"]:
                    critical_issues.append(f"CRITICAL: {result.check_name} failed - {result.error_message or 'See details'}")
                elif result.score < 0.5:
                    critical_issues.append(f"HIGH: {result.check_name} scored {result.score:.2f} - {result.error_message or 'Below acceptable threshold'}")

        return critical_issues

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []

        for result in self.validation_results:
            if not result.passed:
                if result.check_name == "core_infrastructure":
                    recommendations.append("Fix core infrastructure components before proceeding to production")
                elif result.check_name == "performance_system":
                    recommendations.append("Optimize performance system to meet all targets")
                elif result.check_name == "quality_gates":
                    recommendations.append("Address quality gate failures and increase test coverage")
                elif result.check_name == "system_integration":
                    recommendations.append("Fix integration issues and ensure all components communicate properly")
                elif result.check_name == "nasa_pot10_compliance":
                    recommendations.append("Address NASA POT10 compliance requirements before certification")
                elif result.check_name == "end_to_end_workflow":
                    recommendations.append("Fix end-to-end workflow to ensure complete pipeline functionality")
                elif result.check_name == "production_readiness":
                    recommendations.append("Address production readiness issues including resource usage and file availability")

        if not recommendations:
            recommendations.append("All validations passed - System is ready for Phase 7 transition")

        return recommendations

def save_validation_report(report: SystemValidationReport, output_path: Path) -> Path:
    """Save validation report to file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dictionary for JSON serialization
    report_dict = asdict(report)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    return output_path

def print_validation_summary(report: SystemValidationReport):
    """Print validation summary"""
    print("\n" + "=" * 80)
    print("PHASE 6 SYSTEM VALIDATION SUMMARY")
    print("=" * 80)

    print(f"Validation ID: {report.validation_id}")
    print(f"Timestamp: {report.timestamp}")

    print(f"\nOverall Results:")
    print(f"  Overall Passed: {'YES' if report.overall_passed else 'NO'}")
    print(f"  Overall Score: {report.overall_score:.2f}/1.0 ({report.overall_score*100:.1f}%)")

    print(f"\nSystem Status:")
    print(f"  System Working: {'YES' if report.system_working else 'NO'}")
    print(f"  Production Ready: {'YES' if report.production_ready else 'NO'}")
    print(f"  Performance Targets Met: {'YES' if report.performance_targets_met else 'NO'}")
    print(f"  Quality Gates Passed: {'YES' if report.quality_gates_passed else 'NO'}")

    print(f"\nCategory Scores:")
    for category, score in report.category_scores.items():
        print(f"  {category:15} {score:.2f} ({score*100:.1f}%)")

    print(f"\nValidation Results:")
    for result in report.validation_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  {result.check_name:25} {status:4} ({result.score:.2f}) [{result.execution_time:.2f}s]")
        if not result.passed and result.error_message:
            print(f"    Error: {result.error_message}")

    if report.critical_issues:
        print(f"\nCritical Issues ({len(report.critical_issues)}):")
        for i, issue in enumerate(report.critical_issues, 1):
            print(f"  {i}. {issue}")

    if report.recommendations:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")

    print("=" * 80)

async def main():
    """Main function to run system validation"""
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("EMERGENCY PHASE 6 SYSTEM VALIDATION")
    print("=" * 80)

    # Run complete system validation
    validator = SystemValidator()
    report = await validator.run_complete_validation()

    # Print summary
    print_validation_summary(report)

    # Save report
    output_path = Path("emergency/system_validation_report.json")
    save_validation_report(report, output_path)
    print(f"\nDetailed validation report saved to: {output_path}")

    # Return appropriate exit code
    return 0 if report.overall_passed else 1

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)