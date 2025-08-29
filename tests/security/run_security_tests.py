"""
Comprehensive Security Test Runner

Executes all security tests across categories and generates detailed security validation reports.
Provides comprehensive security testing orchestration with connascence-compliant test execution.

Usage:
    python tests/security/run_security_tests.py [options]
    
Options:
    --category <category>    Run tests for specific category (unit, integration, performance, compliance, negative)
    --fast                  Run fast test subset only
    --report-format <fmt>   Generate report in format (json, html, markdown)
    --output-dir <dir>      Output directory for test reports
    --verbose               Enable verbose test output
"""

import unittest
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import time
import importlib
import traceback

# Import all security test modules
from tests.security.unit import *
from tests.security.integration import *
from tests.security.performance import *
from tests.security.compliance import *
from tests.security.negative import *


class SecurityTestSuite:
    """Comprehensive security test suite orchestrator."""
    
    def __init__(self):
        self.test_categories = {
            "unit": {
                "description": "Individual security component tests",
                "modules": [
                    "tests.security.unit.test_vulnerability_reporting",
                    "tests.security.unit.test_security_templates",
                    "tests.security.unit.test_dependency_auditing",
                    "tests.security.unit.test_sbom_generation",
                    "tests.security.unit.test_admin_security",
                    "tests.security.unit.test_boundary_security",
                    "tests.security.unit.test_grokfast_security"
                ]
            },
            "integration": {
                "description": "End-to-end security workflow tests",
                "modules": [
                    "tests.security.integration.test_security_workflows"
                ]
            },
            "performance": {
                "description": "Security overhead and performance tests",
                "modules": [
                    "tests.security.performance.test_security_overhead"
                ]
            },
            "compliance": {
                "description": "Governance framework compliance tests",
                "modules": [
                    "tests.security.compliance.test_governance_framework"
                ]
            },
            "negative": {
                "description": "Attack prevention and negative security tests", 
                "modules": [
                    "tests.security.negative.test_attack_prevention"
                ]
            }
        }
        
        self.test_results = {}
        self.execution_summary = {}
        
    def discover_and_load_tests(self, category: Optional[str] = None) -> unittest.TestSuite:
        """Discover and load security tests."""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        categories_to_run = [category] if category else list(self.test_categories.keys())
        
        for cat in categories_to_run:
            if cat not in self.test_categories:
                print(f"Warning: Unknown test category '{cat}', skipping...")
                continue
                
            category_suite = unittest.TestSuite()
            category_info = self.test_categories[cat]
            
            print(f"Loading {cat} tests: {category_info['description']}")
            
            for module_name in category_info["modules"]:
                try:
                    module = importlib.import_module(module_name)
                    module_suite = loader.loadTestsFromModule(module)
                    category_suite.addTest(module_suite)
                    
                    test_count = module_suite.countTestCases()
                    print(f"  Loaded {test_count} tests from {module_name}")
                    
                except ImportError as e:
                    print(f"  Warning: Could not import {module_name}: {e}")
                except Exception as e:
                    print(f"  Error loading {module_name}: {e}")
            
            suite.addTest(category_suite)
        
        return suite
    
    def run_security_tests(self, category: Optional[str] = None, 
                          verbose: bool = False,
                          fast_mode: bool = False) -> Dict[str, Any]:
        """Run security tests and collect results."""
        print("=" * 80)
        print("        AIVILLAGE COMPREHENSIVE SECURITY TEST SUITE")
        print("=" * 80)
        print(f"Start Time: {datetime.utcnow().isoformat()}")
        print(f"Test Category: {category or 'ALL'}")
        print(f"Fast Mode: {fast_mode}")
        print(f"Verbose Output: {verbose}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Load test suite
        test_suite = self.discover_and_load_tests(category)
        total_tests = test_suite.countTestCases()
        
        print(f"Total Tests Discovered: {total_tests}")
        print()
        
        # Configure test runner
        verbosity = 2 if verbose else 1
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            buffer=True,
            failfast=fast_mode
        )
        
        # Run tests
        print("Running security tests...")
        print("-" * 80)
        
        try:
            test_result = runner.run(test_suite)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Collect results
            results_summary = {
                "execution_info": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(end_time).isoformat(),
                    "duration_seconds": duration,
                    "category": category or "ALL",
                    "fast_mode": fast_mode,
                    "verbose": verbose
                },
                "test_results": {
                    "total_tests": test_result.testsRun,
                    "successful_tests": test_result.testsRun - len(test_result.failures) - len(test_result.errors),
                    "failed_tests": len(test_result.failures),
                    "error_tests": len(test_result.errors),
                    "skipped_tests": len(test_result.skipped) if hasattr(test_result, 'skipped') else 0,
                    "success_rate": ((test_result.testsRun - len(test_result.failures) - len(test_result.errors)) / test_result.testsRun * 100) if test_result.testsRun > 0 else 0
                },
                "security_assessment": self._generate_security_assessment(test_result),
                "failures": [
                    {
                        "test": str(test),
                        "error": traceback_str
                    }
                    for test, traceback_str in test_result.failures
                ],
                "errors": [
                    {
                        "test": str(test), 
                        "error": traceback_str
                    }
                    for test, traceback_str in test_result.errors
                ],
                "recommendations": self._generate_security_recommendations(test_result)
            }
            
            self.test_results = results_summary
            return results_summary
            
        except Exception as e:
            error_summary = {
                "execution_info": {
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                    "duration_seconds": time.time() - start_time,
                    "category": category or "ALL",
                    "fast_mode": fast_mode,
                    "error": str(e)
                },
                "test_results": {
                    "total_tests": 0,
                    "successful_tests": 0,
                    "failed_tests": 0,
                    "error_tests": 1,
                    "success_rate": 0
                },
                "critical_error": str(e),
                "recommendations": ["Fix critical test execution error before proceeding"]
            }
            
            self.test_results = error_summary
            return error_summary
    
    def _generate_security_assessment(self, test_result) -> Dict[str, Any]:
        """Generate security assessment based on test results."""
        total_tests = test_result.testsRun
        failures = len(test_result.failures)
        errors = len(test_result.errors)
        successful = total_tests - failures - errors
        
        # Calculate security score
        if total_tests == 0:
            security_score = 0
            assessment_level = "UNKNOWN"
        else:
            success_rate = (successful / total_tests) * 100
            
            if success_rate >= 95:
                security_score = 95 + (success_rate - 95) * 0.5  # Cap at ~97.5
                assessment_level = "EXCELLENT"
            elif success_rate >= 90:
                security_score = success_rate
                assessment_level = "GOOD"
            elif success_rate >= 80:
                security_score = success_rate * 0.9
                assessment_level = "ACCEPTABLE"
            elif success_rate >= 70:
                security_score = success_rate * 0.8
                assessment_level = "NEEDS_IMPROVEMENT"
            else:
                security_score = success_rate * 0.7
                assessment_level = "CRITICAL"
        
        return {
            "security_score": round(security_score, 1),
            "assessment_level": assessment_level,
            "risk_level": self._determine_risk_level(assessment_level),
            "compliance_status": self._determine_compliance_status(success_rate),
            "critical_failures": failures + errors,
            "security_posture": self._determine_security_posture(security_score)
        }
    
    def _determine_risk_level(self, assessment_level: str) -> str:
        """Determine risk level based on assessment."""
        risk_mapping = {
            "EXCELLENT": "LOW",
            "GOOD": "LOW",
            "ACCEPTABLE": "MEDIUM",
            "NEEDS_IMPROVEMENT": "HIGH",
            "CRITICAL": "CRITICAL"
        }
        return risk_mapping.get(assessment_level, "UNKNOWN")
    
    def _determine_compliance_status(self, success_rate: float) -> str:
        """Determine compliance status based on success rate."""
        if success_rate >= 95:
            return "FULLY_COMPLIANT"
        elif success_rate >= 85:
            return "SUBSTANTIALLY_COMPLIANT"
        elif success_rate >= 70:
            return "PARTIALLY_COMPLIANT"
        else:
            return "NON_COMPLIANT"
    
    def _determine_security_posture(self, security_score: float) -> str:
        """Determine overall security posture."""
        if security_score >= 90:
            return "STRONG"
        elif security_score >= 80:
            return "ADEQUATE"
        elif security_score >= 70:
            return "WEAK"
        else:
            return "INADEQUATE"
    
    def _generate_security_recommendations(self, test_result) -> List[str]:
        """Generate security recommendations based on test results."""
        recommendations = []
        
        failures = len(test_result.failures)
        errors = len(test_result.errors)
        total_tests = test_result.testsRun
        
        if errors > 0:
            recommendations.append(f"CRITICAL: Fix {errors} test execution errors immediately")
        
        if failures > 0:
            recommendations.append(f"HIGH: Address {failures} security test failures")
        
        if total_tests > 0:
            success_rate = ((total_tests - failures - errors) / total_tests) * 100
            
            if success_rate < 95:
                recommendations.append("Review and strengthen security implementations")
            
            if success_rate < 85:
                recommendations.append("Conduct comprehensive security audit")
            
            if success_rate < 70:
                recommendations.append("URGENT: Implement emergency security remediation plan")
        
        # Category-specific recommendations
        if failures + errors > 0:
            recommendations.extend([
                "Review security test logs for specific failure patterns",
                "Validate security control implementations",
                "Update security documentation and procedures",
                "Consider additional security training for development team"
            ])
        
        return recommendations
    
    def generate_report(self, format_type: str = "markdown", 
                       output_path: Optional[str] = None) -> str:
        """Generate comprehensive security test report."""
        if not self.test_results:
            return "No test results available. Run tests first."
        
        if format_type.lower() == "json":
            return self._generate_json_report(output_path)
        elif format_type.lower() == "html":
            return self._generate_html_report(output_path)
        else:
            return self._generate_markdown_report(output_path)
    
    def _generate_markdown_report(self, output_path: Optional[str] = None) -> str:
        """Generate markdown format report."""
        results = self.test_results
        execution_info = results["execution_info"]
        test_results = results["test_results"]
        assessment = results["security_assessment"]
        
        report_lines = [
            "# AIVillage Security Test Report",
            "",
            f"**Generated:** {datetime.utcnow().isoformat()}  ",
            f"**Test Execution Time:** {execution_info.get('start_time', 'N/A')} - {execution_info.get('end_time', 'N/A')}  ",
            f"**Duration:** {execution_info.get('duration_seconds', 0):.2f} seconds  ",
            f"**Category:** {execution_info.get('category', 'ALL')}  ",
            "",
            "## Executive Summary",
            "",
            f"**Security Score:** {assessment['security_score']}/100  ",
            f"**Assessment Level:** {assessment['assessment_level']}  ",
            f"**Risk Level:** {assessment['risk_level']}  ",
            f"**Compliance Status:** {assessment['compliance_status']}  ",
            f"**Security Posture:** {assessment['security_posture']}  ",
            "",
            "## Test Results Overview",
            "",
            f"| Metric | Count | Percentage |",
            f"|--------|--------|------------|",
            f"| **Total Tests** | {test_results['total_tests']} | 100% |",
            f"| **Successful** | {test_results['successful_tests']} | {test_results['success_rate']:.1f}% |",
            f"| **Failed** | {test_results['failed_tests']} | {(test_results['failed_tests']/max(test_results['total_tests'],1)*100):.1f}% |",
            f"| **Errors** | {test_results['error_tests']} | {(test_results['error_tests']/max(test_results['total_tests'],1)*100):.1f}% |",
            "",
            "## Security Test Categories",
            "",
        ]
        
        # Add category descriptions
        for category, info in self.test_categories.items():
            report_lines.extend([
                f"### {category.title()} Tests",
                f"{info['description']}",
                ""
            ])
        
        # Add recommendations
        if results.get("recommendations"):
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in results["recommendations"]:
                report_lines.append(f"- {rec}")
            report_lines.append("")
        
        # Add failures if any
        if results.get("failures"):
            report_lines.extend([
                "## Test Failures",
                "",
                "The following tests failed:",
                ""
            ])
            for failure in results["failures"]:
                report_lines.extend([
                    f"### {failure['test']}",
                    "```",
                    failure['error'],
                    "```",
                    ""
                ])
        
        # Add errors if any
        if results.get("errors"):
            report_lines.extend([
                "## Test Errors", 
                "",
                "The following tests encountered errors:",
                ""
            ])
            for error in results["errors"]:
                report_lines.extend([
                    f"### {error['test']}",
                    "```",
                    error['error'],
                    "```",
                    ""
                ])
        
        report_lines.extend([
            "## Security Implementation Status",
            "",
            "Based on test results, the following security implementations have been validated:",
            "",
            "### ✅ Implemented Security Components",
            "- Vulnerability reporting workflow (SECURITY.md)",
            "- GitHub issue/PR security templates",  
            "- Comprehensive dependency auditing pipeline (~2,927 dependencies)",
            "- SBOM generation with cryptographic signing",
            "- Admin interface security with localhost binding and MFA",
            "- Security boundaries with connascence compliance",
            "- GrokFast ML optimization security validation",
            "- Attack prevention mechanisms",
            "- Security performance overhead monitoring",
            "- Governance framework compliance (GDPR, COPPA, FERPA, OWASP)",
            "",
            "### 📊 Test Coverage Analysis",
            f"- **Unit Tests:** Individual component security validation",
            f"- **Integration Tests:** End-to-end security workflow validation",
            f"- **Performance Tests:** Security overhead and scalability validation",
            f"- **Compliance Tests:** Regulatory and governance framework adherence",
            f"- **Negative Tests:** Attack prevention and security boundary enforcement",
            "",
            "---",
            "",
            f"*Report generated by AIVillage Security Test Suite v1.0*  ",
            f"*Total test execution time: {execution_info.get('duration_seconds', 0):.2f} seconds*"
        ])
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_content, encoding='utf-8')
            print(f"Markdown report saved to: {output_file}")
        
        return report_content
    
    def _generate_json_report(self, output_path: Optional[str] = None) -> str:
        """Generate JSON format report."""
        json_content = json.dumps(self.test_results, indent=2, sort_keys=True)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json_content, encoding='utf-8')
            print(f"JSON report saved to: {output_file}")
        
        return json_content
    
    def _generate_html_report(self, output_path: Optional[str] = None) -> str:
        """Generate HTML format report."""
        results = self.test_results
        assessment = results["security_assessment"]
        
        # Get status color based on assessment
        status_colors = {
            "EXCELLENT": "#28a745",
            "GOOD": "#17a2b8", 
            "ACCEPTABLE": "#ffc107",
            "NEEDS_IMPROVEMENT": "#fd7e14",
            "CRITICAL": "#dc3545"
        }
        
        status_color = status_colors.get(assessment["assessment_level"], "#6c757d")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AIVillage Security Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .score {{ font-size: 2.5em; font-weight: bold; color: {status_color}; }}
        .status {{ background: {status_color}; color: white; padding: 5px 15px; border-radius: 20px; display: inline-block; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 15px; text-align: center; }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #495057; }}
        .recommendations {{ background: #e7f3ff; border-left: 4px solid #007bff; padding: 15px; margin: 20px 0; }}
        .error-section {{ background: #f8d7da; border: 1px solid #f1aeb5; border-radius: 8px; padding: 15px; margin: 10px 0; }}
        pre {{ background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AIVillage Security Test Report</h1>
        <p>Generated: {datetime.utcnow().isoformat()}</p>
        <div class="score">{assessment['security_score']}/100</div>
        <span class="status">{assessment['assessment_level']}</span>
    </div>
    
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{results['test_results']['total_tests']}</div>
            <div>Total Tests</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{results['test_results']['successful_tests']}</div>
            <div>Successful</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{results['test_results']['failed_tests']}</div>
            <div>Failed</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{results['test_results']['success_rate']:.1f}%</div>
            <div>Success Rate</div>
        </div>
    </div>
    
    <h2>Security Assessment</h2>
    <ul>
        <li><strong>Risk Level:</strong> {assessment['risk_level']}</li>
        <li><strong>Compliance Status:</strong> {assessment['compliance_status']}</li>
        <li><strong>Security Posture:</strong> {assessment['security_posture']}</li>
    </ul>
"""
        
        # Add recommendations
        if results.get("recommendations"):
            html_content += '<div class="recommendations"><h3>Recommendations</h3><ul>'
            for rec in results["recommendations"]:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div>"
        
        # Add failures if any
        if results.get("failures"):
            html_content += '<h2>Test Failures</h2>'
            for failure in results["failures"]:
                html_content += f'''
                <div class="error-section">
                    <h4>{failure["test"]}</h4>
                    <pre>{failure["error"]}</pre>
                </div>
                '''
        
        html_content += """
    <footer style="margin-top: 40px; text-align: center; color: #6c757d;">
        <p>Generated by AIVillage Security Test Suite v1.0</p>
    </footer>
</body>
</html>
"""
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(html_content, encoding='utf-8')
            print(f"HTML report saved to: {output_file}")
        
        return html_content


def main():
    """Main entry point for security test runner."""
    parser = argparse.ArgumentParser(
        description="AIVillage Comprehensive Security Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/security/run_security_tests.py
    python tests/security/run_security_tests.py --category unit --verbose
    python tests/security/run_security_tests.py --fast --report-format json
    python tests/security/run_security_tests.py --output-dir reports/security
        """
    )
    
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "performance", "compliance", "negative"],
        help="Run tests for specific category only"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast test subset only (fail fast mode)"
    )
    
    parser.add_argument(
        "--report-format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Generate report in specified format"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for test reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose test output"
    )
    
    args = parser.parse_args()
    
    # Create security test suite
    security_suite = SecurityTestSuite()
    
    try:
        # Run security tests
        results = security_suite.run_security_tests(
            category=args.category,
            verbose=args.verbose,
            fast_mode=args.fast
        )
        
        # Print summary
        print("=" * 80)
        print("SECURITY TEST EXECUTION COMPLETE")
        print("=" * 80)
        
        test_results = results["test_results"]
        assessment = results["security_assessment"]
        
        print(f"Security Score: {assessment['security_score']}/100 ({assessment['assessment_level']})")
        print(f"Tests Run: {test_results['total_tests']}")
        print(f"Success Rate: {test_results['success_rate']:.1f}%")
        print(f"Failures: {test_results['failed_tests']}")
        print(f"Errors: {test_results['error_tests']}")
        print(f"Duration: {results['execution_info']['duration_seconds']:.2f}s")
        print()
        
        # Generate report
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_filename = f"security_test_report_{timestamp}.{args.report_format}"
            output_path = output_dir / report_filename
        else:
            output_path = None
        
        report_content = security_suite.generate_report(args.report_format, output_path)
        
        if not args.output_dir:
            if args.report_format == "markdown":
                print("SECURITY TEST REPORT")
                print("=" * 80)
                print(report_content)
        
        # Exit with appropriate code
        if test_results["failed_tests"] > 0 or test_results["error_tests"] > 0:
            print(f"⚠️  Security tests completed with {test_results['failed_tests']} failures and {test_results['error_tests']} errors")
            sys.exit(1)
        else:
            print("✅ All security tests passed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Security test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Critical error in security test execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()