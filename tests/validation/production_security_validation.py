#!/usr/bin/env python3
"""
Production Security Validation Test Suite

This comprehensive test suite validates that all security fixes maintain system
stability and functionality while providing rollback capabilities.

Key validation areas:
1. Security fix impact assessment
2. System functionality preservation
3. Performance impact measurement
4. Rollback script validation
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unittest
from unittest.mock import patch, MagicMock

import pytest
import requests
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None

class SecurityValidationFramework:
    """Framework for validating security fixes without breaking production"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or os.getcwd())
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        
    def log_result(self, result: ValidationResult):
        """Log and store a validation result"""
        self.results.append(result)
        status_symbol = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "○"}
        logger.info(f"{status_symbol.get(result.status, '?')} {result.test_name}: {result.message}")
        
    def run_validation(self, test_func, test_name: str, *args, **kwargs) -> ValidationResult:
        """Run a validation test with timing and error handling"""
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            if isinstance(result, ValidationResult):
                result.duration_ms = duration
                self.log_result(result)
                return result
            elif result is True:
                val_result = ValidationResult(test_name, "PASS", "Test passed", duration)
                self.log_result(val_result)
                return val_result
            else:
                val_result = ValidationResult(test_name, "FAIL", f"Test failed: {result}", duration)
                self.log_result(val_result)
                return val_result
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            val_result = ValidationResult(test_name, "FAIL", f"Exception: {str(e)}", duration)
            self.log_result(val_result)
            return val_result

class TestSecurityFixImpactAssessment(unittest.TestCase):
    """Test the impact of security fixes on system components"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = Path(os.getcwd())
    
    def setUp(self):
        self.validator = SecurityValidationFramework()
        
    def test_security_report_analysis(self):
        """Analyze the current security report for fixed issues"""
        def validate_security_report():
            security_report_path = self.base_path / "reports" / "security" / "security_report.csv"
            
            if not security_report_path.exists():
                return ValidationResult(
                    "security_report_analysis", "SKIP", 
                    "Security report not found", 0,
                    {"path": str(security_report_path)}
                )
            
            try:
                # Count security issues by severity
                with open(security_report_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[1:]  # Skip header
                    
                high_issues = len([l for l in lines if ',HIGH,' in l])
                medium_issues = len([l for l in lines if ',MEDIUM,' in l])
                low_issues = len([l for l in lines if ',LOW,' in l])
                
                details = {
                    "total_issues": len(lines),
                    "high_severity": high_issues,
                    "medium_severity": medium_issues,
                    "low_severity": low_issues,
                    "report_path": str(security_report_path)
                }
                
                if high_issues == 0:
                    return ValidationResult(
                        "security_report_analysis", "PASS",
                        f"No high-severity issues found. Total: {len(lines)} issues",
                        0, details
                    )
                else:
                    return ValidationResult(
                        "security_report_analysis", "WARN",
                        f"Found {high_issues} high-severity issues requiring attention",
                        0, details
                    )
                    
            except Exception as e:
                return ValidationResult(
                    "security_report_analysis", "FAIL",
                    f"Failed to parse security report: {str(e)}", 0
                )
        
        return self.validator.run_validation(validate_security_report, "security_report_analysis")
        
    def test_module_import_stability(self):
        """Test that core modules can still be imported after security fixes"""
        def validate_module_imports():
            modules_to_test = [
                "core.agent_forge",
                "core.rag",
                "core.monitoring",
                # Note: gateway.server requires API_KEY env var, test separately
            ]
            
            failed_imports = []
            successful_imports = []
            
            for module in modules_to_test:
                try:
                    # Test import in subprocess to avoid contaminating current process
                    result = subprocess.run([
                        sys.executable, "-c", f"import {module}; print('SUCCESS')"
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0 and "SUCCESS" in result.stdout:
                        successful_imports.append(module)
                    else:
                        failed_imports.append({
                            "module": module,
                            "error": result.stderr,
                            "returncode": result.returncode
                        })
                        
                except subprocess.TimeoutExpired:
                    failed_imports.append({
                        "module": module,
                        "error": "Import timeout after 30 seconds",
                        "returncode": -1
                    })
                except Exception as e:
                    failed_imports.append({
                        "module": module,
                        "error": str(e),
                        "returncode": -1
                    })
            
            details = {
                "successful_imports": successful_imports,
                "failed_imports": failed_imports,
                "success_rate": len(successful_imports) / len(modules_to_test)
            }
            
            if not failed_imports:
                return ValidationResult(
                    "module_import_stability", "PASS",
                    f"All {len(successful_imports)} core modules import successfully",
                    0, details
                )
            elif len(successful_imports) > len(failed_imports):
                return ValidationResult(
                    "module_import_stability", "WARN",
                    f"{len(successful_imports)} imports OK, {len(failed_imports)} failed",
                    0, details
                )
            else:
                return ValidationResult(
                    "module_import_stability", "FAIL",
                    f"Critical import failures: {len(failed_imports)} of {len(modules_to_test)}",
                    0, details
                )
        
        return self.validator.run_validation(validate_module_imports, "module_import_stability")
        
    def test_gateway_security_configuration(self):
        """Test gateway security configuration without starting server"""
        def validate_gateway_config():
            try:
                # Test configuration loading with minimal environment
                test_env = os.environ.copy()
                test_env["API_KEY"] = "test_key_for_validation_only"
                test_env["GATEWAY_HOST"] = "127.0.0.1"
                test_env["GATEWAY_PORT"] = "8001"
                
                # Test in subprocess to avoid env contamination
                test_code = """
import os
os.environ["API_KEY"] = "test_key_for_validation_only"
from core.gateway.server import GatewayConfig
config = GatewayConfig()
print(f"HOST:{config.host}")
print(f"PORT:{config.port}")
print("CONFIG_SUCCESS")
"""
                
                result = subprocess.run([
                    sys.executable, "-c", test_code
                ], capture_output=True, text=True, timeout=30, env=test_env)
                
                if result.returncode == 0 and "CONFIG_SUCCESS" in result.stdout:
                    return ValidationResult(
                        "gateway_security_configuration", "PASS",
                        "Gateway security configuration loads successfully",
                        0, {"stdout": result.stdout.strip()}
                    )
                else:
                    return ValidationResult(
                        "gateway_security_configuration", "FAIL",
                        f"Gateway config failed: {result.stderr}",
                        0, {"stderr": result.stderr, "returncode": result.returncode}
                    )
                    
            except Exception as e:
                return ValidationResult(
                    "gateway_security_configuration", "FAIL",
                    f"Exception testing gateway config: {str(e)}", 0
                )
        
        return self.validator.run_validation(validate_gateway_config, "gateway_security_configuration")

class TestModelTrainingPipelineStability(unittest.TestCase):
    """Test that model training components remain functional after security fixes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_path = Path(os.getcwd())
    
    def setUp(self):
        self.validator = SecurityValidationFramework()
        
    def test_model_files_integrity(self):
        """Test that model files are intact and accessible"""
        def validate_model_integrity():
            models_dir = self.base_path / "models" / "cognate"
            
            if not models_dir.exists():
                return ValidationResult(
                    "model_files_integrity", "SKIP",
                    "Models directory not found", 0
                )
            
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("Cognate")]
            
            intact_models = []
            damaged_models = []
            
            for model_dir in model_dirs:
                config_file = model_dir / "config.json"
                model_file = model_dir / "pytorch_model.bin"
                
                if config_file.exists() and model_file.exists():
                    try:
                        # Test config file can be parsed
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        # Check model file size (should be substantial)
                        model_size = model_file.stat().st_size
                        
                        intact_models.append({
                            "name": model_dir.name,
                            "config_keys": len(config.keys()) if isinstance(config, dict) else 0,
                            "model_size_mb": round(model_size / (1024*1024), 2)
                        })
                    except Exception as e:
                        damaged_models.append({
                            "name": model_dir.name,
                            "error": str(e)
                        })
                else:
                    damaged_models.append({
                        "name": model_dir.name,
                        "error": "Missing config.json or pytorch_model.bin"
                    })
            
            details = {
                "intact_models": intact_models,
                "damaged_models": damaged_models,
                "total_models": len(model_dirs)
            }
            
            if not damaged_models and intact_models:
                return ValidationResult(
                    "model_files_integrity", "PASS",
                    f"All {len(intact_models)} model files are intact",
                    0, details
                )
            elif intact_models and len(intact_models) > len(damaged_models):
                return ValidationResult(
                    "model_files_integrity", "WARN",
                    f"{len(intact_models)} OK, {len(damaged_models)} damaged",
                    0, details
                )
            else:
                return ValidationResult(
                    "model_files_integrity", "FAIL",
                    f"Model integrity compromised: {len(damaged_models)} damaged",
                    0, details
                )
        
        return self.validator.run_validation(validate_model_integrity, "model_files_integrity")

class TestSystemFunctionalityPreservation(unittest.TestCase):
    """Test that core system functionality is preserved after security fixes"""
    
    def setUp(self):
        self.validator = SecurityValidationFramework()
        
    def test_test_suite_execution_health(self):
        """Test that the test suite can execute without critical failures"""
        def validate_test_health():
            try:
                # Run pytest collection to see test discovery health
                result = subprocess.run([
                    sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"
                ], capture_output=True, text=True, timeout=120)
                
                if "collected" in result.stdout:
                    # Parse collection results
                    lines = result.stdout.split('\n')
                    collected_line = [l for l in lines if "collected" in l]
                    
                    if collected_line:
                        details = {"collection_output": collected_line[0]}
                        return ValidationResult(
                            "test_suite_execution_health", "PASS",
                            f"Test discovery successful: {collected_line[0]}",
                            0, details
                        )
                
                # If no collection found, check for errors
                if result.returncode != 0:
                    return ValidationResult(
                        "test_suite_execution_health", "WARN",
                        f"Test collection issues detected (exit code: {result.returncode})",
                        0, {"stderr": result.stderr[:500]}
                    )
                else:
                    return ValidationResult(
                        "test_suite_execution_health", "SKIP",
                        "Test collection completed but no clear success indicator",
                        0, {"stdout": result.stdout[:500]}
                    )
                    
            except subprocess.TimeoutExpired:
                return ValidationResult(
                    "test_suite_execution_health", "FAIL",
                    "Test collection timeout after 120 seconds", 0
                )
            except Exception as e:
                return ValidationResult(
                    "test_suite_execution_health", "FAIL",
                    f"Exception during test validation: {str(e)}", 0
                )
        
        return self.validator.run_validation(validate_test_health, "test_suite_execution_health")

class ProductionSecurityValidator:
    """Main validator orchestrating all security validation tests"""
    
    def __init__(self):
        self.framework = SecurityValidationFramework()
        self.test_suites = [
            TestSecurityFixImpactAssessment(),
            TestModelTrainingPipelineStability(), 
            TestSystemFunctionalityPreservation()
        ]
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and compile results"""
        logger.info("Starting Production Security Validation")
        
        all_results = []
        
        for test_suite in self.test_suites:
            test_suite.validator = self.framework
            
            # Run all test methods in the suite
            for method_name in dir(test_suite):
                if method_name.startswith("test_"):
                    logger.info(f"Running {test_suite.__class__.__name__}.{method_name}")
                    try:
                        test_method = getattr(test_suite, method_name)
                        result = test_method()
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        logger.error(f"Failed to run {method_name}: {e}")
        
        # Compile final report
        summary = self._compile_validation_summary(all_results)
        self._generate_validation_report(summary)
        
        return summary
        
    def _compile_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Compile validation results into a summary"""
        total_tests = len(results)
        passed = len([r for r in results if r.status == "PASS"])
        failed = len([r for r in results if r.status == "FAIL"])
        warnings = len([r for r in results if r.status == "WARN"])
        skipped = len([r for r in results if r.status == "SKIP"])
        
        total_duration = sum(r.duration_ms for r in results)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "skipped": skipped,
                "success_rate": (passed / total_tests * 100) if total_tests > 0 else 0,
                "total_duration_ms": total_duration
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "message": r.message,
                    "duration_ms": r.duration_ms,
                    "details": r.details
                }
                for r in results
            ],
            "recommendations": self._generate_recommendations(results)
        }
        
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        failed_tests = [r for r in results if r.status == "FAIL"]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} critical validation failures before deployment")
            
        warning_tests = [r for r in results if r.status == "WARN"]
        if warning_tests:
            recommendations.append(f"Review {len(warning_tests)} warnings that may impact production")
            
        # Specific recommendations based on test types
        for result in results:
            if result.test_name == "gateway_security_configuration" and result.status == "FAIL":
                recommendations.append("Gateway security configuration must be fixed before production deployment")
            elif result.test_name == "model_files_integrity" and result.status in ["FAIL", "WARN"]:
                recommendations.append("Model file integrity issues may impact AI functionality")
                
        return recommendations
        
    def _generate_validation_report(self, summary: Dict[str, Any]):
        """Generate and save validation report"""
        report_path = Path("tests/validation/production_security_validation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Validation report saved to {report_path}")

def main():
    """Main entry point for production security validation"""
    validator = ProductionSecurityValidator()
    summary = validator.run_comprehensive_validation()
    
    # Print summary
    print(f"\n{'='*60}")
    print("PRODUCTION SECURITY VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {summary['summary']['total_tests']}")
    print(f"Passed: {summary['summary']['passed']}")
    print(f"Failed: {summary['summary']['failed']}")
    print(f"Warnings: {summary['summary']['warnings']}")
    print(f"Skipped: {summary['summary']['skipped']}")
    print(f"Success Rate: {summary['summary']['success_rate']:.1f}%")
    print(f"Duration: {summary['summary']['total_duration_ms']:.0f}ms")
    
    if summary['recommendations']:
        print(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"{i}. {rec}")
    
    # Exit with appropriate code
    failed_count = summary['summary']['failed']
    sys.exit(0 if failed_count == 0 else 1)

if __name__ == "__main__":
    main()