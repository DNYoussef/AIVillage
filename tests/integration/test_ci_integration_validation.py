#!/usr/bin/env python3
"""
CI/CD Integration Test Validation
Comprehensive validation for CI/CD pipeline integration testing.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# Add project root to Python path for imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CIIntegrationValidator:
    """Validates CI/CD integration testing functionality."""

    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.project_root = Path(__file__).parents[2]

    def record_result(self, test_name: str, success: bool, **kwargs) -> None:
        """Record test result with metadata."""
        self.test_results[test_name] = {
            "success": success,
            "timestamp": time.time(),
            "duration": kwargs.get("duration", 0.0),
            "details": kwargs.get("details", {}),
            "error": kwargs.get("error"),
        }

        status = "PASS" if success else "FAIL"
        logger.info(f"[{status}] {test_name}")

    async def test_import_validation(self) -> bool:
        """Test that critical packages can be imported."""
        start_time = time.time()
        
        try:
            import_tests = {
                "fastapi": "fastapi",
                "pydantic": "pydantic",
                "pytest": "pytest",
                "asyncio": "asyncio",
                "pathlib": "pathlib",
                "json": "json",
                "logging": "logging",
            }

            failed_imports = []
            successful_imports = []

            for test_name, module_name in import_tests.items():
                try:
                    __import__(module_name)
                    successful_imports.append(module_name)
                except ImportError as e:
                    failed_imports.append((module_name, str(e)))

            success = len(failed_imports) == 0

            self.record_result(
                "import_validation",
                success,
                duration=time.time() - start_time,
                details={
                    "successful_imports": successful_imports,
                    "failed_imports": failed_imports,
                    "total_imports": len(import_tests),
                    "success_rate": len(successful_imports) / len(import_tests),
                },
            )

            return success

        except Exception as e:
            self.record_result(
                "import_validation",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def test_file_structure_validation(self) -> bool:
        """Test that required file structure exists."""
        start_time = time.time()
        
        try:
            required_paths = [
                "tests/integration",
                "tests/pytest.ini", 
                ".github/workflows/main-ci.yml",
                "requirements.txt",
                "src",
                "packages",
            ]

            missing_paths = []
            existing_paths = []

            for path_str in required_paths:
                path = self.project_root / path_str
                if path.exists():
                    existing_paths.append(path_str)
                else:
                    missing_paths.append(path_str)

            success = len(missing_paths) == 0

            self.record_result(
                "file_structure_validation",
                success,
                duration=time.time() - start_time,
                details={
                    "existing_paths": existing_paths,
                    "missing_paths": missing_paths,
                    "total_paths": len(required_paths),
                    "coverage": len(existing_paths) / len(required_paths),
                },
            )

            return success

        except Exception as e:
            self.record_result(
                "file_structure_validation",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def test_pytest_configuration(self) -> bool:
        """Test pytest configuration validity."""
        start_time = time.time()
        
        try:
            pytest_ini_path = self.project_root / "tests" / "pytest.ini"
            
            if not pytest_ini_path.exists():
                self.record_result(
                    "pytest_configuration",
                    False,
                    duration=time.time() - start_time,
                    error="pytest.ini not found",
                )
                return False

            # Check pytest can load configuration
            import configparser
            config = configparser.ConfigParser()
            config.read(pytest_ini_path)

            has_tool_pytest = config.has_section("tool:pytest")
            has_asyncio_mode = "asyncio_mode" in config.get("tool:pytest", "", fallback="")
            
            configuration_valid = has_tool_pytest

            self.record_result(
                "pytest_configuration",
                configuration_valid,
                duration=time.time() - start_time,
                details={
                    "has_tool_pytest": has_tool_pytest,
                    "has_asyncio_mode": has_asyncio_mode,
                    "config_file_exists": True,
                    "configuration_valid": configuration_valid,
                },
            )

            return configuration_valid

        except Exception as e:
            self.record_result(
                "pytest_configuration",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def test_async_support(self) -> bool:
        """Test async/await functionality works correctly."""
        start_time = time.time()
        
        try:
            # Test basic async functionality
            async def simple_async_test():
                await asyncio.sleep(0.01)
                return True

            result = await simple_async_test()

            # Test pytest-asyncio is available
            try:
                import pytest_asyncio
                pytest_asyncio_available = True
            except ImportError:
                pytest_asyncio_available = False

            success = result and pytest_asyncio_available

            self.record_result(
                "async_support",
                success,
                duration=time.time() - start_time,
                details={
                    "basic_async_test": result,
                    "pytest_asyncio_available": pytest_asyncio_available,
                },
            )

            return success

        except Exception as e:
            self.record_result(
                "async_support",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def test_ci_workflow_validation(self) -> bool:
        """Test CI workflow configuration."""
        start_time = time.time()
        
        try:
            workflow_path = self.project_root / ".github" / "workflows" / "main-ci.yml"
            
            if not workflow_path.exists():
                self.record_result(
                    "ci_workflow_validation",
                    False,
                    duration=time.time() - start_time,
                    error="main-ci.yml not found",
                )
                return False

            with open(workflow_path, 'r') as f:
                workflow_content = f.read()

            # Check for key CI components
            has_integration_tests = "integration-tests:" in workflow_content
            has_python_setup = "setup-python@v5" in workflow_content
            has_pytest_runner = "pytest" in workflow_content
            has_timeout_config = "timeout-minutes:" in workflow_content

            workflow_valid = has_integration_tests and has_python_setup and has_pytest_runner

            self.record_result(
                "ci_workflow_validation",
                workflow_valid,
                duration=time.time() - start_time,
                details={
                    "has_integration_tests": has_integration_tests,
                    "has_python_setup": has_python_setup,
                    "has_pytest_runner": has_pytest_runner,
                    "has_timeout_config": has_timeout_config,
                    "workflow_file_exists": True,
                },
            )

            return workflow_valid

        except Exception as e:
            self.record_result(
                "ci_workflow_validation",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def test_environment_setup(self) -> bool:
        """Test environment setup for integration tests."""
        start_time = time.time()
        
        try:
            # Check Python version compatibility
            python_version = sys.version_info
            python_compatible = python_version >= (3, 8)

            # Check environment variables that CI might set
            ci_env_vars = {
                "PYTHONPATH": os.environ.get("PYTHONPATH"),
                "DB_PASSWORD": os.environ.get("DB_PASSWORD", "default"),
                "REDIS_PASSWORD": os.environ.get("REDIS_PASSWORD", "default"),
                "JWT_SECRET": os.environ.get("JWT_SECRET", "default"),
                "AIVILLAGE_ENV": os.environ.get("AIVILLAGE_ENV", "testing"),
            }

            # Check current working directory is correct
            current_dir = Path.cwd()
            in_project_root = (current_dir.name == "AIVillage" or 
                              "AIVillage" in str(current_dir))

            success = python_compatible and in_project_root

            self.record_result(
                "environment_setup",
                success,
                duration=time.time() - start_time,
                details={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "python_compatible": python_compatible,
                    "current_directory": str(current_dir),
                    "in_project_root": in_project_root,
                    "env_vars": ci_env_vars,
                },
            )

            return success

        except Exception as e:
            self.record_result(
                "environment_setup",
                False,
                duration=time.time() - start_time,
                error=str(e),
            )
            return False

    async def run_ci_validation_tests(self) -> Dict[str, Any]:
        """Run all CI/CD integration validation tests."""
        logger.info("Starting CI/CD integration validation tests...")

        test_methods = [
            self.test_import_validation,
            self.test_file_structure_validation,
            self.test_pytest_configuration,
            self.test_async_support,
            self.test_ci_workflow_validation,
            self.test_environment_setup,
        ]

        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                logger.exception(f"Test {test_method.__name__} failed with error: {e}")

        return self.get_validation_summary()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Generate validation summary."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["success"])
        failed_tests = total_tests - passed_tests

        total_duration = time.time() - self.start_time
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        ci_ready = success_rate >= 0.8  # 80% threshold for CI readiness

        return {
            "ci_integration_ready": ci_ready,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": success_rate,
            "total_duration": total_duration,
            "results": self.test_results,
            "recommendations": self._get_ci_recommendations(),
        }

    def _get_ci_recommendations(self) -> List[str]:
        """Get CI/CD integration recommendations."""
        recommendations = []

        for test_name, result in self.test_results.items():
            if not result["success"]:
                if test_name == "import_validation":
                    recommendations.append("Install missing Python dependencies with: pip install -r requirements.txt")
                elif test_name == "pytest_configuration":
                    recommendations.append("Fix pytest.ini configuration - ensure asyncio_mode is properly set")
                elif test_name == "async_support":
                    recommendations.append("Install pytest-asyncio with: pip install pytest-asyncio")
                elif test_name == "file_structure_validation":
                    recommendations.append("Ensure all required project directories and files exist")
                elif test_name == "ci_workflow_validation":
                    recommendations.append("Review .github/workflows/main-ci.yml for integration test configuration")
                else:
                    recommendations.append(f"Fix {test_name}: {result.get('error', 'Review test criteria')}")

        if not recommendations:
            recommendations.append("CI/CD integration is ready - all validation tests passed!")

        return recommendations


@pytest.mark.asyncio
async def test_ci_integration_validation():
    """Pytest entry point for CI/CD integration validation."""
    validator = CIIntegrationValidator()
    report = await validator.run_ci_validation_tests()

    # Save detailed report
    report_path = Path("ci_integration_validation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"CI integration validation: {report['success_rate']:.1%} success rate")
    logger.info(f"Report saved to: {report_path}")

    # Assert CI readiness
    assert report["ci_integration_ready"], f"CI integration validation failed: {report['success_rate']:.1%} success rate"
    
    return report


if __name__ == "__main__":
    asyncio.run(test_ci_integration_validation())