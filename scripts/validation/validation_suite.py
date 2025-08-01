#!/usr/bin/env python3
"""Integrated validation framework for AIVillage.

This module consolidates validation functionality from multiple scripts:
- validate_dependencies.py
- check_quality_gates.py
- verify_docs.py

Provides comprehensive validation including:
- Dependency validation and conflict detection
- Code quality gates and standards compliance
- Documentation verification and completeness
- Production readiness assessment
"""

import ast
import importlib
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import toml
except ImportError:
    toml = None

from ..core import BaseScript, ScriptResult


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationType(Enum):
    """Types of validation checks."""
    DEPENDENCIES = "dependencies"
    QUALITY_GATES = "quality_gates"
    DOCUMENTATION = "documentation"
    IMPORTS = "imports"
    STRUCTURE = "structure"
    TESTS = "tests"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    type: ValidationType
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of validation check."""
    type: ValidationType
    passed: bool
    issues: List[ValidationIssue]
    metrics: Dict[str, Any]
    duration: float


@dataclass
class ValidationConfig:
    """Configuration for validation suite."""
    enabled_validations: List[ValidationType]
    production_mode: bool = False
    strict_mode: bool = False
    exclude_patterns: List[str] = None
    dependency_sources: List[str] = None

    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = [
                "__pycache__",
                ".git",
                ".pytest_cache",
                "node_modules",
                "venv",
                "env",
                ".env",
                "deprecated",
                "archive",
            ]

        if self.dependency_sources is None:
            self.dependency_sources = ["pyproject.toml", "requirements.txt"]


class ValidationSuite(BaseScript):
    """Comprehensive validation suite for AIVillage."""

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        target_dir: Optional[Path] = None,
        **kwargs
    ):
        """Initialize validation suite.

        Args:
            config: Validation configuration
            target_dir: Directory to validate
            **kwargs: Additional arguments for BaseScript
        """
        super().__init__(
            name="validation_suite",
            description="Integrated validation framework for AIVillage",
            **kwargs
        )

        self.config = config or ValidationConfig(
            enabled_validations=[
                ValidationType.DEPENDENCIES,
                ValidationType.QUALITY_GATES,
                ValidationType.DOCUMENTATION,
            ]
        )

        self.target_dir = target_dir or Path.cwd()
        self.python_files: List[Path] = []
        self.validation_results: List[ValidationResult] = []

        self.logger.info(f"ValidationSuite initialized for {self.target_dir}")

    def discover_files(self) -> None:
        """Discover Python files to validate."""
        self.python_files = []

        for pattern in ["**/*.py"]:
            for file_path in self.target_dir.glob(pattern):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in self.config.exclude_patterns):
                    continue

                # Skip very large files
                if file_path.stat().st_size > 1024 * 1024:  # 1MB
                    self.logger.warning(f"Skipping large file: {file_path}")
                    continue

                self.python_files.append(file_path)

        self.logger.info(f"Discovered {len(self.python_files)} Python files")

    def validate_dependencies(self) -> ValidationResult:
        """Validate project dependencies.

        Returns:
            ValidationResult for dependency validation
        """
        import time
        start_time = time.time()
        issues = []
        metrics = {}

        try:
            # Load dependencies from various sources
            all_dependencies = set()

            # Check pyproject.toml
            pyproject_path = self.target_dir / "pyproject.toml"
            if pyproject_path.exists() and toml:
                try:
                    with open(pyproject_path, encoding="utf-8") as f:
                        data = toml.load(f)

                    # Core dependencies
                    core_deps = data.get("project", {}).get("dependencies", [])
                    for dep in core_deps:
                        all_dependencies.add(self._parse_requirement(dep))

                    # Optional dependencies
                    optional_deps = data.get("project", {}).get("optional-dependencies", {})
                    for group, deps in optional_deps.items():
                        for dep in deps:
                            all_dependencies.add(self._parse_requirement(dep))

                    metrics["pyproject_dependencies"] = len(core_deps) + sum(len(deps)
                                                                             for deps in optional_deps.values())

                except Exception as e:
                    issues.append(ValidationIssue(
                        type=ValidationType.DEPENDENCIES,
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to parse pyproject.toml: {e}",
                        file_path=str(pyproject_path)
                    ))

            # Check requirements.txt files
            for req_file in ["requirements.txt", "requirements-dev.txt", "requirements-prod.txt"]:
                req_path = self.target_dir / req_file
                if req_path.exists():
                    try:
                        with open(req_path, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    dep = self._parse_requirement(line)
                                    if dep:
                                        all_dependencies.add(dep)
                    except Exception as e:
                        issues.append(ValidationIssue(
                            type=ValidationType.DEPENDENCIES,
                            severity=ValidationSeverity.WARNING,
                            message=f"Failed to parse {req_file}: {e}",
                            file_path=str(req_path)
                        ))

            # Remove None values
            all_dependencies = {dep for dep in all_dependencies if dep}
            metrics["total_dependencies"] = len(all_dependencies)

            # Test imports
            successful_imports = []
            failed_imports = []

            for package in sorted(all_dependencies):
                import_name = self._get_import_name(package)
                try:
                    importlib.import_module(import_name)
                    successful_imports.append(package)
                except ImportError as e:
                    failed_imports.append(package)
                    issues.append(ValidationIssue(
                        type=ValidationType.DEPENDENCIES,
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to import {package} ({import_name}): {e}",
                        details={"package": package, "import_name": import_name}
                    ))
                except Exception as e:
                    issues.append(ValidationIssue(
                        type=ValidationType.DEPENDENCIES,
                        severity=ValidationSeverity.WARNING,
                        message=f"Unexpected error importing {package}: {e}",
                        details={"package": package, "import_name": import_name}
                    ))

            metrics["successful_imports"] = len(successful_imports)
            metrics["failed_imports"] = len(failed_imports)
            metrics["import_success_rate"] = len(successful_imports) / len(all_dependencies) if all_dependencies else 0

            # Check for version conflicts
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "check"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if result.returncode != 0 and result.stdout:
                    conflicts = result.stdout.strip().split('\n')
                    for conflict in conflicts:
                        if conflict.strip():
                            issues.append(ValidationIssue(
                                type=ValidationType.DEPENDENCIES,
                                severity=ValidationSeverity.ERROR,
                                message=f"Version conflict: {conflict}",
                                details={"pip_check_output": conflict}
                            ))
                    metrics["version_conflicts"] = len(conflicts)
                else:
                    metrics["version_conflicts"] = 0

            except subprocess.TimeoutExpired:
                issues.append(ValidationIssue(
                    type=ValidationType.DEPENDENCIES,
                    severity=ValidationSeverity.WARNING,
                    message="Dependency conflict check timed out"
                ))
            except Exception as e:
                issues.append(ValidationIssue(
                    type=ValidationType.DEPENDENCIES,
                    severity=ValidationSeverity.WARNING,
                    message=f"Failed to check version conflicts: {e}"
                ))

        except Exception as e:
            issues.append(ValidationIssue(
                type=ValidationType.DEPENDENCIES,
                severity=ValidationSeverity.CRITICAL,
                message=f"Dependency validation failed: {e}"
            ))

        duration = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return ValidationResult(
            type=ValidationType.DEPENDENCIES,
            passed=passed,
            issues=issues,
            metrics=metrics,
            duration=duration
        )

    def validate_quality_gates(self) -> ValidationResult:
        """Validate code quality gates.

        Returns:
            ValidationResult for quality gate validation
        """
        import time
        start_time = time.time()
        issues = []
        metrics = {}

        try:
            production_dir = self.target_dir / "src" / "production"
            experimental_dir = self.target_dir / "experimental"

            # Check import separation
            if production_dir.exists():
                production_files = list(production_dir.rglob("*.py"))
                forbidden_imports = []

                for py_file in production_files:
                    try:
                        with open(py_file, encoding="utf-8") as f:
                            content = f.read()

                        # Check for experimental imports
                        if re.search(r"^(from experimental|import experimental)", content, re.MULTILINE):
                            forbidden_imports.append(str(py_file))
                            issues.append(ValidationIssue(
                                type=ValidationType.QUALITY_GATES,
                                severity=ValidationSeverity.ERROR,
                                message="Production code imports experimental modules",
                                file_path=str(py_file)
                            ))

                        # Check for deprecated imports
                        if re.search(r"^(from deprecated|import deprecated)", content, re.MULTILINE):
                            forbidden_imports.append(str(py_file))
                            issues.append(ValidationIssue(
                                type=ValidationType.QUALITY_GATES,
                                severity=ValidationSeverity.ERROR,
                                message="Production code imports deprecated modules",
                                file_path=str(py_file)
                            ))

                    except Exception as e:
                        issues.append(ValidationIssue(
                            type=ValidationType.QUALITY_GATES,
                            severity=ValidationSeverity.WARNING,
                            message=f"Could not check imports in {py_file}: {e}",
                            file_path=str(py_file)
                        ))

                metrics["production_files_checked"] = len(production_files)
                metrics["forbidden_imports"] = len(forbidden_imports)

            # Check for TODOs in production code
            todo_files = []
            if production_dir.exists():
                for py_file in production_dir.rglob("*.py"):
                    try:
                        with open(py_file, encoding="utf-8") as f:
                            content = f.read()

                        if re.search(r"TODO|FIXME|XXX", content, re.IGNORECASE):
                            todo_files.append(str(py_file))
                            if self.config.production_mode:
                                issues.append(ValidationIssue(
                                    type=ValidationType.QUALITY_GATES,
                                    severity=ValidationSeverity.ERROR,
                                    message="TODO found in production code",
                                    file_path=str(py_file)
                                ))
                            else:
                                issues.append(ValidationIssue(
                                    type=ValidationType.QUALITY_GATES,
                                    severity=ValidationSeverity.WARNING,
                                    message="TODO found in production code",
                                    file_path=str(py_file)
                                ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            type=ValidationType.QUALITY_GATES,
                            severity=ValidationSeverity.WARNING,
                            message=f"Could not check TODOs in {py_file}: {e}",
                            file_path=str(py_file)
                        ))

            metrics["todo_files"] = len(todo_files)

            # Check experimental warnings
            experimental_warnings = 0
            if experimental_dir.exists():
                experimental_files = list(experimental_dir.rglob("*.py"))
                for py_file in experimental_files:
                    try:
                        with open(py_file, encoding="utf-8") as f:
                            content = f.read()

                        if re.search(r"warnings\.warn|ExperimentalWarning|warn_experimental", content):
                            experimental_warnings += 1
                        else:
                            issues.append(ValidationIssue(
                                type=ValidationType.QUALITY_GATES,
                                severity=ValidationSeverity.WARNING,
                                message="Experimental file missing warning",
                                file_path=str(py_file)
                            ))
                    except Exception as e:
                        issues.append(ValidationIssue(
                            type=ValidationType.QUALITY_GATES,
                            severity=ValidationSeverity.WARNING,
                            message=f"Could not check experimental warnings in {py_file}: {e}",
                            file_path=str(py_file)
                        ))

                metrics["experimental_files"] = len(experimental_files)
                metrics["experimental_warnings"] = experimental_warnings

            # Check for basic code quality issues
            syntax_errors = 0
            for py_file in self.python_files[:100]:  # Limit to first 100 files
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    # Check syntax
                    try:
                        ast.parse(content)
                    except SyntaxError as e:
                        syntax_errors += 1
                        issues.append(ValidationIssue(
                            type=ValidationType.QUALITY_GATES,
                            severity=ValidationSeverity.ERROR,
                            message=f"Syntax error: {e}",
                            file_path=str(py_file),
                            line_number=e.lineno
                        ))

                except Exception as e:
                    issues.append(ValidationIssue(
                        type=ValidationType.QUALITY_GATES,
                        severity=ValidationSeverity.WARNING,
                        message=f"Could not validate {py_file}: {e}",
                        file_path=str(py_file)
                    ))

            metrics["syntax_errors"] = syntax_errors
            metrics["files_checked"] = min(len(self.python_files), 100)

        except Exception as e:
            issues.append(ValidationIssue(
                type=ValidationType.QUALITY_GATES,
                severity=ValidationSeverity.CRITICAL,
                message=f"Quality gate validation failed: {e}"
            ))

        duration = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return ValidationResult(
            type=ValidationType.QUALITY_GATES,
            passed=passed,
            issues=issues,
            metrics=metrics,
            duration=duration
        )

    def validate_documentation(self) -> ValidationResult:
        """Validate documentation completeness and quality.

        Returns:
            ValidationResult for documentation validation
        """
        import time
        start_time = time.time()
        issues = []
        metrics = {}

        try:
            # Check for required documentation files
            required_docs = [
                "README.md",
                "CONTRIBUTING.md",
                "LICENSE",
            ]

            missing_docs = []
            for doc_file in required_docs:
                doc_path = self.target_dir / doc_file
                if not doc_path.exists():
                    missing_docs.append(doc_file)
                    severity = ValidationSeverity.ERROR if self.config.strict_mode else ValidationSeverity.WARNING
                    issues.append(ValidationIssue(
                        type=ValidationType.DOCUMENTATION,
                        severity=severity,
                        message=f"Missing required documentation: {doc_file}",
                        file_path=doc_file
                    ))

            metrics["missing_required_docs"] = len(missing_docs)

            # Check docstring coverage
            functions_without_docstrings = 0
            classes_without_docstrings = 0
            total_functions = 0
            total_classes = 0

            for py_file in self.python_files[:50]:  # Limit to first 50 files
                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()

                    tree = ast.parse(content)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            if not ast.get_docstring(node):
                                functions_without_docstrings += 1
                                if self.config.strict_mode:
                                    issues.append(ValidationIssue(
                                        type=ValidationType.DOCUMENTATION,
                                        severity=ValidationSeverity.WARNING,
                                        message=f"Function '{node.name}' missing docstring",
                                        file_path=str(py_file),
                                        line_number=node.lineno
                                    ))

                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if not ast.get_docstring(node):
                                classes_without_docstrings += 1
                                if self.config.strict_mode:
                                    issues.append(ValidationIssue(
                                        type=ValidationType.DOCUMENTATION,
                                        severity=ValidationSeverity.WARNING,
                                        message=f"Class '{node.name}' missing docstring",
                                        file_path=str(py_file),
                                        line_number=node.lineno
                                    ))

                except Exception as e:
                    issues.append(ValidationIssue(
                        type=ValidationType.DOCUMENTATION,
                        severity=ValidationSeverity.WARNING,
                        message=f"Could not check docstrings in {py_file}: {e}",
                        file_path=str(py_file)
                    ))

            metrics["total_functions"] = total_functions
            metrics["functions_without_docstrings"] = functions_without_docstrings
            metrics["function_docstring_coverage"] = (
                (total_functions - functions_without_docstrings) / total_functions
                if total_functions > 0 else 1.0
            )

            metrics["total_classes"] = total_classes
            metrics["classes_without_docstrings"] = classes_without_docstrings
            metrics["class_docstring_coverage"] = (
                (total_classes - classes_without_docstrings) / total_classes
                if total_classes > 0 else 1.0
            )

            # Check for documentation directories
            docs_dir = self.target_dir / "docs"
            if docs_dir.exists():
                doc_files = list(docs_dir.rglob("*.md"))
                metrics["documentation_files"] = len(doc_files)
            else:
                metrics["documentation_files"] = 0
                if self.config.strict_mode:
                    issues.append(ValidationIssue(
                        type=ValidationType.DOCUMENTATION,
                        severity=ValidationSeverity.WARNING,
                        message="No docs directory found"
                    ))

        except Exception as e:
            issues.append(ValidationIssue(
                type=ValidationType.DOCUMENTATION,
                severity=ValidationSeverity.CRITICAL,
                message=f"Documentation validation failed: {e}"
            ))

        duration = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return ValidationResult(
            type=ValidationType.DOCUMENTATION,
            passed=passed,
            issues=issues,
            metrics=metrics,
            duration=duration
        )

    def run_all_validations(self) -> List[ValidationResult]:
        """Run all enabled validations.

        Returns:
            List of ValidationResult objects
        """
        results = []

        # Discover files first
        self.discover_files()

        for validation_type in self.config.enabled_validations:
            self.logger.info(f"Running {validation_type.value} validation...")

            try:
                if validation_type == ValidationType.DEPENDENCIES:
                    result = self.validate_dependencies()
                elif validation_type == ValidationType.QUALITY_GATES:
                    result = self.validate_quality_gates()
                elif validation_type == ValidationType.DOCUMENTATION:
                    result = self.validate_documentation()
                else:
                    self.logger.warning(f"Unknown validation type: {validation_type}")
                    continue

                results.append(result)

                status = "PASSED" if result.passed else "FAILED"
                self.logger.info(
                    f"{validation_type.value} validation {status} "
                    f"({len(result.issues)} issues, {result.duration:.2f}s)"
                )

            except Exception as e:
                self.logger.error(f"Validation {validation_type.value} failed: {e}")
                results.append(ValidationResult(
                    type=validation_type,
                    passed=False,
                    issues=[ValidationIssue(
                        type=validation_type,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation failed: {e}"
                    )],
                    metrics={},
                    duration=0.0
                ))

        self.validation_results = results
        return results

    def generate_report(self) -> str:
        """Generate validation report.

        Returns:
            Formatted validation report
        """
        if not self.validation_results:
            return "No validation results available."

        total_issues = sum(len(result.issues) for result in self.validation_results)
        passed_validations = sum(1 for result in self.validation_results if result.passed)
        total_validations = len(self.validation_results)

        report = f"""
# AIVillage Validation Report

## Summary
- **Total Validations**: {total_validations}
- **Passed**: {passed_validations}
- **Failed**: {total_validations - passed_validations}
- **Total Issues**: {total_issues}
- **Overall Status**: {'PASSED' if passed_validations == total_validations else 'FAILED'}

"""

        for result in self.validation_results:
            status = "PASSED" if result.passed else "FAILED"
            error_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for issue in result.issues if issue.severity == ValidationSeverity.WARNING)

            report += f"""## {result.type.value.title()} Validation - {status}
- **Duration**: {result.duration:.2f}s
- **Issues**: {len(result.issues)} (Errors: {error_count}, Warnings: {warning_count})
"""

            if result.metrics:
                report += "- **Metrics**:\n"
                for key, value in result.metrics.items():
                    if isinstance(value, float):
                        report += f"  - {key}: {value:.3f}\n"
                    else:
                        report += f"  - {key}: {value}\n"

            if result.issues:
                report += "- **Issues**:\n"
                for issue in result.issues[:10]:  # Show first 10 issues
                    report += f"  - [{issue.severity.value.upper()}] {issue.message}"
                    if issue.file_path:
                        report += f" ({issue.file_path}"
                        if issue.line_number:
                            report += f":{issue.line_number}"
                        report += ")"
                    report += "\n"

                if len(result.issues) > 10:
                    report += f"  - ... and {len(result.issues) - 10} more issues\n"

            report += "\n"

        return report

    def save_results(self, output_file: Optional[Path] = None) -> None:
        """Save validation results to JSON file.

        Args:
            output_file: Output file path (defaults to validation_results.json)
        """
        if output_file is None:
            output_file = self.target_dir / "validation_results.json"

        results_data = {
            "timestamp": self.start_time,
            "target_dir": str(self.target_dir),
            "config": asdict(self.config),
            "summary": {
                "total_validations": len(self.validation_results),
                "passed_validations": sum(1 for r in self.validation_results if r.passed),
                "total_issues": sum(len(r.issues) for r in self.validation_results),
                "overall_passed": all(r.passed for r in self.validation_results),
            },
            "results": [asdict(result) for result in self.validation_results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str)

        self.logger.info(f"Validation results saved to {output_file}")

    def execute(self) -> ScriptResult:
        """Execute validation suite.

        Returns:
            ScriptResult with validation results
        """
        try:
            if self.dry_run:
                return ScriptResult(
                    success=True,
                    message="Dry run completed - validation configuration verified",
                    data={"config": asdict(self.config)},
                )

            # Run all validations
            results = self.run_all_validations()

            # Generate report
            report = self.generate_report()

            # Save results
            self.save_results()

            # Determine overall success
            overall_success = all(result.passed for result in results)
            total_issues = sum(len(result.issues) for result in results)

            return ScriptResult(
                success=overall_success,
                message=f"Validation {'completed successfully' if overall_success else 'completed with issues'}",
                data={
                    "total_validations": len(results),
                    "passed_validations": sum(1 for r in results if r.passed),
                    "total_issues": total_issues,
                    "report": report,
                    "results_summary": {
                        result.type.value: {
                            "passed": result.passed,
                            "issues": len(result.issues),
                            "duration": result.duration
                        }
                        for result in results
                    }
                },
                metrics={
                    "total_execution_time": sum(r.duration for r in results),
                    "files_discovered": len(self.python_files),
                    "validations_run": len(results),
                },
                warnings=[
                    f"{result.type.value} validation failed"
                    for result in results if not result.passed
                ] if not overall_success else None
            )

        except Exception as e:
            return ScriptResult(
                success=False,
                message=f"Validation suite failed: {e}",
                errors=[str(e)]
            )

    def _parse_requirement(self, req: str) -> Optional[str]:
        """Parse requirement string to extract package name.

        Args:
            req: Requirement string

        Returns:
            Package name or None
        """
        req = req.strip()

        if not req or req.startswith('#'):
            return None

        # Remove environment markers
        if ";" in req:
            req = req.split(";")[0].strip()

        # Remove version specifications
        for op in [">=", "<=", "==", "!=", ">", "<", "~=", "^"]:
            if op in req:
                req = req.split(op)[0].strip()
                break

        # Remove extras
        if "[" in req:
            req = req.split("[")[0].strip()

        # Skip URL requirements
        if req.startswith(("git+", "http", "https://", "file://")):
            return None

        return req

    def _get_import_name(self, package_name: str) -> str:
        """Get import name for package.

        Args:
            package_name: Package name

        Returns:
            Import name
        """
        mapping = {
            "pillow": "PIL",
            "pyyaml": "yaml",
            "scikit-learn": "sklearn",
            "scikit-image": "skimage",
            "opencv-python": "cv2",
            "beautifulsoup4": "bs4",
            "python-dotenv": "dotenv",
            "psycopg2-binary": "psycopg2",
            "sentence-transformers": "sentence_transformers",
            "huggingface-hub": "huggingface_hub",
            "faiss-cpu": "faiss",
            "llama-cpp-python": "llama_cpp",
            "qdrant-client": "qdrant_client",
        }

        return mapping.get(package_name.lower(), package_name.replace("-", "_"))
