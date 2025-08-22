#!/usr/bin/env python3
"""Architectural Fitness Functions for AIVillage

Implements automated monitoring and validation of architectural quality
following connascence management principles. These functions act as
architectural "unit tests" to prevent regression in coupling quality.

Key Functions:
1. Coupling Thresholds - Enforce maximum coupling scores
2. Connascence Violations - Block strong connascence across modules
3. Anti-pattern Detection - Prevent architectural debt
4. Dependency Rules - Enforce clean architecture layers
"""

import ast
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FitnessLevel(Enum):
    """Fitness function severity levels"""

    CRITICAL = "critical"  # Build-breaking violations
    WARNING = "warning"  # Code review flags
    INFO = "info"  # Informational only


@dataclass
class FitnessViolation:
    """Represents a fitness function violation"""

    rule: str
    level: FitnessLevel
    file_path: str
    line_number: int
    description: str
    recommendation: str
    metric_value: float
    threshold: float


@dataclass
class FitnessResults:
    """Results of fitness function evaluation"""

    passed: bool
    total_rules: int
    violations: list[FitnessViolation]
    metrics: dict[str, float]

    @property
    def critical_violations(self) -> list[FitnessViolation]:
        return [v for v in self.violations if v.level == FitnessLevel.CRITICAL]

    @property
    def warning_violations(self) -> list[FitnessViolation]:
        return [v for v in self.violations if v.level == FitnessLevel.WARNING]


class ArchitecturalFitnessChecker:
    """Implements architectural fitness functions for continuous validation"""

    def __init__(self, codebase_path: Path):
        self.codebase_path = codebase_path
        self.violations: list[FitnessViolation] = []

        # Configurable thresholds
        self.thresholds = {
            "max_coupling_score": 12.0,
            "max_method_complexity": 8,
            "max_class_size_lines": 500,
            "max_method_size_lines": 50,
            "max_positional_parameters": 3,
            "max_magic_literals_per_file": 10,
            "max_god_class_methods": 20,
            "max_duplicate_code_percentage": 0.05,
            "min_test_coverage_percentage": 0.80,
        }

    def evaluate_all_fitness_functions(self) -> FitnessResults:
        """Evaluate all architectural fitness functions"""

        self.violations = []

        # Core architectural rules
        self._check_coupling_thresholds()
        self._check_connascence_violations()
        self._check_method_complexity()
        self._check_class_size_limits()
        self._check_positional_parameters()
        self._check_magic_literals()
        self._check_god_classes()
        self._check_duplicate_code()
        self._check_dependency_rules()
        self._check_test_coverage()

        # Calculate overall metrics
        metrics = self._calculate_overall_metrics()

        # Determine if build should pass
        critical_violations = [v for v in self.violations if v.level == FitnessLevel.CRITICAL]
        passed = len(critical_violations) == 0

        return FitnessResults(
            passed=passed, total_rules=10, violations=self.violations, metrics=metrics  # Number of fitness functions
        )

    def _check_coupling_thresholds(self):
        """FF1: Enforce maximum coupling scores per file"""

        try:
            # Run coupling analysis
            result = subprocess.run(
                [sys.executable, "scripts/coupling_metrics.py", str(self.codebase_path), "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.codebase_path.parent,
            )

            if result.returncode == 0:
                metrics = json.loads(result.stdout)
                worst_files = metrics.get("worst_coupled_files", [])

                for file_path, score in worst_files:
                    if score > self.thresholds["max_coupling_score"]:
                        self.violations.append(
                            FitnessViolation(
                                rule="coupling_threshold",
                                level=FitnessLevel.CRITICAL if score > 20 else FitnessLevel.WARNING,
                                file_path=file_path,
                                line_number=1,
                                description=f"File coupling score {score:.1f} exceeds threshold {self.thresholds['max_coupling_score']}",
                                recommendation="Refactor to reduce dependencies, apply dependency injection, or split into smaller modules",
                                metric_value=score,
                                threshold=self.thresholds["max_coupling_score"],
                            )
                        )

        except Exception as e:
            logger.error(f"Failed to check coupling thresholds: {e}")

    def _check_connascence_violations(self):
        """FF2: Block strong connascence across module boundaries"""

        try:
            # Run connascence analysis
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/check_connascence.py",
                    str(self.codebase_path),
                    "--format=json",
                    "--severity=medium",
                ],
                capture_output=True,
                text=True,
                cwd=self.codebase_path.parent,
            )

            if result.returncode == 0:
                violations = json.loads(result.stdout)

                for violation in violations:
                    # Critical: Strong connascence across modules
                    if self._is_cross_module_violation(violation):
                        level = FitnessLevel.CRITICAL
                    elif violation.get("type") in ["connascence_of_position", "connascence_of_algorithm"]:
                        level = FitnessLevel.WARNING
                    else:
                        level = FitnessLevel.INFO

                    self.violations.append(
                        FitnessViolation(
                            rule="connascence_strength",
                            level=level,
                            file_path=violation["file_path"],
                            line_number=violation["line_number"],
                            description=violation["description"],
                            recommendation=violation["recommendation"],
                            metric_value=1.0,  # Binary violation
                            threshold=1.0,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to check connascence violations: {e}")

    def _check_method_complexity(self):
        """FF3: Enforce maximum method complexity"""

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)

                        if complexity > self.thresholds["max_method_complexity"]:
                            self.violations.append(
                                FitnessViolation(
                                    rule="method_complexity",
                                    level=FitnessLevel.CRITICAL if complexity > 15 else FitnessLevel.WARNING,
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Method '{node.name}' has complexity {complexity}, exceeds {self.thresholds['max_method_complexity']}",
                                    recommendation="Break down into smaller methods following Single Responsibility Principle",
                                    metric_value=complexity,
                                    threshold=self.thresholds["max_method_complexity"],
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")

    def _check_class_size_limits(self):
        """FF4: Enforce maximum class size in lines"""

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    lines = f.readlines()

                tree = ast.parse("".join(lines))

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if hasattr(node, "end_lineno"):
                            class_lines = node.end_lineno - node.lineno + 1
                        else:
                            # Fallback for older Python versions
                            class_lines = len([n for n in ast.walk(node) if hasattr(n, "lineno")])

                        if class_lines > self.thresholds["max_class_size_lines"]:
                            self.violations.append(
                                FitnessViolation(
                                    rule="class_size",
                                    level=FitnessLevel.WARNING,
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Class '{node.name}' has {class_lines} lines, exceeds {self.thresholds['max_class_size_lines']}",
                                    recommendation="Split class following Single Responsibility Principle",
                                    metric_value=class_lines,
                                    threshold=self.thresholds["max_class_size_lines"],
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to analyze class sizes in {py_file}: {e}")

    def _check_positional_parameters(self):
        """FF5: Enforce maximum positional parameters"""

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Count non-keyword-only parameters
                        positional_count = len(
                            [arg for arg in node.args.args if arg.arg != "self" and arg.arg != "cls"]
                        ) - len(node.args.kwonlyargs)

                        if positional_count > self.thresholds["max_positional_parameters"]:
                            self.violations.append(
                                FitnessViolation(
                                    rule="positional_parameters",
                                    level=FitnessLevel.WARNING,
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Function '{node.name}' has {positional_count} positional parameters, exceeds {self.thresholds['max_positional_parameters']}",
                                    recommendation="Use keyword-only parameters, data classes, or parameter objects",
                                    metric_value=positional_count,
                                    threshold=self.thresholds["max_positional_parameters"],
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to analyze parameters in {py_file}: {e}")

    def _check_magic_literals(self):
        """FF6: Limit magic literals per file"""

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)
                magic_count = 0

                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant | ast.Num | ast.Str):
                        # Skip certain patterns that are acceptable
                        if self._is_acceptable_literal(node):
                            continue
                        magic_count += 1

                if magic_count > self.thresholds["max_magic_literals_per_file"]:
                    self.violations.append(
                        FitnessViolation(
                            rule="magic_literals",
                            level=FitnessLevel.WARNING,
                            file_path=str(py_file),
                            line_number=1,
                            description=f"File has {magic_count} magic literals, exceeds {self.thresholds['max_magic_literals_per_file']}",
                            recommendation="Extract to named constants or configuration",
                            metric_value=magic_count,
                            threshold=self.thresholds["max_magic_literals_per_file"],
                        )
                    )

            except Exception as e:
                logger.warning(f"Failed to analyze magic literals in {py_file}: {e}")

    def _check_god_classes(self):
        """FF7: Detect and limit god classes"""

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])

                        if method_count > self.thresholds["max_god_class_methods"]:
                            self.violations.append(
                                FitnessViolation(
                                    rule="god_class",
                                    level=FitnessLevel.CRITICAL,
                                    file_path=str(py_file),
                                    line_number=node.lineno,
                                    description=f"Class '{node.name}' has {method_count} methods, exceeds {self.thresholds['max_god_class_methods']}",
                                    recommendation="Split into multiple classes following Single Responsibility Principle",
                                    metric_value=method_count,
                                    threshold=self.thresholds["max_god_class_methods"],
                                )
                            )

            except Exception as e:
                logger.warning(f"Failed to analyze god classes in {py_file}: {e}")

    def _check_duplicate_code(self):
        """FF8: Detect excessive code duplication"""

        try:
            # Simple duplication detection based on similar function signatures
            function_signatures = {}

            for py_file in self.codebase_path.rglob("*.py"):
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create signature based on parameter names and count
                        params = [arg.arg for arg in node.args.args]
                        signature = f"{node.name}({','.join(params)})"

                        if signature not in function_signatures:
                            function_signatures[signature] = []
                        function_signatures[signature].append((str(py_file), node.lineno))

            # Check for duplicates
            total_functions = sum(len(locations) for locations in function_signatures.values())
            duplicate_functions = sum(
                len(locations) - 1 for locations in function_signatures.values() if len(locations) > 1
            )

            if total_functions > 0:
                duplication_ratio = duplicate_functions / total_functions

                if duplication_ratio > self.thresholds["max_duplicate_code_percentage"]:
                    # Report duplicates
                    for signature, locations in function_signatures.items():
                        if len(locations) > 1:
                            for file_path, line_no in locations[1:]:  # Skip first occurrence
                                self.violations.append(
                                    FitnessViolation(
                                        rule="duplicate_code",
                                        level=FitnessLevel.WARNING,
                                        file_path=file_path,
                                        line_number=line_no,
                                        description=f"Duplicate function signature: {signature}",
                                        recommendation="Extract to shared utility or use template method pattern",
                                        metric_value=duplication_ratio,
                                        threshold=self.thresholds["max_duplicate_code_percentage"],
                                    )
                                )

        except Exception as e:
            logger.error(f"Failed to check duplicate code: {e}")

    def _check_dependency_rules(self):
        """FF9: Enforce dependency direction rules"""

        # Define allowed dependency directions (simplified)
        dependency_rules = {
            "core/agents": ["core/shared", "core/rag"],  # Agents can depend on core/rag
            "core/shared": [],  # Core should have minimal dependencies
            "apps/ui": ["infrastructure/api", "core/shared"],  # UI can depend on API/core
            "infrastructure/api": ["core/shared", "core/rag"],  # API can depend on core/rag
        }

        for py_file in self.codebase_path.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    source = f.read()

                tree = ast.parse(source)

                # Check imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import | ast.ImportFrom):
                        module_name = self._get_import_module(node)

                        if module_name and module_name.startswith("packages/"):
                            file_module = self._get_file_module(py_file)

                            if file_module in dependency_rules:
                                allowed_deps = dependency_rules[file_module]

                                if not any(module_name.startswith(allowed) for allowed in allowed_deps):
                                    # Check if this is a disallowed dependency
                                    if module_name != file_module:  # Allow self-imports
                                        self.violations.append(
                                            FitnessViolation(
                                                rule="dependency_direction",
                                                level=FitnessLevel.WARNING,
                                                file_path=str(py_file),
                                                line_number=node.lineno,
                                                description=f"Disallowed dependency: {file_module} -> {module_name}",
                                                recommendation="Follow clean architecture dependency rules",
                                                metric_value=1.0,
                                                threshold=1.0,
                                            )
                                        )

            except Exception as e:
                logger.warning(f"Failed to check dependencies in {py_file}: {e}")

    def _check_test_coverage(self):
        """FF10: Ensure minimum test coverage"""

        try:
            # Run coverage analysis if pytest-cov is available
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--cov=packages", "--cov-report=json", "--quiet"],
                capture_output=True,
                text=True,
                cwd=self.codebase_path.parent,
            )

            if result.returncode == 0:
                # Look for coverage.json
                coverage_file = self.codebase_path.parent / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)

                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0) / 100.0

                    if total_coverage < self.thresholds["min_test_coverage_percentage"]:
                        self.violations.append(
                            FitnessViolation(
                                rule="test_coverage",
                                level=FitnessLevel.WARNING,
                                file_path=".",
                                line_number=1,
                                description=f"Test coverage {total_coverage:.1%} below threshold {self.thresholds['min_test_coverage_percentage']:.1%}",
                                recommendation="Add tests for uncovered code paths",
                                metric_value=total_coverage,
                                threshold=self.thresholds["min_test_coverage_percentage"],
                            )
                        )

        except Exception as e:
            logger.warning(f"Failed to check test coverage: {e}")

    # Helper methods

    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.And | ast.Or):
                complexity += 1
            elif isinstance(child, ast.ListComp):
                complexity += 1

        return complexity

    def _is_cross_module_violation(self, violation: dict[str, Any]) -> bool:
        """Check if connascence violation crosses module boundaries"""
        # Simplified check - in practice would analyze import relationships
        file_path = violation.get("file_path", "")
        return "packages/" in file_path and violation.get("type") in [
            "connascence_of_position",
            "connascence_of_algorithm",
        ]

    def _is_acceptable_literal(self, node) -> bool:
        """Check if literal is acceptable (not magic)"""
        try:
            if hasattr(node, "value"):
                value = node.value
            elif hasattr(node, "n"):  # ast.Num
                value = node.n
            elif hasattr(node, "s"):  # ast.Str
                value = node.s
            else:
                return True

            # Acceptable patterns
            if value in [0, 1, -1, "", None, True, False]:
                return True
            if isinstance(value, str) and len(value) == 0:
                return True

        except:
            pass

        return False

    def _get_import_module(self, node) -> str | None:
        """Get module name from import node"""
        if isinstance(node, ast.ImportFrom):
            return node.module
        elif isinstance(node, ast.Import):
            if node.names and len(node.names) > 0:
                return node.names[0].name
        return None

    def _get_file_module(self, file_path: Path) -> str:
        """Get module path from file path"""
        relative_path = file_path.relative_to(self.codebase_path.parent)
        return str(relative_path.parent).replace("\\", "/")

    def _calculate_overall_metrics(self) -> dict[str, float]:
        """Calculate overall architectural metrics"""

        total_violations = len(self.violations)
        critical_violations = len([v for v in self.violations if v.level == FitnessLevel.CRITICAL])
        warning_violations = len([v for v in self.violations if v.level == FitnessLevel.WARNING])

        return {
            "total_violations": total_violations,
            "critical_violations": critical_violations,
            "warning_violations": warning_violations,
            "fitness_score": max(0.0, 1.0 - (critical_violations * 0.2 + warning_violations * 0.1)),
            "rules_passed": self.thresholds["max_coupling_score"] - total_violations,
            "architecture_debt": critical_violations * 10 + warning_violations * 5,  # Arbitrary units
        }


def main():
    """Run architectural fitness functions"""

    import argparse

    parser = argparse.ArgumentParser(description="Run architectural fitness functions")
    parser.add_argument("--codebase", default="packages", help="Codebase path to analyze")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--fail-on-critical", action="store_true", help="Exit with error on critical violations")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run fitness functions
    checker = ArchitecturalFitnessChecker(Path(args.codebase))
    results = checker.evaluate_all_fitness_functions()

    # Output results
    output_data = {
        "passed": results.passed,
        "total_rules": results.total_rules,
        "violations": [
            {
                "rule": v.rule,
                "level": v.level.value,
                "file": v.file_path,
                "line": v.line_number,
                "description": v.description,
                "recommendation": v.recommendation,
                "value": v.metric_value,
                "threshold": v.threshold,
            }
            for v in results.violations
        ],
        "metrics": results.metrics,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results written to {args.output}")
    else:
        print(json.dumps(output_data, indent=2))

    # Print summary
    print("\n=== FITNESS FUNCTION RESULTS ===")
    print(f"Overall Status: {'PASS' if results.passed else 'FAIL'}")
    print(f"Critical Violations: {len(results.critical_violations)}")
    print(f"Warning Violations: {len(results.warning_violations)}")
    print(f"Fitness Score: {results.metrics['fitness_score']:.2f}")

    # Exit with appropriate code
    if args.fail_on_critical and len(results.critical_violations) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
