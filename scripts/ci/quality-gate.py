#!/usr/bin/env python3
"""
Quality Gate Script for CI/CD Pipeline
Evaluates architectural fitness functions and enforces quality thresholds.
"""

import argparse
import ast
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re
import subprocess
import sys


class QualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    name: str
    value: float
    threshold: float
    passed: bool
    level: QualityLevel
    description: str


@dataclass
class QualityGateResult:
    passed: bool
    overall_score: float
    metrics: list[QualityMetric]
    violations: list[str]
    recommendations: list[str]


class ArchitecturalQualityGate:
    """Enforces architectural quality standards in CI/CD pipeline."""

    def __init__(self):
        self.project_root = Path.cwd()
        self.violations = []
        self.recommendations = []

    def evaluate_quality_gates(
        self,
        coupling_threshold: float = 12.0,
        complexity_threshold: int = 15,
        god_object_threshold: int = 500,
        magic_literal_threshold: float = 20.0,
    ) -> QualityGateResult:
        """Evaluate all quality gates and return comprehensive result."""

        metrics = []

        # 1. Coupling Score Analysis
        coupling_metric = self._evaluate_coupling_score(coupling_threshold)
        metrics.append(coupling_metric)

        # 2. Cyclomatic Complexity
        complexity_metric = self._evaluate_complexity(complexity_threshold)
        metrics.append(complexity_metric)

        # 3. God Object Detection
        god_object_metric = self._evaluate_god_objects(god_object_threshold)
        metrics.append(god_object_metric)

        # 4. Magic Literal Density
        magic_literal_metric = self._evaluate_magic_literals(magic_literal_threshold)
        metrics.append(magic_literal_metric)

        # 5. Connascence Violations
        connascence_metric = self._evaluate_connascence_violations()
        metrics.append(connascence_metric)

        # 6. Anti-pattern Detection
        antipattern_metric = self._evaluate_anti_patterns()
        metrics.append(antipattern_metric)

        # Calculate overall score and determine pass/fail
        overall_score = self._calculate_overall_score(metrics)
        passed = all(metric.passed for metric in metrics)

        return QualityGateResult(
            passed=passed,
            overall_score=overall_score,
            metrics=metrics,
            violations=self.violations,
            recommendations=self.recommendations,
        )

    def _evaluate_coupling_score(self, threshold: float) -> QualityMetric:
        """Evaluate coupling score across modules."""
        try:
            result = subprocess.run(
                ["python", "scripts/coupling_metrics.py", "--json"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                avg_coupling = data.get("average_coupling_score", 0)

                level = self._get_coupling_level(avg_coupling)
                passed = avg_coupling <= threshold

                if not passed:
                    self.violations.append(f"Coupling score {avg_coupling:.1f} exceeds threshold {threshold}")
                    self.recommendations.append("Refactor high-coupling modules using dependency injection")

                return QualityMetric(
                    name="Coupling Score",
                    value=avg_coupling,
                    threshold=threshold,
                    passed=passed,
                    level=level,
                    description="Average coupling score across modules",
                )
            else:
                return self._error_metric("Coupling Score", "Failed to analyze coupling")

        except Exception as e:
            return self._error_metric("Coupling Score", f"Error: {e}")

    def _evaluate_complexity(self, threshold: int) -> QualityMetric:
        """Evaluate cyclomatic complexity."""
        try:
            # Use radon to calculate complexity
            result = subprocess.run(
                ["radon", "cc", ".", "--json"], capture_output=True, text=True, cwd=self.project_root
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                max_complexity = 0
                high_complexity_count = 0

                for file_path, functions in data.items():
                    for func in functions:
                        complexity = func.get("complexity", 0)
                        max_complexity = max(max_complexity, complexity)
                        if complexity > threshold:
                            high_complexity_count += 1

                passed = max_complexity <= threshold
                level = self._get_complexity_level(max_complexity)

                if not passed:
                    self.violations.append(f"Maximum complexity {max_complexity} exceeds threshold {threshold}")
                    self.recommendations.append("Break down complex functions using Extract Method refactoring")

                return QualityMetric(
                    name="Cyclomatic Complexity",
                    value=max_complexity,
                    threshold=threshold,
                    passed=passed,
                    level=level,
                    description="Maximum cyclomatic complexity across codebase",
                )
            else:
                return self._error_metric("Complexity", "Failed to analyze complexity")

        except Exception as e:
            return self._error_metric("Complexity", f"Error: {e}")

    def _evaluate_god_objects(self, threshold: int) -> QualityMetric:
        """Detect God Objects (classes with too many lines)."""
        god_objects = []
        max_lines = 0

        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Count lines in class
                        class_lines = node.end_lineno - node.lineno + 1
                        max_lines = max(max_lines, class_lines)

                        if class_lines > threshold:
                            god_objects.append(
                                {
                                    "file": str(py_file.relative_to(self.project_root)),
                                    "class": node.name,
                                    "lines": class_lines,
                                }
                            )

            except Exception:
                continue  # Skip files that can't be parsed

        passed = len(god_objects) == 0
        level = QualityLevel.EXCELLENT if passed else QualityLevel.POOR

        if not passed:
            self.violations.append(f"Found {len(god_objects)} God Objects")
            self.recommendations.append("Break down large classes using Single Responsibility Principle")

        return QualityMetric(
            name="God Objects",
            value=len(god_objects),
            threshold=0,
            passed=passed,
            level=level,
            description="Number of classes exceeding size threshold",
        )

    def _evaluate_magic_literals(self, threshold: float) -> QualityMetric:
        """Evaluate magic literal density."""
        total_literals = 0
        magic_literals = 0

        # Patterns for magic literals (excluding common constants)
        magic_patterns = [
            r"\b[2-9]\d+\b",  # Numbers > 9
            r"\b0x[0-9a-fA-F]+\b",  # Hex numbers
            r'"[^"]{3,}"(?!\s*[:=])',  # String literals not in assignments
        ]

        excluded_patterns = [
            r"\b[01]\b",  # 0 and 1 are usually not magic
            r"\b(True|False|None)\b",  # Boolean/None literals
            r"__\w+__",  # Dunder methods/attributes
        ]

        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                # Count all literals
                for pattern in magic_patterns:
                    matches = re.findall(pattern, content)
                    total_literals += len(matches)

                    # Count magic literals (exclude common patterns)
                    for match in matches:
                        is_excluded = False
                        for exc_pattern in excluded_patterns:
                            if re.match(exc_pattern, match):
                                is_excluded = True
                                break

                        if not is_excluded:
                            magic_literals += 1

            except Exception:
                continue

        if total_literals > 0:
            magic_density = (magic_literals / total_literals) * 100
        else:
            magic_density = 0

        passed = magic_density <= threshold
        level = self._get_magic_literal_level(magic_density)

        if not passed:
            self.violations.append(f"Magic literal density {magic_density:.1f}% exceeds threshold {threshold}%")
            self.recommendations.append("Replace magic literals with named constants or enums")

        return QualityMetric(
            name="Magic Literal Density",
            value=magic_density,
            threshold=threshold,
            passed=passed,
            level=level,
            description="Percentage of magic literals in code",
        )

    def _evaluate_connascence_violations(self) -> QualityMetric:
        """Check for connascence violations."""
        try:
            result = subprocess.run(
                ["python", "scripts/check_connascence.py", "--count"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                violations = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                passed = violations == 0
                level = QualityLevel.EXCELLENT if passed else QualityLevel.POOR

                if not passed:
                    self.violations.append(f"Found {violations} connascence violations")
                    self.recommendations.append("Refactor strong connascence to weaker forms")

                return QualityMetric(
                    name="Connascence Violations",
                    value=violations,
                    threshold=0,
                    passed=passed,
                    level=level,
                    description="Number of connascence violations detected",
                )
            else:
                return self._error_metric("Connascence", "Failed to check connascence")

        except Exception as e:
            return self._error_metric("Connascence", f"Error: {e}")

    def _evaluate_anti_patterns(self) -> QualityMetric:
        """Detect architectural anti-patterns."""
        try:
            result = subprocess.run(
                ["python", "scripts/detect_anti_patterns.py", "--count"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                antipatterns = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                passed = antipatterns == 0
                level = QualityLevel.EXCELLENT if passed else QualityLevel.POOR

                if not passed:
                    self.violations.append(f"Found {antipatterns} anti-patterns")
                    self.recommendations.append("Eliminate anti-patterns through systematic refactoring")

                return QualityMetric(
                    name="Anti-patterns",
                    value=antipatterns,
                    threshold=0,
                    passed=passed,
                    level=level,
                    description="Number of architectural anti-patterns detected",
                )
            else:
                return self._error_metric("Anti-patterns", "Failed to detect anti-patterns")

        except Exception as e:
            return self._error_metric("Anti-patterns", f"Error: {e}")

    def _calculate_overall_score(self, metrics: list[QualityMetric]) -> float:
        """Calculate weighted overall quality score (0-100)."""
        weights = {
            "Coupling Score": 0.25,
            "Cyclomatic Complexity": 0.20,
            "God Objects": 0.15,
            "Magic Literal Density": 0.15,
            "Connascence Violations": 0.15,
            "Anti-patterns": 0.10,
        }

        total_score = 0
        total_weight = 0

        for metric in metrics:
            weight = weights.get(metric.name, 0.1)

            # Convert metric to 0-100 score
            if metric.name == "Coupling Score":
                score = max(0, 100 - (metric.value * 5))  # Coupling penalty
            elif metric.name == "Cyclomatic Complexity":
                score = max(0, 100 - (metric.value * 3))  # Complexity penalty
            elif metric.name in ["God Objects", "Connascence Violations", "Anti-patterns"]:
                score = 100 if metric.value == 0 else max(0, 100 - (metric.value * 10))
            elif metric.name == "Magic Literal Density":
                score = max(0, 100 - metric.value)
            else:
                score = 100 if metric.passed else 0

            total_score += score * weight
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def _get_coupling_level(self, coupling: float) -> QualityLevel:
        """Determine quality level based on coupling score."""
        if coupling <= 5:
            return QualityLevel.EXCELLENT
        elif coupling <= 8:
            return QualityLevel.GOOD
        elif coupling <= 12:
            return QualityLevel.ACCEPTABLE
        elif coupling <= 20:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _get_complexity_level(self, complexity: int) -> QualityLevel:
        """Determine quality level based on complexity."""
        if complexity <= 5:
            return QualityLevel.EXCELLENT
        elif complexity <= 10:
            return QualityLevel.GOOD
        elif complexity <= 15:
            return QualityLevel.ACCEPTABLE
        elif complexity <= 25:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _get_magic_literal_level(self, density: float) -> QualityLevel:
        """Determine quality level based on magic literal density."""
        if density <= 5:
            return QualityLevel.EXCELLENT
        elif density <= 10:
            return QualityLevel.GOOD
        elif density <= 20:
            return QualityLevel.ACCEPTABLE
        elif density <= 35:
            return QualityLevel.POOR
        else:
            return QualityLevel.CRITICAL

    def _error_metric(self, name: str, error: str) -> QualityMetric:
        """Create error metric for failed analysis."""
        return QualityMetric(
            name=name,
            value=0,
            threshold=0,
            passed=False,
            level=QualityLevel.CRITICAL,
            description=f"Analysis failed: {error}",
        )


def main():
    """Main entry point for quality gate evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate architectural quality gates")
    parser.add_argument("--coupling-threshold", type=float, default=12.0, help="Maximum allowed coupling score")
    parser.add_argument("--complexity-threshold", type=int, default=15, help="Maximum allowed cyclomatic complexity")
    parser.add_argument("--god-object-threshold", type=int, default=500, help="Maximum lines of code per class")
    parser.add_argument(
        "--magic-literal-threshold", type=float, default=20.0, help="Maximum magic literal density percentage"
    )
    parser.add_argument("--output", type=str, default="quality_gate_result.json", help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Initialize quality gate
    quality_gate = ArchitecturalQualityGate()

    # Evaluate quality gates
    result = quality_gate.evaluate_quality_gates(
        coupling_threshold=args.coupling_threshold,
        complexity_threshold=args.complexity_threshold,
        god_object_threshold=args.god_object_threshold,
        magic_literal_threshold=args.magic_literal_threshold,
    )

    # Prepare output data
    output_data = {
        "passed": result.passed,
        "overall_score": result.overall_score,
        "metrics": [
            {
                "name": metric.name,
                "value": metric.value,
                "threshold": metric.threshold,
                "passed": metric.passed,
                "level": metric.level.value,
                "description": metric.description,
            }
            for metric in result.metrics
        ],
        "violations": result.violations,
        "recommendations": result.recommendations,
        "timestamp": subprocess.run(["date", "+%Y-%m-%d %H:%M:%S"], capture_output=True, text=True).stdout.strip(),
    }

    # Write results to file
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    status = "✅ PASSED" if result.passed else "❌ FAILED"
    print(f"Quality Gate: {status}")
    print(f"Overall Score: {result.overall_score:.1f}/100")

    if args.verbose:
        print("\nMetric Details:")
        for metric in result.metrics:
            status_icon = "✅" if metric.passed else "❌"
            print(f"  {status_icon} {metric.name}: {metric.value} (threshold: {metric.threshold})")

        if result.violations:
            print("\nViolations:")
            for violation in result.violations:
                print(f"  - {violation}")

        if result.recommendations:
            print("\nRecommendations:")
            for rec in result.recommendations:
                print(f"  - {rec}")

    # Exit with error code if quality gate failed
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
