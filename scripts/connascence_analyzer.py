#!/usr/bin/env python3
"""
Connascence Analyzer - Architectural Quality Assessment Tool
Analyzes different types of connascence to identify coupling issues
"""

import ast
import json
import sys
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict


@dataclass
class ConnascenceViolation:
    """Represents a connascence violation"""

    type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    file_path: str
    line_number: int
    description: str
    suggestion: str


class ConnascenceAnalyzer:
    """Analyzes connascence in Python codebases"""

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.violations = []

    def analyze_directory(self, directory: Path) -> List[ConnascenceViolation]:
        """Analyze all Python files in directory for connascence"""
        violations = []

        for py_file in directory.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                violations.extend(self._analyze_file(py_file))
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}", file=sys.stderr)

        return violations

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            "venv",
            ".venv",
            "migrations",
            "test_",
            "_test.py",
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file(self, file_path: Path) -> List[ConnascenceViolation]:
        """Analyze single file for connascence violations"""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Analyze different types of connascence
            violations.extend(self._check_connascence_of_name(tree, file_path))
            violations.extend(self._check_connascence_of_type(tree, file_path))
            violations.extend(self._check_connascence_of_meaning(tree, file_path))
            violations.extend(self._check_connascence_of_position(tree, file_path))
            violations.extend(self._check_connascence_of_algorithm(tree, file_path))

        except Exception as e:
            print(f"Error parsing {file_path}: {e}", file=sys.stderr)

        return violations

    def _check_connascence_of_name(self, tree: ast.AST, file_path: Path) -> List[ConnascenceViolation]:
        """Check for name-based connascence"""
        violations = []

        # Find hardcoded names that should be constants
        for node in ast.walk(tree):
            if isinstance(node, ast.Str):
                if len(node.s) > 3 and node.s.isupper():
                    violations.append(
                        ConnascenceViolation(
                            type="Connascence of Name",
                            severity="MEDIUM",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Hardcoded string constant: {node.s}",
                            suggestion="Extract to named constant",
                        )
                    )

        return violations

    def _check_connascence_of_type(self, tree: ast.AST, file_path: Path) -> List[ConnascenceViolation]:
        """Check for type-based connascence"""
        violations = []

        # Look for magic numbers
        for node in ast.walk(tree):
            if isinstance(node, ast.Num):
                if isinstance(node.n, (int, float)) and abs(node.n) > 1 and node.n not in [0, 1, -1]:
                    violations.append(
                        ConnascenceViolation(
                            type="Connascence of Type",
                            severity="LOW",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Magic number: {node.n}",
                            suggestion="Extract to named constant",
                        )
                    )

        return violations

    def _check_connascence_of_meaning(self, tree: ast.AST, file_path: Path) -> List[ConnascenceViolation]:
        """Check for meaning-based connascence"""
        violations = []

        # Look for boolean flags that could indicate meaning connascence
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for functions with multiple boolean parameters
                bool_args = [
                    arg for arg in node.args if isinstance(arg, ast.NameConstant) and isinstance(arg.value, bool)
                ]
                if len(bool_args) >= 2:
                    violations.append(
                        ConnascenceViolation(
                            type="Connascence of Meaning",
                            severity="HIGH",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description="Multiple boolean parameters suggest meaning connascence",
                            suggestion="Use named parameters or enum values",
                        )
                    )

        return violations

    def _check_connascence_of_position(self, tree: ast.AST, file_path: Path) -> List[ConnascenceViolation]:
        """Check for position-based connascence"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Functions with many positional arguments
                if len(node.args) > 4:
                    violations.append(
                        ConnascenceViolation(
                            type="Connascence of Position",
                            severity="HIGH",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Function call with {len(node.args)} positional arguments",
                            suggestion="Use keyword arguments or parameter objects",
                        )
                    )

        return violations

    def _check_connascence_of_algorithm(self, tree: ast.AST, file_path: Path) -> List[ConnascenceViolation]:
        """Check for algorithm-based connascence"""
        violations = []

        # Look for complex conditions that might indicate algorithm connascence
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if self._count_conditions(node.test) > 3:
                    violations.append(
                        ConnascenceViolation(
                            type="Connascence of Algorithm",
                            severity="MEDIUM",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description="Complex conditional logic suggests algorithm connascence",
                            suggestion="Extract to strategy pattern or lookup table",
                        )
                    )

        return violations

    def _count_conditions(self, node: ast.AST) -> int:
        """Count number of conditions in a test"""
        if isinstance(node, ast.BoolOp):
            return sum(self._count_conditions(child) for child in node.values)
        return 1


def main():
    parser = argparse.ArgumentParser(description="Connascence Analysis Tool")
    parser.add_argument("path", help="Path to analyze")
    parser.add_argument("--strict", action="store_true", help="Enable strict mode")
    parser.add_argument("--fail-on-violations", action="store_true", help="Exit with error if violations found")
    parser.add_argument("--output", help="Output file for JSON results")

    args = parser.parse_args()

    analyzer = ConnascenceAnalyzer(strict_mode=args.strict)
    violations = analyzer.analyze_directory(Path(args.path))

    # Generate report
    report = {
        "total_violations": len(violations),
        "violations_by_severity": {
            "LOW": len([v for v in violations if v.severity == "LOW"]),
            "MEDIUM": len([v for v in violations if v.severity == "MEDIUM"]),
            "HIGH": len([v for v in violations if v.severity == "HIGH"]),
            "CRITICAL": len([v for v in violations if v.severity == "CRITICAL"]),
        },
        "violations": [asdict(v) for v in violations],
    }

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(json.dumps(report, indent=2))

    # Print summary to stderr
    print(f"Connascence Analysis: {len(violations)} violations found", file=sys.stderr)
    for severity, count in report["violations_by_severity"].items():
        if count > 0:
            print(f"  {severity}: {count}", file=sys.stderr)

    # Exit with error if requested and violations found
    if args.fail_on_violations and violations:
        high_severity = len([v for v in violations if v.severity in ["HIGH", "CRITICAL"]])
        if high_severity > 0:
            print(f"FAIL: {high_severity} high/critical severity violations found", file=sys.stderr)
            sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
