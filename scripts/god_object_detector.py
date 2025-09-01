#!/usr/bin/env python3
"""
God Object Detector - Architectural Quality Assessment Tool
Identifies classes and modules that violate Single Responsibility Principle
"""

import ast
import json
import sys
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict


@dataclass
class GodObjectViolation:
    """Represents a god object violation"""

    type: str  # "class" or "module"
    name: str
    file_path: str
    line_count: int
    method_count: int
    attribute_count: int
    complexity_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    suggestions: List[str]


class GodObjectDetector:
    """Detects god objects in Python codebases"""

    def __init__(
        self,
        class_line_threshold: int = 500,
        class_method_threshold: int = 20,
        module_line_threshold: int = 1000,
        complexity_threshold: float = 15.0,
    ):
        self.class_line_threshold = class_line_threshold
        self.class_method_threshold = class_method_threshold
        self.module_line_threshold = module_line_threshold
        self.complexity_threshold = complexity_threshold

    def analyze_directory(self, directory: Path) -> List[GodObjectViolation]:
        """Analyze all Python files in directory for god objects"""
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
            "__init__.py",
        ]

        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _analyze_file(self, file_path: Path) -> List[GodObjectViolation]:
        """Analyze single file for god objects"""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            lines = content.split("\n")

            # Check for god modules
            module_violation = self._check_god_module(file_path, lines, tree)
            if module_violation:
                violations.append(module_violation)

            # Check for god classes
            violations.extend(self._check_god_classes(file_path, tree, lines))

        except Exception as e:
            print(f"Error parsing {file_path}: {e}", file=sys.stderr)

        return violations

    def _check_god_module(self, file_path: Path, lines: List[str], tree: ast.AST) -> GodObjectViolation:
        """Check if module is a god object"""
        line_count = len([line for line in lines if line.strip() and not line.strip().startswith("#")])

        # Count module-level elements
        class_count = len([node for node in tree.body if isinstance(node, ast.ClassDef)])
        function_count = len([node for node in tree.body if isinstance(node, ast.FunctionDef)])

        complexity = self._calculate_complexity(tree)

        if line_count > self.module_line_threshold or complexity > self.complexity_threshold:
            severity = self._get_severity(line_count, self.module_line_threshold, complexity)

            suggestions = []
            if class_count > 5:
                suggestions.append(f"Split {class_count} classes into separate modules")
            if function_count > 10:
                suggestions.append(f"Group {function_count} functions into classes or submodules")
            if line_count > self.module_line_threshold:
                suggestions.append(f"Module has {line_count} lines, consider breaking into smaller modules")

            return GodObjectViolation(
                type="module",
                name=file_path.stem,
                file_path=str(file_path),
                line_count=line_count,
                method_count=function_count,
                attribute_count=0,  # Module-level variables are harder to count accurately
                complexity_score=complexity,
                severity=severity,
                description=f"Module exceeds size/complexity thresholds (lines: {line_count}, complexity: {complexity:.1f})",
                suggestions=suggestions,
            )

        return None

    def _check_god_classes(self, file_path: Path, tree: ast.AST, lines: List[str]) -> List[GodObjectViolation]:
        """Check for god classes"""
        violations = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                violation = self._analyze_class(file_path, node, lines)
                if violation:
                    violations.append(violation)

        return violations

    def _analyze_class(self, file_path: Path, class_node: ast.ClassDef, lines: List[str]) -> GodObjectViolation:
        """Analyze a single class for god object patterns"""

        # Count methods and attributes
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        method_count = len(methods)

        # Count attributes (assignments to self.*)
        attributes = set()
        for method in methods:
            for node in ast.walk(method):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attributes.add(target.attr)

        attribute_count = len(attributes)

        # Calculate line count for class
        start_line = class_node.lineno
        end_line = class_node.end_lineno or start_line
        class_lines = [
            line for line in lines[start_line - 1 : end_line] if line.strip() and not line.strip().startswith("#")
        ]
        line_count = len(class_lines)

        # Calculate complexity
        complexity = self._calculate_complexity(class_node)

        # Check thresholds
        is_god_object = (
            line_count > self.class_line_threshold
            or method_count > self.class_method_threshold
            or complexity > self.complexity_threshold
        )

        if is_god_object:
            severity = self._get_severity(line_count, self.class_line_threshold, complexity)

            suggestions = []
            if method_count > self.class_method_threshold:
                suggestions.append(f"Extract some of {method_count} methods into separate classes")
            if attribute_count > 10:
                suggestions.append(f"Group {attribute_count} attributes using composition pattern")
            if complexity > self.complexity_threshold:
                suggestions.append(f"Reduce cyclomatic complexity from {complexity:.1f}")
            if line_count > self.class_line_threshold:
                suggestions.append(f"Split {line_count}-line class using Single Responsibility Principle")

            # Suggest specific patterns
            if method_count > 15:
                suggestions.append("Consider Strategy, Command, or State pattern")
            if attribute_count > 8:
                suggestions.append("Consider using composition or data classes")

            return GodObjectViolation(
                type="class",
                name=class_node.name,
                file_path=str(file_path),
                line_count=line_count,
                method_count=method_count,
                attribute_count=attribute_count,
                complexity_score=complexity,
                severity=severity,
                description=f"Class exceeds thresholds (lines: {line_count}, methods: {method_count}, complexity: {complexity:.1f})",
                suggestions=suggestions,
            )

        return None

    def _calculate_complexity(self, node: ast.AST) -> float:
        """Calculate approximate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1

        return float(complexity)

    def _get_severity(self, line_count: int, threshold: int, complexity: float) -> str:
        """Determine severity based on metrics"""
        if line_count > threshold * 2 or complexity > self.complexity_threshold * 2:
            return "CRITICAL"
        elif line_count > threshold * 1.5 or complexity > self.complexity_threshold * 1.5:
            return "HIGH"
        elif line_count > threshold * 1.2 or complexity > self.complexity_threshold * 1.2:
            return "MEDIUM"
        else:
            return "LOW"


def main():
    parser = argparse.ArgumentParser(description="God Object Detection Tool")
    parser.add_argument("path", help="Path to analyze")
    parser.add_argument("--threshold", type=int, default=500, help="Line count threshold for classes")
    parser.add_argument("--method-threshold", type=int, default=20, help="Method count threshold for classes")
    parser.add_argument("--module-threshold", type=int, default=1000, help="Line count threshold for modules")
    parser.add_argument("--complexity-threshold", type=float, default=15.0, help="Complexity threshold")
    parser.add_argument("--output", help="Output file for JSON results")

    args = parser.parse_args()

    detector = GodObjectDetector(
        class_line_threshold=args.threshold,
        class_method_threshold=args.method_threshold,
        module_line_threshold=args.module_threshold,
        complexity_threshold=args.complexity_threshold,
    )

    violations = detector.analyze_directory(Path(args.path))

    # Generate report
    report = {
        "total_violations": len(violations),
        "violations_by_type": {
            "classes": len([v for v in violations if v.type == "class"]),
            "modules": len([v for v in violations if v.type == "module"]),
        },
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
    print(f"God Object Analysis: {len(violations)} violations found", file=sys.stderr)
    for type_name, count in report["violations_by_type"].items():
        if count > 0:
            print(f"  {type_name}: {count}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
