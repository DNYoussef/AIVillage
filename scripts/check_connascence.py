#!/usr/bin/env python3
"""
Connascence Violation Detection Tool for AIVillage

This tool detects various forms of connascence in Python code, focusing on:
- Static forms: Name, Type, Meaning (magic values), Position, Algorithm
- Dynamic forms: Execution, Timing, Value, Identity

Based on Meilir Page-Jones' connascence theory for reducing coupling.
"""

import argparse
import ast
import collections
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


@dataclass
class ConnascenceViolation:
    """Represents a detected connascence violation."""

    type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: int
    column: int
    description: str
    recommendation: str
    code_snippet: str
    context: dict[str, Any]


class ConnascenceDetector(ast.NodeVisitor):
    """AST visitor that detects connascence violations."""

    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.violations: list[ConnascenceViolation] = []

        # Tracking structures
        self.function_definitions: dict[str, ast.FunctionDef] = {}
        self.class_definitions: dict[str, ast.ClassDef] = {}
        self.imports: set[str] = set()
        self.magic_literals: list[tuple[ast.AST, Any]] = []
        self.global_vars: set[str] = set()
        self.sleep_calls: list[ast.Call] = []
        self.positional_params: list[tuple[ast.FunctionDef, int]] = []

        # Algorithm tracking for duplicate detection
        self.function_hashes: dict[str, list[tuple[str, ast.FunctionDef]]] = collections.defaultdict(list)

    def get_code_snippet(self, node: ast.AST, context_lines: int = 2) -> str:
        """Extract code snippet around the given node."""
        if not hasattr(node, "lineno"):
            return ""

        start_line = max(0, node.lineno - context_lines - 1)
        end_line = min(len(self.source_lines), node.lineno + context_lines)

        lines = []
        for i in range(start_line, end_line):
            marker = ">>>" if i == node.lineno - 1 else "   "
            lines.append(f"{marker} {i+1:3d}: {self.source_lines[i].rstrip()}")

        return "\n".join(lines)

    def _normalize_function_body(self, node: ast.FunctionDef) -> str:
        """Create normalized hash of function body for duplicate detection."""
        # Extract just the structure, not variable names
        body_parts = []
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    body_parts.append(f"return {type(stmt.value).__name__}")
                else:
                    body_parts.append("return")
            elif isinstance(stmt, ast.If):
                body_parts.append("if")
            elif isinstance(stmt, ast.For):
                body_parts.append("for")
            elif isinstance(stmt, ast.While):
                body_parts.append("while")
            elif isinstance(stmt, ast.Assign):
                body_parts.append("assign")
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Call):
                    body_parts.append("call")
                else:
                    body_parts.append("expr")

        return "|".join(body_parts)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Detect connascence violations in function definitions."""
        self.function_definitions[node.name] = node

        # Check for Connascence of Position (>3 positional parameters)
        positional_count = sum(1 for arg in node.args.args if not arg.arg.startswith("_"))
        if positional_count > 3:
            self.positional_params.append((node, positional_count))
            self.violations.append(
                ConnascenceViolation(
                    type="connascence_of_position",
                    severity="high",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Function '{node.name}' has {positional_count} positional parameters (>3)",
                    recommendation="Consider using keyword arguments, data classes, or parameter objects",
                    code_snippet=self.get_code_snippet(node),
                    context={"parameter_count": positional_count, "function_name": node.name},
                )
            )

        # Check for algorithm duplication
        body_hash = self._normalize_function_body(node)
        if len(node.body) > 3:  # Only check substantial functions
            self.function_hashes[body_hash].append((self.file_path, node))

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Detect God Objects and other class-level violations."""
        self.class_definitions[node.name] = node

        # Count methods and lines
        method_count = sum(1 for n in node.body if isinstance(n, ast.FunctionDef))

        # Estimate lines of code (rough approximation)
        if hasattr(node, "end_lineno") and node.end_lineno:
            loc = node.end_lineno - node.lineno
        else:
            loc = len(node.body) * 5  # Rough estimate

        # God Object detection
        if method_count > 20 or loc > 500:
            self.violations.append(
                ConnascenceViolation(
                    type="god_object",
                    severity="critical",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Class '{node.name}' is a God Object: {method_count} methods, ~{loc} lines",
                    recommendation="Split into smaller, focused classes following Single Responsibility Principle",
                    code_snippet=self.get_code_snippet(node),
                    context={"method_count": method_count, "estimated_loc": loc, "class_name": node.name},
                )
            )

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Track imports for dependency analysis."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track imports for dependency analysis."""
        if node.module:
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        """Track global variable usage (Connascence of Identity)."""
        for name in node.names:
            self.global_vars.add(name)
        self.generic_visit(node)

    def visit_Num(self, node: ast.Num):
        """Detect magic numbers (Connascence of Meaning) - deprecated but kept for compatibility."""
        # Skip common "safe" numbers
        if hasattr(node, "n") and node.n not in [0, 1, -1, 2, 10, 100, 1000]:
            self.magic_literals.append((node, node.n))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect magic constants (Connascence of Meaning)."""
        if isinstance(node.value, int | float):
            if node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                self.magic_literals.append((node, node.value))
        elif isinstance(node.value, str):
            # Skip very short strings and common patterns
            if len(node.value) > 3 and not re.match(r"^[a-zA-Z0-9_-]+$", node.value):
                self.magic_literals.append((node, node.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect timing-related calls and other patterns."""
        # Connascence of Timing - sleep() calls
        if (isinstance(node.func, ast.Name) and node.func.id == "sleep") or (
            isinstance(node.func, ast.Attribute) and node.func.attr == "sleep"
        ):
            self.sleep_calls.append(node)
            self.violations.append(
                ConnascenceViolation(
                    type="connascence_of_timing",
                    severity="medium",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description="Sleep-based timing dependency detected",
                    recommendation="Use proper synchronization primitives, events, or async patterns",
                    code_snippet=self.get_code_snippet(node),
                    context={"call_type": "sleep"},
                )
            )

        self.generic_visit(node)

    def finalize_analysis(self):
        """Perform final analysis that requires complete traversal."""
        # Check for algorithm duplicates
        for body_hash, functions in self.function_hashes.items():
            if len(functions) > 1:
                for file_path, func_node in functions:
                    self.violations.append(
                        ConnascenceViolation(
                            type="connascence_of_algorithm",
                            severity="medium",
                            file_path=file_path,
                            line_number=func_node.lineno,
                            column=func_node.col_offset,
                            description=f"Function '{func_node.name}' appears to duplicate algorithm from other functions",
                            recommendation="Extract common algorithm into shared function or module",
                            code_snippet=self.get_code_snippet(func_node),
                            context={
                                "duplicate_count": len(functions),
                                "function_name": func_node.name,
                                "similar_functions": [f.name for _, f in functions if f != func_node],
                            },
                        )
                    )

        # Analyze magic literals in context
        for node, value in self.magic_literals:
            # Context-sensitive analysis
            in_conditional = self._is_in_conditional(node)
            self.violations.append(
                ConnascenceViolation(
                    type="connascence_of_meaning",
                    severity="high" if in_conditional else "medium",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Magic literal '{value}' should be a named constant",
                    recommendation="Replace with a well-named constant or configuration value",
                    code_snippet=self.get_code_snippet(node),
                    context={"literal_value": value, "in_conditional": in_conditional},
                )
            )

        # Check for excessive global usage
        if len(self.global_vars) > 5:
            # Find a representative location (first global usage)
            for node in ast.walk(ast.parse("".join(self.source_lines))):
                if isinstance(node, ast.Global):
                    self.violations.append(
                        ConnascenceViolation(
                            type="connascence_of_identity",
                            severity="high",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            description=f"Excessive global variable usage: {len(self.global_vars)} globals",
                            recommendation="Use dependency injection, configuration objects, or class attributes",
                            code_snippet=self.get_code_snippet(node),
                            context={"global_count": len(self.global_vars), "global_vars": list(self.global_vars)},
                        )
                    )
                    break

    def _is_in_conditional(self, node: ast.AST) -> bool:
        """Check if node is within a conditional statement."""
        # This is a simplified check - in practice you'd walk up the AST
        line_content = self.source_lines[node.lineno - 1] if node.lineno <= len(self.source_lines) else ""
        return any(keyword in line_content for keyword in ["if ", "elif ", "while ", "assert "])


class ConnascenceAnalyzer:
    """Main analyzer that orchestrates connascence detection."""

    def __init__(self, exclusions: list[str] | None = None):
        self.exclusions = exclusions or [
            "test_*",
            "tests/",
            "*_test.py",
            "conftest.py",
            "deprecated/",
            "archive/",
            "experimental/",
            "__pycache__/",
            ".git/",
            "build/",
            "dist/",
            "*.egg-info/",
            "venv*/",
            "*env*/",
        ]
        self.violations: list[ConnascenceViolation] = []
        self.file_stats: dict[str, dict] = {}

    def should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed based on exclusions."""
        path_str = str(file_path)
        for exclusion in self.exclusions:
            if exclusion.endswith("/"):
                if exclusion[:-1] in path_str:
                    return False
            elif "*" in exclusion:
                import fnmatch

                if fnmatch.fnmatch(path_str, exclusion):
                    return False
            elif exclusion in path_str:
                return False
        return True

    def analyze_file(self, file_path: Path) -> list[ConnascenceViolation]:
        """Analyze a single Python file for connascence violations."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
                source_lines = source.splitlines()

            tree = ast.parse(source, filename=str(file_path))
            detector = ConnascenceDetector(str(file_path), source_lines)
            detector.visit(tree)
            detector.finalize_analysis()

            # Collect file statistics
            self.file_stats[str(file_path)] = {
                "functions": len(detector.function_definitions),
                "classes": len(detector.class_definitions),
                "imports": len(detector.imports),
                "globals": len(detector.global_vars),
                "magic_literals": len(detector.magic_literals),
                "violations": len(detector.violations),
            }

            return detector.violations

        except (SyntaxError, UnicodeDecodeError) as e:
            # Return a violation for unparseable files
            return [
                ConnascenceViolation(
                    type="syntax_error",
                    severity="critical",
                    file_path=str(file_path),
                    line_number=getattr(e, "lineno", 1),
                    column=getattr(e, "offset", 0) or 0,
                    description=f"File cannot be parsed: {e}",
                    recommendation="Fix syntax errors before analyzing connascence",
                    code_snippet="",
                    context={"error": str(e)},
                )
            ]

    def analyze_directory(self, directory: Path) -> list[ConnascenceViolation]:
        """Analyze all Python files in a directory tree."""
        all_violations = []

        for py_file in directory.rglob("*.py"):
            if self.should_analyze_file(py_file):
                file_violations = self.analyze_file(py_file)
                all_violations.extend(file_violations)
                self.violations.extend(file_violations)

        return all_violations

    def generate_report(self, violations: list[ConnascenceViolation], output_format: str = "text") -> str:
        """Generate a report of connascence violations."""
        if output_format == "json":
            return json.dumps([asdict(v) for v in violations], indent=2)

        # Text report
        report_lines = ["=" * 80, "CONNASCENCE ANALYSIS REPORT", "=" * 80, ""]

        # Summary
        severity_counts = collections.Counter(v.severity for v in violations)
        type_counts = collections.Counter(v.type for v in violations)

        report_lines.extend(
            [
                f"Total violations: {len(violations)}",
                f"Files analyzed: {len(self.file_stats)}",
                "",
                "Severity breakdown:",
                f"  Critical: {severity_counts['critical']:3d}",
                f"  High:     {severity_counts['high']:3d}",
                f"  Medium:   {severity_counts['medium']:3d}",
                f"  Low:      {severity_counts['low']:3d}",
                "",
                "Violation types:",
            ]
        )

        for violation_type, count in type_counts.most_common():
            report_lines.append(f"  {violation_type:30s}: {count:3d}")

        report_lines.extend(["", "=" * 80, "DETAILED VIOLATIONS", "=" * 80, ""])

        # Group violations by severity
        for severity in ["critical", "high", "medium", "low"]:
            severity_violations = [v for v in violations if v.severity == severity]
            if not severity_violations:
                continue

            report_lines.extend([f"\n{severity.upper()} SEVERITY ({len(severity_violations)} violations)", "-" * 40])

            for v in severity_violations:
                report_lines.extend(
                    [
                        f"\n{v.type}: {v.description}",
                        f"File: {v.file_path}:{v.line_number}:{v.column}",
                        f"Recommendation: {v.recommendation}",
                    ]
                )

                if v.code_snippet:
                    report_lines.extend(["Code context:", v.code_snippet, ""])

        # Summary statistics
        if self.file_stats:
            report_lines.extend(["\n" + "=" * 80, "FILE STATISTICS", "=" * 80, ""])

            total_functions = sum(stats["functions"] for stats in self.file_stats.values())
            total_classes = sum(stats["classes"] for stats in self.file_stats.values())
            total_imports = sum(stats["imports"] for stats in self.file_stats.values())

            report_lines.extend(
                [
                    f"Total functions analyzed: {total_functions}",
                    f"Total classes analyzed: {total_classes}",
                    f"Total imports: {total_imports}",
                    "",
                    "Most problematic files:",
                ]
            )

            # Sort files by violation count
            sorted_files = sorted(self.file_stats.items(), key=lambda x: x[1]["violations"], reverse=True)[:10]

            for file_path, stats in sorted_files:
                if stats["violations"] > 0:
                    report_lines.append(f"  {os.path.basename(file_path):30s}: {stats['violations']:3d} violations")

        return "\n".join(report_lines)


def main():
    """Main entry point for the connascence checker."""
    parser = argparse.ArgumentParser(
        description="Detect connascence violations in Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_connascence.py .                    # Analyze current directory
  python check_connascence.py src/ --json          # JSON output
  python check_connascence.py . --severity high    # Only high+ severity
  python check_connascence.py . --output report.txt # Save to file
        """,
    )

    parser.add_argument("path", help="Path to analyze (file or directory)")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    parser.add_argument(
        "--format", "-f", choices=["text", "json"], default="text", help="Output format (default: text)"
    )
    parser.add_argument(
        "--severity", "-s", choices=["low", "medium", "high", "critical"], help="Minimum severity level to report"
    )
    parser.add_argument("--exclude", "-e", action="append", help="Additional exclusion patterns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize analyzer
    exclusions = None
    if args.exclude:
        exclusions = args.exclude

    analyzer = ConnascenceAnalyzer(exclusions)

    # Analyze path
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Analyzing {target_path}...")

    start_time = time.time()

    if target_path.is_file():
        violations = analyzer.analyze_file(target_path)
    else:
        violations = analyzer.analyze_directory(target_path)

    elapsed = time.time() - start_time

    # Filter by severity if requested
    if args.severity:
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order[args.severity]
        violations = [v for v in violations if severity_order[v.severity] >= min_level]

    # Generate report
    report = analyzer.generate_report(violations, args.format)

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        if args.verbose:
            print(f"Report saved to {args.output}")
    else:
        # Handle encoding issues on Windows
        try:
            print(report)
        except UnicodeEncodeError:
            print(report.encode("ascii", "replace").decode("ascii"))

    if args.verbose:
        print(f"\nAnalysis completed in {elapsed:.2f} seconds")
        print(f"Found {len(violations)} violations")

    # Exit with error code if critical violations found
    critical_count = sum(1 for v in violations if v.severity == "critical")
    return min(critical_count, 1)  # Return 1 if any critical violations


if __name__ == "__main__":
    sys.exit(main())
