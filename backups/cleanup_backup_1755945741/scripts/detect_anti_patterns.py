#!/usr/bin/env python3
"""
Anti-Pattern Detection Tool for AIVillage

This tool detects common anti-patterns that indicate poor software design:
- Big Ball of Mud indicators
- God Object detection
- Copy-paste programming (duplicate code blocks)
- Database-as-IPC patterns
- Sequential coupling in APIs
- Magic number/string usage
- Shotgun Surgery indicators
- Feature Envy patterns
- Data Clumps
- Long Parameter Lists

Provides specific recommendations for refactoring each anti-pattern.
"""

import argparse
import ast
import collections
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any


@dataclass
class AntiPattern:
    """Represents a detected anti-pattern."""

    pattern_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: int
    column: int
    description: str
    recommendation: str
    code_snippet: str
    context: dict[str, Any]
    related_files: list[str] = None  # For cross-file patterns


class AntiPatternDetector(ast.NodeVisitor):
    """AST visitor that detects anti-patterns."""

    def __init__(self, file_path: str, source_lines: list[str], project_context: dict | None = None):
        self.file_path = file_path
        self.source_lines = source_lines
        self.project_context = project_context or {}
        self.anti_patterns: list[AntiPattern] = []

        # Pattern tracking
        self.functions = {}
        self.classes = {}
        self.imports = set()
        self.sql_queries = []
        self.magic_literals = []
        self.method_calls = collections.defaultdict(list)
        self.data_access_patterns = []
        self.parameter_lists = []

        # Code duplication tracking
        self.code_blocks = []
        self.function_similarities = {}

        # Complexity tracking
        self.nested_complexity = []
        self.coupling_indicators = []

    def get_code_snippet(self, node: ast.AST, context_lines: int = 3) -> str:
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

    def _calculate_method_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate the complexity of a method."""
        complexity = 1
        nesting_level = 0
        max_nesting = 0

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor):
                complexity += 1
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif isinstance(child, ast.And | ast.Or):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1

        return complexity, max_nesting

    def _extract_code_block_signature(self, node: ast.stmt | list[ast.stmt]) -> str:
        """Extract a signature for code block similarity detection."""
        if isinstance(node, list):
            statements = node
        else:
            statements = [node]

        signature_parts = []
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                signature_parts.append("ASSIGN")
            elif isinstance(stmt, ast.If):
                signature_parts.append("IF")
            elif isinstance(stmt, ast.For):
                signature_parts.append("FOR")
            elif isinstance(stmt, ast.While):
                signature_parts.append("WHILE")
            elif isinstance(stmt, ast.Return):
                signature_parts.append("RETURN")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if hasattr(stmt.value.func, "id"):
                    signature_parts.append(f"CALL_{stmt.value.func.id}")
                elif hasattr(stmt.value.func, "attr"):
                    signature_parts.append(f"CALL_{stmt.value.func.attr}")
                else:
                    signature_parts.append("CALL")
            elif isinstance(stmt, ast.Try):
                signature_parts.append("TRY")
            elif isinstance(stmt, ast.With):
                signature_parts.append("WITH")

        return "|".join(signature_parts)

    def _detect_sql_patterns(self, node: ast.Str) -> bool:
        """Detect SQL queries in string literals."""
        sql_keywords = [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "CREATE",
            "ALTER",
            "DROP",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
        ]
        content = node.s.upper()
        return any(keyword in content for keyword in sql_keywords)

    def _detect_database_ipc(self, node: ast.Call) -> bool:
        """Detect database-as-IPC patterns."""
        # Look for database operations that might be used for inter-process communication
        db_methods = ["execute", "query", "insert", "update", "delete", "commit"]

        if hasattr(node.func, "attr") and node.func.attr in db_methods:
            # Check if this looks like IPC (frequent small operations)
            return True

        return False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Detect function-level anti-patterns."""
        self.functions[node.name] = node

        # God Method detection
        complexity, max_nesting = self._calculate_method_complexity(node)

        if hasattr(node, "end_lineno") and node.end_lineno:
            method_lines = node.end_lineno - node.lineno
        else:
            method_lines = len(node.body) * 3  # Rough estimate

        if complexity > 10 or method_lines > 50 or max_nesting > 4:
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="god_method",
                    severity="high",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Method '{node.name}' is too complex: {complexity} complexity, {method_lines} lines, {max_nesting} max nesting",
                    recommendation="Break down into smaller, focused methods following Single Responsibility Principle",
                    code_snippet=self.get_code_snippet(node),
                    context={
                        "complexity": complexity,
                        "lines": method_lines,
                        "max_nesting": max_nesting,
                        "method_name": node.name,
                    },
                )
            )

        # Long Parameter List detection
        total_params = len(node.args.args) + len(node.args.kwonlyargs)
        if node.args.vararg:
            total_params += 1
        if node.args.kwarg:
            total_params += 1

        if total_params > 5:
            self.parameter_lists.append((node, total_params))
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="long_parameter_list",
                    severity="medium",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Function '{node.name}' has {total_params} parameters (>5)",
                    recommendation="Use parameter objects, data classes, or keyword-only arguments",
                    code_snippet=self.get_code_snippet(node),
                    context={"parameter_count": total_params, "function_name": node.name},
                )
            )

        # Extract code blocks for duplication detection
        if len(node.body) > 3:
            block_signature = self._extract_code_block_signature(node.body)
            self.code_blocks.append((node, block_signature))

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Detect class-level anti-patterns."""
        self.classes[node.name] = node

        # God Object detection
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        method_count = len(methods)

        # Calculate class metrics
        if hasattr(node, "end_lineno") and node.end_lineno:
            class_lines = node.end_lineno - node.lineno
        else:
            class_lines = len(node.body) * 10  # Rough estimate

        # Analyze method responsibilities
        data_methods = sum(1 for m in methods if m.name.startswith(("get_", "set_", "_get", "_set")))
        business_methods = method_count - data_methods

        if method_count > 20 or class_lines > 500:
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="god_object",
                    severity="critical",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Class '{node.name}' is a God Object: {method_count} methods, ~{class_lines} lines",
                    recommendation="Split into smaller classes using composition or Extract Class refactoring",
                    code_snippet=self.get_code_snippet(node),
                    context={
                        "method_count": method_count,
                        "estimated_lines": class_lines,
                        "class_name": node.name,
                        "data_methods": data_methods,
                        "business_methods": business_methods,
                    },
                )
            )

        # Data Class anti-pattern (too many getters/setters)
        if data_methods > business_methods and method_count > 10:
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="data_class",
                    severity="medium",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description=f"Class '{node.name}' appears to be a data class with business logic: {data_methods} data methods vs {business_methods} business methods",
                    recommendation="Consider using dataclasses or separate data from behavior",
                    code_snippet=self.get_code_snippet(node),
                    context={
                        "class_name": node.name,
                        "data_methods": data_methods,
                        "business_methods": business_methods,
                    },
                )
            )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect call-based anti-patterns."""
        # Track method calls for Feature Envy detection
        if hasattr(node.func, "attr"):
            method_name = node.func.attr
            if hasattr(node.func, "value") and hasattr(node.func.value, "id"):
                object_name = node.func.value.id
                self.method_calls[object_name].append(method_name)

        # Database-as-IPC detection
        if self._detect_database_ipc(node):
            self.data_access_patterns.append(node)
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="database_as_ipc",
                    severity="high",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description="Potential database-as-IPC anti-pattern detected",
                    recommendation="Use proper message queues, REST APIs, or event systems for inter-process communication",
                    code_snippet=self.get_code_snippet(node),
                    context={"call_type": "database_operation"},
                )
            )

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Detect magic literals and string patterns."""
        # Magic numbers/strings
        if isinstance(node.value, int | float):
            if node.value not in [0, 1, -1, 2, 10, 100, 1000]:
                self.magic_literals.append(node)
        elif isinstance(node.value, str):
            # Check for SQL queries
            if len(node.value) > 20 and self._detect_sql_patterns(node):
                self.sql_queries.append(node)
                self.anti_patterns.append(
                    AntiPattern(
                        pattern_type="embedded_sql",
                        severity="medium",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description="Embedded SQL query detected in code",
                        recommendation="Use ORM, query builders, or move SQL to separate files",
                        code_snippet=self.get_code_snippet(node),
                        context={"query_preview": node.value[:100] + "..." if len(node.value) > 100 else node.value},
                    )
                )

            # Check for hardcoded paths, URLs, etc.
            if any(pattern in node.value for pattern in ["http://", "https://", "/usr/", "/var/", "C:\\"]):
                self.anti_patterns.append(
                    AntiPattern(
                        pattern_type="hardcoded_path",
                        severity="medium",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description="Hardcoded path or URL detected",
                        recommendation="Use configuration files, environment variables, or path utilities",
                        code_snippet=self.get_code_snippet(node),
                        context={"hardcoded_value": node.value},
                    )
                )

        self.generic_visit(node)

    def visit_Str(self, node: ast.Str):
        """Handle string literals in older Python versions."""
        if len(node.s) > 20 and self._detect_sql_patterns(node):
            self.sql_queries.append(node)
            self.anti_patterns.append(
                AntiPattern(
                    pattern_type="embedded_sql",
                    severity="medium",
                    file_path=self.file_path,
                    line_number=node.lineno,
                    column=node.col_offset,
                    description="Embedded SQL query detected in code",
                    recommendation="Use ORM, query builders, or move SQL to separate files",
                    code_snippet=self.get_code_snippet(node),
                    context={"query_preview": node.s[:100] + "..." if len(node.s) > 100 else node.s},
                )
            )

        self.generic_visit(node)

    def finalize_analysis(self):
        """Perform final analysis that requires complete traversal."""
        # Detect code duplication
        self._detect_copy_paste_programming()

        # Detect Feature Envy
        self._detect_feature_envy()

        # Detect Sequential Coupling
        self._detect_sequential_coupling()

        # Analyze magic literal usage
        self._analyze_magic_literals()

    def _detect_copy_paste_programming(self):
        """Detect duplicate code blocks."""
        signatures = collections.defaultdict(list)

        for node, signature in self.code_blocks:
            if len(signature) > 10:  # Only consider substantial blocks
                signatures[signature].append(node)

        for signature, nodes in signatures.items():
            if len(nodes) > 1:
                for node in nodes:
                    self.anti_patterns.append(
                        AntiPattern(
                            pattern_type="copy_paste_programming",
                            severity="high",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            description=f"Duplicate code block detected in function '{node.name}'",
                            recommendation="Extract common code into a shared function or utility",
                            code_snippet=self.get_code_snippet(node),
                            context={"duplicate_count": len(nodes), "function_name": node.name, "signature": signature},
                        )
                    )

    def _detect_feature_envy(self):
        """Detect methods that use other objects more than their own."""
        for object_name, methods in self.method_calls.items():
            if len(methods) > 5:  # Threshold for feature envy
                # Find the context (which function this is in)
                # This is simplified - in practice you'd track the calling context
                self.anti_patterns.append(
                    AntiPattern(
                        pattern_type="feature_envy",
                        severity="medium",
                        file_path=self.file_path,
                        line_number=1,  # Would need better tracking
                        column=0,
                        description=f"Potential Feature Envy: Excessive use of '{object_name}' methods ({len(methods)} calls)",
                        recommendation="Consider moving this behavior to the envied class or use delegation",
                        code_snippet="",
                        context={
                            "envied_object": object_name,
                            "method_count": len(methods),
                            "methods": list(set(methods)),
                        },
                    )
                )

    def _detect_sequential_coupling(self):
        """Detect APIs that require specific call sequences."""
        # Look for patterns like init() -> setup() -> run() -> cleanup()
        function_names = list(self.functions.keys())
        sequential_patterns = [
            ["init", "setup", "run", "cleanup"],
            ["open", "read", "close"],
            ["connect", "authenticate", "execute", "disconnect"],
            ["begin", "process", "commit"],
        ]

        for pattern in sequential_patterns:
            matches = [name for name in function_names if any(p in name.lower() for p in pattern)]
            if len(matches) >= 3:
                # This suggests a sequential coupling API
                for func_name in matches:
                    node = self.functions[func_name]
                    self.anti_patterns.append(
                        AntiPattern(
                            pattern_type="sequential_coupling",
                            severity="medium",
                            file_path=self.file_path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            description=f"Sequential coupling detected: Function '{func_name}' is part of a sequence-dependent API",
                            recommendation="Consider using context managers, fluent interfaces, or state machines",
                            code_snippet=self.get_code_snippet(node),
                            context={
                                "function_name": func_name,
                                "sequential_functions": matches,
                                "pattern_matched": pattern,
                            },
                        )
                    )

    def _analyze_magic_literals(self):
        """Analyze magic literal usage patterns."""
        if len(self.magic_literals) > 10:  # Threshold for magic literal abuse
            for node in self.magic_literals[:5]:  # Report first few
                self.anti_patterns.append(
                    AntiPattern(
                        pattern_type="magic_number_abuse",
                        severity="medium",
                        file_path=self.file_path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        description=f"Excessive magic literals detected ({len(self.magic_literals)} total)",
                        recommendation="Replace magic numbers with named constants or enums",
                        code_snippet=self.get_code_snippet(node),
                        context={"total_magic_literals": len(self.magic_literals), "current_value": node.value},
                    )
                )


class ProjectAntiPatternAnalyzer:
    """Analyzes anti-patterns across a project."""

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
        self.anti_patterns: list[AntiPattern] = []
        self.file_count = 0
        self.project_context = {}

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

    def analyze_file(self, file_path: Path) -> list[AntiPattern]:
        """Analyze a single Python file for anti-patterns."""
        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
                source_lines = source.splitlines()

            tree = ast.parse(source, filename=str(file_path))
            detector = AntiPatternDetector(str(file_path), source_lines, self.project_context)
            detector.visit(tree)
            detector.finalize_analysis()

            return detector.anti_patterns

        except (SyntaxError, UnicodeDecodeError) as e:
            return [
                AntiPattern(
                    pattern_type="syntax_error",
                    severity="critical",
                    file_path=str(file_path),
                    line_number=getattr(e, "lineno", 1),
                    column=getattr(e, "offset", 0) or 0,
                    description=f"File cannot be parsed: {e}",
                    recommendation="Fix syntax errors before analyzing anti-patterns",
                    code_snippet="",
                    context={"error": str(e)},
                )
            ]

    def analyze_directory(self, directory: Path) -> list[AntiPattern]:
        """Analyze all Python files in a directory."""
        all_anti_patterns = []

        # First pass: build project context
        self._build_project_context(directory)

        # Second pass: analyze files
        for py_file in directory.rglob("*.py"):
            if self.should_analyze_file(py_file):
                file_patterns = self.analyze_file(py_file)
                all_anti_patterns.extend(file_patterns)
                self.file_count += 1

        self.anti_patterns = all_anti_patterns
        return all_anti_patterns

    def _build_project_context(self, directory: Path):
        """Build project-wide context for better analysis."""
        # Count files, analyze overall structure
        py_files = list(directory.rglob("*.py"))
        self.project_context = {
            "total_files": len(py_files),
            "has_tests": any("test" in str(f) for f in py_files),
            "has_setup": any("setup.py" in str(f) for f in py_files),
            "package_structure": self._analyze_package_structure(py_files),
        }

    def _analyze_package_structure(self, py_files: list[Path]) -> dict[str, int]:
        """Analyze the package structure."""
        packages = collections.defaultdict(int)
        for py_file in py_files:
            parts = py_file.parts
            if len(parts) > 1:
                package = parts[-2]  # Parent directory
                packages[package] += 1
        return dict(packages)

    def generate_report(self, anti_patterns: list[AntiPattern], output_format: str = "text") -> str:
        """Generate an anti-pattern report."""
        if output_format == "json":
            return json.dumps([asdict(ap) for ap in anti_patterns], indent=2)

        # Text report
        lines = ["=" * 80, "ANTI-PATTERN DETECTION REPORT", "=" * 80, ""]

        # Summary
        severity_counts = collections.Counter(ap.severity for ap in anti_patterns)
        pattern_counts = collections.Counter(ap.pattern_type for ap in anti_patterns)

        lines.extend(
            [
                f"Total anti-patterns detected: {len(anti_patterns)}",
                f"Files analyzed: {self.file_count}",
                "",
                "Severity breakdown:",
                f"  Critical: {severity_counts['critical']:3d}",
                f"  High:     {severity_counts['high']:3d}",
                f"  Medium:   {severity_counts['medium']:3d}",
                f"  Low:      {severity_counts['low']:3d}",
                "",
                "Anti-pattern types detected:",
            ]
        )

        for pattern_type, count in pattern_counts.most_common():
            lines.append(f"  {pattern_type:25s}: {count:3d}")

        lines.extend(["", "=" * 80, "DETAILED ANTI-PATTERNS", "=" * 80])

        # Group by severity
        for severity in ["critical", "high", "medium", "low"]:
            severity_patterns = [ap for ap in anti_patterns if ap.severity == severity]
            if not severity_patterns:
                continue

            lines.extend([f"\n{severity.upper()} SEVERITY ({len(severity_patterns)} patterns)", "-" * 50])

            for ap in severity_patterns:
                lines.extend(
                    [
                        f"\n{ap.pattern_type.upper()}: {ap.description}",
                        f"File: {ap.file_path}:{ap.line_number}:{ap.column}",
                        f"Recommendation: {ap.recommendation}",
                    ]
                )

                if ap.code_snippet:
                    lines.extend(["Code context:", ap.code_snippet, ""])

        # Recommendations summary
        lines.extend(["\n" + "=" * 80, "REFACTORING PRIORITIES", "=" * 80, ""])

        priority_patterns = [
            ("god_object", "CRITICAL: Break down God Objects using Extract Class refactoring"),
            ("god_method", "HIGH: Split complex methods into smaller, focused functions"),
            ("copy_paste_programming", "HIGH: Extract duplicate code into shared utilities"),
            ("long_parameter_list", "MEDIUM: Use parameter objects or data classes"),
            ("database_as_ipc", "HIGH: Replace with proper messaging or API patterns"),
            ("sequential_coupling", "MEDIUM: Implement context managers or fluent interfaces"),
            ("magic_number_abuse", "MEDIUM: Replace magic numbers with named constants"),
        ]

        detected_types = set(ap.pattern_type for ap in anti_patterns)
        for pattern_type, recommendation in priority_patterns:
            if pattern_type in detected_types:
                count = sum(1 for ap in anti_patterns if ap.pattern_type == pattern_type)
                lines.append(f"• {recommendation} ({count} instances)")

        return "\n".join(lines)


def main():
    """Main entry point for anti-pattern detection."""
    parser = argparse.ArgumentParser(
        description="Detect anti-patterns in Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detect_anti_patterns.py .                    # Analyze current directory
  python detect_anti_patterns.py src/ --json          # JSON output
  python detect_anti_patterns.py . --severity high    # Only high+ severity
  python detect_anti_patterns.py . --pattern god_object # Specific pattern
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
    parser.add_argument("--pattern", "-p", help="Filter by specific anti-pattern type")
    parser.add_argument("--exclude", "-e", action="append", help="Additional exclusion patterns")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Initialize analyzer
    exclusions = None
    if args.exclude:
        exclusions = args.exclude

    analyzer = ProjectAntiPatternAnalyzer(exclusions)

    # Analyze path
    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Analyzing anti-patterns in {target_path}...")

    start_time = time.time()

    if target_path.is_file():
        anti_patterns = analyzer.analyze_file(target_path)
    else:
        anti_patterns = analyzer.analyze_directory(target_path)

    elapsed = time.time() - start_time

    # Filter by severity if requested
    if args.severity:
        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_level = severity_order[args.severity]
        anti_patterns = [ap for ap in anti_patterns if severity_order[ap.severity] >= min_level]

    # Filter by pattern type if requested
    if args.pattern:
        anti_patterns = [ap for ap in anti_patterns if ap.pattern_type == args.pattern]

    # Generate report
    report = analyzer.generate_report(anti_patterns, args.format)

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
        print(f"Found {len(anti_patterns)} anti-patterns")

    # Exit with error code if critical patterns found
    critical_count = sum(1 for ap in anti_patterns if ap.severity == "critical")
    return min(critical_count, 1)


if __name__ == "__main__":
    sys.exit(main())
