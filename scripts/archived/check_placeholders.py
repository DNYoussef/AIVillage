#!/usr/bin/env python3
"""Placeholder and Stub Detection Script.

This script scans the codebase for placeholder functions, stub implementations,
and other non-functional code patterns that need to be implemented.
"""

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlaceholderIssue:
    """Represents a placeholder or stub issue found in code."""

    file_path: str
    line_number: int
    function_name: str
    issue_type: str
    description: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    code_snippet: str


class PlaceholderDetector:
    """Detects placeholder functions and stub implementations."""

    def __init__(self, src_dir: str = "src") -> None:
        self.src_dir = Path(src_dir)
        self.issues: list[PlaceholderIssue] = []

        # Patterns to detect different types of placeholders
        self.stub_patterns = {
            "empty_return_none": re.compile(r"^\s*return\s+None\s*$"),
            "empty_return_dict": re.compile(r"^\s*return\s+\{\}\s*$"),
            "empty_return_list": re.compile(r"^\s*return\s+\[\]\s*$"),
            "just_pass": re.compile(r"^\s*pass\s*$"),
            "just_ellipsis": re.compile(r"^\s*\.\.\.\s*$"),
            "todo_comment": re.compile(r"#.*TODO|#.*FIXME|#.*XXX|#.*HACK"),
            "placeholder_comment": re.compile(
                r"#.*placeholder|#.*stub|#.*not.implemented", re.IGNORECASE
            ),
        }

        # Keywords that indicate placeholders
        self.placeholder_keywords = [
            "NotImplementedError",
            "raise NotImplementedError",
            "placeholder",
            "stub",
            "mock",
            "dummy",
            "fake",
            "test_data",
        ]

    def scan_directory(self) -> list[PlaceholderIssue]:
        """Scan the source directory for placeholder issues."""
        print(f"Scanning {self.src_dir} for placeholder functions...")

        for py_file in self.src_dir.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            self._scan_file(py_file)

        return self.issues

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            "__pycache__",
            ".pytest_cache",
            "tests/",
            "deprecated/",
            "experimental/",
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def _scan_file(self, file_path: Path) -> None:
        """Scan a single Python file for placeholder issues."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Parse AST for function analysis
            try:
                tree = ast.parse(content, filename=str(file_path))
                self._analyze_ast(tree, file_path, lines)
            except SyntaxError as e:
                print(f"Syntax error in {file_path}: {e}")
                return

            # Pattern-based analysis
            self._analyze_patterns(file_path, lines)

        except Exception as e:
            print(f"Error scanning {file_path}: {e}")

    def _analyze_ast(self, tree: ast.AST, file_path: Path, lines: list[str]) -> None:
        """Analyze AST for function-level issues."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                self._analyze_function(node, file_path, lines)

    def _analyze_function(
        self, func_node: ast.FunctionDef, file_path: Path, lines: list[str]
    ) -> None:
        """Analyze a function for placeholder patterns."""
        func_name = func_node.name

        # Skip abstract methods and dunder methods
        if (func_name.startswith("__") and func_name.endswith("__")) or any(
            isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
            for decorator in func_node.decorator_list
        ):
            return

        # Get function body
        if not func_node.body:
            return

        # Check for single-statement functions that might be stubs
        if len(func_node.body) == 1:
            stmt = func_node.body[0]

            # Check for raise NotImplementedError
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if (
                    hasattr(stmt.exc.func, "id")
                    and stmt.exc.func.id == "NotImplementedError"
                ):
                    # This is intentional - check if it has proper documentation
                    if not self._has_proper_todo_documentation(stmt):
                        severity = "high"
                    else:
                        severity = "medium"  # Documented unimplemented feature

                    self._add_issue(
                        file_path,
                        stmt.lineno,
                        func_name,
                        "not_implemented",
                        "Function raises NotImplementedError",
                        severity,
                        self._get_code_snippet(lines, stmt.lineno),
                    )

            # Check for simple return statements
            elif isinstance(stmt, ast.Return):
                return_value = stmt.value

                if return_value is None:
                    # return (no value) - might be intentional
                    pass
                elif isinstance(return_value, ast.Constant):
                    if return_value.value is None:
                        self._add_issue(
                            file_path,
                            stmt.lineno,
                            func_name,
                            "return_none",
                            "Function only returns None",
                            "medium",
                            self._get_code_snippet(lines, stmt.lineno),
                        )
                elif (
                    isinstance(return_value, ast.Dict | ast.List)
                    and len(
                        return_value.elts
                        if hasattr(return_value, "elts")
                        else (
                            return_value.values
                            if hasattr(return_value, "values")
                            else []
                        )
                    )
                    == 0
                ):
                    self._add_issue(
                        file_path,
                        stmt.lineno,
                        func_name,
                        "empty_return",
                        "Function returns empty collection",
                        "medium",
                        self._get_code_snippet(lines, stmt.lineno),
                    )

            # Check for pass statement
            elif isinstance(stmt, ast.Pass):
                # Check if this is an abstract method (acceptable)
                if not self._is_abstract_method(func_node):
                    self._add_issue(
                        file_path,
                        stmt.lineno,
                        func_name,
                        "pass_only",
                        "Function only contains pass statement",
                        "high",
                        self._get_code_snippet(lines, stmt.lineno),
                    )

    def _analyze_patterns(self, file_path: Path, lines: list[str]) -> None:
        """Analyze file for pattern-based placeholder detection."""
        for i, line in enumerate(lines, 1):
            # Check for TODO/FIXME comments that might indicate incomplete work
            if self.stub_patterns["todo_comment"].search(line):
                # Only flag TODOs that seem to indicate incomplete implementation
                if any(
                    keyword in line.lower()
                    for keyword in ["implement", "finish", "complete"]
                ):
                    self._add_issue(
                        file_path,
                        i,
                        "N/A",
                        "todo_implementation",
                        "TODO comment indicates incomplete implementation",
                        "low",
                        line.strip(),
                    )

            # Check for placeholder comments
            if self.stub_patterns["placeholder_comment"].search(line):
                self._add_issue(
                    file_path,
                    i,
                    "N/A",
                    "placeholder_comment",
                    "Comment indicates placeholder code",
                    "medium",
                    line.strip(),
                )

    def _has_proper_todo_documentation(self, node: ast.Raise) -> bool:
        """Check if NotImplementedError has proper documentation/issue tracking."""
        if isinstance(node.exc, ast.Call) and len(node.exc.args) > 0:
            if isinstance(node.exc.args[0], ast.Constant):
                message = node.exc.args[0].value
                if isinstance(message, str):
                    # Check for issue tracking URLs or proper documentation
                    return any(
                        pattern in message.lower()
                        for pattern in [
                            "github.com/issues",
                            "track progress",
                            "roadmap",
                            "planned feature",
                            "atlantis",
                        ]
                    )
        return False

    def _is_abstract_method(self, func_node: ast.FunctionDef) -> bool:
        """Check if function is an abstract method."""
        for decorator in func_node.decorator_list:
            if (
                isinstance(decorator, ast.Name) and decorator.id == "abstractmethod"
            ) or (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "abstractmethod"
            ):
                return True
        return False

    def _add_issue(
        self,
        file_path: Path,
        line_number: int,
        function_name: str,
        issue_type: str,
        description: str,
        severity: str,
        code_snippet: str,
    ) -> None:
        """Add a placeholder issue to the list."""
        issue = PlaceholderIssue(
            file_path=str(file_path.relative_to(Path.cwd())),
            line_number=line_number,
            function_name=function_name,
            issue_type=issue_type,
            description=description,
            severity=severity,
            code_snippet=code_snippet,
        )
        self.issues.append(issue)

    def _get_code_snippet(
        self, lines: list[str], line_number: int, context: int = 1
    ) -> str:
        """Get code snippet around the line."""
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        return "\n".join(lines[start:end])

    def generate_report(self) -> str:
        """Generate a report of all placeholder issues."""
        if not self.issues:
            return "🎉 No placeholder issues found!"

        # Group by severity
        critical = [i for i in self.issues if i.severity == "critical"]
        high = [i for i in self.issues if i.severity == "high"]
        medium = [i for i in self.issues if i.severity == "medium"]
        low = [i for i in self.issues if i.severity == "low"]

        report = []
        report.append("# Placeholder and Stub Detection Report")
        report.append("")
        report.append(f"**Total issues found: {len(self.issues)}**")
        report.append("")
        report.append(f"- 🔴 Critical: {len(critical)}")
        report.append(f"- 🟠 High: {len(high)}")
        report.append(f"- 🟡 Medium: {len(medium)}")
        report.append(f"- 🔵 Low: {len(low)}")
        report.append("")

        for severity, issues in [
            ("Critical", critical),
            ("High", high),
            ("Medium", medium),
            ("Low", low),
        ]:
            if issues:
                report.append(f"## {severity} Issues")
                report.append("")

                for issue in issues:
                    report.append(f"### {issue.file_path}:{issue.line_number}")
                    if issue.function_name != "N/A":
                        report.append(f"**Function:** `{issue.function_name}`")
                    report.append(f"**Type:** {issue.issue_type}")
                    report.append(f"**Description:** {issue.description}")
                    report.append("")
                    report.append("```python")
                    report.append(issue.code_snippet)
                    report.append("```")
                    report.append("")

        return "\n".join(report)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect placeholder functions and stubs"
    )
    parser.add_argument("--src-dir", default="src", help="Source directory to scan")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit with code 1 if critical issues found",
    )
    parser.add_argument(
        "--fail-on-high",
        action="store_true",
        help="Exit with code 1 if high or critical issues found",
    )

    args = parser.parse_args()

    if not os.path.exists(args.src_dir):
        print(f"Error: Source directory '{args.src_dir}' not found")
        sys.exit(1)

    detector = PlaceholderDetector(args.src_dir)
    issues = detector.scan_directory()
    report = detector.generate_report()

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Check exit conditions
    critical_count = len([i for i in issues if i.severity == "critical"])
    high_count = len([i for i in issues if i.severity == "high"])

    if args.fail_on_critical and critical_count > 0:
        print(f"\n❌ Exiting with error: {critical_count} critical issues found")
        sys.exit(1)

    if args.fail_on_high and (critical_count > 0 or high_count > 0):
        print(
            f"\n❌ Exiting with error: {critical_count + high_count} critical/high issues found"
        )
        sys.exit(1)

    if issues:
        print(f"\n⚠️  Found {len(issues)} placeholder issues")
    else:
        print("\n✅ No placeholder issues found")


if __name__ == "__main__":
    main()
