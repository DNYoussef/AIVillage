#!/usr/bin/env python3
"""
Automated Stub Scanner - AIVillage Critical Stubs Audit Tool

Scans the entire repository for stub functions/classes and categorizes them
by priority for the Critical Stubs audit gate.

Categories:
- CRITICAL: Runtime-blocking stubs (pass, NotImplementedError, raw TODO)
- WARNING: Documentation stubs (docstring-only functions)
- INFO: Commented stubs (# TODO, # FIXME, etc.)

Zero-tolerance acceptance gate: No CRITICAL stubs allowed.
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StubReport:
    """Stub detection report."""

    file_path: str
    line_number: int
    function_name: str
    class_name: str
    stub_type: str  # 'pass', 'NotImplementedError', 'docstring_only', 'todo_comment'
    severity: str  # 'CRITICAL', 'WARNING', 'INFO'
    context: str  # Code snippet


class StubDetectorVisitor(ast.NodeVisitor):
    """AST visitor to detect stub functions and classes."""

    def __init__(self, file_path: str, source_lines: list[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.stubs: list[StubReport] = []
        self.class_stack: list[str] = []

    def visit_ClassDef(self, node):
        """Visit class definition to track nested classes."""
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        """Visit function definition to detect stubs."""
        self._check_function_for_stubs(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition to detect stubs."""
        self._check_function_for_stubs(node)
        self.generic_visit(node)

    def _check_function_for_stubs(self, node):
        """Check if function/method is a stub."""
        class_name = ".".join(self.class_stack) if self.class_stack else ""
        function_name = node.name

        # Get function body
        if not node.body:
            return

        # Check for different stub patterns
        body_statements = [
            stmt
            for stmt in node.body
            if not isinstance(stmt, ast.Expr)
            or not isinstance(stmt.value, ast.Constant)
        ]

        # Pattern 1: Only pass statement
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self._report_stub(
                node.lineno,
                function_name,
                class_name,
                stub_type="pass",
                severity="CRITICAL",
                context=self._get_context(node.lineno),
            )
            return

        # Pattern 2: Only NotImplementedError
        if len(body_statements) == 1:
            stmt = body_statements[0]
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if (
                    isinstance(stmt.exc.func, ast.Name)
                    and stmt.exc.func.id == "NotImplementedError"
                ):
                    self._report_stub(
                        node.lineno,
                        function_name,
                        class_name,
                        stub_type="NotImplementedError",
                        severity="CRITICAL",
                        context=self._get_context(node.lineno),
                    )
                    return

        # Pattern 3: Only docstring (no real implementation)
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        ):
            if isinstance(node.body[0].value.value, str):  # Docstring
                self._report_stub(
                    node.lineno,
                    function_name,
                    class_name,
                    stub_type="docstring_only",
                    severity="WARNING",
                    context=self._get_context(node.lineno),
                )
                return

        # Pattern 4: Function with TODO/FIXME comments
        for i, line in enumerate(
            self.source_lines[node.lineno - 1 : node.end_lineno], node.lineno
        ):
            if any(
                marker in line.upper()
                for marker in ["# TODO", "# FIXME", "# STUB", "# PLACEHOLDER"]
            ):
                self._report_stub(
                    i,
                    function_name,
                    class_name,
                    stub_type="todo_comment",
                    severity="INFO",
                    context=line.strip(),
                )

    def _report_stub(
        self,
        line_num: int,
        func_name: str,
        class_name: str,
        stub_type: str,
        severity: str,
        context: str,
    ):
        """Add a stub report."""
        self.stubs.append(
            StubReport(
                file_path=self.file_path,
                line_number=line_num,
                function_name=func_name,
                class_name=class_name,
                stub_type=stub_type,
                severity=severity,
                context=context,
            )
        )

    def _get_context(self, line_num: int, context_lines: int = 3) -> str:
        """Get code context around the line."""
        start = max(0, line_num - context_lines - 1)
        end = min(len(self.source_lines), line_num + context_lines)

        context_lines_list = []
        for i in range(start, end):
            prefix = ">>>" if i == line_num - 1 else "   "
            context_lines_list.append(
                f"{prefix} {i + 1:4d}: {self.source_lines[i].rstrip()}"
            )

        return "\n".join(context_lines_list)


def scan_file(file_path: Path) -> list[StubReport]:
    """Scan a single Python file for stubs."""
    try:
        with open(file_path, encoding="utf-8") as f:
            source = f.read()
            source_lines = source.splitlines()

        # Parse AST
        tree = ast.parse(source, filename=str(file_path))

        # Visit nodes to detect stubs
        visitor = StubDetectorVisitor(str(file_path), source_lines)
        visitor.visit(tree)

        return visitor.stubs

    except (UnicodeDecodeError, SyntaxError) as e:
        print(f"WARNING: Could not parse {file_path}: {e}")
        return []


def scan_repository(
    root_path: Path, exclude_patterns: set[str] = None
) -> dict[str, list[StubReport]]:
    """Scan entire repository for stubs."""
    if exclude_patterns is None:
        exclude_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "venv",
            "env",
            ".venv",
            ".env",
            "tmp_",
            "archive",
            "deprecated",
            "legacy",
        }

    all_stubs = {}

    # Find all Python files
    for py_file in root_path.rglob("*.py"):
        # Skip excluded directories
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue

        stubs = scan_file(py_file)
        if stubs:
            relative_path = py_file.relative_to(root_path)
            all_stubs[str(relative_path)] = stubs

    return all_stubs


def generate_report(
    all_stubs: dict[str, list[StubReport]], output_file: Path = None
) -> bool:
    """Generate comprehensive stub report."""

    # Categorize stubs
    critical_stubs = []
    warning_stubs = []
    info_stubs = []

    for file_path, stubs in all_stubs.items():
        for stub in stubs:
            if stub.severity == "CRITICAL":
                critical_stubs.append(stub)
            elif stub.severity == "WARNING":
                warning_stubs.append(stub)
            else:
                info_stubs.append(stub)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("AIVillage Critical Stubs Audit Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary
    total_files_with_stubs = len(all_stubs)
    total_critical = len(critical_stubs)
    total_warnings = len(warning_stubs)
    total_info = len(info_stubs)

    report_lines.append("## SUMMARY")
    report_lines.append(f"Files with stubs: {total_files_with_stubs}")
    report_lines.append(f"[CRITICAL] stubs: {total_critical}")
    report_lines.append(f"[WARNING] stubs: {total_warnings}")
    report_lines.append(f"[INFO] stubs: {total_info}")
    report_lines.append("")

    # Acceptance gate status
    gate_status = "PASS" if total_critical == 0 else "FAIL"
    report_lines.append(f"## ACCEPTANCE GATE STATUS: {gate_status}")
    if total_critical > 0:
        report_lines.append(
            "FAIL: Repository contains CRITICAL stubs that block production deployment."
        )
        report_lines.append("   All CRITICAL stubs must be fixed before acceptance.")
    else:
        report_lines.append(
            "PASS: No CRITICAL stubs found. Repository ready for production."
        )
    report_lines.append("")

    # Critical stubs detail
    if critical_stubs:
        report_lines.append("## [CRITICAL] STUBS (Must Fix)")
        report_lines.append(
            "These stubs will cause runtime failures and block production deployment:"
        )
        report_lines.append("")

        for i, stub in enumerate(critical_stubs, 1):
            report_lines.append(f"### {i}. {stub.file_path}:{stub.line_number}")
            location = (
                f"{stub.class_name}.{stub.function_name}"
                if stub.class_name
                else stub.function_name
            )
            report_lines.append(f"**Location:** `{location}`")
            report_lines.append(f"**Stub Type:** `{stub.stub_type}`")
            report_lines.append("**Context:**")
            report_lines.append("```python")
            report_lines.append(stub.context)
            report_lines.append("```")
            report_lines.append("")

    # Warning stubs summary
    if warning_stubs:
        report_lines.append("## [WARNING] STUBS (Should Fix)")
        report_lines.append(
            f"Found {len(warning_stubs)} docstring-only functions. These should be implemented or documented as intentionally empty."
        )
        report_lines.append("")

        for stub in warning_stubs[:10]:  # Show first 10
            location = (
                f"{stub.class_name}.{stub.function_name}"
                if stub.class_name
                else stub.function_name
            )
            report_lines.append(
                f"- `{stub.file_path}:{stub.line_number}` - `{location}`"
            )

        if len(warning_stubs) > 10:
            report_lines.append(f"... and {len(warning_stubs) - 10} more")
        report_lines.append("")

    # Info stubs summary
    if info_stubs:
        report_lines.append("## [INFO] STUBS (For Reference)")
        report_lines.append(
            f"Found {len(info_stubs)} TODO/FIXME comments. These are tracked for future work."
        )
        report_lines.append("")

    # File-by-file breakdown
    report_lines.append("## FILES WITH STUBS")
    for file_path, stubs in sorted(all_stubs.items()):
        critical_count = sum(1 for s in stubs if s.severity == "CRITICAL")
        warning_count = sum(1 for s in stubs if s.severity == "WARNING")
        info_count = sum(1 for s in stubs if s.severity == "INFO")

        status = (
            "[CRITICAL]"
            if critical_count > 0
            else "[WARNING]"
            if warning_count > 0
            else "[INFO]"
        )
        report_lines.append(
            f"{status} `{file_path}` - {critical_count}C/{warning_count}W/{info_count}I"
        )

    report_lines.append("")
    report_lines.append("## NEXT STEPS")
    if total_critical > 0:
        report_lines.append("1. [CRITICAL] Fix all CRITICAL stubs (runtime blockers)")
        report_lines.append(
            "2. [WARNING] Review WARNING stubs (docstring-only functions)"
        )
        report_lines.append(
            "3. [INFO] Prioritize INFO stubs (TODO comments) for future work"
        )
    else:
        report_lines.append("1. [PASS] All CRITICAL stubs resolved")
        report_lines.append("2. [WARNING] Review remaining WARNING stubs if desired")
        report_lines.append("3. [INFO] Address INFO stubs (TODOs) in future iterations")

    # Output report
    report_text = "\n".join(report_lines)

    if output_file:
        output_file.write_text(report_text, encoding="utf-8")
        print(f"Full report saved to: {output_file}")
    else:
        print(report_text)

    # Return pass/fail status
    return total_critical == 0


def main():
    """Main entry point for stub scanner."""
    if len(sys.argv) > 1:
        root_path = Path(sys.argv[1])
    else:
        root_path = Path.cwd()

    if not root_path.exists():
        print(f"ERROR: Path does not exist: {root_path}")
        sys.exit(1)

    print(f"Scanning {root_path} for stubs...")

    # Scan repository
    all_stubs = scan_repository(root_path)

    # Generate report
    output_file = root_path / "STUB_AUDIT_REPORT.md"
    passed = generate_report(all_stubs, output_file)

    # Exit with appropriate code
    if passed:
        print("\nSTUB AUDIT PASSED - No critical stubs found")
        sys.exit(0)
    else:
        print("\nSTUB AUDIT FAILED - Critical stubs must be fixed")
        sys.exit(1)


if __name__ == "__main__":
    main()
