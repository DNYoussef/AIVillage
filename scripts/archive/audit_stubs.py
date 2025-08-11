#!/usr/bin/env python3
"""Comprehensive stub detection leveraging AIVillage's linting infrastructure.
Outputs GitHub issues, pre-commit compatible reports, and human-readable analysis.
"""

import ast
from collections import defaultdict
import json
from pathlib import Path


class StubAuditor:
    def __init__(self) -> None:
        self.stub_patterns = {
            "not_implemented": "Raises NotImplementedError",
            "misleading_log": "Logs action but does nothing",
            "empty_pass": "Empty function with pass",
            "todo_only": "Contains only TODO comment",
        }

    def analyze_file(self, filepath: Path) -> list[dict]:
        """Analyze a single file for stub implementations."""
        stubs = []

        try:
            with open(filepath, encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (SyntaxError, UnicodeDecodeError):
            return stubs

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                stub_type = self._classify_stub(node)
                if stub_type:
                    stubs.append(
                        {
                            "file": str(filepath),
                            "function": node.name,
                            "line": node.lineno,
                            "type": stub_type,
                            "description": self.stub_patterns[stub_type],
                        }
                    )

        return stubs

    def _classify_stub(self, node: ast.FunctionDef) -> str:
        """Classify the type of stub implementation."""
        # Check for NotImplementedError
        for child in ast.walk(node):
            if isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call):
                if getattr(child.exc.func, "id", "") == "NotImplementedError":
                    return "not_implemented"

        # Check for misleading logging (especially "evolving")
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            if self._is_misleading_log(node.body[0].value):
                return "misleading_log"

        # Check for empty pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return "empty_pass"

        return None

    def _is_misleading_log(self, node) -> bool:
        """Check if this is a misleading log statement."""
        if isinstance(node, ast.Call) and hasattr(node.func, "attr"):
            if node.func.attr in ["info", "debug", "warning"]:
                # Check for misleading words
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        text = str(arg.value).lower()
                        if any(word in text for word in ["evolving", "processing", "implementing"]):
                            return True
        return False

    def generate_report(self, all_stubs: list[dict]) -> None:
        """Generate multiple report formats."""
        # Human-readable report
        with open("stub_audit_report.md", "w") as f:
            f.write("# Stub Implementation Audit\n\n")
            f.write(f"Total stubs found: {len(all_stubs)}\n\n")

            by_type = defaultdict(list)
            for stub in all_stubs:
                by_type[stub["type"]].append(stub)

            for stub_type, stubs in by_type.items():
                f.write(f"## {self.stub_patterns[stub_type]} ({len(stubs)})\n\n")
                for stub in stubs:
                    f.write(f"- `{stub['file']}:{stub['line']}` - {stub['function']}()\n")

        # GitHub issues format
        issues = self._create_github_issues(all_stubs)
        with open("github_issues.json", "w") as f:
            json.dump(issues, f, indent=2)

    def _create_github_issues(self, stubs: list[dict]) -> list[dict]:
        """Create GitHub issues for PowerShell script."""
        by_file = defaultdict(list)
        for stub in stubs:
            by_file[stub["file"]].append(stub)

        issues = []
        for filepath, file_stubs in by_file.items():
            issue = {
                "title": f"Clean up {len(file_stubs)} stub implementations in {filepath}",
                "body": self._create_issue_body(filepath, file_stubs),
                "labels": ["technical-debt", "cleanup", "sprint-1"],
            }
            issues.append(issue)

        return issues

    def _create_issue_body(self, filepath: str, stubs: list[dict]) -> str:
        """Create detailed issue body."""
        body = f"## File: `{filepath}`\n\n"
        body += "### Stubs to Fix:\n\n"

        for stub in stubs:
            body += f"- [ ] Line {stub['line']}: `{stub['function']}()` - {stub['description']}\n"

        body += "\n### Acceptance Criteria:\n"
        body += "- [ ] All misleading implementations replaced with clear NotImplementedError\n"
        body += "- [ ] Each NotImplementedError includes tracking issue URL\n"
        body += "- [ ] Documentation updated to reflect actual status\n"

        return body


# Run the audit
if __name__ == "__main__":
    auditor = StubAuditor()
    all_stubs = []

    for py_file in Path().rglob("*.py"):
        if "__pycache__" not in str(py_file):
            stubs = auditor.analyze_file(py_file)
            all_stubs.extend(stubs)

    auditor.generate_report(all_stubs)
    print(f"Found {len(all_stubs)} stubs. See stub_audit_report.md for details.")
