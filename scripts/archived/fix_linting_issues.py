#!/usr/bin/env python3
"""Systematic linting issue detection and fixing.

Analyzes the codebase for linting issues and provides automated fixes where possible.
"""

import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LintingIssue:
    """Represents a linting issue."""

    file_path: str
    line_number: int
    column: int
    error_code: str
    error_message: str
    severity: str = "error"
    auto_fixable: bool = False
    suggested_fix: str = ""


@dataclass
class LintingReport:
    """Comprehensive linting analysis report."""

    total_files_scanned: int = 0
    total_issues: int = 0
    issues_by_type: dict[str, int] = field(default_factory=dict)
    issues_by_file: dict[str, list[LintingIssue]] = field(default_factory=lambda: defaultdict(list))
    auto_fixable_count: int = 0
    critical_issues: list[LintingIssue] = field(default_factory=list)


class LintingAnalyzer:
    """Analyzes and fixes linting issues in the codebase."""

    def __init__(self, source_dirs: list[str] = None):
        self.source_dirs = source_dirs or ["src", "tests", "scripts"]
        self.exclude_patterns = [
            "__pycache__",
            "*.pyc",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            ".pytest_cache",
            "deprecated",
        ]

        # Issue categorization
        self.critical_issues = [
            "E999",  # SyntaxError
            "F821",  # undefined name
            "F822",  # undefined name in __all__
            "F823",  # local variable referenced before assignment
        ]

        self.auto_fixable_issues = {
            "W292": self._fix_no_newline_at_end,
            "E303": self._fix_too_many_blank_lines,
            "E302": self._fix_expected_2_blank_lines,
            "E261": self._fix_inline_comment_spacing,
            "E262": self._fix_inline_comment_no_space,
            "W291": self._fix_trailing_whitespace,
            "E231": self._fix_missing_whitespace_after_comma,
            "E225": self._fix_missing_whitespace_around_operator,
        }

    def analyze_codebase(self) -> LintingReport:
        """Analyze the entire codebase for linting issues."""
        print("Starting comprehensive linting analysis...")

        report = LintingReport()

        # Get all Python files
        python_files = self._get_python_files()
        report.total_files_scanned = len(python_files)

        print(f"Scanning {len(python_files)} Python files...")

        # Run different linters
        flake8_issues = self._run_flake8(python_files)
        mypy_issues = self._run_mypy(python_files)
        bandit_issues = self._run_bandit(python_files)

        # Combine all issues
        all_issues = flake8_issues + mypy_issues + bandit_issues

        # Process issues
        for issue in all_issues:
            report.total_issues += 1
            report.issues_by_file[issue.file_path].append(issue)

            # Categorize by type
            if issue.error_code not in report.issues_by_type:
                report.issues_by_type[issue.error_code] = 0
            report.issues_by_type[issue.error_code] += 1

            # Check if auto-fixable
            if issue.error_code in self.auto_fixable_issues:
                issue.auto_fixable = True
                report.auto_fixable_count += 1

            # Check if critical
            if issue.error_code in self.critical_issues:
                report.critical_issues.append(issue)

        return report

    def _get_python_files(self) -> list[Path]:
        """Get all Python files in source directories."""
        python_files = []

        for source_dir in self.source_dirs:
            if not Path(source_dir).exists():
                continue

            for file_path in Path(source_dir).rglob("*.py"):
                # Skip excluded patterns
                if any(pattern in str(file_path) for pattern in self.exclude_patterns):
                    continue
                python_files.append(file_path)

        return sorted(python_files)

    def _run_flake8(self, files: list[Path]) -> list[LintingIssue]:
        """Run flake8 linter and parse results."""
        issues = []

        try:
            # Run flake8 with JSON output if available, otherwise parse text
            cmd = ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)

            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue

                try:
                    parts = line.split(":", 4)
                    if len(parts) >= 5:
                        file_path, line_num, col, code, message = parts

                        issue = LintingIssue(
                            file_path=file_path,
                            line_number=int(line_num),
                            column=int(col),
                            error_code=code,
                            error_message=message.strip(),
                            severity="error" if code.startswith("E") else "warning",
                        )
                        issues.append(issue)

                except (ValueError, IndexError) as e:
                    print(f"Failed to parse flake8 line: {line} - {e}")

        except subprocess.TimeoutExpired:
            print("Flake8 analysis timed out")
        except FileNotFoundError:
            print("Flake8 not found - skipping flake8 analysis")
        except Exception as e:
            print(f"Flake8 analysis failed: {e}")

        return issues

    def _run_mypy(self, files: list[Path]) -> list[LintingIssue]:
        """Run mypy type checker and parse results."""
        issues = []

        try:
            # Run mypy on each file individually to avoid overwhelming output
            for file_path in files[:10]:  # Limit to first 10 files to avoid timeout
                cmd = [
                    "mypy",
                    "--ignore-missing-imports",
                    "--no-error-summary",
                    str(file_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)

                for line in result.stdout.strip().split("\n"):
                    if not line or "Success:" in line:
                        continue

                    # Parse mypy output: file.py:line: level: message
                    match = re.match(r"^(.+?):(\d+):\s*(error|warning|note):\s*(.+)$", line)
                    if match:
                        file_name, line_num, level, message = match.groups()

                        issue = LintingIssue(
                            file_path=file_name,
                            line_number=int(line_num),
                            column=0,
                            error_code="MYPY",
                            error_message=message,
                            severity=level,
                        )
                        issues.append(issue)

        except subprocess.TimeoutExpired:
            print("MyPy analysis timed out")
        except FileNotFoundError:
            print("MyPy not found - skipping type checking")
        except Exception as e:
            print(f"MyPy analysis failed: {e}")

        return issues

    def _run_bandit(self, files: list[Path]) -> list[LintingIssue]:
        """Run bandit security checker and parse results."""
        issues = []

        try:
            cmd = ["bandit", "-f", "json", "-q"] + [str(f) for f in files]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)

            if result.stdout:
                data = json.loads(result.stdout)

                for finding in data.get("results", []):
                    issue = LintingIssue(
                        file_path=finding["filename"],
                        line_number=finding["line_number"],
                        column=0,
                        error_code=f"B{finding['test_id'][1:]}",  # Convert to Bandit format
                        error_message=finding["issue_text"],
                        severity=finding["issue_severity"].lower(),
                    )
                    issues.append(issue)

        except subprocess.TimeoutExpired:
            print("Bandit analysis timed out")
        except FileNotFoundError:
            print("Bandit not found - skipping security analysis")
        except json.JSONDecodeError:
            print("Failed to parse Bandit JSON output")
        except Exception as e:
            print(f"Bandit analysis failed: {e}")

        return issues

    def fix_auto_fixable_issues(self, report: LintingReport) -> dict[str, int]:
        """Automatically fix issues that can be safely fixed."""
        fixes_applied = defaultdict(int)

        print(f"Attempting to fix {report.auto_fixable_count} auto-fixable issues...")

        for file_path, issues in report.issues_by_file.items():
            file_modified = False

            try:
                with open(file_path, encoding="utf-8") as f:
                    original_content = f.read()
                    lines = original_content.splitlines()

                modified_lines = lines.copy()

                # Sort issues by line number (descending) to avoid line number shifts
                auto_fixable = [issue for issue in issues if issue.auto_fixable]
                auto_fixable.sort(key=lambda x: x.line_number, reverse=True)

                for issue in auto_fixable:
                    if issue.error_code in self.auto_fixable_issues:
                        fixer = self.auto_fixable_issues[issue.error_code]

                        try:
                            modified_lines = fixer(modified_lines, issue)
                            fixes_applied[issue.error_code] += 1
                            file_modified = True

                        except Exception as e:
                            print(f"Failed to fix {issue.error_code} in {file_path}:{issue.line_number} - {e}")

                # Write back if modified
                if file_modified:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write("\n".join(modified_lines) + "\n")

                    print(f"Applied fixes to {file_path}")

            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return dict(fixes_applied)

    def _fix_no_newline_at_end(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Fix missing newline at end of file."""
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    def _fix_too_many_blank_lines(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Fix too many blank lines."""
        line_idx = issue.line_number - 1

        # Remove excessive blank lines
        while (
            line_idx > 0
            and line_idx < len(lines)
            and lines[line_idx].strip() == ""
            and lines[line_idx - 1].strip() == ""
        ):
            lines.pop(line_idx)

        return lines

    def _fix_expected_2_blank_lines(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Add expected blank lines before class/function definitions."""
        line_idx = issue.line_number - 1

        if line_idx > 0 and line_idx < len(lines):
            current_line = lines[line_idx].strip()

            # Check if this is a class or function definition
            if current_line.startswith(("class ", "def ", "async def ")):
                # Count existing blank lines above
                blank_count = 0
                check_idx = line_idx - 1

                while check_idx >= 0 and lines[check_idx].strip() == "":
                    blank_count += 1
                    check_idx -= 1

                # Add blank lines if needed
                if blank_count < 2:
                    lines.insert(line_idx, "")

        return lines

    def _fix_inline_comment_spacing(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Fix inline comment spacing."""
        line_idx = issue.line_number - 1

        if line_idx < len(lines):
            line = lines[line_idx]
            # Fix spacing around inline comments
            lines[line_idx] = re.sub(r"(\S)#(\S)", r"\1  # \2", line)

        return lines

    def _fix_inline_comment_no_space(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Fix missing space after # in comments."""
        line_idx = issue.line_number - 1

        if line_idx < len(lines):
            line = lines[line_idx]
            lines[line_idx] = re.sub(r"#(\S)", r"# \1", line)

        return lines

    def _fix_trailing_whitespace(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Remove trailing whitespace."""
        line_idx = issue.line_number - 1

        if line_idx < len(lines):
            lines[line_idx] = lines[line_idx].rstrip()

        return lines

    def _fix_missing_whitespace_after_comma(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Add whitespace after commas."""
        line_idx = issue.line_number - 1

        if line_idx < len(lines):
            line = lines[line_idx]
            lines[line_idx] = re.sub(r",(\S)", r", \1", line)

        return lines

    def _fix_missing_whitespace_around_operator(self, lines: list[str], issue: LintingIssue) -> list[str]:
        """Add whitespace around operators."""
        line_idx = issue.line_number - 1

        if line_idx < len(lines):
            line = lines[line_idx]
            # Fix common operators
            line = re.sub(r"(\w)=(\w)", r"\1 = \2", line)
            line = re.sub(r"(\w)\+(\w)", r"\1 + \2", line)
            line = re.sub(r"(\w)-(\w)", r"\1 - \2", line)
            line = re.sub(r"(\w)\*(\w)", r"\1 * \2", line)
            line = re.sub(r"(\w)/(\w)", r"\1 / \2", line)
            lines[line_idx] = line

        return lines

    def generate_report(self, report: LintingReport, fixes_applied: dict[str, int] = None) -> str:
        """Generate a comprehensive linting report."""
        report_lines = [
            "# Linting Analysis Report",
            f"**Generated**: {Path.cwd()!s} at {subprocess.check_output(['date'], text=True).strip()}",
            "",
            "## Summary",
            f"- **Files Scanned**: {report.total_files_scanned}",
            f"- **Total Issues**: {report.total_issues}",
            f"- **Auto-fixable Issues**: {report.auto_fixable_count}",
            f"- **Critical Issues**: {len(report.critical_issues)}",
            "",
        ]

        if fixes_applied:
            report_lines.extend(["## Fixes Applied", ""])
            for error_code, count in sorted(fixes_applied.items()):
                report_lines.append(f"- **{error_code}**: {count} fixes applied")
            report_lines.append("")

        # Issues by type
        if report.issues_by_type:
            report_lines.extend(["## Issues by Type", ""])

            for error_code, count in sorted(report.issues_by_type.items(), key=lambda x: -x[1]):
                severity = "[CRITICAL]" if error_code in self.critical_issues else "[WARNING]"
                fixable = "[AUTO-FIXABLE]" if error_code in self.auto_fixable_issues else ""
                report_lines.append(f"- **{error_code}**: {count} occurrences {severity} {fixable}")

            report_lines.append("")

        # Critical issues
        if report.critical_issues:
            report_lines.extend(["## Critical Issues (Require Manual Fix)", ""])

            for issue in report.critical_issues[:10]:  # Show first 10
                report_lines.append(
                    f"- **{issue.file_path}:{issue.line_number}** - {issue.error_code}: {issue.error_message}"
                )

            if len(report.critical_issues) > 10:
                report_lines.append(f"... and {len(report.critical_issues) - 10} more")

            report_lines.append("")

        # Top problematic files
        if report.issues_by_file:
            report_lines.extend(["## Most Problematic Files", ""])

            file_issues = [(file_path, len(issues)) for file_path, issues in report.issues_by_file.items()]
            file_issues.sort(key=lambda x: -x[1])

            for file_path, issue_count in file_issues[:10]:
                report_lines.append(f"- **{file_path}**: {issue_count} issues")

            report_lines.append("")

        # Recommendations
        report_lines.extend(
            [
                "## Recommendations",
                "",
                "### Immediate Actions",
            ]
        )

        if report.critical_issues:
            report_lines.append(
                f"1. **Fix critical issues**: {len(report.critical_issues)} critical issues need immediate attention"
            )

        if report.auto_fixable_count > 0:
            report_lines.append(f"2. **Run auto-fixes**: {report.auto_fixable_count} issues can be automatically fixed")

        if report.total_issues > 50:
            report_lines.append("3. **Consider incremental fixes**: Large number of issues - fix incrementally")

        report_lines.extend(
            [
                "",
                "### Long-term Improvements",
                "1. **Add pre-commit hooks**: Prevent new linting issues",
                "2. **Configure IDE linting**: Real-time issue detection",
                "3. **Regular linting**: Run weekly linting analysis",
                "4. **Team standards**: Establish coding standards and enforcement",
            ]
        )

        return "\n".join(report_lines)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and fix linting issues")
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        default=["src", "tests", "scripts"],
        help="Source directories to analyze",
    )
    parser.add_argument("--fix", action="store_true", help="Apply automatic fixes")
    parser.add_argument(
        "--report-file",
        default="linting_analysis_report.md",
        help="Output file for the report",
    )
    parser.add_argument("--export-json", help="Export detailed results as JSON")

    args = parser.parse_args()

    # Create analyzer
    analyzer = LintingAnalyzer(source_dirs=args.source_dirs)

    # Run analysis
    print("Running comprehensive linting analysis...")
    report = analyzer.analyze_codebase()

    fixes_applied = {}
    if args.fix:
        print("Applying automatic fixes...")
        fixes_applied = analyzer.fix_auto_fixable_issues(report)

        total_fixes = sum(fixes_applied.values())
        if total_fixes > 0:
            print(f"Applied {total_fixes} automatic fixes")

            # Re-run analysis to see remaining issues
            print("Re-analyzing after fixes...")
            report = analyzer.analyze_codebase()
        else:
            print("No automatic fixes were applied")

    # Generate report
    report_content = analyzer.generate_report(report, fixes_applied)

    # Save report
    with open(args.report_file, "w") as f:
        f.write(report_content)

    print(f"Report saved to {args.report_file}")

    # Export JSON if requested
    if args.export_json:
        json_data = {
            "summary": {
                "total_files": report.total_files_scanned,
                "total_issues": report.total_issues,
                "auto_fixable": report.auto_fixable_count,
                "critical_issues": len(report.critical_issues),
            },
            "issues_by_type": report.issues_by_type,
            "critical_issues": [
                {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "code": issue.error_code,
                    "message": issue.error_message,
                }
                for issue in report.critical_issues
            ],
            "fixes_applied": fixes_applied,
        }

        with open(args.export_json, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"JSON data exported to {args.export_json}")

    # Print summary
    print("\nAnalysis Complete:")
    print(f"   Files: {report.total_files_scanned}")
    print(f"   Issues: {report.total_issues}")
    print(f"   Critical: {len(report.critical_issues)}")
    print(f"   Auto-fixable: {report.auto_fixable_count}")

    if fixes_applied:
        print(f"   Fixed: {sum(fixes_applied.values())}")

    # Exit with error code if critical issues exist
    if report.critical_issues:
        print(f"\n{len(report.critical_issues)} critical issues require manual attention")
        sys.exit(1)
    else:
        print("\nNo critical issues found")
        sys.exit(0)


if __name__ == "__main__":
    main()
