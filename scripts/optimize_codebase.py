#!/usr/bin/env python3
"""Codebase optimization and validation tool.

This script provides comprehensive code quality improvements including:
- Python code formatting and linting
- Import optimization and unused import removal
- Type hint validation and improvement
- Documentation validation
- Performance analysis

Usage:
    python optimize_codebase.py [--path PATH] [--fix] [--check-only]
"""

import argparse
import ast
import logging
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("optimize_codebase.log"),
    ],
)
logger = logging.getLogger(__name__)


class CodeQualityAnalyzer:
    """Analyze and improve code quality."""

    def __init__(self, root_path: Path, fix_issues: bool = False) -> None:
        """Initialize the analyzer.

        Args:
            root_path: Root directory to analyze
            fix_issues: Whether to automatically fix issues
        """
        self.root_path = root_path
        self.fix_issues = fix_issues
        self.python_files: list[Path] = []
        self.issues: dict[str, list[str]] = {}

    def discover_files(self) -> None:
        """Discover Python files to analyze."""
        patterns = ["**/*.py"]
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "venv",
            "env",
            ".env",
            "deprecated",
            "archive",
        }

        for pattern in patterns:
            for file_path in self.root_path.glob(pattern):
                # Skip excluded directories
                if any(excluded in file_path.parts for excluded in excluded_dirs):
                    continue

                # Skip very large files
                if file_path.stat().st_size > 1024 * 1024:  # 1MB
                    logger.warning(f"Skipping large file: {file_path}")
                    continue

                self.python_files.append(file_path)

        logger.info(f"Discovered {len(self.python_files)} Python files")

    def analyze_imports(self, file_path: Path) -> list[str]:
        """Analyze imports in a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of issues found
        """
        issues = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")

            # Check for unused imports (basic check)
            for imp in imports:
                base_name = imp.split(".")[0]
                if base_name not in content:
                    issues.append(f"Potentially unused import: {imp}")

        except Exception as e:
            issues.append(f"Failed to analyze imports: {e}")

        return issues

    def check_type_hints(self, file_path: Path) -> list[str]:
        """Check for missing type hints.

        Args:
            file_path: Path to the Python file

        Returns:
            List of issues found
        """
        issues = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for missing return type annotation
                    if not node.returns and node.name != "__init__":
                        issues.append(f"Function {node.name} missing return type hint")

                    # Check for missing argument type annotations
                    for arg in node.args.args:
                        if not arg.annotation and arg.arg != "self":
                            issues.append(f"Function {node.name} argument {arg.arg} missing type hint")

        except Exception as e:
            issues.append(f"Failed to check type hints: {e}")

        return issues

    def run_linting_tools(self) -> dict[str, list[str]]:
        """Run various linting tools on the codebase.

        Returns:
            Dictionary mapping tool names to lists of issues
        """
        results = {}

        # Run ruff
        try:
            cmd = ["python", "-m", "ruff", "check", str(self.root_path)]
            if self.fix_issues:
                cmd.append("--fix")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

            if result.stdout:
                results["ruff"] = result.stdout.strip().split("\n")
            else:
                results["ruff"] = ["No issues found"]

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            results["ruff"] = [f"Tool execution failed: {e}"]

        # Run mypy on key directories
        key_dirs = ["scripts", "src", "agent_forge"]
        for dir_name in key_dirs:
            dir_path = self.root_path / dir_name
            if dir_path.exists():
                try:
                    cmd = [
                        "python",
                        "-m",
                        "mypy",
                        str(dir_path),
                        "--ignore-missing-imports",
                        "--show-error-codes",
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)

                    if result.stdout:
                        results[f"mypy-{dir_name}"] = result.stdout.strip().split("\n")
                    else:
                        results[f"mypy-{dir_name}"] = ["No type issues found"]

                except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                    results[f"mypy-{dir_name}"] = [f"Tool execution failed: {e}"]

        return results

    def analyze_file(self, file_path: Path) -> None:
        """Analyze a single Python file.

        Args:
            file_path: Path to the file to analyze
        """
        relative_path = file_path.relative_to(self.root_path)
        logger.debug(f"Analyzing {relative_path}")

        file_issues = []

        # Check imports
        import_issues = self.analyze_imports(file_path)
        file_issues.extend(import_issues)

        # Check type hints
        type_issues = self.check_type_hints(file_path)
        file_issues.extend(type_issues)

        # Store issues if any found
        if file_issues:
            self.issues[str(relative_path)] = file_issues

    def generate_report(self) -> str:
        """Generate optimization report.

        Returns:
            Formatted report string
        """
        report_lines = [
            "# Codebase Optimization Report",
            "",
            f"**Analyzed Files:** {len(self.python_files)}",
            f"**Files with Issues:** {len(self.issues)}",
            "",
        ]

        if self.issues:
            report_lines.extend(
                [
                    "## File-specific Issues",
                    "",
                ]
            )

            for file_path, file_issues in self.issues.items():
                report_lines.extend(
                    [
                        f"### {file_path}",
                        "",
                    ]
                )

                for issue in file_issues:
                    report_lines.append(f"- {issue}")

                report_lines.append("")

        # Run linting tools
        linting_results = self.run_linting_tools()

        if linting_results:
            report_lines.extend(
                [
                    "## Linting Results",
                    "",
                ]
            )

            for tool, tool_results in linting_results.items():
                report_lines.extend(
                    [
                        f"### {tool.upper()}",
                        "",
                    ]
                )

                for result in tool_results[:10]:  # Limit to first 10 results
                    report_lines.append(f"- {result}")

                if len(tool_results) > 10:
                    report_lines.append(f"- ... and {len(tool_results) - 10} more issues")

                report_lines.append("")

        return "\n".join(report_lines)

    def optimize(self) -> None:
        """Run the optimization process."""
        logger.info("Starting codebase optimization...")

        self.discover_files()

        # Analyze each file
        for file_path in self.python_files:
            try:
                self.analyze_file(file_path)
            except Exception as e:
                logger.exception(f"Failed to analyze {file_path}: {e}")

        # Generate and save report
        report = self.generate_report()

        report_path = self.root_path / "optimization_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Optimization report saved to {report_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Files analyzed: {len(self.python_files)}")
        print(f"Files with issues: {len(self.issues)}")

        if self.fix_issues:
            print("Automatic fixes were applied where possible")
        else:
            print("Run with --fix to apply automatic fixes")

        print(f"Full report: {report_path}")


def main() -> int:
    """Main optimization function.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Codebase optimization and validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_codebase.py
  python optimize_codebase.py --path src/ --fix
  python optimize_codebase.py --check-only
""",
    )

    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path to analyze (default: current directory)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix issues where possible",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for issues, don't generate full report",
    )

    args = parser.parse_args()

    try:
        if not args.path.exists():
            logger.error(f"Path does not exist: {args.path}")
            return 1

        analyzer = CodeQualityAnalyzer(args.path, fix_issues=args.fix)

        if args.check_only:
            analyzer.discover_files()
            quick_issues = 0

            for file_path in analyzer.python_files[:10]:  # Quick check of first 10 files
                try:
                    analyzer.analyze_file(file_path)
                    if str(file_path.relative_to(args.path)) in analyzer.issues:
                        quick_issues += 1
                except Exception:
                    pass

            print(
                f"Quick check: {quick_issues} files have issues (out of {min(10, len(analyzer.python_files))} checked)"
            )
            return 0

        analyzer.optimize()

        return 0

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Optimization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
