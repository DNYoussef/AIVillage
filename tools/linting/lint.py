#!/usr/bin/env python3
"""AIVillage Unified Linting System
Comprehensive code quality and linting orchestrator
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


class LintingOrchestrator:
    """Unified linting system for AIVillage project."""

    def __init__(self, config_path: str | None = None):
        self.project_root = Path.cwd()
        self.results = {}
        self.start_time = time.time()
        self.config = self.load_config(config_path)
        self.available_tools = self.detect_available_tools()

    def load_config(self, config_path: str | None = None) -> dict[str, Any]:
        """Load linting configuration."""
        default_config = {
            "python": {
                "line_length": 88,
                "target_python": "py310",
                "exclude_dirs": [
                    ".git",
                    "__pycache__",
                    "build",
                    "dist",
                    "venv",
                    "env",
                    ".mypy_cache",
                    "archived",
                ],
                "tools": {
                    "ruff": {"enabled": True, "fix": False, "config": "pyproject.toml"},
                    "black": {"enabled": True, "check": True},
                    "isort": {"enabled": True, "check": True},
                    "mypy": {"enabled": True, "non_blocking": True},
                    "flake8": {"enabled": True, "config": ".flake8"},
                },
            },
            "performance": {"max_workers": 4, "timeout_per_tool": 300},
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_config = json.load(f)
                self.merge_config(default_config, custom_config)
            except Exception as e:
                print(f"Warning: Failed to load config {config_path}: {e}")

        return default_config

    def merge_config(self, base: dict, custom: dict) -> None:
        """Recursively merge custom config into base config."""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_config(base[key], value)
            else:
                base[key] = value

    def detect_available_tools(self) -> dict[str, bool]:
        """Detect which linting tools are available."""
        tools = {}

        # Test Ruff (CLI)
        try:
            result = subprocess.run(
                ["ruff", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tools["ruff"] = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["ruff"] = False

        # Test Black (Python module)
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import black; print('ok')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tools["black"] = result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["black"] = False

        # Test isort
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import isort; print('ok')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tools["isort"] = result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["isort"] = False

        # Test mypy
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import mypy; print('ok')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tools["mypy"] = result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["mypy"] = False

        # Test flake8
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import flake8; print('ok')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            tools["flake8"] = result.returncode == 0 and "ok" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            tools["flake8"] = False

        return tools

    def find_python_files(
        self, path: str = ".", exclude_dirs: list[str] = None
    ) -> list[Path]:
        """Find all Python files, excluding specified directories."""
        if exclude_dirs is None:
            exclude_dirs = self.config["python"]["exclude_dirs"]

        python_files = []
        search_path = Path(path)

        if search_path.is_file() and search_path.suffix == ".py":
            return [search_path]

        for py_file in search_path.rglob("*.py"):
            # Check if file is in excluded directory
            if any(excluded in py_file.parts for excluded in exclude_dirs):
                continue
            python_files.append(py_file)

        return sorted(python_files)

    def run_ruff(self, files: list[Path] = None, fix: bool = False) -> dict[str, Any]:
        """Run Ruff linting."""
        if not self.available_tools.get("ruff", False):
            return {"status": "skipped", "reason": "ruff not available"}

        start_time = time.time()
        cmd = ["ruff", "check"]

        if fix:
            cmd.append("--fix")

        if files:
            cmd.extend([str(f) for f in files])
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config["performance"]["timeout_per_tool"],
            )

            # Count actual issues (each line is typically one issue)
            issues_count = (
                len(
                    [
                        line
                        for line in result.stdout.splitlines()
                        if line.strip() and not line.startswith("warning:")
                    ]
                )
                if result.stdout
                else 0
            )

            return {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - start_time,
                "files_processed": len(files) if files else "all",
                "issues_found": issues_count,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration": time.time() - start_time}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_black(
        self, files: list[Path] = None, check_only: bool = True
    ) -> dict[str, Any]:
        """Run Black formatter."""
        if not self.available_tools.get("black", False):
            return {"status": "skipped", "reason": "black not available"}

        start_time = time.time()
        cmd = [sys.executable, "-m", "black"]

        if check_only:
            cmd.append("--check")

        cmd.extend(["--line-length", str(self.config["python"]["line_length"])])

        if files:
            cmd.extend([str(f) for f in files])
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config["performance"]["timeout_per_tool"],
            )

            return {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - start_time,
                "files_processed": len(files) if files else "all",
                "issues_found": len(
                    [
                        line
                        for line in result.stdout.splitlines()
                        if "would reformat" in line or "error:" in line.lower()
                    ]
                ),
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration": time.time() - start_time}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_isort(
        self, files: list[Path] = None, check_only: bool = True
    ) -> dict[str, Any]:
        """Run isort import sorting."""
        if not self.available_tools.get("isort", False):
            return {"status": "skipped", "reason": "isort not available"}

        start_time = time.time()
        cmd = [sys.executable, "-m", "isort"]

        if check_only:
            cmd.append("--check-only")

        if files:
            cmd.extend([str(f) for f in files])
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config["performance"]["timeout_per_tool"],
            )

            return {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - start_time,
                "files_processed": len(files) if files else "all",
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration": time.time() - start_time}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_mypy(self, files: list[Path] = None) -> dict[str, Any]:
        """Run mypy type checking."""
        if not self.available_tools.get("mypy", False):
            return {"status": "skipped", "reason": "mypy not available"}

        start_time = time.time()
        cmd = [sys.executable, "-m", "mypy"]

        if files:
            cmd.extend([str(f) for f in files])
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config["performance"]["timeout_per_tool"],
            )

            return {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - start_time,
                "files_processed": len(files) if files else "all",
                "non_blocking": self.config["python"]["tools"]["mypy"]["non_blocking"],
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration": time.time() - start_time}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def run_flake8(self, files: list[Path] = None) -> dict[str, Any]:
        """Run flake8 linting."""
        if not self.available_tools.get("flake8", False):
            return {"status": "skipped", "reason": "flake8 not available"}

        start_time = time.time()
        cmd = [sys.executable, "-m", "flake8"]

        if files:
            cmd.extend([str(f) for f in files])
        else:
            cmd.append(".")

        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.config["performance"]["timeout_per_tool"],
            )

            # Count actual issues for flake8 (each line is an issue)
            issues_count = (
                len([line for line in result.stdout.splitlines() if line.strip()])
                if result.stdout
                else 0
            )

            return {
                "status": "completed",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": time.time() - start_time,
                "files_processed": len(files) if files else "all",
                "issues_found": issues_count,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "duration": time.time() - start_time}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def lint_python(
        self, path: str = ".", fix: bool = False, parallel: bool = True
    ) -> dict[str, Any]:
        """Run comprehensive Python linting."""
        print(f"[PYTHON] Running Python linting on: {path}")

        python_files = self.find_python_files(path)
        print(f"Found {len(python_files)} Python files")

        if not python_files:
            return {"status": "no_files", "message": "No Python files found"}

        linting_functions = []

        # Add enabled tools
        if self.config["python"]["tools"]["ruff"]["enabled"]:
            linting_functions.append(("ruff", lambda: self.run_ruff(fix=fix)))

        if self.config["python"]["tools"]["black"]["enabled"]:
            linting_functions.append(
                ("black", lambda: self.run_black(check_only=not fix))
            )

        if self.config["python"]["tools"]["isort"]["enabled"]:
            linting_functions.append(
                ("isort", lambda: self.run_isort(check_only=not fix))
            )

        if self.config["python"]["tools"]["mypy"]["enabled"]:
            linting_functions.append(("mypy", lambda: self.run_mypy()))

        if self.config["python"]["tools"]["flake8"]["enabled"]:
            linting_functions.append(("flake8", lambda: self.run_flake8()))

        results = {}

        if parallel and len(linting_functions) > 1:
            # Run tools in parallel
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.config["performance"]["max_workers"]
            ) as executor:
                futures = {
                    executor.submit(func): name for name, func in linting_functions
                }

                for future in concurrent.futures.as_completed(futures):
                    tool_name = futures[future]
                    try:
                        results[tool_name] = future.result()
                        print(f"[OK] {tool_name} completed")
                    except Exception as e:
                        results[tool_name] = {"status": "error", "error": str(e)}
                        print(f"[FAIL] {tool_name} failed: {e}")
        else:
            # Run tools sequentially
            for tool_name, func in linting_functions:
                print(f"[RUNNING] {tool_name}...")
                results[tool_name] = func()
                print(f"[OK] {tool_name} completed")

        return {
            "status": "completed",
            "files_found": len(python_files),
            "tools_run": len(results),
            "results": results,
            "summary": self.generate_summary(results),
        }

    def generate_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate summary of linting results."""
        total_issues = 0
        completed_tools = 0
        failed_tools = 0
        skipped_tools = 0

        for tool, result in results.items():
            if result["status"] == "completed":
                completed_tools += 1
                if "issues_found" in result:
                    total_issues += result["issues_found"]
            elif result["status"] == "skipped":
                skipped_tools += 1
            else:
                failed_tools += 1

        return {
            "total_issues": total_issues,
            "completed_tools": completed_tools,
            "failed_tools": failed_tools,
            "skipped_tools": skipped_tools,
            "success_rate": completed_tools / len(results) if results else 0,
        }

    def generate_report(
        self, results: dict[str, Any], output_format: str = "json"
    ) -> str:
        """Generate comprehensive linting report."""
        report_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "total_duration": time.time() - self.start_time,
                "available_tools": self.available_tools,
                "config": self.config,
            },
            "results": results,
        }

        if output_format == "json":
            return json.dumps(report_data, indent=2)
        if output_format == "summary":
            return self.generate_text_summary(report_data)
        return json.dumps(report_data, indent=2)

    def generate_text_summary(self, report_data: dict[str, Any]) -> str:
        """Generate human-readable text summary."""
        lines = []
        lines.append("=" * 70)
        lines.append("AIVillage Linting Report Summary")
        lines.append("=" * 70)
        lines.append(f"Generated: {report_data['metadata']['timestamp']}")
        lines.append(f"Duration: {report_data['metadata']['total_duration']:.2f}s")
        lines.append("")

        if "results" in report_data and "summary" in report_data["results"]:
            summary = report_data["results"]["summary"]
            lines.append("[SUMMARY]:")
            lines.append(f"  Total Issues: {summary['total_issues']}")
            lines.append(f"  Tools Completed: {summary['completed_tools']}")
            lines.append(f"  Tools Failed: {summary['failed_tools']}")
            lines.append(f"  Tools Skipped: {summary['skipped_tools']}")
            lines.append(f"  Success Rate: {summary['success_rate']:.1%}")
            lines.append("")

        lines.append("[TOOLS] Available Tools:")
        for tool, available in report_data["metadata"]["available_tools"].items():
            status = "[OK]" if available else "[FAIL]"
            lines.append(f"  {status} {tool}")

        return "\n".join(lines)


def main():
    """Main entry point for the unified linting system."""
    parser = argparse.ArgumentParser(description="AIVillage Unified Linting System")
    parser.add_argument(
        "path", nargs="?", default=".", help="Path to lint (default: current directory)"
    )
    parser.add_argument("--fix", action="store_true", help="Apply fixes where possible")
    parser.add_argument("--config", help="Path to custom configuration file")
    parser.add_argument(
        "--output",
        choices=["json", "summary", "both"],
        default="summary",
        help="Output format",
    )
    parser.add_argument("--output-file", help="Save report to file")
    parser.add_argument(
        "--parallel", action="store_true", default=True, help="Run tools in parallel"
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel execution"
    )

    args = parser.parse_args()

    if args.no_parallel:
        args.parallel = False

    # Create orchestrator
    orchestrator = LintingOrchestrator(args.config)

    # Print available tools
    print("[TOOLS] Available Tools:")
    for tool, available in orchestrator.available_tools.items():
        status = "[OK]" if available else "[FAIL]"
        print(f"  {status} {tool}")
    print()

    # Run linting
    results = orchestrator.lint_python(args.path, fix=args.fix, parallel=args.parallel)

    # Generate report
    if args.output in ["json", "both"]:
        json_report = orchestrator.generate_report(results, "json")
        if args.output_file:
            report_file = f"{args.output_file}.json"
            with open(report_file, "w") as f:
                f.write(json_report)
            print(f"[REPORT] JSON report saved to: {report_file}")
        elif args.output == "json":
            print(json_report)

    if args.output in ["summary", "both"]:
        summary_report = orchestrator.generate_report(results, "summary")
        if args.output_file and args.output == "both":
            report_file = f"{args.output_file}.txt"
            with open(report_file, "w") as f:
                f.write(summary_report)
            print(f"[REPORT] Summary report saved to: {report_file}")
        else:
            print(summary_report)

    # Exit with appropriate code
    if (
        results.get("summary", {}).get("failed_tools", 0) > 0
        or results.get("summary", {}).get("total_issues", 0) > 0
    ):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
