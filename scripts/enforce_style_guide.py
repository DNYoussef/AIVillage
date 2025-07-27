#!/usr/bin/env python3
"""Automated Style Guide Enforcement System
Enforces consistent style across automation scripts with comprehensive checks.
"""

import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StyleGuideEnforcer:
    """Automated style guide enforcement for automation scripts"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.scripts_dir = self.project_root / "scripts"

        # Style guide rules
        self.style_rules = {
            "header_requirements": {
                "shebang": "#!/usr/bin/env python3",
                "docstring": True,
                "imports_order": ["standard", "third_party", "local"],
            },
            "naming_conventions": {
                "functions": "snake_case",
                "classes": "PascalCase",
                "constants": "UPPER_CASE",
                "variables": "snake_case",
            },
            "documentation": {
                "function_docstrings": True,
                "class_docstrings": True,
                "type_hints": True,
            },
            "performance": {
                "max_function_length": 50,
                "max_complexity": 10,
                "max_line_length": 88,
            },
        }

        # Tools configuration
        self.tools = {
            "ruff": {
                "command": ["python", "-m", "ruff"],
                "fix_flag": "--fix",
                "config_file": "pyproject.toml",
            },
            "black": {
                "command": ["python", "-m", "black"],
                "config_file": "pyproject.toml",
            },
            "mypy": {
                "command": ["python", "-m", "mypy"],
                "config_file": "pyproject.toml",
            },
            "bandit": {
                "command": ["python", "-m", "bandit"],
                "args": ["-r", "--exclude", "tests"],
            },
        }

    def check_script_structure(self, script_path: Path) -> dict:
        """Check if script follows required structure"""
        issues = []

        try:
            with open(script_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
        except Exception as e:
            return {"status": "error", "issues": [f"Cannot read file: {e}"]}

        # Check shebang
        if not lines or not lines[0].startswith("#!/usr/bin/env python3"):
            issues.append("Missing or incorrect shebang line")

        # Check for module docstring
        docstring_found = False
        in_docstring = False
        quote_type = None

        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            if not docstring_found:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    docstring_found = True
                    quote_type = stripped[:3]
                    if len(stripped) > 3 and stripped.endswith(quote_type):
                        break  # Single line docstring
                    in_docstring = True
                elif not stripped.startswith("#!/") and not stripped.startswith("#"):
                    break  # Hit non-comment, non-shebang code
            elif in_docstring and stripped.endswith(quote_type):
                break

        if not docstring_found:
            issues.append("Missing module docstring")

        # Check imports organization
        import_sections = {"standard": [], "third_party": [], "local": []}
        current_section = None

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                module = self._extract_module_name(stripped)
                section = self._classify_import(module)
                import_sections[section].append(line)

        # Check function and class definitions for docstrings
        function_issues = self._check_function_docstrings(content)
        issues.extend(function_issues)

        return {
            "status": "checked",
            "issues": issues,
            "import_sections": import_sections,
        }

    def _extract_module_name(self, import_line: str) -> str:
        """Extract module name from import statement"""
        if import_line.startswith("from "):
            return import_line.split()[1].split(".")[0]
        return import_line.split()[1].split(".")[0]

    def _classify_import(self, module: str) -> str:
        """Classify import as standard, third-party, or local"""
        standard_libs = {
            "os",
            "sys",
            "json",
            "time",
            "datetime",
            "pathlib",
            "typing",
            "argparse",
            "logging",
            "subprocess",
            "collections",
            "re",
            "functools",
            "itertools",
            "math",
            "random",
        }

        local_modules = {
            "agent_forge",
            "agents",
            "communications",
            "core",
            "utils",
            "ingestion",
        }

        if module in standard_libs:
            return "standard"
        if module in local_modules:
            return "local"
        return "third_party"

    def _check_function_docstrings(self, content: str) -> list[str]:
        """Check for missing function/class docstrings"""
        issues = []
        lines = content.splitlines()

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class "):
                # Look for docstring in next few lines
                docstring_found = False
                for j in range(i + 1, min(i + 5, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    if next_line.startswith('"""') or next_line.startswith("'''"):
                        docstring_found = True
                        break
                    if next_line and not next_line.startswith("#"):
                        break

                if not docstring_found:
                    name = stripped.split("(")[0].split()[1]
                    prefix = "Function" if stripped.startswith("def") else "Class"
                    issues.append(
                        f"{prefix} '{name}' missing docstring at line {i + 1}"
                    )

        return issues

    def run_tool_checks(self, target_path: str, fix: bool = False) -> dict:
        """Run style checking tools"""
        results = {}

        for tool_name, config in self.tools.items():
            logger.info(f"Running {tool_name}...")

            cmd = config["command"][:]

            # Add fix flag if supported and requested
            if fix and "fix_flag" in config:
                cmd.append(config["fix_flag"])

            # Add tool-specific arguments
            if "args" in config:
                cmd.extend(config["args"])

            # Add target path
            cmd.append(target_path)

            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                )

                results[tool_name] = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "passed": result.returncode == 0,
                }

            except Exception as e:
                results[tool_name] = {
                    "returncode": -1,
                    "error": str(e),
                    "passed": False,
                }

        return results

    def check_script(self, script_path: Path, fix: bool = False) -> dict:
        """Perform comprehensive style check on a script"""
        logger.info(f"Checking {script_path.name}...")

        # Structure check
        structure_result = self.check_script_structure(script_path)

        # Tool checks
        tool_results = self.run_tool_checks(str(script_path), fix=fix)

        # Combine results
        all_passed = len(structure_result["issues"]) == 0 and all(
            result["passed"] for result in tool_results.values()
        )

        return {
            "script": script_path.name,
            "passed": all_passed,
            "structure": structure_result,
            "tools": tool_results,
        }

    def check_all_scripts(self, fix: bool = False) -> dict:
        """Check all Python scripts in the scripts directory"""
        logger.info("Checking all automation scripts...")

        python_scripts = list(self.scripts_dir.glob("*.py"))
        results = {}
        summary = {"total": len(python_scripts), "passed": 0, "failed": 0}

        for script in python_scripts:
            if script.name.startswith("__"):
                continue  # Skip __init__.py files

            result = self.check_script(script, fix=fix)
            results[script.name] = result

            if result["passed"]:
                summary["passed"] += 1
            else:
                summary["failed"] += 1

        return {
            "summary": summary,
            "results": results,
            "timestamp": str(Path().cwd()),
        }

    def generate_report(self, results: dict, output_file: str = None) -> str:
        """Generate a comprehensive style guide report"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("AUTOMATION SCRIPTS STYLE GUIDE REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        summary = results["summary"]
        report_lines.append(f"Total Scripts: {summary['total']}")
        report_lines.append(f"Passed: {summary['passed']}")
        report_lines.append(f"Failed: {summary['failed']}")
        report_lines.append(
            f"Success Rate: {summary['passed'] / summary['total'] * 100:.1f}%"
        )
        report_lines.append("")

        # Detailed results
        for script_name, result in results["results"].items():
            report_lines.append("-" * 40)
            report_lines.append(f"Script: {script_name}")
            report_lines.append(f"Status: {'PASS' if result['passed'] else 'FAIL'}")
            report_lines.append("")

            # Structure issues
            if result["structure"]["issues"]:
                report_lines.append("Structure Issues:")
                for issue in result["structure"]["issues"]:
                    report_lines.append(f"  - {issue}")
                report_lines.append("")

            # Tool results
            for tool_name, tool_result in result["tools"].items():
                status = "PASS" if tool_result["passed"] else "FAIL"
                report_lines.append(f"{tool_name.upper()}: {status}")

                if not tool_result["passed"] and "stdout" in tool_result:
                    if tool_result["stdout"].strip():
                        report_lines.append(
                            f"  Output: {tool_result['stdout'][:200]}..."
                        )
                report_lines.append("")

        report_content = "\n".join(report_lines)

        if output_file:
            with open(output_file, "w") as f:
                f.write(report_content)
            logger.info(f"Report saved to {output_file}")

        return report_content


def main():
    parser = argparse.ArgumentParser(description="Automated style guide enforcement")
    parser.add_argument("--script", help="Check specific script")
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues where possible"
    )
    parser.add_argument("--report", help="Output report file")
    parser.add_argument("--project-root", help="Project root directory")

    args = parser.parse_args()

    enforcer = StyleGuideEnforcer(args.project_root)

    if args.script:
        script_path = enforcer.scripts_dir / args.script
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            sys.exit(1)

        result = enforcer.check_script(script_path, fix=args.fix)
        print(json.dumps(result, indent=2))

        if not result["passed"]:
            sys.exit(1)
    else:
        results = enforcer.check_all_scripts(fix=args.fix)
        report = enforcer.generate_report(results, args.report)

        if not args.report:
            print(report)

        if results["summary"]["failed"] > 0:
            sys.exit(1)


if __name__ == "__main__":
    main()
