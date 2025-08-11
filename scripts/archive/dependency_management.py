#!/usr/bin/env python3
"""Dependency management script for AIVillage.

This script provides tools for managing dependencies, checking compatibility,
and resolving version conflicts.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import sys


@dataclass
class DependencyInfo:
    """Information about a dependency."""

    name: str
    version: str
    required_by: list[str]
    conflicts: list[str]
    is_vulnerable: bool = False
    vulnerability_details: str | None = None


class DependencyManager:
    """Manages dependencies and resolves conflicts."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.requirements_file = self.project_root / "requirements.txt"
        self.pyproject_file = self.project_root / "pyproject.toml"

    def get_installed_packages(self) -> dict[str, str]:
        """Get currently installed packages and their versions."""
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True,
        )

        packages = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}

    def get_dependency_conflicts(self) -> list[str]:
        """Get dependency conflicts from pip check."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "check"],
                capture_output=True,
                text=True,
                check=False,  # Don't raise on non-zero exit
            )

            if result.returncode != 0:
                return result.stdout.split("\n") if result.stdout else []
            return []
        except Exception as e:
            print(f"Error checking dependencies: {e}")
            return []

    def parse_requirements(self) -> dict[str, str]:
        """Parse requirements.txt to get required dependencies."""
        requirements = {}

        if self.requirements_file.exists():
            with open(self.requirements_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Extract package name and version
                        match = re.match(r"^([a-zA-Z0-9\-_]+)([>=<~!]+.*)?$", line)
                        if match:
                            name = match.group(1).lower()
                            version = match.group(2) or ""
                            requirements[name] = version

        return requirements

    def check_numpy_compatibility(self) -> tuple[bool, str]:
        """Check NumPy compatibility with other packages."""
        installed = self.get_installed_packages()
        numpy_version = installed.get("numpy", "")

        if not numpy_version:
            return False, "NumPy is not installed"

        # Check if NumPy 2.x is installed
        major_version = int(numpy_version.split(".")[0])
        if major_version >= 2:
            return (
                False,
                f"NumPy {numpy_version} (2.x) may be incompatible with some packages. Consider downgrading to 1.x",
            )

        return True, f"NumPy {numpy_version} is compatible"

    def suggest_numpy_fix(self) -> list[str]:
        """Suggest fixes for NumPy compatibility issues."""
        return [
            "# Fix NumPy compatibility issues:",
            "pip install 'numpy>=1.24.3,<2.0.0'",
            "",
            "# If you have compiled packages that need NumPy 1.x:",
            "pip install --force-reinstall 'numpy>=1.24.3,<2.0.0'",
            "",
            "# Update requirements.txt to pin NumPy version:",
            "numpy>=1.24.3,<2.0.0",
        ]

    def get_vulnerable_packages(self) -> list[DependencyInfo]:
        """Get list of potentially vulnerable packages based on known issues."""
        # Known vulnerable versions (this is a basic check - in production use safety or similar)
        known_vulnerabilities = {
            "pillow": {
                "vulnerable_versions": ["<10.0.0"],
                "description": "Potential security vulnerabilities in older Pillow versions",
            },
            "requests": {
                "vulnerable_versions": ["<2.31.0"],
                "description": "Potential security vulnerabilities in older requests versions",
            },
            "urllib3": {
                "vulnerable_versions": ["<2.0.0"],
                "description": "Potential security vulnerabilities in older urllib3 versions",
            },
            "cryptography": {
                "vulnerable_versions": ["<41.0.0"],
                "description": "Potential security vulnerabilities in older cryptography versions",
            },
        }

        installed = self.get_installed_packages()
        vulnerable = []

        for package, details in known_vulnerabilities.items():
            if package in installed:
                version = installed[package]
                # Simple version check (in production use proper version parsing)
                if any(
                    self._is_version_vulnerable(version, vuln_pattern)
                    for vuln_pattern in details["vulnerable_versions"]
                ):
                    vulnerable.append(
                        DependencyInfo(
                            name=package,
                            version=version,
                            required_by=[],
                            conflicts=[],
                            is_vulnerable=True,
                            vulnerability_details=details["description"],
                        )
                    )

        return vulnerable

    def _is_version_vulnerable(self, version: str, pattern: str) -> bool:
        """Simple version comparison (basic implementation)."""
        # This is a simplified check - in production use packaging.version
        if pattern.startswith("<"):
            target_version = pattern[1:]
            return version < target_version
        return False

    def generate_security_report(self) -> str:
        """Generate a security report for dependencies."""
        report = []
        report.append("# AIVillage Dependency Security Report")
        report.append("=" * 50)
        report.append("")

        # Check NumPy compatibility
        numpy_ok, numpy_msg = self.check_numpy_compatibility()
        if numpy_ok:
            report.append(f"[OK] NumPy: {numpy_msg}")
        else:
            report.append(f"[WARNING] NumPy: {numpy_msg}")
            report.append("")
            report.extend(self.suggest_numpy_fix())
            report.append("")

        # Check for conflicts
        conflicts = self.get_dependency_conflicts()
        if conflicts:
            report.append("## Dependency Conflicts")
            report.append("-" * 25)
            for conflict in conflicts:
                if conflict.strip():
                    report.append(f"[WARNING] {conflict}")
            report.append("")
        else:
            report.append("[OK] No dependency conflicts detected")
            report.append("")

        # Check for vulnerable packages
        vulnerable = self.get_vulnerable_packages()
        if vulnerable:
            report.append("## Potentially Vulnerable Packages")
            report.append("-" * 35)
            for pkg in vulnerable:
                report.append(f"[WARNING] {pkg.name} {pkg.version}: {pkg.vulnerability_details}")
            report.append("")
        else:
            report.append("[OK] No known vulnerable packages detected")
            report.append("")

        # Recommendations
        report.append("## Recommendations")
        report.append("-" * 20)
        report.append("1. Regularly update dependencies to latest secure versions")
        report.append("2. Use version pinning for critical dependencies")
        report.append("3. Monitor security advisories for used packages")
        report.append("4. Consider using tools like safety or bandit for security scanning")
        report.append("5. Test thoroughly after dependency updates")

        return "\n".join(report)

    def create_dependency_lock_file(self) -> str:
        """Create a dependency lock file with exact versions."""
        installed = self.get_installed_packages()

        lock_content = []
        lock_content.append("# Dependency lock file - exact versions")
        lock_content.append("# Generated by AIVillage dependency manager")
        lock_content.append("")

        # Sort packages for consistency
        for package, version in sorted(installed.items()):
            lock_content.append(f"{package}=={version}")

        return "\n".join(lock_content)

    def suggest_dependency_fixes(self) -> list[str]:
        """Suggest fixes for common dependency issues."""
        fixes = []

        # Check for common issues
        conflicts = self.get_dependency_conflicts()
        if conflicts:
            fixes.append("# Fix dependency conflicts:")
            fixes.append("pip install --upgrade --force-reinstall <conflicting_package>")
            fixes.append("")

        # NumPy compatibility
        numpy_ok, _ = self.check_numpy_compatibility()
        if not numpy_ok:
            fixes.extend(self.suggest_numpy_fix())
            fixes.append("")

        # General recommendations
        fixes.extend(
            [
                "# General dependency management:",
                "pip install --upgrade pip setuptools wheel",
                "pip install --upgrade --force-reinstall pydantic fastapi",
                "",
                "# If you have persistent conflicts, try:",
                "pip-autoremove -y  # Remove unused dependencies",
                "pip install --no-deps <package>  # Install without dependencies",
            ]
        )

        return fixes


def main() -> None:
    """Main function for dependency management."""
    import argparse

    parser = argparse.ArgumentParser(description="AIVillage Dependency Manager")
    parser.add_argument("--report", action="store_true", help="Generate security report")
    parser.add_argument("--lock", action="store_true", help="Create dependency lock file")
    parser.add_argument("--fix", action="store_true", help="Show suggested fixes")
    parser.add_argument("--check-numpy", action="store_true", help="Check NumPy compatibility")

    args = parser.parse_args()

    manager = DependencyManager()

    if args.report:
        print(manager.generate_security_report())

    elif args.lock:
        lock_content = manager.create_dependency_lock_file()
        lock_file = manager.project_root / "requirements-lock.txt"
        with open(lock_file, "w") as f:
            f.write(lock_content)
        print(f"Dependency lock file created: {lock_file}")

    elif args.fix:
        fixes = manager.suggest_dependency_fixes()
        print("\n".join(fixes))

    elif args.check_numpy:
        numpy_ok, msg = manager.check_numpy_compatibility()
        print(f"NumPy status: {msg}")
        if not numpy_ok:
            print("\nSuggested fixes:")
            print("\n".join(manager.suggest_numpy_fix()))

    else:
        # Default: show brief status
        print("AIVillage Dependency Status:")
        print("-" * 30)

        numpy_ok, numpy_msg = manager.check_numpy_compatibility()
        status_symbol = "[OK]" if numpy_ok else "[WARNING]"
        print(f"{status_symbol} {numpy_msg}")

        conflicts = manager.get_dependency_conflicts()
        if conflicts:
            print(f"[WARNING] {len(conflicts)} dependency conflicts detected")
        else:
            print("[OK] No dependency conflicts")

        vulnerable = manager.get_vulnerable_packages()
        if vulnerable:
            print(f"[WARNING] {len(vulnerable)} potentially vulnerable packages")
        else:
            print("[OK] No known vulnerable packages")

        print("\nUse --report for detailed analysis")


if __name__ == "__main__":
    main()
