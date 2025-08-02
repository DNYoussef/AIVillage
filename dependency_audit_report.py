#!/usr/bin/env python3
"""AIVillage Dependency Audit Report Generator
Comprehensive analysis of all requirements files and their dependencies.
"""

from collections import defaultdict
import json
import os
import re


class DependencyAuditor:
    def __init__(self):
        self.requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "requirements-test.txt",
            "agent_forge/requirements.txt",
            "agent_forge/requirements_evomerge.txt",
            "communications/requirements.txt",
            "mcp_servers/hyperag/requirements.txt",
            "production/evolution/evomerge/requirements.txt",
            "experimental/services/services/wave_bridge/requirements.txt",
            "experimental/services/services/wave_bridge/requirements_enhanced.txt",
            "experimental/services/services/gateway/requirements.txt",
            "experimental/services/services/twin/requirements.txt",
            "tests/requirements.txt"
        ]

        # Known security-sensitive packages
        self.security_packages = {
            "pillow", "requests", "urllib3", "cryptography", "jinja2",
            "pyyaml", "setuptools", "wheel", "flask", "django", "sqlalchemy",
            "tornado", "werkzeug", "paramiko", "pycrypto", "pycryptodome",
            "lxml", "bleach", "markdown", "twisted", "pyopenssl"
        }

        # Heavy ML/AI packages that impact performance
        self.heavy_packages = {
            "torch", "tensorflow", "transformers", "accelerate", "bitsandbytes",
            "triton", "xformers", "torchvision", "torchaudio", "datasets",
            "sentence-transformers", "faiss-cpu", "faiss-gpu", "opencv-python"
        }

        # Production-critical packages
        self.critical_packages = {
            "fastapi", "uvicorn", "pydantic", "sqlalchemy", "redis", "celery",
            "gunicorn", "nginx", "postgres", "docker", "kubernetes"
        }

    def parse_requirements_file(self, filepath: str) -> dict[str, str]:
        """Parse a requirements file and extract package names and versions."""
        packages = {}

        if not os.path.exists(filepath):
            return packages

        with open(filepath, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Skip git repositories for now
                if line.startswith("git+"):
                    continue

                # Parse different version specifications
                if "==" in line:
                    pkg, version = line.split("==", 1)
                    packages[pkg.strip()] = version.strip()
                elif ">=" in line:
                    match = re.match(r"([^>=<!\s]+)>=([^,<>=!\s]+)", line)
                    if match:
                        pkg, version = match.groups()
                        packages[pkg.strip()] = f">={version.strip()}"
                elif "~=" in line:
                    match = re.match(r"([^~=<!\s]+)~=([^,<>=!\s]+)", line)
                    if match:
                        pkg, version = match.groups()
                        packages[pkg.strip()] = f"~={version.strip()}"
                elif "<" in line:
                    match = re.match(r"([^<>=!\s]+)<([^,<>=!\s]+)", line)
                    if match:
                        pkg, version = match.groups()
                        packages[pkg.strip()] = f"<{version.strip()}"
                else:
                    # Package name only
                    pkg = re.split(r"[<>=!]", line)[0].strip()
                    if pkg:
                        packages[pkg] = "latest"

        return packages

    def analyze_all_requirements(self) -> dict:
        """Analyze all requirements files and return comprehensive data."""
        all_packages = defaultdict(list)
        file_stats = {}
        version_conflicts = defaultdict(set)

        for filepath in self.requirements_files:
            packages = self.parse_requirements_file(filepath)
            file_stats[filepath] = {
                "exists": os.path.exists(filepath),
                "package_count": len(packages),
                "packages": packages
            }

            for pkg, version in packages.items():
                all_packages[pkg].append((filepath, version))
                if version != "latest":
                    # Extract just the version number for conflict detection
                    clean_version = re.sub(r"[><=~!]+", "", version)
                    version_conflicts[pkg].add(clean_version)

        return {
            "all_packages": dict(all_packages),
            "file_stats": file_stats,
            "version_conflicts": {k: v for k, v in version_conflicts.items() if len(v) > 1}
        }

    def analyze_security_issues(self, all_packages: dict) -> dict:
        """Analyze potential security issues."""
        security_issues = {
            "vulnerable_packages": [],
            "outdated_packages": [],
            "security_sensitive": []
        }

        for pkg in all_packages:
            if pkg.lower() in self.security_packages:
                security_issues["security_sensitive"].append(pkg)

        return security_issues

    def analyze_performance_impact(self, all_packages: dict) -> dict:
        """Analyze performance impact of heavy dependencies."""
        performance_analysis = {
            "heavy_packages": [],
            "size_estimate": {},
            "cuda_dependencies": [],
            "compilation_required": []
        }

        for pkg in all_packages:
            if pkg.lower() in self.heavy_packages:
                performance_analysis["heavy_packages"].append(pkg)

            # Check for CUDA-related packages
            if any(cuda_term in pkg.lower() for cuda_term in ["cuda", "gpu", "triton"]):
                performance_analysis["cuda_dependencies"].append(pkg)

            # Packages that often require compilation
            if any(compile_term in pkg.lower() for compile_term in
                   ["torch", "numpy", "scipy", "pandas", "pillow", "lxml"]):
                performance_analysis["compilation_required"].append(pkg)

        return performance_analysis

    def check_python_compatibility(self, all_packages: dict) -> dict:
        """Check Python version compatibility."""
        compatibility = {
            "python_310_compatible": [],
            "python_311_compatible": [],
            "python_312_compatible": [],
            "potential_issues": []
        }

        # This would require checking PyPI metadata, simplified for now
        # Known packages with Python version restrictions
        version_sensitive = {
            "tensorflow": "Limited Python 3.12 support",
            "torch": "Good Python 3.10+ support",
            "transformers": "Good Python 3.10+ support",
            "bitsandbytes": "Limited Windows/newer Python support"
        }

        for pkg in all_packages:
            if pkg.lower() in version_sensitive:
                compatibility["potential_issues"].append({
                    "package": pkg,
                    "note": version_sensitive[pkg.lower()]
                })

        return compatibility

    def generate_recommendations(self, analysis_data: dict) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Version conflicts
        if analysis_data["version_conflicts"]:
            recommendations.append(
                f"CRITICAL: Resolve {len(analysis_data['version_conflicts'])} version conflicts"
            )

        # Security recommendations
        security_count = len(analysis_data["security_analysis"]["security_sensitive"])
        if security_count > 0:
            recommendations.append(
                f"SECURITY: Monitor {security_count} security-sensitive packages"
            )

        # Performance recommendations
        heavy_count = len(analysis_data["performance_analysis"]["heavy_packages"])
        if heavy_count > 10:
            recommendations.append(
                f"PERFORMANCE: Consider optimizing {heavy_count} heavy dependencies"
            )

        # File consolidation
        active_files = sum(1 for stats in analysis_data["file_stats"].values()
                          if stats["exists"] and stats["package_count"] > 0)
        if active_files > 8:
            recommendations.append(
                f"MAINTENANCE: Consolidate {active_files} requirements files"
            )

        return recommendations

    def generate_report(self) -> dict:
        """Generate comprehensive dependency audit report."""
        print("Analyzing dependencies...")

        # Main analysis
        analysis_data = self.analyze_all_requirements()

        # Additional analyses
        analysis_data["security_analysis"] = self.analyze_security_issues(
            analysis_data["all_packages"]
        )
        analysis_data["performance_analysis"] = self.analyze_performance_impact(
            analysis_data["all_packages"]
        )
        analysis_data["compatibility_analysis"] = self.check_python_compatibility(
            analysis_data["all_packages"]
        )

        # Generate recommendations
        analysis_data["recommendations"] = self.generate_recommendations(analysis_data)

        # Summary statistics
        analysis_data["summary"] = {
            "total_unique_packages": len(analysis_data["all_packages"]),
            "total_requirements_files": len([f for f, stats in analysis_data["file_stats"].items()
                                           if stats["exists"]]),
            "active_requirements_files": len([f for f, stats in analysis_data["file_stats"].items()
                                            if stats["exists"] and stats["package_count"] > 0]),
            "version_conflicts_count": len(analysis_data["version_conflicts"]),
            "security_sensitive_count": len(analysis_data["security_analysis"]["security_sensitive"]),
            "heavy_packages_count": len(analysis_data["performance_analysis"]["heavy_packages"])
        }

        return analysis_data

def main():
    """Main execution function."""
    auditor = DependencyAuditor()
    report = auditor.generate_report()

    # Save report to JSON
    with open("dependency_audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("AIVILLAGE DEPENDENCY AUDIT REPORT")
    print("="*60)

    summary = report["summary"]
    print(f"Total unique packages: {summary['total_unique_packages']}")
    print(f"Active requirements files: {summary['active_requirements_files']}")
    print(f"Version conflicts: {summary['version_conflicts_count']}")
    print(f"Security-sensitive packages: {summary['security_sensitive_count']}")
    print(f"Heavy ML/AI packages: {summary['heavy_packages_count']}")

    print("\nTop Recommendations:")
    for i, rec in enumerate(report["recommendations"][:5], 1):
        print(f"{i}. {rec}")

    print("\nTop Version Conflicts:")
    for pkg, versions in list(report["version_conflicts"].items())[:5]:
        print(f"  - {pkg}: {sorted(versions)}")

    print("\nFull report saved to: dependency_audit_report.json")

    return report

if __name__ == "__main__":
    main()
