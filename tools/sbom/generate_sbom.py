#!/usr/bin/env python3
"""
SBOM (Software Bill of Materials) Generator for AIVillage

This script generates comprehensive SBOMs in multiple formats:
- CycloneDX JSON format
- SPDX JSON format
- Human-readable report

Supports both pip packages and system analysis.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pkg_resources

try:
    from cyclonedx.model import Component, ComponentType
    from cyclonedx.model.bom import Bom
    from cyclonedx.output.json import JsonV1Dot4

    CYCLONE_AVAILABLE = True
except ImportError:
    CYCLONE_AVAILABLE = False
    print("Warning: cyclonedx-python not available. Install with: pip install cyclonedx-bom")


class SBOMGenerator:
    """Generate Software Bill of Materials for AIVillage."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.timestamp = datetime.utcnow().isoformat() + "Z"
        self.project_name = "AIVillage"
        self.project_version = self._get_project_version()

    def _get_project_version(self) -> str:
        """Extract project version from various sources."""
        # Try pyproject.toml
        pyproject = self.project_root / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                for line in content.split("\n"):
                    if line.startswith("version ="):
                        return line.split("=")[1].strip().strip("\"'")
            except Exception:
                pass

        # Try setup.py
        setup_py = self.project_root / "setup.py"
        if setup_py.exists():
            try:
                content = setup_py.read_text()
                # Simple regex for version
                import re

                match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if match:
                    return match.group(1)
            except Exception:
                pass

        # Fallback to git tag or default
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True, cwd=self.project_root
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return "1.0.0-dev"

    def _get_pip_packages(self) -> list[dict[str, Any]]:
        """Get list of installed pip packages with metadata."""
        packages = []

        try:
            # Get pip freeze output
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Warning: pip freeze failed: {result.stderr}")
                return packages

            for line in result.stdout.strip().split("\n"):
                if line and not line.startswith("#") and "==" in line:
                    name, version = line.split("==", 1)

                    # Get additional metadata
                    package_info = {
                        "name": name.strip(),
                        "version": version.strip(),
                        "type": "library",
                        "scope": "required",
                        "source": "pip",
                    }

                    # Try to get more details using pkg_resources
                    try:
                        dist = pkg_resources.get_distribution(name)
                        package_info.update(
                            {
                                "location": dist.location,
                                "requires": [str(req) for req in dist.requires()],
                                "metadata": {
                                    "summary": getattr(dist, "summary", ""),
                                    "homepage": getattr(dist, "homepage", ""),
                                    "license": self._extract_license(dist),
                                },
                            }
                        )
                    except Exception as e:
                        package_info["metadata"] = {"error": str(e)}

                    packages.append(package_info)

        except Exception as e:
            print(f"Error getting pip packages: {e}")

        return packages

    def _extract_license(self, dist) -> str:
        """Extract license information from package metadata."""
        try:
            if hasattr(dist, "get_metadata"):
                metadata = dist.get_metadata("METADATA")
                for line in metadata.split("\n"):
                    if line.startswith("License:"):
                        return line.split(":", 1)[1].strip()
                    elif line.startswith("Classifier: License"):
                        return line.split("::")[-1].strip()
        except Exception:
            pass
        return "Unknown"

    def _get_system_info(self) -> dict[str, Any]:
        """Get system and environment information."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os": {
                "name": os.name,
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "environment": {
                "path": os.environ.get("PATH", ""),
                "python_path": os.environ.get("PYTHONPATH", ""),
                "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            },
        }

    def _get_project_files(self) -> list[dict[str, Any]]:
        """Get key project files with checksums."""
        key_files = [
            "requirements.txt",
            "requirements-prod.txt",
            "requirements-dev.txt",
            "requirements-security.txt",
            "constraints.txt",
            "pyproject.toml",
            "setup.py",
            "Dockerfile",
            "docker-compose.yml",
        ]

        files = []
        for filename in key_files:
            file_path = self.project_root / filename
            if file_path.exists():
                try:
                    content = file_path.read_bytes()
                    files.append(
                        {
                            "name": filename,
                            "path": str(file_path.relative_to(self.project_root)),
                            "size": len(content),
                            "checksums": {
                                "sha256": hashlib.sha256(content).hexdigest(),
                                "md5": hashlib.md5(content).hexdigest(),
                            },
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        }
                    )
                except Exception as e:
                    files.append(
                        {"name": filename, "path": str(file_path.relative_to(self.project_root)), "error": str(e)}
                    )

        return files

    def generate_cyclonedx(self) -> str | None:
        """Generate CycloneDX SBOM."""
        if not CYCLONE_AVAILABLE:
            return None

        try:
            bom = Bom()

            # Add main component
            main_component = Component(
                name=self.project_name, version=self.project_version, component_type=ComponentType.APPLICATION
            )
            bom.metadata.component = main_component

            # Add package components
            packages = self._get_pip_packages()
            for pkg in packages:
                component = Component(name=pkg["name"], version=pkg["version"], component_type=ComponentType.LIBRARY)
                bom.components.add(component)

            # Generate JSON
            json_output = JsonV1Dot4(bom)
            return json_output.output_as_string()

        except Exception as e:
            print(f"Error generating CycloneDX SBOM: {e}")
            return None

    def generate_spdx(self) -> dict[str, Any]:
        """Generate SPDX SBOM."""
        packages = self._get_pip_packages()
        system_info = self._get_system_info()
        project_files = self._get_project_files()

        return {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{self.project_name}-{self.project_version}",
            "documentNamespace": f"https://github.com/aivillage/{self.project_name.lower()}/sbom/{uuid.uuid4()}",
            "creationInfo": {
                "created": self.timestamp,
                "creators": ["Tool: AIVillage-SBOM-Generator"],
                "licenseListVersion": "3.21",
            },
            "packages": [
                {
                    "SPDXID": f"SPDXRef-Package-{pkg['name']}",
                    "name": pkg["name"],
                    "versionInfo": pkg["version"],
                    "packageDownloadLocation": "NOASSERTION",
                    "filesAnalyzed": False,
                    "licenseConcluded": pkg.get("metadata", {}).get("license", "NOASSERTION"),
                    "licenseDeclared": pkg.get("metadata", {}).get("license", "NOASSERTION"),
                    "copyrightText": "NOASSERTION",
                    "externalRefs": [
                        {
                            "referenceCategory": "PACKAGE-MANAGER",
                            "referenceType": "purl",
                            "referenceLocator": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
                        }
                    ],
                }
                for pkg in packages
            ],
            "relationships": [
                {
                    "spdxElementId": "SPDXRef-DOCUMENT",
                    "relationshipType": "DESCRIBES",
                    "relatedSpdxElement": f"SPDXRef-Package-{self.project_name}",
                }
            ]
            + [
                {
                    "spdxElementId": f"SPDXRef-Package-{self.project_name}",
                    "relationshipType": "DEPENDS_ON",
                    "relatedSpdxElement": f"SPDXRef-Package-{pkg['name']}",
                }
                for pkg in packages
            ],
            "annotations": [
                {
                    "annotationType": "OTHER",
                    "annotator": "Tool: AIVillage-SBOM-Generator",
                    "annotationDate": self.timestamp,
                    "annotationComment": f"Generated for {system_info['platform']}",
                }
            ],
            "metadata": {
                "system_info": system_info,
                "project_files": project_files,
                "package_count": len(packages),
                "generation_tool": "AIVillage SBOM Generator v1.0",
            },
        }

    def generate_human_readable(self) -> str:
        """Generate human-readable SBOM report."""
        packages = self._get_pip_packages()
        system_info = self._get_system_info()
        project_files = self._get_project_files()

        report = f"""
# AIVillage Software Bill of Materials (SBOM)
Generated: {self.timestamp}
Project: {self.project_name} v{self.project_version}

## System Information
- Platform: {system_info['platform']}
- Python: {system_info['python_version']} ({system_info['python_implementation']})
- Architecture: {system_info['architecture']} ({system_info['machine']})
- OS: {system_info['os']['system']} {system_info['os']['release']}

## Environment
- Virtual Environment: {system_info['environment']['virtual_env'] or 'None'}
- Conda Environment: {system_info['environment']['conda_env'] or 'None'}

## Package Dependencies ({len(packages)} total)
"""

        # Sort packages by name
        packages_sorted = sorted(packages, key=lambda x: x["name"].lower())

        for pkg in packages_sorted:
            report += f"\n### {pkg['name']} {pkg['version']}\n"
            if "metadata" in pkg and "summary" in pkg["metadata"]:
                report += f"- Summary: {pkg['metadata']['summary']}\n"
            if "metadata" in pkg and "license" in pkg["metadata"]:
                report += f"- License: {pkg['metadata']['license']}\n"
            if "requires" in pkg and pkg["requires"]:
                report += f"- Dependencies: {', '.join(pkg['requires'])}\n"

        report += f"\n## Project Files ({len(project_files)} key files)\n"
        for file_info in project_files:
            if "error" in file_info:
                report += f"\n- {file_info['name']}: ERROR - {file_info['error']}\n"
            else:
                report += f"\n- {file_info['name']} ({file_info['size']} bytes)\n"
                report += f"  - SHA256: {file_info['checksums']['sha256']}\n"
                report += f"  - Modified: {file_info['modified']}\n"

        report += """
## Security & Compliance Notes
- All dependencies should be regularly scanned for vulnerabilities
- Use `pip-audit --desc` to check for known CVEs
- Use `safety check` for additional vulnerability scanning
- Review license compliance for all dependencies
- Consider using `pip-licenses` for detailed license analysis

## Verification
To verify this SBOM:
1. Check package versions: `pip freeze`
2. Scan for vulnerabilities: `pip-audit --desc`
3. Check for security issues: `safety check`
4. Verify checksums of key files

Generated by AIVillage SBOM Generator v1.0
"""

        return report

    def generate_all_formats(self, output_dir: Path) -> dict[str, str]:
        """Generate SBOM in all available formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}

        # Generate CycloneDX
        cyclonedx_content = self.generate_cyclonedx()
        if cyclonedx_content:
            cyclonedx_file = output_dir / "sbom-cyclonedx.json"
            cyclonedx_file.write_text(cyclonedx_content)
            results["cyclonedx"] = str(cyclonedx_file)
            print(f"✅ Generated CycloneDX SBOM: {cyclonedx_file}")
        else:
            print("❌ Could not generate CycloneDX SBOM (missing dependency)")

        # Generate SPDX
        spdx_content = self.generate_spdx()
        spdx_file = output_dir / "sbom-spdx.json"
        with open(spdx_file, "w") as f:
            json.dump(spdx_content, f, indent=2, sort_keys=True)
        results["spdx"] = str(spdx_file)
        print(f"✅ Generated SPDX SBOM: {spdx_file}")

        # Generate human-readable
        human_content = self.generate_human_readable()
        human_file = output_dir / "sbom-report.md"
        human_file.write_text(human_content)
        results["human"] = str(human_file)
        print(f"✅ Generated Human-readable SBOM: {human_file}")

        # Generate summary
        packages = self._get_pip_packages()
        summary = {
            "generation_time": self.timestamp,
            "project": {"name": self.project_name, "version": self.project_version},
            "statistics": {
                "total_packages": len(packages),
                "unique_licenses": len(set(pkg.get("metadata", {}).get("license", "Unknown") for pkg in packages)),
                "formats_generated": list(results.keys()),
            },
            "files": results,
        }

        summary_file = output_dir / "sbom-summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        results["summary"] = str(summary_file)
        print(f"✅ Generated SBOM Summary: {summary_file}")

        return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Generate Software Bill of Materials (SBOM) for AIVillage")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path.cwd() / "artifacts" / "sbom",
        help="Output directory for SBOM files (default: ./artifacts/sbom/)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["cyclonedx", "spdx", "human", "all"],
        default="all",
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--project-root",
        "-r",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        print(f"Generating SBOM for project at: {args.project_root}")
        print(f"Output directory: {args.output}")
        print(f"Format: {args.format}")

    generator = SBOMGenerator(args.project_root)

    if args.format == "all":
        results = generator.generate_all_formats(args.output)
        print(f"\n🎉 Successfully generated SBOM in {len(results)} formats")
        if args.verbose:
            for format_name, file_path in results.items():
                print(f"  - {format_name}: {file_path}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)

        if args.format == "cyclonedx":
            content = generator.generate_cyclonedx()
            if content:
                output_file = args.output / "sbom-cyclonedx.json"
                output_file.write_text(content)
                print(f"✅ Generated CycloneDX SBOM: {output_file}")
            else:
                print("❌ Could not generate CycloneDX SBOM")
                sys.exit(1)
        elif args.format == "spdx":
            content = generator.generate_spdx()
            output_file = args.output / "sbom-spdx.json"
            with open(output_file, "w") as f:
                json.dump(content, f, indent=2, sort_keys=True)
            print(f"✅ Generated SPDX SBOM: {output_file}")
        elif args.format == "human":
            content = generator.generate_human_readable()
            output_file = args.output / "sbom-report.md"
            output_file.write_text(content)
            print(f"✅ Generated Human-readable SBOM: {output_file}")


if __name__ == "__main__":
    main()
