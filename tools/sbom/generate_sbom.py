#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for AIVillage
Generates comprehensive SBOM including Python, Rust, and Go dependencies.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def get_python_dependencies() -> List[Dict[str, Any]]:
    """Get Python dependencies from pip."""
    try:
        result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return []


def get_rust_dependencies() -> List[Dict[str, Any]]:
    """Get Rust dependencies from Cargo.lock if available."""
    rust_deps = []

    # Check multiple possible Cargo.lock locations
    lock_files = ["Cargo.lock", "build/core-build/Cargo.lock", "packages/p2p/betanet-bounty/Cargo.lock"]

    for lock_file in lock_files:
        if Path(lock_file).exists():
            try:
                # Parse Cargo.lock for dependency info
                with open(lock_file, "r") as f:
                    content = f.read()
                    # Simple parsing - in production use proper TOML parser
                    lines = content.split("\n")
                    current_package = None

                    for line in lines:
                        if line.startswith("[[package]]"):
                            current_package = {}
                        elif line.startswith("name = "):
                            if current_package is not None:
                                current_package["name"] = line.split('"')[1]
                        elif line.startswith("version = "):
                            if current_package is not None:
                                current_package["version"] = line.split('"')[1]
                                rust_deps.append(
                                    {
                                        "name": current_package["name"],
                                        "version": current_package["version"],
                                        "type": "rust",
                                    }
                                )
                                current_package = None
            except Exception as e:
                print(f"Warning: Could not parse {lock_file}: {e}")

    return rust_deps


def get_go_dependencies() -> List[Dict[str, Any]]:
    """Get Go dependencies from go.mod if available."""
    go_deps = []

    # Check for Go modules
    go_mod_files = ["go.mod", "clients/rust/scion-sidecar/go.mod"]

    for mod_file in go_mod_files:
        if Path(mod_file).exists():
            try:
                with open(mod_file, "r") as f:
                    content = f.read()
                    lines = content.split("\n")

                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("//") and not line.startswith("module"):
                            if " v" in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    go_deps.append({"name": parts[0], "version": parts[1], "type": "go"})
            except Exception as e:
                print(f"Warning: Could not parse {mod_file}: {e}")

    return go_deps


def get_javascript_dependencies() -> List[Dict[str, Any]]:
    """Get JavaScript dependencies from package.json files."""
    js_deps = []

    # Look for package.json files
    package_files = list(Path(".").rglob("package.json"))

    for package_file in package_files:
        try:
            with open(package_file, "r") as f:
                package_data = json.load(f)

                # Get dependencies
                for dep_type in ["dependencies", "devDependencies"]:
                    if dep_type in package_data:
                        for name, version in package_data[dep_type].items():
                            js_deps.append({"name": name, "version": version, "type": "javascript"})
        except Exception as e:
            print(f"Warning: Could not parse {package_file}: {e}")

    return js_deps


def generate_sbom(output_dir: str, verbose: bool = False) -> Dict[str, Any]:
    """Generate comprehensive SBOM."""

    print("🔍 Generating Software Bill of Materials (SBOM)...")

    # Collect all dependencies
    python_deps = get_python_dependencies()
    rust_deps = get_rust_dependencies()
    go_deps = get_go_dependencies()
    js_deps = get_javascript_dependencies()

    if verbose:
        print(f"   Found {len(python_deps)} Python dependencies")
        print(f"   Found {len(rust_deps)} Rust dependencies")
        print(f"   Found {len(go_deps)} Go dependencies")
        print(f"   Found {len(js_deps)} JavaScript dependencies")

    # Create SBOM structure
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "version": 1,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "component": {
                "type": "application",
                "name": "AIVillage",
                "version": "1.0.0",
                "description": "AI Village distributed intelligence platform",
            },
            "tools": [{"vendor": "AIVillage", "name": "sbom-generator", "version": "1.0.0"}],
        },
        "components": python_deps + rust_deps + go_deps + js_deps,
        "summary": {
            "total_components": len(python_deps + rust_deps + go_deps + js_deps),
            "python_components": len(python_deps),
            "rust_components": len(rust_deps),
            "go_components": len(go_deps),
            "javascript_components": len(js_deps),
        },
    }

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write SBOM files
    sbom_file = Path(output_dir) / "aivillage-sbom.json"
    with open(sbom_file, "w") as f:
        json.dump(sbom, f, indent=2)

    # Write human-readable summary
    summary_file = Path(output_dir) / "sbom-summary.txt"
    with open(summary_file, "w") as f:
        f.write("AIVillage Software Bill of Materials\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        f.write(f"Total Components: {sbom['summary']['total_components']}\n\n")

        for dep_type in ["python", "rust", "go", "javascript"]:
            count = sbom["summary"][f"{dep_type}_components"]
            f.write(f"{dep_type.title()} Dependencies: {count}\n")

        f.write("\n" + "=" * 40 + "\n")
        f.write("Detailed Component List:\n\n")

        for component in sbom["components"]:
            f.write(f"- {component['name']} v{component['version']} ({component.get('type', 'unknown')})\n")

    print(f"✅ SBOM generated successfully!")
    print(f"   📄 SBOM file: {sbom_file}")
    print(f"   📋 Summary file: {summary_file}")

    return sbom


def main():
    parser = argparse.ArgumentParser(description="Generate SBOM for AIVillage")
    parser.add_argument("--output", default="artifacts/sbom", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        sbom = generate_sbom(args.output, args.verbose)
        print(f"🎉 SBOM generation completed with {sbom['summary']['total_components']} components")
        return 0
    except Exception as e:
        print(f"❌ Error generating SBOM: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
