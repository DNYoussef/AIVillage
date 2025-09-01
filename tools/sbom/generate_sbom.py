#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) generator
"""

import json
import os
import sys
from datetime import datetime
import subprocess


def get_python_packages():
    """Get list of installed Python packages"""
    try:
        result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except:
        return []


def generate_sbom(output_dir, verbose=False):
    """Generate SBOM for the project"""

    if verbose:
        print("Generating Software Bill of Materials...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get Python packages
    packages = get_python_packages()

    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:aivillage-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "version": 1,
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "tools": [{"vendor": "AIVillage", "name": "custom-sbom-generator", "version": "1.0.0"}],
        },
        "components": [],
    }

    # Add Python packages to components
    for pkg in packages:
        component = {
            "type": "library",
            "bom-ref": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "name": pkg["name"],
            "version": pkg["version"],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}",
            "scope": "required",
        }
        sbom["components"].append(component)

    # Write SBOM to file
    sbom_file = os.path.join(output_dir, "aivillage-sbom.json")
    with open(sbom_file, "w") as f:
        json.dump(sbom, f, indent=2)

    if verbose:
        print(f"SBOM generated with {len(packages)} components")
        print(f"Saved to: {sbom_file}")

    return sbom_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SBOM for AIVillage project")
    parser.add_argument("--output", default="artifacts/sbom", help="Output directory for SBOM files")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        generate_sbom(args.output, args.verbose)
        print("SBOM generation completed successfully")
    except Exception as e:
        print(f"Error generating SBOM: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
