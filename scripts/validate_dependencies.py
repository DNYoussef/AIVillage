#!/usr/bin/env python3
"""Dependency validation script for AIVillage pyproject.toml consolidation.
Validates that all dependencies can be resolved and imported.
"""

import importlib
from pathlib import Path
import subprocess
import sys

import toml


def load_pyproject_dependencies() -> dict[str, list[str]]:
    """Load dependencies from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    with open(pyproject_path, encoding="utf-8") as f:
        data = toml.load(f)

    dependencies = {
        "core": data.get("project", {}).get("dependencies", []),
        **data.get("project", {}).get("optional-dependencies", {}),
    }

    return dependencies


def parse_requirement(req: str) -> str:
    """Extract package name from requirement string."""
    # Handle various requirement formats
    req = req.strip()

    # Remove environment markers
    if ";" in req:
        req = req.split(";")[0].strip()

    # Remove version specifications
    for op in [">=", "<=", "==", "!=", ">", "<", "~=", "^"]:
        if op in req:
            req = req.split(op)[0].strip()

    # Remove extras (e.g., package[extra])
    if "[" in req:
        req = req.split("[")[0].strip()

    # Handle URL requirements
    if req.startswith("git+") or req.startswith("http"):
        return None  # Skip URL requirements for import testing

    return req


def get_import_name(package_name: str) -> str:
    """Get the import name for a package (may differ from package name)."""
    # Common package name to import name mappings
    mapping = {
        "pillow": "PIL",
        "pyyaml": "yaml",
        "scikit-learn": "sklearn",
        "scikit-image": "skimage",
        "opencv-python": "cv2",
        "beautifulsoup4": "bs4",
        "python-dotenv": "dotenv",
        "python-multipart": "multipart",
        "python-jose": "jose",
        "msgpack": "msgpack",
        "psycopg2-binary": "psycopg2",
        "sentence-transformers": "sentence_transformers",
        "huggingface-hub": "huggingface_hub",
        "faiss-cpu": "faiss",
        "llama-cpp-python": "llama_cpp",
        "qdrant-client": "qdrant_client",
        "transformers": "transformers",
        "gradio": "gradio",
        "streamlit": "streamlit",
    }

    return mapping.get(package_name.lower(), package_name.replace("-", "_"))


def check_imports(packages: set[str]) -> tuple[list[str], list[str]]:
    """Check if packages can be imported."""
    successful = []
    failed = []

    for package in sorted(packages):
        if not package:  # Skip empty packages
            continue

        import_name = get_import_name(package)

        try:
            importlib.import_module(import_name)
            successful.append(package)
            print(f"OK {package} ({import_name})")
        except ImportError as e:
            failed.append(package)
            print(f"FAIL {package} ({import_name}): {e}")
        except Exception as e:
            failed.append(package)
            print(f"WARN {package} ({import_name}): Unexpected error - {e}")

    return successful, failed


def check_version_conflicts() -> bool:
    """Check for version conflicts using pip."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("OK No version conflicts detected")
            return True
        print("FAIL Version conflicts detected:")
        print(result.stdout)
        print(result.stderr)
        return False

    except subprocess.TimeoutExpired:
        print("WARN Timeout while checking version conflicts")
        return False
    except Exception as e:
        print(f"WARN Error checking version conflicts: {e}")
        return False


def generate_requirements_txt() -> None:
    """Generate requirements.txt files for backward compatibility."""
    dependencies = load_pyproject_dependencies()

    # Generate core requirements.txt
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write("# Generated from pyproject.toml - do not edit manually\n")
        f.write("# Use 'pip install -e .' for installation\n\n")
        for dep in dependencies["core"]:
            f.write(f"{dep}\n")

    # Generate optional requirements files
    for group, deps in dependencies.items():
        if group == "core":
            continue

        filename = f"requirements-{group}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"# Generated from pyproject.toml [{group}] - do not edit manually\n"
            )
            f.write(f"# Use 'pip install -e \".[{group}]\"' for installation\n\n")
            for dep in deps:
                f.write(f"{dep}\n")


def main():
    """Main validation function."""
    print("AIVillage Dependency Validation")
    print("=" * 40)

    try:
        # Load dependencies
        dependencies = load_pyproject_dependencies()
        print(f"Loaded {len(dependencies)} dependency groups")

        # Extract all unique packages
        all_packages = set()
        for group, deps in dependencies.items():
            packages = [parse_requirement(dep) for dep in deps]
            packages = [p for p in packages if p]  # Remove None values
            all_packages.update(packages)
            print(f"  {group}: {len(packages)} packages")

        print(f"\nTotal unique packages: {len(all_packages)}")
        print("-" * 40)

        # Check imports
        print("Testing imports...")
        successful, failed = check_imports(all_packages)

        print("\nImport Results:")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")

        if failed:
            print("\nFailed imports:")
            for package in failed:
                print(f"    - {package}")

        # Check version conflicts
        print("\nChecking version conflicts...")
        conflicts_ok = check_version_conflicts()

        # Generate requirements files for compatibility
        print("\nGenerating backward compatibility files...")
        generate_requirements_txt()
        print("Generated requirements.txt files")

        # Summary
        print("\nValidation Summary:")
        success_rate = len(successful) / len(all_packages) * 100 if all_packages else 0
        print(f"  Import success rate: {success_rate:.1f}%")
        print(f"  Version conflicts: {'None' if conflicts_ok else 'Found'}")
        print(
            f"  Status: {'PASS' if len(failed) == 0 and conflicts_ok else 'ISSUES FOUND'}"
        )

        if failed or not conflicts_ok:
            sys.exit(1)

    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
