#!/usr/bin/env python3
"""Fix dependency conflicts in AIVillage project.

This script provides automated fixes for common dependency issues.
"""

from pathlib import Path
import subprocess
import sys


def run_command(cmd: list[str], description: str = "") -> bool:
    """Run a command and return success status."""
    if description:
        print(f"Running: {description}")

    print(f"$ {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def fix_numpy_compatibility():
    """Fix NumPy 2.x compatibility issues."""
    print("=== Fixing NumPy Compatibility ===")

    # Ensure NumPy 1.x is installed for compatibility
    commands = [
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "numpy>=1.24.3,<2.0.0",
            ],
            "Installing compatible NumPy version",
        ),
    ]

    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False

    return True


def fix_core_dependencies():
    """Fix core dependency conflicts."""
    print("\n=== Fixing Core Dependencies ===")

    # Core packages that should be updated
    core_packages = [
        "pydantic>=2.8.2",
        "fastapi>=0.95.1",
        "uvicorn>=0.22.0",
        "httpx>=0.27.0",
        "aiohttp>=3.10.5",
        "requests>=2.32.3",
        "urllib3>=2.2.0",
        "certifi",
        "pillow>=10.4.0",
    ]

    for package in core_packages:
        if not run_command(
            [sys.executable, "-m", "pip", "install", "--upgrade", package],
            f"Updating {package}",
        ):
            print(f"Warning: Failed to update {package}")

    return True


def fix_tenacity_conflicts():
    """Fix tenacity version conflicts."""
    print("\n=== Fixing Tenacity Conflicts ===")

    # Downgrade tenacity to resolve conflicts with langchain and llama-index
    return run_command(
        [sys.executable, "-m", "pip", "install", "tenacity<9.0.0,>=8.2.0"],
        "Installing compatible tenacity version",
    )


def fix_typer_conflicts():
    """Fix typer version conflicts."""
    print("\n=== Fixing Typer Conflicts ===")

    # Downgrade typer to resolve conflicts with langroid
    return run_command(
        [sys.executable, "-m", "pip", "install", "typer<0.10.0,>=0.9.0"],
        "Installing compatible typer version",
    )


def fix_protobuf_conflicts():
    """Fix protobuf version conflicts."""
    print("\n=== Fixing Protobuf Conflicts ===")

    # Downgrade protobuf to resolve conflicts
    return run_command(
        [sys.executable, "-m", "pip", "install", "protobuf<5.0.0,>=4.21.6"],
        "Installing compatible protobuf version",
    )


def update_requirements_txt():
    """Update requirements.txt with fixed versions."""
    print("\n=== Updating requirements.txt ===")

    project_root = Path.cwd()
    requirements_file = project_root / "requirements.txt"

    if not requirements_file.exists():
        print("No requirements.txt found, skipping update")
        return True

    # Read current requirements
    with open(requirements_file) as f:
        lines = f.readlines()

    # Update specific packages with fixed versions
    fixes = {
        "numpy": "numpy>=1.24.3,<2.0.0",
        "tenacity": "tenacity<9.0.0,>=8.2.0",
        "typer": "typer<0.10.0,>=0.9.0",
        "protobuf": "protobuf<5.0.0,>=4.21.6",
        "pydantic": "pydantic>=2.8.2",
        "fastapi": "fastapi>=0.95.1",
        "aiohttp": "aiohttp>=3.10.5",
        "pillow": "pillow>=10.4.0",
    }

    updated_lines = []
    updated_packages = set()

    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name
            package_name = (
                line.split(">=")[0].split("==")[0].split("<")[0].split(">")[0]
            )
            if package_name in fixes:
                updated_lines.append(fixes[package_name])
                updated_packages.add(package_name)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # Add any missing fixed packages
    for package, fix in fixes.items():
        if package not in updated_packages:
            updated_lines.append(fix)

    # Write updated requirements
    with open(requirements_file, "w") as f:
        for line in updated_lines:
            f.write(line + "\n")

    print(f"Updated {requirements_file}")
    return True


def cleanup_conflicts():
    """Clean up remaining conflicts."""
    print("\n=== Cleaning Up Conflicts ===")

    # Try to resolve remaining conflicts by reinstalling problem packages
    commands = [
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "pip",
            ],
            "Updating pip",
        ),
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "setuptools",
                "wheel",
            ],
            "Updating build tools",
        ),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

    return True


def verify_fixes():
    """Verify that fixes have been applied successfully."""
    print("\n=== Verifying Fixes ===")

    # Check for remaining conflicts
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            print("✓ All dependency conflicts resolved!")
            return True
        print("⚠ Some conflicts remain:")
        if result.stdout:
            print(result.stdout)
        return False

    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False


def main():
    """Main function to fix all dependency issues."""
    print("AIVillage Dependency Fix Script")
    print("=" * 40)

    fixes = [
        fix_numpy_compatibility,
        fix_tenacity_conflicts,
        fix_typer_conflicts,
        fix_protobuf_conflicts,
        fix_core_dependencies,
        update_requirements_txt,
        cleanup_conflicts,
        verify_fixes,
    ]

    success_count = 0
    for fix_func in fixes:
        try:
            if fix_func():
                success_count += 1
            else:
                print(f"Warning: {fix_func.__name__} completed with issues")
        except Exception as e:
            print(f"Error in {fix_func.__name__}: {e}")

    print("\n=== Summary ===")
    print(f"Completed {success_count}/{len(fixes)} fixes")

    if success_count == len(fixes):
        print("✓ All dependency fixes applied successfully!")
        print("\nNext steps:")
        print("1. Test your application to ensure everything works")
        print("2. Run tests: python -m pytest")
        print("3. Consider creating a virtual environment for isolated development")
    else:
        print("⚠ Some fixes encountered issues")
        print("You may need to manually resolve remaining conflicts")
        print("Run 'pip check' to see remaining issues")


if __name__ == "__main__":
    main()
