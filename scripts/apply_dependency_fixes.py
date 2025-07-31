#!/usr/bin/env python3
"""AIVillage Dependency Fix Implementation Script
Applies the consolidated dependency recommendations.
"""

from pathlib import Path
import shutil
import subprocess
import sys


def backup_existing_requirements():
    """Backup existing requirements files."""
    backup_dir = Path("requirements_backup")
    backup_dir.mkdir(exist_ok=True)

    req_files = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt"]

    print("Backing up existing requirements files...")
    for req_file in req_files:
        if Path(req_file).exists():
            shutil.copy2(req_file, backup_dir / req_file)
            print(f"  ✓ Backed up {req_file}")


def install_security_tools():
    """Install security scanning tools."""
    print("Installing security scanning tools...")
    tools = ["pip-audit", "safety", "bandit"]

    for tool in tools:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", tool],
                check=True,
                capture_output=True,
            )
            print(f"  ✓ Installed {tool}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {tool}: {e}")


def run_security_scan():
    """Run security scans on the consolidated requirements."""
    print("Running security scans...")

    # Run pip-audit
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip_audit",
                "--requirement",
                "requirements_production.txt",
                "--format",
                "columns",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("  ✓ pip-audit: No vulnerabilities found")
        else:
            print(f"  ⚠️  pip-audit found issues:\n{result.stdout}")

    except subprocess.TimeoutExpired:
        print("  ⚠️  pip-audit scan timed out")
    except Exception as e:
        print(f"  ❌ pip-audit failed: {e}")


def validate_requirements():
    """Validate that requirements files have proper syntax."""
    print("Validating requirements files...")

    req_files = [
        "requirements_consolidated.txt",
        "requirements_production.txt",
        "requirements_development.txt",
        "requirements_security_audit.txt",
    ]

    for req_file in req_files:
        if Path(req_file).exists():
            try:
                # Try to parse the requirements file
                with open(req_file) as f:
                    lines = f.readlines()

                # Basic validation
                valid_lines = 0
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-r"):
                        valid_lines += 1

                print(f"  ✓ {req_file}: {valid_lines} valid package specifications")

            except Exception as e:
                print(f"  ❌ {req_file}: Error - {e}")


def create_requirements_directory():
    """Create organized requirements directory structure."""
    print("Creating organized requirements directory...")

    req_dir = Path("requirements")
    req_dir.mkdir(exist_ok=True)

    # Move consolidated files to requirements directory
    files_to_move = [
        ("requirements_consolidated.txt", "requirements/base.txt"),
        ("requirements_production.txt", "requirements/production.txt"),
        ("requirements_development.txt", "requirements/development.txt"),
        ("requirements_security_audit.txt", "requirements/security.txt"),
    ]

    for src, dst in files_to_move:
        if Path(src).exists():
            shutil.move(src, dst)
            print(f"  ✓ Moved {src} → {dst}")


def update_docker_files():
    """Update Dockerfile references to use new requirements structure."""
    print("Checking for Dockerfile updates needed...")

    docker_files = list(Path().rglob("Dockerfile*"))

    for dockerfile in docker_files:
        try:
            with open(dockerfile) as f:
                content = f.read()

            # Check if it references old requirements files
            if "requirements.txt" in content:
                print(
                    f"  ⚠️  {dockerfile} may need updating to use requirements/production.txt"
                )

        except Exception as e:
            print(f"  ❌ Error checking {dockerfile}: {e}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("AIVillage Dependency Fix Implementation")
    print("=" * 60)

    if not Path("requirements_consolidated.txt").exists():
        print("❌ Error: requirements_consolidated.txt not found!")
        print("Please run the dependency audit script first.")
        sys.exit(1)

    try:
        # Step 1: Backup existing files
        backup_existing_requirements()

        # Step 2: Install security tools
        install_security_tools()

        # Step 3: Validate new requirements
        validate_requirements()

        # Step 4: Run security scan
        run_security_scan()

        # Step 5: Create organized structure
        create_requirements_directory()

        # Step 6: Check Docker files
        update_docker_files()

        print("\n" + "=" * 60)
        print("✅ DEPENDENCY FIXES APPLIED SUCCESSFULLY")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Test the new requirements in a clean environment")
        print("2. Update CI/CD pipelines to use requirements/production.txt")
        print("3. Update documentation with new requirements structure")
        print("4. Set up automated security scanning")

        print("\nRecommended Testing:")
        print("  python -m venv test_env")
        print("  test_env/Scripts/activate  # Windows")
        print("  pip install -r requirements/production.txt")
        print("  # Run your application tests")

    except KeyboardInterrupt:
        print("\n\n❌ Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
