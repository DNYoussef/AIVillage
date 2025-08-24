#!/usr/bin/env python3
"""
Setup script for architectural fitness function tools

This script installs and configures all necessary tools for architectural analysis.
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parent.parent


def install_requirements():
    """Install required packages for architectural analysis"""
    requirements_file = PROJECT_ROOT / "config" / "requirements-architecture.txt"

    if not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}")
        return False

    print("Installing architectural analysis requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False


def setup_directories():
    """Create necessary directories"""
    directories = ["reports/architecture", "reports/ci", "tests/architecture", "config"]

    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def validate_installation():
    """Validate that all tools are working correctly"""
    print("\nValidating installation...")

    # Test imports
    required_modules = ["pytest", "networkx", "matplotlib", "seaborn", "pandas", "numpy", "radon", "yaml", "jinja2"]

    failed_imports = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)

    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False

    print("\n✅ All modules imported successfully")
    return True


def run_test_fitness_functions():
    """Run a quick test of the fitness functions"""
    print("\nRunning test fitness functions...")

    test_file = PROJECT_ROOT / "tests" / "architecture" / "test_fitness_functions.py"
    if not test_file.exists():
        print("❌ Fitness functions test file not found")
        return False

    try:
        # Run a dry-run test
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        if result.returncode == 0:
            print("✅ Fitness functions tests are ready")
            return True
        else:
            print(f"❌ Test collection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    print("🏗️  Setting up AIVillage Architectural Fitness Functions")
    print("=" * 60)

    success = True

    # Step 1: Setup directories
    print("\n📁 Creating directories...")
    setup_directories()

    # Step 2: Install requirements
    print("\n📦 Installing requirements...")
    if not install_requirements():
        success = False

    # Step 3: Validate installation
    print("\n🔍 Validating installation...")
    if not validate_installation():
        success = False

    # Step 4: Test fitness functions
    print("\n🧪 Testing fitness functions...")
    if not run_test_fitness_functions():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run fitness functions: python -m pytest tests/architecture/")
        print("2. Run analysis: python scripts/architectural_analysis.py")
        print("3. Setup CI: python scripts/ci_integration.py --mode github-actions")
    else:
        print("❌ Setup completed with errors!")
        print("Please check the error messages above and resolve any issues.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
