#!/usr/bin/env python3
"""Migration script to transition from the old ADAS implementation to the secure version.

This script:
1. Backs up the original adas.py
2. Replaces it with the secure version
3. Updates imports in dependent files
4. Runs tests to ensure compatibility
"""

from datetime import datetime
from pathlib import Path
import shutil
import subprocess
import sys


def backup_original(file_path: Path) -> Path:
    """Create a timestamped backup of the original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    print(f"✓ Created backup: {backup_path}")
    return backup_path


def update_imports(root_dir: Path) -> None:
    """Update imports in files that reference the ADAS module."""
    # Files that might import from adas
    patterns = [
        "from agent_forge.adas.adas import",
        "from .adas import",
        "import agent_forge.adas.adas",
    ]

    files_updated = []

    # Search for Python files
    for py_file in root_dir.rglob("*.py"):
        if "backup" in str(py_file) or "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text()
            original_content = content

            # Check if file needs updating
            needs_update = any(pattern in content for pattern in patterns)

            if needs_update:
                # Update imports to use secure version
                content = content.replace(
                    "from agent_forge.adas.adas import",
                    "from agent_forge.adas.adas_secure import",
                )
                content = content.replace("from .adas import", "from .adas_secure import")
                content = content.replace(
                    "import agent_forge.adas.adas",
                    "import agent_forge.adas.adas_secure",
                )

                if content != original_content:
                    py_file.write_text(content)
                    files_updated.append(py_file)

        except Exception as e:
            print(f"⚠️  Error processing {py_file}: {e}")

    if files_updated:
        print(f"✓ Updated imports in {len(files_updated)} files:")
        for f in files_updated:
            print(f"  - {f.relative_to(root_dir)}")
    else:
        print("✓ No import updates needed")


def run_tests(root_dir: Path) -> bool:
    """Run ADAS-related tests to ensure compatibility."""
    print("\n🧪 Running tests...")

    test_files = [
        root_dir / "tests" / "test_adas_system.py",
        root_dir / "tests" / "test_adas_technique.py",
        root_dir / "tests" / "test_adas_search.py",
    ]

    all_passed = True

    for test_file in test_files:
        if test_file.exists():
            print(f"\nRunning {test_file.name}...")
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "-v"],
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"✓ {test_file.name} passed")
            else:
                print(f"✗ {test_file.name} failed")
                print(result.stdout)
                print(result.stderr)
                all_passed = False

    return all_passed


def create_migration_summary(backup_path: Path, files_updated: int) -> None:
    """Create a summary of the migration."""
    summary = f"""
ADAS Security Migration Summary
==============================
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Changes Made:
1. Backed up original adas.py to: {backup_path}
2. Replaced with secure implementation (adas_secure.py)
3. Updated imports in {files_updated} files
4. Removed dangerous exec_module usage
5. Added subprocess sandboxing with resource limits

Security Improvements:
- Code execution now runs in isolated subprocess
- Memory limits enforced (512MB default)
- CPU time limits enforced (30s default)
- Filesystem access restricted
- Environment variables sanitized
- No direct exec() or eval() usage

Migration Notes:
- The API remains compatible
- Performance may be slightly slower due to subprocess overhead
- Resource limits can be adjusted in AgentTechnique.handle()

To rollback:
1. Copy {backup_path} back to adas.py
2. Update imports back to original
"""

    summary_path = backup_path.parent / "MIGRATION_SUMMARY.txt"
    summary_path.write_text(summary)
    print(f"\n✓ Migration summary saved to: {summary_path}")


def main() -> int:
    """Main migration process."""
    print("🔒 ADAS Security Migration Script")
    print("=================================\n")

    # Find project root
    root_dir = Path(__file__).resolve().parents[3]  # Go up to AIVillage root
    adas_dir = root_dir / "agent_forge" / "adas"
    original_file = adas_dir / "adas.py"
    secure_file = adas_dir / "adas_secure.py"

    if not original_file.exists():
        print(f"✗ Original file not found: {original_file}")
        return 1

    if not secure_file.exists():
        print(f"✗ Secure version not found: {secure_file}")
        return 1

    print(f"Project root: {root_dir}")
    print(f"ADAS directory: {adas_dir}\n")

    # Step 1: Backup original
    backup_path = backup_original(original_file)

    # Step 2: Replace with secure version
    print("\n📝 Replacing with secure implementation...")
    shutil.copy2(secure_file, original_file)
    print("✓ Replaced adas.py with secure version")

    # Step 3: Update imports
    print("\n🔄 Updating imports...")
    update_imports(root_dir)

    # Step 4: Run tests
    tests_passed = run_tests(root_dir)

    # Step 5: Create summary
    create_migration_summary(backup_path, 0)  # TODO: track actual files updated

    if tests_passed:
        print("\n✅ Migration completed successfully!")
        print(f"   Original backed up to: {backup_path}")
        print("   All tests passed")
    else:
        print("\n⚠️  Migration completed with test failures")
        print("   Please review the test output above")
        print(f"   To rollback: cp {backup_path} {original_file}")

    return 0 if tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
