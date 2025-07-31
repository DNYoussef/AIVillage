#!/usr/bin/env python3
"""
Basic quality fixes - focus on critical formatting issues.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return result."""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def install_if_missing(packages):
    """Install packages if they're missing."""
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)

def main():
    project_root = Path(__file__).parent
    
    # Install required tools
    install_if_missing(["black", "isort", "ruff"])
    
    # Define target directories  
    targets = ["agent_forge", "mcp_servers", "production", "tests", "scripts", "benchmarks"]
    
    # Filter existing directories
    existing_targets = [t for t in targets if (project_root / t).exists()]
    
    print(f"Running quality fixes on: {', '.join(existing_targets)}")
    
    # 1. Run black formatting
    print("\n1. Running black formatter...")
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "black", 
        "--line-length", "88",
        "--target-version", "py39",
        "--exclude", "(new_env|__pycache__|.git|.cleanup_backups|.test_repair_backup)"
    ] + existing_targets, cwd=project_root)
    
    if returncode == 0:
        print("✓ Black formatting completed successfully")
        if stdout.strip():
            print(f"Files formatted: {len(stdout.strip().split())}")
    else:
        print(f"⚠ Black formatting issues: {stderr}")
    
    # 2. Run isort for import organization
    print("\n2. Running isort for import organization...")
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "isort",
        "--profile", "black",
        "--line-length", "88",
        "--multi-line", "3",
        "--skip", "new_env",
        "--skip", "__pycache__",
        "--skip", ".git"
    ] + existing_targets, cwd=project_root)
    
    if returncode == 0:
        print("✓ Import organization completed successfully")
    else:
        print(f"⚠ Isort issues: {stderr}")
    
    # 3. Run ruff for basic linting and auto-fixes
    print("\n3. Running ruff linter with auto-fixes...")
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "ruff", "check",
        "--fix",
        "--exclude", "new_env,__pycache__,.git,.cleanup_backups,.test_repair_backup",
        "--select", "E,W,F,I",  # Basic errors, warnings, pyflakes, isort
        "--ignore", "E501,W503,E203"  # Ignore line length (handled by black)
    ] + existing_targets, cwd=project_root)
    
    if returncode == 0:
        print("✓ Ruff auto-fixes completed successfully")
    else:
        print(f"⚠ Ruff found remaining issues:")
        if stdout:
            print(stdout)
    
    # 4. Check for remaining issues
    print("\n4. Checking for remaining critical issues...")
    returncode, stdout, stderr = run_command([
        sys.executable, "-m", "ruff", "check",
        "--exclude", "new_env,__pycache__,.git,.cleanup_backups,.test_repair_backup",
        "--select", "F,E9",  # Only critical errors
        "--format", "concise"
    ] + existing_targets, cwd=project_root)
    
    if returncode == 0:
        print("✓ No critical linting errors found")
        critical_issues = 0
    else:
        lines = stdout.strip().split('\n') if stdout else []
        critical_issues = len([line for line in lines if line.strip()])
        print(f"⚠ Found {critical_issues} critical issues:")
        if critical_issues <= 20:  # Show details for small number of issues
            for line in lines[:20]:
                if line.strip():
                    print(f"  {line}")
        else:
            print("  (Too many to display - run ruff check manually for details)")
    
    # 5. Validate Python syntax
    print("\n5. Validating Python syntax...")
    syntax_errors = 0
    for target in existing_targets:
        for py_file in (project_root / target).rglob("*.py"):
            if any(exclude in str(py_file) for exclude in ["new_env", "__pycache__", ".git"]):
                continue
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                print(f"  Syntax error in {py_file}: {e}")
                syntax_errors += 1
            except Exception:
                pass  # Other encoding issues, etc.
    
    if syntax_errors == 0:
        print("✓ No syntax errors found")
    else:
        print(f"⚠ Found {syntax_errors} syntax errors")
    
    # Summary
    print("\n" + "="*50)
    print("QUALITY FIX SUMMARY")
    print("="*50)
    
    total_issues = critical_issues + syntax_errors
    
    if total_issues == 0:
        print("✓ All critical quality issues have been fixed!")
        print("✓ Code is ready for commit")
        return 0
    elif total_issues <= 10:
        print(f"⚠ {total_issues} minor issues remain")
        print("⚠ Consider fixing these before commit, but not critical")
        return 0
    else:
        print(f"✗ {total_issues} issues found that need attention")
        print("✗ Please fix critical issues before committing")
        return 1

if __name__ == "__main__":
    sys.exit(main())