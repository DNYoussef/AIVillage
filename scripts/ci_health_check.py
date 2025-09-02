#!/usr/bin/env python3
"""
CI Health Check Script
Validates CI environment before running tests
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major != 3 or version.minor < 9:
        print(f"[FAIL] Python version {version.major}.{version.minor} not supported. Requires Python 3.9+")
        return False
    print(f"[PASS] Python version {version.major}.{version.minor}.{version.micro} OK")
    return True

def check_required_packages():
    """Check if required packages are available"""
    required_packages = [
        'pytest',
        'pytest_asyncio',
        'pytest_cov',
        'pytest_mock'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"[PASS] {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"[FAIL] {package} missing")
    
    if missing_packages:
        print(f"Install missing packages: pip install {' '.join(missing_packages)}")
        return False
    return True

def check_environment_variables():
    """Check required environment variables"""
    env_vars = {
        'PYTHONPATH': 'Python path configuration',
        'AIVILLAGE_ENV': 'Application environment'
    }
    
    all_ok = True
    for var, description in env_vars.items():
        if var in os.environ:
            value = os.environ[var]
            display_value = value[:50] + '...' if len(value) > 50 else value
            print(f"[PASS] {var}: {display_value}")
        else:
            print(f"[WARN] {var} not set ({description})")
    
    return all_ok

def check_file_structure():
    """Check required file structure"""
    required_paths = [
        'tests',
        'requirements.txt',
        'pyproject.toml'
    ]
    
    all_ok = True
    root_path = Path.cwd()
    
    for path_str in required_paths:
        path = root_path / path_str
        if path.exists():
            print(f"[PASS] {path_str} exists")
        else:
            print(f"[WARN] {path_str} missing")
    
    return all_ok

def main():
    """Run all health checks"""
    print("Running CI Health Checks...")
    print("=" * 50)
    
    checks = [
        check_python_version,
        check_required_packages,
        check_environment_variables,
        check_file_structure
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
            print()
        except Exception as e:
            print(f"[FAIL] Check failed with error: {e}")
            results.append(False)
            print()
    
    critical_checks = results[:2]
    if all(critical_checks):
        print("[PASS] All critical CI health checks passed!")
        return 0
    else:
        print("[FAIL] Some critical CI health checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
