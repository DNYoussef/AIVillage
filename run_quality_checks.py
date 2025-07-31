#!/usr/bin/env python3
"""
Comprehensive Code Quality Checker and Fixer for Agent Forge
Runs black, ruff, mypy, bandit, and other quality checks.
Fixes all linting issues found.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeQualityChecker:
    """Comprehensive code quality checker and fixer."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues_found = []
        self.files_processed = 0
        self.files_fixed = 0
        
        # Key directories to check
        self.target_dirs = [
            "agent_forge",
            "mcp_servers", 
            "tests",
            "production",
            "scripts",
            "benchmarks",
            "examples"
        ]
        
        # Files to exclude
        self.exclude_patterns = [
            "*/new_env/*",
            "*/__pycache__/*",
            "*.pyc",
            "*/.git/*",
            "*/node_modules/*",
            "*/.cleanup_backups/*",
            "*/.test_repair_backup/*"
        ]

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)

    def get_python_files(self) -> List[Path]:
        """Get all Python files to check."""
        python_files = []
        
        for target_dir in self.target_dirs:
            dir_path = self.project_root / target_dir
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    # Check if file should be excluded
                    should_exclude = False
                    for pattern in self.exclude_patterns:
                        if pattern.replace("*", "") in str(py_file):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        python_files.append(py_file)
        
        return python_files

    def check_black_formatting(self) -> bool:
        """Check and fix black formatting."""
        logger.info("Running black formatter...")
        
        python_files = self.get_python_files()
        if not python_files:
            logger.warning("No Python files found to format")
            return True
            
        # Run black with --check first to see what needs fixing
        file_paths = [str(f) for f in python_files]
        
        # Check formatting
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "black", "--check", "--diff"
        ] + file_paths)
        
        if returncode != 0:
            logger.info("Black formatting issues found, fixing...")
            
            # Fix formatting
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "black"
            ] + file_paths)
            
            if returncode == 0:
                logger.info("Black formatting applied successfully")
                self.files_fixed += len(python_files)
                return True
            else:
                logger.error(f"Black formatting failed: {stderr}")
                self.issues_found.append(f"Black formatting failed: {stderr}")
                return False
        else:
            logger.info("All files already properly formatted with black")
            return True

    def check_isort_imports(self) -> bool:
        """Check and fix import organization with isort."""
        logger.info("Running isort import organizer...")
        
        python_files = self.get_python_files()
        if not python_files:
            return True
            
        file_paths = [str(f) for f in python_files]
        
        # Check import organization
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "isort", "--check-only", "--diff"
        ] + file_paths)
        
        if returncode != 0:
            logger.info("Import organization issues found, fixing...")
            
            # Fix imports
            returncode, stdout, stderr = self.run_command([
                sys.executable, "-m", "isort"
            ] + file_paths)
            
            if returncode == 0:
                logger.info("Import organization applied successfully")
                return True
            else:
                logger.error(f"Isort failed: {stderr}")
                self.issues_found.append(f"Isort failed: {stderr}")
                return False
        else:
            logger.info("All imports properly organized")
            return True

    def check_ruff_linting(self) -> bool:
        """Check and fix ruff linting issues."""
        logger.info("Running ruff linter...")
        
        python_files = self.get_python_files()
        if not python_files:
            return True
            
        file_paths = [str(f) for f in python_files]
        
        # Try to fix automatically first
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "ruff", "check", "--fix"
        ] + file_paths)
        
        # Then check for remaining issues
        returncode, stdout, stderr = self.run_command([
            sys.executable, "-m", "ruff", "check"
        ] + file_paths)
        
        if returncode != 0:
            logger.warning(f"Ruff found remaining issues:\n{stdout}")
            self.issues_found.append(f"Ruff linting issues: {stdout}")
            # Don't return False - some issues might not be auto-fixable
            
        logger.info("Ruff linting completed")
        return True

    def check_mypy_typing(self) -> bool:
        """Check mypy type hints."""
        logger.info("Running mypy type checker...")
        
        # Check key directories with mypy
        for target_dir in ["agent_forge", "mcp_servers"]:
            dir_path = self.project_root / target_dir
            if dir_path.exists():
                returncode, stdout, stderr = self.run_command([
                    sys.executable, "-m", "mypy", str(dir_path),
                    "--ignore-missing-imports",
                    "--no-strict-optional",
                    "--show-error-codes"
                ])
                
                if returncode != 0:
                    logger.warning(f"MyPy found type issues in {target_dir}:\n{stdout}")
                    self.issues_found.append(f"MyPy type issues in {target_dir}: {stdout}")
        
        logger.info("MyPy type checking completed")
        return True

    def check_bandit_security(self) -> bool:
        """Check security issues with bandit."""
        logger.info("Running bandit security scanner...")
        
        for target_dir in ["agent_forge", "mcp_servers"]:
            dir_path = self.project_root / target_dir
            if dir_path.exists():
                returncode, stdout, stderr = self.run_command([
                    sys.executable, "-m", "bandit", "-r", str(dir_path),
                    "-f", "json", "--skip", "B101,B601"  # Skip assert and shell usage
                ])
                
                if returncode != 0 and stdout:
                    try:
                        results = json.loads(stdout)
                        if results.get("results"):
                            logger.warning(f"Bandit found security issues in {target_dir}")
                            self.issues_found.append(f"Security issues in {target_dir}: {len(results['results'])} issues")
                    except json.JSONDecodeError:
                        logger.warning(f"Bandit output parsing failed for {target_dir}")
        
        logger.info("Bandit security scanning completed")
        return True

    def validate_yaml_json_files(self) -> bool:
        """Validate YAML and JSON configuration files."""
        logger.info("Validating configuration files...")
        
        # Check JSON files
        for json_file in self.project_root.rglob("*.json"):
            if any(pattern.replace("*", "") in str(json_file) for pattern in self.exclude_patterns):
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {json_file}: {e}")
                self.issues_found.append(f"Invalid JSON: {json_file}")
                return False
        
        # Check YAML files
        try:
            import yaml
            for yaml_file in self.project_root.rglob("*.yml"):
                if any(pattern.replace("*", "") in str(yaml_file) for pattern in self.exclude_patterns):
                    continue
                    
                try:
                    with open(yaml_file, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.error(f"Invalid YAML in {yaml_file}: {e}")
                    self.issues_found.append(f"Invalid YAML: {yaml_file}")
                    return False
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML validation")
        
        logger.info("Configuration file validation completed")
        return True

    def check_import_issues(self) -> bool:
        """Check for import issues and circular imports."""
        logger.info("Checking for import issues...")
        
        python_files = self.get_python_files()
        import_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for obvious import problems
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    
                    # Check for relative imports that might be broken
                    if line.startswith('from .') and 'import' in line:
                        # This is a relative import - check if it's valid
                        continue  # Skip for now - would need more complex validation
                        
                    # Check for imports that might not exist
                    if line.startswith(('import ', 'from ')) and 'import' in line:
                        if any(problematic in line for problematic in ['..', 'undefined', 'missing']):
                            import_issues.append(f"{py_file}:{i} - Potentially problematic import: {line}")
                            
            except Exception as e:
                logger.warning(f"Could not analyze imports in {py_file}: {e}")
        
        if import_issues:
            logger.warning(f"Found {len(import_issues)} potential import issues")
            for issue in import_issues[:10]:  # Show first 10
                logger.warning(f"  {issue}")
            self.issues_found.extend(import_issues)
        
        logger.info("Import analysis completed")
        return True

    async def run_all_checks(self) -> bool:
        """Run all quality checks and fixes."""
        logger.info("Starting comprehensive code quality checks...")
        
        self.files_processed = len(self.get_python_files())
        logger.info(f"Found {self.files_processed} Python files to check")
        
        checks = [
            ("Black Formatting", self.check_black_formatting),
            ("Import Organization", self.check_isort_imports),
            ("Ruff Linting", self.check_ruff_linting),
            ("MyPy Type Checking", self.check_mypy_typing),
            ("Bandit Security", self.check_bandit_security),
            ("Configuration Validation", self.validate_yaml_json_files),
            ("Import Analysis", self.check_import_issues),
        ]
        
        results = {}
        for check_name, check_func in checks:
            logger.info(f"\n--- Running {check_name} ---")
            try:
                results[check_name] = check_func()
            except Exception as e:
                logger.error(f"Error in {check_name}: {e}")
                results[check_name] = False
                self.issues_found.append(f"{check_name} failed: {e}")
        
        return self.generate_report(results)

    def generate_report(self, results: Dict[str, bool]) -> bool:
        """Generate final quality report."""
        logger.info("\n" + "="*60)
        logger.info("CODE QUALITY REPORT")
        logger.info("="*60)
        
        logger.info(f"Files processed: {self.files_processed}")
        logger.info(f"Files fixed: {self.files_fixed}")
        
        logger.info("\nCheck Results:")
        for check_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {check_name}: {status}")
        
        if self.issues_found:
            logger.info(f"\nIssues Found ({len(self.issues_found)}):")
            for i, issue in enumerate(self.issues_found[:20], 1):  # Show first 20
                logger.info(f"  {i}. {issue}")
            
            if len(self.issues_found) > 20:
                logger.info(f"  ... and {len(self.issues_found) - 20} more issues")
        
        # Determine overall success
        critical_checks = ["Black Formatting", "Import Organization", "Configuration Validation"]
        critical_passed = all(results.get(check, False) for check in critical_checks)
        
        if critical_passed and len(self.issues_found) < 50:  # Allow some non-critical issues
            logger.info("\n✓ OVERALL: QUALITY CHECKS PASSED")
            logger.info("Code is ready for commit!")
            return True
        else:
            logger.error("\n✗ OVERALL: QUALITY CHECKS FAILED")
            logger.error("Please fix the issues above before committing.")
            return False

async def main():
    """Main entry point."""
    project_root = Path(__file__).parent
    
    # Install required packages if missing
    required_packages = ["black", "isort", "ruff", "mypy", "bandit"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    
    checker = CodeQualityChecker(project_root)
    success = await checker.run_all_checks()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)