#!/usr/bin/env python3
"""
Immediate Security Fixes for Production Deployment

This script applies the critical security fixes identified in the security pre-flight monitoring:
1. Adds missing pragma allowlist comments to test passwords
2. Validates the fixes were applied correctly
3. Tests security validation after fixes

Usage:
    python scripts/security/apply_security_fixes.py [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityFixApplier:
    """Applies critical security fixes for production readiness."""
    
    def __init__(self, base_path: str, dry_run: bool = False):
        self.base_path = Path(base_path)
        self.dry_run = dry_run
        self.fixes_applied = []
        
        # Critical fixes needed based on monitoring report
        self.critical_fixes = [
            {
                "file": "tests/security/test_auth_system.py",
                "line": 364,
                "pattern": r'username="resetuser", password="test_reset_password_123!" # nosec B106 - test password',
                "replacement": r'username="resetuser", password="test_reset_password_123!" # nosec B106 - test password  # pragma: allowlist secret'
            },
            {
                "file": "tests/security/test_auth_system.py", 
                "line": 396,
                "pattern": r'username="mfadisable", email="mfadisable@example.com", password="test_auth_password_123!"',
                "replacement": r'username="mfadisable", email="mfadisable@example.com", password="test_auth_password_123!"  # pragma: allowlist secret'
            },
            {
                "file": "tests/security/test_auth_system.py",
                "line": 498, 
                "pattern": r'username="nonexistent_user_12345", password="any_password"',
                "replacement": r'username="nonexistent_user_12345", password="any_password"  # pragma: allowlist secret'
            },
            {
                "file": "tests/security/test_auth_system.py",
                "line": 504,
                "pattern": r'username="realuser", email="real@example.com", password="test_real_password_123!" # nosec B106 - test password',
                "replacement": r'username="realuser", email="real@example.com", password="test_real_password_123!" # nosec B106 - test password  # pragma: allowlist secret'
            },
            {
                "file": "tests/security/test_auth_system.py",
                "line": 509,
                "pattern": r'username="realuser", password="wrong_password"',
                "replacement": r'username="realuser", password="wrong_password"  # pragma: allowlist secret'
            }
        ]

    def apply_fix(self, fix_info: dict) -> bool:
        """Apply a single security fix."""
        file_path = self.base_path / fix_info["file"]
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
            
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Check if the line exists and matches
            line_idx = fix_info["line"] - 1  # Convert to 0-based index
            if line_idx >= len(lines):
                logger.error(f"Line {fix_info['line']} doesn't exist in {fix_info['file']}")
                return False
                
            original_line = lines[line_idx].rstrip()
            
            # Check if fix is already applied
            if "pragma: allowlist secret" in original_line:
                logger.info(f"Fix already applied at {fix_info['file']}:{fix_info['line']}")
                return True
                
            # Apply the fix using regex replacement
            pattern = fix_info["pattern"]
            replacement = fix_info["replacement"]
            
            if re.search(pattern, original_line):
                new_line = re.sub(pattern, replacement, original_line)
                
                if self.dry_run:
                    logger.info(f"[DRY-RUN] Would fix {fix_info['file']}:{fix_info['line']}")
                    logger.info(f"  FROM: {original_line}")
                    logger.info(f"  TO:   {new_line}")
                else:
                    lines[line_idx] = new_line + '\n'
                    
                    # Write back the file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                        
                    logger.info(f"✅ Fixed {fix_info['file']}:{fix_info['line']}")
                    
                self.fixes_applied.append(fix_info)
                return True
            else:
                logger.warning(f"Pattern not found at {fix_info['file']}:{fix_info['line']}")
                logger.warning(f"Expected pattern: {pattern}")
                logger.warning(f"Actual line: {original_line}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying fix to {fix_info['file']}: {e}")
            return False

    def apply_all_fixes(self) -> bool:
        """Apply all critical security fixes."""
        logger.info("Applying critical security fixes for production deployment...")
        
        success_count = 0
        for fix_info in self.critical_fixes:
            if self.apply_fix(fix_info):
                success_count += 1
                
        logger.info(f"Applied {success_count}/{len(self.critical_fixes)} fixes successfully")
        
        if success_count == len(self.critical_fixes):
            logger.info("✅ All critical security fixes applied successfully!")
            return True
        else:
            logger.error(f"❌ {len(self.critical_fixes) - success_count} fixes failed")
            return False

    def validate_fixes(self) -> bool:
        """Validate that fixes were applied correctly."""
        logger.info("Validating security fixes...")
        
        for fix_info in self.fixes_applied:
            file_path = self.base_path / fix_info["file"]
            line_num = fix_info["line"]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                if line_num <= len(lines):
                    line_content = lines[line_num - 1].rstrip()
                    if "pragma: allowlist secret" not in line_content:
                        logger.error(f"❌ Fix validation failed at {fix_info['file']}:{line_num}")
                        return False
                    else:
                        logger.info(f"✅ Fix validated at {fix_info['file']}:{line_num}")
                        
            except Exception as e:
                logger.error(f"Error validating {fix_info['file']}: {e}")
                return False
                
        logger.info("✅ All fixes validated successfully!")
        return True

    def run_security_validation(self) -> bool:
        """Run the security validation script to test the fixes."""
        logger.info("Running security validation to test fixes...")
        
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, "scripts/validate_secret_sanitization.py", "--production-ready"],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            logger.info(f"Security validation exit code: {result.returncode}")
            
            if result.returncode == 0:
                logger.info("✅ Security validation PASSED after fixes!")
                return True
            elif result.returncode == 1:
                logger.warning("⚠️ Security validation PASSED WITH WARNINGS")
                return True
            else:
                logger.error("❌ Security validation still FAILING after fixes")
                logger.error("STDOUT:", result.stdout)
                logger.error("STDERR:", result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Security validation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running security validation: {e}")
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Apply critical security fixes for production deployment")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without applying changes")
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent.parent
    
    logger.info("🔒 SCION Security Fix Application Tool")
    logger.info("=" * 50)
    
    if args.dry_run:
        logger.info("Running in DRY-RUN mode - no changes will be made")
    
    # Apply the fixes
    fixer = SecurityFixApplier(base_path, dry_run=args.dry_run)
    
    if not fixer.apply_all_fixes():
        logger.error("❌ Failed to apply all security fixes")
        return 1
        
    if not args.dry_run:
        # Validate the fixes
        if not fixer.validate_fixes():
            logger.error("❌ Fix validation failed")
            return 1
            
        # Run security validation
        if not fixer.run_security_validation():
            logger.error("❌ Security validation still failing after fixes")
            return 1
            
        logger.info("🎉 All security fixes applied and validated successfully!")
        logger.info("🚀 Ready for production deployment!")
    else:
        logger.info("🔍 Dry-run complete - use without --dry-run to apply fixes")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())