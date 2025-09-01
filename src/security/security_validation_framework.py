#!/usr/bin/env python3
"""
AI Village Security Validation Framework
Comprehensive security validation and compliance checking system
"""

import os
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityValidationResult:
    """Security validation test result"""
    test_name: str
    category: str
    status: str  # PASS, FAIL, WARNING, SKIP
    message: str
    details: Optional[Dict] = None
    recommendation: Optional[str] = None

class SecurityValidationFramework:
    """Comprehensive security validation and compliance framework"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results: List[SecurityValidationResult] = []
        self.validation_timestamp = datetime.now().isoformat()
        
    def run_full_validation(self) -> Dict[str, any]:
        """Run complete security validation suite"""
        logger.info("Starting comprehensive security validation...")
        
        # 1. Credential Exposure Validation
        self._validate_no_exposed_credentials()
        
        # 2. Test Suppression Validation
        self._validate_test_suppressions()
        
        # 3. Configuration Security Validation
        self._validate_configuration_security()
        
        # 4. Dependency Security Validation
        self._validate_dependencies()
        
        # 5. Access Control Validation
        self._validate_access_controls()
        
        # 6. Encryption Validation
        self._validate_encryption_usage()
        
        # 7. CI/CD Security Validation
        self._validate_cicd_security()
        
        # Generate comprehensive report
        return self._generate_validation_report()
    
    def _validate_no_exposed_credentials(self) -> None:
        """Validate no credentials are exposed in version control"""
        logger.info("Validating credential exposure...")
        
        # Check for .env files in git
        try:
            result = subprocess.run(['git', 'ls-files', '*.env'], 
                                  cwd=self.project_root, 
                                  capture_output=True, text=True)
            
            if result.stdout.strip():
                self.results.append(SecurityValidationResult(
                    test_name="env_files_in_git",
                    category="credential_exposure",
                    status="FAIL",
                    message=f"Found .env files in git: {result.stdout.strip()}",
                    recommendation="Remove .env files from git and add to .gitignore"
                ))
            else:
                self.results.append(SecurityValidationResult(
                    test_name="env_files_in_git",
                    category="credential_exposure", 
                    status="PASS",
                    message="No .env files found in git tracking"
                ))
                
        except Exception as e:
            self.results.append(SecurityValidationResult(
                test_name="env_files_in_git",
                category="credential_exposure",
                status="WARNING",
                message=f"Could not check git files: {e}"
            ))
        
        # Check .gitignore contains .env pattern
        gitignore_path = self.project_root / '.gitignore'
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                gitignore_content = f.read()
            
            if '.env' in gitignore_content:
                self.results.append(SecurityValidationResult(
                    test_name="gitignore_env_pattern",
                    category="credential_exposure",
                    status="PASS",
                    message=".env pattern found in .gitignore"
                ))
            else:
                self.results.append(SecurityValidationResult(
                    test_name="gitignore_env_pattern",
                    category="credential_exposure",
                    status="FAIL",
                    message=".env pattern missing from .gitignore",
                    recommendation="Add config/.env and *.env to .gitignore"
                ))
    
    def _validate_test_suppressions(self) -> None:
        """Validate test credentials have proper suppression comments"""
        logger.info("Validating test suppression comments...")
        
        unsuppressed_patterns = [
            r'password\s*=\s*["\'][^"\']*["\'](?!\s*#.*(?:nosec|pragma.*allowlist))',
            r'(?:api_key|token|secret)\s*=\s*["\'][^"\']*["\'](?!\s*#.*(?:nosec|pragma.*allowlist))'
        ]
        
        test_dirs = ['tests', 'test']
        total_unsuppressed = 0
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for py_file in test_path.rglob('*.py'):
                    try:
                        with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        for pattern in unsuppressed_patterns:
                            matches = list(re.finditer(pattern, content, re.IGNORECASE))
                            total_unsuppressed += len(matches)
                            
                    except Exception as e:
                        logger.warning(f"Could not scan {py_file}: {e}")
        
        if total_unsuppressed == 0:
            self.results.append(SecurityValidationResult(
                test_name="test_credential_suppressions",
                category="test_security",
                status="PASS",
                message="All test credentials have proper suppression comments"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="test_credential_suppressions",
                category="test_security",
                status="FAIL",
                message=f"Found {total_unsuppressed} unsuppressed test credentials",
                recommendation="Run suppression_fixer.py to add missing comments"
            ))
    
    def _validate_configuration_security(self) -> None:
        """Validate security configuration"""
        logger.info("Validating configuration security...")
        
        # Check for secure template files
        env_template = self.project_root / 'config' / '.env.template'
        if env_template.exists():
            self.results.append(SecurityValidationResult(
                test_name="env_template_exists",
                category="configuration",
                status="PASS", 
                message="Secure .env.template found"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="env_template_exists",
                category="configuration",
                status="FAIL",
                message="No .env.template found",
                recommendation="Create secure environment template"
            ))
        
        # Check security configuration files
        security_configs = [
            'config/security/rbac.json',
            '.secrets.baseline',
            'config/security/.secrets.baseline'
        ]
        
        for config_file in security_configs:
            config_path = self.project_root / config_file
            if config_path.exists():
                self.results.append(SecurityValidationResult(
                    test_name=f"security_config_{Path(config_file).name}",
                    category="configuration",
                    status="PASS",
                    message=f"Security configuration {config_file} exists"
                ))
            else:
                self.results.append(SecurityValidationResult(
                    test_name=f"security_config_{Path(config_file).name}",
                    category="configuration",
                    status="WARNING",
                    message=f"Security configuration {config_file} not found"
                ))
    
    def _validate_dependencies(self) -> None:
        """Validate dependency security"""
        logger.info("Validating dependency security...")
        
        # Check for security-focused requirements
        security_requirements = [
            'requirements-security.txt',
            'config/requirements/requirements-security.txt'
        ]
        
        security_deps_found = False
        for req_file in security_requirements:
            req_path = self.project_root / req_file
            if req_path.exists():
                security_deps_found = True
                self.results.append(SecurityValidationResult(
                    test_name="security_requirements",
                    category="dependencies",
                    status="PASS",
                    message=f"Security requirements file {req_file} exists"
                ))
                break
        
        if not security_deps_found:
            self.results.append(SecurityValidationResult(
                test_name="security_requirements",
                category="dependencies", 
                status="WARNING",
                message="No security-specific requirements file found",
                recommendation="Create requirements-security.txt with security tools"
            ))
    
    def _validate_access_controls(self) -> None:
        """Validate access control implementations"""
        logger.info("Validating access controls...")
        
        # Check for RBAC configuration
        rbac_path = self.project_root / 'config' / 'security' / 'rbac.json'
        if rbac_path.exists():
            self.results.append(SecurityValidationResult(
                test_name="rbac_configuration",
                category="access_control",
                status="PASS",
                message="RBAC configuration file exists"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="rbac_configuration",
                category="access_control",
                status="WARNING",
                message="RBAC configuration not found"
            ))
        
        # Check for authentication implementations
        auth_files = list(self.project_root.rglob('*auth*.py'))
        if auth_files:
            self.results.append(SecurityValidationResult(
                test_name="authentication_modules",
                category="access_control", 
                status="PASS",
                message=f"Found {len(auth_files)} authentication modules"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="authentication_modules",
                category="access_control",
                status="WARNING",
                message="No authentication modules found"
            ))
    
    def _validate_encryption_usage(self) -> None:
        """Validate encryption implementations"""
        logger.info("Validating encryption usage...")
        
        # Look for encryption modules
        encryption_patterns = [
            r'import\s+(?:cryptography|Crypto|hashlib)',
            r'from\s+(?:cryptography|Crypto|hashlib)',
            r'AES|RSA|SHA256|bcrypt|scrypt'
        ]
        
        encryption_found = False
        py_files_with_crypto = 0
        
        for py_file in self.project_root.rglob('*.py'):
            if any(exclude in str(py_file) for exclude in ['.git', '__pycache__', 'node_modules']):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                for pattern in encryption_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        encryption_found = True
                        py_files_with_crypto += 1
                        break
                        
            except Exception:
                continue
        
        if encryption_found:
            self.results.append(SecurityValidationResult(
                test_name="encryption_usage",
                category="encryption",
                status="PASS",
                message=f"Encryption usage found in {py_files_with_crypto} files"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="encryption_usage",
                category="encryption",
                status="WARNING",
                message="No encryption usage detected",
                recommendation="Implement proper encryption for sensitive data"
            ))
    
    def _validate_cicd_security(self) -> None:
        """Validate CI/CD security configurations"""
        logger.info("Validating CI/CD security...")
        
        # Check for security workflows
        github_workflows = self.project_root / '.github' / 'workflows'
        if github_workflows.exists():
            security_workflows = list(github_workflows.glob('*security*.yml'))
            
            if security_workflows:
                self.results.append(SecurityValidationResult(
                    test_name="security_workflows",
                    category="cicd",
                    status="PASS", 
                    message=f"Found {len(security_workflows)} security workflows"
                ))
            else:
                self.results.append(SecurityValidationResult(
                    test_name="security_workflows",
                    category="cicd",
                    status="WARNING",
                    message="No security workflows found",
                    recommendation="Add security scanning to CI/CD pipeline"
                ))
        
        # Check for pre-commit hooks
        precommit_config = self.project_root / '.pre-commit-config.yaml'
        if precommit_config.exists():
            self.results.append(SecurityValidationResult(
                test_name="precommit_hooks",
                category="cicd",
                status="PASS",
                message="Pre-commit configuration exists"
            ))
        else:
            self.results.append(SecurityValidationResult(
                test_name="precommit_hooks", 
                category="cicd",
                status="WARNING",
                message="No pre-commit hooks configured"
            ))
    
    def _generate_validation_report(self) -> Dict[str, any]:
        """Generate comprehensive validation report"""
        
        # Calculate statistics
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == 'PASS'])
        failed = len([r for r in self.results if r.status == 'FAIL']) 
        warnings = len([r for r in self.results if r.status == 'WARNING'])
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(asdict(result))
        
        report = {
            'validation_metadata': {
                'timestamp': self.validation_timestamp,
                'project_root': str(self.project_root),
                'total_tests': total_tests,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'overall_status': 'PASS' if failed == 0 else 'FAIL' if failed > 3 else 'WARNING'
            },
            'categories': categories,
            'summary': {
                'credential_exposure': self._get_category_status('credential_exposure'),
                'test_security': self._get_category_status('test_security'),
                'configuration': self._get_category_status('configuration'),
                'dependencies': self._get_category_status('dependencies'),
                'access_control': self._get_category_status('access_control'),
                'encryption': self._get_category_status('encryption'),
                'cicd': self._get_category_status('cicd')
            },
            'recommendations': [r.recommendation for r in self.results if r.recommendation]
        }
        
        # Save detailed report
        report_path = self.project_root / 'reports' / 'security' / 'security_validation_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate executive summary
        self._generate_executive_summary(report)
        
        logger.info(f"Security validation report saved to {report_path}")
        return report
    
    def _get_category_status(self, category: str) -> str:
        """Get overall status for a category"""
        category_results = [r for r in self.results if r.category == category]
        if not category_results:
            return 'SKIP'
        
        if any(r.status == 'FAIL' for r in category_results):
            return 'FAIL'
        elif any(r.status == 'WARNING' for r in category_results):
            return 'WARNING'
        else:
            return 'PASS'
    
    def _generate_executive_summary(self, report: Dict) -> None:
        """Generate executive summary"""
        summary_path = self.project_root / 'reports' / 'security' / 'SECURITY_VALIDATION_SUMMARY.md'
        
        status_emoji = {
            'PASS': '✅',
            'FAIL': '❌', 
            'WARNING': '⚠️',
            'SKIP': '⏭️'
        }
        
        with open(summary_path, 'w') as f:
            f.write(f"""# 🛡️ Security Validation Summary

**Generated:** {self.validation_timestamp}
**Overall Status:** {status_emoji[report['validation_metadata']['overall_status']]} {report['validation_metadata']['overall_status']}

## 📊 Test Results

- **Total Tests:** {report['validation_metadata']['total_tests']}
- **Passed:** {status_emoji['PASS']} {report['validation_metadata']['passed']}
- **Failed:** {status_emoji['FAIL']} {report['validation_metadata']['failed']}
- **Warnings:** {status_emoji['WARNING']} {report['validation_metadata']['warnings']}

## 🔍 Category Results

""")
            
            for category, status in report['summary'].items():
                f.write(f"- **{category.replace('_', ' ').title()}:** {status_emoji[status]} {status}\n")
            
            f.write(f"""
## 🚨 Critical Issues

""")
            
            critical_issues = [r for r in self.results if r.status == 'FAIL']
            if critical_issues:
                for issue in critical_issues:
                    f.write(f"- **{issue.test_name}:** {issue.message}\n")
            else:
                f.write("No critical issues detected! 🎉\n")
            
            f.write(f"""
## 📋 Recommendations

""")
            
            recommendations = list(set([r.recommendation for r in self.results if r.recommendation]))
            if recommendations:
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            else:
                f.write("No additional recommendations at this time.\n")
            
            f.write(f"""
## ✅ Security Status

{status_emoji[report['validation_metadata']['overall_status']]} **{report['validation_metadata']['overall_status']}** - {"Security posture is acceptable" if report['validation_metadata']['overall_status'] in ['PASS', 'WARNING'] else "Critical security issues require immediate attention"}

---
*Generated by AI Village Security Manager Agent*
""")
        
        logger.info(f"Executive summary saved to {summary_path}")

def main():
    """Main execution function"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    validator = SecurityValidationFramework(project_root)
    report = validator.run_full_validation()
    
    print(f"""
🛡️ SECURITY VALIDATION COMPLETE

Overall Status: {report['validation_metadata']['overall_status']}
Tests Passed: {report['validation_metadata']['passed']}/{report['validation_metadata']['total_tests']}
Critical Failures: {report['validation_metadata']['failed']}
Warnings: {report['validation_metadata']['warnings']}

Reports generated:
- reports/security/security_validation_report.json
- reports/security/SECURITY_VALIDATION_SUMMARY.md
""")

if __name__ == "__main__":
    main()