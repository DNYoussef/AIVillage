#!/usr/bin/env python3
"""
Emergency Security Audit and Remediation System
Security Manager Agent - AIVillage Emergency Response
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityFinding:
    """Security vulnerability finding"""
    file_path: str
    line_number: int
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    type: str      # exposed_credential, unsuppressed_test_password, etc.
    description: str
    value_pattern: str
    recommendation: str
    status: str = "DETECTED"  # DETECTED, SUPPRESSED, FIXED, IGNORED

@dataclass
class CredentialExposure:
    """Exposed credential information"""
    credential_type: str
    value_hash: str  # SHA-256 hash for tracking without storing actual value
    file_path: str
    line_number: int
    exposure_severity: str
    rotation_required: bool = True

class EmergencySecurityAuditor:
    """Comprehensive security audit and remediation system"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.credential_exposures: List[CredentialExposure] = []
        self.scan_timestamp = datetime.now().isoformat()
        
        # Dangerous patterns that should never be in production
        self.critical_patterns = {
            'api_keys': r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\'][a-z0-9-_]{20,}["\']',
            'passwords': r'(?i)(password|passwd|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']',
            'secrets': r'(?i)(secret|token)\s*[=:]\s*["\'][a-z0-9-_]{16,}["\']',
            'redis_password': r'REDIS_PASSWORD\s*=\s*["\']([^"\']+)["\']',
            'openrouter_key': r'OPENROUTER_API_KEY\s*=\s*["\']([^"\']+)["\']'
        }
        
        # Test patterns that need suppression comments
        self.test_patterns = {
            'test_passwords': r'password\s*=\s*["\']test[^"\']*["\'](?!.*(?:nosec|pragma.*allowlist))',
            'test_credentials': r'(?:api_key|token|secret)\s*=\s*["\']test[^"\']*["\'](?!.*(?:nosec|pragma.*allowlist))',
        }
        
    def scan_codebase(self) -> None:
        """Comprehensive codebase security scan"""
        logger.info(f"Starting emergency security scan of {self.project_root}")
        
        # Scan for critical exposures first
        self._scan_critical_exposures()
        
        # Scan for unsuppressed test credentials
        self._scan_test_credentials()
        
        # Scan for hardcoded secrets in configuration files
        self._scan_configuration_files()
        
        # Generate comprehensive report
        self._generate_security_report()
        
        logger.info(f"Security scan completed. Found {len(self.findings)} findings.")
    
    def _scan_critical_exposures(self) -> None:
        """Scan for critical credential exposures"""
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.swarm', 'models'}
        
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file() and not any(exc in str(file_path) for exc in exclude_dirs):
                if file_path.suffix in ['.py', '.js', '.ts', '.env', '.yaml', '.yml', '.json', '.md']:
                    self._scan_file_for_critical_patterns(file_path)
    
    def _scan_file_for_critical_patterns(self, file_path: Path) -> None:
        """Scan individual file for critical security patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for pattern_name, pattern in self.critical_patterns.items():
                    matches = list(re.finditer(pattern, content, re.MULTILINE))
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        line_content = lines[line_number - 1] if line_number <= len(lines) else ""
                        
                        # Skip if already suppressed
                        if any(supp in line_content.lower() for supp in ['nosec', 'pragma: allowlist', 'allowlist secret']):
                            continue
                            
                        severity = "CRITICAL" if pattern_name in ['api_keys', 'openrouter_key', 'redis_password'] else "HIGH"
                        
                        finding = SecurityFinding(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_number,
                            severity=severity,
                            type=f"exposed_{pattern_name}",
                            description=f"Exposed {pattern_name.replace('_', ' ')} in {file_path.name}",
                            value_pattern=match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
                            recommendation=f"Move to environment variables and add to .gitignore"
                        )
                        
                        self.findings.append(finding)
                        
                        # Track credential exposures
                        if len(match.groups()) > 0:
                            credential_value = match.group(1) if match.groups() else match.group(0)
                            exposure = CredentialExposure(
                                credential_type=pattern_name,
                                value_hash=hashlib.sha256(credential_value.encode()).hexdigest(),
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_number,
                                exposure_severity=severity
                            )
                            self.credential_exposures.append(exposure)
                            
        except Exception as e:
            logger.warning(f"Could not scan {file_path}: {e}")
    
    def _scan_test_credentials(self) -> None:
        """Scan for test credentials missing suppression comments"""
        test_dirs = {'tests', 'test', 'spec'}
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for file_path in test_path.rglob('*.py'):
                    self._scan_file_for_test_patterns(file_path)
    
    def _scan_file_for_test_patterns(self, file_path: Path) -> None:
        """Scan test files for unsuppressed credentials"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for pattern_name, pattern in self.test_patterns.items():
                    matches = list(re.finditer(pattern, content, re.MULTILINE))
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\n') + 1
                        
                        finding = SecurityFinding(
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_number,
                            severity="HIGH",
                            type="unsuppressed_test_credential",
                            description=f"Test credential without suppression comment",
                            value_pattern=match.group(0)[:50],
                            recommendation="Add # nosec B106 - test credential or # pragma: allowlist secret"
                        )
                        
                        self.findings.append(finding)
                        
        except Exception as e:
            logger.warning(f"Could not scan test file {file_path}: {e}")
    
    def _scan_configuration_files(self) -> None:
        """Scan configuration files for security issues"""
        config_patterns = ['*.env*', '*.conf', '*.ini', 'config.*']
        
        for pattern in config_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    self._scan_file_for_critical_patterns(file_path)
    
    def generate_suppression_fixes(self) -> Dict[str, List[str]]:
        """Generate automatic fixes for suppression comments"""
        fixes = {}
        
        for finding in self.findings:
            if finding.type == "unsuppressed_test_credential":
                file_path = finding.file_path
                if file_path not in fixes:
                    fixes[file_path] = []
                
                fix_instruction = {
                    'line': finding.line_number,
                    'original': finding.value_pattern,
                    'fix': f"{finding.value_pattern}  # nosec B106 - test credential",
                    'type': 'add_suppression'
                }
                fixes[file_path].append(fix_instruction)
        
        return fixes
    
    def _generate_security_report(self) -> None:
        """Generate comprehensive security report"""
        report = {
            'scan_metadata': {
                'scan_timestamp': self.scan_timestamp,
                'project_root': str(self.project_root),
                'total_findings': len(self.findings),
                'critical_findings': len([f for f in self.findings if f.severity == 'CRITICAL']),
                'high_findings': len([f for f in self.findings if f.severity == 'HIGH']),
            },
            'critical_exposures': [asdict(exp) for exp in self.credential_exposures],
            'security_findings': [asdict(finding) for finding in self.findings],
            'suppression_fixes': self.generate_suppression_fixes(),
            'immediate_actions_required': self._get_immediate_actions()
        }
        
        # Save detailed report
        report_path = self.project_root / 'reports' / 'security' / 'emergency_audit_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Security report saved to {report_path}")
        
        # Generate executive summary
        self._generate_executive_summary(report)
    
    def _get_immediate_actions(self) -> List[str]:
        """Get list of immediate actions required"""
        actions = []
        
        # Check for critical exposures
        critical_findings = [f for f in self.findings if f.severity == 'CRITICAL']
        if critical_findings:
            actions.append("🚨 IMMEDIATE: Rotate all exposed API keys and passwords")
            actions.append("🚨 IMMEDIATE: Remove all .env files from version control")
            actions.append("🚨 IMMEDIATE: Add .env* to .gitignore")
        
        # Check for high severity issues
        high_findings = [f for f in self.findings if f.severity == 'HIGH']
        if high_findings:
            actions.append("⚠️  HIGH PRIORITY: Add suppression comments to test credentials")
            actions.append("⚠️  HIGH PRIORITY: Implement automated secret scanning")
        
        return actions
    
    def _generate_executive_summary(self, report: Dict) -> None:
        """Generate executive summary report"""
        summary_path = self.project_root / 'reports' / 'security' / 'EMERGENCY_SECURITY_SUMMARY.md'
        
        with open(summary_path, 'w') as f:
            f.write(f"""# 🚨 EMERGENCY Security Audit Summary
            
**Generated:** {self.scan_timestamp}
**Security Manager:** AIVillage Emergency Response Agent

## 🔥 Critical Status

### Immediate Threat Level: {"🚨 CRITICAL" if report['scan_metadata']['critical_findings'] > 0 else "⚠️  HIGH"}

- **Total Findings:** {report['scan_metadata']['total_findings']}
- **Critical Exposures:** {report['scan_metadata']['critical_findings']}
- **High Priority Issues:** {report['scan_metadata']['high_findings']}

### 🚨 Critical Actions Required

""")
            
            for action in report['immediate_actions_required']:
                f.write(f"- {action}\n")
            
            f.write(f"""
### 📊 Exposure Breakdown

**Exposed Credentials:** {len(report['critical_exposures'])}
""")
            
            # Group exposures by type
            exposure_types = {}
            for exp in report['critical_exposures']:
                exp_type = exp['credential_type']
                exposure_types[exp_type] = exposure_types.get(exp_type, 0) + 1
            
            for exp_type, count in exposure_types.items():
                f.write(f"- {exp_type.replace('_', ' ').title()}: {count}\n")
            
            f.write(f"""
### 🛠 Remediation Status

Files requiring suppression fixes: {len(report['suppression_fixes'])}

### 📋 Next Steps

1. **IMMEDIATE**: Rotate exposed credentials
2. **HIGH**: Apply suppression fixes to test files  
3. **MEDIUM**: Deploy automated security scanning
4. **LOW**: Implement security validation framework

---
*Generated by AI Village Security Manager Agent*
""")
        
        logger.info(f"Executive summary saved to {summary_path}")

def main():
    """Main execution function"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    auditor = EmergencySecurityAuditor(project_root)
    auditor.scan_codebase()
    
    print(f"""
🚨 EMERGENCY SECURITY AUDIT COMPLETE

Total Findings: {len(auditor.findings)}
Critical Exposures: {len([f for f in auditor.findings if f.severity == 'CRITICAL'])}
Credential Exposures: {len(auditor.credential_exposures)}

Reports generated in: {project_root}/reports/security/
- emergency_audit_report.json
- EMERGENCY_SECURITY_SUMMARY.md

⚠️  IMMEDIATE ACTION REQUIRED: Review and implement fixes immediately!
""")

if __name__ == "__main__":
    main()