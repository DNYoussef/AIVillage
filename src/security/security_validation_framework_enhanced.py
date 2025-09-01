#!/usr/bin/env python3
"""
Enhanced Security Validation Framework
Comprehensive security validation with MCP integration for AIVillage
"""

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import subprocess

# Configure logging
logger = logging.getLogger(__name__)


class SecurityValidationLevel(Enum):
    """Security validation severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"


class SecurityCheckType(Enum):
    """Types of security checks"""
    STATIC_ANALYSIS = "static_analysis"
    CREDENTIAL_SCAN = "credential_scan"
    DEPENDENCY_AUDIT = "dependency_audit"
    CONFIGURATION_REVIEW = "configuration_review"
    PERMISSION_ANALYSIS = "permission_analysis"
    ENCRYPTION_VALIDATION = "encryption_validation"


@dataclass
class SecurityFinding:
    """Individual security finding"""
    id: str
    check_type: SecurityCheckType
    severity: SecurityValidationLevel
    title: str
    description: str
    file_path: str
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None
    cvss_score: Optional[float] = None
    cwe_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class ValidationReport:
    """Complete security validation report"""
    scan_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    findings: List[SecurityFinding] = field(default_factory=list)
    scanned_files: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityValidationFramework:
    """Enhanced security validation framework with MCP integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SecurityValidationFramework")
        self.scan_patterns = self._initialize_scan_patterns()
        self.credential_patterns = self._initialize_credential_patterns()
        self.configuration_checks = self._initialize_configuration_checks()
        
    def _initialize_scan_patterns(self) -> Dict[str, Dict]:
        """Initialize security scan patterns"""
        return {
            "hardcoded_credentials": {
                "patterns": [
                    r'password\s*=\s*["\'][^"\']{3,}["\']',
                    r'secret\s*=\s*["\'][^"\']{10,}["\']',
                    r'api_key\s*=\s*["\'][^"\']{10,}["\']',
                    r'token\s*=\s*["\'][^"\']{10,}["\']',
                    r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']+["\']',
                ],
                "severity": SecurityValidationLevel.CRITICAL,
                "cwe_id": "CWE-798"
            },
            "insecure_protocols": {
                "patterns": [
                    r'http://(?!localhost|127\.0\.0\.1)',
                    r'ftp://',
                    r'telnet://',
                    r'import\s+telnetlib',
                ],
                "severity": SecurityValidationLevel.HIGH,
                "cwe_id": "CWE-319"
            },
            "weak_encryption": {
                "patterns": [
                    r'hashlib\.md5\(',
                    r'hashlib\.sha1\(',
                    r'DES\(',
                    r'RC4\(',
                ],
                "severity": SecurityValidationLevel.MEDIUM,
                "cwe_id": "CWE-327"
            },
            "sql_injection_risk": {
                "patterns": [
                    r'execute\s*\(\s*["\'][^"\']*%s[^"\']*["\']',
                    r'query\s*\(\s*["\'][^"\']*\+[^"\']*["\']',
                    r'format\s*\(\s*["\'][^"\']*SELECT[^"\']*["\']',
                ],
                "severity": SecurityValidationLevel.HIGH,
                "cwe_id": "CWE-89"
            },
            "debug_information_leak": {
                "patterns": [
                    r'print\s*\(\s*password',
                    r'logger\.\w+\s*\(\s*.*password',
                    r'debug.*=.*True',
                ],
                "severity": SecurityValidationLevel.MEDIUM,
                "cwe_id": "CWE-489"
            }
        }
    
    def _initialize_credential_patterns(self) -> List[Dict]:
        """Initialize credential detection patterns"""
        return [
            {
                "name": "AWS Keys",
                "pattern": r'AKIA[0-9A-Z]{16}',
                "severity": SecurityValidationLevel.CRITICAL
            },
            {
                "name": "SSH Private Keys",
                "pattern": r'-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----',
                "severity": SecurityValidationLevel.CRITICAL
            },
            {
                "name": "Generic Secrets",
                "pattern": r'(?i)(secret|password|passwd|pwd|token|key)\s*[:=]\s*["\'][^"\']{8,}["\']',
                "severity": SecurityValidationLevel.HIGH
            },
            {
                "name": "Database URLs",
                "pattern": r'(mongodb|mysql|postgresql|redis)://[^:]+:[^@]+@',
                "severity": SecurityValidationLevel.HIGH
            },
            {
                "name": "API Tokens",
                "pattern": r'["\'][a-zA-Z0-9]{32,}["\']',
                "severity": SecurityValidationLevel.MEDIUM
            }
        ]
    
    def _initialize_configuration_checks(self) -> List[Dict]:
        """Initialize configuration security checks"""
        return [
            {
                "name": "Insecure Bind Interface",
                "pattern": r'bind.*0\.0\.0\.0|host.*0\.0\.0\.0',
                "severity": SecurityValidationLevel.CRITICAL,
                "description": "Service binding to all interfaces (0.0.0.0)"
            },
            {
                "name": "Debug Mode Enabled", 
                "pattern": r'debug\s*=\s*True',
                "severity": SecurityValidationLevel.MEDIUM,
                "description": "Debug mode enabled in production code"
            },
            {
                "name": "Weak CORS Configuration",
                "pattern": r'allow_origins=\["?\*"?\]',
                "severity": SecurityValidationLevel.HIGH,
                "description": "CORS configured to allow all origins"
            },
            {
                "name": "Insecure Random",
                "pattern": r'random\.random\(\)|random\.randint\(',
                "severity": SecurityValidationLevel.MEDIUM,
                "description": "Using non-cryptographic random functions"
            }
        ]

    async def run_comprehensive_scan(self, scan_path: str) -> ValidationReport:
        """Run comprehensive security validation scan"""
        scan_id = hashlib.sha256(f"{scan_path}-{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        report = ValidationReport(scan_id=scan_id, start_time=datetime.now())
        
        self.logger.info(f"Starting comprehensive security scan {scan_id} on {scan_path}")
        
        try:
            # Collect all Python files
            python_files = await self._collect_python_files(scan_path)
            report.scanned_files = set(python_files)
            
            # Run different types of security checks
            checks = [
                self._run_static_analysis(python_files),
                self._run_credential_scan(python_files),
                self._run_configuration_review(python_files),
                self._run_dependency_audit(scan_path),
            ]
            
            # Execute all checks concurrently
            check_results = await asyncio.gather(*checks, return_exceptions=True)
            
            # Aggregate findings
            for result in check_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Security check failed: {result}")
                    continue
                report.findings.extend(result)
            
            # Update statistics
            self._calculate_report_statistics(report)
            
        except Exception as e:
            self.logger.error(f"Comprehensive security scan failed: {e}")
            raise
        finally:
            report.end_time = datetime.now()
            
        self.logger.info(f"Security scan {scan_id} completed. Found {report.total_findings} issues.")
        return report

    async def _collect_python_files(self, scan_path: str) -> List[str]:
        """Collect all Python files for scanning"""
        python_files = []
        
        if os.path.isfile(scan_path) and scan_path.endswith('.py'):
            return [scan_path]
            
        for root, dirs, files in os.walk(scan_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
                    
        return python_files

    async def _run_static_analysis(self, files: List[str]) -> List[SecurityFinding]:
        """Run static analysis security checks"""
        findings = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                # Check against all patterns
                for check_name, check_data in self.scan_patterns.items():
                    for pattern in check_data["patterns"]:
                        matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                        
                        for match in matches:
                            line_number = content[:match.start()].count('\\n') + 1
                            
                            finding = SecurityFinding(
                                id=f"SA-{hashlib.sha256(f'{file_path}-{line_number}-{check_name}'.encode()).hexdigest()[:8]}",
                                check_type=SecurityCheckType.STATIC_ANALYSIS,
                                severity=check_data["severity"],
                                title=f"Static Analysis: {check_name.replace('_', ' ').title()}",
                                description=f"Potential security issue detected in {file_path}",
                                file_path=file_path,
                                line_number=line_number,
                                code_snippet=match.group(0),
                                cwe_id=check_data.get("cwe_id"),
                                recommendation=self._get_recommendation(check_name)
                            )
                            findings.append(finding)
                            
            except Exception as e:
                self.logger.warning(f"Failed to analyze file {file_path}: {e}")
                
        return findings

    async def _run_credential_scan(self, files: List[str]) -> List[SecurityFinding]:
        """Run credential exposure scan"""
        findings = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for cred_check in self.credential_patterns:
                    matches = re.finditer(cred_check["pattern"], content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        # Skip if it's a comment or test file
                        if self._is_legitimate_occurrence(match.group(0), file_path):
                            continue
                            
                        line_number = content[:match.start()].count('\\n') + 1
                        
                        cred_name = cred_check["name"]
                        finding_id = f"CRED-{hashlib.sha256(f'{file_path}-{line_number}-{cred_name}'.encode()).hexdigest()[:8]}"
                        finding = SecurityFinding(
                            id=finding_id,
                            check_type=SecurityCheckType.CREDENTIAL_SCAN,
                            severity=cred_check["severity"],
                            title=f"Credential Exposure: {cred_name}",
                            description=f"Potential {cred_check['name'].lower()} found in {file_path}",
                            file_path=file_path,
                            line_number=line_number,
                            code_snippet=self._sanitize_code_snippet(match.group(0)),
                            recommendation="Move credentials to environment variables or secure configuration"
                        )
                        findings.append(finding)
                        
            except Exception as e:
                self.logger.warning(f"Failed credential scan on file {file_path}: {e}")
                
        return findings

    async def _run_configuration_review(self, files: List[str]) -> List[SecurityFinding]:
        """Run configuration security review"""
        findings = []
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                for config_check in self.configuration_checks:
                    matches = re.finditer(config_check["pattern"], content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        line_number = content[:match.start()].count('\\n') + 1
                        
                        finding = SecurityFinding(
                            id=f"CFG-{hashlib.sha256(f'{file_path}-{line_number}-{config_check[\"name\"]}'.encode()).hexdigest()[:8]}",
                            check_type=SecurityCheckType.CONFIGURATION_REVIEW,
                            severity=config_check["severity"],
                            title=f"Configuration Issue: {config_check['name']}",
                            description=config_check["description"],
                            file_path=file_path,
                            line_number=line_number,
                            code_snippet=match.group(0),
                            recommendation=self._get_config_recommendation(config_check["name"])
                        )
                        findings.append(finding)
                        
            except Exception as e:
                self.logger.warning(f"Failed configuration review on file {file_path}: {e}")
                
        return findings

    async def _run_dependency_audit(self, scan_path: str) -> List[SecurityFinding]:
        """Run dependency security audit"""
        findings = []
        
        try:
            # Look for requirements files
            req_files = ['requirements.txt', 'Pipfile', 'pyproject.toml', 'setup.py']
            
            for req_file in req_files:
                req_path = os.path.join(scan_path, req_file)
                if os.path.exists(req_path):
                    # Run safety check if available
                    try:
                        result = subprocess.run(['safety', 'check', '-r', req_path], 
                                              capture_output=True, text=True, timeout=60)
                        if result.returncode != 0 and "vulnerabilities found" in result.stdout:
                            finding = SecurityFinding(
                                id=f"DEP-{hashlib.sha256(req_path.encode()).hexdigest()[:8]}",
                                check_type=SecurityCheckType.DEPENDENCY_AUDIT,
                                severity=SecurityValidationLevel.HIGH,
                                title="Vulnerable Dependencies Detected",
                                description=f"Security vulnerabilities found in dependencies: {req_path}",
                                file_path=req_path,
                                code_snippet=result.stdout[:500],
                                recommendation="Update vulnerable dependencies to secure versions"
                            )
                            findings.append(finding)
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        self.logger.warning("Safety tool not available for dependency audit")
                        
        except Exception as e:
            self.logger.warning(f"Failed dependency audit: {e}")
            
        return findings

    def _is_legitimate_occurrence(self, match_text: str, file_path: str) -> bool:
        """Check if credential match is legitimate (test, example, etc.)"""
        # Skip test files and examples
        if any(x in file_path.lower() for x in ['test', 'example', 'demo', 'mock']):
            return True
            
        # Skip obvious placeholder values
        placeholder_patterns = [
            'password', 'secret', 'key', 'token', 'changeme', 'placeholder',
            'example', 'test', 'demo', 'mock', 'fake', 'dummy'
        ]
        
        return any(placeholder in match_text.lower() for placeholder in placeholder_patterns)

    def _sanitize_code_snippet(self, code: str) -> str:
        """Sanitize sensitive code snippets"""
        # Mask potential sensitive values
        return re.sub(r'(["\'][^"\']{3,}["\'])', '"***REDACTED***"', code)

    def _get_recommendation(self, check_name: str) -> str:
        """Get security recommendation for finding type"""
        recommendations = {
            "hardcoded_credentials": "Store credentials in environment variables or secure configuration",
            "insecure_protocols": "Use HTTPS/TLS encrypted protocols instead",
            "weak_encryption": "Use strong cryptographic algorithms like SHA-256 or AES",
            "sql_injection_risk": "Use parameterized queries or ORM methods",
            "debug_information_leak": "Remove debug statements and disable debug mode in production"
        }
        return recommendations.get(check_name, "Review and remediate security issue")

    def _get_config_recommendation(self, config_name: str) -> str:
        """Get configuration security recommendation"""
        recommendations = {
            "Insecure Bind Interface": "Bind only to specific interfaces (127.0.0.1 for localhost)",
            "Debug Mode Enabled": "Disable debug mode in production environments",
            "Weak CORS Configuration": "Configure CORS with specific allowed origins",
            "Insecure Random": "Use secrets module for cryptographic purposes"
        }
        return recommendations.get(config_name, "Review and secure configuration")

    def _calculate_report_statistics(self, report: ValidationReport):
        """Calculate report statistics"""
        report.total_findings = len(report.findings)
        
        for finding in report.findings:
            if finding.severity == SecurityValidationLevel.CRITICAL:
                report.critical_count += 1
            elif finding.severity == SecurityValidationLevel.HIGH:
                report.high_count += 1
            elif finding.severity == SecurityValidationLevel.MEDIUM:
                report.medium_count += 1
            else:
                report.low_count += 1

    async def export_report(self, report: ValidationReport, output_path: str) -> str:
        """Export validation report to JSON"""
        try:
            report_data = {
                "scan_id": report.scan_id,
                "start_time": report.start_time.isoformat(),
                "end_time": report.end_time.isoformat() if report.end_time else None,
                "statistics": {
                    "total_findings": report.total_findings,
                    "critical": report.critical_count,
                    "high": report.high_count,
                    "medium": report.medium_count,
                    "low": report.low_count
                },
                "scanned_files": list(report.scanned_files),
                "findings": [
                    {
                        "id": finding.id,
                        "type": finding.check_type.value,
                        "severity": finding.severity.value,
                        "title": finding.title,
                        "description": finding.description,
                        "file": finding.file_path,
                        "line": finding.line_number,
                        "code": finding.code_snippet,
                        "recommendation": finding.recommendation,
                        "cwe_id": finding.cwe_id,
                        "created_at": finding.created_at.isoformat()
                    }
                    for finding in report.findings
                ],
                "metadata": report.metadata
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            self.logger.info(f"Security validation report exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export report: {e}")
            raise


# Security configuration externalization utility
class SecurityConfigurationManager:
    """Manage security configuration externalization"""
    
    def __init__(self):
        self.required_env_vars = {
            "REDIS_PASSWORD": "Redis authentication password",
            "SECRET_KEY": "Application secret key", 
            "DATABASE_URL": "Database connection string",
            "JWT_SECRET": "JWT signing secret",
            "ENCRYPTION_KEY": "Data encryption key"
        }
    
    def validate_environment(self) -> List[str]:
        """Validate required environment variables"""
        missing_vars = []
        
        for var_name, description in self.required_env_vars.items():
            if not os.getenv(var_name):
                missing_vars.append(f"{var_name}: {description}")
                
        return missing_vars
    
    def generate_env_template(self) -> str:
        """Generate .env template file"""
        template_lines = [
            "# AIVillage Security Configuration",
            "# Generated by SecurityConfigurationManager",
            f"# Generated at: {datetime.now().isoformat()}",
            "",
        ]
        
        for var_name, description in self.required_env_vars.items():
            template_lines.extend([
                f"# {description}",
                f"{var_name}=CHANGE_ME_TO_SECURE_VALUE",
                ""
            ])
            
        return "\\n".join(template_lines)


async def main():
    """Main execution for security validation"""
    framework = SecurityValidationFramework()
    
    # Run comprehensive security scan
    report = await framework.run_comprehensive_scan("C:\\Users\\17175\\Desktop\\AIVillage\\src")
    
    # Export report
    await framework.export_report(report, "security_validation_report.json")
    
    print(f"Security validation completed:")
    print(f"  Total findings: {report.total_findings}")
    print(f"  Critical: {report.critical_count}")
    print(f"  High: {report.high_count}")
    print(f"  Medium: {report.medium_count}")
    print(f"  Low: {report.low_count}")


if __name__ == "__main__":
    asyncio.run(main())