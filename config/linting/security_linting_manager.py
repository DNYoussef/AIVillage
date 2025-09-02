"""
AIVillage Security Linting Manager
Comprehensive security scanning with MCP integration
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
import yaml

from src.performance.caching_manager import cache_manager, CacheKey

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Standardized security finding structure"""
    tool: str
    rule_id: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    title: str
    description: str
    file_path: str
    line_number: int
    column_number: Optional[int]
    cwe_id: Optional[str]
    cve_id: Optional[str]
    confidence: str  # 'high', 'medium', 'low'
    fix_suggestion: Optional[str]
    references: List[str]
    fingerprint: str
    timestamp: str


@dataclass
class SecurityScanResult:
    """Security scan results container"""
    tool: str
    scan_type: str  # 'sast', 'secrets', 'dependencies', 'container'
    status: str  # 'passed', 'failed', 'error'
    findings: List[SecurityFinding]
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    execution_time: float
    coverage_stats: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: str


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    block_on_critical: bool = True
    block_on_high: bool = False
    max_medium_issues: int = 10
    max_low_issues: int = 50
    required_tools: List[str] = None
    severity_thresholds: Dict[str, int] = None
    exempted_rules: List[str] = None
    custom_rules_path: Optional[str] = None


class SecurityLintingManager:
    """
    Comprehensive security linting with MCP coordination
    Handles SAST, secrets detection, dependency scanning, and container security
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/linting/unified_config.yml")
        self.config = self._load_security_config()
        self.policy = SecurityPolicy(**self.config.get("security_policy", {}))
        
        # Security tool configurations
        self.sast_tools = {
            "bandit": self._run_bandit_sast,
            "semgrep": self._run_semgrep_sast,
            "pysec": self._run_pysec_analysis,
        }
        
        self.secrets_tools = {
            "detect-secrets": self._run_detect_secrets,
            "trufflesec": self._run_trufflesec,
            "gitleaks": self._run_gitleaks,
        }
        
        self.dependency_tools = {
            "pip-audit": self._run_pip_audit,
            "safety": self._run_safety_scan,
            "osv-scanner": self._run_osv_scanner,
        }
        
        self.container_tools = {
            "trivy": self._run_trivy_scan,
            "snyk": self._run_snyk_container,
        }
        
        # Results storage
        self.scan_results: Dict[str, SecurityScanResult] = {}
        self.all_findings: List[SecurityFinding] = []
        
        # Known false positives and exemptions
        self.exemptions = self._load_exemptions()

    def _load_security_config(self) -> Dict[str, Any]:
        """Load security-specific configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                return full_config.get("security", {})
        
        # Default security configuration
        return {
            "block_on_critical": True,
            "auto_fix": False,
            "report_format": "sarif",
            "fail_on_severity": "HIGH",
            "security_policy": {
                "block_on_critical": True,
                "block_on_high": False,
                "max_medium_issues": 10,
                "required_tools": ["bandit", "detect-secrets", "pip-audit"],
                "severity_thresholds": {
                    "critical": 0,
                    "high": 2,
                    "medium": 10,
                    "low": 50
                }
            },
            "tools": {
                "bandit": {
                    "enabled": True,
                    "config_file": ".bandit.yml",
                    "severity": ["HIGH", "MEDIUM"],
                    "confidence": ["HIGH", "MEDIUM"]
                },
                "semgrep": {
                    "enabled": True,
                    "rulesets": ["security", "owasp-top-ten", "cwe-top-25"],
                    "custom_rules": "config/security/custom-semgrep-rules"
                },
                "detect_secrets": {
                    "enabled": True,
                    "baseline": ".secrets.baseline",
                    "plugins": ["all"]
                }
            }
        }

    def _load_exemptions(self) -> Dict[str, List[str]]:
        """Load security finding exemptions"""
        exemptions_file = Path("config/security/exemptions.yml")
        if exemptions_file.exists():
            with open(exemptions_file, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            "false_positives": [],
            "accepted_risks": [],
            "temp_exemptions": []
        }

    async def run_comprehensive_security_scan(self, target_paths: List[str]) -> Dict[str, SecurityScanResult]:
        """Run comprehensive security scanning across all categories"""
        logger.info("Starting comprehensive security scan...")
        
        # Initialize MCP coordination
        await self._initialize_security_mcp()
        
        # Check cache first
        cache_key = self._generate_cache_key(target_paths, "comprehensive")
        cached_result = await self._get_cached_results(cache_key)
        if cached_result:
            logger.info("Using cached comprehensive security scan results")
            return cached_result
        
        # Run parallel security scans
        scan_tasks = []
        
        # SAST (Static Application Security Testing)
        scan_tasks.append(self.run_sast_analysis(target_paths))
        
        # Secrets detection
        scan_tasks.append(self.run_secrets_detection(target_paths))
        
        # Dependency vulnerability scanning
        scan_tasks.append(self.run_dependency_scanning())
        
        # Infrastructure as Code security
        scan_tasks.append(self.run_iac_security_scan(target_paths))
        
        # Execute all scans in parallel
        scan_results = await asyncio.gather(*scan_tasks, return_exceptions=True)
        
        # Merge results
        comprehensive_results = {}
        for result in scan_results:
            if isinstance(result, dict):
                comprehensive_results.update(result)
            elif isinstance(result, Exception):
                logger.error(f"Security scan failed: {result}")
        
        # Aggregate findings and apply policy
        aggregated_findings = await self._aggregate_security_findings(comprehensive_results)
        policy_result = await self._apply_security_policy(aggregated_findings)
        
        # Cache results
        await self._cache_results(cache_key, comprehensive_results, ttl=3600)
        
        # Update MCP with findings
        await self._update_security_patterns(aggregated_findings)
        
        return comprehensive_results

    async def run_sast_analysis(self, target_paths: List[str]) -> Dict[str, SecurityScanResult]:
        """Run Static Application Security Testing"""
        logger.info("Running SAST analysis...")
        
        sast_results = {}
        
        # Run bandit for Python security issues
        if any(Path(p).glob("**/*.py") for p in target_paths):
            try:
                bandit_result = await self._run_bandit_sast(target_paths)
                sast_results["bandit"] = bandit_result
            except Exception as e:
                logger.error(f"Bandit scan failed: {e}")
        
        # Run Semgrep for multi-language security analysis
        try:
            semgrep_result = await self._run_semgrep_sast(target_paths)
            sast_results["semgrep"] = semgrep_result
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
        
        return sast_results

    async def _run_bandit_sast(self, target_paths: List[str]) -> SecurityScanResult:
        """Run Bandit security scanner for Python"""
        start_time = datetime.now()
        
        try:
            # Build bandit command
            cmd = [
                "bandit",
                "-f", "json",
                "-r",
                "-ll",  # Medium and high severity only
                "-o", "/tmp/bandit_output.json"
            ]
            
            # Add target paths
            python_paths = []
            for path in target_paths:
                path_obj = Path(path)
                if path_obj.is_file() and path_obj.suffix == ".py":
                    python_paths.append(str(path_obj))
                elif path_obj.is_dir():
                    python_files = list(path_obj.rglob("*.py"))
                    if python_files:
                        python_paths.append(str(path_obj))
            
            if not python_paths:
                return self._create_empty_scan_result("bandit", "sast")
            
            cmd.extend(python_paths)
            
            # Execute bandit
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse results
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            if Path("/tmp/bandit_output.json").exists():
                with open("/tmp/bandit_output.json", 'r') as f:
                    bandit_data = json.load(f)
                
                for issue in bandit_data.get("results", []):
                    severity = self._normalize_severity(issue.get("issue_severity", "LOW"))
                    confidence = issue.get("issue_confidence", "LOW").lower()
                    
                    finding = SecurityFinding(
                        tool="bandit",
                        rule_id=issue.get("test_id", "unknown"),
                        severity=severity,
                        title=issue.get("test_name", "Security Issue"),
                        description=issue.get("issue_text", ""),
                        file_path=issue.get("filename", ""),
                        line_number=issue.get("line_number", 0),
                        column_number=issue.get("col_offset"),
                        cwe_id=self._extract_cwe_from_bandit(issue),
                        cve_id=None,
                        confidence=confidence,
                        fix_suggestion=self._generate_bandit_fix_suggestion(issue),
                        references=self._get_bandit_references(issue),
                        fingerprint=self._generate_fingerprint("bandit", issue),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    # Skip exempted findings
                    if not self._is_finding_exempted(finding):
                        findings.append(finding)
                        
                        # Count by severity
                        if severity == "critical":
                            critical_count += 1
                        elif severity == "high":
                            high_count += 1
                        elif severity == "medium":
                            medium_count += 1
                        else:
                            low_count += 1
            
            return SecurityScanResult(
                tool="bandit",
                scan_type="sast",
                status="passed" if not findings or (critical_count == 0 and high_count <= 2) else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"files_scanned": len(python_paths)},
                metadata={"bandit_version": "1.7.5", "command": " ".join(cmd)},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return self._create_error_scan_result("bandit", "sast", str(e))

    async def _run_semgrep_sast(self, target_paths: List[str]) -> SecurityScanResult:
        """Run Semgrep security analysis"""
        start_time = datetime.now()
        
        try:
            cmd = [
                "semgrep",
                "--config=security",
                "--config=owasp-top-ten", 
                "--config=cwe-top-25",
                "--json",
                "--quiet",
                "--no-git-ignore"
            ]
            
            # Add custom rules if available
            custom_rules_path = Path("config/security/custom-semgrep-rules")
            if custom_rules_path.exists():
                cmd.extend(["--config", str(custom_rules_path)])
            
            cmd.extend(target_paths)
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=400)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            if result.stdout.strip():
                try:
                    semgrep_data = json.loads(result.stdout)
                    
                    for finding_data in semgrep_data.get("results", []):
                        severity = self._map_semgrep_severity(finding_data.get("extra", {}).get("severity", "INFO"))
                        
                        finding = SecurityFinding(
                            tool="semgrep",
                            rule_id=finding_data.get("check_id", "unknown"),
                            severity=severity,
                            title=finding_data.get("extra", {}).get("message", "Security Issue"),
                            description=finding_data.get("extra", {}).get("message", ""),
                            file_path=finding_data.get("path", ""),
                            line_number=finding_data.get("start", {}).get("line", 0),
                            column_number=finding_data.get("start", {}).get("col", 0),
                            cwe_id=self._extract_cwe_from_semgrep(finding_data),
                            cve_id=None,
                            confidence="high",
                            fix_suggestion=self._generate_semgrep_fix_suggestion(finding_data),
                            references=finding_data.get("extra", {}).get("references", []),
                            fingerprint=self._generate_fingerprint("semgrep", finding_data),
                            timestamp=datetime.now().isoformat()
                        )
                        
                        if not self._is_finding_exempted(finding):
                            findings.append(finding)
                            
                            if severity == "critical":
                                critical_count += 1
                            elif severity == "high":
                                high_count += 1
                            elif severity == "medium":
                                medium_count += 1
                            else:
                                low_count += 1
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Semgrep JSON output")
            
            return SecurityScanResult(
                tool="semgrep",
                scan_type="sast",
                status="passed" if critical_count == 0 and high_count <= 3 else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"paths_scanned": len(target_paths)},
                metadata={"semgrep_version": "1.45.0", "rulesets": ["security", "owasp-top-ten", "cwe-top-25"]},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return self._create_error_scan_result("semgrep", "sast", str(e))

    async def run_secrets_detection(self, target_paths: List[str]) -> Dict[str, SecurityScanResult]:
        """Run secrets detection across multiple tools"""
        logger.info("Running secrets detection...")
        
        secrets_results = {}
        
        # Run detect-secrets
        try:
            detect_secrets_result = await self._run_detect_secrets(target_paths)
            secrets_results["detect-secrets"] = detect_secrets_result
        except Exception as e:
            logger.error(f"detect-secrets scan failed: {e}")
        
        # Run additional secrets scanners if available
        try:
            trufflesec_result = await self._run_trufflesec(target_paths)
            secrets_results["trufflesec"] = trufflesec_result
        except Exception as e:
            logger.debug(f"TruffleHog not available: {e}")
        
        return secrets_results

    async def _run_detect_secrets(self, target_paths: List[str]) -> SecurityScanResult:
        """Run detect-secrets scanner"""
        start_time = datetime.now()
        
        try:
            # Create temporary baseline if none exists
            baseline_path = Path(".secrets.baseline")
            temp_baseline = False
            
            if not baseline_path.exists():
                temp_baseline = True
                cmd_scan = ["detect-secrets", "scan", "--baseline", str(baseline_path)]
                cmd_scan.extend(target_paths)
                subprocess.run(cmd_scan, capture_output=True, timeout=120)
            
            # Audit the baseline to find new secrets
            cmd_audit = ["detect-secrets", "audit", str(baseline_path), "--display-results"]
            result = subprocess.run(cmd_audit, capture_output=True, text=True, timeout=120)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            # Parse baseline file for secrets
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline_data = json.load(f)
                
                for file_path, secrets in baseline_data.get("results", {}).items():
                    for secret in secrets:
                        # Only report unreviewed secrets as findings
                        if not secret.get("is_verified", False):
                            severity = self._determine_secret_severity(secret)
                            
                            finding = SecurityFinding(
                                tool="detect-secrets",
                                rule_id=secret.get("type", "unknown"),
                                severity=severity,
                                title=f"Potential {secret.get('type', 'Secret')} Found",
                                description=f"Potential secret detected in {file_path}",
                                file_path=file_path,
                                line_number=secret.get("line_number", 0),
                                column_number=None,
                                cwe_id="CWE-798",  # Use of Hard-coded Credentials
                                cve_id=None,
                                confidence="medium",
                                fix_suggestion="Move secret to environment variable or secure vault",
                                references=["https://owasp.org/Top10/A07_2021-Identification_and_Authentication_Failures/"],
                                fingerprint=secret.get("hashed_secret", ""),
                                timestamp=datetime.now().isoformat()
                            )
                            
                            if not self._is_finding_exempted(finding):
                                findings.append(finding)
                                
                                if severity == "critical":
                                    critical_count += 1
                                elif severity == "high":
                                    high_count += 1
                                else:
                                    medium_count += 1
            
            # Clean up temporary baseline
            if temp_baseline and baseline_path.exists():
                baseline_path.unlink()
            
            return SecurityScanResult(
                tool="detect-secrets",
                scan_type="secrets",
                status="passed" if critical_count == 0 else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"files_scanned": len(target_paths)},
                metadata={"baseline_path": str(baseline_path)},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"detect-secrets scan failed: {e}")
            return self._create_error_scan_result("detect-secrets", "secrets", str(e))

    async def _run_trufflesec(self, target_paths: List[str]) -> SecurityScanResult:
        """Run TruffleHog secrets scanner"""
        start_time = datetime.now()
        
        try:
            # Check if trufflehog is available
            which_result = subprocess.run(["which", "trufflehog"], capture_output=True)
            if which_result.returncode != 0:
                return self._create_empty_scan_result("trufflesec", "secrets", "TruffleHog not installed")
            
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            for target_path in target_paths:
                cmd = ["trufflehog", "filesystem", "--json", "--no-update", str(target_path)]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=200)
                
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            finding_data = json.loads(line)
                            
                            if finding_data.get("verified", False):  # Only verified secrets
                                severity = "high"  # Verified secrets are always high severity
                                
                                finding = SecurityFinding(
                                    tool="trufflesec",
                                    rule_id=finding_data.get("DetectorName", "unknown"),
                                    severity=severity,
                                    title=f"Verified {finding_data.get('DetectorName', 'Secret')} Found",
                                    description=f"Verified secret detected: {finding_data.get('Raw', '')[:50]}...",
                                    file_path=finding_data.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file", ""),
                                    line_number=finding_data.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("line", 0),
                                    column_number=None,
                                    cwe_id="CWE-798",
                                    cve_id=None,
                                    confidence="high",
                                    fix_suggestion="Immediately rotate this secret and move to secure vault",
                                    references=["https://github.com/trufflesecurity/trufflehog"],
                                    fingerprint=finding_data.get("Raw", "")[:20],
                                    timestamp=datetime.now().isoformat()
                                )
                                
                                if not self._is_finding_exempted(finding):
                                    findings.append(finding)
                                    high_count += 1
                        
                        except json.JSONDecodeError:
                            continue
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return SecurityScanResult(
                tool="trufflesec",
                scan_type="secrets",
                status="passed" if high_count == 0 else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"paths_scanned": len(target_paths)},
                metadata={"verified_only": True},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"TruffleHog scan failed: {e}")
            return self._create_error_scan_result("trufflesec", "secrets", str(e))

    async def run_dependency_scanning(self) -> Dict[str, SecurityScanResult]:
        """Run dependency vulnerability scanning"""
        logger.info("Running dependency vulnerability scanning...")
        
        dependency_results = {}
        
        # Run pip-audit for Python dependencies
        try:
            pip_audit_result = await self._run_pip_audit()
            dependency_results["pip-audit"] = pip_audit_result
        except Exception as e:
            logger.error(f"pip-audit scan failed: {e}")
        
        # Run safety for Python dependencies
        try:
            safety_result = await self._run_safety_scan()
            dependency_results["safety"] = safety_result
        except Exception as e:
            logger.error(f"safety scan failed: {e}")
        
        return dependency_results

    async def _run_pip_audit(self) -> SecurityScanResult:
        """Run pip-audit for Python dependency vulnerabilities"""
        start_time = datetime.now()
        
        try:
            cmd = ["pip-audit", "--format=json", "--progress-spinner=off"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            if result.stdout.strip():
                try:
                    audit_data = json.loads(result.stdout)
                    
                    for vuln in audit_data.get("vulnerabilities", []):
                        # Map pip-audit severity to our severity levels
                        aliases = vuln.get("aliases", [])
                        severity = self._determine_dependency_severity(aliases)
                        
                        finding = SecurityFinding(
                            tool="pip-audit",
                            rule_id=vuln.get("id", "unknown"),
                            severity=severity,
                            title=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                            description=vuln.get("description", ""),
                            file_path="requirements.txt",  # Simplified
                            line_number=0,
                            column_number=None,
                            cwe_id=self._extract_cwe_from_aliases(aliases),
                            cve_id=self._extract_cve_from_aliases(aliases),
                            confidence="high",
                            fix_suggestion=f"Update {vuln.get('package', 'package')} to version {vuln.get('fix_versions', ['latest'])[0] if vuln.get('fix_versions') else 'latest'}",
                            references=vuln.get("aliases", []),
                            fingerprint=f"{vuln.get('package', 'unknown')}-{vuln.get('id', 'unknown')}",
                            timestamp=datetime.now().isoformat()
                        )
                        
                        if not self._is_finding_exempted(finding):
                            findings.append(finding)
                            
                            if severity == "critical":
                                critical_count += 1
                            elif severity == "high":
                                high_count += 1
                            elif severity == "medium":
                                medium_count += 1
                            else:
                                low_count += 1
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-audit JSON output")
            
            return SecurityScanResult(
                tool="pip-audit",
                scan_type="dependencies",
                status="passed" if critical_count == 0 and high_count <= 2 else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"packages_scanned": len(audit_data.get("vulnerabilities", []))},
                metadata={"audit_version": "2.6.1"},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"pip-audit scan failed: {e}")
            return self._create_error_scan_result("pip-audit", "dependencies", str(e))

    async def _run_safety_scan(self) -> SecurityScanResult:
        """Run Safety scan for Python dependencies"""
        start_time = datetime.now()
        
        try:
            cmd = ["safety", "check", "--json", "--full-report"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            findings = []
            critical_count = high_count = medium_count = low_count = 0
            
            if result.stdout.strip():
                try:
                    safety_data = json.loads(result.stdout)
                    
                    for vuln in safety_data.get("vulnerabilities", []):
                        severity = self._map_safety_severity(vuln.get("vulnerability", {}).get("v", ""))
                        
                        finding = SecurityFinding(
                            tool="safety",
                            rule_id=vuln.get("vulnerability", {}).get("id", "unknown"),
                            severity=severity,
                            title=f"Vulnerable package: {vuln.get('package_name', 'unknown')}",
                            description=vuln.get("vulnerability", {}).get("advisory", ""),
                            file_path="requirements.txt",
                            line_number=0,
                            column_number=None,
                            cwe_id=vuln.get("vulnerability", {}).get("cwe", ""),
                            cve_id=vuln.get("vulnerability", {}).get("cve", ""),
                            confidence="high",
                            fix_suggestion=f"Update to version >= {vuln.get('vulnerable_spec', 'latest')}",
                            references=[vuln.get("vulnerability", {}).get("more_info_url", "")],
                            fingerprint=f"{vuln.get('package_name', 'unknown')}-{vuln.get('vulnerability', {}).get('id', 'unknown')}",
                            timestamp=datetime.now().isoformat()
                        )
                        
                        if not self._is_finding_exempted(finding):
                            findings.append(finding)
                            
                            if severity == "critical":
                                critical_count += 1
                            elif severity == "high":
                                high_count += 1
                            elif severity == "medium":
                                medium_count += 1
                            else:
                                low_count += 1
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Safety JSON output")
            
            return SecurityScanResult(
                tool="safety",
                scan_type="dependencies",
                status="passed" if critical_count == 0 else "failed",
                findings=findings,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                execution_time=execution_time,
                coverage_stats={"packages_checked": len(safety_data.get("vulnerabilities", []))},
                metadata={"safety_version": "2.3.1"},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return self._create_error_scan_result("safety", "dependencies", str(e))

    async def run_iac_security_scan(self, target_paths: List[str]) -> Dict[str, SecurityScanResult]:
        """Run Infrastructure as Code security scanning"""
        logger.info("Running Infrastructure as Code security scanning...")
        
        iac_results = {}
        
        # Look for IaC files
        iac_files = []
        for path in target_paths:
            path_obj = Path(path)
            if path_obj.is_dir():
                iac_files.extend(path_obj.rglob("*.tf"))      # Terraform
                iac_files.extend(path_obj.rglob("*.yml"))     # Ansible, K8s
                iac_files.extend(path_obj.rglob("*.yaml"))    # Ansible, K8s
                iac_files.extend(path_obj.rglob("Dockerfile")) # Docker
                iac_files.extend(path_obj.rglob("docker-compose*.yml"))
        
        if not iac_files:
            logger.info("No Infrastructure as Code files found")
            return iac_results
        
        # Run relevant IaC scanners
        try:
            # This would run tools like Checkov, Terrascan, etc.
            # For now, return empty result as these tools may not be installed
            iac_results["checkov"] = self._create_empty_scan_result("checkov", "iac", "IaC scanning feature disabled")
        except Exception as e:
            logger.error(f"IaC scanning failed: {e}")
        
        return iac_results

    # Utility methods for security scanning

    def _normalize_severity(self, severity: str) -> str:
        """Normalize severity levels across tools"""
        severity_upper = severity.upper()
        
        severity_mapping = {
            "CRITICAL": "critical",
            "HIGH": "high", 
            "MEDIUM": "medium",
            "LOW": "low",
            "INFO": "low",
            "WARNING": "medium",
            "ERROR": "high"
        }
        
        return severity_mapping.get(severity_upper, "medium")

    def _map_semgrep_severity(self, severity: str) -> str:
        """Map Semgrep severity to our severity levels"""
        mapping = {
            "ERROR": "high",
            "WARNING": "medium", 
            "INFO": "low"
        }
        return mapping.get(severity.upper(), "medium")

    def _determine_secret_severity(self, secret: Dict[str, Any]) -> str:
        """Determine severity level for detected secrets"""
        secret_type = secret.get("type", "").lower()
        
        # High-risk secret types
        high_risk_types = [
            "private_key", "rsa_private_key", "ssh_key",
            "aws_access_key", "gcp_service_account", "azure_storage_account_key",
            "github_token", "slack_token", "stripe_key"
        ]
        
        # Critical secret types
        critical_types = [
            "private_key_encrypted", "certificate",
            "database_connection_string"
        ]
        
        if any(t in secret_type for t in critical_types):
            return "critical"
        elif any(t in secret_type for t in high_risk_types):
            return "high"
        else:
            return "medium"

    def _determine_dependency_severity(self, aliases: List[str]) -> str:
        """Determine severity from CVE aliases"""
        for alias in aliases:
            if "CRITICAL" in alias.upper():
                return "critical"
            elif "HIGH" in alias.upper():
                return "high"
        
        # Default based on CVSS if available
        return "medium"

    def _map_safety_severity(self, advisory: str) -> str:
        """Map Safety advisory to severity"""
        advisory_lower = advisory.lower()
        
        if any(word in advisory_lower for word in ["critical", "severe", "remote code execution", "rce"]):
            return "critical"
        elif any(word in advisory_lower for word in ["high", "dangerous", "vulnerability"]):
            return "high"
        else:
            return "medium"

    def _generate_fingerprint(self, tool: str, finding_data: Dict[str, Any]) -> str:
        """Generate unique fingerprint for finding deduplication"""
        import hashlib
        
        key_components = [
            tool,
            str(finding_data.get("filename", finding_data.get("path", ""))),
            str(finding_data.get("line_number", finding_data.get("start", {}).get("line", 0))),
            str(finding_data.get("test_id", finding_data.get("check_id", "unknown")))
        ]
        
        fingerprint_string = "|".join(key_components)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]

    def _is_finding_exempted(self, finding: SecurityFinding) -> bool:
        """Check if finding is exempted"""
        fingerprint = finding.fingerprint
        
        return (
            fingerprint in self.exemptions.get("false_positives", []) or
            fingerprint in self.exemptions.get("accepted_risks", []) or
            finding.rule_id in self.exemptions.get("temp_exemptions", [])
        )

    def _extract_cwe_from_bandit(self, issue: Dict[str, Any]) -> Optional[str]:
        """Extract CWE ID from Bandit issue"""
        # Bandit test ID to CWE mapping (simplified)
        cwe_mapping = {
            "B105": "CWE-798",  # Hardcoded password
            "B106": "CWE-798",  # Hardcoded password
            "B107": "CWE-798",  # Hardcoded password
            "B108": "CWE-377",  # Insecure temp file
            "B110": "CWE-377",  # Try except pass
            "B602": "CWE-78",   # Subprocess shell=True
            "B608": "CWE-89",   # SQL injection
        }
        
        test_id = issue.get("test_id", "")
        return cwe_mapping.get(test_id)

    def _extract_cwe_from_semgrep(self, finding: Dict[str, Any]) -> Optional[str]:
        """Extract CWE ID from Semgrep finding"""
        metadata = finding.get("extra", {}).get("metadata", {})
        return metadata.get("cwe")

    def _extract_cwe_from_aliases(self, aliases: List[str]) -> Optional[str]:
        """Extract CWE ID from vulnerability aliases"""
        for alias in aliases:
            if alias.startswith("CWE-"):
                return alias
        return None

    def _extract_cve_from_aliases(self, aliases: List[str]) -> Optional[str]:
        """Extract CVE ID from vulnerability aliases"""
        for alias in aliases:
            if alias.startswith("CVE-"):
                return alias
        return None

    def _generate_bandit_fix_suggestion(self, issue: Dict[str, Any]) -> Optional[str]:
        """Generate fix suggestion for Bandit issue"""
        test_id = issue.get("test_id", "")
        
        fix_suggestions = {
            "B105": "Move hardcoded password to environment variable",
            "B106": "Use environment variable or secure credential store",
            "B108": "Use tempfile.NamedTemporaryFile() for secure temp files",
            "B602": "Avoid shell=True or use shlex.quote() for user input",
            "B608": "Use parameterized queries to prevent SQL injection"
        }
        
        return fix_suggestions.get(test_id)

    def _generate_semgrep_fix_suggestion(self, finding: Dict[str, Any]) -> Optional[str]:
        """Generate fix suggestion for Semgrep finding"""
        rule_id = finding.get("check_id", "")
        
        if "sql-injection" in rule_id:
            return "Use parameterized queries or ORM methods"
        elif "xss" in rule_id:
            return "Sanitize user input and use proper encoding"
        elif "hardcoded" in rule_id:
            return "Move sensitive data to environment variables"
        
        return "Review security best practices for this finding"

    def _get_bandit_references(self, issue: Dict[str, Any]) -> List[str]:
        """Get reference URLs for Bandit issue"""
        test_id = issue.get("test_id", "")
        return [f"https://bandit.readthedocs.io/en/latest/plugins/{test_id.lower()}.html"]

    def _create_empty_scan_result(self, tool: str, scan_type: str, reason: str = "") -> SecurityScanResult:
        """Create empty scan result"""
        return SecurityScanResult(
            tool=tool,
            scan_type=scan_type,
            status="skipped",
            findings=[],
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            execution_time=0.0,
            coverage_stats={"reason": reason},
            metadata={"skipped": True},
            timestamp=datetime.now().isoformat()
        )

    def _create_error_scan_result(self, tool: str, scan_type: str, error: str) -> SecurityScanResult:
        """Create error scan result"""
        return SecurityScanResult(
            tool=tool,
            scan_type=scan_type,
            status="error",
            findings=[],
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            execution_time=0.0,
            coverage_stats={},
            metadata={"error": error},
            timestamp=datetime.now().isoformat()
        )

    def _generate_cache_key(self, target_paths: List[str], scan_type: str) -> str:
        """Generate cache key for security scan results"""
        import hashlib
        
        paths_str = "|".join(sorted(target_paths))
        config_str = json.dumps(self.config, sort_keys=True)
        
        combined = f"{scan_type}|{paths_str}|{config_str}"
        return f"security_scan_{hashlib.sha256(combined.encode()).hexdigest()[:16]}"

    async def _get_cached_results(self, cache_key: str) -> Optional[Dict[str, SecurityScanResult]]:
        """Get cached security scan results"""
        try:
            cached_data = await cache_manager.get(cache_key, "security")
            if cached_data:
                # Convert dict back to SecurityScanResult objects
                results = {}
                for tool, data in cached_data.items():
                    if isinstance(data, dict):
                        # Reconstruct SecurityScanResult from dict
                        findings = [SecurityFinding(**f) for f in data.get("findings", [])]
                        data["findings"] = findings
                        results[tool] = SecurityScanResult(**data)
                return results
        except Exception as e:
            logger.warning(f"Failed to retrieve cached results: {e}")
        
        return None

    async def _cache_results(self, cache_key: str, results: Dict[str, SecurityScanResult], ttl: int):
        """Cache security scan results"""
        try:
            # Convert SecurityScanResult objects to dicts for caching
            cacheable_results = {}
            for tool, result in results.items():
                cacheable_results[tool] = asdict(result)
            
            await cache_manager.set(cache_key, cacheable_results, ttl=ttl, service="security")
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    async def _initialize_security_mcp(self):
        """Initialize MCP coordination for security scanning"""
        try:
            # This would initialize MCP hooks for security coordination
            logger.debug("Initializing security MCP coordination")
        except Exception as e:
            logger.warning(f"MCP initialization failed: {e}")

    async def _aggregate_security_findings(self, results: Dict[str, SecurityScanResult]) -> List[SecurityFinding]:
        """Aggregate and deduplicate security findings"""
        all_findings = []
        seen_fingerprints = set()
        
        for tool_result in results.values():
            for finding in tool_result.findings:
                if finding.fingerprint not in seen_fingerprints:
                    all_findings.append(finding)
                    seen_fingerprints.add(finding.fingerprint)
        
        # Sort by severity and confidence
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_findings.sort(key=lambda x: (severity_order.get(x.severity, 3), x.confidence != "high"))
        
        return all_findings

    async def _apply_security_policy(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Apply security policy to determine if build should pass/fail"""
        critical_count = sum(1 for f in findings if f.severity == "critical")
        high_count = sum(1 for f in findings if f.severity == "high")
        medium_count = sum(1 for f in findings if f.severity == "medium")
        low_count = sum(1 for f in findings if f.severity == "low")
        
        policy_result = {
            "passed": True,
            "block_deployment": False,
            "reasons": []
        }
        
        if self.policy.block_on_critical and critical_count > 0:
            policy_result["passed"] = False
            policy_result["block_deployment"] = True
            policy_result["reasons"].append(f"{critical_count} critical security issues found")
        
        if self.policy.block_on_high and high_count > 0:
            policy_result["passed"] = False
            policy_result["block_deployment"] = True
            policy_result["reasons"].append(f"{high_count} high severity security issues found")
        
        if medium_count > self.policy.max_medium_issues:
            policy_result["passed"] = False
            policy_result["reasons"].append(f"{medium_count} medium issues exceed threshold of {self.policy.max_medium_issues}")
        
        return policy_result

    async def _update_security_patterns(self, findings: List[SecurityFinding]):
        """Update MCP memory with security patterns for learning"""
        try:
            # This would update MCP memory with security patterns
            pattern_data = {
                "timestamp": datetime.now().isoformat(),
                "total_findings": len(findings),
                "severity_distribution": {
                    "critical": sum(1 for f in findings if f.severity == "critical"),
                    "high": sum(1 for f in findings if f.severity == "high"),
                    "medium": sum(1 for f in findings if f.severity == "medium"),
                    "low": sum(1 for f in findings if f.severity == "low"),
                },
                "common_issues": [f.rule_id for f in findings[:10]]  # Top 10 most common
            }
            
            await cache_manager.set("security_patterns_latest", pattern_data, ttl=86400, service="security")
            logger.debug("Security patterns updated in MCP memory")
        except Exception as e:
            logger.warning(f"Failed to update security patterns: {e}")


# Global security linting manager instance
security_linting_manager = SecurityLintingManager()


async def run_security_scan_cli():
    """CLI entry point for security scanning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVillage Security Linting Pipeline")
    parser.add_argument("--scan-type", choices=["sast", "secrets", "dependencies", "iac", "all"], 
                       default="all", help="Type of security scan")
    parser.add_argument("--paths", nargs="+", default=["."], help="Paths to scan")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--format", choices=["json", "sarif", "text"], default="json")
    parser.add_argument("--fail-on", choices=["critical", "high", "medium", "low"], default="critical")
    
    args = parser.parse_args()
    
    manager = security_linting_manager
    
    # Run scan based on type
    if args.scan_type == "sast":
        results = await manager.run_sast_analysis(args.paths)
    elif args.scan_type == "secrets":
        results = await manager.run_secrets_detection(args.paths)
    elif args.scan_type == "dependencies":
        results = await manager.run_dependency_scanning()
    elif args.scan_type == "iac":
        results = await manager.run_iac_security_scan(args.paths)
    else:
        results = await manager.run_comprehensive_security_scan(args.paths)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({tool: asdict(result) for tool, result in results.items()}, f, indent=2, default=str)
    else:
        print(json.dumps({tool: asdict(result) for tool, result in results.items()}, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(run_security_scan_cli())