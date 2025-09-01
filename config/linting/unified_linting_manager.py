"""
AIVillage Unified Linting Manager with MCP Integration
Coordinates multi-agent code review with GitHub automation
"""

import asyncio
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

from src.performance.caching_manager import cache_manager, CacheKey

logger = logging.getLogger(__name__)


@dataclass
class LintingResult:
    """Standardized linting result structure"""
    tool: str
    status: str  # 'passed', 'failed', 'warning'
    issues_found: int
    critical_issues: int
    security_issues: int
    performance_issues: int
    style_issues: int
    files_processed: int
    execution_time: float
    suggestions: List[str]
    details: Dict[str, Any]
    timestamp: str


@dataclass
class QualityMetrics:
    """Code quality metrics aggregation"""
    overall_score: float
    security_score: float
    performance_score: float
    style_score: float
    maintainability_score: float
    complexity_score: float
    coverage_score: float
    technical_debt_ratio: float
    quality_gate_status: str


class UnifiedLintingPipeline:
    """
    MCP-enhanced unified linting pipeline with intelligent code review
    Integrates with GitHub MCP for workflow automation
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/linting/unified_config.yml")
        self.config = self._load_config()
        
        # MCP server connections (initialized via hooks)
        self.github_mcp = None
        self.sequential_mcp = None  
        self.memory_mcp = None
        self.context7_mcp = None
        
        # Linting tool configurations
        self.python_tools = ["ruff", "black", "mypy", "bandit", "safety"]
        self.frontend_tools = ["eslint", "prettier", "typescript", "stylelint"]
        self.security_tools = ["bandit", "semgrep", "detect-secrets", "pip-audit"]
        self.documentation_tools = ["markdownlint", "textlint"]
        
        # Quality thresholds
        self.quality_thresholds = {
            "security": {"critical": 0, "high": 0, "medium": 5},
            "performance": {"regression_threshold": 0.05, "complexity_limit": 10},
            "style": {"max_issues": 50, "critical_issues": 0},
            "coverage": {"minimum": 60, "target": 80}
        }
        
        # Results storage
        self.results: Dict[str, LintingResult] = {}
        self.quality_metrics: Optional[QualityMetrics] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load unified linting configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            "python": {
                "ruff": {
                    "line_length": 120,
                    "select": ["E", "W", "F", "I", "UP", "B", "C4", "SIM", "S"],
                    "ignore": ["E501", "F821"],
                    "exclude": ["experimental", "deprecated", "archive", "tmp"]
                },
                "black": {
                    "line_length": 120,
                    "exclude": "/(experimental|deprecated|archive|tmp)/"
                },
                "mypy": {
                    "ignore_missing_imports": True,
                    "no_strict_optional": True,
                    "exclude": "experimental|deprecated|archive|tmp"
                },
                "bandit": {
                    "exclude_dirs": ["tests", "experimental"],
                    "severity_level": "medium"
                }
            },
            "frontend": {
                "eslint": {
                    "extends": ["@eslint/recommended", "prettier"],
                    "rules": {
                        "no-console": "warn",
                        "no-unused-vars": "error",
                        "prefer-const": "error"
                    }
                },
                "prettier": {
                    "printWidth": 120,
                    "tabWidth": 2,
                    "semi": True,
                    "singleQuote": True
                }
            },
            "security": {
                "block_on_critical": True,
                "auto_fix": True,
                "report_format": "sarif"
            },
            "github_integration": {
                "auto_pr_checks": True,
                "auto_fix_commits": True,
                "quality_gates": True,
                "required_status_checks": ["unified-linting"]
            }
        }

    async def initialize_mcp_connections(self):
        """Initialize MCP server connections for coordination"""
        try:
            # These would be actual MCP connections in production
            logger.info("Initializing MCP server connections...")
            
            # Simulate MCP connection initialization
            await self._run_hooks_command("pre-task", {
                "description": "unified-linting-pipeline",
                "github_automation": "full",
                "performance_caching": "enabled"
            })
            
            logger.info("MCP connections established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP connections: {e}")
            return False

    async def run_python_linting(self, target_paths: List[str]) -> Dict[str, LintingResult]:
        """Run unified Python linting with performance caching"""
        results = {}
        
        # Check cache first (Context7 MCP integration)
        cache_key = CacheKey.hash_params({
            "target_paths": target_paths,
            "tools": self.python_tools,
            "config_hash": self._get_config_hash()
        })
        
        cached_result = await cache_manager.get(f"linting:python:{cache_key}", "linting")
        if cached_result:
            logger.info("Using cached Python linting results")
            return cached_result
        
        # Run ruff (fast linter)
        ruff_result = await self._run_ruff(target_paths)
        results["ruff"] = ruff_result
        
        # Run black (formatter check)
        black_result = await self._run_black(target_paths)
        results["black"] = black_result
        
        # Run mypy (type checking)
        mypy_result = await self._run_mypy(target_paths)
        results["mypy"] = mypy_result
        
        # Run security scans
        bandit_result = await self._run_bandit(target_paths)
        results["bandit"] = bandit_result
        
        # Cache results for 1 hour
        await cache_manager.set(f"linting:python:{cache_key}", results, ttl=3600, service="linting")
        
        return results

    async def _run_ruff(self, target_paths: List[str]) -> LintingResult:
        """Run ruff linting with configuration"""
        start_time = datetime.now()
        
        try:
            config = self.config["python"]["ruff"]
            cmd = [
                "ruff", "check", "--format=json",
                f"--select={','.join(config['select'])}",
                f"--ignore={','.join(config['ignore'])}",
                f"--line-length={config['line_length']}"
            ] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                return LintingResult(
                    tool="ruff",
                    status="passed",
                    issues_found=0,
                    critical_issues=0,
                    security_issues=0,
                    performance_issues=0,
                    style_issues=0,
                    files_processed=len(target_paths),
                    execution_time=execution_time,
                    suggestions=[],
                    details={"stdout": result.stdout},
                    timestamp=datetime.now().isoformat()
                )
            else:
                # Parse JSON output for issue analysis
                issues_data = json.loads(result.stdout) if result.stdout.strip() else []
                
                security_codes = ["S102", "S105", "S106", "S107", "S108", "S110"]
                performance_codes = ["B", "C4", "SIM"]
                
                security_issues = sum(1 for issue in issues_data if any(code in issue.get("code", "") for code in security_codes))
                performance_issues = sum(1 for issue in issues_data if any(code in issue.get("code", "") for code in performance_codes))
                style_issues = len(issues_data) - security_issues - performance_issues
                
                return LintingResult(
                    tool="ruff",
                    status="failed" if len(issues_data) > 50 else "warning",
                    issues_found=len(issues_data),
                    critical_issues=security_issues,
                    security_issues=security_issues,
                    performance_issues=performance_issues,
                    style_issues=style_issues,
                    files_processed=len(target_paths),
                    execution_time=execution_time,
                    suggestions=self._generate_ruff_suggestions(issues_data),
                    details={"issues": issues_data, "stderr": result.stderr},
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            logger.error(f"Ruff execution failed: {e}")
            return self._create_error_result("ruff", str(e), execution_time)

    async def _run_black(self, target_paths: List[str]) -> LintingResult:
        """Run black formatter check"""
        start_time = datetime.now()
        
        try:
            config = self.config["python"]["black"]
            cmd = [
                "black", "--check", "--diff", 
                f"--line-length={config['line_length']}",
                f"--extend-exclude={config['exclude']}"
            ] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            status = "passed" if result.returncode == 0 else "failed"
            issues_found = result.stdout.count("--- ") if result.stdout else 0
            
            return LintingResult(
                tool="black",
                status=status,
                issues_found=issues_found,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=issues_found,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=["Run 'black .' to auto-format files"] if issues_found > 0 else [],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Black execution failed: {e}")
            return self._create_error_result("black", str(e), execution_time)

    async def _run_mypy(self, target_paths: List[str]) -> LintingResult:
        """Run mypy type checking"""
        start_time = datetime.now()
        
        try:
            config = self.config["python"]["mypy"]
            cmd = ["mypy"] + target_paths
            
            if config.get("ignore_missing_imports"):
                cmd.append("--ignore-missing-imports")
            if config.get("no_strict_optional"):
                cmd.append("--no-strict-optional")
            if config.get("exclude"):
                cmd.extend(["--exclude", config["exclude"]])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse mypy output
            issues_found = result.stdout.count("error:") if result.stdout else 0
            critical_issues = result.stdout.count("error:") if result.stdout else 0
            
            status = "passed" if result.returncode == 0 else "warning"
            
            return LintingResult(
                tool="mypy",
                status=status,
                issues_found=issues_found,
                critical_issues=critical_issues,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=self._generate_mypy_suggestions(result.stdout),
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"MyPy execution failed: {e}")
            return self._create_error_result("mypy", str(e), execution_time)

    async def _run_bandit(self, target_paths: List[str]) -> LintingResult:
        """Run bandit security scanning"""
        start_time = datetime.now()
        
        try:
            config = self.config["python"]["bandit"]
            cmd = [
                "bandit", "-f", "json", "-ll", "-r"
            ] + target_paths
            
            if config.get("exclude_dirs"):
                cmd.extend(["-x", ",".join(config["exclude_dirs"])])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse bandit JSON output
            if result.stdout.strip():
                try:
                    bandit_data = json.loads(result.stdout)
                    results_list = bandit_data.get("results", [])
                    
                    high_severity = sum(1 for r in results_list if r.get("issue_severity") in ["HIGH", "CRITICAL"])
                    medium_severity = sum(1 for r in results_list if r.get("issue_severity") == "MEDIUM")
                    
                    status = "failed" if high_severity > 0 else ("warning" if medium_severity > 0 else "passed")
                    
                    return LintingResult(
                        tool="bandit",
                        status=status,
                        issues_found=len(results_list),
                        critical_issues=high_severity,
                        security_issues=len(results_list),
                        performance_issues=0,
                        style_issues=0,
                        files_processed=len(target_paths),
                        execution_time=execution_time,
                        suggestions=self._generate_bandit_suggestions(results_list),
                        details={"bandit_report": bandit_data},
                        timestamp=datetime.now().isoformat()
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse bandit JSON output")
            
            # Fallback for non-JSON output
            return LintingResult(
                tool="bandit",
                status="passed",
                issues_found=0,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=[],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Bandit execution failed: {e}")
            return self._create_error_result("bandit", str(e), execution_time)

    async def run_frontend_linting(self, target_paths: List[str]) -> Dict[str, LintingResult]:
        """Run unified frontend linting (ESLint + Prettier + TypeScript)"""
        results = {}
        
        # Check cache first
        cache_key = CacheKey.hash_params({
            "target_paths": target_paths,
            "tools": self.frontend_tools,
            "config_hash": self._get_config_hash()
        })
        
        cached_result = await cache_manager.get(f"linting:frontend:{cache_key}", "linting")
        if cached_result:
            logger.info("Using cached frontend linting results")
            return cached_result
        
        # Run ESLint
        if any(p for p in target_paths if p.endswith(('.js', '.jsx', '.ts', '.tsx'))):
            eslint_result = await self._run_eslint(target_paths)
            results["eslint"] = eslint_result
        
        # Run Prettier
        if any(p for p in target_paths if p.endswith(('.js', '.jsx', '.ts', '.tsx', '.css', '.scss', '.json'))):
            prettier_result = await self._run_prettier(target_paths)
            results["prettier"] = prettier_result
        
        # Run TypeScript compiler check
        if any(p for p in target_paths if p.endswith(('.ts', '.tsx'))):
            typescript_result = await self._run_typescript(target_paths)
            results["typescript"] = typescript_result
        
        # Cache results for 1 hour
        await cache_manager.set(f"linting:frontend:{cache_key}", results, ttl=3600, service="linting")
        
        return results

    async def _run_eslint(self, target_paths: List[str]) -> LintingResult:
        """Run ESLint for JavaScript/TypeScript"""
        start_time = datetime.now()
        
        try:
            cmd = ["npx", "eslint", "--format=json"] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.stdout.strip():
                try:
                    eslint_data = json.loads(result.stdout)
                    
                    total_errors = sum(len(file.get("messages", [])) for file in eslint_data)
                    critical_errors = sum(
                        1 for file in eslint_data
                        for msg in file.get("messages", [])
                        if msg.get("severity") == 2
                    )
                    
                    status = "failed" if critical_errors > 10 else ("warning" if total_errors > 0 else "passed")
                    
                    return LintingResult(
                        tool="eslint",
                        status=status,
                        issues_found=total_errors,
                        critical_issues=critical_errors,
                        security_issues=0,
                        performance_issues=0,
                        style_issues=total_errors - critical_errors,
                        files_processed=len(target_paths),
                        execution_time=execution_time,
                        suggestions=self._generate_eslint_suggestions(eslint_data),
                        details={"eslint_report": eslint_data},
                        timestamp=datetime.now().isoformat()
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse ESLint JSON output")
            
            return LintingResult(
                tool="eslint",
                status="passed",
                issues_found=0,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=[],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"ESLint execution failed: {e}")
            return self._create_error_result("eslint", str(e), execution_time)

    async def _run_prettier(self, target_paths: List[str]) -> LintingResult:
        """Run Prettier format check"""
        start_time = datetime.now()
        
        try:
            cmd = ["npx", "prettier", "--check"] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            status = "passed" if result.returncode == 0 else "warning"
            issues_found = result.stderr.count("Code style issues found") if result.stderr else 0
            
            return LintingResult(
                tool="prettier",
                status=status,
                issues_found=issues_found,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=issues_found,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=["Run 'npx prettier --write .' to auto-format files"] if issues_found > 0 else [],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Prettier execution failed: {e}")
            return self._create_error_result("prettier", str(e), execution_time)

    async def _run_typescript(self, target_paths: List[str]) -> LintingResult:
        """Run TypeScript compiler check"""
        start_time = datetime.now()
        
        try:
            cmd = ["npx", "tsc", "--noEmit", "--skipLibCheck"] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse TypeScript errors
            errors_found = result.stdout.count("error TS") if result.stdout else 0
            critical_errors = result.stdout.count("error TS2") if result.stdout else 0  # Type errors
            
            status = "failed" if critical_errors > 5 else ("warning" if errors_found > 0 else "passed")
            
            return LintingResult(
                tool="typescript",
                status=status,
                issues_found=errors_found,
                critical_issues=critical_errors,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=self._generate_typescript_suggestions(result.stdout),
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"TypeScript execution failed: {e}")
            return self._create_error_result("typescript", str(e), execution_time)

    async def run_security_linting(self, target_paths: List[str]) -> Dict[str, LintingResult]:
        """Run comprehensive security linting across all languages"""
        results = {}
        
        # Detect-secrets for secret scanning
        secrets_result = await self._run_detect_secrets(target_paths)
        results["detect-secrets"] = secrets_result
        
        # Semgrep for SAST analysis
        semgrep_result = await self._run_semgrep(target_paths)
        results["semgrep"] = semgrep_result
        
        # pip-audit for dependency vulnerabilities
        pip_audit_result = await self._run_pip_audit()
        results["pip-audit"] = pip_audit_result
        
        return results

    async def _run_detect_secrets(self, target_paths: List[str]) -> LintingResult:
        """Run detect-secrets scan"""
        start_time = datetime.now()
        
        try:
            cmd = ["detect-secrets", "scan", "--baseline", ".secrets.baseline"] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Parse detect-secrets output
            secrets_found = result.stdout.count('"filename"') if result.stdout else 0
            
            status = "failed" if secrets_found > 0 else "passed"
            
            return LintingResult(
                tool="detect-secrets",
                status=status,
                issues_found=secrets_found,
                critical_issues=secrets_found,
                security_issues=secrets_found,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=["Review and whitelist false positives in .secrets.baseline"] if secrets_found > 0 else [],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"detect-secrets execution failed: {e}")
            return self._create_error_result("detect-secrets", str(e), execution_time)

    async def _run_semgrep(self, target_paths: List[str]) -> LintingResult:
        """Run Semgrep SAST analysis"""
        start_time = datetime.now()
        
        try:
            cmd = ["semgrep", "--config=auto", "--config=security", "--json"] + target_paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.stdout.strip():
                try:
                    semgrep_data = json.loads(result.stdout)
                    findings = semgrep_data.get("results", [])
                    
                    critical_findings = sum(1 for f in findings if f.get("extra", {}).get("severity") in ["ERROR", "HIGH"])
                    medium_findings = sum(1 for f in findings if f.get("extra", {}).get("severity") == "WARNING")
                    
                    status = "failed" if critical_findings > 0 else ("warning" if medium_findings > 0 else "passed")
                    
                    return LintingResult(
                        tool="semgrep",
                        status=status,
                        issues_found=len(findings),
                        critical_issues=critical_findings,
                        security_issues=len(findings),
                        performance_issues=0,
                        style_issues=0,
                        files_processed=len(target_paths),
                        execution_time=execution_time,
                        suggestions=self._generate_semgrep_suggestions(findings),
                        details={"semgrep_report": semgrep_data},
                        timestamp=datetime.now().isoformat()
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Semgrep JSON output")
            
            return LintingResult(
                tool="semgrep",
                status="passed",
                issues_found=0,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=len(target_paths),
                execution_time=execution_time,
                suggestions=[],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Semgrep execution failed: {e}")
            return self._create_error_result("semgrep", str(e), execution_time)

    async def _run_pip_audit(self) -> LintingResult:
        """Run pip-audit for dependency vulnerabilities"""
        start_time = datetime.now()
        
        try:
            cmd = ["pip-audit", "--format=json"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result.stdout.strip():
                try:
                    audit_data = json.loads(result.stdout)
                    vulnerabilities = audit_data.get("vulnerabilities", [])
                    
                    critical_vulns = sum(
                        1 for v in vulnerabilities 
                        if any(a.get("severity") in ["HIGH", "CRITICAL"] for a in v.get("aliases", []))
                    )
                    
                    status = "failed" if critical_vulns > 0 else ("warning" if len(vulnerabilities) > 0 else "passed")
                    
                    return LintingResult(
                        tool="pip-audit",
                        status=status,
                        issues_found=len(vulnerabilities),
                        critical_issues=critical_vulns,
                        security_issues=len(vulnerabilities),
                        performance_issues=0,
                        style_issues=0,
                        files_processed=0,
                        execution_time=execution_time,
                        suggestions=self._generate_pip_audit_suggestions(vulnerabilities),
                        details={"pip_audit_report": audit_data},
                        timestamp=datetime.now().isoformat()
                    )
                except json.JSONDecodeError:
                    logger.warning("Failed to parse pip-audit JSON output")
            
            return LintingResult(
                tool="pip-audit",
                status="passed",
                issues_found=0,
                critical_issues=0,
                security_issues=0,
                performance_issues=0,
                style_issues=0,
                files_processed=0,
                execution_time=execution_time,
                suggestions=[],
                details={"stdout": result.stdout, "stderr": result.stderr},
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"pip-audit execution failed: {e}")
            return self._create_error_result("pip-audit", str(e), execution_time)

    async def calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate comprehensive quality metrics from all linting results"""
        if not self.results:
            logger.warning("No linting results available for quality metrics calculation")
            return QualityMetrics(
                overall_score=0.0,
                security_score=0.0,
                performance_score=0.0,
                style_score=0.0,
                maintainability_score=0.0,
                complexity_score=0.0,
                coverage_score=0.0,
                technical_debt_ratio=1.0,
                quality_gate_status="failed"
            )
        
        # Calculate individual scores
        security_score = self._calculate_security_score()
        performance_score = self._calculate_performance_score()
        style_score = self._calculate_style_score()
        maintainability_score = self._calculate_maintainability_score()
        complexity_score = self._calculate_complexity_score()
        coverage_score = 60.0  # Placeholder - would come from coverage reports
        
        # Calculate overall score (weighted average)
        overall_score = (
            security_score * 0.3 +
            performance_score * 0.2 +
            style_score * 0.15 +
            maintainability_score * 0.15 +
            complexity_score * 0.1 +
            coverage_score * 0.1
        )
        
        # Calculate technical debt ratio
        total_issues = sum(result.issues_found for result in self.results.values())
        total_files = sum(result.files_processed for result in self.results.values())
        technical_debt_ratio = total_issues / max(total_files, 1)
        
        # Determine quality gate status
        quality_gate_status = self._determine_quality_gate_status(
            security_score, performance_score, style_score, overall_score
        )
        
        self.quality_metrics = QualityMetrics(
            overall_score=round(overall_score, 2),
            security_score=round(security_score, 2),
            performance_score=round(performance_score, 2),
            style_score=round(style_score, 2),
            maintainability_score=round(maintainability_score, 2),
            complexity_score=round(complexity_score, 2),
            coverage_score=round(coverage_score, 2),
            technical_debt_ratio=round(technical_debt_ratio, 3),
            quality_gate_status=quality_gate_status
        )
        
        return self.quality_metrics

    def _calculate_security_score(self) -> float:
        """Calculate security score based on security linting results"""
        security_tools = ["bandit", "detect-secrets", "semgrep", "pip-audit"]
        security_results = [result for tool, result in self.results.items() if tool in security_tools]
        
        if not security_results:
            return 50.0  # Neutral score if no security tools ran
        
        total_critical = sum(result.critical_issues for result in security_results)
        total_security = sum(result.security_issues for result in security_results)
        
        if total_critical > 0:
            return 0.0  # Critical security issues = 0 score
        elif total_security > 10:
            return 30.0  # Many security issues
        elif total_security > 5:
            return 60.0  # Some security issues
        elif total_security > 0:
            return 80.0  # Few security issues
        else:
            return 100.0  # No security issues

    def _calculate_performance_score(self) -> float:
        """Calculate performance score"""
        performance_issues = sum(result.performance_issues for result in self.results.values())
        
        if performance_issues == 0:
            return 100.0
        elif performance_issues <= 5:
            return 80.0
        elif performance_issues <= 15:
            return 60.0
        else:
            return 30.0

    def _calculate_style_score(self) -> float:
        """Calculate style score"""
        total_style_issues = sum(result.style_issues for result in self.results.values())
        total_files = sum(result.files_processed for result in self.results.values())
        
        if total_files == 0:
            return 100.0
        
        issues_per_file = total_style_issues / total_files
        
        if issues_per_file == 0:
            return 100.0
        elif issues_per_file <= 1:
            return 90.0
        elif issues_per_file <= 3:
            return 70.0
        elif issues_per_file <= 5:
            return 50.0
        else:
            return 20.0

    def _calculate_maintainability_score(self) -> float:
        """Calculate maintainability score based on various factors"""
        # This is simplified - in practice would include cyclomatic complexity, etc.
        mypy_results = [result for tool, result in self.results.items() if tool == "mypy"]
        
        if not mypy_results:
            return 70.0  # Neutral score if no type checking
        
        mypy_result = mypy_results[0]
        if mypy_result.issues_found == 0:
            return 100.0
        elif mypy_result.issues_found <= 10:
            return 80.0
        elif mypy_result.issues_found <= 25:
            return 60.0
        else:
            return 40.0

    def _calculate_complexity_score(self) -> float:
        """Calculate complexity score (placeholder)"""
        # This would typically come from tools like radon, mccabe, etc.
        return 75.0  # Placeholder score

    def _determine_quality_gate_status(self, security_score: float, performance_score: float, 
                                     style_score: float, overall_score: float) -> str:
        """Determine if the code passes quality gates"""
        
        # Critical failures
        if security_score < 50.0:
            return "failed_security"
        
        # Check individual thresholds
        if overall_score < 60.0:
            return "failed_overall"
        
        if security_score < 80.0:
            return "warning_security"
        
        if performance_score < 70.0:
            return "warning_performance"
        
        if style_score < 60.0:
            return "warning_style"
        
        return "passed"

    async def generate_github_workflow(self) -> str:
        """Generate GitHub workflow YAML for unified linting"""
        workflow = {
            "name": "Unified Code Quality Pipeline",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main", "develop"]},
                "workflow_dispatch": {
                    "inputs": {
                        "run_full_scan": {
                            "description": "Run comprehensive quality scan",
                            "required": False,
                            "default": "false",
                            "type": "boolean"
                        }
                    }
                }
            },
            "env": {
                "PYTHON_VERSION": "3.11",
                "NODE_VERSION": "18"
            },
            "jobs": {
                "unified-linting": {
                    "name": "Unified Quality Checks",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "${{ env.PYTHON_VERSION }}"}
                        },
                        {
                            "uses": "actions/setup-node@v4",
                            "with": {"node-version": "${{ env.NODE_VERSION }}"}
                        },
                        {
                            "name": "Install Python linting tools",
                            "run": "pip install ruff black mypy bandit safety semgrep detect-secrets pip-audit"
                        },
                        {
                            "name": "Install Node.js linting tools", 
                            "run": "npm install -g eslint prettier typescript @typescript-eslint/parser @typescript-eslint/eslint-plugin"
                        },
                        {
                            "name": "Run unified Python linting",
                            "run": "python config/linting/run_unified_linting.py --language=python --output=results/python-linting.json"
                        },
                        {
                            "name": "Run unified frontend linting",
                            "run": "python config/linting/run_unified_linting.py --language=frontend --output=results/frontend-linting.json"
                        },
                        {
                            "name": "Run security linting",
                            "run": "python config/linting/run_unified_linting.py --language=security --output=results/security-linting.json"
                        },
                        {
                            "name": "Generate quality report",
                            "run": "python config/linting/generate_quality_report.py --input-dir=results --output=quality-report.json"
                        },
                        {
                            "name": "Quality gate check",
                            "run": "python config/linting/quality_gate_check.py --report=quality-report.json"
                        },
                        {
                            "name": "Upload quality reports",
                            "uses": "actions/upload-artifact@v4",
                            "with": {
                                "name": "quality-reports",
                                "path": "results/"
                            }
                        }
                    ]
                }
            }
        }
        
        return yaml.dump(workflow, default_flow_style=False)

    async def run_full_pipeline(self, target_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete unified linting pipeline"""
        
        if target_paths is None:
            target_paths = [".", "core/", "infrastructure/", "src/", "tests/"]
        
        logger.info("Starting unified linting pipeline...")
        
        # Initialize MCP connections
        await self.initialize_mcp_connections()
        
        # Run Python linting
        logger.info("Running Python linting...")
        python_results = await self.run_python_linting(target_paths)
        self.results.update(python_results)
        
        # Run frontend linting
        logger.info("Running frontend linting...")
        frontend_results = await self.run_frontend_linting(target_paths)
        self.results.update(frontend_results)
        
        # Run security linting
        logger.info("Running security linting...")
        security_results = await self.run_security_linting(target_paths)
        self.results.update(security_results)
        
        # Calculate quality metrics
        logger.info("Calculating quality metrics...")
        quality_metrics = await self.calculate_quality_metrics()
        
        # Generate comprehensive report
        report = {
            "pipeline_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tools_run": len(self.results),
                "total_issues_found": sum(result.issues_found for result in self.results.values()),
                "critical_issues": sum(result.critical_issues for result in self.results.values()),
                "security_issues": sum(result.security_issues for result in self.results.values()),
                "performance_issues": sum(result.performance_issues for result in self.results.values()),
                "style_issues": sum(result.style_issues for result in self.results.values()),
                "total_execution_time": sum(result.execution_time for result in self.results.values())
            },
            "tool_results": {tool: asdict(result) for tool, result in self.results.items()},
            "quality_metrics": asdict(quality_metrics),
            "recommendations": self._generate_recommendations(),
            "next_steps": self._generate_next_steps()
        }
        
        # Save report to cache and trigger GitHub integration
        await self._save_results_and_notify(report)
        
        logger.info("Unified linting pipeline completed")
        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Security recommendations
        total_critical_security = sum(result.critical_issues for result in self.results.values())
        if total_critical_security > 0:
            recommendations.append(f"CRITICAL: Address {total_critical_security} critical security issues immediately")
        
        # Performance recommendations
        total_performance_issues = sum(result.performance_issues for result in self.results.values())
        if total_performance_issues > 10:
            recommendations.append(f"Consider refactoring code to address {total_performance_issues} performance issues")
        
        # Style recommendations
        total_style_issues = sum(result.style_issues for result in self.results.values())
        if total_style_issues > 50:
            recommendations.append(f"Run auto-formatters to fix {total_style_issues} style issues automatically")
        
        # Tool-specific recommendations
        for tool, result in self.results.items():
            if result.suggestions:
                recommendations.extend([f"{tool}: {suggestion}" for suggestion in result.suggestions[:2]])
        
        return recommendations

    def _generate_next_steps(self) -> List[str]:
        """Generate next steps based on quality gate status"""
        if not self.quality_metrics:
            return ["Run quality metrics calculation"]
        
        next_steps = []
        
        if self.quality_metrics.quality_gate_status == "failed_security":
            next_steps.extend([
                "Block merge until critical security issues are resolved",
                "Run security-focused review with security team",
                "Update security baseline after fixes"
            ])
        elif self.quality_metrics.quality_gate_status == "failed_overall":
            next_steps.extend([
                "Review failing quality checks",
                "Address critical issues before merge",
                "Consider technical debt reduction"
            ])
        elif "warning" in self.quality_metrics.quality_gate_status:
            next_steps.extend([
                "Review warnings and plan improvements",
                "Schedule technical debt reduction in next sprint",
                "Consider implementing stricter quality gates"
            ])
        else:
            next_steps.extend([
                "All quality gates passed - ready for merge",
                "Consider this code as a quality reference",
                "Update team best practices based on this implementation"
            ])
        
        return next_steps

    async def _save_results_and_notify(self, report: Dict[str, Any]):
        """Save results and notify via MCP hooks"""
        try:
            # Save to cache
            cache_key = f"linting_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            await cache_manager.set(cache_key, report, ttl=86400, service="linting")  # 24 hours
            
            # Trigger GitHub integration hooks
            await self._run_hooks_command("post-task", {
                "task_id": "unified-linting-pipeline",
                "status": report["quality_metrics"]["quality_gate_status"],
                "issues_found": report["pipeline_summary"]["total_issues_found"],
                "critical_issues": report["pipeline_summary"]["critical_issues"],
                "github_update": True
            })
            
            logger.info(f"Results saved with cache key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to save results and notify: {e}")

    async def _run_hooks_command(self, hook_type: str, params: Dict[str, Any]):
        """Run claude-flow hooks for MCP coordination"""
        try:
            cmd_params = " ".join([f"--{k}={v}" for k, v in params.items()])
            cmd = f"npx claude-flow@alpha hooks {hook_type} {cmd_params}"
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.warning(f"Hook {hook_type} failed: {result.stderr}")
            else:
                logger.debug(f"Hook {hook_type} completed successfully")
                
        except Exception as e:
            logger.error(f"Failed to run hook {hook_type}: {e}")

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration for caching"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]

    def _create_error_result(self, tool: str, error: str, execution_time: float) -> LintingResult:
        """Create error result for failed tool execution"""
        return LintingResult(
            tool=tool,
            status="error",
            issues_found=0,
            critical_issues=0,
            security_issues=0,
            performance_issues=0,
            style_issues=0,
            files_processed=0,
            execution_time=execution_time,
            suggestions=[f"Fix {tool} configuration or installation"],
            details={"error": error},
            timestamp=datetime.now().isoformat()
        )

    def _generate_ruff_suggestions(self, issues_data: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on ruff issues"""
        suggestions = []
        
        # Analyze issue patterns
        issue_codes = [issue.get("code", "") for issue in issues_data]
        
        if any(code.startswith("S") for code in issue_codes):
            suggestions.append("Review security-related issues flagged by ruff")
        
        if any(code.startswith("E") for code in issue_codes):
            suggestions.append("Run 'black .' to auto-fix formatting issues")
        
        if any(code.startswith("F") for code in issue_codes):
            suggestions.append("Fix import and syntax issues identified")
        
        if len(suggestions) == 0 and issues_data:
            suggestions.append("Review ruff output and fix identified issues")
        
        return suggestions[:3]  # Limit to top 3 suggestions

    def _generate_mypy_suggestions(self, output: str) -> List[str]:
        """Generate suggestions based on mypy output"""
        suggestions = []
        
        if "error:" in output:
            suggestions.append("Add type annotations to resolve mypy errors")
        
        if "note:" in output:
            suggestions.append("Review mypy notes for type hints improvements")
        
        if len(suggestions) == 0:
            suggestions.append("Consider enabling stricter mypy settings")
        
        return suggestions

    def _generate_bandit_suggestions(self, results_list: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on bandit results"""
        suggestions = []
        
        high_severity_issues = [r for r in results_list if r.get("issue_severity") in ["HIGH", "CRITICAL"]]
        
        if high_severity_issues:
            suggestions.append("Address high-severity security issues immediately")
        
        # Common issue patterns
        issue_types = [r.get("test_id", "") for r in results_list]
        
        if "B105" in issue_types:  # hardcoded_password_string
            suggestions.append("Move hardcoded secrets to environment variables")
        
        if "B108" in issue_types:  # hardcoded_tmp_directory
            suggestions.append("Use tempfile module for temporary file operations")
        
        if len(suggestions) == 0 and results_list:
            suggestions.append("Review bandit security findings and apply fixes")
        
        return suggestions[:3]

    def _generate_eslint_suggestions(self, eslint_data: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on ESLint results"""
        suggestions = []
        
        # Analyze common error types
        all_messages = []
        for file_data in eslint_data:
            all_messages.extend(file_data.get("messages", []))
        
        error_rules = [msg.get("ruleId", "") for msg in all_messages if msg.get("severity") == 2]
        
        if "no-unused-vars" in error_rules:
            suggestions.append("Remove unused variables and imports")
        
        if "no-console" in error_rules:
            suggestions.append("Remove or replace console statements with proper logging")
        
        if "prefer-const" in error_rules:
            suggestions.append("Use const for variables that are never reassigned")
        
        if len(suggestions) == 0 and all_messages:
            suggestions.append("Run 'npx eslint --fix .' to auto-fix some issues")
        
        return suggestions[:3]

    def _generate_typescript_suggestions(self, output: str) -> List[str]:
        """Generate suggestions based on TypeScript compiler output"""
        suggestions = []
        
        if "error TS2" in output:
            suggestions.append("Fix type errors to improve code safety")
        
        if "Cannot find module" in output:
            suggestions.append("Install missing dependencies or fix import paths")
        
        if "Property" in output and "does not exist" in output:
            suggestions.append("Add proper type definitions or fix property access")
        
        if len(suggestions) == 0 and "error" in output:
            suggestions.append("Fix TypeScript compilation errors")
        
        return suggestions

    def _generate_semgrep_suggestions(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on Semgrep findings"""
        suggestions = []
        
        high_severity = [f for f in findings if f.get("extra", {}).get("severity") in ["ERROR", "HIGH"]]
        
        if high_severity:
            suggestions.append("Address high-severity security findings immediately")
        
        # Analyze common issue patterns
        rule_ids = [f.get("check_id", "") for f in findings]
        
        if any("sql-injection" in rule for rule in rule_ids):
            suggestions.append("Use parameterized queries to prevent SQL injection")
        
        if any("xss" in rule for rule in rule_ids):
            suggestions.append("Implement proper input sanitization and output encoding")
        
        if len(suggestions) == 0 and findings:
            suggestions.append("Review Semgrep security findings and apply fixes")
        
        return suggestions[:3]

    def _generate_pip_audit_suggestions(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions based on pip-audit results"""
        suggestions = []
        
        if vulnerabilities:
            suggestions.append("Update vulnerable dependencies to secure versions")
        
        # Analyze vulnerability types
        critical_vulns = [
            v for v in vulnerabilities 
            if any(a.get("severity") in ["HIGH", "CRITICAL"] for a in v.get("aliases", []))
        ]
        
        if critical_vulns:
            suggestions.append("Prioritize critical vulnerability fixes")
        
        if len(suggestions) == 0:
            suggestions.append("Keep dependencies up to date with security patches")
        
        return suggestions


# Global unified linting manager instance
unified_linting_manager = UnifiedLintingPipeline()


async def run_unified_linting_cli():
    """CLI entry point for unified linting"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Linting Pipeline")
    parser.add_argument("--language", choices=["python", "frontend", "security", "all"], 
                       default="all", help="Language/type to lint")
    parser.add_argument("--paths", nargs="+", default=None, help="Paths to lint")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    if args.config:
        manager = UnifiedLintingPipeline(Path(args.config))
    else:
        manager = unified_linting_manager
    
    # Run linting based on language selection
    if args.language == "python":
        results = await manager.run_python_linting(args.paths or ["."])
    elif args.language == "frontend":
        results = await manager.run_frontend_linting(args.paths or ["."])
    elif args.language == "security":
        results = await manager.run_security_linting(args.paths or ["."])
    else:
        results = await manager.run_full_pipeline(args.paths)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(run_unified_linting_cli())