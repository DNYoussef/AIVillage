"""
Security Gates System - Prompt 11

Comprehensive security validation framework for AIVillage components with
automated gate checks, policy enforcement, and integration validation.

Security Integration Point: All components pass security gates consistently
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import re
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class GateResult(Enum):
    """Security gate validation results."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class SecurityIssue:
    """Security issue found during validation."""

    severity: str
    category: str
    description: str
    location: str
    recommendation: str
    cve_refs: list[str] = field(default_factory=list)


@dataclass
class GateReport:
    """Security gate validation report."""

    gate_name: str
    result: GateResult
    execution_time: float
    issues: list[SecurityIssue] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.result == GateResult.PASS

    @property
    def critical_issues(self) -> list[SecurityIssue]:
        return [issue for issue in self.issues if issue.severity in ["CRITICAL", "HIGH"]]


class SecurityGate(ABC):
    """Abstract base class for security gates."""

    def __init__(self, name: str, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.name = name
        self.security_level = security_level
        self.enabled = True

    @abstractmethod
    def validate(self, context: dict[str, Any]) -> GateReport:
        """Validate security requirements."""
        pass

    def should_run(self, context: dict[str, Any]) -> bool:
        """Check if gate should run in current context."""
        return self.enabled


class HttpsEnforcementGate(SecurityGate):
    """Validates HTTPS enforcement in production configurations."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        super().__init__("HTTPS Enforcement", security_level)

    def validate(self, context: dict[str, Any]) -> GateReport:
        start_time = time.time()
        issues = []

        project_root = context.get("project_root", Path.cwd())

        # Check production source files for HTTP URLs
        src_dirs = [
            project_root / "src" / "production",
            project_root / "src" / "core",
        ]

        http_pattern = re.compile(r'http://[^\s\'"]+', re.IGNORECASE)

        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Skip test files and comments
                    if "test_" in py_file.name or py_file.name.endswith("_test.py"):
                        continue

                    matches = http_pattern.findall(content)
                    if matches:
                        for match in matches:
                            # Skip common safe patterns
                            if any(safe in match.lower() for safe in ["localhost", "127.0.0.1", "example.com"]):
                                continue

                            issues.append(
                                SecurityIssue(
                                    severity="MEDIUM",
                                    category="HTTPS_ENFORCEMENT",
                                    description=f"HTTP URL found in production code: {match}",
                                    location=str(py_file.relative_to(project_root)),
                                    recommendation="Replace with HTTPS URL or use secure configuration",
                                )
                            )

                except Exception as e:
                    logger.warning(f"Error scanning {py_file}: {e}")

        # Check configuration files
        config_files = [
            project_root / "config" / "production.yaml",
            project_root / "docker-compose.yml",
            project_root / "docker-compose.prod.yml",
        ]

        for config_file in config_files:
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        content = f.read()

                    if "http://" in content and "localhost" not in content:
                        issues.append(
                            SecurityIssue(
                                severity="HIGH",
                                category="HTTPS_ENFORCEMENT",
                                description=f"HTTP configuration in production file: {config_file.name}",
                                location=str(config_file.relative_to(project_root)),
                                recommendation="Use HTTPS for all production endpoints",
                            )
                        )

                except Exception as e:
                    logger.warning(f"Error scanning config {config_file}: {e}")

        execution_time = time.time() - start_time

        # Determine result
        if any(issue.severity == "HIGH" for issue in issues):
            result = GateResult.FAIL
        elif issues:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateReport(
            gate_name=self.name,
            result=result,
            execution_time=execution_time,
            issues=issues,
            metadata={"files_scanned": len(list((project_root / "src").rglob("*.py")))},
        )


class PickleSecurityGate(SecurityGate):
    """Validates that unsafe pickle usage has been eliminated."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        super().__init__("Pickle Security", security_level)

    def validate(self, context: dict[str, Any]) -> GateReport:
        start_time = time.time()
        issues = []

        project_root = context.get("project_root", Path.cwd())

        # Patterns to detect unsafe pickle usage
        unsafe_patterns = [
            (r"import\s+pickle", "Direct pickle import"),
            (r"from\s+pickle\s+import", "Pickle function import"),
            (r"pickle\.loads?\(", "Unsafe pickle load/loads call"),
            (r"pickle\.dumps?\(", "Direct pickle dump/dumps call"),
            (r"cPickle", "Legacy cPickle usage"),
        ]

        # Safe patterns that are allowed
        safe_patterns = [
            r"# SAFE: pickle replacement",
            r"SecureSerializer",
            r"secure_serializer",
        ]

        src_dirs = [project_root / "src"]

        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Check for safe patterns first
                    has_safe_pattern = any(re.search(pattern, content, re.IGNORECASE) for pattern in safe_patterns)

                    for pattern, description in unsafe_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip if file has safe patterns (secure replacement)
                            if has_safe_pattern:
                                continue

                            line_num = content[: match.start()].count("\n") + 1

                            issues.append(
                                SecurityIssue(
                                    severity="CRITICAL",
                                    category="UNSAFE_SERIALIZATION",
                                    description=f"{description} found at line {line_num}",
                                    location=f"{py_file.relative_to(project_root)}:{line_num}",
                                    recommendation="Replace with SecureSerializer from core.security.secure_serializer",
                                    cve_refs=["CVE-2022-42969", "CWE-502"],
                                )
                            )

                except Exception as e:
                    logger.warning(f"Error scanning {py_file}: {e}")

        execution_time = time.time() - start_time

        # Critical issues = fail
        result = GateResult.FAIL if issues else GateResult.PASS

        return GateReport(
            gate_name=self.name,
            result=result,
            execution_time=execution_time,
            issues=issues,
            metadata={"patterns_checked": len(unsafe_patterns)},
        )


class DependencySecurityGate(SecurityGate):
    """Validates dependencies for known security vulnerabilities."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        super().__init__("Dependency Security", security_level)

    def validate(self, context: dict[str, Any]) -> GateReport:
        start_time = time.time()
        issues = []

        project_root = context.get("project_root", Path.cwd())

        # Check for requirements files
        req_files = [
            project_root / "requirements.txt",
            project_root / "pyproject.toml",
            project_root / "Pipfile",
        ]

        found_files = [f for f in req_files if f.exists()]

        if not found_files:
            issues.append(
                SecurityIssue(
                    severity="MEDIUM",
                    category="DEPENDENCY_MANAGEMENT",
                    description="No dependency files found",
                    location="project root",
                    recommendation="Create requirements.txt or pyproject.toml",
                )
            )

        # Try to run safety check if available
        try:
            result = subprocess.run(
                ["safety", "check", "--json"], cwd=project_root, capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                # No vulnerabilities found
                pass
            else:
                # Parse safety output for vulnerabilities
                if result.stdout:
                    try:
                        import json

                        vulns = json.loads(result.stdout)
                        for vuln in vulns:
                            issues.append(
                                SecurityIssue(
                                    severity="HIGH",
                                    category="VULNERABLE_DEPENDENCY",
                                    description=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                                    location="dependencies",
                                    recommendation=f"Update to version {vuln.get('safe_versions', 'latest')}",
                                    cve_refs=[vuln.get("cve", "")],
                                )
                            )
                    except json.JSONDecodeError:
                        logger.warning("Could not parse safety check output")

        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Safety not available - add warning
            issues.append(
                SecurityIssue(
                    severity="LOW",
                    category="SECURITY_TOOLING",
                    description="Safety dependency checker not available",
                    location="tooling",
                    recommendation="Install safety: pip install safety",
                )
            )

        execution_time = time.time() - start_time

        # Determine result
        if any(issue.severity == "HIGH" for issue in issues):
            result = GateResult.FAIL
        elif issues:
            result = GateResult.WARN
        else:
            result = GateResult.PASS

        return GateReport(
            gate_name=self.name,
            result=result,
            execution_time=execution_time,
            issues=issues,
            metadata={"dependency_files": [str(f.name) for f in found_files]},
        )


class SecretScanningGate(SecurityGate):
    """Scans for hardcoded secrets and credentials."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        super().__init__("Secret Scanning", security_level)

        # Common secret patterns
        self.secret_patterns = [
            (r'password\s*=\s*["\']([^"\']{8,})["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\']([^"\']{16,})["\']', "API key"),
            (r'secret_key\s*=\s*["\']([^"\']{16,})["\']', "Secret key"),
            (r'token\s*=\s*["\']([^"\']{16,})["\']', "Access token"),
            (r"Bearer\s+([A-Za-z0-9\-_]+)", "Bearer token"),
            (r"[A-Za-z0-9]{20,}", "Generic secret-like string"),
        ]

    def validate(self, context: dict[str, Any]) -> GateReport:
        start_time = time.time()
        issues = []

        project_root = context.get("project_root", Path.cwd())

        # Patterns that indicate safe usage
        safe_indicators = [
            "test",
            "example",
            "placeholder",
            "dummy",
            "fake",
            "TODO",
            "REPLACE",
            "your_",
            "env.get",
            "os.environ",
        ]

        src_dirs = [project_root / "src", project_root / "scripts"]

        for src_dir in src_dirs:
            if not src_dir.exists():
                continue

            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    for pattern, description in self.secret_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            matched_text = match.group(1) if match.groups() else match.group(0)

                            # Skip if looks like safe placeholder
                            if any(safe in matched_text.lower() for safe in safe_indicators):
                                continue

                            # Skip very common/simple patterns
                            if len(matched_text) < 8 or matched_text.lower() in ["password", "secret"]:
                                continue

                            line_num = content[: match.start()].count("\n") + 1

                            issues.append(
                                SecurityIssue(
                                    severity="HIGH",
                                    category="HARDCODED_SECRET",
                                    description=f"{description} found at line {line_num}",
                                    location=f"{py_file.relative_to(project_root)}:{line_num}",
                                    recommendation="Use environment variables or secure configuration",
                                    cve_refs=["CWE-798"],
                                )
                            )

                except Exception as e:
                    logger.warning(f"Error scanning {py_file}: {e}")

        execution_time = time.time() - start_time

        # High severity issues = fail
        result = GateResult.FAIL if any(issue.severity == "HIGH" for issue in issues) else GateResult.PASS

        return GateReport(
            gate_name=self.name,
            result=result,
            execution_time=execution_time,
            issues=issues,
            metadata={"patterns_checked": len(self.secret_patterns)},
        )


class SecurityGateRunner:
    """Manages and executes security gates."""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.security_level = security_level
        self.gates = []
        self.reports = []

        # Register default gates
        self.register_gate(HttpsEnforcementGate(security_level))
        self.register_gate(PickleSecurityGate(security_level))
        self.register_gate(DependencySecurityGate(security_level))
        self.register_gate(SecretScanningGate(security_level))

    def register_gate(self, gate: SecurityGate):
        """Register a security gate."""
        self.gates.append(gate)

    def run_all_gates(self, context: dict[str, Any] = None) -> dict[str, Any]:
        """Run all registered security gates."""
        if context is None:
            context = {"project_root": Path.cwd()}

        print(f"\n=== Security Gates Validation - {self.security_level.value.title()} ===")

        self.reports = []
        passed = 0
        failed = 0
        warnings = 0

        for gate in self.gates:
            if not gate.should_run(context):
                continue

            print(f"\nRunning {gate.name}...")
            report = gate.validate(context)
            self.reports.append(report)

            if report.result == GateResult.PASS:
                passed += 1
                print(f"  [PASS] - {report.execution_time:.2f}s")
            elif report.result == GateResult.WARN:
                warnings += 1
                print(f"  [WARN] - {len(report.issues)} issues - {report.execution_time:.2f}s")
            elif report.result == GateResult.FAIL:
                failed += 1
                print(f"  [FAIL] - {len(report.critical_issues)} critical issues - {report.execution_time:.2f}s")

                # Show critical issues
                for issue in report.critical_issues[:3]:  # Show first 3
                    print(f"    - {issue.severity}: {issue.description}")

        # Summary
        total_gates = len([r for r in self.reports])
        all_passed = failed == 0

        print("\n=== Security Gates Summary ===")
        print(f"Gates run: {total_gates}")
        print(f"Passed: {passed}")
        print(f"Warnings: {warnings}")
        print(f"Failed: {failed}")
        print(f"Overall status: {'[PASS]' if all_passed else '[FAIL]'}")

        return {
            "total_gates": total_gates,
            "passed": passed,
            "warnings": warnings,
            "failed": failed,
            "all_passed": all_passed,
            "reports": self.reports,
            "security_level": self.security_level.value,
        }

    def get_security_summary(self) -> dict[str, Any]:
        """Get comprehensive security summary."""
        all_issues = []
        for report in self.reports:
            all_issues.extend(report.issues)

        # Group by severity
        issues_by_severity = {}
        for issue in all_issues:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)

        # Group by category
        issues_by_category = {}
        for issue in all_issues:
            if issue.category not in issues_by_category:
                issues_by_category[issue.category] = []
            issues_by_category[issue.category].append(issue)

        return {
            "total_issues": len(all_issues),
            "by_severity": {k: len(v) for k, v in issues_by_severity.items()},
            "by_category": {k: len(v) for k, v in issues_by_category.items()},
            "critical_issues": [issue for issue in all_issues if issue.severity == "CRITICAL"],
            "recommendations": list(set(issue.recommendation for issue in all_issues)),
        }


# Convenience functions
def run_security_gates(
    project_root: Path = None, security_level: SecurityLevel = SecurityLevel.PRODUCTION
) -> dict[str, Any]:
    """Run all security gates for a project."""
    context = {"project_root": project_root or Path.cwd()}
    runner = SecurityGateRunner(security_level)
    return runner.run_all_gates(context)


def validate_component_security(component_path: Path, security_level: SecurityLevel = SecurityLevel.PRODUCTION) -> bool:
    """Validate security for a specific component."""
    context = {"project_root": component_path.parent, "component_path": component_path}
    runner = SecurityGateRunner(security_level)
    results = runner.run_all_gates(context)
    return results["all_passed"]
