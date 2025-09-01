#!/usr/bin/env python3
"""
WebSocket Security Audit Script

Comprehensive security audit tool that:
1. Scans all WebSocket implementations for RCE vulnerabilities
2. Validates security fixes are in place
3. Tests for additional security issues
4. Generates detailed security report
"""

import re
import json
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security finding."""

    file_path: str
    line_number: int
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    vulnerability_type: str
    description: str
    code_snippet: str
    recommendation: str


@dataclass
class SecurityReport:
    """Complete security audit report."""

    total_files_scanned: int
    websocket_files_found: int
    critical_findings: List[SecurityFinding]
    high_findings: List[SecurityFinding]
    medium_findings: List[SecurityFinding]
    low_findings: List[SecurityFinding]
    fixes_validated: List[str]
    scan_timestamp: datetime


class WebSocketSecurityAuditor:
    """Comprehensive WebSocket security auditor."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.findings: List[SecurityFinding] = []
        self.fixes_validated: List[str] = []

        # Dangerous patterns to search for
        self.dangerous_patterns = {
            "eval_usage": {
                "pattern": r"eval\s*\(",
                "severity": "CRITICAL",
                "type": "Remote Code Execution",
                "description": "Use of eval() function can lead to RCE",
                "recommendation": "Replace with json.loads() or ast.literal_eval()",
            },
            "exec_usage": {
                "pattern": r"exec\s*\(",
                "severity": "CRITICAL",
                "type": "Remote Code Execution",
                "description": "Use of exec() function can lead to RCE",
                "recommendation": "Remove exec() usage or replace with safe alternatives",
            },
            "subprocess_direct": {
                "pattern": r"subprocess\.(call|check_output|run|Popen)\s*\(",
                "severity": "HIGH",
                "type": "Command Injection",
                "description": "Direct subprocess usage without input validation",
                "recommendation": "Add input validation and use shell=False",
            },
            "os_system": {
                "pattern": r"os\.system\s*\(",
                "severity": "CRITICAL",
                "type": "Command Injection",
                "description": "Use of os.system() can lead to command injection",
                "recommendation": "Replace with subprocess with proper validation",
            },
            "import_builtin": {
                "pattern": r"__import__\s*\(",
                "severity": "HIGH",
                "type": "Dynamic Import",
                "description": "Dynamic imports can be dangerous with user input",
                "recommendation": "Use static imports or validate input thoroughly",
            },
            "compile_usage": {
                "pattern": r"compile\s*\(",
                "severity": "HIGH",
                "type": "Code Compilation",
                "description": "Dynamic code compilation can be dangerous",
                "recommendation": "Avoid dynamic compilation or validate thoroughly",
            },
        }

        # WebSocket-specific patterns
        self.websocket_patterns = {
            "websocket_receive": r"websocket\.receive_text\(\)",
            "websocket_handler": r"async def.*websocket.*\(.*WebSocket",
            "message_processing": r"message\s*=.*data",
        }

        # Safe patterns we expect to see
        self.safe_patterns = {
            "json_loads": r"json\.loads\s*\(",
            "ast_literal_eval": r"ast\.literal_eval\s*\(",
            "input_validation": r"validate.*message|message.*validation",
            "security_check": r"security|validate|sanitize",
        }

    def scan_project(self) -> SecurityReport:
        """Perform comprehensive security scan of the project."""
        logger.info(f"Starting security audit of {self.project_root}")

        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        websocket_files = []

        logger.info(f"Found {len(python_files)} Python files to scan")

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check if this is a WebSocket-related file
                if self._is_websocket_file(content):
                    websocket_files.append(file_path)
                    logger.info(f"Scanning WebSocket file: {file_path}")

                # Scan for security issues
                self._scan_file(file_path, content)

            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")

        logger.info(f"Found {len(websocket_files)} WebSocket-related files")

        # Validate specific fixes
        self._validate_rce_fixes()

        # Generate report
        report = self._generate_report(len(python_files), len(websocket_files))

        return report

    def _is_websocket_file(self, content: str) -> bool:
        """Check if file contains WebSocket-related code."""
        websocket_indicators = [
            "WebSocket",
            "websocket",
            "ws:",
            "receive_text",
            "send_json",
            "WebSocketDisconnect",
            "websockets.",
            "fastapi.*WebSocket",
        ]

        return any(re.search(indicator, content, re.IGNORECASE) for indicator in websocket_indicators)

    def _scan_file(self, file_path: Path, content: str) -> None:
        """Scan individual file for security issues."""
        lines = content.split("\n")

        for pattern_name, pattern_info in self.dangerous_patterns.items():
            pattern = pattern_info["pattern"]

            for line_num, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    # Check if this is a false positive (e.g., model.eval())
                    if self._is_false_positive(line, pattern_name):
                        continue

                    finding = SecurityFinding(
                        file_path=str(file_path),
                        line_number=line_num,
                        severity=pattern_info["severity"],
                        vulnerability_type=pattern_info["type"],
                        description=pattern_info["description"],
                        code_snippet=line.strip(),
                        recommendation=pattern_info["recommendation"],
                    )

                    self.findings.append(finding)

                    logger.warning(f"SECURITY FINDING: {pattern_info['severity']} - {file_path}:{line_num}")

    def _is_false_positive(self, line: str, pattern_name: str) -> bool:
        """Check if detection is a false positive."""
        false_positive_patterns = {
            "eval_usage": [
                r"model\.eval\(\)",  # PyTorch model evaluation
                r"\.eval\(\)",  # General method calls
                r"#.*eval\(",  # Comments
                r"\".*eval\(",  # String literals
                r"'.*eval\(",  # String literals
            ]
        }

        if pattern_name in false_positive_patterns:
            for fp_pattern in false_positive_patterns[pattern_name]:
                if re.search(fp_pattern, line):
                    return True

        return False

    def _validate_rce_fixes(self) -> None:
        """Validate that specific RCE fixes are in place."""
        logger.info("Validating RCE vulnerability fixes...")

        # Check the specific files we fixed
        files_to_check = [
            "infrastructure/gateway/unified_api_gateway.py",
            "infrastructure/gateway/enhanced_unified_api_gateway.py",
        ]

        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                self._validate_websocket_fix(full_path)

    def _validate_websocket_fix(self, file_path: Path) -> None:
        """Validate that WebSocket handler is properly secured."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Check that eval() is NOT used with websocket data
            eval_with_data_pattern = r"eval\s*\(\s*data\s*\)"
            if re.search(eval_with_data_pattern, content):
                finding = SecurityFinding(
                    file_path=str(file_path),
                    line_number=0,
                    severity="CRITICAL",
                    vulnerability_type="RCE Vulnerability Not Fixed",
                    description="WebSocket still uses eval() with user data",
                    code_snippet="eval(data)",
                    recommendation="Replace with json.loads(data)",
                )
                self.findings.append(finding)
                logger.error(f"RCE FIX VALIDATION FAILED: {file_path}")
                return

            # Check that json.loads is used instead
            if re.search(r"json\.loads\s*\(\s*data\s*\)", content):
                self.fixes_validated.append(f"WebSocket RCE fix validated in {file_path}")
                logger.info(f"RCE fix validation PASSED: {file_path}")

            # Check for input validation
            if re.search(r"allowed_types|validate.*message|security.*check", content, re.IGNORECASE):
                self.fixes_validated.append(f"Input validation found in {file_path}")
                logger.info(f"Input validation found in {file_path}")

            # Check for error handling
            if re.search(r"JSONDecodeError|except.*json", content):
                self.fixes_validated.append(f"JSON error handling found in {file_path}")
                logger.info(f"JSON error handling validated in {file_path}")

        except Exception as e:
            logger.error(f"Could not validate {file_path}: {e}")

    def _generate_report(self, total_files: int, websocket_files: int) -> SecurityReport:
        """Generate comprehensive security report."""
        critical_findings = [f for f in self.findings if f.severity == "CRITICAL"]
        high_findings = [f for f in self.findings if f.severity == "HIGH"]
        medium_findings = [f for f in self.findings if f.severity == "MEDIUM"]
        low_findings = [f for f in self.findings if f.severity == "LOW"]

        report = SecurityReport(
            total_files_scanned=total_files,
            websocket_files_found=websocket_files,
            critical_findings=critical_findings,
            high_findings=high_findings,
            medium_findings=medium_findings,
            low_findings=low_findings,
            fixes_validated=self.fixes_validated,
            scan_timestamp=datetime.now(),
        )

        return report

    def save_report(self, report: SecurityReport, output_file: str) -> None:
        """Save security report to file."""
        report_data = {
            "scan_summary": {
                "timestamp": report.scan_timestamp.isoformat(),
                "total_files_scanned": report.total_files_scanned,
                "websocket_files_found": report.websocket_files_found,
                "total_findings": len(report.critical_findings)
                + len(report.high_findings)
                + len(report.medium_findings)
                + len(report.low_findings),
                "critical_count": len(report.critical_findings),
                "high_count": len(report.high_findings),
                "medium_count": len(report.medium_findings),
                "low_count": len(report.low_findings),
                "fixes_validated_count": len(report.fixes_validated),
            },
            "critical_findings": [
                {
                    "file": f.file_path,
                    "line": f.line_number,
                    "type": f.vulnerability_type,
                    "description": f.description,
                    "code": f.code_snippet,
                    "recommendation": f.recommendation,
                }
                for f in report.critical_findings
            ],
            "high_findings": [
                {
                    "file": f.file_path,
                    "line": f.line_number,
                    "type": f.vulnerability_type,
                    "description": f.description,
                    "code": f.code_snippet,
                    "recommendation": f.recommendation,
                }
                for f in report.high_findings
            ],
            "fixes_validated": report.fixes_validated,
        }

        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Security report saved to {output_file}")

    def print_summary(self, report: SecurityReport) -> None:
        """Print security report summary."""
        print("\n" + "=" * 80)
        print("WEBSOCKET SECURITY AUDIT REPORT")
        print("=" * 80)
        print(f"Scan Date: {report.scan_timestamp}")
        print(f"Files Scanned: {report.total_files_scanned}")
        print(f"WebSocket Files: {report.websocket_files_found}")
        print("\nFINDINGS SUMMARY:")
        print(f"  🔴 CRITICAL: {len(report.critical_findings)}")
        print(f"  🟠 HIGH:     {len(report.high_findings)}")
        print(f"  🟡 MEDIUM:   {len(report.medium_findings)}")
        print(f"  🟢 LOW:      {len(report.low_findings)}")

        print(f"\n✅ FIXES VALIDATED: {len(report.fixes_validated)}")
        for fix in report.fixes_validated:
            print(f"  - {fix}")

        if report.critical_findings:
            print("\n🚨 CRITICAL VULNERABILITIES FOUND:")
            for finding in report.critical_findings[:5]:  # Show first 5
                print(f"  - {finding.file_path}:{finding.line_number}")
                print(f"    {finding.vulnerability_type}: {finding.description}")
                print(f"    Code: {finding.code_snippet}")
                print(f"    Fix: {finding.recommendation}")
                print()

        if len(report.critical_findings) == 0 and len(report.high_findings) == 0:
            print("\n🎉 NO CRITICAL OR HIGH SEVERITY VULNERABILITIES FOUND!")

        print("=" * 80)


def main():
    """Main function to run security audit."""
    import argparse

    parser = argparse.ArgumentParser(description="WebSocket Security Auditor")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="websocket_security_report.json", help="Output file for detailed report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run security audit
    auditor = WebSocketSecurityAuditor(args.project_root)
    report = auditor.scan_project()

    # Save and display report
    auditor.save_report(report, args.output)
    auditor.print_summary(report)

    # Exit with error code if critical vulnerabilities found
    if len(report.critical_findings) > 0:
        print("\n❌ CRITICAL VULNERABILITIES FOUND - IMMEDIATE ACTION REQUIRED!")
        return 1
    elif len(report.high_findings) > 0:
        print("\n⚠️  HIGH SEVERITY ISSUES FOUND - REVIEW RECOMMENDED")
        return 2
    else:
        print("\n✅ SECURITY AUDIT PASSED")
        return 0


if __name__ == "__main__":
    exit(main())
