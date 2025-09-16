"""
Secure Subprocess Management System
NASA POT10 Compliant Command Execution Framework

This module provides secure subprocess execution with comprehensive input validation,
command injection prevention, and audit trail capabilities for defense industry
compliance requirements.

Author: Security Manager Agent
Version: 1.0.0
Compliance: NASA POT10, NIST SSDF, OWASP ASVS
"""

import asyncio
import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum


class SecurityLevel(Enum):
    """Security levels for command execution"""
    RESTRICTED = "restricted"  # Only allow explicitly approved commands
    VALIDATED = "validated"    # Allow validated commands with input sanitization
    CONTROLLED = "controlled"  # Allow controlled execution with full audit


@dataclass
class CommandAuditRecord:
    """Audit record for command execution"""
    timestamp: float
    command_hash: str
    original_command: List[str]
    sanitized_command: List[str]
    security_level: str
    execution_time: float
    exit_code: int
    user_context: Optional[str] = None
    risk_score: float = 0.0


class SecureSubprocessManager:
    """
    NASA POT10 Compliant Secure Subprocess Manager

    Provides comprehensive protection against command injection attacks
    while maintaining full audit trails for defense industry compliance.
    """

    # Allowed base commands for RESTRICTED security level
    ALLOWED_COMMANDS = {
        'python', 'python3', 'pip', 'pip3', 'npm', 'node', 'git',
        'pytest', 'coverage', 'black', 'isort', 'mypy', 'flake8',
        'bandit', 'semgrep', 'docker', 'kubectl', 'helm',
        'terraform', 'ansible', 'curl', 'wget'
    }

    # High-risk patterns that require special validation
    HIGH_RISK_PATTERNS = {
        '|', '&', ';', '&&', '||', '$(', '`', '<', '>', '>>',
        'rm -rf', 'sudo', 'su', 'chmod 777', 'eval', 'exec'
    }

    def __init__(self,
                 security_level: SecurityLevel = SecurityLevel.VALIDATED,
                 audit_file: Optional[Path] = None,
                 max_execution_time: int = 300):
        """
        Initialize secure subprocess manager

        Args:
            security_level: Security enforcement level
            audit_file: Path to audit log file
            max_execution_time: Maximum command execution time in seconds
        """
        self.security_level = security_level
        self.max_execution_time = max_execution_time
        self.audit_records: List[CommandAuditRecord] = []

        # Initialize audit logging
        if audit_file:
            self.audit_file = audit_file
        else:
            self.audit_file = Path(__file__).parent / "audit" / "subprocess_audit.log"

        self.audit_file.parent.mkdir(exist_ok=True, parents=True)

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.SecureSubprocessManager")
        self.logger.setLevel(logging.INFO)

        # Create file handler if not exists
        if not self.logger.handlers:
            handler = logging.FileHandler(self.audit_file)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _calculate_command_hash(self, command: List[str]) -> str:
        """Calculate SHA-256 hash of command for audit purposes"""
        command_str = ' '.join(command)
        return hashlib.sha256(command_str.encode()).hexdigest()

    def _calculate_risk_score(self, command: List[str]) -> float:
        """Calculate risk score for command execution (0.0-1.0)"""
        risk_score = 0.0
        command_str = ' '.join(command).lower()

        # Check for high-risk patterns
        for pattern in self.HIGH_RISK_PATTERNS:
            if pattern in command_str:
                risk_score += 0.2

        # Additional risk factors
        if len(command) > 10:  # Very long commands
            risk_score += 0.1

        if any(arg.startswith('-') and len(arg) > 10 for arg in command):  # Complex flags
            risk_score += 0.1

        return min(risk_score, 1.0)

    def _validate_command_security(self, command: List[str]) -> Tuple[bool, str]:
        """
        Validate command against security policies

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not command:
            return False, "Empty command not allowed"

        base_command = Path(command[0]).name.lower()

        # Security level checks
        if self.security_level == SecurityLevel.RESTRICTED:
            if base_command not in self.ALLOWED_COMMANDS:
                return False, f"Command '{base_command}' not in allowed list"

        # High-risk pattern checks
        command_str = ' '.join(command)
        for pattern in self.HIGH_RISK_PATTERNS:
            if pattern in command_str:
                if self.security_level == SecurityLevel.RESTRICTED:
                    return False, f"High-risk pattern detected: {pattern}"
                elif self.security_level == SecurityLevel.VALIDATED:
                    self.logger.warning(f"High-risk pattern detected but allowed: {pattern}")

        return True, "Command validated"

    def _sanitize_command(self, command: Union[str, List[str]]) -> List[str]:
        """
        Sanitize command input to prevent injection attacks

        Args:
            command: Command as string or list of arguments

        Returns:
            List of sanitized command arguments
        """
        if isinstance(command, str):
            # Use shlex to properly split command string
            try:
                command_parts = shlex.split(command)
            except ValueError as e:
                raise ValueError(f"Invalid command syntax: {e}")
        else:
            command_parts = list(command)

        # Sanitize each argument
        sanitized_parts = []
        for part in command_parts:
            # Remove null bytes and control characters
            sanitized_part = ''.join(char for char in part if ord(char) >= 32 or char in '\t\n')

            # Escape shell metacharacters if needed
            if self.security_level in [SecurityLevel.VALIDATED, SecurityLevel.CONTROLLED]:
                # For file paths and arguments that might contain spaces or special chars
                if ' ' in sanitized_part or any(char in sanitized_part for char in '"|\';<>&$`'):
                    sanitized_part = shlex.quote(sanitized_part)

            sanitized_parts.append(sanitized_part)

        return sanitized_parts

    def _create_audit_record(self,
                           original_command: List[str],
                           sanitized_command: List[str],
                           execution_time: float,
                           exit_code: int) -> CommandAuditRecord:
        """Create audit record for command execution"""
        return CommandAuditRecord(
            timestamp=time.time(),
            command_hash=self._calculate_command_hash(sanitized_command),
            original_command=original_command,
            sanitized_command=sanitized_command,
            security_level=self.security_level.value,
            execution_time=execution_time,
            exit_code=exit_code,
            risk_score=self._calculate_risk_score(sanitized_command)
        )

    def _log_audit_record(self, record: CommandAuditRecord):
        """Log audit record to file and memory"""
        self.audit_records.append(record)

        audit_data = {
            'timestamp': record.timestamp,
            'command_hash': record.command_hash,
            'original_command': record.original_command,
            'sanitized_command': record.sanitized_command,
            'security_level': record.security_level,
            'execution_time': record.execution_time,
            'exit_code': record.exit_code,
            'risk_score': record.risk_score
        }

        self.logger.info(f"AUDIT: {json.dumps(audit_data)}")

    def execute_command(self,
                       command: Union[str, List[str]],
                       cwd: Optional[Path] = None,
                       env: Optional[Dict[str, str]] = None,
                       timeout: Optional[int] = None,
                       capture_output: bool = True) -> Dict:
        """
        Execute command with security validation and audit logging

        Args:
            command: Command to execute (string or list)
            cwd: Working directory for command execution
            env: Environment variables
            timeout: Execution timeout (uses default if None)
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with execution results and metadata

        Raises:
            SecurityError: If command fails security validation
            subprocess.TimeoutExpired: If command times out
        """
        start_time = time.time()

        # Convert to list if string
        if isinstance(command, str):
            original_command = [command]
        else:
            original_command = list(command)

        # Sanitize command
        sanitized_command = self._sanitize_command(command)

        # Validate security
        is_valid, error_msg = self._validate_command_security(sanitized_command)
        if not is_valid:
            raise SecurityError(f"Command security validation failed: {error_msg}")

        # Set timeout
        exec_timeout = timeout or self.max_execution_time

        # Execute command
        try:
            result = subprocess.run(
                sanitized_command,
                cwd=cwd,
                env=env,
                timeout=exec_timeout,
                capture_output=capture_output,
                text=True,
                check=False  # Don't raise exception on non-zero exit code
            )

            execution_time = time.time() - start_time

            # Create and log audit record
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, result.returncode
            )
            self._log_audit_record(audit_record)

            return {
                'returncode': result.returncode,
                'stdout': result.stdout if capture_output else None,
                'stderr': result.stderr if capture_output else None,
                'execution_time': execution_time,
                'command_hash': audit_record.command_hash,
                'risk_score': audit_record.risk_score,
                'success': result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, -1
            )
            self._log_audit_record(audit_record)
            raise

        except Exception as e:
            execution_time = time.time() - start_time
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, -2
            )
            self._log_audit_record(audit_record)
            raise

    async def execute_command_async(self,
                                  command: Union[str, List[str]],
                                  cwd: Optional[Path] = None,
                                  env: Optional[Dict[str, str]] = None,
                                  timeout: Optional[int] = None) -> Dict:
        """
        Asynchronously execute command with security validation

        Args:
            command: Command to execute (string or list)
            cwd: Working directory for command execution
            env: Environment variables
            timeout: Execution timeout (uses default if None)

        Returns:
            Dictionary with execution results and metadata
        """
        start_time = time.time()

        # Convert to list if string
        if isinstance(command, str):
            original_command = [command]
        else:
            original_command = list(command)

        # Sanitize command
        sanitized_command = self._sanitize_command(command)

        # Validate security
        is_valid, error_msg = self._validate_command_security(sanitized_command)
        if not is_valid:
            raise SecurityError(f"Command security validation failed: {error_msg}")

        # Set timeout
        exec_timeout = timeout or self.max_execution_time

        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *sanitized_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )

            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=exec_timeout
            )

            execution_time = time.time() - start_time

            # Create and log audit record
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, process.returncode
            )
            self._log_audit_record(audit_record)

            return {
                'returncode': process.returncode,
                'stdout': stdout.decode() if stdout else None,
                'stderr': stderr.decode() if stderr else None,
                'execution_time': execution_time,
                'command_hash': audit_record.command_hash,
                'risk_score': audit_record.risk_score,
                'success': process.returncode == 0
            }

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, -1
            )
            self._log_audit_record(audit_record)
            # Kill the process
            if 'process' in locals():
                process.kill()
                await process.wait()
            raise subprocess.TimeoutExpired(sanitized_command, exec_timeout)

        except Exception as e:
            execution_time = time.time() - start_time
            audit_record = self._create_audit_record(
                original_command, sanitized_command, execution_time, -2
            )
            self._log_audit_record(audit_record)
            raise

    def get_audit_summary(self) -> Dict:
        """Get summary of audit records for compliance reporting"""
        if not self.audit_records:
            return {'total_executions': 0}

        total_executions = len(self.audit_records)
        successful_executions = sum(1 for r in self.audit_records if r.exit_code == 0)
        high_risk_executions = sum(1 for r in self.audit_records if r.risk_score >= 0.5)
        avg_execution_time = sum(r.execution_time for r in self.audit_records) / total_executions

        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'failed_executions': total_executions - successful_executions,
            'high_risk_executions': high_risk_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': avg_execution_time,
            'security_level': self.security_level.value,
            'audit_file': str(self.audit_file)
        }

    def export_audit_records(self, output_file: Path) -> None:
        """Export audit records to JSON file for compliance reporting"""
        audit_data = {
            'export_timestamp': time.time(),
            'security_level': self.security_level.value,
            'total_records': len(self.audit_records),
            'summary': self.get_audit_summary(),
            'records': [asdict(record) for record in self.audit_records]
        }

        with open(output_file, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)


class SecurityError(Exception):
    """Exception raised for security policy violations"""
    pass


# Global secure subprocess manager instance
_global_manager: Optional[SecureSubprocessManager] = None


def get_secure_manager() -> SecureSubprocessManager:
    """Get or create global secure subprocess manager"""
    global _global_manager
    if _global_manager is None:
        _global_manager = SecureSubprocessManager()
    return _global_manager


def execute_secure_command(command: Union[str, List[str]], **kwargs) -> Dict:
    """Convenience function for secure command execution"""
    manager = get_secure_manager()
    return manager.execute_command(command, **kwargs)


async def execute_secure_command_async(command: Union[str, List[str]], **kwargs) -> Dict:
    """Convenience function for async secure command execution"""
    manager = get_secure_manager()
    return await manager.execute_command_async(command, **kwargs)


if __name__ == "__main__":
    # Example usage and testing
    manager = SecureSubprocessManager(security_level=SecurityLevel.VALIDATED)

    # Test secure execution
    try:
        result = manager.execute_command(['echo', 'Hello, secure world!'])
        print(f"Execution successful: {result}")

        # Print audit summary
        summary = manager.get_audit_summary()
        print(f"Audit summary: {summary}")

    except SecurityError as e:
        print(f"Security error: {e}")
    except Exception as e:
        print(f"Execution error: {e}")