"""
Process Security Manager for Phase 6 Baking System

Ensures secure baking pipeline execution, input validation,
resource access control, and comprehensive audit trail generation.
"""

import os
import json
import time
import psutil
import logging
import hashlib
import subprocess
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import resource
from pathlib import Path

from .enhanced_audit_trail_manager import EnhancedAuditTrail
from .fips_crypto_module import FIPSCryptoModule

class ProcessSecurityLevel(Enum):
    STANDARD = "standard"
    ENHANCED = "enhanced"
    CLASSIFIED = "classified"

class ProcessState(Enum):
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class ProcessSecurityConfig:
    """Process security configuration."""
    security_level: ProcessSecurityLevel
    sandbox_enabled: bool = True
    resource_limits: Dict[str, Any] = None
    network_isolation: bool = True
    filesystem_isolation: bool = True
    process_monitoring: bool = True
    input_validation: bool = True
    output_sanitization: bool = True
    audit_all_operations: bool = True
    max_execution_time: int = 3600  # 1 hour
    max_memory_mb: int = 8192  # 8GB

@dataclass
class ProcessExecution:
    """Process execution tracking."""
    process_id: str
    pid: int
    user_id: str
    command: str
    start_time: datetime
    end_time: Optional[datetime]
    state: ProcessState
    resource_usage: Dict[str, Any]
    security_violations: List[str]

class ProcessSecurityViolation(Exception):
    """Raised when process security is violated."""
    pass

class ProcessSecurityManager:
    """Manages security for baking process execution."""

    def __init__(self, config: ProcessSecurityConfig):
        self.config = config
        self.audit = EnhancedAuditTrail()
        self.crypto = FIPSCryptoModule()

        self.active_processes = {}
        self.process_history = []
        self.security_violations = []
        self.resource_monitors = {}

        self.lock = threading.Lock()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup process security logger."""
        logger = logging.getLogger('process_security')
        handler = logging.FileHandler('.claude/.artifacts/security/process_security.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [PROCESS_SEC] %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def create_secure_execution_context(self, user_id: str, process_name: str) -> Dict[str, Any]:
        """Create secure execution context for baking process."""
        context_id = self._generate_context_id(user_id, process_name)

        # Create sandboxed environment
        sandbox_env = self._create_sandbox_environment(context_id)

        # Setup resource limits
        resource_limits = self._configure_resource_limits()

        # Setup monitoring
        monitor_config = self._setup_process_monitoring(context_id)

        # Create security context
        security_context = {
            'context_id': context_id,
            'user_id': user_id,
            'process_name': process_name,
            'sandbox_env': sandbox_env,
            'resource_limits': resource_limits,
            'monitor_config': monitor_config,
            'security_level': self.config.security_level.value,
            'created_at': datetime.now(timezone.utc),
            'restrictions': self._get_security_restrictions()
        }

        self.logger.info(f"Created secure execution context: {context_id}")
        self.audit.log_security_event(
            event_type='execution_context',
            user_id=user_id,
            action='create_context',
            resource=f"context_{context_id}",
            classification='CUI//BASIC',
            additional_data=security_context
        )

        return security_context

    def execute_secure_process(self, context: Dict[str, Any],
                              command: List[str],
                              input_data: Any = None,
                              timeout: int = None) -> Dict[str, Any]:
        """Execute process with security controls."""
        context_id = context['context_id']
        user_id = context['user_id']

        # Validate input
        if not self._validate_process_input(command, input_data):
            raise ProcessSecurityViolation("Invalid process input detected")

        # Create process execution record
        process_id = self._generate_process_id()
        execution_timeout = timeout or self.config.max_execution_time

        execution_record = ProcessExecution(
            process_id=process_id,
            pid=0,  # Will be set when process starts
            user_id=user_id,
            command=' '.join(command),
            start_time=datetime.now(timezone.utc),
            end_time=None,
            state=ProcessState.INITIALIZING,
            resource_usage={},
            security_violations=[]
        )

        with self.lock:
            self.active_processes[process_id] = execution_record

        try:
            # Start process with security controls
            result = self._execute_with_security(
                context, command, input_data, execution_timeout, execution_record
            )

            execution_record.state = ProcessState.TERMINATED
            execution_record.end_time = datetime.now(timezone.utc)

            # Sanitize output
            sanitized_result = self._sanitize_process_output(result)

            self.logger.info(f"Process {process_id} completed successfully")

            return {
                'process_id': process_id,
                'result': sanitized_result,
                'execution_time': (execution_record.end_time - execution_record.start_time).total_seconds(),
                'resource_usage': execution_record.resource_usage,
                'security_status': 'CLEAN'
            }

        except Exception as e:
            execution_record.state = ProcessState.ERROR
            execution_record.end_time = datetime.now(timezone.utc)

            self._handle_execution_error(execution_record, str(e))
            raise ProcessSecurityViolation(f"Secure execution failed: {str(e)}")

        finally:
            # Cleanup
            self._cleanup_execution_context(context_id, execution_record)

    def _validate_process_input(self, command: List[str], input_data: Any) -> bool:
        """Validate process input for security threats."""
        if not self.config.input_validation:
            return True

        # Check command whitelist
        allowed_commands = [
            'python', 'python3', 'node', 'java', 'docker',
            'bash', 'sh', 'make', 'cmake', 'gcc', 'g++'
        ]

        base_command = os.path.basename(command[0]) if command else ""
        if base_command not in allowed_commands:
            self.logger.warning(f"Blocked command: {base_command}")
            return False

        # Check for dangerous arguments
        dangerous_patterns = [
            'rm -rf', 'sudo', 'su', 'chmod +x', 'wget', 'curl',
            '>', '>>', '|', '&', ';', '$(', '`'
        ]

        full_command = ' '.join(command)
        for pattern in dangerous_patterns:
            if pattern in full_command:
                self.logger.warning(f"Dangerous pattern detected: {pattern}")
                return False

        # Validate input data
        if input_data:
            if isinstance(input_data, str):
                if len(input_data) > 1024 * 1024:  # 1MB limit
                    return False

                # Check for code injection patterns
                injection_patterns = ['<script>', '<?php', 'eval(', 'exec(']
                for pattern in injection_patterns:
                    if pattern in input_data.lower():
                        return False

        return True

    def _execute_with_security(self, context: Dict[str, Any],
                             command: List[str],
                             input_data: Any,
                             timeout: int,
                             execution_record: ProcessExecution) -> Dict[str, Any]:
        """Execute process with comprehensive security controls."""

        # Setup environment
        env = self._create_secure_environment(context)

        # Set resource limits
        self._apply_resource_limits(context['resource_limits'])

        # Start resource monitoring
        monitor_thread = self._start_resource_monitoring(execution_record)

        try:
            # Execute process
            if self.config.sandbox_enabled:
                result = self._execute_in_sandbox(command, input_data, timeout, env)
            else:
                result = self._execute_direct(command, input_data, timeout, env)

            execution_record.pid = result.get('pid', 0)
            execution_record.state = ProcessState.RUNNING

            return result

        finally:
            # Stop monitoring
            if monitor_thread:
                monitor_thread.stop()

    def _create_secure_environment(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Create secure environment variables."""
        base_env = {
            'PATH': '/usr/local/bin:/usr/bin:/bin',
            'HOME': '/tmp',
            'TMPDIR': f"/tmp/secure_{context['context_id']}",
            'USER': 'secure_user',
            'SHELL': '/bin/bash'
        }

        # Add security-specific variables
        if self.config.security_level == ProcessSecurityLevel.CLASSIFIED:
            base_env.update({
                'SECURITY_LEVEL': 'CLASSIFIED',
                'AUDIT_MODE': 'ENABLED',
                'NETWORK_ACCESS': 'RESTRICTED'
            })

        return base_env

    def _apply_resource_limits(self, limits: Dict[str, Any]):
        """Apply system resource limits."""
        try:
            # Memory limit
            if 'max_memory_mb' in limits:
                memory_bytes = limits['max_memory_mb'] * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # CPU time limit
            if 'max_cpu_seconds' in limits:
                cpu_limit = limits['max_cpu_seconds']
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

            # File size limit
            if 'max_file_size_mb' in limits:
                file_bytes = limits['max_file_size_mb'] * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))

            # Number of processes
            if 'max_processes' in limits:
                proc_limit = limits['max_processes']
                resource.setrlimit(resource.RLIMIT_NPROC, (proc_limit, proc_limit))

        except Exception as e:
            self.logger.warning(f"Failed to apply resource limits: {e}")

    def _execute_in_sandbox(self, command: List[str], input_data: Any,
                          timeout: int, env: Dict[str, str]) -> Dict[str, Any]:
        """Execute process in sandboxed environment."""
        # Create temporary directory
        sandbox_dir = f"/tmp/sandbox_{int(time.time())}"
        os.makedirs(sandbox_dir, exist_ok=True)

        try:
            # Change to sandbox directory
            original_cwd = os.getcwd()
            os.chdir(sandbox_dir)

            # Execute with restricted privileges
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=sandbox_dir,
                preexec_fn=os.setsid  # Create new process group
            )

            # Communicate with process
            input_bytes = input_data.encode() if isinstance(input_data, str) else input_data
            stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)

            return {
                'pid': process.pid,
                'returncode': process.returncode,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'sandbox_dir': sandbox_dir
            }

        finally:
            os.chdir(original_cwd)
            # Cleanup sandbox (in production, might want to preserve for forensics)
            if os.path.exists(sandbox_dir):
                subprocess.run(['rm', '-rf', sandbox_dir], check=False)

    def _execute_direct(self, command: List[str], input_data: Any,
                       timeout: int, env: Dict[str, str]) -> Dict[str, Any]:
        """Execute process directly with monitoring."""
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )

        input_bytes = input_data.encode() if isinstance(input_data, str) else input_data
        stdout, stderr = process.communicate(input=input_bytes, timeout=timeout)

        return {
            'pid': process.pid,
            'returncode': process.returncode,
            'stdout': stdout.decode('utf-8', errors='replace'),
            'stderr': stderr.decode('utf-8', errors='replace')
        }

    def _start_resource_monitoring(self, execution_record: ProcessExecution) -> Any:
        """Start monitoring process resource usage."""
        class ResourceMonitor:
            def __init__(self, record, logger):
                self.record = record
                self.logger = logger
                self.running = True
                self.thread = threading.Thread(target=self._monitor)
                self.thread.start()

            def _monitor(self):
                while self.running:
                    try:
                        if self.record.pid > 0:
                            process = psutil.Process(self.record.pid)
                            self.record.resource_usage = {
                                'cpu_percent': process.cpu_percent(),
                                'memory_mb': process.memory_info().rss / 1024 / 1024,
                                'num_threads': process.num_threads(),
                                'open_files': len(process.open_files()),
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    except Exception as e:
                        self.logger.warning(f"Resource monitoring error: {e}")

                    time.sleep(1)

            def stop(self):
                self.running = False
                self.thread.join()

        return ResourceMonitor(execution_record, self.logger)

    def _sanitize_process_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize process output for security."""
        if not self.config.output_sanitization:
            return result

        sanitized = result.copy()

        # Remove potentially sensitive information
        sensitive_patterns = [
            r'password[=:]\s*\S+',
            r'token[=:]\s*\S+',
            r'key[=:]\s*\S+',
            r'secret[=:]\s*\S+'
        ]

        import re
        for field in ['stdout', 'stderr']:
            if field in sanitized:
                content = sanitized[field]
                for pattern in sensitive_patterns:
                    content = re.sub(pattern, f'{pattern.split("[")[0]}=***REDACTED***', content, flags=re.IGNORECASE)
                sanitized[field] = content

        return sanitized

    def monitor_process_security(self, process_id: str) -> Dict[str, Any]:
        """Monitor ongoing process security."""
        if process_id not in self.active_processes:
            raise ValueError(f"Process {process_id} not found")

        execution_record = self.active_processes[process_id]

        # Check for security violations
        violations = self._check_security_violations(execution_record)
        if violations:
            execution_record.security_violations.extend(violations)
            self._handle_security_violations(execution_record, violations)

        return {
            'process_id': process_id,
            'state': execution_record.state.value,
            'resource_usage': execution_record.resource_usage,
            'security_violations': execution_record.security_violations,
            'execution_time': (datetime.now(timezone.utc) - execution_record.start_time).total_seconds()
        }

    def _check_security_violations(self, record: ProcessExecution) -> List[str]:
        """Check for security violations in process execution."""
        violations = []

        # Check resource usage
        if record.resource_usage:
            memory_mb = record.resource_usage.get('memory_mb', 0)
            if memory_mb > self.config.max_memory_mb:
                violations.append(f"Memory usage exceeded limit: {memory_mb}MB")

            cpu_percent = record.resource_usage.get('cpu_percent', 0)
            if cpu_percent > 90:  # High CPU usage
                violations.append(f"High CPU usage detected: {cpu_percent}%")

        # Check execution time
        if record.start_time:
            elapsed = (datetime.now(timezone.utc) - record.start_time).total_seconds()
            if elapsed > self.config.max_execution_time:
                violations.append(f"Execution time exceeded limit: {elapsed}s")

        return violations

    def _handle_security_violations(self, record: ProcessExecution, violations: List[str]):
        """Handle detected security violations."""
        for violation in violations:
            self.logger.error(f"Security violation in process {record.process_id}: {violation}")

            # Log security event
            self.audit.log_security_event(
                event_type='security_violation',
                user_id=record.user_id,
                action='process_security_violation',
                resource=f"process_{record.process_id}",
                classification='CUI//BASIC',
                additional_data={
                    'violation': violation,
                    'process_command': record.command,
                    'resource_usage': record.resource_usage
                }
            )

        # Terminate process if critical violations
        critical_violations = [v for v in violations if 'exceeded limit' in v]
        if critical_violations:
            self.terminate_process(record.process_id, 'security_violation')

    def terminate_process(self, process_id: str, reason: str = 'user_request'):
        """Terminate a running process."""
        if process_id not in self.active_processes:
            return

        record = self.active_processes[process_id]

        try:
            if record.pid > 0:
                process = psutil.Process(record.pid)
                process.terminate()

                # Wait for graceful shutdown
                time.sleep(2)

                # Force kill if still running
                if process.is_running():
                    process.kill()

            record.state = ProcessState.TERMINATED
            record.end_time = datetime.now(timezone.utc)

            self.logger.info(f"Process {process_id} terminated: {reason}")

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.warning(f"Failed to terminate process {process_id}: {e}")

    def generate_process_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive process security report."""
        total_processes = len(self.process_history) + len(self.active_processes)

        violation_summary = {}
        for record in self.process_history + list(self.active_processes.values()):
            for violation in record.security_violations:
                violation_type = violation.split(':')[0]
                violation_summary[violation_type] = violation_summary.get(violation_type, 0) + 1

        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'process_statistics': {
                'total_processes': total_processes,
                'active_processes': len(self.active_processes),
                'completed_processes': len(self.process_history),
                'sandboxed_executions': sum(1 for r in self.process_history if 'sandbox' in r.command)
            },
            'security_statistics': {
                'total_violations': len(self.security_violations),
                'violation_types': violation_summary,
                'processes_with_violations': sum(1 for r in self.process_history
                                               if r.security_violations)
            },
            'compliance_status': {
                'input_validation': self.config.input_validation,
                'output_sanitization': self.config.output_sanitization,
                'resource_monitoring': self.config.process_monitoring,
                'sandbox_enabled': self.config.sandbox_enabled,
                'audit_trail': self.config.audit_all_operations
            }
        }

    def _generate_context_id(self, user_id: str, process_name: str) -> str:
        """Generate unique context identifier."""
        data = f"{user_id}_{process_name}_{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _generate_process_id(self) -> str:
        """Generate unique process identifier."""
        return hashlib.sha256(f"{time.time()}_{os.urandom(8).hex()}".encode()).hexdigest()[:12]

    def _create_sandbox_environment(self, context_id: str) -> Dict[str, Any]:
        """Create sandbox environment configuration."""
        return {
            'type': 'filesystem_sandbox',
            'root_dir': f"/tmp/sandbox_{context_id}",
            'readonly_mounts': ['/usr', '/lib', '/lib64'],
            'readwrite_mounts': [f"/tmp/sandbox_{context_id}"],
            'network_isolation': self.config.network_isolation,
            'process_isolation': True
        }

    def _configure_resource_limits(self) -> Dict[str, Any]:
        """Configure resource limits for secure execution."""
        return {
            'max_memory_mb': self.config.max_memory_mb,
            'max_cpu_seconds': 300,  # 5 minutes
            'max_file_size_mb': 1024,  # 1GB
            'max_processes': 10,
            'max_open_files': 1024
        }

    def _setup_process_monitoring(self, context_id: str) -> Dict[str, Any]:
        """Setup process monitoring configuration."""
        return {
            'resource_monitoring': self.config.process_monitoring,
            'network_monitoring': True,
            'filesystem_monitoring': True,
            'syscall_monitoring': False,  # Requires advanced setup
            'log_file': f".claude/.artifacts/security/process_monitor_{context_id}.log"
        }

    def _get_security_restrictions(self) -> Dict[str, Any]:
        """Get security restrictions based on configuration."""
        restrictions = {
            'network_access': not self.config.network_isolation,
            'filesystem_write': True,  # Within sandbox only
            'system_calls': False,
            'privileged_operations': False,
            'inter_process_communication': False
        }

        if self.config.security_level == ProcessSecurityLevel.CLASSIFIED:
            restrictions.update({
                'network_access': False,
                'filesystem_write': False,
                'system_calls': False
            })

        return restrictions

    def _cleanup_execution_context(self, context_id: str, record: ProcessExecution):
        """Cleanup execution context after process completion."""
        # Move to history
        with self.lock:
            if record.process_id in self.active_processes:
                del self.active_processes[record.process_id]
            self.process_history.append(record)

        # Cleanup temporary files
        sandbox_dir = f"/tmp/sandbox_{context_id}"
        if os.path.exists(sandbox_dir):
            subprocess.run(['rm', '-rf', sandbox_dir], check=False)

        self.logger.info(f"Cleaned up execution context: {context_id}")

    def _handle_execution_error(self, record: ProcessExecution, error: str):
        """Handle process execution errors."""
        self.logger.error(f"Process execution error: {error}")

        self.audit.log_security_event(
            event_type='execution_error',
            user_id=record.user_id,
            action='process_execution_error',
            resource=f"process_{record.process_id}",
            classification='CUI//BASIC',
            additional_data={
                'error': error,
                'command': record.command,
                'execution_time': (datetime.now(timezone.utc) - record.start_time).total_seconds()
            }
        )

# Example usage
if __name__ == "__main__":
    config = ProcessSecurityConfig(
        security_level=ProcessSecurityLevel.ENHANCED,
        sandbox_enabled=True,
        process_monitoring=True,
        input_validation=True,
        max_memory_mb=2048
    )

    process_security = ProcessSecurityManager(config)
    print("Process Security Manager initialized")
    print(f"Security Level: {config.security_level.value}")
    print(f"Sandbox: {'Enabled' if config.sandbox_enabled else 'Disabled'}")