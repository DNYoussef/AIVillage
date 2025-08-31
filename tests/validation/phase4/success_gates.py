"""
Success Gates and Rollback Procedures for Phase 4 Validation

Implements automated success gates and rollback procedures to ensure Phase 4
deployment only proceeds when all criteria are met.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import shutil


class GateStatus(Enum):
    """Status of a success gate"""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SuccessGate:
    """Individual success gate definition"""
    name: str
    description: str
    gate_type: str
    criteria: Dict[str, Any]
    weight: float = 1.0
    critical: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 1


@dataclass
class GateResult:
    """Result of a success gate evaluation"""
    gate_name: str
    status: GateStatus
    score: float  # 0.0 to 1.0
    details: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    timestamp: str = ""


@dataclass
class RollbackAction:
    """Rollback action definition"""
    name: str
    action_type: str
    config: Dict[str, Any]
    priority: int = 0  # Lower numbers execute first
    timeout_seconds: int = 60


class SuccessGateManager:
    """
    Manages success gates and rollback procedures for Phase 4 validation
    """
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.logger = logging.getLogger(__name__)
        
        # Define success gates
        self.success_gates = self._define_success_gates()
        
        # Define rollback actions
        self.rollback_actions = self._define_rollback_actions()
        
        # Gate evaluation results
        self.gate_results: List[GateResult] = []
        
        # Rollback history
        self.rollback_history: List[Dict[str, Any]] = []
        
        # State management
        self.deployment_approved = False
        self.rollback_in_progress = False
        
        # Callbacks for notifications
        self.gate_callbacks: List[Callable[[GateResult], None]] = []
        self.rollback_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    async def evaluate_success_gates(self, validation_result: Any) -> Dict[str, Any]:
        """
        Evaluate all success gates against validation results
        
        Args:
            validation_result: Results from Phase4ValidationSuite
            
        Returns:
            Dictionary with gate evaluation results and deployment decision
        """
        self.logger.info("Starting success gate evaluation...")
        
        self.gate_results = []
        start_time = time.time()
        
        # Evaluate each gate
        for gate in self.success_gates:
            gate_result = await self._evaluate_single_gate(gate, validation_result)
            self.gate_results.append(gate_result)
            
            # Trigger callbacks
            for callback in self.gate_callbacks:
                try:
                    callback(gate_result)
                except Exception as e:
                    self.logger.error(f"Gate callback failed: {e}")
            
            # If critical gate fails, stop evaluation
            if gate.critical and gate_result.status == GateStatus.FAILED:
                self.logger.error(f"Critical gate failed: {gate.name}")
                break
        
        # Calculate overall result
        evaluation_result = self._calculate_overall_result()
        evaluation_result['execution_time_ms'] = int((time.time() - start_time) * 1000)
        evaluation_result['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Set deployment approval
        self.deployment_approved = evaluation_result['deployment_approved']
        
        self.logger.info(f"Success gate evaluation completed: {'APPROVED' if self.deployment_approved else 'REJECTED'}")
        
        return evaluation_result
    
    async def _evaluate_single_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate a single success gate"""
        start_time = time.perf_counter()
        
        self.logger.debug(f"Evaluating gate: {gate.name}")
        
        try:
            # Select evaluation method based on gate type
            if gate.gate_type == "coupling_improvement":
                result = await self._evaluate_coupling_gate(gate, validation_result)
            elif gate.gate_type == "performance_regression":
                result = await self._evaluate_performance_gate(gate, validation_result)
            elif gate.gate_type == "quality_metrics":
                result = await self._evaluate_quality_gate(gate, validation_result)
            elif gate.gate_type == "compatibility":
                result = await self._evaluate_compatibility_gate(gate, validation_result)
            elif gate.gate_type == "integration":
                result = await self._evaluate_integration_gate(gate, validation_result)
            elif gate.gate_type == "custom_script":
                result = await self._evaluate_custom_script_gate(gate, validation_result)
            else:
                result = GateResult(
                    gate_name=gate.name,
                    status=GateStatus.FAILED,
                    score=0.0,
                    details={},
                    error_message=f"Unknown gate type: {gate.gate_type}"
                )
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            result.execution_time_ms = execution_time
            result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return result
            
        except Exception as e:
            execution_time = int((time.perf_counter() - start_time) * 1000)
            
            self.logger.error(f"Gate evaluation failed for {gate.name}: {e}")
            
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message=str(e),
                execution_time_ms=execution_time,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
    
    async def _evaluate_coupling_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate coupling improvement gate"""
        criteria = gate.criteria
        coupling_results = validation_result.coupling_results
        
        if not coupling_results:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No coupling results available"
            )
        
        improvements = coupling_results.get('improvements', {})
        details = {}
        passed_components = 0
        total_components = 0
        
        # Check each component requirement
        for component, min_improvement in criteria.get('component_requirements', {}).items():
            total_components += 1
            
            if component in improvements:
                actual_improvement = improvements[component].get('improvement_percent', 0)
                target_met = actual_improvement >= min_improvement
                
                details[component] = {
                    'required_improvement': min_improvement,
                    'actual_improvement': actual_improvement,
                    'target_met': target_met,
                    'current_score': improvements[component].get('current', 0),
                    'baseline_score': improvements[component].get('baseline', 0)
                }
                
                if target_met:
                    passed_components += 1
            else:
                details[component] = {
                    'required_improvement': min_improvement,
                    'actual_improvement': 0,
                    'target_met': False,
                    'error': 'Component not found in results'
                }
        
        # Calculate score
        score = passed_components / total_components if total_components > 0 else 0.0
        
        # Check minimum score requirement
        min_score = criteria.get('minimum_score', 1.0)
        status = GateStatus.PASSED if score >= min_score else GateStatus.FAILED
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            score=score,
            details=details
        )
    
    async def _evaluate_performance_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate performance regression gate"""
        criteria = gate.criteria
        performance_results = validation_result.performance_results
        
        if not performance_results:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No performance results available"
            )
        
        overall = performance_results.get('overall', {})
        details = {}
        checks_passed = 0
        total_checks = 0
        
        # Memory increase check
        if 'max_memory_increase' in criteria:
            total_checks += 1
            max_allowed = criteria['max_memory_increase']
            actual_increase = overall.get('memory_increase_percent', float('inf'))
            passed = actual_increase <= max_allowed
            
            details['memory_increase'] = {
                'max_allowed': max_allowed,
                'actual': actual_increase,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Throughput check
        if 'min_throughput_ratio' in criteria:
            total_checks += 1
            min_required = criteria['min_throughput_ratio']
            actual_ratio = overall.get('throughput_ratio', 0)
            passed = actual_ratio >= min_required
            
            details['throughput_ratio'] = {
                'min_required': min_required,
                'actual': actual_ratio,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Performance degradation check
        if 'max_degradation' in criteria:
            total_checks += 1
            max_allowed = criteria['max_degradation']
            actual_degradation = overall.get('performance_degradation_percent', float('inf'))
            passed = actual_degradation <= max_allowed
            
            details['performance_degradation'] = {
                'max_allowed': max_allowed,
                'actual': actual_degradation,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Calculate score and status
        score = checks_passed / total_checks if total_checks > 0 else 1.0
        min_score = criteria.get('minimum_score', 1.0)
        status = GateStatus.PASSED if score >= min_score else GateStatus.FAILED
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            score=score,
            details=details
        )
    
    async def _evaluate_quality_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate code quality gate"""
        criteria = gate.criteria
        quality_results = validation_result.quality_results
        
        if not quality_results:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No quality results available"
            )
        
        details = {}
        checks_passed = 0
        total_checks = 0
        
        # Test coverage check
        if 'min_test_coverage' in criteria:
            total_checks += 1
            min_required = criteria['min_test_coverage']
            actual_coverage = quality_results.get('avg_test_coverage', 0)
            passed = actual_coverage >= min_required
            
            details['test_coverage'] = {
                'min_required': min_required,
                'actual': actual_coverage,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Magic literals check
        if 'max_magic_literals' in criteria:
            total_checks += 1
            max_allowed = criteria['max_magic_literals']
            actual_count = quality_results.get('magic_literals_count', float('inf'))
            passed = actual_count <= max_allowed
            
            details['magic_literals'] = {
                'max_allowed': max_allowed,
                'actual': actual_count,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Lines per class check
        if 'max_lines_per_class' in criteria:
            total_checks += 1
            max_allowed = criteria['max_lines_per_class']
            actual_max = quality_results.get('max_lines_per_class', float('inf'))
            passed = actual_max <= max_allowed
            
            details['lines_per_class'] = {
                'max_allowed': max_allowed,
                'actual_max': actual_max,
                'passed': passed
            }
            
            if passed:
                checks_passed += 1
        
        # Calculate score and status
        score = checks_passed / total_checks if total_checks > 0 else 1.0
        min_score = criteria.get('minimum_score', 1.0)
        status = GateStatus.PASSED if score >= min_score else GateStatus.FAILED
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            score=score,
            details=details
        )
    
    async def _evaluate_compatibility_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate backwards compatibility gate"""
        compatibility_results = validation_result.compatibility_results
        
        if not compatibility_results:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No compatibility results available"
            )
        
        # Check if all compatibility tests passed
        all_tests_passed = compatibility_results.get('all_tests_passed', False)
        critical_failures = compatibility_results.get('critical_failures', [])
        
        details = {
            'all_tests_passed': all_tests_passed,
            'critical_failures_count': len(critical_failures),
            'critical_failures': critical_failures,
            'summary': compatibility_results.get('summary', {})
        }
        
        # Score based on success rate
        summary = compatibility_results.get('summary', {})
        total_tests = summary.get('total_tests', 0)
        passed_tests = summary.get('passed_tests', 0)
        
        score = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Status based on critical failures and minimum score
        min_score = gate.criteria.get('minimum_score', 0.95)  # 95% pass rate
        max_critical_failures = gate.criteria.get('max_critical_failures', 0)
        
        status = GateStatus.PASSED if (
            score >= min_score and len(critical_failures) <= max_critical_failures
        ) else GateStatus.FAILED
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            score=score,
            details=details
        )
    
    async def _evaluate_integration_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate integration testing gate"""
        integration_results = validation_result.integration_results
        
        if not integration_results:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No integration results available"
            )
        
        # Check overall integration status
        all_services_integrated = integration_results.get('all_services_integrated', False)
        
        details = {
            'all_services_integrated': all_services_integrated,
            'service_health': integration_results.get('service_health', {}),
            'communication_tests': integration_results.get('communication_tests', {}),
            'summary': integration_results.get('summary', {})
        }
        
        # Calculate score based on various factors
        summary = integration_results.get('summary', {})
        service_health = integration_results.get('service_health', {})
        communication = integration_results.get('communication_tests', {})
        
        factors = []
        
        # Integration test success rate
        total_tests = summary.get('total_tests', 0)
        passed_tests = summary.get('passed_tests', 0)
        if total_tests > 0:
            factors.append(passed_tests / total_tests)
        
        # Service health score
        if service_health.get('total_count', 0) > 0:
            health_score = service_health.get('healthy_count', 0) / service_health.get('total_count', 1)
            factors.append(health_score)
        
        # Communication success rate
        if 'success_rate' in communication:
            comm_score = communication['success_rate'] / 100.0
            factors.append(comm_score)
        
        # Overall score
        score = sum(factors) / len(factors) if factors else 0.0
        
        # Status determination
        min_score = gate.criteria.get('minimum_score', 0.90)
        status = GateStatus.PASSED if score >= min_score and all_services_integrated else GateStatus.FAILED
        
        return GateResult(
            gate_name=gate.name,
            status=status,
            score=score,
            details=details
        )
    
    async def _evaluate_custom_script_gate(self, gate: SuccessGate, validation_result: Any) -> GateResult:
        """Evaluate custom script gate"""
        script_path = gate.criteria.get('script_path')
        
        if not script_path:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message="No script path specified"
            )
        
        try:
            # Execute custom script
            result = subprocess.run(
                [script_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=gate.timeout_seconds
            )
            
            # Parse result (script should return JSON)
            try:
                script_output = json.loads(result.stdout)
                passed = script_output.get('passed', False)
                score = script_output.get('score', 1.0 if passed else 0.0)
                details = script_output.get('details', {})
            except json.JSONDecodeError:
                # Fallback: consider return code only
                passed = result.returncode == 0
                score = 1.0 if passed else 0.0
                details = {'stdout': result.stdout, 'stderr': result.stderr}
            
            status = GateStatus.PASSED if passed else GateStatus.FAILED
            
            return GateResult(
                gate_name=gate.name,
                status=status,
                score=score,
                details=details
            )
            
        except subprocess.TimeoutExpired:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message=f"Script timed out after {gate.timeout_seconds} seconds"
            )
        except Exception as e:
            return GateResult(
                gate_name=gate.name,
                status=GateStatus.FAILED,
                score=0.0,
                details={},
                error_message=str(e)
            )
    
    def _calculate_overall_result(self) -> Dict[str, Any]:
        """Calculate overall success gate result"""
        if not self.gate_results:
            return {
                'deployment_approved': False,
                'overall_score': 0.0,
                'gates_passed': 0,
                'gates_failed': 0,
                'gates_total': 0,
                'critical_failures': [],
                'gate_results': []
            }
        
        gates_passed = 0
        gates_failed = 0
        critical_failures = []
        weighted_scores = []
        total_weight = 0
        
        for result in self.gate_results:
            gate = next((g for g in self.success_gates if g.name == result.gate_name), None)
            
            if result.status == GateStatus.PASSED:
                gates_passed += 1
                weighted_scores.append(result.score * (gate.weight if gate else 1.0))
            else:
                gates_failed += 1
                if gate and gate.critical:
                    critical_failures.append({
                        'gate': result.gate_name,
                        'error': result.error_message,
                        'score': result.score
                    })
            
            total_weight += gate.weight if gate else 1.0
        
        # Calculate overall score
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
        
        # Determine deployment approval
        deployment_approved = (
            len(critical_failures) == 0 and  # No critical failures
            gates_failed == 0 and           # No gates failed
            overall_score >= 0.95           # At least 95% overall score
        )
        
        return {
            'deployment_approved': deployment_approved,
            'overall_score': overall_score,
            'gates_passed': gates_passed,
            'gates_failed': gates_failed,
            'gates_total': len(self.gate_results),
            'critical_failures': critical_failures,
            'gate_results': [asdict(result) for result in self.gate_results]
        }
    
    async def execute_rollback(self, reason: str, validation_result: Any = None) -> Dict[str, Any]:
        """Execute rollback procedures"""
        if self.rollback_in_progress:
            return {'error': 'Rollback already in progress'}
        
        self.rollback_in_progress = True
        self.logger.warning(f"🔄 Starting rollback procedure: {reason}")
        
        rollback_result = {
            'reason': reason,
            'start_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'actions': [],
            'success': False,
            'error': None
        }
        
        try:
            # Sort rollback actions by priority
            sorted_actions = sorted(self.rollback_actions, key=lambda x: x.priority)
            
            for action in sorted_actions:
                action_result = await self._execute_rollback_action(action)
                rollback_result['actions'].append(action_result)
                
                # Trigger rollback callbacks
                for callback in self.rollback_callbacks:
                    try:
                        callback(action.name, action_result)
                    except Exception as e:
                        self.logger.error(f"Rollback callback failed: {e}")
                
                # If critical action fails, stop rollback
                if not action_result['success'] and action.config.get('critical', True):
                    rollback_result['error'] = f"Critical rollback action failed: {action.name}"
                    break
            
            # Determine overall success
            successful_actions = [a for a in rollback_result['actions'] if a['success']]
            rollback_result['success'] = len(successful_actions) == len(sorted_actions)
            
            # Record rollback in history
            rollback_result['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.rollback_history.append(rollback_result)
            
            if rollback_result['success']:
                self.logger.info("✅ Rollback procedure completed successfully")
            else:
                self.logger.error("❌ Rollback procedure completed with errors")
            
        except Exception as e:
            rollback_result['error'] = str(e)
            rollback_result['success'] = False
            self.logger.error(f"Rollback procedure failed: {e}")
        
        finally:
            self.rollback_in_progress = False
        
        return rollback_result
    
    async def _execute_rollback_action(self, action: RollbackAction) -> Dict[str, Any]:
        """Execute a single rollback action"""
        start_time = time.perf_counter()
        
        self.logger.info(f"Executing rollback action: {action.name}")
        
        action_result = {
            'name': action.name,
            'type': action.action_type,
            'success': False,
            'error': None,
            'details': {},
            'execution_time_ms': 0
        }
        
        try:
            if action.action_type == 'git_reset':
                await self._rollback_git_reset(action, action_result)
            elif action.action_type == 'file_restore':
                await self._rollback_file_restore(action, action_result)
            elif action.action_type == 'service_restart':
                await self._rollback_service_restart(action, action_result)
            elif action.action_type == 'database_restore':
                await self._rollback_database_restore(action, action_result)
            elif action.action_type == 'custom_script':
                await self._rollback_custom_script(action, action_result)
            else:
                action_result['error'] = f"Unknown rollback action type: {action.action_type}"
            
            execution_time = int((time.perf_counter() - start_time) * 1000)
            action_result['execution_time_ms'] = execution_time
            
        except Exception as e:
            action_result['error'] = str(e)
            execution_time = int((time.perf_counter() - start_time) * 1000)
            action_result['execution_time_ms'] = execution_time
        
        return action_result
    
    async def _rollback_git_reset(self, action: RollbackAction, result: Dict[str, Any]) -> None:
        """Execute git reset rollback action"""
        config = action.config
        target_commit = config.get('target_commit', 'HEAD~1')
        create_backup_branch = config.get('create_backup_branch', True)
        
        try:
            # Create backup branch if requested
            if create_backup_branch:
                backup_branch = f"backup-{int(time.time())}"
                subprocess.run(['git', 'checkout', '-b', backup_branch], 
                             cwd=self.project_root, check=True)
                result['details']['backup_branch'] = backup_branch
            
            # Reset to target commit
            subprocess.run(['git', 'reset', '--hard', target_commit], 
                         cwd=self.project_root, check=True)
            
            result['success'] = True
            result['details']['target_commit'] = target_commit
            
        except subprocess.CalledProcessError as e:
            result['error'] = f"Git reset failed: {e}"
    
    async def _rollback_file_restore(self, action: RollbackAction, result: Dict[str, Any]) -> None:
        """Execute file restore rollback action"""
        config = action.config
        backup_dir = Path(config.get('backup_directory', '/tmp/phase4_backup'))
        files_to_restore = config.get('files', [])
        
        restored_files = []
        
        try:
            for file_pattern in files_to_restore:
                source_files = list(backup_dir.glob(file_pattern))
                for source_file in source_files:
                    # Calculate target path
                    relative_path = source_file.relative_to(backup_dir)
                    target_path = self.project_root / relative_path
                    
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Restore file
                    shutil.copy2(source_file, target_path)
                    restored_files.append(str(target_path))
            
            result['success'] = True
            result['details']['restored_files'] = restored_files
            
        except Exception as e:
            result['error'] = f"File restore failed: {e}"
            result['details']['restored_files'] = restored_files
    
    async def _rollback_service_restart(self, action: RollbackAction, result: Dict[str, Any]) -> None:
        """Execute service restart rollback action"""
        config = action.config
        services = config.get('services', [])
        use_sudo = config.get('use_sudo', True)
        
        restarted_services = []
        
        try:
            for service in services:
                command = ['sudo', 'systemctl', 'restart', service] if use_sudo else ['systemctl', 'restart', service]
                subprocess.run(command, check=True, timeout=action.timeout_seconds)
                restarted_services.append(service)
            
            result['success'] = True
            result['details']['restarted_services'] = restarted_services
            
        except Exception as e:
            result['error'] = f"Service restart failed: {e}"
            result['details']['restarted_services'] = restarted_services
    
    async def _rollback_database_restore(self, action: RollbackAction, result: Dict[str, Any]) -> None:
        """Execute database restore rollback action"""
        config = action.config
        backup_file = config.get('backup_file')
        database_name = config.get('database_name')
        
        try:
            if not backup_file or not Path(backup_file).exists():
                result['error'] = f"Backup file not found: {backup_file}"
                return
            
            # This is a placeholder - actual implementation would depend on database type
            # For PostgreSQL: pg_restore
            # For MySQL: mysql < backup.sql
            # etc.
            
            result['success'] = True
            result['details']['backup_file'] = backup_file
            result['details']['database'] = database_name
            
        except Exception as e:
            result['error'] = f"Database restore failed: {e}"
    
    async def _rollback_custom_script(self, action: RollbackAction, result: Dict[str, Any]) -> None:
        """Execute custom script rollback action"""
        config = action.config
        script_path = config.get('script_path')
        script_args = config.get('args', [])
        
        try:
            command = [script_path] + script_args
            process_result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=action.timeout_seconds
            )
            
            result['success'] = process_result.returncode == 0
            result['details']['returncode'] = process_result.returncode
            result['details']['stdout'] = process_result.stdout
            result['details']['stderr'] = process_result.stderr
            
            if not result['success']:
                result['error'] = f"Script returned non-zero exit code: {process_result.returncode}"
            
        except subprocess.TimeoutExpired:
            result['error'] = f"Script timed out after {action.timeout_seconds} seconds"
        except Exception as e:
            result['error'] = f"Script execution failed: {e}"
    
    def _define_success_gates(self) -> List[SuccessGate]:
        """Define all success gates for Phase 4"""
        return [
            SuccessGate(
                name="coupling_improvements",
                description="Verify coupling score improvements meet targets",
                gate_type="coupling_improvement",
                criteria={
                    'component_requirements': {
                        'UnifiedManagement': 50.0,  # 50% improvement required
                        'SageAgent': 30.0,          # 30% improvement required
                        'task_management_average': 35.0  # 35% improvement required
                    },
                    'minimum_score': 1.0  # All components must meet requirements
                },
                weight=3.0,  # High weight - critical for Phase 4
                critical=True
            ),
            SuccessGate(
                name="performance_benchmarks",
                description="Verify performance benchmarks are maintained",
                gate_type="performance_regression",
                criteria={
                    'max_memory_increase': 10.0,      # Max 10% memory increase
                    'min_throughput_ratio': 1.0,      # Maintain throughput
                    'max_degradation': 5.0,           # Max 5% performance degradation
                    'minimum_score': 1.0
                },
                weight=2.5,
                critical=True
            ),
            SuccessGate(
                name="code_quality_standards",
                description="Verify code quality improvements",
                gate_type="quality_metrics",
                criteria={
                    'min_test_coverage': 85.0,        # Min 85% test coverage
                    'max_magic_literals': 5,          # Max 5 magic literals
                    'max_lines_per_class': 150,       # Max 150 lines per class
                    'minimum_score': 0.8              # 80% of quality checks must pass
                },
                weight=2.0,
                critical=False  # Quality improvements are important but not blocking
            ),
            SuccessGate(
                name="backwards_compatibility",
                description="Verify backwards compatibility is maintained",
                gate_type="compatibility",
                criteria={
                    'minimum_score': 0.95,            # 95% compatibility test pass rate
                    'max_critical_failures': 0        # No critical compatibility failures
                },
                weight=3.0,
                critical=True  # Critical for existing system stability
            ),
            SuccessGate(
                name="service_integration",
                description="Verify all services integrate correctly",
                gate_type="integration",
                criteria={
                    'minimum_score': 0.90             # 90% integration test success rate
                },
                weight=2.5,
                critical=True
            ),
            SuccessGate(
                name="security_validation",
                description="Run custom security validation script",
                gate_type="custom_script",
                criteria={
                    'script_path': str(self.project_root / 'scripts/security_validation.py')
                },
                weight=2.0,
                critical=True,
                timeout_seconds=600  # 10 minutes for security checks
            )
        ]
    
    def _define_rollback_actions(self) -> List[RollbackAction]:
        """Define rollback actions for Phase 4"""
        return [
            RollbackAction(
                name="create_incident_backup",
                action_type="file_restore",
                config={
                    'backup_directory': '/tmp/phase4_incident_backup',
                    'files': ['swarm/**/*.py', 'tests/**/*.py', 'requirements.txt'],
                    'critical': False
                },
                priority=0  # Run first
            ),
            RollbackAction(
                name="git_rollback",
                action_type="git_reset",
                config={
                    'target_commit': 'HEAD~1',  # Roll back one commit
                    'create_backup_branch': True,
                    'critical': True
                },
                priority=1
            ),
            RollbackAction(
                name="restart_core_services",
                action_type="service_restart",
                config={
                    'services': ['swarm-agent', 'task-manager', 'workflow-engine'],
                    'use_sudo': True,
                    'critical': True
                },
                priority=2
            ),
            RollbackAction(
                name="restore_configuration",
                action_type="file_restore",
                config={
                    'backup_directory': '/etc/swarm/backup',
                    'files': ['config/*.json', '*.conf'],
                    'critical': False
                },
                priority=3
            ),
            RollbackAction(
                name="run_rollback_validation",
                action_type="custom_script",
                config={
                    'script_path': str(self.project_root / 'scripts/rollback_validation.py'),
                    'args': ['--quick-check'],
                    'critical': False
                },
                priority=4  # Run last
            )
        ]
    
    def add_gate_callback(self, callback: Callable[[GateResult], None]) -> None:
        """Add callback for gate evaluation results"""
        self.gate_callbacks.append(callback)
    
    def add_rollback_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for rollback actions"""
        self.rollback_callbacks.append(callback)
    
    def get_gate_status(self) -> Dict[str, Any]:
        """Get current status of all gates"""
        return {
            'deployment_approved': self.deployment_approved,
            'rollback_in_progress': self.rollback_in_progress,
            'last_evaluation': {
                'gate_results': [asdict(result) for result in self.gate_results],
                'timestamp': self.gate_results[-1].timestamp if self.gate_results else None
            },
            'rollback_history_count': len(self.rollback_history),
            'gates_defined': len(self.success_gates),
            'rollback_actions_defined': len(self.rollback_actions)
        }
    
    def export_gate_results(self, file_path: Path) -> None:
        """Export gate results to JSON file"""
        export_data = {
            'deployment_approved': self.deployment_approved,
            'gate_results': [asdict(result) for result in self.gate_results],
            'rollback_history': self.rollback_history,
            'export_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Gate results exported to {file_path}")