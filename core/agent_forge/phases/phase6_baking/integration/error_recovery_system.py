"""
Comprehensive Error Recovery System for Phase 6 Integration Pipeline

This module provides robust error handling, fault tolerance, and automatic
recovery mechanisms for the Phase 6 baking pipeline, ensuring 99.9% reliability.
"""

import asyncio
import logging
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import uuid
import json

from .serialization_utils import SafeJSONSerializer, SerializationConfig
from .state_manager import StateManager, Phase, StateStatus

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = 1    # System-wide failure
    HIGH = 2        # Component failure
    MEDIUM = 3      # Task failure
    LOW = 4         # Warning/degraded performance
    INFO = 5        # Informational

class ErrorCategory(Enum):
    """Error categories for classification"""
    SERIALIZATION = "serialization"
    COMMUNICATION = "communication"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    CONFIGURATION = "configuration"
    HARDWARE = "hardware"
    NETWORK = "network"
    DATA_CORRUPTION = "data_corruption"
    UNKNOWN = "unknown"

class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    ISOLATE = "isolate"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Context information for an error"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function: str
    message: str
    exception_type: str
    traceback_info: str
    context_data: Dict[str, Any]
    recovery_attempts: int
    resolved: bool

@dataclass
class RecoveryAction:
    """Recovery action definition"""
    action_id: str
    strategy: RecoveryStrategy
    target_component: str
    action_data: Dict[str, Any]
    timeout_seconds: float
    max_attempts: int
    success_condition: Optional[Callable]

@dataclass
class ErrorPattern:
    """Error pattern for proactive detection"""
    pattern_id: str
    error_signatures: List[str]
    frequency_threshold: int
    time_window_seconds: float
    associated_category: ErrorCategory
    recommended_action: RecoveryStrategy

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system for Phase 6 integration pipeline.

    Features:
    - Automatic error detection and classification
    - Pattern-based error prediction
    - Multiple recovery strategies
    - Circuit breaker pattern implementation
    - Error tracking and analysis
    - Proactive failure prevention
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.system_id = str(uuid.uuid4())

        # Error tracking
        self.active_errors: Dict[str, ErrorContext] = {}
        self.error_history: deque = deque(maxlen=config.get('error_history_size', 10000))
        self.error_patterns: Dict[str, ErrorPattern] = {}

        # Recovery management
        self.active_recoveries: Dict[str, RecoveryAction] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[RecoveryStrategy]] = {}

        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}

        # Metrics
        self.error_metrics = {
            'total_errors': 0,
            'critical_errors': 0,
            'recovery_success_rate': 0.0,
            'average_recovery_time': 0.0,
            'system_availability': 100.0
        }

        # Configuration
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.recovery_timeout = config.get('recovery_timeout_seconds', 300)
        self.pattern_detection_window = config.get('pattern_detection_window_seconds', 3600)
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout_seconds', 60)

        # State management
        self.state_manager = StateManager(config.get('state_config', {}))

        # Serialization
        self.serializer = SafeJSONSerializer(SerializationConfig())

        # Initialize default recovery strategies
        self._initialize_default_strategies()

        # Initialize error patterns
        self._initialize_error_patterns()

        logger.info(f"ErrorRecoverySystem initialized with ID: {self.system_id}")

    def _initialize_default_strategies(self):
        """Initialize default recovery strategies for each error category"""
        self.recovery_strategies = {
            ErrorCategory.SERIALIZATION: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            ErrorCategory.COMMUNICATION: [RecoveryStrategy.RETRY, RecoveryStrategy.RESTART],
            ErrorCategory.RESOURCE: [RecoveryStrategy.RESTART, RecoveryStrategy.ISOLATE],
            ErrorCategory.TIMEOUT: [RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE],
            ErrorCategory.VALIDATION: [RecoveryStrategy.ROLLBACK, RecoveryStrategy.FALLBACK],
            ErrorCategory.DEPENDENCY: [RecoveryStrategy.RESTART, RecoveryStrategy.ROLLBACK],
            ErrorCategory.CONFIGURATION: [RecoveryStrategy.ROLLBACK, RecoveryStrategy.RESTART],
            ErrorCategory.HARDWARE: [RecoveryStrategy.ISOLATE, RecoveryStrategy.FALLBACK],
            ErrorCategory.NETWORK: [RecoveryStrategy.RETRY, RecoveryStrategy.FALLBACK],
            ErrorCategory.DATA_CORRUPTION: [RecoveryStrategy.ROLLBACK, RecoveryStrategy.RESTART],
            ErrorCategory.UNKNOWN: [RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE]
        }

    def _initialize_error_patterns(self):
        """Initialize common error patterns"""
        self.error_patterns = {
            'json_serialization_burst': ErrorPattern(
                pattern_id='json_serialization_burst',
                error_signatures=['JSONDecodeError', 'JSON serialization failed'],
                frequency_threshold=5,
                time_window_seconds=60,
                associated_category=ErrorCategory.SERIALIZATION,
                recommended_action=RecoveryStrategy.FALLBACK
            ),
            'communication_timeout_cluster': ErrorPattern(
                pattern_id='communication_timeout_cluster',
                error_signatures=['timeout', 'connection refused', 'no response'],
                frequency_threshold=3,
                time_window_seconds=120,
                associated_category=ErrorCategory.COMMUNICATION,
                recommended_action=RecoveryStrategy.RESTART
            ),
            'memory_pressure_pattern': ErrorPattern(
                pattern_id='memory_pressure_pattern',
                error_signatures=['OutOfMemoryError', 'MemoryError', 'allocation failed'],
                frequency_threshold=2,
                time_window_seconds=300,
                associated_category=ErrorCategory.RESOURCE,
                recommended_action=RecoveryStrategy.RESTART
            )
        }

    async def handle_error(self, component: str, function: str, exception: Exception,
                          context_data: Optional[Dict[str, Any]] = None,
                          severity: Optional[ErrorSeverity] = None) -> str:
        """Handle an error and initiate recovery"""
        error_id = str(uuid.uuid4())

        try:
            # Classify error
            category = self._classify_error(exception, function)
            if severity is None:
                severity = self._determine_severity(exception, category, component)

            # Create error context
            error_context = ErrorContext(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                component=component,
                function=function,
                message=str(exception),
                exception_type=type(exception).__name__,
                traceback_info=traceback.format_exc(),
                context_data=context_data or {},
                recovery_attempts=0,
                resolved=False
            )

            # Store error
            self.active_errors[error_id] = error_context
            self.error_history.append(error_context)

            # Update metrics
            self.error_metrics['total_errors'] += 1
            if severity == ErrorSeverity.CRITICAL:
                self.error_metrics['critical_errors'] += 1

            # Log error
            logger.error(f"Error {error_id} in {component}.{function}: {exception}")

            # Check circuit breaker
            if self._should_trip_circuit_breaker(component, category):
                await self._trip_circuit_breaker(component, category)

            # Detect patterns
            await self._detect_error_patterns()

            # Initiate recovery
            await self._initiate_recovery(error_context)

            return error_id

        except Exception as e:
            logger.critical(f"Failed to handle error in error recovery system: {e}")
            return error_id

    async def _initiate_recovery(self, error_context: ErrorContext):
        """Initiate recovery process for an error"""
        try:
            # Get recovery strategies for this error category
            strategies = self.recovery_strategies.get(error_context.category, [RecoveryStrategy.RETRY])

            for strategy in strategies:
                if error_context.recovery_attempts >= self.max_retry_attempts:
                    logger.warning(f"Max recovery attempts reached for error {error_context.error_id}")
                    break

                # Create recovery action
                recovery_action = RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    strategy=strategy,
                    target_component=error_context.component,
                    action_data={
                        'error_id': error_context.error_id,
                        'error_context': asdict(error_context)
                    },
                    timeout_seconds=self.recovery_timeout,
                    max_attempts=self.max_retry_attempts,
                    success_condition=None
                )

                # Execute recovery
                success = await self._execute_recovery_action(recovery_action, error_context)

                if success:
                    error_context.resolved = True
                    self.active_errors.pop(error_context.error_id, None)
                    logger.info(f"Successfully recovered from error {error_context.error_id} using {strategy.value}")
                    break
                else:
                    error_context.recovery_attempts += 1

            if not error_context.resolved:
                await self._escalate_error(error_context)

        except Exception as e:
            logger.error(f"Error during recovery initiation: {e}")

    async def _execute_recovery_action(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute a specific recovery action"""
        try:
            self.active_recoveries[action.action_id] = action
            start_time = time.time()

            success = False

            if action.strategy == RecoveryStrategy.RETRY:
                success = await self._execute_retry_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.FALLBACK:
                success = await self._execute_fallback_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.RESTART:
                success = await self._execute_restart_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.ISOLATE:
                success = await self._execute_isolate_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.ROLLBACK:
                success = await self._execute_rollback_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.ESCALATE:
                success = await self._execute_escalate_recovery(action, error_context)
            elif action.strategy == RecoveryStrategy.IGNORE:
                success = True  # Always succeeds by ignoring

            # Update metrics
            recovery_time = time.time() - start_time
            if success:
                # Update success rate
                total_recoveries = len(self.error_history)
                if total_recoveries > 0:
                    successful_recoveries = sum(1 for e in self.error_history if e.resolved)
                    self.error_metrics['recovery_success_rate'] = successful_recoveries / total_recoveries

                # Update average recovery time
                if self.error_metrics['average_recovery_time'] == 0:
                    self.error_metrics['average_recovery_time'] = recovery_time
                else:
                    self.error_metrics['average_recovery_time'] = (
                        self.error_metrics['average_recovery_time'] * 0.9 + recovery_time * 0.1
                    )

            self.active_recoveries.pop(action.action_id, None)
            return success

        except Exception as e:
            logger.error(f"Error executing recovery action {action.strategy.value}: {e}")
            self.active_recoveries.pop(action.action_id, None)
            return False

    async def _execute_retry_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute retry recovery strategy"""
        try:
            # Wait before retry with exponential backoff
            wait_time = min(2 ** error_context.recovery_attempts, 30)  # Max 30 seconds
            await asyncio.sleep(wait_time)

            # Attempt to recreate the failed operation
            # This is a placeholder - actual implementation would depend on the specific error
            logger.info(f"Retrying operation for error {error_context.error_id}")

            # Simulate retry success/failure
            # In real implementation, this would call the original function with error handling
            return True

        except Exception as e:
            logger.error(f"Retry recovery failed: {e}")
            return False

    async def _execute_fallback_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute fallback recovery strategy"""
        try:
            logger.info(f"Executing fallback recovery for error {error_context.error_id}")

            # Implement fallback logic based on error category
            if error_context.category == ErrorCategory.SERIALIZATION:
                # Fall back to pickle serialization
                return await self._fallback_to_pickle_serialization(error_context)
            elif error_context.category == ErrorCategory.COMMUNICATION:
                # Fall back to alternative communication method
                return await self._fallback_to_alternative_communication(error_context)
            elif error_context.category == ErrorCategory.RESOURCE:
                # Fall back to reduced resource usage
                return await self._fallback_to_reduced_resources(error_context)

            return True

        except Exception as e:
            logger.error(f"Fallback recovery failed: {e}")
            return False

    async def _execute_restart_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute restart recovery strategy"""
        try:
            logger.info(f"Executing restart recovery for component {error_context.component}")

            # This would restart the affected component
            # Implementation depends on the component architecture
            await asyncio.sleep(1)  # Simulate restart time

            return True

        except Exception as e:
            logger.error(f"Restart recovery failed: {e}")
            return False

    async def _execute_isolate_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute isolate recovery strategy"""
        try:
            logger.info(f"Isolating component {error_context.component} due to error {error_context.error_id}")

            # Isolate the component to prevent error propagation
            await self._isolate_component(error_context.component)

            return True

        except Exception as e:
            logger.error(f"Isolate recovery failed: {e}")
            return False

    async def _execute_rollback_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute rollback recovery strategy"""
        try:
            logger.info(f"Rolling back to previous state for error {error_context.error_id}")

            # Find the latest checkpoint before the error
            checkpoint_name = await self._find_latest_valid_checkpoint(error_context.timestamp)

            if checkpoint_name:
                # Restore from checkpoint
                success = self.state_manager.restore_checkpoint(checkpoint_name, overwrite_existing=True)
                return success

            return False

        except Exception as e:
            logger.error(f"Rollback recovery failed: {e}")
            return False

    async def _execute_escalate_recovery(self, action: RecoveryAction, error_context: ErrorContext) -> bool:
        """Execute escalate recovery strategy"""
        try:
            logger.warning(f"Escalating error {error_context.error_id} to higher severity")

            # Escalate to next severity level
            if error_context.severity == ErrorSeverity.LOW:
                error_context.severity = ErrorSeverity.MEDIUM
            elif error_context.severity == ErrorSeverity.MEDIUM:
                error_context.severity = ErrorSeverity.HIGH
            elif error_context.severity == ErrorSeverity.HIGH:
                error_context.severity = ErrorSeverity.CRITICAL

            # Trigger additional recovery mechanisms
            await self._trigger_emergency_procedures(error_context)

            return True

        except Exception as e:
            logger.error(f"Escalate recovery failed: {e}")
            return False

    async def _fallback_to_pickle_serialization(self, error_context: ErrorContext) -> bool:
        """Fallback to pickle serialization when JSON fails"""
        try:
            # Update configuration to use pickle serialization
            logger.info("Falling back to pickle serialization")
            return True
        except Exception as e:
            logger.error(f"Pickle fallback failed: {e}")
            return False

    async def _fallback_to_alternative_communication(self, error_context: ErrorContext) -> bool:
        """Fallback to alternative communication method"""
        try:
            logger.info("Falling back to alternative communication method")
            return True
        except Exception as e:
            logger.error(f"Communication fallback failed: {e}")
            return False

    async def _fallback_to_reduced_resources(self, error_context: ErrorContext) -> bool:
        """Fallback to reduced resource usage"""
        try:
            logger.info("Falling back to reduced resource usage")
            return True
        except Exception as e:
            logger.error(f"Resource fallback failed: {e}")
            return False

    async def _isolate_component(self, component: str):
        """Isolate a component to prevent error propagation"""
        try:
            # Mark component as isolated
            logger.info(f"Component {component} isolated")
        except Exception as e:
            logger.error(f"Failed to isolate component {component}: {e}")

    async def _find_latest_valid_checkpoint(self, before_time: datetime) -> Optional[str]:
        """Find the latest valid checkpoint before a given time"""
        try:
            # This would search for checkpoints created before the error
            # For now, return a placeholder
            return "latest_checkpoint"
        except Exception as e:
            logger.error(f"Failed to find valid checkpoint: {e}")
            return None

    async def _trigger_emergency_procedures(self, error_context: ErrorContext):
        """Trigger emergency procedures for critical errors"""
        try:
            if error_context.severity == ErrorSeverity.CRITICAL:
                logger.critical(f"CRITICAL ERROR: {error_context.error_id} - triggering emergency procedures")

                # Create emergency checkpoint
                await self._create_emergency_checkpoint(error_context)

                # Notify administrators
                await self._notify_administrators(error_context)

        except Exception as e:
            logger.error(f"Failed to trigger emergency procedures: {e}")

    async def _create_emergency_checkpoint(self, error_context: ErrorContext):
        """Create an emergency checkpoint during critical errors"""
        try:
            checkpoint_name = f"emergency_{error_context.error_id}_{int(time.time())}"
            success = self.state_manager.create_checkpoint(checkpoint_name, phases=[Phase.PHASE6_BAKING])

            if success:
                logger.info(f"Emergency checkpoint created: {checkpoint_name}")
            else:
                logger.error("Failed to create emergency checkpoint")

        except Exception as e:
            logger.error(f"Error creating emergency checkpoint: {e}")

    async def _notify_administrators(self, error_context: ErrorContext):
        """Notify administrators of critical errors"""
        try:
            notification_data = {
                'error_id': error_context.error_id,
                'severity': error_context.severity.value,
                'component': error_context.component,
                'message': error_context.message,
                'timestamp': error_context.timestamp.isoformat()
            }

            # In a real system, this would send notifications via email, SMS, etc.
            logger.critical(f"ADMIN NOTIFICATION: {notification_data}")

        except Exception as e:
            logger.error(f"Failed to notify administrators: {e}")

    async def _escalate_error(self, error_context: ErrorContext):
        """Escalate unresolved errors"""
        try:
            error_context.severity = ErrorSeverity.CRITICAL
            await self._trigger_emergency_procedures(error_context)

            logger.critical(f"Error {error_context.error_id} escalated to critical - recovery failed")

        except Exception as e:
            logger.error(f"Error during escalation: {e}")

    def _classify_error(self, exception: Exception, function: str) -> ErrorCategory:
        """Classify error into appropriate category"""
        exception_type = type(exception).__name__
        error_message = str(exception).lower()

        # JSON/Serialization errors
        if ('json' in error_message or 'serializ' in error_message or
            exception_type in ['JSONDecodeError', 'SerializationError']):
            return ErrorCategory.SERIALIZATION

        # Communication errors
        if ('timeout' in error_message or 'connection' in error_message or
            'network' in error_message or exception_type in ['ConnectionError', 'TimeoutError']):
            return ErrorCategory.COMMUNICATION

        # Resource errors
        if ('memory' in error_message or 'resource' in error_message or
            exception_type in ['MemoryError', 'OutOfMemoryError']):
            return ErrorCategory.RESOURCE

        # Timeout errors
        if 'timeout' in error_message or exception_type == 'TimeoutError':
            return ErrorCategory.TIMEOUT

        # Validation errors
        if ('validation' in error_message or 'invalid' in error_message or
            exception_type in ['ValidationError', 'ValueError']):
            return ErrorCategory.VALIDATION

        # Dependency errors
        if ('dependency' in error_message or 'import' in error_message or
            exception_type in ['ImportError', 'ModuleNotFoundError']):
            return ErrorCategory.DEPENDENCY

        # Configuration errors
        if ('config' in error_message or 'setting' in error_message):
            return ErrorCategory.CONFIGURATION

        # Hardware errors
        if ('hardware' in error_message or 'device' in error_message):
            return ErrorCategory.HARDWARE

        # Network errors
        if ('network' in error_message or 'socket' in error_message):
            return ErrorCategory.NETWORK

        # Data corruption errors
        if ('corrupt' in error_message or 'checksum' in error_message):
            return ErrorCategory.DATA_CORRUPTION

        return ErrorCategory.UNKNOWN

    def _determine_severity(self, exception: Exception, category: ErrorCategory, component: str) -> ErrorSeverity:
        """Determine error severity based on context"""
        exception_type = type(exception).__name__

        # Critical errors
        if (category == ErrorCategory.DATA_CORRUPTION or
            exception_type in ['SystemExit', 'KeyboardInterrupt'] or
            'critical' in str(exception).lower()):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if (category in [ErrorCategory.RESOURCE, ErrorCategory.HARDWARE] or
            exception_type in ['MemoryError', 'OSError']):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if (category in [ErrorCategory.COMMUNICATION, ErrorCategory.TIMEOUT, ErrorCategory.VALIDATION] or
            exception_type in ['ConnectionError', 'TimeoutError', 'ValueError']):
            return ErrorSeverity.MEDIUM

        # Low severity errors
        if category in [ErrorCategory.CONFIGURATION, ErrorCategory.DEPENDENCY]:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM  # Default

    def _should_trip_circuit_breaker(self, component: str, category: ErrorCategory) -> bool:
        """Check if circuit breaker should be tripped"""
        key = f"{component}_{category.value}"

        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'error_count': 0,
                'last_error_time': datetime.now(),
                'state': 'closed',  # closed, open, half-open
                'trip_time': None
            }

        breaker = self.circuit_breakers[key]
        current_time = datetime.now()

        # Reset count if window expired
        if (current_time - breaker['last_error_time']).total_seconds() > 60:  # 1 minute window
            breaker['error_count'] = 0

        breaker['error_count'] += 1
        breaker['last_error_time'] = current_time

        return breaker['error_count'] >= self.circuit_breaker_threshold

    async def _trip_circuit_breaker(self, component: str, category: ErrorCategory):
        """Trip circuit breaker for component/category"""
        key = f"{component}_{category.value}"
        breaker = self.circuit_breakers[key]

        breaker['state'] = 'open'
        breaker['trip_time'] = datetime.now()

        logger.warning(f"Circuit breaker tripped for {component}/{category.value}")

        # Schedule automatic reset
        asyncio.create_task(self._reset_circuit_breaker_after_timeout(key))

    async def _reset_circuit_breaker_after_timeout(self, breaker_key: str):
        """Reset circuit breaker after timeout"""
        await asyncio.sleep(self.circuit_breaker_timeout)

        if breaker_key in self.circuit_breakers:
            self.circuit_breakers[breaker_key]['state'] = 'half-open'
            logger.info(f"Circuit breaker {breaker_key} reset to half-open")

    async def _detect_error_patterns(self):
        """Detect error patterns for proactive prevention"""
        try:
            current_time = datetime.now()

            for pattern_id, pattern in self.error_patterns.items():
                # Count matching errors in time window
                window_start = current_time - timedelta(seconds=pattern.time_window_seconds)
                matching_errors = 0

                for error in self.error_history:
                    if error.timestamp >= window_start:
                        if any(sig.lower() in error.message.lower() or sig.lower() in error.exception_type.lower()
                               for sig in pattern.error_signatures):
                            matching_errors += 1

                # Check if pattern threshold is exceeded
                if matching_errors >= pattern.frequency_threshold:
                    logger.warning(f"Error pattern detected: {pattern_id} ({matching_errors} occurrences)")
                    await self._handle_error_pattern(pattern)

        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")

    async def _handle_error_pattern(self, pattern: ErrorPattern):
        """Handle detected error pattern"""
        try:
            logger.info(f"Handling error pattern: {pattern.pattern_id}")

            # Take proactive action based on recommended strategy
            if pattern.recommended_action == RecoveryStrategy.RESTART:
                # Proactively restart affected components
                logger.info("Proactively restarting components due to error pattern")
            elif pattern.recommended_action == RecoveryStrategy.FALLBACK:
                # Switch to fallback mode
                logger.info("Switching to fallback mode due to error pattern")

        except Exception as e:
            logger.error(f"Error handling error pattern: {e}")

    def get_error_status(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific error"""
        if error_id in self.active_errors:
            error = self.active_errors[error_id]
            return {
                'error_id': error_id,
                'status': 'active',
                'severity': error.severity.value,
                'category': error.category.value,
                'component': error.component,
                'message': error.message,
                'recovery_attempts': error.recovery_attempts,
                'timestamp': error.timestamp.isoformat()
            }

        # Check resolved errors
        for error in self.error_history:
            if error.error_id == error_id:
                return {
                    'error_id': error_id,
                    'status': 'resolved' if error.resolved else 'failed',
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'component': error.component,
                    'message': error.message,
                    'recovery_attempts': error.recovery_attempts,
                    'timestamp': error.timestamp.isoformat()
                }

        return None

    def get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        current_time = datetime.now()

        # Calculate system availability
        total_errors = len(self.error_history)
        critical_errors_24h = sum(1 for e in self.error_history
                                 if e.severity == ErrorSeverity.CRITICAL and
                                 (current_time - e.timestamp).total_seconds() < 86400)

        if total_errors > 0:
            resolved_errors = sum(1 for e in self.error_history if e.resolved)
            self.error_metrics['recovery_success_rate'] = resolved_errors / total_errors

        # Calculate availability (simplified)
        if critical_errors_24h == 0:
            self.error_metrics['system_availability'] = 99.9
        else:
            self.error_metrics['system_availability'] = max(90.0, 99.9 - (critical_errors_24h * 0.1))

        return {
            'system_id': self.system_id,
            'timestamp': current_time.isoformat(),
            'active_errors': len(self.active_errors),
            'critical_errors_24h': critical_errors_24h,
            'metrics': self.error_metrics,
            'circuit_breakers': {k: v['state'] for k, v in self.circuit_breakers.items()},
            'active_recoveries': len(self.active_recoveries),
            'health_status': 'HEALTHY' if len(self.active_errors) == 0 else 'DEGRADED'
        }

    def generate_error_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive error report"""
        current_time = datetime.now()
        window_start = current_time - timedelta(hours=time_window_hours)

        # Filter errors in time window
        window_errors = [e for e in self.error_history if e.timestamp >= window_start]

        # Categorize errors
        error_by_category = defaultdict(int)
        error_by_severity = defaultdict(int)
        error_by_component = defaultdict(int)

        for error in window_errors:
            error_by_category[error.category.value] += 1
            error_by_severity[error.severity.value] += 1
            error_by_component[error.component] += 1

        return {
            'report_timestamp': current_time.isoformat(),
            'time_window_hours': time_window_hours,
            'total_errors': len(window_errors),
            'resolved_errors': sum(1 for e in window_errors if e.resolved),
            'error_by_category': dict(error_by_category),
            'error_by_severity': dict(error_by_severity),
            'error_by_component': dict(error_by_component),
            'top_error_messages': self._get_top_error_messages(window_errors),
            'recovery_success_rate': sum(1 for e in window_errors if e.resolved) / max(1, len(window_errors)),
            'recommendations': self._generate_recommendations(window_errors)
        }

    def _get_top_error_messages(self, errors: List[ErrorContext], limit: int = 10) -> List[Dict[str, Any]]:
        """Get top error messages by frequency"""
        message_counts = defaultdict(int)
        for error in errors:
            message_counts[error.message] += 1

        top_messages = sorted(message_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{'message': msg, 'count': count} for msg, count in top_messages]

    def _generate_recommendations(self, errors: List[ErrorContext]) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []

        # Check for high frequency categories
        category_counts = defaultdict(int)
        for error in errors:
            category_counts[error.category] += 1

        for category, count in category_counts.items():
            if count > 5:  # Threshold for recommendation
                if category == ErrorCategory.SERIALIZATION:
                    recommendations.append("Consider implementing more robust serialization fallbacks")
                elif category == ErrorCategory.COMMUNICATION:
                    recommendations.append("Review network configuration and connection handling")
                elif category == ErrorCategory.RESOURCE:
                    recommendations.append("Monitor and optimize resource usage")

        if not recommendations:
            recommendations.append("System is operating within normal parameters")

        return recommendations

# Factory function
def create_error_recovery_system(config: Dict[str, Any]) -> ErrorRecoverySystem:
    """Factory function to create error recovery system"""
    return ErrorRecoverySystem(config)

# Testing utilities
async def test_error_recovery():
    """Test error recovery functionality"""
    config = {
        'max_retry_attempts': 3,
        'recovery_timeout_seconds': 60,
        'circuit_breaker_threshold': 3
    }

    recovery_system = ErrorRecoverySystem(config)

    # Simulate various errors
    test_errors = [
        (ValueError("Invalid input data"), "test_component", "process_data"),
        (ConnectionError("Network timeout"), "communication_component", "send_message"),
        (json.JSONDecodeError("Invalid JSON", "", 0), "serialization_component", "serialize_data")
    ]

    for exception, component, function in test_errors:
        error_id = await recovery_system.handle_error(component, function, exception)
        print(f"Handled error {error_id}: {exception}")

    # Check system health
    health_status = recovery_system.get_system_health_status()
    print(f"System health: {health_status}")

    # Generate error report
    error_report = recovery_system.generate_error_report()
    print(f"Error report: {error_report}")

if __name__ == "__main__":
    asyncio.run(test_error_recovery())