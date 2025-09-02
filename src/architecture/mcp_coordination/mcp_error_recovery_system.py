"""
MCP Error Recovery System - Systematic Failure Detection and Automatic Recovery

This module provides comprehensive error detection, analysis, and recovery for MCP server
coordination in CI/CD pipelines. Designed to handle the complex failure modes that can
occur in distributed MCP environments.

Key Features:
- Multi-layered error detection (network, protocol, application level)
- Automatic recovery strategies with progressive escalation
- Circuit breaker patterns for cascade failure prevention
- Error pattern analysis and learning
- Comprehensive failure audit trails
- Integration with monitoring and alerting systems
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Union
from uuid import uuid4

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class ErrorCategory(Enum):
    """Categories of errors that can occur"""
    NETWORK_ERROR = "network_error"
    PROTOCOL_ERROR = "protocol_error"
    AUTHENTICATION_ERROR = "authentication_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    APPLICATION_ERROR = "application_error"
    CONFIGURATION_ERROR = "configuration_error"
    DEPENDENCY_ERROR = "dependency_error"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    RESTART = "restart"
    FAILOVER = "failover"
    DEGRADE = "degrade"
    ESCALATE = "escalate"
    MANUAL = "manual"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing if recovery is possible


@dataclass
class ErrorEvent:
    """Represents an error event in the system"""
    error_id: str = field(default_factory=lambda: f"error_{uuid4().hex[:8]}")
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.APPLICATION_ERROR
    
    # Error context
    component: str = "unknown"
    server_id: Optional[str] = None
    operation: str = "unknown"
    
    # Error details
    message: str = ""
    exception_type: str = ""
    stack_trace: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery tracking
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for preventing cascade failures"""
    name: str
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    
    # State tracking
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None
    half_open_calls: int = 0
    
    # Metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    
    def should_allow_call(self) -> bool:
        """Check if a call should be allowed through the circuit breaker"""
        current_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self.next_attempt_time and current_time >= self.next_attempt_time:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record a successful call"""
        self.total_calls += 1
        self.successful_calls += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                # Enough successful calls, close the circuit
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed call"""
        self.total_calls += 1
        self.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                # Open the circuit
                self.state = CircuitState.OPEN
                self.next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)
        elif self.state == CircuitState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.state = CircuitState.OPEN
            self.next_attempt_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)


class ErrorPattern:
    """Analyzes and tracks error patterns"""
    
    def __init__(self, window_minutes: int = 10):
        self.window_minutes = window_minutes
        self.error_history: Deque[ErrorEvent] = deque(maxlen=1000)
        self.pattern_cache: Dict[str, Any] = {}
        self.last_analysis = datetime.now()
    
    def add_error(self, error: ErrorEvent):
        """Add an error to the pattern analysis"""
        self.error_history.append(error)
        
        # Clear cache when new data arrives
        if len(self.error_history) % 10 == 0:
            self.pattern_cache.clear()
    
    def get_error_rate(self, component: str = None, category: ErrorCategory = None) -> float:
        """Get error rate for a component or category"""
        cache_key = f"rate_{component}_{category}_{self.window_minutes}"
        
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        relevant_errors = [
            e for e in self.error_history 
            if e.timestamp >= cutoff_time and
            (component is None or e.component == component) and
            (category is None or e.category == category)
        ]
        
        # Calculate rate per minute
        rate = len(relevant_errors) / max(self.window_minutes, 1)
        self.pattern_cache[cache_key] = rate
        return rate
    
    def get_top_error_sources(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top error sources by frequency"""
        cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        source_counts = defaultdict(int)
        for error in recent_errors:
            source = f"{error.component}:{error.category.value}"
            source_counts[source] += 1
        
        return [
            {"source": source, "count": count}
            for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        ]
    
    def detect_error_spikes(self, threshold_multiplier: float = 3.0) -> List[Dict[str, Any]]:
        """Detect sudden spikes in error rates"""
        current_rate = self.get_error_rate()
        
        # Compare with historical average
        historical_cutoff = datetime.now() - timedelta(minutes=self.window_minutes * 3)
        historical_errors = [
            e for e in self.error_history 
            if e.timestamp >= historical_cutoff and e.timestamp <= datetime.now() - timedelta(minutes=self.window_minutes)
        ]
        
        historical_rate = len(historical_errors) / max(self.window_minutes * 2, 1)
        
        spikes = []
        if current_rate > historical_rate * threshold_multiplier:
            spikes.append({
                "type": "overall_spike",
                "current_rate": current_rate,
                "historical_rate": historical_rate,
                "multiplier": current_rate / max(historical_rate, 0.01)
            })
        
        return spikes


class RecoveryExecutor:
    """Executes recovery strategies for different types of errors"""
    
    def __init__(self):
        self.recovery_handlers: Dict[RecoveryStrategy, Callable] = {}
        self.register_default_handlers()
        self.recovery_history: List[Dict[str, Any]] = []
    
    def register_default_handlers(self):
        """Register default recovery handlers"""
        self.recovery_handlers[RecoveryStrategy.RETRY] = self._retry_handler
        self.recovery_handlers[RecoveryStrategy.RESTART] = self._restart_handler
        self.recovery_handlers[RecoveryStrategy.FAILOVER] = self._failover_handler
        self.recovery_handlers[RecoveryStrategy.DEGRADE] = self._degrade_handler
        self.recovery_handlers[RecoveryStrategy.ESCALATE] = self._escalate_handler
    
    async def execute_recovery(self, error: ErrorEvent, strategy: RecoveryStrategy) -> bool:
        """Execute a recovery strategy for an error"""
        start_time = time.perf_counter()
        
        try:
            handler = self.recovery_handlers.get(strategy)
            if not handler:
                logger.error(f"No recovery handler for strategy: {strategy}")
                return False
            
            success = await handler(error)
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Record recovery attempt
            self.recovery_history.append({
                "error_id": error.error_id,
                "strategy": strategy.value,
                "success": success,
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat()
            })
            
            if success:
                error.recovery_successful = True
                error.resolved_at = datetime.now()
                logger.info(f"Recovery successful for error {error.error_id} using {strategy.value}")
            else:
                logger.warning(f"Recovery failed for error {error.error_id} using {strategy.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
    
    async def _retry_handler(self, error: ErrorEvent) -> bool:
        """Retry the failed operation"""
        max_retries = 3
        delay_seconds = [1, 2, 4]  # Exponential backoff
        
        for attempt in range(max_retries):
            try:
                # Wait before retry
                if attempt > 0:
                    await asyncio.sleep(delay_seconds[min(attempt - 1, len(delay_seconds) - 1)])
                
                # Simulate retry of the original operation
                success = await self._simulate_operation_retry(error)
                
                if success:
                    logger.info(f"Retry successful for {error.component}:{error.operation} (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(f"Retry failed for {error.component}:{error.operation} (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.error(f"Retry attempt {attempt + 1} failed: {e}")
        
        return False
    
    async def _restart_handler(self, error: ErrorEvent) -> bool:
        """Restart the failed component"""
        try:
            logger.info(f"Restarting component: {error.component}")
            
            # Simulate component restart
            await asyncio.sleep(2.0)  # Simulate restart time
            
            # Verify restart was successful
            success = await self._verify_component_health(error.component)
            
            if success:
                logger.info(f"Component restart successful: {error.component}")
            else:
                logger.error(f"Component restart failed: {error.component}")
            
            return success
            
        except Exception as e:
            logger.error(f"Restart handler failed: {e}")
            return False
    
    async def _failover_handler(self, error: ErrorEvent) -> bool:
        """Failover to backup component or server"""
        try:
            logger.info(f"Initiating failover for: {error.component}")
            
            # Find backup server/component
            backup_component = await self._find_backup_component(error.component)
            
            if backup_component:
                # Switch to backup
                await self._activate_backup_component(backup_component)
                logger.info(f"Failover successful to: {backup_component}")
                return True
            else:
                logger.warning(f"No backup component available for: {error.component}")
                return False
                
        except Exception as e:
            logger.error(f"Failover handler failed: {e}")
            return False
    
    async def _degrade_handler(self, error: ErrorEvent) -> bool:
        """Degrade functionality to maintain core operations"""
        try:
            logger.info(f"Degrading functionality for: {error.component}")
            
            # Activate degraded mode
            degradation_config = await self._get_degradation_config(error.component)
            await self._apply_degradation(error.component, degradation_config)
            
            logger.info(f"Degradation applied for: {error.component}")
            return True
            
        except Exception as e:
            logger.error(f"Degradation handler failed: {e}")
            return False
    
    async def _escalate_handler(self, error: ErrorEvent) -> bool:
        """Escalate to human intervention"""
        try:
            logger.critical(f"Escalating error for manual intervention: {error.error_id}")
            
            # Create escalation report
            escalation_report = {
                "error_id": error.error_id,
                "severity": error.severity.value,
                "component": error.component,
                "message": error.message,
                "context": error.context_data,
                "escalated_at": datetime.now().isoformat()
            }
            
            # Send to monitoring/alerting system
            await self._send_escalation_alert(escalation_report)
            
            logger.info(f"Escalation initiated for error: {error.error_id}")
            return True
            
        except Exception as e:
            logger.error(f"Escalation handler failed: {e}")
            return False
    
    # Simulation methods (would be replaced with real implementations)
    
    async def _simulate_operation_retry(self, error: ErrorEvent) -> bool:
        """Simulate retrying the original operation"""
        # For demo, simulate 70% success rate on retry
        import random
        return random.random() > 0.3
    
    async def _verify_component_health(self, component: str) -> bool:
        """Verify that a component is healthy after restart"""
        # Simulate health check
        await asyncio.sleep(0.5)
        import random
        return random.random() > 0.2  # 80% success rate
    
    async def _find_backup_component(self, component: str) -> Optional[str]:
        """Find a backup component for failover"""
        # Simulate finding backup
        backup_map = {
            "memory_server": "memory_server_backup",
            "github_server": "github_server_backup",
            "hyperag_server": "hyperag_server_backup"
        }
        return backup_map.get(component)
    
    async def _activate_backup_component(self, backup_component: str):
        """Activate a backup component"""
        await asyncio.sleep(1.0)  # Simulate activation time
    
    async def _get_degradation_config(self, component: str) -> Dict[str, Any]:
        """Get degradation configuration for a component"""
        return {
            "reduce_functionality": True,
            "use_cache_only": True,
            "disable_non_critical_features": True
        }
    
    async def _apply_degradation(self, component: str, config: Dict[str, Any]):
        """Apply degradation configuration"""
        await asyncio.sleep(0.5)  # Simulate configuration application
    
    async def _send_escalation_alert(self, report: Dict[str, Any]):
        """Send escalation alert to monitoring system"""
        # In real implementation, this would integrate with alerting systems
        logger.critical(f"ESCALATION ALERT: {json.dumps(report, indent=2)}")


class MCPErrorRecoverySystem:
    """
    Comprehensive Error Recovery System for MCP Coordination
    
    Provides:
    - Multi-layered error detection and classification
    - Circuit breaker patterns for cascade failure prevention
    - Automatic recovery with progressive strategy escalation
    - Error pattern analysis and learning
    - Comprehensive audit trails and monitoring integration
    """
    
    def __init__(self, memory_client=None):
        self.memory_client = memory_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Error tracking and analysis
        self.error_events: Dict[str, ErrorEvent] = {}
        self.error_pattern = ErrorPattern()
        self.recovery_executor = RecoveryExecutor()
        
        # Circuit breakers for different components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Recovery configuration
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.max_recovery_attempts = 3
        self.escalation_threshold_minutes = 15
        
        # Monitoring
        self.error_detection_active = True
        self.recovery_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            "total_errors": 0,
            "recovered_errors": 0,
            "escalated_errors": 0,
            "circuit_breaker_trips": 0,
            "recovery_success_rate": 0.0
        }
        
        self.logger.info("MCP Error Recovery System initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorCategory, List[RecoveryStrategy]]:
        """Initialize default recovery strategies for different error types"""
        return {
            ErrorCategory.NETWORK_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.FAILOVER, RecoveryStrategy.ESCALATE],
            ErrorCategory.PROTOCOL_ERROR: [RecoveryStrategy.RESTART, RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE],
            ErrorCategory.AUTHENTICATION_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.RESTART, RecoveryStrategy.ESCALATE],
            ErrorCategory.TIMEOUT_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.DEGRADE, RecoveryStrategy.FAILOVER],
            ErrorCategory.RESOURCE_ERROR: [RecoveryStrategy.DEGRADE, RecoveryStrategy.RESTART, RecoveryStrategy.ESCALATE],
            ErrorCategory.APPLICATION_ERROR: [RecoveryStrategy.RETRY, RecoveryStrategy.RESTART, RecoveryStrategy.ESCALATE],
            ErrorCategory.CONFIGURATION_ERROR: [RecoveryStrategy.RESTART, RecoveryStrategy.ESCALATE],
            ErrorCategory.DEPENDENCY_ERROR: [RecoveryStrategy.FAILOVER, RecoveryStrategy.DEGRADE, RecoveryStrategy.ESCALATE]
        }
    
    async def initialize(self) -> bool:
        """Initialize the error recovery system"""
        try:
            # Start recovery monitoring
            self.recovery_task = asyncio.create_task(self._recovery_monitoring_loop())
            
            self.logger.info("Error recovery system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error recovery system: {e}")
            return False
    
    async def report_error(self, component: str, operation: str, exception: Exception, 
                          server_id: str = None, context: Dict[str, Any] = None) -> str:
        """Report an error to the recovery system"""
        try:
            # Classify the error
            error_category = self._classify_error(exception)
            severity = self._assess_severity(error_category, exception)
            
            # Create error event
            error_event = ErrorEvent(
                severity=severity,
                category=error_category,
                component=component,
                server_id=server_id,
                operation=operation,
                message=str(exception),
                exception_type=type(exception).__name__,
                stack_trace=self._extract_stack_trace(exception),
                context_data=context or {}
            )
            
            # Store error event
            self.error_events[error_event.error_id] = error_event
            self.error_pattern.add_error(error_event)
            self.stats["total_errors"] += 1
            
            # Update circuit breaker
            circuit_breaker = self._get_circuit_breaker(component)
            circuit_breaker.record_failure()
            
            if circuit_breaker.state == CircuitState.OPEN:
                self.stats["circuit_breaker_trips"] += 1
                self.logger.warning(f"Circuit breaker opened for component: {component}")
            
            # Store in memory for persistence
            await self._store_error_in_memory(error_event)
            
            self.logger.error(f"Error reported: {error_event.error_id} - {component}:{operation} - {exception}")
            
            # Trigger immediate recovery if critical
            if severity in [ErrorSeverity.CRITICAL, ErrorSeverity.CATASTROPHIC]:
                asyncio.create_task(self._attempt_immediate_recovery(error_event))
            
            return error_event.error_id
            
        except Exception as e:
            self.logger.error(f"Failed to report error: {e}")
            return ""
    
    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify an error into a category"""
        exception_type = type(exception).__name__
        message = str(exception).lower()
        
        if "network" in message or "connection" in message or exception_type in ["ConnectionError", "NetworkError"]:
            return ErrorCategory.NETWORK_ERROR
        elif "timeout" in message or exception_type in ["TimeoutError", "asyncio.TimeoutError"]:
            return ErrorCategory.TIMEOUT_ERROR
        elif "auth" in message or "permission" in message or exception_type in ["AuthenticationError", "PermissionError"]:
            return ErrorCategory.AUTHENTICATION_ERROR
        elif "protocol" in message or "json" in message or exception_type in ["ProtocolError", "JSONDecodeError"]:
            return ErrorCategory.PROTOCOL_ERROR
        elif "resource" in message or "memory" in message or exception_type in ["MemoryError", "ResourceError"]:
            return ErrorCategory.RESOURCE_ERROR
        elif "config" in message or exception_type in ["ConfigurationError", "ValueError"]:
            return ErrorCategory.CONFIGURATION_ERROR
        elif "dependency" in message or exception_type in ["ImportError", "ModuleNotFoundError"]:
            return ErrorCategory.DEPENDENCY_ERROR
        else:
            return ErrorCategory.APPLICATION_ERROR
    
    def _assess_severity(self, category: ErrorCategory, exception: Exception) -> ErrorSeverity:
        """Assess the severity of an error"""
        # High severity categories
        if category in [ErrorCategory.AUTHENTICATION_ERROR, ErrorCategory.CONFIGURATION_ERROR]:
            return ErrorSeverity.HIGH
        
        # Critical system errors
        if category == ErrorCategory.RESOURCE_ERROR:
            return ErrorSeverity.CRITICAL
        
        # Check exception type
        exception_type = type(exception).__name__
        if exception_type in ["MemoryError", "SystemExit", "KeyboardInterrupt"]:
            return ErrorSeverity.CATASTROPHIC
        elif exception_type in ["ConnectionError", "TimeoutError"]:
            return ErrorSeverity.HIGH
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _extract_stack_trace(self, exception: Exception) -> str:
        """Extract stack trace from exception"""
        import traceback
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    def _get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(
                name=component,
                failure_threshold=5,
                recovery_timeout_seconds=60
            )
        
        return self.circuit_breakers[component]
    
    async def check_circuit_breaker(self, component: str) -> bool:
        """Check if operation should be allowed through circuit breaker"""
        circuit_breaker = self._get_circuit_breaker(component)
        return circuit_breaker.should_allow_call()
    
    async def record_success(self, component: str):
        """Record a successful operation"""
        circuit_breaker = self._get_circuit_breaker(component)
        circuit_breaker.record_success()
    
    async def _attempt_immediate_recovery(self, error_event: ErrorEvent):
        """Attempt immediate recovery for critical errors"""
        strategies = self.recovery_strategies.get(error_event.category, [RecoveryStrategy.ESCALATE])
        
        for strategy in strategies:
            try:
                self.logger.info(f"Attempting {strategy.value} recovery for critical error: {error_event.error_id}")
                
                success = await self.recovery_executor.execute_recovery(error_event, strategy)
                error_event.recovery_attempts += 1
                error_event.recovery_strategy = strategy
                
                if success:
                    self.stats["recovered_errors"] += 1
                    self._update_success_rate()
                    break
                    
            except Exception as e:
                self.logger.error(f"Recovery attempt failed: {e}")
    
    async def _recovery_monitoring_loop(self):
        """Monitor and attempt recovery for accumulated errors"""
        while self.error_detection_active:
            try:
                await self._analyze_error_patterns()
                await self._attempt_pending_recoveries()
                await self._check_escalation_conditions()
                await self._cleanup_resolved_errors()
                
                await asyncio.sleep(10.0)  # Recovery check every 10 seconds
                
            except asyncio.CancelledError:
                self.logger.info("Recovery monitoring cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in recovery monitoring loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _analyze_error_patterns(self):
        """Analyze error patterns and detect anomalies"""
        # Check for error spikes
        spikes = self.error_pattern.detect_error_spikes()
        
        for spike in spikes:
            self.logger.warning(f"Error spike detected: {spike}")
            
            # Create synthetic error for spike pattern
            spike_error = ErrorEvent(
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.APPLICATION_ERROR,
                component="system",
                operation="error_pattern_analysis",
                message=f"Error spike detected: {spike['multiplier']:.2f}x increase"
            )
            
            # Attempt recovery for spike condition
            asyncio.create_task(self._attempt_immediate_recovery(spike_error))
        
        # Get top error sources
        top_sources = self.error_pattern.get_top_error_sources()
        if top_sources:
            self.logger.info(f"Top error sources: {top_sources}")
    
    async def _attempt_pending_recoveries(self):
        """Attempt recovery for unresolved errors"""
        current_time = datetime.now()
        
        for error_event in self.error_events.values():
            if (not error_event.recovery_successful and 
                error_event.recovery_attempts < self.max_recovery_attempts and
                error_event.severity != ErrorSeverity.LOW):
                
                # Don't retry too frequently
                if error_event.recovery_attempts > 0:
                    time_since_last = current_time - error_event.timestamp
                    min_wait = timedelta(minutes=error_event.recovery_attempts * 2)
                    if time_since_last < min_wait:
                        continue
                
                # Attempt recovery
                strategies = self.recovery_strategies.get(error_event.category, [])
                if error_event.recovery_attempts < len(strategies):
                    strategy = strategies[error_event.recovery_attempts]
                    
                    success = await self.recovery_executor.execute_recovery(error_event, strategy)
                    error_event.recovery_attempts += 1
                    error_event.recovery_strategy = strategy
                    
                    if success:
                        self.stats["recovered_errors"] += 1
                        self._update_success_rate()
    
    async def _check_escalation_conditions(self):
        """Check if errors should be escalated"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=self.escalation_threshold_minutes)
        
        unresolved_critical_errors = [
            error for error in self.error_events.values()
            if (not error.recovery_successful and 
                error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.CATASTROPHIC] and
                error.timestamp >= cutoff_time and
                error.recovery_attempts >= self.max_recovery_attempts)
        ]
        
        for error in unresolved_critical_errors:
            if not error.recovery_strategy or error.recovery_strategy != RecoveryStrategy.ESCALATE:
                await self.recovery_executor.execute_recovery(error, RecoveryStrategy.ESCALATE)
                error.recovery_strategy = RecoveryStrategy.ESCALATE
                self.stats["escalated_errors"] += 1
    
    async def _cleanup_resolved_errors(self):
        """Clean up resolved errors from memory"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=1)  # Keep errors for 1 hour
        
        resolved_errors = [
            error_id for error_id, error in self.error_events.items()
            if (error.recovery_successful and 
                error.resolved_at and 
                error.resolved_at < cutoff_time)
        ]
        
        for error_id in resolved_errors:
            del self.error_events[error_id]
    
    def _update_success_rate(self):
        """Update recovery success rate"""
        total_recovery_attempts = self.stats["recovered_errors"] + (self.stats["total_errors"] - self.stats["recovered_errors"])
        if total_recovery_attempts > 0:
            self.stats["recovery_success_rate"] = self.stats["recovered_errors"] / total_recovery_attempts
    
    async def _store_error_in_memory(self, error_event: ErrorEvent):
        """Store error event in memory for persistence"""
        if not self.memory_client:
            return
        
        try:
            error_data = {
                "error_id": error_event.error_id,
                "timestamp": error_event.timestamp.isoformat(),
                "severity": error_event.severity.value,
                "category": error_event.category.value,
                "component": error_event.component,
                "server_id": error_event.server_id,
                "operation": error_event.operation,
                "message": error_event.message,
                "exception_type": error_event.exception_type,
                "context_data": error_event.context_data,
                "recovery_attempts": error_event.recovery_attempts
            }
            
            if hasattr(self.memory_client, 'store'):
                await self.memory_client.store(
                    f"error_{error_event.error_id}",
                    error_data,
                    "error_recovery_system"
                )
                
        except Exception as e:
            self.logger.warning(f"Failed to store error in memory: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the error recovery system"""
        current_time = datetime.now()
        
        # Circuit breaker status
        circuit_status = {}
        for name, breaker in self.circuit_breakers.items():
            circuit_status[name] = {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "success_rate": breaker.successful_calls / max(breaker.total_calls, 1),
                "last_failure": breaker.last_failure_time.isoformat() if breaker.last_failure_time else None
            }
        
        # Recent error summary
        recent_errors = [
            error for error in self.error_events.values()
            if error.timestamp >= current_time - timedelta(minutes=60)
        ]
        
        error_summary = {
            "total_recent": len(recent_errors),
            "by_severity": {},
            "by_category": {},
            "unresolved": len([e for e in recent_errors if not e.recovery_successful])
        }
        
        for error in recent_errors:
            severity = error.severity.name
            category = error.category.value
            
            error_summary["by_severity"][severity] = error_summary["by_severity"].get(severity, 0) + 1
            error_summary["by_category"][category] = error_summary["by_category"].get(category, 0) + 1
        
        return {
            "system_status": {
                "error_detection_active": self.error_detection_active,
                "recovery_task_running": self.recovery_task and not self.recovery_task.done(),
                "total_circuit_breakers": len(self.circuit_breakers),
                "active_errors": len(self.error_events)
            },
            "statistics": self.stats.copy(),
            "circuit_breakers": circuit_status,
            "recent_errors": error_summary,
            "error_patterns": {
                "current_error_rate": self.error_pattern.get_error_rate(),
                "top_sources": self.error_pattern.get_top_error_sources(3)
            }
        }
    
    async def shutdown(self):
        """Shutdown the error recovery system"""
        self.logger.info("Shutting down error recovery system...")
        
        self.error_detection_active = False
        
        if self.recovery_task and not self.recovery_task.done():
            self.recovery_task.cancel()
            try:
                await self.recovery_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Error recovery system shutdown complete")


# Example usage and testing
async def example_error_recovery_usage():
    """Example of how to use the MCP Error Recovery System"""
    
    # Create error recovery system
    recovery_system = MCPErrorRecoverySystem()
    await recovery_system.initialize()
    
    # Simulate some errors
    try:
        raise ConnectionError("Failed to connect to MCP server")
    except Exception as e:
        error_id = await recovery_system.report_error("memory_server", "connect", e, context={"retry_count": 0})
        print(f"Reported error: {error_id}")
    
    try:
        raise TimeoutError("Operation timed out after 30 seconds")
    except Exception as e:
        error_id = await recovery_system.report_error("github_server", "api_call", e)
        print(f"Reported error: {error_id}")
    
    # Check circuit breaker
    can_proceed = await recovery_system.check_circuit_breaker("memory_server")
    print(f"Can proceed with memory_server operations: {can_proceed}")
    
    # Let recovery system work
    await asyncio.sleep(5.0)
    
    # Get status
    status = recovery_system.get_system_status()
    print(f"Recovery system status: {json.dumps(status, indent=2, default=str)}")
    
    await recovery_system.shutdown()


if __name__ == "__main__":
    asyncio.run(example_error_recovery_usage())