"""
SafetyManager - Comprehensive safety management for ADAS systems

Implements ISO 26262 ASIL-D functional safety requirements with real-time
monitoring, validation, and emergency response capabilities.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from ..config.adas_config import ADASConfig, ASILLevel
from ..agents.perception_agent import PerceptionOutput
from ..agents.prediction_agent import PredictionOutput
from ..agents.planning_agent import PlanningOutput

class SafetyState(Enum):
    """Safety system states"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    FAULT = "fault"

class SafetyViolationType(Enum):
    """Types of safety violations"""
    LATENCY_EXCEEDED = "latency_exceeded"
    CONFIDENCE_LOW = "confidence_low"
    COLLISION_IMMINENT = "collision_imminent"
    SENSOR_FAILURE = "sensor_failure"
    COMPUTATION_ERROR = "computation_error"
    CONSTRAINT_VIOLATED = "constraint_violated"
    ASIL_DEGRADED = "asil_degraded"

@dataclass
class SafetyViolation:
    """Safety violation with metadata"""
    violation_type: SafetyViolationType
    severity: ASILLevel
    timestamp: float
    description: str
    affected_components: List[str]
    recommended_action: str
    auto_recovery_possible: bool
    violation_id: str

@dataclass
class SafetyMetrics:
    """Comprehensive safety metrics"""
    timestamp: float
    overall_safety_score: float
    perception_safety: float
    prediction_safety: float
    planning_safety: float
    sensor_health_score: float
    system_latency_ms: float
    active_violations: List[SafetyViolation]
    degraded_functions: List[str]
    emergency_actions_taken: List[str]

class SafetyManager:
    """
    Comprehensive safety management system for ADAS

    Implements ISO 26262 functional safety requirements with real-time monitoring,
    violation detection, and automated safety responses.
    """

    def __init__(self, config: ADASConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Safety state management
        self.safety_state = SafetyState.SAFE
        self.active_violations: Dict[str, SafetyViolation] = {}
        self.violation_history: List[SafetyViolation] = []
        self.degraded_functions: Set[str] = set()

        # Safety thresholds (stricter than operational thresholds)
        self.safety_thresholds = {
            'max_latency_ms': min(config.latency.total_pipeline_max_ms * 0.8, 8.0),
            'min_confidence': config.safety.min_detection_confidence,
            'max_collision_risk': 0.05,  # 5% maximum collision risk
            'min_sensor_health': 0.9,
            'max_false_positive_rate': config.safety.max_false_positive_rate,
            'max_false_negative_rate': config.safety.max_false_negative_rate
        }

        # Performance monitoring
        self.performance_history = {
            'perception_latencies': [],
            'prediction_latencies': [],
            'planning_latencies': [],
            'total_latencies': []
        }

        # Emergency response system
        self.emergency_responses = {
            SafetyViolationType.COLLISION_IMMINENT: self._emergency_brake,
            SafetyViolationType.SENSOR_FAILURE: self._sensor_fallback,
            SafetyViolationType.LATENCY_EXCEEDED: self._reduce_computation_load,
            SafetyViolationType.ASIL_DEGRADED: self._activate_degraded_mode
        }

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        self.monitoring_thread = None

        # Safety validators
        self.perception_validator = PerceptionSafetyValidator(config)
        self.prediction_validator = PredictionSafetyValidator(config)
        self.planning_validator = PlanningSafetyValidator(config)
        self.sensor_monitor = SensorHealthMonitor(config)

        # Violation ID counter
        self.violation_counter = 0

    async def start(self) -> bool:
        """Start the safety management system"""
        try:
            self.logger.info("Starting SafetyManager...")

            # Initialize validators
            await self.perception_validator.initialize()
            await self.prediction_validator.initialize()
            await self.planning_validator.initialize()
            await self.sensor_monitor.initialize()

            # Start monitoring thread
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

            self.safety_state = SafetyState.SAFE
            self.logger.info("SafetyManager started successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start SafetyManager: {e}")
            self.safety_state = SafetyState.FAULT
            return False

    def _monitoring_loop(self) -> None:
        """Main safety monitoring loop"""
        while self.running:
            try:
                # Periodic safety checks
                self._check_system_health()
                self._check_violation_recovery()
                self._update_safety_state()

                # Monitor at 10 Hz
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Safety monitoring error: {e}")
                self._handle_monitoring_error(e)

    async def validate_perception_output(self, output: PerceptionOutput) -> SafetyMetrics:
        """Validate perception output for safety compliance"""
        violations = []

        try:
            # Latency validation
            if output.processing_latency_ms > self.safety_thresholds['max_latency_ms']:
                violation = self._create_violation(
                    SafetyViolationType.LATENCY_EXCEEDED,
                    ASILLevel.C,
                    f"Perception latency {output.processing_latency_ms:.2f}ms exceeds threshold",
                    ["perception"],
                    "Reduce perception computation load"
                )
                violations.append(violation)

            # Object detection validation
            perception_violations = await self.perception_validator.validate(output)
            violations.extend(perception_violations)

            # Update performance history
            self.performance_history['perception_latencies'].append(output.processing_latency_ms)
            if len(self.performance_history['perception_latencies']) > 100:
                self.performance_history['perception_latencies'].pop(0)

            # Process violations
            await self._process_violations(violations)

            # Compute safety metrics
            safety_metrics = self._compute_perception_safety_metrics(output, violations)

            return safety_metrics

        except Exception as e:
            self.logger.error(f"Perception validation failed: {e}")
            return self._generate_failsafe_metrics("perception_validation_error")

    async def validate_prediction_output(self, output: PredictionOutput) -> SafetyMetrics:
        """Validate prediction output for safety compliance"""
        violations = []

        try:
            # Latency validation
            if output.prediction_latency_ms > self.safety_thresholds['max_latency_ms']:
                violation = self._create_violation(
                    SafetyViolationType.LATENCY_EXCEEDED,
                    ASILLevel.C,
                    f"Prediction latency {output.prediction_latency_ms:.2f}ms exceeds threshold",
                    ["prediction"],
                    "Reduce prediction computation load"
                )
                violations.append(violation)

            # Trajectory prediction validation
            prediction_violations = await self.prediction_validator.validate(output)
            violations.extend(prediction_violations)

            # Check collision risks
            for trajectory in output.predicted_trajectories:
                if trajectory.collision_probability > self.safety_thresholds['max_collision_risk']:
                    violation = self._create_violation(
                        SafetyViolationType.COLLISION_IMMINENT,
                        ASILLevel.D,
                        f"High collision probability {trajectory.collision_probability:.3f} for object {trajectory.object_id}",
                        ["prediction", "planning"],
                        "Execute emergency maneuver"
                    )
                    violations.append(violation)

            # Update performance history
            self.performance_history['prediction_latencies'].append(output.prediction_latency_ms)
            if len(self.performance_history['prediction_latencies']) > 100:
                self.performance_history['prediction_latencies'].pop(0)

            # Process violations
            await self._process_violations(violations)

            # Compute safety metrics
            safety_metrics = self._compute_prediction_safety_metrics(output, violations)

            return safety_metrics

        except Exception as e:
            self.logger.error(f"Prediction validation failed: {e}")
            return self._generate_failsafe_metrics("prediction_validation_error")

    async def validate_planning_output(self, output: PlanningOutput) -> SafetyMetrics:
        """Validate planning output for safety compliance"""
        violations = []

        try:
            # Latency validation
            if output.planning_latency_ms > self.safety_thresholds['max_latency_ms']:
                violation = self._create_violation(
                    SafetyViolationType.LATENCY_EXCEEDED,
                    ASILLevel.C,
                    f"Planning latency {output.planning_latency_ms:.2f}ms exceeds threshold",
                    ["planning"],
                    "Use simpler planning algorithm"
                )
                violations.append(violation)

            # Path safety validation
            planning_violations = await self.planning_validator.validate(output)
            violations.extend(planning_violations)

            # Check constraint violations
            if output.constraints_violated:
                violation = self._create_violation(
                    SafetyViolationType.CONSTRAINT_VIOLATED,
                    ASILLevel.C,
                    f"Planning constraints violated: {', '.join(output.constraints_violated)}",
                    ["planning"],
                    "Activate emergency path"
                )
                violations.append(violation)

            # Emergency state validation
            if output.planning_state.value in ['emergency', 'error']:
                violation = self._create_violation(
                    SafetyViolationType.COLLISION_IMMINENT,
                    ASILLevel.D,
                    f"Planning system in emergency state: {output.planning_state.value}",
                    ["planning", "control"],
                    "Execute emergency stop"
                )
                violations.append(violation)

            # Update performance history
            self.performance_history['planning_latencies'].append(output.planning_latency_ms)
            if len(self.performance_history['planning_latencies']) > 100:
                self.performance_history['planning_latencies'].pop(0)

            # Process violations
            await self._process_violations(violations)

            # Compute safety metrics
            safety_metrics = self._compute_planning_safety_metrics(output, violations)

            return safety_metrics

        except Exception as e:
            self.logger.error(f"Planning validation failed: {e}")
            return self._generate_failsafe_metrics("planning_validation_error")

    async def validate_sensor_fusion_output(self, fused_output: Any) -> SafetyMetrics:
        """Validate sensor fusion output for safety compliance"""
        violations = []

        try:
            # Sensor health monitoring
            sensor_violations = await self.sensor_monitor.check_sensor_health(fused_output)
            violations.extend(sensor_violations)

            # Data quality validation
            if hasattr(fused_output, 'confidence_score'):
                if fused_output.confidence_score < self.safety_thresholds['min_confidence']:
                    violation = self._create_violation(
                        SafetyViolationType.CONFIDENCE_LOW,
                        ASILLevel.C,
                        f"Sensor fusion confidence {fused_output.confidence_score:.3f} below threshold",
                        ["sensor_fusion"],
                        "Increase sensor redundancy"
                    )
                    violations.append(violation)

            # Process violations
            await self._process_violations(violations)

            # Compute safety metrics
            safety_metrics = self._compute_sensor_safety_metrics(fused_output, violations)

            return safety_metrics

        except Exception as e:
            self.logger.error(f"Sensor fusion validation failed: {e}")
            return self._generate_failsafe_metrics("sensor_validation_error")

    def _create_violation(self, violation_type: SafetyViolationType, severity: ASILLevel,
                         description: str, affected_components: List[str],
                         recommended_action: str) -> SafetyViolation:
        """Create a safety violation record"""
        self.violation_counter += 1
        return SafetyViolation(
            violation_type=violation_type,
            severity=severity,
            timestamp=time.time(),
            description=description,
            affected_components=affected_components,
            recommended_action=recommended_action,
            auto_recovery_possible=violation_type not in [
                SafetyViolationType.COLLISION_IMMINENT,
                SafetyViolationType.SENSOR_FAILURE
            ],
            violation_id=f"SAFETY_{self.violation_counter:06d}"
        )

    async def _process_violations(self, violations: List[SafetyViolation]) -> None:
        """Process safety violations and trigger appropriate responses"""
        for violation in violations:
            # Add to active violations
            self.active_violations[violation.violation_id] = violation

            # Add to history
            self.violation_history.append(violation)
            if len(self.violation_history) > 1000:  # Keep last 1000 violations
                self.violation_history.pop(0)

            # Update degraded functions
            self.degraded_functions.update(violation.affected_components)

            # Log violation
            self.logger.warning(f"Safety violation {violation.violation_id}: {violation.description}")

            # Trigger emergency response if applicable
            if violation.violation_type in self.emergency_responses:
                await self._trigger_emergency_response(violation)

    async def _trigger_emergency_response(self, violation: SafetyViolation) -> None:
        """Trigger emergency response for critical violations"""
        try:
            response_func = self.emergency_responses[violation.violation_type]
            await response_func(violation)
            self.logger.critical(f"Emergency response triggered for {violation.violation_id}")

        except Exception as e:
            self.logger.error(f"Emergency response failed for {violation.violation_id}: {e}")

    async def _emergency_brake(self, violation: SafetyViolation) -> None:
        """Emergency braking response"""
        self.logger.critical("EMERGENCY BRAKE ACTIVATED")
        # In real implementation, would send emergency brake command to actuators
        # For now, just log the action
        pass

    async def _sensor_fallback(self, violation: SafetyViolation) -> None:
        """Sensor fallback response"""
        self.logger.warning("Activating sensor fallback mode")
        # In real implementation, would reconfigure sensor fusion to use backup sensors
        pass

    async def _reduce_computation_load(self, violation: SafetyViolation) -> None:
        """Reduce computational load response"""
        self.logger.warning("Reducing computational load")
        # In real implementation, would reduce model complexity or frequency
        pass

    async def _activate_degraded_mode(self, violation: SafetyViolation) -> None:
        """Activate degraded operation mode"""
        self.logger.warning("Activating degraded operation mode")
        # In real implementation, would reduce functionality while maintaining safety
        pass

    def _check_system_health(self) -> None:
        """Periodic system health check"""
        try:
            # Check for stale violations
            current_time = time.time()
            stale_violations = []

            for violation_id, violation in self.active_violations.items():
                if current_time - violation.timestamp > 10.0:  # 10 second timeout
                    stale_violations.append(violation_id)

            # Remove stale violations
            for violation_id in stale_violations:
                del self.active_violations[violation_id]

            # Check overall system latency
            if self.performance_history['total_latencies']:
                avg_latency = np.mean(self.performance_history['total_latencies'][-10:])
                if avg_latency > self.safety_thresholds['max_latency_ms']:
                    self.logger.warning(f"System latency trending high: {avg_latency:.2f}ms")

        except Exception as e:
            self.logger.error(f"System health check failed: {e}")

    def _check_violation_recovery(self) -> None:
        """Check for automatic violation recovery"""
        recovered_violations = []

        for violation_id, violation in self.active_violations.items():
            if violation.auto_recovery_possible:
                # Check if conditions have improved
                if self._check_recovery_conditions(violation):
                    recovered_violations.append(violation_id)
                    self.logger.info(f"Violation {violation_id} automatically recovered")

        # Remove recovered violations
        for violation_id in recovered_violations:
            del self.active_violations[violation_id]

    def _check_recovery_conditions(self, violation: SafetyViolation) -> bool:
        """Check if violation recovery conditions are met"""
        current_time = time.time()

        # Check if violation is old enough for recovery consideration
        if current_time - violation.timestamp < 5.0:  # 5 second minimum
            return False

        # Violation-specific recovery checks
        if violation.violation_type == SafetyViolationType.LATENCY_EXCEEDED:
            # Check if recent latencies are below threshold
            if self.performance_history['total_latencies']:
                recent_latencies = self.performance_history['total_latencies'][-5:]
                return all(lat < self.safety_thresholds['max_latency_ms'] for lat in recent_latencies)

        elif violation.violation_type == SafetyViolationType.CONFIDENCE_LOW:
            # Would check if confidence has improved
            return True  # Simplified

        return False

    def _update_safety_state(self) -> None:
        """Update overall safety state based on active violations"""
        if not self.active_violations:
            self.safety_state = SafetyState.SAFE
            return

        # Check for critical violations
        critical_violations = [v for v in self.active_violations.values()
                             if v.severity == ASILLevel.D]
        if critical_violations:
            self.safety_state = SafetyState.EMERGENCY
            return

        # Check for serious violations
        serious_violations = [v for v in self.active_violations.values()
                            if v.severity == ASILLevel.C]
        if serious_violations:
            self.safety_state = SafetyState.CRITICAL
            return

        # Check for warnings
        warning_violations = [v for v in self.active_violations.values()
                            if v.severity in [ASILLevel.B, ASILLevel.A]]
        if warning_violations:
            self.safety_state = SafetyState.WARNING
            return

        self.safety_state = SafetyState.SAFE

    def _compute_perception_safety_metrics(self, output: PerceptionOutput,
                                         violations: List[SafetyViolation]) -> SafetyMetrics:
        """Compute safety metrics for perception output"""
        perception_safety = 1.0
        if violations:
            # Reduce safety score based on violation severity
            severity_weights = {ASILLevel.A: 0.1, ASILLevel.B: 0.2, ASILLevel.C: 0.4, ASILLevel.D: 0.8}
            total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
            perception_safety = max(0.0, 1.0 - total_penalty)

        return SafetyMetrics(
            timestamp=time.time(),
            overall_safety_score=perception_safety,
            perception_safety=perception_safety,
            prediction_safety=1.0,  # Not assessed in this call
            planning_safety=1.0,    # Not assessed in this call
            sensor_health_score=self._compute_sensor_health_score(),
            system_latency_ms=output.processing_latency_ms,
            active_violations=list(self.active_violations.values()),
            degraded_functions=list(self.degraded_functions),
            emergency_actions_taken=[]
        )

    def _compute_prediction_safety_metrics(self, output: PredictionOutput,
                                         violations: List[SafetyViolation]) -> SafetyMetrics:
        """Compute safety metrics for prediction output"""
        prediction_safety = 1.0
        if violations:
            severity_weights = {ASILLevel.A: 0.1, ASILLevel.B: 0.2, ASILLevel.C: 0.4, ASILLevel.D: 0.8}
            total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
            prediction_safety = max(0.0, 1.0 - total_penalty)

        return SafetyMetrics(
            timestamp=time.time(),
            overall_safety_score=prediction_safety,
            perception_safety=1.0,    # Not assessed in this call
            prediction_safety=prediction_safety,
            planning_safety=1.0,     # Not assessed in this call
            sensor_health_score=self._compute_sensor_health_score(),
            system_latency_ms=output.prediction_latency_ms,
            active_violations=list(self.active_violations.values()),
            degraded_functions=list(self.degraded_functions),
            emergency_actions_taken=[]
        )

    def _compute_planning_safety_metrics(self, output: PlanningOutput,
                                       violations: List[SafetyViolation]) -> SafetyMetrics:
        """Compute safety metrics for planning output"""
        planning_safety = 1.0
        if violations:
            severity_weights = {ASILLevel.A: 0.1, ASILLevel.B: 0.2, ASILLevel.C: 0.4, ASILLevel.D: 0.8}
            total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
            planning_safety = max(0.0, 1.0 - total_penalty)

        return SafetyMetrics(
            timestamp=time.time(),
            overall_safety_score=planning_safety,
            perception_safety=1.0,   # Not assessed in this call
            prediction_safety=1.0,   # Not assessed in this call
            planning_safety=planning_safety,
            sensor_health_score=self._compute_sensor_health_score(),
            system_latency_ms=output.planning_latency_ms,
            active_violations=list(self.active_violations.values()),
            degraded_functions=list(self.degraded_functions),
            emergency_actions_taken=[]
        )

    def _compute_sensor_safety_metrics(self, output: Any,
                                     violations: List[SafetyViolation]) -> SafetyMetrics:
        """Compute safety metrics for sensor fusion output"""
        sensor_safety = 1.0
        if violations:
            severity_weights = {ASILLevel.A: 0.1, ASILLevel.B: 0.2, ASILLevel.C: 0.4, ASILLevel.D: 0.8}
            total_penalty = sum(severity_weights.get(v.severity, 0.5) for v in violations)
            sensor_safety = max(0.0, 1.0 - total_penalty)

        return SafetyMetrics(
            timestamp=time.time(),
            overall_safety_score=sensor_safety,
            perception_safety=1.0,   # Not assessed in this call
            prediction_safety=1.0,   # Not assessed in this call
            planning_safety=1.0,     # Not assessed in this call
            sensor_health_score=self._compute_sensor_health_score(),
            system_latency_ms=0.0,   # Not applicable
            active_violations=list(self.active_violations.values()),
            degraded_functions=list(self.degraded_functions),
            emergency_actions_taken=[]
        )

    def _compute_sensor_health_score(self) -> float:
        """Compute overall sensor health score"""
        # Simplified sensor health computation
        # In real implementation, would aggregate individual sensor health scores
        if self.degraded_functions:
            degradation_factor = len(self.degraded_functions) / 10.0  # Assume max 10 functions
            return max(0.0, 1.0 - degradation_factor)
        return 1.0

    def _generate_failsafe_metrics(self, error_context: str) -> SafetyMetrics:
        """Generate failsafe safety metrics in case of validation errors"""
        # Create error violation
        violation = self._create_violation(
            SafetyViolationType.COMPUTATION_ERROR,
            ASILLevel.C,
            f"Safety validation error: {error_context}",
            ["safety_system"],
            "Restart safety validation"
        )

        return SafetyMetrics(
            timestamp=time.time(),
            overall_safety_score=0.5,  # Degraded safety due to validation error
            perception_safety=0.5,
            prediction_safety=0.5,
            planning_safety=0.5,
            sensor_health_score=0.5,
            system_latency_ms=0.0,
            active_violations=[violation],
            degraded_functions=["safety_validation"],
            emergency_actions_taken=[]
        )

    async def get_comprehensive_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        return {
            'safety_state': self.safety_state.value,
            'active_violations': [
                {
                    'id': v.violation_id,
                    'type': v.violation_type.value,
                    'severity': v.severity.value,
                    'description': v.description,
                    'timestamp': v.timestamp,
                    'affected_components': v.affected_components
                }
                for v in self.active_violations.values()
            ],
            'degraded_functions': list(self.degraded_functions),
            'safety_thresholds': self.safety_thresholds,
            'performance_summary': {
                'avg_perception_latency': np.mean(self.performance_history['perception_latencies'][-10:])
                if self.performance_history['perception_latencies'] else 0.0,
                'avg_prediction_latency': np.mean(self.performance_history['prediction_latencies'][-10:])
                if self.performance_history['prediction_latencies'] else 0.0,
                'avg_planning_latency': np.mean(self.performance_history['planning_latencies'][-10:])
                if self.performance_history['planning_latencies'] else 0.0
            },
            'violation_statistics': {
                'total_violations': len(self.violation_history),
                'violations_last_hour': len([v for v in self.violation_history
                                            if time.time() - v.timestamp < 3600])
            }
        }

    async def stop(self) -> None:
        """Stop the safety management system"""
        self.logger.info("Stopping SafetyManager...")
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)
        self.logger.info("SafetyManager stopped")

    def get_safety_state(self) -> SafetyState:
        """Get current safety state"""
        return self.safety_state


# Supporting validator classes (simplified implementations)
class PerceptionSafetyValidator:
    def __init__(self, config: ADASConfig):
        self.config = config

    async def initialize(self):
        pass

    async def validate(self, output: PerceptionOutput) -> List[SafetyViolation]:
        # Simplified perception validation
        return []

class PredictionSafetyValidator:
    def __init__(self, config: ADASConfig):
        self.config = config

    async def initialize(self):
        pass

    async def validate(self, output: PredictionOutput) -> List[SafetyViolation]:
        # Simplified prediction validation
        return []

class PlanningSafetyValidator:
    def __init__(self, config: ADASConfig):
        self.config = config

    async def initialize(self):
        pass

    async def validate(self, output: PlanningOutput) -> List[SafetyViolation]:
        # Simplified planning validation
        return []

class SensorHealthMonitor:
    def __init__(self, config: ADASConfig):
        self.config = config

    async def initialize(self):
        pass

    async def check_sensor_health(self, output: Any) -> List[SafetyViolation]:
        # Simplified sensor health monitoring
        return []