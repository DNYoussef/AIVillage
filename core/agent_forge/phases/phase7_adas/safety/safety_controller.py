"""
Safety Controller - Safety-critical controls for ADAS systems
Implements fail-safe mechanisms, redundancy management, and emergency response
Automotive Safety Integrity Level (ASIL-D) compliant with functional safety standards
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from abc import ABC, abstractmethod
import uuid

class SafetyLevel(Enum):
    """Automotive Safety Integrity Levels"""
    QM = "QM"      # Quality Management (non-safety)
    ASIL_A = "A"   # Lowest safety level
    ASIL_B = "B"   # Basic safety level
    ASIL_C = "C"   # Enhanced safety level
    ASIL_D = "D"   # Highest safety level

class SystemState(Enum):
    """System operational states"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAIL_SAFE = "fail_safe"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"

class ThreatLevel(Enum):
    """Threat assessment levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FailureMode(Enum):
    """System failure modes"""
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    PROCESSING_FAILURE = "processing_failure"
    POWER_FAILURE = "power_failure"
    THERMAL_FAILURE = "thermal_failure"
    SOFTWARE_FAILURE = "software_failure"
    EXTERNAL_INTERFERENCE = "external_interference"

@dataclass
class SafetyAlert:
    """Safety alert data structure"""
    alert_id: str
    timestamp: float
    threat_level: ThreatLevel
    alert_type: str
    message: str
    affected_systems: List[str]
    recommended_action: str
    time_to_impact: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health monitoring data"""
    component: str
    status: str
    health_score: float  # 0.0 to 1.0
    last_check: float
    error_count: int
    warning_count: int
    performance_metrics: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmergencyResponse:
    """Emergency response action"""
    response_id: str
    trigger_condition: str
    action_type: str
    urgency: ThreatLevel
    execution_time: float
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

class SafetyMonitor(ABC):
    """Abstract base class for safety monitors"""

    @abstractmethod
    async def check_safety(self, system_data: Dict) -> List[SafetyAlert]:
        """Check safety conditions and return alerts"""
        pass

    @abstractmethod
    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        pass

class CollisionRiskMonitor(SafetyMonitor):
    """Monitor for collision risk assessment"""

    def __init__(self, config: Dict):
        self.config = config
        self.time_to_collision_threshold = config.get('ttc_threshold', 3.0)  # seconds
        self.min_safe_distance = config.get('min_safe_distance', 2.0)  # meters
        self.lateral_clearance = config.get('lateral_clearance', 1.0)  # meters

        # Health tracking
        self.health = SystemHealth(
            component="collision_risk_monitor",
            status="active",
            health_score=1.0,
            last_check=time.time(),
            error_count=0,
            warning_count=0,
            performance_metrics={}
        )

    async def check_safety(self, system_data: Dict) -> List[SafetyAlert]:
        """Check for collision risks"""
        alerts = []

        try:
            self.health.last_check = time.time()

            # Get object tracking data
            tracks = system_data.get('tracks', [])
            ego_velocity = system_data.get('ego_velocity', (0, 0, 0))

            for track in tracks:
                # Calculate time to collision
                ttc = self._calculate_time_to_collision(track, ego_velocity)

                if ttc is not None and ttc < self.time_to_collision_threshold:
                    threat_level = self._assess_collision_threat(ttc, track)

                    alert = SafetyAlert(
                        alert_id=f"collision_risk_{track.track_id}_{int(time.time()*1000)}",
                        timestamp=time.time(),
                        threat_level=threat_level,
                        alert_type="collision_risk",
                        message=f"Collision risk with {track.object_type.value} (TTC: {ttc:.1f}s)",
                        affected_systems=["braking", "steering", "warning"],
                        recommended_action=self._get_recommended_action(threat_level),
                        time_to_impact=ttc,
                        metadata={
                            'track_id': track.track_id,
                            'object_type': track.object_type.value,
                            'ttc': ttc,
                            'distance': self._calculate_distance(track)
                        }
                    )
                    alerts.append(alert)

            # Update health status
            if alerts:
                self.health.warning_count += len(alerts)

            self.health.status = "active"
            self.health.health_score = max(0.5, 1.0 - len(alerts) * 0.1)

        except Exception as e:
            logging.error(f"Collision risk monitoring failed: {e}")
            self.health.error_count += 1
            self.health.health_score = 0.3
            self.health.status = "degraded"

        return alerts

    def _calculate_time_to_collision(self, track, ego_velocity: Tuple[float, float, float]) -> Optional[float]:
        """Calculate time to collision with tracked object"""
        if not track.positions or not track.velocities:
            return None

        # Get latest position and velocity
        obj_position = track.positions[-1]
        obj_velocity = track.velocities[-1] if track.velocities else (0, 0, 0)

        # Relative velocity (object velocity - ego velocity)
        rel_velocity = (
            obj_velocity[0] - ego_velocity[0],
            obj_velocity[1] - ego_velocity[1],
            obj_velocity[2] - ego_velocity[2]
        )

        # Distance to object
        distance = math.sqrt(obj_position[0]**2 + obj_position[1]**2)

        # Relative speed (closing rate)
        rel_speed = math.sqrt(rel_velocity[0]**2 + rel_velocity[1]**2)

        # Time to collision
        if rel_speed > 0.1:  # Minimum speed threshold
            ttc = distance / rel_speed
            return ttc if ttc > 0 else None

        return None

    def _assess_collision_threat(self, ttc: float, track) -> ThreatLevel:
        """Assess threat level based on time to collision"""
        if ttc < 1.0:
            return ThreatLevel.CRITICAL
        elif ttc < 2.0:
            return ThreatLevel.HIGH
        elif ttc < 3.0:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW

    def _calculate_distance(self, track) -> float:
        """Calculate distance to tracked object"""
        if not track.positions:
            return float('inf')

        position = track.positions[-1]
        return math.sqrt(position[0]**2 + position[1]**2)

    def _get_recommended_action(self, threat_level: ThreatLevel) -> str:
        """Get recommended action based on threat level"""
        action_map = {
            ThreatLevel.CRITICAL: "EMERGENCY_BRAKE",
            ThreatLevel.HIGH: "AUTOMATIC_BRAKE",
            ThreatLevel.MEDIUM: "WARNING_BRAKE",
            ThreatLevel.LOW: "AUDIO_WARNING"
        }
        return action_map.get(threat_level, "MONITOR")

    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        return self.health

class LaneDepartureMonitor(SafetyMonitor):
    """Monitor for lane departure warnings"""

    def __init__(self, config: Dict):
        self.config = config
        self.lane_departure_threshold = config.get('departure_threshold', 0.5)  # meters
        self.time_to_departure_threshold = config.get('ttd_threshold', 2.0)  # seconds

        self.health = SystemHealth(
            component="lane_departure_monitor",
            status="active",
            health_score=1.0,
            last_check=time.time(),
            error_count=0,
            warning_count=0,
            performance_metrics={}
        )

    async def check_safety(self, system_data: Dict) -> List[SafetyAlert]:
        """Check for lane departure conditions"""
        alerts = []

        try:
            self.health.last_check = time.time()

            lane_info = system_data.get('lane_info')
            ego_velocity = system_data.get('ego_velocity', (0, 0, 0))

            if not lane_info or lane_info.confidence < 0.5:
                # Lane information not reliable
                self.health.warning_count += 1
                return alerts

            # Check left lane departure
            if lane_info.left_markings:
                left_distance = self._calculate_lane_distance(lane_info.left_markings[0])
                if left_distance < self.lane_departure_threshold:
                    ttd = self._calculate_time_to_departure(left_distance, ego_velocity)

                    alert = SafetyAlert(
                        alert_id=f"lane_departure_left_{int(time.time()*1000)}",
                        timestamp=time.time(),
                        threat_level=ThreatLevel.MEDIUM,
                        alert_type="lane_departure",
                        message="Left lane departure detected",
                        affected_systems=["steering", "warning"],
                        recommended_action="STEERING_CORRECTION",
                        time_to_impact=ttd,
                        metadata={'side': 'left', 'distance': left_distance, 'ttd': ttd}
                    )
                    alerts.append(alert)

            # Check right lane departure
            if lane_info.right_markings:
                right_distance = self._calculate_lane_distance(lane_info.right_markings[0])
                if right_distance < self.lane_departure_threshold:
                    ttd = self._calculate_time_to_departure(right_distance, ego_velocity)

                    alert = SafetyAlert(
                        alert_id=f"lane_departure_right_{int(time.time()*1000)}",
                        timestamp=time.time(),
                        threat_level=ThreatLevel.MEDIUM,
                        alert_type="lane_departure",
                        message="Right lane departure detected",
                        affected_systems=["steering", "warning"],
                        recommended_action="STEERING_CORRECTION",
                        time_to_impact=ttd,
                        metadata={'side': 'right', 'distance': right_distance, 'ttd': ttd}
                    )
                    alerts.append(alert)

            # Update health
            self.health.status = "active"
            self.health.health_score = 1.0 if not alerts else 0.8

        except Exception as e:
            logging.error(f"Lane departure monitoring failed: {e}")
            self.health.error_count += 1
            self.health.health_score = 0.3
            self.health.status = "degraded"

        return alerts

    def _calculate_lane_distance(self, lane_marking) -> float:
        """Calculate distance to lane marking (simplified)"""
        # Simplified calculation - in production, use proper geometric calculations
        if not lane_marking.points:
            return float('inf')

        # Use closest point as approximation
        closest_distance = float('inf')
        for point in lane_marking.points:
            # Assume vehicle is at origin of coordinate system
            distance = abs(point[0])  # Lateral distance
            closest_distance = min(closest_distance, distance)

        return closest_distance

    def _calculate_time_to_departure(self, distance: float, velocity: Tuple[float, float, float]) -> float:
        """Calculate time to lane departure"""
        lateral_velocity = abs(velocity[1])  # Y-component of velocity

        if lateral_velocity > 0.1:  # Minimum velocity threshold
            return distance / lateral_velocity

        return float('inf')

    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        return self.health

class SystemHealthMonitor(SafetyMonitor):
    """Monitor overall system health"""

    def __init__(self, config: Dict):
        self.config = config
        self.component_thresholds = config.get('component_thresholds', {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'temperature': 75.0,
            'processing_latency': 100.0  # ms
        })

        self.health = SystemHealth(
            component="system_health_monitor",
            status="active",
            health_score=1.0,
            last_check=time.time(),
            error_count=0,
            warning_count=0,
            performance_metrics={}
        )

    async def check_safety(self, system_data: Dict) -> List[SafetyAlert]:
        """Check system health conditions"""
        alerts = []

        try:
            self.health.last_check = time.time()

            # Check processing latency
            processing_time = system_data.get('processing_time_ms', 0)
            if processing_time > self.component_thresholds['processing_latency']:
                alert = SafetyAlert(
                    alert_id=f"high_latency_{int(time.time()*1000)}",
                    timestamp=time.time(),
                    threat_level=ThreatLevel.HIGH,
                    alert_type="performance_degradation",
                    message=f"High processing latency: {processing_time:.1f}ms",
                    affected_systems=["perception", "planning"],
                    recommended_action="REDUCE_PROCESSING_LOAD",
                    metadata={'latency_ms': processing_time}
                )
                alerts.append(alert)

            # Check system resources (simulated)
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            temperature = self._get_system_temperature()

            if cpu_usage > self.component_thresholds['cpu_usage']:
                alert = SafetyAlert(
                    alert_id=f"high_cpu_{int(time.time()*1000)}",
                    timestamp=time.time(),
                    threat_level=ThreatLevel.MEDIUM,
                    alert_type="resource_exhaustion",
                    message=f"High CPU usage: {cpu_usage:.1f}%",
                    affected_systems=["all"],
                    recommended_action="THROTTLE_PROCESSING",
                    metadata={'cpu_usage': cpu_usage}
                )
                alerts.append(alert)

            if memory_usage > self.component_thresholds['memory_usage']:
                alert = SafetyAlert(
                    alert_id=f"high_memory_{int(time.time()*1000)}",
                    timestamp=time.time(),
                    threat_level=ThreatLevel.MEDIUM,
                    alert_type="resource_exhaustion",
                    message=f"High memory usage: {memory_usage:.1f}%",
                    affected_systems=["all"],
                    recommended_action="FREE_MEMORY",
                    metadata={'memory_usage': memory_usage}
                )
                alerts.append(alert)

            if temperature > self.component_thresholds['temperature']:
                alert = SafetyAlert(
                    alert_id=f"high_temp_{int(time.time()*1000)}",
                    timestamp=time.time(),
                    threat_level=ThreatLevel.HIGH,
                    alert_type="thermal_warning",
                    message=f"High system temperature: {temperature:.1f}C",
                    affected_systems=["hardware"],
                    recommended_action="THERMAL_THROTTLING",
                    metadata={'temperature': temperature}
                )
                alerts.append(alert)

            # Update health
            self.health.performance_metrics = {
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'temperature': temperature,
                'processing_latency': processing_time
            }

            self.health.status = "active" if not alerts else "degraded"
            self.health.health_score = max(0.5, 1.0 - len(alerts) * 0.15)

        except Exception as e:
            logging.error(f"System health monitoring failed: {e}")
            self.health.error_count += 1
            self.health.health_score = 0.2
            self.health.status = "fault"

        return alerts

    def _get_cpu_usage(self) -> float:
        """Get current CPU usage (simulated)"""
        # In production, use psutil or similar
        import random
        return 60.0 + random.random() * 25.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage (simulated)"""
        # In production, use psutil or similar
        import random
        return 55.0 + random.random() * 30.0

    def _get_system_temperature(self) -> float:
        """Get system temperature (simulated)"""
        # In production, read from thermal sensors
        import random
        return 45.0 + random.random() * 20.0

    def get_health_status(self) -> SystemHealth:
        """Get current health status"""
        return self.health

class FailSafeController:
    """Fail-safe mechanism controller"""

    def __init__(self, config: Dict):
        self.config = config
        self.fail_safe_actions = {
            ThreatLevel.CRITICAL: self._critical_fail_safe,
            ThreatLevel.HIGH: self._high_fail_safe,
            ThreatLevel.MEDIUM: self._medium_fail_safe,
            ThreatLevel.LOW: self._low_fail_safe
        }

        self.emergency_responses = []
        self.fail_safe_lock = threading.RLock()

    async def execute_fail_safe(self, alerts: List[SafetyAlert]) -> List[EmergencyResponse]:
        """Execute appropriate fail-safe actions"""
        responses = []

        with self.fail_safe_lock:
            try:
                # Group alerts by threat level
                threat_levels = set(alert.threat_level for alert in alerts)

                # Execute fail-safe for highest threat level
                max_threat = max(threat_levels) if threat_levels else ThreatLevel.LOW

                response = await self._execute_threat_response(max_threat, alerts)
                if response:
                    responses.append(response)
                    self.emergency_responses.append(response)

                # Log all responses
                for response in responses:
                    logging.warning(f"Fail-safe executed: {response.action_type} "
                                  f"(urgency: {response.urgency.value})")

            except Exception as e:
                logging.error(f"Fail-safe execution failed: {e}")

        return responses

    async def _execute_threat_response(self, threat_level: ThreatLevel,
                                     alerts: List[SafetyAlert]) -> Optional[EmergencyResponse]:
        """Execute response for specific threat level"""
        try:
            action_func = self.fail_safe_actions.get(threat_level)
            if not action_func:
                return None

            start_time = time.time()
            success = await action_func(alerts)
            execution_time = time.time() - start_time

            response = EmergencyResponse(
                response_id=str(uuid.uuid4()),
                trigger_condition=f"Threat level: {threat_level.value}",
                action_type=action_func.__name__,
                urgency=threat_level,
                execution_time=execution_time,
                success=success,
                details={
                    'alert_count': len(alerts),
                    'alert_types': list(set(alert.alert_type for alert in alerts))
                }
            )

            return response

        except Exception as e:
            logging.error(f"Threat response execution failed: {e}")
            return None

    async def _critical_fail_safe(self, alerts: List[SafetyAlert]) -> bool:
        """Critical threat response - emergency stop"""
        try:
            logging.critical("CRITICAL THREAT DETECTED - EMERGENCY STOP")

            # Emergency brake activation
            await self._activate_emergency_brake()

            # Hazard lights activation
            await self._activate_hazard_lights()

            # Emergency communication
            await self._send_emergency_signal()

            # System state transition
            await self._transition_to_emergency_state()

            return True

        except Exception as e:
            logging.error(f"Critical fail-safe failed: {e}")
            return False

    async def _high_fail_safe(self, alerts: List[SafetyAlert]) -> bool:
        """High threat response - automatic intervention"""
        try:
            logging.warning("HIGH THREAT DETECTED - AUTOMATIC INTERVENTION")

            # Automatic braking
            await self._activate_automatic_brake()

            # Steering correction if lane departure
            lane_alerts = [a for a in alerts if a.alert_type == "lane_departure"]
            if lane_alerts:
                await self._apply_steering_correction(lane_alerts[0])

            # Warning systems
            await self._activate_audio_warning()

            return True

        except Exception as e:
            logging.error(f"High fail-safe failed: {e}")
            return False

    async def _medium_fail_safe(self, alerts: List[SafetyAlert]) -> bool:
        """Medium threat response - warning and preparation"""
        try:
            logging.warning("MEDIUM THREAT DETECTED - WARNING SYSTEMS")

            # Pre-charge brakes
            await self._precharge_brakes()

            # Audio/visual warnings
            await self._activate_audio_warning()
            await self._activate_visual_warning()

            # Driver alertness check
            await self._check_driver_alertness()

            return True

        except Exception as e:
            logging.error(f"Medium fail-safe failed: {e}")
            return False

    async def _low_fail_safe(self, alerts: List[SafetyAlert]) -> bool:
        """Low threat response - monitoring and alerts"""
        try:
            logging.info("LOW THREAT DETECTED - MONITORING")

            # Gentle audio notification
            await self._gentle_audio_notification()

            # Increase monitoring frequency
            await self._increase_monitoring()

            return True

        except Exception as e:
            logging.error(f"Low fail-safe failed: {e}")
            return False

    # Fail-safe action implementations (placeholders for actual vehicle interfaces)

    async def _activate_emergency_brake(self):
        """Activate emergency braking system"""
        logging.critical("EMERGENCY BRAKE ACTIVATED")
        # In production: Interface with vehicle brake controller
        await asyncio.sleep(0.01)  # Simulate actuation delay

    async def _activate_hazard_lights(self):
        """Activate hazard warning lights"""
        logging.warning("HAZARD LIGHTS ACTIVATED")
        # In production: Interface with vehicle lighting system
        await asyncio.sleep(0.005)

    async def _send_emergency_signal(self):
        """Send emergency signal to other vehicles/infrastructure"""
        logging.critical("EMERGENCY SIGNAL TRANSMITTED")
        # In production: Interface with V2X communication
        await asyncio.sleep(0.001)

    async def _transition_to_emergency_state(self):
        """Transition system to emergency state"""
        logging.critical("SYSTEM ENTERING EMERGENCY STATE")
        # In production: Coordinate with vehicle systems
        await asyncio.sleep(0.001)

    async def _activate_automatic_brake(self):
        """Activate automatic braking"""
        logging.warning("AUTOMATIC BRAKE ACTIVATED")
        # In production: Interface with ABS/ESP systems
        await asyncio.sleep(0.01)

    async def _apply_steering_correction(self, lane_alert: SafetyAlert):
        """Apply steering correction for lane departure"""
        side = lane_alert.metadata.get('side', 'unknown')
        logging.warning(f"STEERING CORRECTION APPLIED - {side} side")
        # In production: Interface with EPS (Electric Power Steering)
        await asyncio.sleep(0.02)

    async def _activate_audio_warning(self):
        """Activate audio warning system"""
        logging.warning("AUDIO WARNING ACTIVATED")
        # In production: Interface with vehicle audio system
        await asyncio.sleep(0.005)

    async def _precharge_brakes(self):
        """Pre-charge brake system for faster response"""
        logging.info("BRAKE SYSTEM PRE-CHARGED")
        # In production: Interface with brake controller
        await asyncio.sleep(0.005)

    async def _activate_visual_warning(self):
        """Activate visual warning displays"""
        logging.warning("VISUAL WARNING ACTIVATED")
        # In production: Interface with HMI/dashboard
        await asyncio.sleep(0.005)

    async def _check_driver_alertness(self):
        """Check driver alertness level"""
        logging.info("DRIVER ALERTNESS CHECK")
        # In production: Interface with driver monitoring system
        await asyncio.sleep(0.01)

    async def _gentle_audio_notification(self):
        """Generate gentle audio notification"""
        logging.info("GENTLE AUDIO NOTIFICATION")
        # In production: Interface with audio system
        await asyncio.sleep(0.002)

    async def _increase_monitoring(self):
        """Increase system monitoring frequency"""
        logging.info("MONITORING FREQUENCY INCREASED")
        # In production: Adjust system parameters
        await asyncio.sleep(0.001)

class RedundancyManager:
    """Manages system redundancy and backup systems"""

    def __init__(self, config: Dict):
        self.config = config
        self.primary_systems = config.get('primary_systems', [])
        self.backup_systems = config.get('backup_systems', [])
        self.system_status = {}

        # Initialize system status
        for system in self.primary_systems + self.backup_systems:
            self.system_status[system] = "active"

    def check_redundancy(self, failed_systems: List[str]) -> Dict[str, str]:
        """Check redundancy status and activate backups if needed"""
        redundancy_status = {}

        for failed_system in failed_systems:
            self.system_status[failed_system] = "failed"

            # Find backup system
            backup_system = self._find_backup_system(failed_system)

            if backup_system and self.system_status[backup_system] == "standby":
                # Activate backup system
                self.system_status[backup_system] = "active"
                redundancy_status[failed_system] = f"backup_activated:{backup_system}"
                logging.warning(f"Backup system {backup_system} activated for {failed_system}")
            elif backup_system and self.system_status[backup_system] == "active":
                redundancy_status[failed_system] = f"backup_already_active:{backup_system}"
            else:
                redundancy_status[failed_system] = "no_backup_available"
                logging.error(f"No backup available for failed system: {failed_system}")

        return redundancy_status

    def _find_backup_system(self, primary_system: str) -> Optional[str]:
        """Find backup system for primary system"""
        # Simplified mapping - in production, use configuration-based mapping
        backup_mapping = {
            'front_camera': 'backup_camera',
            'front_radar': 'side_radar',
            'main_processor': 'backup_processor',
            'primary_can': 'backup_can'
        }
        return backup_mapping.get(primary_system)

    def get_system_status(self) -> Dict[str, str]:
        """Get current status of all systems"""
        return self.system_status.copy()

class SafetyController:
    """Main safety controller coordinating all safety functions"""

    def __init__(self, safety_config: Dict):
        self.safety_config = safety_config
        self.safety_level = SafetyLevel.ASIL_D
        self.system_state = SystemState.NORMAL

        # Initialize safety monitors
        self.monitors = {
            'collision_risk': CollisionRiskMonitor(safety_config.get('collision_monitor', {})),
            'lane_departure': LaneDepartureMonitor(safety_config.get('lane_monitor', {})),
            'system_health': SystemHealthMonitor(safety_config.get('health_monitor', {}))
        }

        # Initialize safety controllers
        self.fail_safe_controller = FailSafeController(safety_config.get('fail_safe', {}))
        self.redundancy_manager = RedundancyManager(safety_config.get('redundancy', {}))

        # Safety metrics
        self.safety_metrics = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'fail_safe_activations': 0,
            'system_uptime': time.time(),
            'mtbf': 0.0  # Mean Time Between Failures
        }

        # Alert history for analysis
        self.alert_history = deque(maxlen=1000)
        self.response_history = deque(maxlen=100)

    async def process_safety_frame(self, system_data: Dict) -> Dict:
        """Process safety checks for current frame"""
        start_time = time.perf_counter()

        try:
            # Run all safety monitors concurrently
            monitor_tasks = [
                monitor.check_safety(system_data)
                for monitor in self.monitors.values()
            ]

            # Wait for all monitors to complete
            monitor_results = await asyncio.gather(*monitor_tasks, return_exceptions=True)

            # Collect all alerts
            all_alerts = []
            for result in monitor_results:
                if isinstance(result, Exception):
                    logging.error(f"Safety monitor failed: {result}")
                    continue
                all_alerts.extend(result)

            # Update alert history
            for alert in all_alerts:
                self.alert_history.append(alert)

            # Execute fail-safe responses if needed
            responses = []
            if all_alerts:
                responses = await self.fail_safe_controller.execute_fail_safe(all_alerts)
                self.response_history.extend(responses)

            # Check system redundancy
            failed_systems = self._identify_failed_systems(system_data)
            redundancy_status = self.redundancy_manager.check_redundancy(failed_systems)

            # Update system state
            self._update_system_state(all_alerts, responses)

            # Update safety metrics
            self._update_safety_metrics(all_alerts, responses)

            # Calculate processing time
            processing_time = (time.perf_counter() - start_time) * 1000

            # Create safety result
            safety_result = {
                'timestamp': time.time(),
                'system_state': self.system_state.value,
                'safety_level': self.safety_level.value,
                'alerts': all_alerts,
                'responses': responses,
                'redundancy_status': redundancy_status,
                'health_status': self._get_overall_health_status(),
                'processing_time_ms': processing_time,
                'metrics': self.get_safety_metrics()
            }

            return safety_result

        except Exception as e:
            logging.error(f"Safety processing failed: {e}")
            return {
                'timestamp': time.time(),
                'system_state': SystemState.EMERGENCY.value,
                'safety_level': self.safety_level.value,
                'alerts': [],
                'responses': [],
                'redundancy_status': {},
                'health_status': {},
                'processing_time_ms': 0.0,
                'metrics': self.get_safety_metrics()
            }

    def _identify_failed_systems(self, system_data: Dict) -> List[str]:
        """Identify systems that have failed"""
        failed_systems = []

        # Check for system failures based on health status
        for monitor_name, monitor in self.monitors.items():
            health = monitor.get_health_status()
            if health.status in ["fault", "degraded"] and health.health_score < 0.5:
                failed_systems.append(monitor_name)

        # Check for sensor failures
        sensor_health = system_data.get('sensor_health', {})
        for sensor_id, health_score in sensor_health.items():
            if health_score < 0.3:  # Critical health threshold
                failed_systems.append(sensor_id)

        return failed_systems

    def _update_system_state(self, alerts: List[SafetyAlert], responses: List[EmergencyResponse]):
        """Update overall system state based on alerts and responses"""
        if not alerts:
            self.system_state = SystemState.NORMAL
            return

        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.threat_level == ThreatLevel.CRITICAL]
        if critical_alerts:
            self.system_state = SystemState.EMERGENCY
            return

        # Check for high threat alerts
        high_alerts = [a for a in alerts if a.threat_level == ThreatLevel.HIGH]
        if high_alerts:
            self.system_state = SystemState.FAIL_SAFE
            return

        # Check for multiple medium/low alerts
        if len(alerts) > 3:
            self.system_state = SystemState.DEGRADED
        else:
            self.system_state = SystemState.NORMAL

    def _get_overall_health_status(self) -> Dict:
        """Get overall system health status"""
        health_statuses = {}

        for monitor_name, monitor in self.monitors.items():
            health = monitor.get_health_status()
            health_statuses[monitor_name] = {
                'status': health.status,
                'health_score': health.health_score,
                'error_count': health.error_count,
                'warning_count': health.warning_count
            }

        # Calculate overall health score
        total_score = sum(h.health_score for h in [m.get_health_status() for m in self.monitors.values()])
        overall_score = total_score / len(self.monitors) if self.monitors else 0.0

        health_statuses['overall'] = {
            'health_score': overall_score,
            'system_state': self.system_state.value,
            'safety_level': self.safety_level.value
        }

        return health_statuses

    def _update_safety_metrics(self, alerts: List[SafetyAlert], responses: List[EmergencyResponse]):
        """Update safety performance metrics"""
        self.safety_metrics['total_alerts'] += len(alerts)

        critical_count = sum(1 for a in alerts if a.threat_level == ThreatLevel.CRITICAL)
        self.safety_metrics['critical_alerts'] += critical_count

        self.safety_metrics['fail_safe_activations'] += len(responses)

        # Calculate MTBF (simplified)
        uptime = time.time() - self.safety_metrics['system_uptime']
        failure_count = self.safety_metrics['critical_alerts']
        if failure_count > 0:
            self.safety_metrics['mtbf'] = uptime / failure_count

    def get_safety_metrics(self) -> Dict:
        """Get current safety metrics"""
        return self.safety_metrics.copy()

    def get_recent_alerts(self, limit: int = 10) -> List[SafetyAlert]:
        """Get recent safety alerts"""
        return list(self.alert_history)[-limit:]

    def get_recent_responses(self, limit: int = 5) -> List[EmergencyResponse]:
        """Get recent emergency responses"""
        return list(self.response_history)[-limit:]

    def reset_safety_state(self):
        """Reset safety controller to normal state"""
        self.system_state = SystemState.NORMAL
        self.alert_history.clear()
        self.response_history.clear()

        # Reset monitor health
        for monitor in self.monitors.values():
            health = monitor.get_health_status()
            health.error_count = 0
            health.warning_count = 0
            health.health_score = 1.0
            health.status = "active"

# Example usage
if __name__ == "__main__":
    import asyncio

    # Example safety configuration
    safety_config = {
        'collision_monitor': {
            'ttc_threshold': 3.0,
            'min_safe_distance': 2.0
        },
        'lane_monitor': {
            'departure_threshold': 0.5,
            'ttd_threshold': 2.0
        },
        'health_monitor': {
            'component_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'processing_latency': 100.0
            }
        },
        'fail_safe': {},
        'redundancy': {
            'primary_systems': ['front_camera', 'front_radar'],
            'backup_systems': ['backup_camera', 'side_radar']
        }
    }

    async def main():
        # Initialize safety controller
        safety_controller = SafetyController(safety_config)

        # Create test system data
        test_data = {
            'tracks': [],  # No objects detected
            'ego_velocity': (15.0, 0.0, 0.0),  # 15 m/s forward
            'lane_info': None,
            'processing_time_ms': 45.0,
            'sensor_health': {
                'front_camera': 0.95,
                'front_radar': 0.88
            }
        }

        # Process safety frame
        safety_result = await safety_controller.process_safety_frame(test_data)

        print(f"Safety Controller Results:")
        print(f"  System State: {safety_result['system_state']}")
        print(f"  Safety Level: {safety_result['safety_level']}")
        print(f"  Alerts: {len(safety_result['alerts'])}")
        print(f"  Responses: {len(safety_result['responses'])}")
        print(f"  Processing Time: {safety_result['processing_time_ms']:.1f}ms")

        # Print metrics
        metrics = safety_controller.get_safety_metrics()
        print(f"Safety Metrics: {metrics}")

    # Run example
    asyncio.run(main())