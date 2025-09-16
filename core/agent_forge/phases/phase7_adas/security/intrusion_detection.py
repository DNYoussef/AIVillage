"""
Automotive Intrusion Detection System (IDS)
Implements anomaly detection, cyber attack detection, and system integrity monitoring
Compliant with UN R155 and ISO/SAE 21434 standards
"""

import os
import time
import json
import numpy as np
import threading
import queue
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import logging
import statistics
from datetime import datetime, timedelta

class ThreatLevel(Enum):
    """Security threat severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AttackType(Enum):
    """Types of detected attacks"""
    CAN_FLOODING = "can_flooding"
    REPLAY_ATTACK = "replay_attack"
    MASQUERADE_ATTACK = "masquerade"
    FUZZING_ATTACK = "fuzzing"
    DENIAL_OF_SERVICE = "dos"
    MAN_IN_THE_MIDDLE = "mitm"
    INJECTION_ATTACK = "injection"
    TAMPERING = "tampering"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"

class EventType(Enum):
    """Security event types"""
    SENSOR_ANOMALY = "sensor_anomaly"
    NETWORK_ANOMALY = "network_anomaly"
    SYSTEM_ANOMALY = "system_anomaly"
    AUTHENTICATION_FAILURE = "auth_failure"
    INTEGRITY_VIOLATION = "integrity_violation"
    COMMUNICATION_ERROR = "comm_error"
    PERFORMANCE_DEGRADATION = "perf_degradation"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: float
    event_type: EventType
    threat_level: ThreatLevel
    attack_type: Optional[AttackType]
    source: str
    description: str
    data: Dict[str, Any]
    confidence: float  # 0.0 to 1.0
    mitigation_actions: List[str]

@dataclass
class SensorData:
    """Sensor data structure for anomaly detection"""
    sensor_id: str
    timestamp: float
    value: float
    unit: str
    source: str
    quality: float  # Data quality indicator (0.0 to 1.0)

@dataclass
class CANMessage:
    """CAN message structure for network monitoring"""
    can_id: int
    timestamp: float
    data: bytes
    dlc: int
    source_ecu: str
    frequency: Optional[float] = None

class AnomalyDetector:
    """Base class for anomaly detection algorithms"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.is_trained = False

    def train(self, data: List[Any]) -> bool:
        """Train the anomaly detection model"""
        raise NotImplementedError

    def detect(self, data: Any) -> Tuple[bool, float]:
        """Detect anomaly in data. Returns (is_anomaly, confidence)"""
        raise NotImplementedError

    def update(self, data: Any, is_anomaly: bool):
        """Update model with new data and feedback"""
        pass

class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical anomaly detection using Z-score and IQR methods"""

    def __init__(self, name: str, sensitivity: float = 2.5):
        super().__init__(name)
        self.sensitivity = sensitivity  # Z-score threshold
        self.historical_data = deque(maxlen=1000)
        self.mean = 0.0
        self.std = 0.0
        self.q1 = 0.0
        self.q3 = 0.0
        self.iqr = 0.0

    def train(self, data: List[float]) -> bool:
        """Train with historical normal data"""
        if len(data) < 10:
            self.logger.warning("Insufficient training data")
            return False

        self.historical_data.extend(data)
        self._update_statistics()
        self.is_trained = True
        self.logger.info(f"Statistical detector trained with {len(data)} samples")
        return True

    def _update_statistics(self):
        """Update statistical parameters"""
        if len(self.historical_data) < 2:
            return

        data_list = list(self.historical_data)
        self.mean = statistics.mean(data_list)
        self.std = statistics.stdev(data_list) if len(data_list) > 1 else 0.0

        # Calculate quartiles for IQR method
        sorted_data = sorted(data_list)
        n = len(sorted_data)
        self.q1 = sorted_data[n // 4] if n >= 4 else self.mean
        self.q3 = sorted_data[3 * n // 4] if n >= 4 else self.mean
        self.iqr = self.q3 - self.q1

    def detect(self, value: float) -> Tuple[bool, float]:
        """Detect anomaly using Z-score and IQR methods"""
        if not self.is_trained:
            return False, 0.0

        # Z-score method
        z_score = abs(value - self.mean) / max(self.std, 1e-10)
        z_anomaly = z_score > self.sensitivity

        # IQR method
        iqr_lower = self.q1 - 1.5 * self.iqr
        iqr_upper = self.q3 + 1.5 * self.iqr
        iqr_anomaly = value < iqr_lower or value > iqr_upper

        # Combined detection
        is_anomaly = z_anomaly or iqr_anomaly
        confidence = min(z_score / self.sensitivity, 1.0) if z_anomaly else 0.0

        if iqr_anomaly and not z_anomaly:
            confidence = 0.7  # Medium confidence for IQR-only detection

        return is_anomaly, confidence

    def update(self, value: float, is_anomaly: bool):
        """Update model with new data"""
        if not is_anomaly:  # Only add normal data to training set
            self.historical_data.append(value)
            if len(self.historical_data) % 50 == 0:  # Periodic retraining
                self._update_statistics()

class FrequencyAnomalyDetector(AnomalyDetector):
    """Detect anomalies in message/event frequencies"""

    def __init__(self, name: str, window_size: int = 60):
        super().__init__(name)
        self.window_size = window_size  # seconds
        self.event_windows = defaultdict(lambda: deque(maxlen=1000))
        self.normal_frequencies = {}
        self.frequency_thresholds = {}

    def train(self, event_history: Dict[str, List[float]]) -> bool:
        """Train with historical event timestamps"""
        for event_type, timestamps in event_history.items():
            if len(timestamps) < 10:
                continue

            # Calculate frequencies in time windows
            frequencies = self._calculate_frequencies(timestamps)

            if frequencies:
                self.normal_frequencies[event_type] = {
                    'mean': statistics.mean(frequencies),
                    'std': statistics.stdev(frequencies) if len(frequencies) > 1 else 0.0
                }

                # Set threshold at 3 standard deviations
                mean_freq = self.normal_frequencies[event_type]['mean']
                std_freq = self.normal_frequencies[event_type]['std']
                self.frequency_thresholds[event_type] = {
                    'min': max(0, mean_freq - 3 * std_freq),
                    'max': mean_freq + 3 * std_freq
                }

        self.is_trained = True
        self.logger.info(f"Frequency detector trained for {len(self.normal_frequencies)} event types")
        return True

    def _calculate_frequencies(self, timestamps: List[float]) -> List[float]:
        """Calculate event frequencies in time windows"""
        if len(timestamps) < 2:
            return []

        timestamps = sorted(timestamps)
        frequencies = []

        start_time = timestamps[0]
        end_time = timestamps[-1]

        current_time = start_time
        while current_time + self.window_size <= end_time:
            window_end = current_time + self.window_size
            count = sum(1 for ts in timestamps if current_time <= ts < window_end)
            frequencies.append(count / self.window_size)  # Events per second
            current_time += self.window_size / 2  # 50% overlap

        return frequencies

    def detect(self, event_type: str, timestamp: float) -> Tuple[bool, float]:
        """Detect frequency anomaly for event type"""
        if event_type not in self.normal_frequencies:
            return False, 0.0

        # Add timestamp to window
        self.event_windows[event_type].append(timestamp)

        # Calculate current frequency
        current_time = time.time()
        recent_events = [ts for ts in self.event_windows[event_type]
                        if current_time - ts <= self.window_size]

        current_frequency = len(recent_events) / self.window_size

        # Check against thresholds
        thresholds = self.frequency_thresholds[event_type]

        if current_frequency < thresholds['min']:
            # Too low frequency (potential DoS or system failure)
            confidence = min((thresholds['min'] - current_frequency) / thresholds['min'], 1.0)
            return True, confidence
        elif current_frequency > thresholds['max']:
            # Too high frequency (potential flooding attack)
            confidence = min((current_frequency - thresholds['max']) / thresholds['max'], 1.0)
            return True, confidence

        return False, 0.0

class PatternAnomalyDetector(AnomalyDetector):
    """Detect anomalous patterns in sequential data"""

    def __init__(self, name: str, sequence_length: int = 5):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.normal_patterns = set()
        self.pattern_frequencies = defaultdict(int)
        self.recent_sequence = deque(maxlen=sequence_length)

    def train(self, sequences: List[List[Any]]) -> bool:
        """Train with normal sequence patterns"""
        for sequence in sequences:
            if len(sequence) >= self.sequence_length:
                for i in range(len(sequence) - self.sequence_length + 1):
                    pattern = tuple(sequence[i:i + self.sequence_length])
                    self.normal_patterns.add(pattern)
                    self.pattern_frequencies[pattern] += 1

        self.is_trained = True
        self.logger.info(f"Pattern detector trained with {len(self.normal_patterns)} patterns")
        return True

    def detect(self, event: Any) -> Tuple[bool, float]:
        """Detect pattern anomaly"""
        if not self.is_trained:
            return False, 0.0

        self.recent_sequence.append(event)

        if len(self.recent_sequence) < self.sequence_length:
            return False, 0.0

        current_pattern = tuple(self.recent_sequence)

        if current_pattern not in self.normal_patterns:
            # Calculate confidence based on how rare the pattern is
            confidence = 1.0  # Unknown pattern gets high confidence
            return True, confidence

        # Pattern is known - check frequency
        frequency = self.pattern_frequencies[current_pattern]
        if frequency < 2:  # Very rare pattern
            return True, 0.7

        return False, 0.0

class NetworkIntrusionDetector:
    """Network-level intrusion detection for CAN bus and V2X"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.can_detectors = {}
        self.v2x_detectors = {}
        self.message_history = defaultdict(list)
        self.attack_signatures = {}
        self._load_attack_signatures()

    def _load_attack_signatures(self):
        """Load known attack signatures"""
        self.attack_signatures = {
            AttackType.CAN_FLOODING: {
                'min_frequency': 100,  # messages per second
                'duration_threshold': 5  # seconds
            },
            AttackType.REPLAY_ATTACK: {
                'exact_match_threshold': 3,  # identical messages
                'time_window': 1.0  # seconds
            },
            AttackType.FUZZING_ATTACK: {
                'random_data_threshold': 0.8,  # entropy threshold
                'frequency_threshold': 50  # messages per second
            }
        }

    def initialize_can_monitoring(self, can_id: int):
        """Initialize monitoring for specific CAN ID"""
        detector_name = f"CAN_{can_id:X}"
        self.can_detectors[can_id] = {
            'frequency': FrequencyAnomalyDetector(f"{detector_name}_freq"),
            'pattern': PatternAnomalyDetector(f"{detector_name}_pattern"),
            'last_messages': deque(maxlen=100)
        }

    def analyze_can_message(self, message: CANMessage) -> List[SecurityEvent]:
        """Analyze CAN message for intrusion indicators"""
        events = []

        if message.can_id not in self.can_detectors:
            self.initialize_can_monitoring(message.can_id)

        detectors = self.can_detectors[message.can_id]

        # Store message for analysis
        detectors['last_messages'].append(message)
        self.message_history[message.can_id].append(message)

        # Check for flooding attack
        flooding_event = self._detect_can_flooding(message)
        if flooding_event:
            events.append(flooding_event)

        # Check for replay attack
        replay_event = self._detect_replay_attack(message)
        if replay_event:
            events.append(replay_event)

        # Check for fuzzing attack
        fuzzing_event = self._detect_fuzzing_attack(message)
        if fuzzing_event:
            events.append(fuzzing_event)

        # Frequency anomaly detection
        is_freq_anomaly, freq_confidence = detectors['frequency'].detect(
            f"CAN_{message.can_id:X}", message.timestamp
        )
        if is_freq_anomaly:
            events.append(SecurityEvent(
                event_id=f"FREQ_ANOMALY_{message.can_id:X}_{int(message.timestamp)}",
                timestamp=message.timestamp,
                event_type=EventType.NETWORK_ANOMALY,
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                source=f"CAN_ID_{message.can_id:X}",
                description=f"Frequency anomaly detected for CAN ID 0x{message.can_id:X}",
                data={'can_id': message.can_id, 'frequency_confidence': freq_confidence},
                confidence=freq_confidence,
                mitigation_actions=['monitor_traffic', 'rate_limit']
            ))

        # Pattern anomaly detection
        is_pattern_anomaly, pattern_confidence = detectors['pattern'].detect(message.data)
        if is_pattern_anomaly:
            events.append(SecurityEvent(
                event_id=f"PATTERN_ANOMALY_{message.can_id:X}_{int(message.timestamp)}",
                timestamp=message.timestamp,
                event_type=EventType.NETWORK_ANOMALY,
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                source=f"CAN_ID_{message.can_id:X}",
                description=f"Pattern anomaly detected for CAN ID 0x{message.can_id:X}",
                data={'can_id': message.can_id, 'data': message.data.hex(), 'pattern_confidence': pattern_confidence},
                confidence=pattern_confidence,
                mitigation_actions=['analyze_pattern', 'investigate_source']
            ))

        return events

    def _detect_can_flooding(self, message: CANMessage) -> Optional[SecurityEvent]:
        """Detect CAN bus flooding attack"""
        recent_messages = [msg for msg in self.message_history[message.can_id]
                          if message.timestamp - msg.timestamp <= 1.0]

        frequency = len(recent_messages)
        threshold = self.attack_signatures[AttackType.CAN_FLOODING]['min_frequency']

        if frequency > threshold:
            return SecurityEvent(
                event_id=f"CAN_FLOOD_{message.can_id:X}_{int(message.timestamp)}",
                timestamp=message.timestamp,
                event_type=EventType.NETWORK_ANOMALY,
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.CAN_FLOODING,
                source=f"CAN_ID_{message.can_id:X}",
                description=f"CAN flooding detected: {frequency} msg/s > {threshold} threshold",
                data={'can_id': message.can_id, 'frequency': frequency, 'threshold': threshold},
                confidence=min(frequency / threshold, 1.0),
                mitigation_actions=['rate_limit', 'block_source', 'alert_operator']
            )

        return None

    def _detect_replay_attack(self, message: CANMessage) -> Optional[SecurityEvent]:
        """Detect replay attack on CAN bus"""
        if message.can_id not in self.can_detectors:
            return None

        recent_messages = self.can_detectors[message.can_id]['last_messages']
        threshold = self.attack_signatures[AttackType.REPLAY_ATTACK]['exact_match_threshold']
        time_window = self.attack_signatures[AttackType.REPLAY_ATTACK]['time_window']

        # Count exact matches within time window
        exact_matches = 0
        for prev_msg in recent_messages:
            if (message.timestamp - prev_msg.timestamp <= time_window and
                message.data == prev_msg.data and
                message.dlc == prev_msg.dlc):
                exact_matches += 1

        if exact_matches >= threshold:
            return SecurityEvent(
                event_id=f"REPLAY_{message.can_id:X}_{int(message.timestamp)}",
                timestamp=message.timestamp,
                event_type=EventType.NETWORK_ANOMALY,
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.REPLAY_ATTACK,
                source=f"CAN_ID_{message.can_id:X}",
                description=f"Replay attack detected: {exact_matches} identical messages",
                data={'can_id': message.can_id, 'matches': exact_matches, 'data': message.data.hex()},
                confidence=min(exact_matches / threshold, 1.0),
                mitigation_actions=['block_duplicate', 'verify_source', 'sequence_check']
            )

        return None

    def _detect_fuzzing_attack(self, message: CANMessage) -> Optional[SecurityEvent]:
        """Detect fuzzing attack based on data randomness"""
        if len(message.data) == 0:
            return None

        # Calculate entropy of message data
        entropy = self._calculate_entropy(message.data)
        threshold = self.attack_signatures[AttackType.FUZZING_ATTACK]['random_data_threshold']

        # Check frequency
        recent_messages = [msg for msg in self.message_history[message.can_id]
                          if message.timestamp - msg.timestamp <= 1.0]
        frequency = len(recent_messages)
        freq_threshold = self.attack_signatures[AttackType.FUZZING_ATTACK]['frequency_threshold']

        if entropy > threshold and frequency > freq_threshold:
            return SecurityEvent(
                event_id=f"FUZZING_{message.can_id:X}_{int(message.timestamp)}",
                timestamp=message.timestamp,
                event_type=EventType.NETWORK_ANOMALY,
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.FUZZING_ATTACK,
                source=f"CAN_ID_{message.can_id:X}",
                description=f"Fuzzing attack detected: high entropy ({entropy:.2f}) and frequency ({frequency})",
                data={'can_id': message.can_id, 'entropy': entropy, 'frequency': frequency},
                confidence=min((entropy - threshold) / (1.0 - threshold), 1.0),
                mitigation_actions=['block_source', 'validate_data', 'alert_security']
            )

        return None

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if len(data) == 0:
            return 0.0

        # Count byte frequencies
        freq = defaultdict(int)
        for byte in data:
            freq[byte] += 1

        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        for count in freq.values():
            p = count / data_len
            if p > 0:
                entropy -= p * np.log2(p)

        return entropy / 8.0  # Normalize to 0-1 range

class SensorAnomalyDetector:
    """Detect anomalies in sensor data"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sensor_detectors = {}
        self.sensor_baselines = {}

    def initialize_sensor(self, sensor_id: str, sensor_type: str):
        """Initialize anomaly detection for sensor"""
        self.sensor_detectors[sensor_id] = {
            'statistical': StatisticalAnomalyDetector(f"{sensor_id}_stats"),
            'type': sensor_type,
            'history': deque(maxlen=1000)
        }

        # Set sensor-specific parameters
        if sensor_type == 'speed':
            self.sensor_detectors[sensor_id]['min_value'] = 0.0
            self.sensor_detectors[sensor_id]['max_value'] = 200.0  # km/h
        elif sensor_type == 'temperature':
            self.sensor_detectors[sensor_id]['min_value'] = -40.0
            self.sensor_detectors[sensor_id]['max_value'] = 150.0  # Celsius
        elif sensor_type == 'pressure':
            self.sensor_detectors[sensor_id]['min_value'] = 0.0
            self.sensor_detectors[sensor_id]['max_value'] = 1000.0  # kPa

    def analyze_sensor_data(self, sensor_data: SensorData) -> List[SecurityEvent]:
        """Analyze sensor data for anomalies"""
        events = []

        if sensor_data.sensor_id not in self.sensor_detectors:
            # Auto-detect sensor type based on sensor_id
            sensor_type = self._detect_sensor_type(sensor_data.sensor_id)
            self.initialize_sensor(sensor_data.sensor_id, sensor_type)

        detector_info = self.sensor_detectors[sensor_data.sensor_id]
        detector_info['history'].append(sensor_data)

        # Range check
        range_event = self._check_sensor_range(sensor_data)
        if range_event:
            events.append(range_event)

        # Statistical anomaly detection
        is_anomaly, confidence = detector_info['statistical'].detect(sensor_data.value)
        if is_anomaly:
            events.append(SecurityEvent(
                event_id=f"SENSOR_ANOMALY_{sensor_data.sensor_id}_{int(sensor_data.timestamp)}",
                timestamp=sensor_data.timestamp,
                event_type=EventType.SENSOR_ANOMALY,
                threat_level=ThreatLevel.MEDIUM,
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                source=sensor_data.sensor_id,
                description=f"Statistical anomaly in sensor {sensor_data.sensor_id}",
                data={
                    'sensor_id': sensor_data.sensor_id,
                    'value': sensor_data.value,
                    'unit': sensor_data.unit,
                    'confidence': confidence,
                    'quality': sensor_data.quality
                },
                confidence=confidence,
                mitigation_actions=['verify_sensor', 'check_calibration', 'investigate_interference']
            ))

        # Data quality check
        if sensor_data.quality < 0.5:
            events.append(SecurityEvent(
                event_id=f"SENSOR_QUALITY_{sensor_data.sensor_id}_{int(sensor_data.timestamp)}",
                timestamp=sensor_data.timestamp,
                event_type=EventType.SENSOR_ANOMALY,
                threat_level=ThreatLevel.LOW,
                attack_type=None,
                source=sensor_data.sensor_id,
                description=f"Low data quality for sensor {sensor_data.sensor_id}",
                data={
                    'sensor_id': sensor_data.sensor_id,
                    'quality': sensor_data.quality,
                    'threshold': 0.5
                },
                confidence=1.0 - sensor_data.quality,
                mitigation_actions=['check_sensor_connection', 'calibrate_sensor']
            ))

        # Update detector with normal data
        if not is_anomaly and sensor_data.quality > 0.8:
            detector_info['statistical'].update(sensor_data.value, False)

        return events

    def _detect_sensor_type(self, sensor_id: str) -> str:
        """Auto-detect sensor type from sensor ID"""
        sensor_id_lower = sensor_id.lower()
        if 'speed' in sensor_id_lower or 'velocity' in sensor_id_lower:
            return 'speed'
        elif 'temp' in sensor_id_lower:
            return 'temperature'
        elif 'press' in sensor_id_lower:
            return 'pressure'
        elif 'accel' in sensor_id_lower:
            return 'acceleration'
        elif 'gyro' in sensor_id_lower:
            return 'gyroscope'
        else:
            return 'generic'

    def _check_sensor_range(self, sensor_data: SensorData) -> Optional[SecurityEvent]:
        """Check if sensor value is within expected range"""
        detector_info = self.sensor_detectors[sensor_data.sensor_id]

        min_value = detector_info.get('min_value', float('-inf'))
        max_value = detector_info.get('max_value', float('inf'))

        if sensor_data.value < min_value or sensor_data.value > max_value:
            return SecurityEvent(
                event_id=f"RANGE_ERROR_{sensor_data.sensor_id}_{int(sensor_data.timestamp)}",
                timestamp=sensor_data.timestamp,
                event_type=EventType.SENSOR_ANOMALY,
                threat_level=ThreatLevel.HIGH,
                attack_type=AttackType.TAMPERING,
                source=sensor_data.sensor_id,
                description=f"Sensor value out of range: {sensor_data.value} not in [{min_value}, {max_value}]",
                data={
                    'sensor_id': sensor_data.sensor_id,
                    'value': sensor_data.value,
                    'min_value': min_value,
                    'max_value': max_value
                },
                confidence=1.0,
                mitigation_actions=['verify_sensor_integrity', 'check_tampering', 'isolate_sensor']
            )

        return None

class SystemIntegrityMonitor:
    """Monitor system integrity and detect tampering"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.integrity_baselines = {}
        self.performance_baselines = {}
        self.system_metrics = defaultdict(list)

    def establish_baseline(self, component: str, metrics: Dict[str, float]):
        """Establish baseline metrics for system component"""
        self.integrity_baselines[component] = {
            'metrics': metrics.copy(),
            'timestamp': time.time(),
            'hash': hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()
        }

        self.logger.info(f"Baseline established for component: {component}")

    def monitor_system_integrity(self, component: str, current_metrics: Dict[str, float]) -> List[SecurityEvent]:
        """Monitor system integrity against baseline"""
        events = []

        if component not in self.integrity_baselines:
            self.establish_baseline(component, current_metrics)
            return events

        baseline = self.integrity_baselines[component]

        # Check for metric deviations
        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline['metrics']:
                continue

            baseline_value = baseline['metrics'][metric_name]

            # Calculate relative deviation
            if baseline_value != 0:
                deviation = abs(current_value - baseline_value) / abs(baseline_value)
            else:
                deviation = abs(current_value)

            # Flag significant deviations (>20% change)
            if deviation > 0.2:
                events.append(SecurityEvent(
                    event_id=f"INTEGRITY_{component}_{metric_name}_{int(time.time())}",
                    timestamp=time.time(),
                    event_type=EventType.INTEGRITY_VIOLATION,
                    threat_level=ThreatLevel.MEDIUM,
                    attack_type=AttackType.TAMPERING,
                    source=component,
                    description=f"Integrity deviation in {component}.{metric_name}",
                    data={
                        'component': component,
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'deviation': deviation
                    },
                    confidence=min(deviation / 0.2, 1.0),
                    mitigation_actions=['verify_component', 'check_tampering', 'restore_backup']
                ))

        return events

class SecurityEventLogger:
    """Security event logging and management"""

    def __init__(self, log_file: str = "security_events.log"):
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.events = deque(maxlen=10000)
        self.event_counts = defaultdict(int)
        self.severity_counts = defaultdict(int)

    def log_event(self, event: SecurityEvent):
        """Log security event"""
        self.events.append(event)

        # Update counters
        if event.attack_type:
            self.event_counts[event.attack_type.value] += 1
        self.severity_counts[event.threat_level.value] += 1

        # Write to log file
        log_entry = {
            'timestamp': datetime.fromtimestamp(event.timestamp).isoformat(),
            'event': asdict(event)
        }

        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write event log: {e}")

        # Log to system logger based on severity
        if event.threat_level == ThreatLevel.CRITICAL:
            self.logger.critical(f"CRITICAL SECURITY EVENT: {event.description}")
        elif event.threat_level == ThreatLevel.HIGH:
            self.logger.error(f"HIGH SECURITY EVENT: {event.description}")
        elif event.threat_level == ThreatLevel.MEDIUM:
            self.logger.warning(f"MEDIUM SECURITY EVENT: {event.description}")
        else:
            self.logger.info(f"LOW SECURITY EVENT: {event.description}")

    def get_recent_events(self, since: Optional[float] = None, count: Optional[int] = None) -> List[SecurityEvent]:
        """Get recent security events"""
        events = list(self.events)

        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        if count is not None:
            events = events[-count:]

        return events

    def get_event_summary(self) -> Dict[str, Any]:
        """Get summary of security events"""
        return {
            'total_events': len(self.events),
            'event_counts': dict(self.event_counts),
            'severity_counts': dict(self.severity_counts),
            'recent_critical': len([e for e in self.events if e.threat_level == ThreatLevel.CRITICAL and time.time() - e.timestamp < 3600])
        }

class AutomotiveIntrusionDetectionSystem:
    """Main automotive intrusion detection system"""

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.network_detector = NetworkIntrusionDetector()
        self.sensor_detector = SensorAnomalyDetector()
        self.integrity_monitor = SystemIntegrityMonitor()
        self.event_logger = SecurityEventLogger(f"ids_events_{vehicle_id}.log")

        # Processing queue and thread
        self.event_queue = queue.Queue(maxsize=1000)
        self.processing_thread = None
        self.running = False

        # Detection statistics
        self.stats = {
            'messages_analyzed': 0,
            'sensors_monitored': 0,
            'events_detected': 0,
            'attacks_prevented': 0
        }

    def start(self):
        """Start the intrusion detection system"""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

        self.logger.info(f"Automotive IDS started for vehicle {self.vehicle_id}")

    def stop(self):
        """Stop the intrusion detection system"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.logger.info("Automotive IDS stopped")

    def process_can_message(self, message: CANMessage):
        """Process CAN message for intrusion detection"""
        try:
            self.event_queue.put(('can_message', message), block=False)
            self.stats['messages_analyzed'] += 1
        except queue.Full:
            self.logger.warning("Event queue full, dropping CAN message")

    def process_sensor_data(self, sensor_data: SensorData):
        """Process sensor data for anomaly detection"""
        try:
            self.event_queue.put(('sensor_data', sensor_data), block=False)
            self.stats['sensors_monitored'] += 1
        except queue.Full:
            self.logger.warning("Event queue full, dropping sensor data")

    def monitor_system_component(self, component: str, metrics: Dict[str, float]):
        """Monitor system component integrity"""
        try:
            self.event_queue.put(('system_metrics', (component, metrics)), block=False)
        except queue.Full:
            self.logger.warning("Event queue full, dropping system metrics")

    def _processing_loop(self):
        """Main processing loop for security events"""
        while self.running:
            try:
                # Get item from queue with timeout
                item_type, data = self.event_queue.get(timeout=1.0)

                events = []

                if item_type == 'can_message':
                    events = self.network_detector.analyze_can_message(data)
                elif item_type == 'sensor_data':
                    events = self.sensor_detector.analyze_sensor_data(data)
                elif item_type == 'system_metrics':
                    component, metrics = data
                    events = self.integrity_monitor.monitor_system_integrity(component, metrics)

                # Process detected events
                for event in events:
                    self.event_logger.log_event(event)
                    self.stats['events_detected'] += 1

                    # Trigger mitigation actions for high-severity events
                    if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        self._trigger_mitigation(event)

                self.event_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")

    def _trigger_mitigation(self, event: SecurityEvent):
        """Trigger mitigation actions for security event"""
        self.logger.warning(f"Triggering mitigation for event: {event.event_id}")

        # Execute mitigation actions
        for action in event.mitigation_actions:
            self._execute_mitigation_action(action, event)

        self.stats['attacks_prevented'] += 1

    def _execute_mitigation_action(self, action: str, event: SecurityEvent):
        """Execute specific mitigation action"""
        # In a real system, these would interface with actual vehicle systems
        if action == 'block_source':
            self.logger.info(f"Blocking source: {event.source}")
        elif action == 'rate_limit':
            self.logger.info(f"Applying rate limiting to: {event.source}")
        elif action == 'alert_operator':
            self.logger.critical(f"OPERATOR ALERT: {event.description}")
        elif action == 'isolate_sensor':
            self.logger.warning(f"Isolating sensor: {event.source}")
        elif action == 'verify_component':
            self.logger.info(f"Initiating component verification: {event.source}")
        else:
            self.logger.info(f"Executing mitigation action: {action} for {event.source}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        event_summary = self.event_logger.get_event_summary()

        return {
            'vehicle_id': self.vehicle_id,
            'running': self.running,
            'timestamp': time.time(),
            'statistics': self.stats.copy(),
            'event_summary': event_summary,
            'queue_size': self.event_queue.qsize(),
            'sensors_monitored': len(self.sensor_detector.sensor_detectors),
            'can_ids_monitored': len(self.network_detector.can_detectors)
        }

    def get_recent_threats(self, severity: Optional[ThreatLevel] = None, count: int = 10) -> List[SecurityEvent]:
        """Get recent security threats"""
        recent_events = self.event_logger.get_recent_events(count=count * 2)  # Get more to filter

        if severity:
            recent_events = [e for e in recent_events if e.threat_level == severity]

        return recent_events[-count:]

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize IDS
    ids = AutomotiveIntrusionDetectionSystem("VEHICLE_001")
    ids.start()

    # Simulate CAN message processing
    can_message = CANMessage(
        can_id=0x123,
        timestamp=time.time(),
        data=b'\x01\x02\x03\x04',
        dlc=4,
        source_ecu='ENGINE_ECU'
    )
    ids.process_can_message(can_message)

    # Simulate sensor data processing
    sensor_data = SensorData(
        sensor_id='SPEED_SENSOR_1',
        timestamp=time.time(),
        value=85.5,
        unit='km/h',
        source='WHEEL_SENSOR',
        quality=0.95
    )
    ids.process_sensor_data(sensor_data)

    # Simulate system monitoring
    ids.monitor_system_component('ENGINE_CONTROLLER', {
        'cpu_usage': 45.2,
        'memory_usage': 68.1,
        'temperature': 75.3
    })

    # Wait for processing
    time.sleep(2)

    # Get system status
    status = ids.get_system_status()
    print(f"IDS Status: {json.dumps(status, indent=2)}")

    # Get recent threats
    threats = ids.get_recent_threats(ThreatLevel.HIGH)
    print(f"Recent high threats: {len(threats)}")

    # Stop IDS
    ids.stop()