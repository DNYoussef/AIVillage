"""
ADAS Configuration Module

Centralized configuration for all ADAS components with safety-critical parameters
and ISO 26262 ASIL-D compliance settings.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import time

class ASILLevel(Enum):
    """Automotive Safety Integrity Level"""
    A = "ASIL-A"
    B = "ASIL-B"
    C = "ASIL-C"
    D = "ASIL-D"

class SensorType(Enum):
    """Supported sensor types"""
    CAMERA = "camera"
    RADAR = "radar"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GPS = "gps"

@dataclass
class LatencyConstraints:
    """Real-time latency constraints for ADAS operations"""
    perception_max_ms: float = 5.0
    prediction_max_ms: float = 8.0
    planning_max_ms: float = 10.0
    sensor_fusion_max_ms: float = 3.0
    safety_monitor_max_ms: float = 2.0
    total_pipeline_max_ms: float = 10.0

@dataclass
class SafetyConstraints:
    """Safety-critical operational constraints"""
    min_detection_confidence: float = 0.95
    max_false_positive_rate: float = 0.001
    max_false_negative_rate: float = 0.0001
    emergency_brake_distance_m: float = 5.0
    min_following_distance_m: float = 10.0
    max_lateral_acceleration_mps2: float = 4.0
    max_longitudinal_deceleration_mps2: float = 8.0

@dataclass
class SensorConfig:
    """Individual sensor configuration"""
    sensor_type: SensorType
    update_frequency_hz: float
    fov_horizontal_deg: float
    fov_vertical_deg: float
    max_range_m: float
    accuracy_m: float
    reliability_score: float
    asil_level: ASILLevel
    enabled: bool = True

@dataclass
class EdgeConfig:
    """Edge deployment configuration"""
    target_platform: str = "NVIDIA Jetson AGX Xavier"
    max_power_watts: float = 30.0
    target_fps: int = 30
    model_precision: str = "FP16"
    batch_size: int = 1
    tensorrt_optimization: bool = True
    cuda_streams: int = 4

@dataclass
class V2XConfig:
    """Vehicle-to-Everything communication configuration"""
    enabled: bool = True
    communication_range_m: float = 300.0
    update_frequency_hz: float = 10.0
    protocols: List[str] = field(default_factory=lambda: ["DSRC", "C-V2X"])
    encryption_enabled: bool = True
    message_priority_levels: int = 8

class ADASConfig:
    """Comprehensive ADAS system configuration"""

    def __init__(self, config_file: Optional[str] = None):
        self.timestamp = time.time()
        self.version = "1.0.0"

        # Core constraints
        self.latency = LatencyConstraints()
        self.safety = SafetyConstraints()
        self.edge = EdgeConfig()
        self.v2x = V2XConfig()

        # Sensor configurations
        self.sensors = self._initialize_default_sensors()

        # System settings
        self.system_settings = {
            "max_concurrent_threads": 8,
            "memory_pool_size_mb": 2048,
            "log_level": "INFO",
            "telemetry_enabled": True,
            "diagnostic_mode": False,
            "fail_safe_mode_enabled": True
        }

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

        self._validate_configuration()

    def _initialize_default_sensors(self) -> Dict[str, SensorConfig]:
        """Initialize default sensor configuration"""
        return {
            "front_camera": SensorConfig(
                sensor_type=SensorType.CAMERA,
                update_frequency_hz=30.0,
                fov_horizontal_deg=60.0,
                fov_vertical_deg=40.0,
                max_range_m=150.0,
                accuracy_m=0.5,
                reliability_score=0.98,
                asil_level=ASILLevel.D
            ),
            "front_radar": SensorConfig(
                sensor_type=SensorType.RADAR,
                update_frequency_hz=20.0,
                fov_horizontal_deg=30.0,
                fov_vertical_deg=10.0,
                max_range_m=200.0,
                accuracy_m=0.1,
                reliability_score=0.99,
                asil_level=ASILLevel.D
            ),
            "front_lidar": SensorConfig(
                sensor_type=SensorType.LIDAR,
                update_frequency_hz=10.0,
                fov_horizontal_deg=120.0,
                fov_vertical_deg=30.0,
                max_range_m=100.0,
                accuracy_m=0.05,
                reliability_score=0.97,
                asil_level=ASILLevel.C
            ),
            "imu": SensorConfig(
                sensor_type=SensorType.IMU,
                update_frequency_hz=100.0,
                fov_horizontal_deg=360.0,
                fov_vertical_deg=360.0,
                max_range_m=0.0,
                accuracy_m=0.01,
                reliability_score=0.999,
                asil_level=ASILLevel.D
            ),
            "gps": SensorConfig(
                sensor_type=SensorType.GPS,
                update_frequency_hz=5.0,
                fov_horizontal_deg=360.0,
                fov_vertical_deg=90.0,
                max_range_m=float('inf'),
                accuracy_m=1.0,
                reliability_score=0.95,
                asil_level=ASILLevel.B
            )
        }

    def _validate_configuration(self) -> bool:
        """Validate configuration for safety compliance"""
        try:
            # Validate latency constraints
            if self.latency.total_pipeline_max_ms <= 0:
                raise ValueError("Total pipeline latency must be positive")

            # Validate safety constraints
            if self.safety.min_detection_confidence < 0.9:
                raise ValueError("Detection confidence too low for ASIL-D")

            # Validate sensor configurations
            asil_d_sensors = sum(1 for sensor in self.sensors.values()
                               if sensor.asil_level == ASILLevel.D and sensor.enabled)
            if asil_d_sensors < 2:
                raise ValueError("At least 2 ASIL-D sensors required for redundancy")

            # Validate edge configuration
            if self.edge.max_power_watts > 50.0:
                logging.warning("Power consumption may exceed automotive limits")

            return True

        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            raise

    def get_sensor_by_type(self, sensor_type: SensorType) -> List[SensorConfig]:
        """Get all sensors of a specific type"""
        return [sensor for sensor in self.sensors.values()
                if sensor.sensor_type == sensor_type and sensor.enabled]

    def get_asil_d_sensors(self) -> List[SensorConfig]:
        """Get all ASIL-D rated sensors"""
        return [sensor for sensor in self.sensors.values()
                if sensor.asil_level == ASILLevel.D and sensor.enabled]

    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            "version": self.version,
            "timestamp": self.timestamp,
            "latency": self.latency.__dict__,
            "safety": self.safety.__dict__,
            "edge": self.edge.__dict__,
            "v2x": self.v2x.__dict__,
            "sensors": {name: {
                "sensor_type": sensor.sensor_type.value,
                "update_frequency_hz": sensor.update_frequency_hz,
                "fov_horizontal_deg": sensor.fov_horizontal_deg,
                "fov_vertical_deg": sensor.fov_vertical_deg,
                "max_range_m": sensor.max_range_m,
                "accuracy_m": sensor.accuracy_m,
                "reliability_score": sensor.reliability_score,
                "asil_level": sensor.asil_level.value,
                "enabled": sensor.enabled
            } for name, sensor in self.sensors.items()},
            "system_settings": self.system_settings
        }

        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def load_from_file(self, filename: str) -> None:
        """Load configuration from JSON file"""
        with open(filename, 'r') as f:
            config_dict = json.load(f)

        # Update configurations
        if "latency" in config_dict:
            for key, value in config_dict["latency"].items():
                setattr(self.latency, key, value)

        if "safety" in config_dict:
            for key, value in config_dict["safety"].items():
                setattr(self.safety, key, value)

        if "sensors" in config_dict:
            for name, sensor_data in config_dict["sensors"].items():
                self.sensors[name] = SensorConfig(
                    sensor_type=SensorType(sensor_data["sensor_type"]),
                    update_frequency_hz=sensor_data["update_frequency_hz"],
                    fov_horizontal_deg=sensor_data["fov_horizontal_deg"],
                    fov_vertical_deg=sensor_data["fov_vertical_deg"],
                    max_range_m=sensor_data["max_range_m"],
                    accuracy_m=sensor_data["accuracy_m"],
                    reliability_score=sensor_data["reliability_score"],
                    asil_level=ASILLevel(sensor_data["asil_level"]),
                    enabled=sensor_data["enabled"]
                )

        self._validate_configuration()

    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance targets for system monitoring"""
        return {
            "max_latency_ms": self.latency.total_pipeline_max_ms,
            "min_fps": 1000.0 / self.latency.total_pipeline_max_ms,
            "min_detection_confidence": self.safety.min_detection_confidence,
            "max_power_watts": self.edge.max_power_watts,
            "target_fps": self.edge.target_fps
        }