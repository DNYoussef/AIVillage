"""
ADAS (Advanced Driver Assistance Systems) Module
Automotive Safety Integrity Level (ASIL-D) compliant implementation

This module provides a comprehensive ADAS framework including:
- Real-time perception and object detection
- Multi-sensor fusion capabilities
- Safety-critical control systems
- Automotive-grade reliability and performance

Modules:
- core: Main ADAS pipeline and processing engine
- sensors: Multi-sensor fusion and calibration
- perception: Object detection, tracking, and scene understanding
- safety: Safety-critical controls and fail-safe mechanisms
"""

from .core.adas_pipeline import AdasPipeline, SensorData, ProcessingResult
from .sensors.sensor_fusion import SensorFusion, FusedDetection
from .perception.perception_engine import PerceptionEngine, Detection, Track
from .safety.safety_controller import SafetyController, SafetyAlert

__version__ = "1.0.0"
__author__ = "ADAS Development Team"
__license__ = "Proprietary - Automotive Safety Critical"

# Safety level compliance
__safety_level__ = "ASIL-D"
__compliance_standards__ = [
    "ISO 26262 (Functional Safety)",
    "ISO 21448 (SOTIF)",
    "AUTOSAR Adaptive Platform"
]

__all__ = [
    # Core pipeline
    'AdasPipeline',
    'SensorData',
    'ProcessingResult',

    # Sensor fusion
    'SensorFusion',
    'FusedDetection',

    # Perception
    'PerceptionEngine',
    'Detection',
    'Track',

    # Safety
    'SafetyController',
    'SafetyAlert'
]