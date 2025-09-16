"""
ADAS Sensor Fusion Agent - Phase 7
Multi-sensor data integration with real-time processing
"""

import asyncio
import logging
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from concurrent.futures import ThreadPoolExecutor
import queue

class SensorType(Enum):
    CAMERA = "camera"
    RADAR = "radar"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GPS = "gps"

@dataclass
class SensorData:
    """Standardized sensor data structure"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Dict[str, Any]
    confidence: float
    health_status: str
    processing_latency: float = 0.0

@dataclass
class FusedOutput:
    """Fused sensor output structure"""
    timestamp: float
    objects: List[Dict[str, Any]]
    environment: Dict[str, Any]
    vehicle_state: Dict[str, Any]
    confidence_map: Dict[str, float]
    fusion_latency: float
    quality_metrics: Dict[str, float]

class SensorFusionAgent:
    """
    Advanced multi-sensor fusion agent for ADAS systems
    Integrates camera, radar, lidar, and other sensor data in real-time
    """

    def __init__(self, agent_id: str = "sensor_fusion_001"):
        self.agent_id = agent_id
        self.logger = self._setup_logging()

        # Real-time constraints
        self.max_processing_time = 0.008  # 8ms target
        self.fusion_frequency = 100  # 100Hz

        # Sensor management
        self.active_sensors: Dict[str, SensorData] = {}
        self.sensor_calibration: Dict[str, Dict] = {}
        self.sensor_health: Dict[str, float] = {}

        # Fusion pipeline
        self.fusion_queue = queue.PriorityQueue(maxsize=1000)
        self.output_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.is_running = False

        # Performance tracking
        self.performance_metrics = {
            'fusion_latency': [],
            'data_drops': 0,
            'fusion_rate': 0,
            'sensor_sync_rate': 0
        }

        # Safety monitoring
        self.safety_thresholds = {
            'max_latency': 0.010,  # 10ms max
            'min_sensor_health': 0.8,
            'max_data_age': 0.050  # 50ms max age
        }

        self.executor = ThreadPoolExecutor(max_workers=4)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"ADAS.SensorFusion.{self.agent_id}")
        logger.setLevel(logging.INFO)

        # Real-time handler with minimal overhead
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self) -> bool:
        """Initialize sensor fusion system"""
        try:
            self.logger.info("Initializing ADAS Sensor Fusion Agent")

            # Load sensor calibration data
            await self._load_sensor_calibration()

            # Initialize fusion algorithms
            await self._initialize_fusion_algorithms()

            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._fusion_processing_loop,
                daemon=True
            )
            self.processing_thread.start()

            self.logger.info("Sensor Fusion Agent initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _load_sensor_calibration(self):
        """Load sensor calibration parameters"""
        # Camera calibration
        self.sensor_calibration[SensorType.CAMERA.value] = {
            'intrinsic_matrix': np.eye(3),
            'distortion_coeffs': np.zeros(5),
            'resolution': (1920, 1080),
            'fov': 60.0
        }

        # Radar calibration
        self.sensor_calibration[SensorType.RADAR.value] = {
            'range_resolution': 0.1,
            'angular_resolution': 1.0,
            'max_range': 200.0,
            'mounting_position': [0, 0, 1.2]
        }

        # Lidar calibration
        self.sensor_calibration[SensorType.LIDAR.value] = {
            'vertical_fov': 40.0,
            'horizontal_fov': 360.0,
            'range_accuracy': 0.03,
            'mounting_position': [0, 0, 1.8]
        }

        self.logger.info("Sensor calibration loaded")

    async def _initialize_fusion_algorithms(self):
        """Initialize fusion algorithms"""
        # Kalman filter for object tracking
        self.object_trackers = {}

        # Particle filter for localization
        self.localization_filter = None

        # Occupancy grid for environment mapping
        self.occupancy_grid = np.zeros((200, 200))

        self.logger.info("Fusion algorithms initialized")

    def _fusion_processing_loop(self):
        """Main fusion processing loop"""
        self.logger.info("Starting fusion processing loop")

        while self.is_running:
            try:
                # Process fusion queue with timeout
                try:
                    priority, sensor_data = self.fusion_queue.get(timeout=0.001)
                    processing_start = time.perf_counter()

                    # Perform sensor fusion
                    fused_result = self._perform_fusion(sensor_data)

                    # Check processing time
                    processing_time = time.perf_counter() - processing_start
                    if processing_time > self.max_processing_time:
                        self.logger.warning(
                            f"Fusion processing exceeded time limit: {processing_time*1000:.2f}ms"
                        )

                    # Update performance metrics
                    self.performance_metrics['fusion_latency'].append(processing_time)
                    if len(self.performance_metrics['fusion_latency']) > 1000:
                        self.performance_metrics['fusion_latency'].pop(0)

                    # Output fused data
                    if fused_result:
                        self.output_queue.put(fused_result)

                    self.fusion_queue.task_done()

                except queue.Empty:
                    # No data available, continue
                    continue

            except Exception as e:
                self.logger.error(f"Fusion processing error: {e}")
                continue

    def _perform_fusion(self, sensor_data_list: List[SensorData]) -> Optional[FusedOutput]:
        """Perform multi-sensor fusion"""
        try:
            fusion_start = time.perf_counter()

            # Group sensor data by type
            sensor_groups = self._group_sensors_by_type(sensor_data_list)

            # Perform temporal alignment
            aligned_data = self._temporal_alignment(sensor_groups)

            # Spatial fusion
            fused_objects = self._spatial_fusion(aligned_data)

            # Environment fusion
            environment_state = self._environment_fusion(aligned_data)

            # Vehicle state fusion
            vehicle_state = self._vehicle_state_fusion(aligned_data)

            # Calculate confidence metrics
            confidence_map = self._calculate_confidence(sensor_groups)

            fusion_time = time.perf_counter() - fusion_start

            # Quality assessment
            quality_metrics = self._assess_fusion_quality(
                fused_objects, environment_state, vehicle_state
            )

            return FusedOutput(
                timestamp=time.time(),
                objects=fused_objects,
                environment=environment_state,
                vehicle_state=vehicle_state,
                confidence_map=confidence_map,
                fusion_latency=fusion_time,
                quality_metrics=quality_metrics
            )

        except Exception as e:
            self.logger.error(f"Fusion processing failed: {e}")
            return None

    def _group_sensors_by_type(self, sensor_data_list: List[SensorData]) -> Dict[SensorType, List[SensorData]]:
        """Group sensor data by sensor type"""
        groups = {sensor_type: [] for sensor_type in SensorType}

        for sensor_data in sensor_data_list:
            groups[sensor_data.sensor_type].append(sensor_data)

        return groups

    def _temporal_alignment(self, sensor_groups: Dict[SensorType, List[SensorData]]) -> Dict[SensorType, List[SensorData]]:
        """Align sensor data temporally"""
        reference_time = time.time()
        aligned_data = {}

        for sensor_type, data_list in sensor_groups.items():
            # Filter data within temporal window
            aligned_list = []
            for data in data_list:
                age = reference_time - data.timestamp
                if age <= self.safety_thresholds['max_data_age']:
                    aligned_list.append(data)
                else:
                    self.logger.warning(f"Dropping old data from {sensor_type.value}: age={age*1000:.1f}ms")

            aligned_data[sensor_type] = aligned_list

        return aligned_data

    def _spatial_fusion(self, aligned_data: Dict[SensorType, List[SensorData]]) -> List[Dict[str, Any]]:
        """Perform spatial fusion of object detections"""
        fused_objects = []

        try:
            # Camera objects
            camera_objects = self._extract_camera_objects(aligned_data.get(SensorType.CAMERA, []))

            # Radar objects
            radar_objects = self._extract_radar_objects(aligned_data.get(SensorType.RADAR, []))

            # Lidar objects
            lidar_objects = self._extract_lidar_objects(aligned_data.get(SensorType.LIDAR, []))

            # Associate and fuse objects
            fused_objects = self._associate_and_fuse_objects(
                camera_objects, radar_objects, lidar_objects
            )

        except Exception as e:
            self.logger.error(f"Spatial fusion failed: {e}")

        return fused_objects

    def _extract_camera_objects(self, camera_data: List[SensorData]) -> List[Dict[str, Any]]:
        """Extract objects from camera data"""
        objects = []
        for data in camera_data:
            # Extract detected objects from camera
            if 'detections' in data.data:
                for detection in data.data['detections']:
                    objects.append({
                        'type': 'camera',
                        'class': detection.get('class', 'unknown'),
                        'bbox': detection.get('bbox', [0, 0, 0, 0]),
                        'confidence': detection.get('confidence', 0.0),
                        'sensor_id': data.sensor_id,
                        'timestamp': data.timestamp
                    })
        return objects

    def _extract_radar_objects(self, radar_data: List[SensorData]) -> List[Dict[str, Any]]:
        """Extract objects from radar data"""
        objects = []
        for data in radar_data:
            if 'targets' in data.data:
                for target in data.data['targets']:
                    objects.append({
                        'type': 'radar',
                        'range': target.get('range', 0.0),
                        'azimuth': target.get('azimuth', 0.0),
                        'velocity': target.get('velocity', 0.0),
                        'rcs': target.get('rcs', 0.0),
                        'sensor_id': data.sensor_id,
                        'timestamp': data.timestamp
                    })
        return objects

    def _extract_lidar_objects(self, lidar_data: List[SensorData]) -> List[Dict[str, Any]]:
        """Extract objects from lidar data"""
        objects = []
        for data in lidar_data:
            if 'point_cloud' in data.data:
                # Process point cloud for object detection
                # This would typically involve clustering and classification
                objects.append({
                    'type': 'lidar',
                    'points': len(data.data['point_cloud']),
                    'clusters': data.data.get('clusters', []),
                    'sensor_id': data.sensor_id,
                    'timestamp': data.timestamp
                })
        return objects

    def _associate_and_fuse_objects(self, camera_objects: List, radar_objects: List, lidar_objects: List) -> List[Dict[str, Any]]:
        """Associate and fuse objects from multiple sensors"""
        fused_objects = []

        # Simple association based on proximity (would be more sophisticated in production)
        for cam_obj in camera_objects:
            fused_obj = {
                'id': f"obj_{len(fused_objects)}",
                'class': cam_obj['class'],
                'confidence': cam_obj['confidence'],
                'sensors': ['camera'],
                'position': self._estimate_position_from_camera(cam_obj),
                'velocity': [0.0, 0.0],  # Will be updated with radar data
                'timestamp': cam_obj['timestamp']
            }

            # Look for corresponding radar detection
            for radar_obj in radar_objects:
                if self._objects_match(cam_obj, radar_obj):
                    fused_obj['sensors'].append('radar')
                    fused_obj['velocity'] = [radar_obj['velocity'], 0.0]
                    fused_obj['confidence'] = min(1.0, fused_obj['confidence'] + 0.2)

            fused_objects.append(fused_obj)

        return fused_objects

    def _estimate_position_from_camera(self, camera_obj: Dict) -> List[float]:
        """Estimate 3D position from camera detection"""
        # Simplified position estimation
        bbox = camera_obj['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Convert to world coordinates (simplified)
        world_x = (center_x - 960) * 0.1  # Rough conversion
        world_y = 10.0  # Assume 10m ahead

        return [world_x, world_y]

    def _objects_match(self, cam_obj: Dict, radar_obj: Dict) -> bool:
        """Check if camera and radar objects match"""
        # Simplified matching logic
        cam_pos = self._estimate_position_from_camera(cam_obj)
        radar_x = radar_obj['range'] * np.sin(np.radians(radar_obj['azimuth']))
        radar_y = radar_obj['range'] * np.cos(np.radians(radar_obj['azimuth']))

        distance = np.sqrt((cam_pos[0] - radar_x)**2 + (cam_pos[1] - radar_y)**2)
        return distance < 2.0  # 2m threshold

    def _environment_fusion(self, aligned_data: Dict[SensorType, List[SensorData]]) -> Dict[str, Any]:
        """Fuse environmental information"""
        environment = {
            'weather': 'clear',
            'visibility': 1.0,
            'road_condition': 'dry',
            'lighting': 'daylight',
            'timestamp': time.time()
        }

        # Extract from camera data
        camera_data = aligned_data.get(SensorType.CAMERA, [])
        if camera_data:
            # Analyze image for environmental conditions
            environment['visibility'] = self._estimate_visibility(camera_data[0])

        return environment

    def _vehicle_state_fusion(self, aligned_data: Dict[SensorType, List[SensorData]]) -> Dict[str, Any]:
        """Fuse vehicle state information"""
        vehicle_state = {
            'position': [0.0, 0.0, 0.0],
            'orientation': [0.0, 0.0, 0.0],
            'velocity': [0.0, 0.0, 0.0],
            'acceleration': [0.0, 0.0, 0.0],
            'timestamp': time.time()
        }

        # Extract from GPS data
        gps_data = aligned_data.get(SensorType.GPS, [])
        if gps_data:
            gps = gps_data[0].data
            vehicle_state['position'] = [
                gps.get('latitude', 0.0),
                gps.get('longitude', 0.0),
                gps.get('altitude', 0.0)
            ]

        # Extract from IMU data
        imu_data = aligned_data.get(SensorType.IMU, [])
        if imu_data:
            imu = imu_data[0].data
            vehicle_state['acceleration'] = [
                imu.get('accel_x', 0.0),
                imu.get('accel_y', 0.0),
                imu.get('accel_z', 0.0)
            ]
            vehicle_state['orientation'] = [
                imu.get('roll', 0.0),
                imu.get('pitch', 0.0),
                imu.get('yaw', 0.0)
            ]

        return vehicle_state

    def _estimate_visibility(self, camera_data: SensorData) -> float:
        """Estimate visibility from camera data"""
        # Simplified visibility estimation
        if 'image_stats' in camera_data.data:
            brightness = camera_data.data['image_stats'].get('mean_brightness', 128)
            contrast = camera_data.data['image_stats'].get('contrast', 50)

            # Higher brightness and contrast = better visibility
            visibility = min(1.0, (brightness / 255.0) * (contrast / 100.0))
            return max(0.1, visibility)

        return 0.8  # Default good visibility

    def _calculate_confidence(self, sensor_groups: Dict[SensorType, List[SensorData]]) -> Dict[str, float]:
        """Calculate confidence metrics for fusion output"""
        confidence_map = {}

        for sensor_type, data_list in sensor_groups.items():
            if data_list:
                avg_confidence = np.mean([data.confidence for data in data_list])
                avg_health = np.mean([self.sensor_health.get(data.sensor_id, 1.0) for data in data_list])

                confidence_map[sensor_type.value] = avg_confidence * avg_health
            else:
                confidence_map[sensor_type.value] = 0.0

        # Overall fusion confidence
        available_sensors = sum(1 for conf in confidence_map.values() if conf > 0.5)
        confidence_map['fusion_overall'] = min(1.0, available_sensors / 3.0)  # Expect at least 3 sensors

        return confidence_map

    def _assess_fusion_quality(self, objects: List, environment: Dict, vehicle_state: Dict) -> Dict[str, float]:
        """Assess quality of fusion output"""
        quality_metrics = {
            'object_detection_quality': 0.0,
            'environment_quality': 0.0,
            'localization_quality': 0.0,
            'temporal_consistency': 0.0,
            'overall_quality': 0.0
        }

        # Object detection quality
        if objects:
            avg_confidence = np.mean([obj.get('confidence', 0.0) for obj in objects])
            sensor_diversity = np.mean([len(obj.get('sensors', [])) for obj in objects])
            quality_metrics['object_detection_quality'] = avg_confidence * (sensor_diversity / 3.0)

        # Environment quality
        if environment.get('visibility', 0) > 0.5:
            quality_metrics['environment_quality'] = 0.9
        else:
            quality_metrics['environment_quality'] = 0.5

        # Localization quality
        if vehicle_state.get('position', [0, 0, 0])[0] != 0 or vehicle_state.get('position', [0, 0, 0])[1] != 0:
            quality_metrics['localization_quality'] = 0.9
        else:
            quality_metrics['localization_quality'] = 0.3

        # Temporal consistency (simplified)
        quality_metrics['temporal_consistency'] = 0.8

        # Overall quality
        quality_metrics['overall_quality'] = np.mean([
            quality_metrics['object_detection_quality'],
            quality_metrics['environment_quality'],
            quality_metrics['localization_quality'],
            quality_metrics['temporal_consistency']
        ])

        return quality_metrics

    async def process_sensor_data(self, sensor_data: SensorData) -> bool:
        """Process incoming sensor data"""
        try:
            # Validate sensor data
            if not self._validate_sensor_data(sensor_data):
                return False

            # Update sensor health
            self._update_sensor_health(sensor_data)

            # Add to fusion queue with priority
            priority = self._calculate_priority(sensor_data)

            try:
                self.fusion_queue.put_nowait((priority, [sensor_data]))
                return True
            except queue.Full:
                self.performance_metrics['data_drops'] += 1
                self.logger.warning(f"Fusion queue full, dropping data from {sensor_data.sensor_id}")
                return False

        except Exception as e:
            self.logger.error(f"Error processing sensor data: {e}")
            return False

    def _validate_sensor_data(self, sensor_data: SensorData) -> bool:
        """Validate incoming sensor data"""
        # Check timestamp
        age = time.time() - sensor_data.timestamp
        if age > self.safety_thresholds['max_data_age']:
            self.logger.warning(f"Sensor data too old: {age*1000:.1f}ms")
            return False

        # Check confidence
        if sensor_data.confidence < 0.1:
            self.logger.warning(f"Sensor confidence too low: {sensor_data.confidence}")
            return False

        # Check health status
        if sensor_data.health_status != 'healthy':
            self.logger.warning(f"Sensor health issue: {sensor_data.health_status}")
            return False

        return True

    def _update_sensor_health(self, sensor_data: SensorData):
        """Update sensor health tracking"""
        sensor_id = sensor_data.sensor_id

        # Health based on data quality and latency
        health_score = sensor_data.confidence * 0.7

        if sensor_data.processing_latency < 0.005:  # 5ms
            health_score += 0.3
        elif sensor_data.processing_latency < 0.010:  # 10ms
            health_score += 0.2
        else:
            health_score += 0.1

        self.sensor_health[sensor_id] = health_score

    def _calculate_priority(self, sensor_data: SensorData) -> int:
        """Calculate processing priority for sensor data"""
        # Higher priority = lower number
        priority = 10

        # Camera and lidar get higher priority
        if sensor_data.sensor_type in [SensorType.CAMERA, SensorType.LIDAR]:
            priority -= 3
        elif sensor_data.sensor_type == SensorType.RADAR:
            priority -= 2

        # Higher confidence gets higher priority
        if sensor_data.confidence > 0.8:
            priority -= 2
        elif sensor_data.confidence > 0.6:
            priority -= 1

        return max(1, priority)

    async def get_fused_output(self, timeout: float = 0.001) -> Optional[FusedOutput]:
        """Get latest fused output"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = self.performance_metrics.copy()

        if metrics['fusion_latency']:
            metrics['avg_fusion_latency'] = np.mean(metrics['fusion_latency'])
            metrics['max_fusion_latency'] = np.max(metrics['fusion_latency'])
            metrics['latency_violations'] = sum(1 for lat in metrics['fusion_latency']
                                              if lat > self.max_processing_time)

        metrics['active_sensors'] = len(self.active_sensors)
        metrics['sensor_health_avg'] = np.mean(list(self.sensor_health.values())) if self.sensor_health else 0.0

        return metrics

    async def shutdown(self):
        """Shutdown sensor fusion agent"""
        self.logger.info("Shutting down Sensor Fusion Agent")

        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)

        self.executor.shutdown(wait=True)

        self.logger.info("Sensor Fusion Agent shutdown complete")

# Example usage for testing
if __name__ == "__main__":
    async def test_sensor_fusion():
        agent = SensorFusionAgent()

        if await agent.initialize():
            print("Sensor Fusion Agent initialized successfully")

            # Simulate sensor data
            camera_data = SensorData(
                sensor_id="cam_front",
                sensor_type=SensorType.CAMERA,
                timestamp=time.time(),
                data={
                    'detections': [
                        {'class': 'car', 'bbox': [100, 200, 300, 400], 'confidence': 0.95}
                    ],
                    'image_stats': {'mean_brightness': 180, 'contrast': 70}
                },
                confidence=0.95,
                health_status='healthy'
            )

            await agent.process_sensor_data(camera_data)

            # Wait for processing
            await asyncio.sleep(0.1)

            # Get fused output
            output = await agent.get_fused_output()
            if output:
                print(f"Fused output: {len(output.objects)} objects detected")
                print(f"Fusion latency: {output.fusion_latency*1000:.2f}ms")

            metrics = await agent.get_performance_metrics()
            print(f"Performance metrics: {metrics}")

            await agent.shutdown()

    asyncio.run(test_sensor_fusion())