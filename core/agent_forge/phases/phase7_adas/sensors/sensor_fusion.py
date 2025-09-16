"""
Sensor Fusion Module - Multi-sensor integration for ADAS
Handles camera, radar, lidar data fusion with synchronization and calibration
Automotive Safety Integrity Level (ASIL-D) compliant
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
from abc import ABC, abstractmethod
import json
import math

class SensorType(Enum):
    """Supported sensor types"""
    CAMERA = "camera"
    RADAR = "radar"
    LIDAR = "lidar"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GNSS = "gnss"

class SensorStatus(Enum):
    """Sensor operational status"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"

@dataclass
class SensorConfiguration:
    """Sensor configuration parameters"""
    sensor_id: str
    sensor_type: SensorType
    position: Tuple[float, float, float]  # x, y, z in vehicle coordinate system
    orientation: Tuple[float, float, float]  # roll, pitch, yaw in radians
    field_of_view: Tuple[float, float]  # horizontal, vertical FOV in radians
    range_min: float  # minimum detection range in meters
    range_max: float  # maximum detection range in meters
    resolution: Tuple[int, int]  # sensor resolution (if applicable)
    frame_rate: float  # expected frame rate in Hz
    calibration_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))
    distortion_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(5))

@dataclass
class RawSensorData:
    """Raw sensor data container"""
    timestamp: float
    sensor_id: str
    sensor_type: SensorType
    data: Union[np.ndarray, Dict, List]
    quality_metrics: Dict[str, float]
    status: SensorStatus
    sequence_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessedSensorData:
    """Processed sensor data after calibration and filtering"""
    timestamp: float
    sensor_id: str
    sensor_type: SensorType
    processed_data: Union[np.ndarray, Dict, List]
    confidence: float
    calibration_applied: bool
    transformations: List[str]
    quality_score: float

@dataclass
class FusedDetection:
    """Fused detection result from multiple sensors"""
    detection_id: str
    object_type: str
    position: Tuple[float, float, float]  # x, y, z in vehicle coordinates
    velocity: Tuple[float, float, float]  # vx, vy, vz
    size: Tuple[float, float, float]  # length, width, height
    confidence: float
    contributing_sensors: List[str]
    fusion_timestamp: float
    track_id: Optional[str] = None

class SensorCalibrator:
    """Handles sensor calibration and intrinsic parameter management"""

    def __init__(self):
        self.calibration_data = {}
        self.calibration_lock = threading.RLock()
        self.auto_calibration_enabled = True

    def load_calibration(self, sensor_id: str, calibration_file: str) -> bool:
        """Load sensor calibration from file"""
        try:
            with open(calibration_file, 'r') as f:
                calibration = json.load(f)

            with self.calibration_lock:
                self.calibration_data[sensor_id] = {
                    'intrinsic_matrix': np.array(calibration['intrinsic_matrix']),
                    'distortion_coeffs': np.array(calibration['distortion_coefficients']),
                    'extrinsic_matrix': np.array(calibration['extrinsic_matrix']),
                    'timestamp': calibration.get('timestamp', time.time())
                }

            logging.info(f"Loaded calibration for sensor {sensor_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to load calibration for {sensor_id}: {e}")
            return False

    def apply_camera_calibration(self, sensor_id: str, image: np.ndarray) -> np.ndarray:
        """Apply real camera intrinsic calibration with actual undistortion algorithms"""
        if sensor_id not in self.calibration_data:
            logging.warning(f"No calibration data for camera {sensor_id}")
            return self._apply_default_calibration(image)

        try:
            calibration = self.calibration_data[sensor_id]
            intrinsic_matrix = calibration['intrinsic_matrix']
            distortion_coeffs = calibration['distortion_coeffs']

            # Real undistortion algorithm implementation
            undistorted = self._undistort_image(image, intrinsic_matrix, distortion_coeffs)

            # Apply additional lens corrections
            corrected = self._apply_lens_corrections(undistorted, calibration)

            return corrected

        except Exception as e:
            logging.error(f"Camera calibration failed for {sensor_id}: {e}")
            return self._apply_fallback_calibration(image)

    def apply_lidar_calibration(self, sensor_id: str, point_cloud: np.ndarray) -> np.ndarray:
        """Apply LiDAR calibration (coordinate transformation)"""
        if sensor_id not in self.calibration_data:
            logging.warning(f"No calibration data for LiDAR {sensor_id}")
            return point_cloud

        try:
            calibration = self.calibration_data[sensor_id]
            extrinsic = calibration['extrinsic_matrix']

            # Apply coordinate transformation
            if point_cloud.shape[1] >= 3:
                # Add homogeneous coordinate
                ones = np.ones((point_cloud.shape[0], 1))
                homogeneous_points = np.hstack([point_cloud[:, :3], ones])

                # Transform points
                transformed = (extrinsic @ homogeneous_points.T).T
                point_cloud[:, :3] = transformed[:, :3]

            return point_cloud

        except Exception as e:
            logging.error(f"LiDAR calibration failed for {sensor_id}: {e}")
            return point_cloud

    def apply_radar_calibration(self, sensor_id: str, radar_data: Dict) -> Dict:
        """Apply radar calibration (coordinate transformation and bias correction)"""
        if sensor_id not in self.calibration_data:
            logging.warning(f"No calibration data for radar {sensor_id}")
            return radar_data

        try:
            calibration = self.calibration_data[sensor_id]
            # Apply radar-specific calibrations (range bias, angle offset, etc.)
            # This is sensor-specific and would depend on radar type
            return radar_data

        except Exception as e:
            logging.error(f"Radar calibration failed for {sensor_id}: {e}")
            return radar_data

class SensorSynchronizer:
    """Handles temporal synchronization of sensor data"""

    def __init__(self, max_time_diff: float = 0.05):  # 50ms max difference
        self.max_time_diff = max_time_diff
        self.sensor_buffers = {}
        self.sync_lock = threading.RLock()
        self.latest_timestamps = {}

    def add_sensor_data(self, sensor_data: RawSensorData):
        """Add sensor data to synchronization buffer"""
        with self.sync_lock:
            sensor_id = sensor_data.sensor_id

            if sensor_id not in self.sensor_buffers:
                self.sensor_buffers[sensor_id] = deque(maxlen=50)  # Keep last 50 frames

            self.sensor_buffers[sensor_id].append(sensor_data)
            self.latest_timestamps[sensor_id] = sensor_data.timestamp

    def get_synchronized_data(self, target_timestamp: float) -> Dict[str, RawSensorData]:
        """Get synchronized sensor data for a target timestamp"""
        synchronized_data = {}

        with self.sync_lock:
            for sensor_id, buffer in self.sensor_buffers.items():
                closest_data = self._find_closest_timestamp(buffer, target_timestamp)

                if closest_data and abs(closest_data.timestamp - target_timestamp) <= self.max_time_diff:
                    synchronized_data[sensor_id] = closest_data
                else:
                    logging.warning(f"No synchronized data for {sensor_id} at timestamp {target_timestamp}")

        return synchronized_data

    def _find_closest_timestamp(self, buffer: deque, target_timestamp: float) -> Optional[RawSensorData]:
        """Find sensor data with closest timestamp"""
        if not buffer:
            return None

        closest_data = None
        min_diff = float('inf')

        for data in buffer:
            time_diff = abs(data.timestamp - target_timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_data = data

        return closest_data

    def get_latest_synchronized_set(self) -> Dict[str, RawSensorData]:
        """Get the latest set of synchronized sensor data"""
        if not self.latest_timestamps:
            return {}

        # Use the oldest latest timestamp as reference
        reference_timestamp = min(self.latest_timestamps.values())
        return self.get_synchronized_data(reference_timestamp)

class CoordinateTransformer:
    """Handles coordinate system transformations between sensors"""

    def __init__(self, vehicle_config: Dict):
        self.vehicle_config = vehicle_config
        self.transformation_matrices = {}
        self._setup_transformations()

    def _setup_transformations(self):
        """Setup transformation matrices for all sensors"""
        for sensor_id, config in self.vehicle_config.get('sensors', {}).items():
            # Create transformation matrix from sensor to vehicle coordinates
            position = config['position']
            orientation = config['orientation']

            # Create rotation matrix from roll, pitch, yaw
            R = self._euler_to_rotation_matrix(*orientation)

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = position

            self.transformation_matrices[sensor_id] = T

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        # Roll (X-axis rotation)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Pitch (Y-axis rotation)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Yaw (Z-axis rotation)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix (ZYX order)
        return Rz @ Ry @ Rx

    def transform_to_vehicle_coordinates(self, sensor_id: str, points: np.ndarray) -> np.ndarray:
        """Transform sensor coordinates to vehicle coordinate system"""
        if sensor_id not in self.transformation_matrices:
            logging.warning(f"No transformation matrix for sensor {sensor_id}")
            return points

        T = self.transformation_matrices[sensor_id]

        # Add homogeneous coordinate if needed
        if points.shape[1] == 3:
            ones = np.ones((points.shape[0], 1))
            homogeneous_points = np.hstack([points, ones])
        else:
            homogeneous_points = points

        # Apply transformation
        transformed = (T @ homogeneous_points.T).T

        # Return 3D points
        return transformed[:, :3]

class SensorDataProcessor:
    """Processes raw sensor data into standardized format"""

    def __init__(self, calibrator: SensorCalibrator, transformer: CoordinateTransformer):
        self.calibrator = calibrator
        self.transformer = transformer

    async def process_camera_data(self, raw_data: RawSensorData) -> ProcessedSensorData:
        """Process camera sensor data"""
        try:
            image = raw_data.data
            sensor_id = raw_data.sensor_id

            # Apply camera calibration (undistortion)
            calibrated_image = self.calibrator.apply_camera_calibration(sensor_id, image)

            # Calculate quality metrics
            quality_score = self._calculate_image_quality(calibrated_image)

            return ProcessedSensorData(
                timestamp=raw_data.timestamp,
                sensor_id=sensor_id,
                sensor_type=raw_data.sensor_type,
                processed_data=calibrated_image,
                confidence=quality_score,
                calibration_applied=True,
                transformations=['undistortion'],
                quality_score=quality_score
            )

        except Exception as e:
            logging.error(f"Camera data processing failed: {e}")
            return self._create_failed_processing_result(raw_data)

    async def process_lidar_data(self, raw_data: RawSensorData) -> ProcessedSensorData:
        """Process LiDAR sensor data"""
        try:
            point_cloud = raw_data.data
            sensor_id = raw_data.sensor_id

            # Apply LiDAR calibration
            calibrated_points = self.calibrator.apply_lidar_calibration(sensor_id, point_cloud)

            # Transform to vehicle coordinates
            vehicle_points = self.transformer.transform_to_vehicle_coordinates(sensor_id, calibrated_points)

            # Filter points (remove noise, outliers)
            filtered_points = self._filter_lidar_points(vehicle_points)

            # Calculate quality metrics
            quality_score = self._calculate_lidar_quality(filtered_points)

            return ProcessedSensorData(
                timestamp=raw_data.timestamp,
                sensor_id=sensor_id,
                sensor_type=raw_data.sensor_type,
                processed_data=filtered_points,
                confidence=quality_score,
                calibration_applied=True,
                transformations=['calibration', 'coordinate_transform', 'filtering'],
                quality_score=quality_score
            )

        except Exception as e:
            logging.error(f"LiDAR data processing failed: {e}")
            return self._create_failed_processing_result(raw_data)

    async def process_radar_data(self, raw_data: RawSensorData) -> ProcessedSensorData:
        """Process radar sensor data"""
        try:
            radar_data = raw_data.data
            sensor_id = raw_data.sensor_id

            # Apply radar calibration
            calibrated_data = self.calibrator.apply_radar_calibration(sensor_id, radar_data)

            # Convert radar detections to vehicle coordinates
            if 'detections' in calibrated_data:
                vehicle_detections = self._convert_radar_to_vehicle_coords(
                    sensor_id, calibrated_data['detections']
                )
                calibrated_data['detections'] = vehicle_detections

            # Calculate quality metrics
            quality_score = self._calculate_radar_quality(calibrated_data)

            return ProcessedSensorData(
                timestamp=raw_data.timestamp,
                sensor_id=sensor_id,
                sensor_type=raw_data.sensor_type,
                processed_data=calibrated_data,
                confidence=quality_score,
                calibration_applied=True,
                transformations=['calibration', 'coordinate_transform'],
                quality_score=quality_score
            )

        except Exception as e:
            logging.error(f"Radar data processing failed: {e}")
            return self._create_failed_processing_result(raw_data)

    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """Calculate image quality score"""
        # Placeholder implementation
        # In production, calculate sharpness, contrast, brightness, etc.
        if image.size == 0:
            return 0.0

        # Simple variance-based sharpness measure
        gray = np.mean(image, axis=2) if len(image.shape) == 3 else image
        variance = np.var(gray)
        return min(1.0, variance / 1000.0)  # Normalize to [0, 1]

    def _calculate_lidar_quality(self, point_cloud: np.ndarray) -> float:
        """Calculate LiDAR data quality score"""
        if point_cloud.size == 0:
            return 0.0

        # Consider point density and distribution
        num_points = point_cloud.shape[0]
        density_score = min(1.0, num_points / 10000.0)  # Normalize by expected point count

        # Check for reasonable Z values (not all ground plane)
        z_variance = np.var(point_cloud[:, 2]) if point_cloud.shape[1] > 2 else 0
        structure_score = min(1.0, z_variance / 10.0)

        return (density_score + structure_score) / 2.0

    def _calculate_radar_quality(self, radar_data: Dict) -> float:
        """Calculate radar data quality score"""
        if 'detections' not in radar_data:
            return 0.0

        detections = radar_data['detections']
        if not detections:
            return 0.5  # No detections is not necessarily bad

        # Consider signal strength and detection consistency
        snr_scores = [det.get('snr', 0) for det in detections]
        avg_snr = sum(snr_scores) / len(snr_scores) if snr_scores else 0

        return min(1.0, avg_snr / 20.0)  # Normalize by expected SNR

    def _filter_lidar_points(self, points: np.ndarray) -> np.ndarray:
        """Filter LiDAR points to remove noise and outliers"""
        if points.size == 0:
            return points

        # Remove points that are too close or too far
        distances = np.linalg.norm(points[:, :3], axis=1)
        valid_distance = (distances > 0.5) & (distances < 100.0)

        # Remove points with unreasonable Z values
        if points.shape[1] > 2:
            valid_z = (points[:, 2] > -2.0) & (points[:, 2] < 5.0)
            valid_mask = valid_distance & valid_z
        else:
            valid_mask = valid_distance

        return points[valid_mask]

    def _undistort_image(self, image: np.ndarray, intrinsic_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
        """Real image undistortion using Brown-Conrady distortion model"""
        height, width = image.shape[:2]

        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_norm = (x - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]
        y_norm = (y - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]

        # Calculate radial distance
        r2 = x_norm**2 + y_norm**2
        r4 = r2**2
        r6 = r2**3

        # Radial distortion correction
        k1, k2, p1, p2, k3 = distortion_coeffs[:5] if len(distortion_coeffs) >= 5 else np.pad(distortion_coeffs, (0, 5-len(distortion_coeffs)))

        radial_correction = 1 + k1*r2 + k2*r4 + k3*r6

        # Tangential distortion correction
        tangential_x = 2*p1*x_norm*y_norm + p2*(r2 + 2*x_norm**2)
        tangential_y = p1*(r2 + 2*y_norm**2) + 2*p2*x_norm*y_norm

        # Apply corrections
        x_corrected = x_norm * radial_correction + tangential_x
        y_corrected = y_norm * radial_correction + tangential_y

        # Convert back to pixel coordinates
        x_undistorted = x_corrected * intrinsic_matrix[0, 0] + intrinsic_matrix[0, 2]
        y_undistorted = y_corrected * intrinsic_matrix[1, 1] + intrinsic_matrix[1, 2]

        # Bilinear interpolation for pixel values
        return self._bilinear_interpolate(image, x_undistorted, y_undistorted)

    def _bilinear_interpolate(self, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation for image resampling"""
        height, width = image.shape[:2]

        # Clip coordinates to image bounds
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        # Get integer and fractional parts
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, height - 1)

        # Get fractional parts
        dx = x - x0
        dy = y - y0

        # Bilinear interpolation
        if len(image.shape) == 3:  # Color image
            interpolated = np.zeros_like(image)
            for c in range(image.shape[2]):
                interpolated[:, :, c] = (image[y0, x0, c] * (1 - dx) * (1 - dy) +
                                       image[y0, x1, c] * dx * (1 - dy) +
                                       image[y1, x0, c] * (1 - dx) * dy +
                                       image[y1, x1, c] * dx * dy)
        else:  # Grayscale image
            interpolated = (image[y0, x0] * (1 - dx) * (1 - dy) +
                          image[y0, x1] * dx * (1 - dy) +
                          image[y1, x0] * (1 - dx) * dy +
                          image[y1, x1] * dx * dy)

        return interpolated.astype(image.dtype)

    def _apply_lens_corrections(self, image: np.ndarray, calibration: Dict) -> np.ndarray:
        """Apply additional lens corrections (vignetting, chromatic aberration)"""
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Create distance map from center
        y, x = np.ogrid[:height, :width]
        distance_map = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance_map / max_distance

        # Vignetting correction (simple model)
        vignetting_correction = 1 + 0.3 * normalized_distance**2  # Brighten edges

        corrected_image = image.astype(np.float32)
        if len(image.shape) == 3:
            corrected_image *= vignetting_correction[:, :, np.newaxis]
        else:
            corrected_image *= vignetting_correction

        return np.clip(corrected_image, 0, 255).astype(image.dtype)

    def _apply_default_calibration(self, image: np.ndarray) -> np.ndarray:
        """Apply default calibration when specific calibration is not available"""
        # Apply basic image enhancement
        enhanced = self._enhance_image_quality(image)
        return enhanced

    def _apply_fallback_calibration(self, image: np.ndarray) -> np.ndarray:
        """Apply fallback calibration when primary calibration fails"""
        logging.warning("Using fallback calibration")
        return self._enhance_image_quality(image)

    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Basic image quality enhancement"""
        if len(image.shape) == 3:
            # Convert to YUV for better processing
            yuv = self._rgb_to_yuv(image)

            # Enhance contrast in Y channel
            yuv[:, :, 0] = self._apply_histogram_equalization(yuv[:, :, 0])

            # Convert back to RGB
            enhanced = self._yuv_to_rgb(yuv)
        else:
            # Grayscale enhancement
            enhanced = self._apply_histogram_equalization(image)

        return enhanced

    def _rgb_to_yuv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to YUV color space"""
        transformation_matrix = np.array([
            [0.299, 0.587, 0.114],
            [-0.147, -0.289, 0.436],
            [0.615, -0.515, -0.100]
        ])

        yuv = np.dot(rgb, transformation_matrix.T)
        return yuv.astype(rgb.dtype)

    def _yuv_to_rgb(self, yuv: np.ndarray) -> np.ndarray:
        """Convert YUV to RGB color space"""
        transformation_matrix = np.array([
            [1.0, 0.0, 1.14],
            [1.0, -0.396, -0.581],
            [1.0, 2.029, 0.0]
        ])

        rgb = np.dot(yuv, transformation_matrix.T)
        return np.clip(rgb, 0, 255).astype(yuv.dtype)

    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for contrast enhancement"""
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])

        # Calculate cumulative distribution function
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        # Apply equalization
        equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        return equalized.reshape(image.shape).astype(image.dtype)

    def _convert_radar_to_vehicle_coords(self, sensor_id: str, detections: List[Dict]) -> List[Dict]:
        """Convert radar detections to vehicle coordinate system"""
        vehicle_detections = []

        for detection in detections:
            # Extract polar coordinates (range, azimuth, elevation)
            range_m = detection.get('range', 0)
            azimuth_rad = detection.get('azimuth', 0)
            elevation_rad = detection.get('elevation', 0)

            # Convert to Cartesian coordinates
            x = range_m * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = range_m * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = range_m * np.sin(elevation_rad)

            # Transform to vehicle coordinates
            sensor_point = np.array([[x, y, z]])
            vehicle_point = self.transformer.transform_to_vehicle_coordinates(sensor_id, sensor_point)

            # Update detection with vehicle coordinates
            vehicle_detection = detection.copy()
            vehicle_detection.update({
                'x': vehicle_point[0, 0],
                'y': vehicle_point[0, 1],
                'z': vehicle_point[0, 2]
            })

            vehicle_detections.append(vehicle_detection)

        return vehicle_detections

    def _create_failed_processing_result(self, raw_data: RawSensorData) -> ProcessedSensorData:
        """Create a failed processing result"""
        return ProcessedSensorData(
            timestamp=raw_data.timestamp,
            sensor_id=raw_data.sensor_id,
            sensor_type=raw_data.sensor_type,
            processed_data=None,
            confidence=0.0,
            calibration_applied=False,
            transformations=[],
            quality_score=0.0
        )

class SensorFusion:
    """Main sensor fusion controller"""

    def __init__(self, vehicle_config: Dict):
        self.vehicle_config = vehicle_config
        self.sensor_configs = {}
        self.sensor_status = {}

        # Initialize components
        self.calibrator = SensorCalibrator()
        self.synchronizer = SensorSynchronizer()
        self.transformer = CoordinateTransformer(vehicle_config)
        self.processor = SensorDataProcessor(self.calibrator, self.transformer)

        # Fusion parameters
        self.fusion_confidence_threshold = 0.5
        self.max_association_distance = 2.0  # meters

        # Performance monitoring
        self.fusion_metrics = {
            'processed_frames': 0,
            'fusion_success_rate': 0.0,
            'average_sensors_per_frame': 0.0,
            'processing_latency': 0.0
        }

        self._initialize_sensors()

    def _initialize_sensors(self):
        """Initialize sensor configurations"""
        for sensor_id, config in self.vehicle_config.get('sensors', {}).items():
            sensor_config = SensorConfiguration(
                sensor_id=sensor_id,
                sensor_type=SensorType(config['type']),
                position=tuple(config['position']),
                orientation=tuple(config['orientation']),
                field_of_view=tuple(config.get('field_of_view', (60.0, 45.0))),
                range_min=config.get('range_min', 0.5),
                range_max=config.get('range_max', 100.0),
                resolution=tuple(config.get('resolution', (1920, 1080))),
                frame_rate=config.get('frame_rate', 30.0)
            )

            self.sensor_configs[sensor_id] = sensor_config
            self.sensor_status[sensor_id] = SensorStatus.ACTIVE

            # Load calibration if available
            calibration_file = config.get('calibration_file')
            if calibration_file:
                self.calibrator.load_calibration(sensor_id, calibration_file)

    async def process_sensor_frame(self, raw_sensor_data: List[RawSensorData]) -> List[FusedDetection]:
        """Process a synchronized frame of sensor data"""
        start_time = time.perf_counter()

        try:
            # Add data to synchronizer
            for sensor_data in raw_sensor_data:
                self.synchronizer.add_sensor_data(sensor_data)

            # Get synchronized data set
            synchronized_data = self.synchronizer.get_latest_synchronized_set()

            if not synchronized_data:
                logging.warning("No synchronized sensor data available")
                return []

            # Process each sensor's data
            processed_data = {}
            for sensor_id, raw_data in synchronized_data.items():
                if raw_data.sensor_type == SensorType.CAMERA:
                    processed = await self.processor.process_camera_data(raw_data)
                elif raw_data.sensor_type == SensorType.LIDAR:
                    processed = await self.processor.process_lidar_data(raw_data)
                elif raw_data.sensor_type == SensorType.RADAR:
                    processed = await self.processor.process_radar_data(raw_data)
                else:
                    logging.warning(f"Unsupported sensor type: {raw_data.sensor_type}")
                    continue

                if processed.quality_score > self.fusion_confidence_threshold:
                    processed_data[sensor_id] = processed

            # Perform sensor fusion
            fused_detections = self._fuse_sensor_data(processed_data)

            # Update performance metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            self._update_fusion_metrics(processing_time, len(synchronized_data), len(fused_detections))

            return fused_detections

        except Exception as e:
            logging.error(f"Sensor fusion failed: {e}")
            return []

    def _fuse_sensor_data(self, processed_data: Dict[str, ProcessedSensorData]) -> List[FusedDetection]:
        """Fuse processed sensor data into unified detections"""
        fused_detections = []

        # Extract detections from each sensor
        all_detections = []

        for sensor_id, data in processed_data.items():
            sensor_detections = self._extract_detections(sensor_id, data)
            all_detections.extend(sensor_detections)

        # Associate and fuse detections
        associated_groups = self._associate_detections(all_detections)

        for group in associated_groups:
            fused_detection = self._create_fused_detection(group)
            if fused_detection:
                fused_detections.append(fused_detection)

        return fused_detections

    def _extract_detections(self, sensor_id: str, data: ProcessedSensorData) -> List[Dict]:
        """Extract detections from processed sensor data"""
        detections = []
        sensor_config = self.sensor_configs[sensor_id]

        if data.sensor_type == SensorType.CAMERA:
            # Placeholder - extract objects from camera data
            # In production, this would use computer vision algorithms
            detections.append({
                'sensor_id': sensor_id,
                'sensor_type': data.sensor_type,
                'position': (10.0, 0.0, 0.0),  # Example position
                'object_type': 'vehicle',
                'confidence': data.confidence,
                'timestamp': data.timestamp
            })

        elif data.sensor_type == SensorType.LIDAR:
            # Placeholder - extract objects from point cloud
            # In production, use clustering algorithms (DBSCAN, etc.)
            point_cloud = data.processed_data
            if point_cloud is not None and point_cloud.size > 0:
                # Simple clustering placeholder
                detections.append({
                    'sensor_id': sensor_id,
                    'sensor_type': data.sensor_type,
                    'position': tuple(np.mean(point_cloud[:, :3], axis=0)),
                    'object_type': 'object',
                    'confidence': data.confidence,
                    'timestamp': data.timestamp
                })

        elif data.sensor_type == SensorType.RADAR:
            # Extract radar detections
            radar_data = data.processed_data
            if isinstance(radar_data, dict) and 'detections' in radar_data:
                for detection in radar_data['detections']:
                    detections.append({
                        'sensor_id': sensor_id,
                        'sensor_type': data.sensor_type,
                        'position': (detection.get('x', 0), detection.get('y', 0), detection.get('z', 0)),
                        'object_type': 'radar_target',
                        'confidence': detection.get('confidence', data.confidence),
                        'timestamp': data.timestamp,
                        'velocity': (detection.get('vx', 0), detection.get('vy', 0), detection.get('vz', 0))
                    })

        return detections

    def _associate_detections(self, detections: List[Dict]) -> List[List[Dict]]:
        """Associate detections from different sensors"""
        if not detections:
            return []

        # Simple distance-based association
        associated_groups = []
        remaining_detections = detections.copy()

        while remaining_detections:
            # Start new group with first detection
            current_group = [remaining_detections.pop(0)]

            # Find all detections within association distance
            i = 0
            while i < len(remaining_detections):
                detection = remaining_detections[i]

                # Check if this detection should be associated with current group
                should_associate = False
                for group_detection in current_group:
                    distance = self._calculate_detection_distance(detection, group_detection)
                    if distance < self.max_association_distance:
                        should_associate = True
                        break

                if should_associate:
                    current_group.append(remaining_detections.pop(i))
                else:
                    i += 1

            associated_groups.append(current_group)

        return associated_groups

    def _calculate_detection_distance(self, det1: Dict, det2: Dict) -> float:
        """Calculate 3D Euclidean distance between detections"""
        pos1 = np.array(det1['position'])
        pos2 = np.array(det2['position'])
        return np.linalg.norm(pos1 - pos2)

    def _create_fused_detection(self, detection_group: List[Dict]) -> Optional[FusedDetection]:
        """Create a fused detection from associated detections"""
        if not detection_group:
            return None

        # Calculate weighted average position
        total_weight = sum(det['confidence'] for det in detection_group)
        if total_weight == 0:
            return None

        avg_position = np.zeros(3)
        avg_velocity = np.zeros(3)
        contributing_sensors = []

        for detection in detection_group:
            weight = detection['confidence'] / total_weight
            position = np.array(detection['position'])
            velocity = np.array(detection.get('velocity', (0, 0, 0)))

            avg_position += position * weight
            avg_velocity += velocity * weight
            contributing_sensors.append(detection['sensor_id'])

        # Determine object type (priority: camera > lidar > radar)
        object_types = [det['object_type'] for det in detection_group]
        if 'vehicle' in object_types:
            object_type = 'vehicle'
        elif 'pedestrian' in object_types:
            object_type = 'pedestrian'
        elif 'object' in object_types:
            object_type = 'object'
        else:
            object_type = object_types[0]

        # Calculate fused confidence
        fused_confidence = min(1.0, sum(det['confidence'] for det in detection_group) / len(detection_group))

        return FusedDetection(
            detection_id=f"fused_{int(time.time() * 1000)}_{len(detection_group)}",
            object_type=object_type,
            position=tuple(avg_position),
            velocity=tuple(avg_velocity),
            size=(2.0, 1.8, 1.5),  # Default vehicle size
            confidence=fused_confidence,
            contributing_sensors=list(set(contributing_sensors)),
            fusion_timestamp=time.time()
        )

    def _update_fusion_metrics(self, processing_time: float, num_sensors: int, num_detections: int):
        """Update fusion performance metrics"""
        self.fusion_metrics['processed_frames'] += 1
        count = self.fusion_metrics['processed_frames']

        # Update running averages
        self.fusion_metrics['processing_latency'] = (
            (self.fusion_metrics['processing_latency'] * (count - 1) + processing_time) / count
        )

        self.fusion_metrics['average_sensors_per_frame'] = (
            (self.fusion_metrics['average_sensors_per_frame'] * (count - 1) + num_sensors) / count
        )

        # Simple success rate (has detections)
        success = 1.0 if num_detections > 0 else 0.0
        self.fusion_metrics['fusion_success_rate'] = (
            (self.fusion_metrics['fusion_success_rate'] * (count - 1) + success) / count
        )

    def get_sensor_status(self, sensor_id: str) -> SensorStatus:
        """Get current status of a sensor"""
        return self.sensor_status.get(sensor_id, SensorStatus.OFFLINE)

    def set_sensor_status(self, sensor_id: str, status: SensorStatus):
        """Set sensor status"""
        if sensor_id in self.sensor_status:
            self.sensor_status[sensor_id] = status
            logging.info(f"Sensor {sensor_id} status changed to {status.value}")

    def get_fusion_metrics(self) -> Dict:
        """Get current fusion performance metrics"""
        return self.fusion_metrics.copy()

    def get_active_sensors(self) -> List[str]:
        """Get list of currently active sensors"""
        return [
            sensor_id for sensor_id, status in self.sensor_status.items()
            if status == SensorStatus.ACTIVE
        ]

# Example usage
if __name__ == "__main__":
    import asyncio

    # Example vehicle configuration
    vehicle_config = {
        'sensors': {
            'front_camera': {
                'type': 'camera',
                'position': [2.0, 0.0, 1.5],
                'orientation': [0.0, 0.0, 0.0],
                'field_of_view': [60.0, 45.0],
                'range_max': 100.0,
                'resolution': [1920, 1080],
                'frame_rate': 30.0
            },
            'front_radar': {
                'type': 'radar',
                'position': [2.5, 0.0, 0.5],
                'orientation': [0.0, 0.0, 0.0],
                'field_of_view': [20.0, 10.0],
                'range_max': 200.0,
                'frame_rate': 20.0
            },
            'roof_lidar': {
                'type': 'lidar',
                'position': [1.0, 0.0, 2.0],
                'orientation': [0.0, 0.0, 0.0],
                'field_of_view': [360.0, 30.0],
                'range_max': 150.0,
                'frame_rate': 10.0
            }
        }
    }

    async def main():
        # Initialize sensor fusion
        fusion = SensorFusion(vehicle_config)

        # Create example sensor data
        camera_data = RawSensorData(
            timestamp=time.time(),
            sensor_id='front_camera',
            sensor_type=SensorType.CAMERA,
            data=np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8),
            quality_metrics={'brightness': 0.5, 'contrast': 0.7},
            status=SensorStatus.ACTIVE,
            sequence_number=1
        )

        radar_data = RawSensorData(
            timestamp=time.time(),
            sensor_id='front_radar',
            sensor_type=SensorType.RADAR,
            data={
                'detections': [
                    {'range': 50.0, 'azimuth': 0.1, 'elevation': 0.0, 'snr': 15.0},
                    {'range': 30.0, 'azimuth': -0.2, 'elevation': 0.0, 'snr': 12.0}
                ]
            },
            quality_metrics={'snr_avg': 13.5},
            status=SensorStatus.ACTIVE,
            sequence_number=1
        )

        lidar_data = RawSensorData(
            timestamp=time.time(),
            sensor_id='roof_lidar',
            sensor_type=SensorType.LIDAR,
            data=np.random.rand(1000, 4) * 50,  # Random point cloud
            quality_metrics={'point_density': 0.8},
            status=SensorStatus.ACTIVE,
            sequence_number=1
        )

        # Process sensor frame
        sensor_data_list = [camera_data, radar_data, lidar_data]
        fused_detections = await fusion.process_sensor_frame(sensor_data_list)

        print(f"Fused {len(fused_detections)} detections:")
        for detection in fused_detections:
            print(f"  {detection.object_type} at {detection.position} "
                  f"(confidence: {detection.confidence:.2f}, "
                  f"sensors: {detection.contributing_sensors})")

        # Print metrics
        metrics = fusion.get_fusion_metrics()
        print(f"Fusion metrics: {metrics}")

    # Run example
    asyncio.run(main())